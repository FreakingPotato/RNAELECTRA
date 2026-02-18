import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import (
    Trainer,
    TrainingArguments,
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertModel,
    ModernBertPreTrainedModel
)
import random
from .tokenizer import NucEL_Tokenizer  # Relative import within model package


# Custom ModernBert discriminator for ELECTRA
class ModernBertForTokenClassification(ModernBertPreTrainedModel):
    """ModernBERT discriminator for ELECTRA"""
    def __init__(self, config):
        super().__init__(config)
        self.bert = ModernBertModel(config)
        self.discriminator_predictions = nn.Linear(config.hidden_size, 1)
        # self.init_weights()
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        **kwargs  
    ):
        # Remove token_type_ids from params as ModernBERT doesn't use it
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        sequence_output = outputs[0]
        logits = self.discriminator_predictions(sequence_output).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.bool()
                active_logits = logits[active_loss]
                active_labels = labels.float()[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits, labels.float())
        
        return type('ElectraOutput', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        })

# --- ElectraPretrainingModel adapted for ModernBERT ---
# ELECTRA Pretraining Model with Temperature Sampling
class ElectraPretrainingModel(nn.Module):
    def __init__(self, generator, discriminator, tokenizer, 
                 temperature=0.9, top_k=5, weight_factor=50.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.tokenizer = tokenizer
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.temperature = temperature
        self.top_k = top_k
        self.weight_factor = weight_factor  # Weight factor for discriminator loss
        
    def sample_from_logits(self, logits, temperature=1.0, top_k=0):
        """Sample from logits with temperature and optional top-k filtering"""
        # Apply temperature
        if temperature != 1.0:
            scaled_logits = logits / temperature
        else:
            scaled_logits = logits
            
        # Optional top-k filtering
        if top_k > 0:
            # Get top-k values and their indices
            top_k_logits, top_k_indices = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)), dim=-1)
            
            # Create a mask of invalid (non top-k) tokens
            indices_to_remove = scaled_logits < top_k_logits[..., -1].unsqueeze(-1)
            
            # Set probabilities of non-top-k tokens to 0
            filtered_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))
        else:
            filtered_logits = scaled_logits
        
        # Convert to probabilities
        probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
        
        # Sample from the distribution
        sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
        
        # Reshape back to match input shape
        sampled_tokens = sampled_tokens.view(logits.size()[:-1])
        
        return sampled_tokens

    def forward(self, input_ids, attention_mask, labels=None, masked_indices=None):
        # If labels and masked_indices are not provided, create them
        if labels is None or masked_indices is None:
            # Create labels as a copy of input IDs
            labels = input_ids.clone()
            
            # Create masked indices with 15% probability
            probability_matrix = torch.full(input_ids.shape, 0.15)
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in input_ids.tolist()
            ]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            # Create masked input_ids for generator
            masked_input = input_ids.clone()
            masked_input[masked_indices] = self.mask_token_id
            
            # For generator labels, use -100 for non-masked tokens
            gen_labels = labels.clone()
            gen_labels[~masked_indices] = -100
        else:
            masked_input = input_ids.clone()
            masked_input[masked_indices] = self.mask_token_id
            gen_labels = labels.clone()
            gen_labels[~masked_indices] = -100
        
        # --- Generator Forward Pass ---
        gen_outputs = self.generator(
            input_ids=masked_input,
            attention_mask=attention_mask,
            labels=gen_labels
        )
        generator_loss = gen_outputs.loss
        generator_logits = gen_outputs.logits

        # Sample tokens from generator predictions with temperature
        if self.training:
            # During training, use temperature sampling
            sampled_tokens = self.sample_from_logits(
                generator_logits, 
                temperature=self.temperature, 
                top_k=self.top_k
            )
        else:
            # During evaluation, use argmax for deterministic results
            sampled_tokens = torch.argmax(generator_logits, dim=-1)

        # --- Create Corrupted Input for Discriminator ---
        # Replace masked tokens with generator predictions
        corrupted_input_ids = input_ids.clone()
        corrupted_input_ids[masked_indices] = sampled_tokens[masked_indices]

        # Create binary labels for discriminator:
        # 1 if the token is the original one, 0 if it was replaced
        disc_labels = (corrupted_input_ids == labels).long()

        # --- Discriminator Forward Pass ---
        disc_outputs = self.discriminator(
            input_ids=corrupted_input_ids,
            attention_mask=attention_mask,
            labels=disc_labels
        )
        discriminator_loss = disc_outputs.loss

        # Total loss is a weighted sum of generator and discriminator losses
        total_loss = generator_loss + self.weight_factor * discriminator_loss

        return {
            "loss": total_loss,
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
            "generator_logits": generator_logits,
            "sampled_tokens": sampled_tokens,
            "disc_labels": disc_labels,
            "disc_logits": disc_outputs.logits if hasattr(disc_outputs, 'logits') else None
        }

class ElectraTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation for ELECTRA with handling for num_items_in_batch parameter
        which is passed by the trainer but not used in our implementation
        """
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            masked_indices=inputs.get("masked_indices")
        )
        
        # Log component losses periodically
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                'generator_loss': outputs['generator_loss'].item(),
                'discriminator_loss': outputs['discriminator_loss'].item(),
                'weighted_disc_loss': (model.weight_factor * outputs['discriminator_loss']).item(),
                'loss_ratio': (model.weight_factor * outputs['discriminator_loss'] / outputs['generator_loss']).item() 
                if outputs['generator_loss'] > 0 else 0
            })
        
        return (outputs['loss'], outputs) if return_outputs else outputs['loss']
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate method to separately evaluate generator and discriminator performance
        """
        # Run standard evaluation
        eval_output = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Get eval dataset if not provided
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        # Custom evaluation metrics
        generator_metrics = self._evaluate_generator(eval_dataset)
        discriminator_metrics = self._evaluate_discriminator(eval_dataset)
        
        # Log all metrics
        for key, value in generator_metrics.items():
            eval_output[f"{metric_key_prefix}_generator_{key}"] = value
            self.log({f"{metric_key_prefix}_generator_{key}": value})
            
        for key, value in discriminator_metrics.items():
            eval_output[f"{metric_key_prefix}_discriminator_{key}"] = value
            self.log({f"{metric_key_prefix}_discriminator_{key}": value})
            
        return eval_output
    
    def _evaluate_generator(self, eval_dataset):
        """Evaluate generator MLM accuracy separately"""
        model = self.model
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Switch to eval mode
        model.eval()
        
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        
        for batch in eval_dataloader:
            batch = self._prepare_inputs(batch)
            
            with torch.no_grad():
                # Create masked inputs and labels
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                masked_indices = batch["masked_indices"]
                labels = batch["labels"]
                
                masked_input = input_ids.clone()
                masked_input[masked_indices] = model.mask_token_id
                
                # Generator forward pass
                gen_labels = labels.clone()
                gen_labels[~masked_indices] = -100
                
                gen_outputs = model.generator(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    labels=gen_labels
                )
                
                # Compute generator accuracy
                logits = gen_outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                # Only count masked tokens
                active_tokens = masked_indices & attention_mask.bool()
                active_preds = predictions[active_tokens]
                active_labels = labels[active_tokens]
                
                correct = (active_preds == active_labels).sum().item()
                total_correct += correct
                total_tokens += active_tokens.sum().item()
                total_loss += gen_outputs.loss.item() * active_tokens.sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {"loss": avg_loss, "accuracy": accuracy, "perplexity": perplexity}
    
    def _evaluate_discriminator(self, eval_dataset):
        """Evaluate discriminator replaced token detection accuracy separately"""
        model = self.model
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Switch to eval mode
        model.eval()
        
        total_loss = 0
        total_tokens = 0
        total_correct = 0
        total_masked_correct = 0
        total_masked_tokens = 0
        
        for batch in eval_dataloader:
            batch = self._prepare_inputs(batch)
            
            with torch.no_grad():
                # Get batch data
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                masked_indices = batch["masked_indices"]
                labels = batch["labels"]
                
                # Create masked inputs
                masked_input = input_ids.clone()
                masked_input[masked_indices] = model.mask_token_id
                
                # Generator forward pass (to get replacements)
                gen_outputs = model.generator(
                    input_ids=masked_input,
                    attention_mask=attention_mask
                )
                
                # Sample tokens from generator predictions
                sampled_tokens = torch.argmax(gen_outputs.logits, dim=-1)
                
                # Create corrupted input
                corrupted_input_ids = input_ids.clone()
                corrupted_input_ids[masked_indices] = sampled_tokens[masked_indices]
                
                # Create binary labels
                disc_labels = (corrupted_input_ids == labels).long()
                
                # Discriminator forward pass
                disc_outputs = model.discriminator(
                    input_ids=corrupted_input_ids,
                    attention_mask=attention_mask,
                    labels=disc_labels
                )
                
                # Compute discriminator accuracy
                logits = disc_outputs.logits
                predictions = (torch.sigmoid(logits) > 0.5).long()
                
                # Count all tokens
                active_tokens = attention_mask.bool()
                active_preds = predictions[active_tokens]
                active_labels = disc_labels[active_tokens]
                
                correct = (active_preds == active_labels).sum().item()
                total_correct += correct
                total_tokens += active_tokens.sum().item()
                
                # Count only masked tokens (the replaced ones)
                masked_active = active_tokens & masked_indices
                masked_preds = predictions[masked_active]
                masked_labels = disc_labels[masked_active]
                
                masked_correct = (masked_preds == masked_labels).sum().item()
                total_masked_correct += masked_correct
                total_masked_tokens += masked_active.sum().item()
                
                total_loss += disc_outputs.loss.item() * active_tokens.sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        masked_accuracy = total_masked_correct / total_masked_tokens if total_masked_tokens > 0 else 0
        
        return {
            "loss": avg_loss, 
            "accuracy": accuracy, 
            "masked_accuracy": masked_accuracy
        }
