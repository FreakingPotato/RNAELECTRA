"""
Task and Model Configuration for BEACON Benchmark
Based on: BEACON: Benchmark for Comprehensive RNA Tasks and Language Models (NeurIPS 2024)

Configuration includes:
- Task definitions with metrics, problem types, and training settings
- Model configurations for different RNA foundation models
- Max lengths derived from BEACON Paper Table 1
"""

# =============================================================================
# Task Configuration (Based on BEACON Paper Table 1 & Appendix A.1)
# =============================================================================

TASK_CONFIG = {
    # Function Tasks
    "spl": {
        "folder": "SpliceAI",
        "problem_type": "token_classification",
        "num_labels": 3,
        "metric": "top_k_accuracy",
        "metric_for_best_model": "top_k_accuracy",
        "greater_is_better": True,
        "epochs": 30,
        "max_length": 100,
        "label_column": "labels",
    },
    "apa": {
        "folder": "Isoform",
        "problem_type": "regression",
        "num_labels": 1,
        "metric": "r2",
        "metric_for_best_model": "r2",
        "greater_is_better": True,
        "epochs": 30,
        "max_length": 186,
        "label_column": "proximal_isoform_proportion",
    },
    "ncrna": {
        "folder": "NoncodingRNAFamily",
        "problem_type": "single_label_classification",
        "num_labels": 13,
        "metric": "accuracy",
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "epochs": 30,
        "max_length": 1182,
        "label_column": "label",
    },
    "modif": {
        "folder": "Modification",
        "problem_type": "multi_label_classification",
        "num_labels": 12,
        "metric": "auc",
        "metric_for_best_model": "auc",
        "greater_is_better": True,
        "epochs": 30,
        "max_length": 101,
        "label_column": "label",
    },
    "mrl": {
        "folder": "MeanRibosomeLoading",
        "problem_type": "regression",
        "num_labels": 1,
        "metric": "r2",
        "metric_for_best_model": "r2",
        "greater_is_better": True,
        "epochs": 30,
        "max_length": 100,
        "label_column": "label",
    },
    # Engineering Tasks
    "prs": {
        "folder": "ProgrammableRNASwitches",
        "problem_type": "regression",
        "num_labels": 3,
        "metric": "r2",
        "metric_for_best_model": "r2",
        "greater_is_better": True,
        "epochs": 30,
        "max_length": 148,
        "label_column": ["ON", "OFF", "ON_OFF"],
    },
    "cri_on": {
        "folder": "CRISPROnTarget",
        "problem_type": "regression",
        "num_labels": 1,
        "metric": "spearman",
        "metric_for_best_model": "spearman",
        "greater_is_better": True,
        "epochs": 30,
        "max_length": 23,
        "label_column": "label",
    },
    "cri_off": {
        "folder": "CRISPROffTarget",
        "problem_type": "regression",
        "num_labels": 1,
        "metric": "spearman",
        "metric_for_best_model": "spearman",
        "greater_is_better": True,
        "epochs": 30,
        "max_length": 50,  # 23 + 23 + special tokens
        "label_column": "label",
        "sequence_columns": ["sgrna", "target"],  # Multiple input columns
    },
}

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_CONFIG = {
    # Released RNAElectra encoder. Use --model_path to point at a local
    # checkpoint directory instead.
    "RNAElectra": {
        "path": "FreakingPotato/RNAElectra",
        "trust_remote_code": True,
    },
}


def get_task_config(task_name: str) -> dict:
    """Get task configuration by name."""
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASK_CONFIG.keys())}")
    return TASK_CONFIG[task_name]


def get_model_config(model_name: str) -> dict:
    """Get model configuration by name."""
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_CONFIG.keys())}")
    return MODEL_CONFIG[model_name]


def list_tasks() -> list:
    """List all available tasks."""
    return list(TASK_CONFIG.keys())


def list_models() -> list:
    """List all available models."""
    return list(MODEL_CONFIG.keys())


def get_task_max_length(task_name: str) -> int:
    """Get the maximum sequence length for a task."""
    return get_task_config(task_name)["max_length"]


def get_label_column(task_name: str) -> str:
    """Get the label column name for a task."""
    return get_task_config(task_name)["label_column"]