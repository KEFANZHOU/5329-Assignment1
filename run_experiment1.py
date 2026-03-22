import os
from typing import Any, Dict, List, Optional

from experiment_report_utils import plot_standard_history_bundle, read_json
from experiment_runner import run_experiment_suite


def _experiment1_summary_extra(
    condition_name: str,
    condition_kwargs: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    train_result: Dict[str, Any],
    eval_result: Dict[str, float],
    history: List[Dict[str, Any]],
    cond_dir: str,
    save_dir: str,
) -> Dict[str, Any]:
    return {
        "norm_name": condition_kwargs["norm_name"],
        "norm_groups": condition_kwargs["norm_groups"],
    }


def run_experiment1_norm_in_assignment(
    output_root: str = "exp_outputs/experiment1_norm",
    train_npz: str = "_data/train.npz",
    dev_npz: str = "_data/dev.npz",
    word_emb_json: str = "_data/word_emb.json",
    char_emb_json: str = "_data/char_emb.json",
    train_eval_json: str = "_data/train_eval.json",
    dev_eval_json: str = "_data/dev_eval.json",
    num_steps: int = 30000,
    checkpoint: int = 200,
    batch_size: int = 8,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
    early_stop: Optional[int] = None,
    optimizer_name: str = "adam",
    scheduler_name: str = "none",
    loss_name: str = "qa_ce",
    norm_groups: int = 8,
    plot_results: bool = False,
) -> Dict[str, Any]:
    """
    Experiment 1:
    Compare LayerNorm vs GroupNorm under the same optimizer, scheduler, and training budget.
    """
    run_seeds = [seed] if seeds is None else list(seeds)
    effective_early_stop = num_steps if early_stop is None else early_stop

    experiment_spec = {
        "title": "Experiment 1: Effect of Normalization Strategy",
        "based_on": "shared experiment runner with common training/evaluation flow",
        "research_question": (
            "Under the same optimizer, scheduler, data split, and training budget, "
            "do LayerNorm and GroupNorm affect training stability and final performance differently?"
        ),
        "hypothesis": (
            "LayerNorm is expected to outperform GroupNorm because it is more naturally "
            "aligned with token-level feature normalization and may yield more stable optimization."
        ),
        "conditions": ["layer_norm", "group_norm"],
        "controlled_variables": {
            "optimizer_name": optimizer_name,
            "scheduler_name": scheduler_name,
            "loss_name": loss_name,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "checkpoint": checkpoint,
            "seeds": run_seeds,
            "early_stop": "disabled" if early_stop is None else effective_early_stop,
            "same_official_eval": True,
            "same_data": True,
            "same_model_size": True,
        },
        "metrics": [
            "train_loss",
            "dev_loss",
            "dev_f1",
            "dev_em",
            "official_eval_f1",
            "official_eval_em",
            "official_eval_loss",
            "train_f1_minus_dev_f1",
            "dev_f1_std",
            "dev_loss_std",
            "first_step_dev_f1_ge_1",
            "first_step_dev_f1_ge_3",
            "step_of_best_logged_dev_f1",
        ],
        "analysis_focus": [
            "Which normalization converges more stably",
            "Which normalization reaches effective learning earlier",
            "Which normalization achieves higher final Dev F1 / EM",
            "Whether the difference mainly appears in optimization dynamics or final generalization",
        ],
    }

    conditions = {
        "layer_norm": {"norm_name": "layer_norm", "norm_groups": norm_groups},
        "group_norm": {"norm_name": "group_norm", "norm_groups": norm_groups},
    }

    base_train_kwargs = {
        "train_npz": train_npz,
        "dev_npz": dev_npz,
        "word_emb_json": word_emb_json,
        "char_emb_json": char_emb_json,
        "train_eval_json": train_eval_json,
        "dev_eval_json": dev_eval_json,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "checkpoint": checkpoint,
        "seed": run_seeds[0],
        "early_stop": effective_early_stop,
        "optimizer_name": optimizer_name,
        "scheduler_name": scheduler_name,
        "loss_name": loss_name,
    }

    return run_experiment_suite(
        title="Experiment 1",
        output_root=output_root,
        experiment_spec=experiment_spec,
        conditions=conditions,
        base_train_kwargs=base_train_kwargs,
        seeds=run_seeds,
        plot_results=plot_results,
        result_table_title="Experiment 1 Results",
        summary_extra_fn=_experiment1_summary_extra,
    )


def plot_experiment1_results(
    output_root: str = "exp_outputs/experiment1_norm",
    histories: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> None:
    if histories is None:
        histories = read_json(os.path.join(output_root, "histories.json"))

    plot_standard_history_bundle(output_root=output_root, histories=histories)
    print(f"Saved experiment 1 plots to {output_root}")


if __name__ == "__main__":
    run_experiment1_norm_in_assignment()
