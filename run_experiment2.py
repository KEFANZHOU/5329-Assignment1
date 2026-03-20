import os
from typing import Any, Dict, List, Optional

from experiment_report_utils import plot_standard_history_bundle, read_json
from experiment_runner import run_experiment_suite


def _experiment2_summary_extra(
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
        "optimizer_name": train_kwargs["optimizer_name"],
        "scheduler_name": condition_kwargs["scheduler_name"],
    }


def run_experiment2(
    output_root: str = "exp_outputs/experiment2_scheduler",
    num_steps: int = 30000,
    checkpoint: int = 200,
    batch_size: int = 8,
    seed: int = 42,
    early_stop: Optional[int] = None,
    optimizer_name: str = "sgd_momentum",
    loss_name: str = "qa_nll",
    schedulers_to_test: Optional[List[str]] = None,
    lr_step_size: int = 5000,
    lr_gamma: float = 0.5,
    plot_results: bool = False,
) -> Dict[str, Any]:
    if schedulers_to_test is None:
        schedulers_to_test = ["none", "step", "cosine"]

    effective_early_stop = num_steps if early_stop is None else early_stop

    experiment_spec = {
        "title": "Experiment 2: Effect of Learning-Rate Scheduling on Late-Stage Optimization",
        "conditions": schedulers_to_test,
        "controlled_variables": {
            "optimizer_name": optimizer_name,
            "loss_name": loss_name,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "checkpoint": checkpoint,
            "seed": seed,
            "early_stop": "disabled" if early_stop is None else effective_early_stop,
            "lr_step_size": lr_step_size,
            "lr_gamma": lr_gamma,
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
    }

    conditions = {
        scheduler_name: {"scheduler_name": scheduler_name}
        for scheduler_name in schedulers_to_test
    }

    base_train_kwargs = {
        "num_steps": num_steps,
        "checkpoint": checkpoint,
        "batch_size": batch_size,
        "seed": seed,
        "early_stop": effective_early_stop,
        "optimizer_name": optimizer_name,
        "loss_name": loss_name,
        "lr_step_size": lr_step_size,
        "lr_gamma": lr_gamma,
    }

    return run_experiment_suite(
        title="Experiment 2",
        output_root=output_root,
        experiment_spec=experiment_spec,
        conditions=conditions,
        base_train_kwargs=base_train_kwargs,
        plot_results=plot_results,
        result_table_title="Experiment 2 Results",
        bundle_filename="experiment2_results.json",
        summary_extra_fn=_experiment2_summary_extra,
    )


def plot_experiment2_results(
    output_root: str = "exp_outputs/experiment2_scheduler",
    histories: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> None:
    if histories is None:
        histories = read_json(os.path.join(output_root, "histories.json"))

    plot_standard_history_bundle(output_root=output_root, histories=histories)
    print(f"Saved experiment 2 plots to {output_root}")


if __name__ == "__main__":
    run_experiment2()
