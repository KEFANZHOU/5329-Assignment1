import os
from typing import Any, Dict, List, Optional

from experiment_report_utils import plot_standard_history_bundle, read_json
from experiment_runner import run_experiment_suite


def _experiment3_summary_extra(
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
        "init_name": condition_kwargs["init_name"],
        "optimizer_name": train_kwargs["optimizer_name"],
        "scheduler_name": train_kwargs["scheduler_name"],
        "loss_name": train_kwargs["loss_name"],
        "activation": train_kwargs["activation"],
    }


def run_experiment3(
    output_root: str = "exp_outputs/experiment3_init",
    num_steps: int = 20000,
    checkpoint: int = 200,
    batch_size: int = 8,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
    early_stop: Optional[int] = None,
    optimizer_name: str = "adam",
    scheduler_name: str = "lambda",
    loss_name: str = "qa_nll",
    activation: str = "relu",
    init_names: Optional[List[str]] = None,
    plot_results: bool = False,
) -> Dict[str, Any]:
    run_seeds = [seed] if seeds is None else list(seeds)
    if init_names is None:
        init_names = ["kaiming", "xavier"]

    effective_early_stop = num_steps if early_stop is None else early_stop

    experiment_spec = {
        "title": "Experiment 3: Effect of Weight Initialization",
        "research_question": (
            "With activation, optimizer, scheduler, loss, and all other hyperparameters fixed, "
            "does Kaiming initialization or Xavier initialization lead to better training dynamics "
            "and final QA performance?"
        ),
        "conditions": init_names,
        "controlled_variables": {
            "optimizer_name": optimizer_name,
            "scheduler_name": scheduler_name,
            "loss_name": loss_name,
            "activation": activation,
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
            "dev_f1_std",
            "dev_loss_std",
            "first_step_dev_f1_ge_1",
            "first_step_dev_f1_ge_3",
            "step_of_best_logged_dev_f1",
        ],
        "notes": {
            "project_loss_name_for_cross_entropy": "qa_nll",
            "compared_initializations": init_names,
        },
    }

    conditions = {
        init_name: {"init_name": init_name}
        for init_name in init_names
    }

    base_train_kwargs = {
        "num_steps": num_steps,
        "checkpoint": checkpoint,
        "batch_size": batch_size,
        "seed": run_seeds[0],
        "early_stop": effective_early_stop,
        "optimizer_name": optimizer_name,
        "scheduler_name": scheduler_name,
        "loss_name": loss_name,
        "activation": activation,
    }

    return run_experiment_suite(
        title="Experiment 3",
        output_root=output_root,
        experiment_spec=experiment_spec,
        conditions=conditions,
        base_train_kwargs=base_train_kwargs,
        seeds=run_seeds,
        plot_results=plot_results,
        result_table_title="Experiment 3 Results",
        summary_extra_fn=_experiment3_summary_extra,
    )


def plot_experiment3_results(
    output_root: str = "exp_outputs/experiment3_init",
    histories: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> None:
    if histories is None:
        histories = read_json(os.path.join(output_root, "histories.json"))

    plot_standard_history_bundle(output_root=output_root, histories=histories)
    print(f"Saved experiment 3 plots to {output_root}")


if __name__ == "__main__":
    run_experiment3()
