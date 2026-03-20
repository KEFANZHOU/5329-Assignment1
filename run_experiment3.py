import csv
import json
import os
from statistics import pstdev
from typing import Any, Dict, List, Optional

from EvaluateTools.evaluate import evaluate
from TrainTools.train import train
from experiment_report_utils import (
    DEFAULT_RESULT_COLUMNS,
    plot_standard_history_bundle,
    print_table,
    read_json,
)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Effect of Weight Initialization
# ─────────────────────────────────────────────────────────────────────────────


def _mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _get_series(hist: List[Dict[str, Any]], key: str) -> List[Any]:
    return [row[key] for row in hist if key in row and row[key] is not None]


def _safe_std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return float(pstdev(values))


def _first_step_meeting(hist: List[Dict[str, Any]], key: str, threshold: float) -> Optional[int]:
    for row in hist:
        if key in row and row[key] is not None and row[key] >= threshold:
            return row.get("step")
    return None


def _step_of_best(hist: List[Dict[str, Any]], key: str, mode: str = "max") -> Optional[int]:
    candidates = [row for row in hist if key in row and row[key] is not None and "step" in row]
    if not candidates:
        return None
    best_row = max(candidates, key=lambda r: r[key]) if mode == "max" else min(candidates, key=lambda r: r[key])
    return best_row["step"]


def run_official_evaluation(
    train_metrics: Dict[str, Any],
    save_dir: str,
    log_dir: str,
    ckpt_path_key: str = "best_ckpt_path",
) -> Dict[str, float]:
    config = train_metrics["config"]
    ckpt_name = os.path.basename(train_metrics[ckpt_path_key])

    return evaluate(
        dev_npz=config["dev_npz"],
        word_emb_json=config["word_emb_json"],
        char_emb_json=config["char_emb_json"],
        dev_eval_json=config["dev_eval_json"],
        save_dir=save_dir,
        log_dir=log_dir,
        ckpt_name=ckpt_name,
        batch_size=config["batch_size"],
        test_num_batches=-1,
        loss_name=config["loss_name"],
        para_limit=config["para_limit"],
        ques_limit=config["ques_limit"],
        char_limit=config["char_limit"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        glove_dim=config["glove_dim"],
        char_dim=config["char_dim"],
        dropout=config["dropout"],
        dropout_char=config["dropout_char"],
        pretrained_char=config["pretrained_char"],
        norm_name=config["norm_name"],
        norm_groups=config["norm_groups"],
        activation=config["activation"],
        init_name=config["init_name"],
        use_batch_norm=config["use_batch_norm"],
    )


def run_experiment3(
    output_root: str = "exp_outputs/experiment3_init",
    num_steps: int = 60000,
    checkpoint: int = 200,
    batch_size: int = 8,
    seed: int = 42,
    optimizer_name: str = "adam",
    scheduler_name: str = "lambda",
    loss_name: str = "qa_ce",
    activation: str = "relu",
    init_names: Optional[List[str]] = None,
    plot_results: bool = False,
) -> Dict[str, Any]:
    if init_names is None:
        init_names = ["kaiming", "xavier"]

    output_root = _mkdir(output_root)

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
            "seed": seed,
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
            "project_loss_name_for_cross_entropy": "qa_ce",
            "compared_initializations": init_names,
        },
    }
    _write_json(os.path.join(output_root, "experiment_spec.json"), experiment_spec)

    results: Dict[str, Dict[str, Any]] = {}
    histories: Dict[str, List[Dict[str, Any]]] = {}

    for init_name in init_names:
        print("\n" + "=" * 80)
        print(f"Running condition: init = {init_name}")
        print("=" * 80)

        cond_dir = _mkdir(os.path.join(output_root, init_name))
        save_dir = _mkdir(os.path.join(cond_dir, "checkpoints"))
        train_log_dir = _mkdir(os.path.join(cond_dir, "train_logs"))
        eval_log_dir = _mkdir(os.path.join(cond_dir, "official_eval"))

        train_result = train(
            save_dir=save_dir,
            log_dir=train_log_dir,
            ckpt_name="model.pt",
            num_steps=num_steps,
            batch_size=batch_size,
            checkpoint=checkpoint,
            seed=seed,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            loss_name=loss_name,
            activation=activation,
            init_name=init_name,
        )

        history = train_result.get("history", [])
        histories[init_name] = history

        _write_json(os.path.join(cond_dir, "history.json"), history)
        _write_csv(os.path.join(cond_dir, "history.csv"), history)
        _write_json(os.path.join(cond_dir, "train_return.json"), train_result)

        print(f"\nRunning official evaluation for initialization '{init_name}'...")
        eval_result = run_official_evaluation(
            train_metrics=train_result,
            save_dir=save_dir,
            log_dir=eval_log_dir,
        )
        _write_json(os.path.join(eval_log_dir, "metrics.json"), eval_result)

        dev_f1_series = _get_series(history, "dev_f1")
        dev_loss_series = _get_series(history, "dev_loss")
        train_loss_series = _get_series(history, "train_loss")

        results[init_name] = {
            "condition": init_name,
            "init_name": init_name,
            "optimizer_name": optimizer_name,
            "scheduler_name": scheduler_name,
            "loss_name": loss_name,
            "activation": activation,
            "best_f1_during_training": train_result["best_f1"],
            "best_em_during_training": train_result["best_em"],
            "best_step_during_training": train_result.get("best_step"),
            "official_eval_f1": eval_result["f1"],
            "official_eval_em": eval_result["exact_match"],
            "official_eval_loss": eval_result["loss"],
            "first_logged_train_loss": train_loss_series[0] if train_loss_series else None,
            "last_logged_train_loss": train_loss_series[-1] if train_loss_series else None,
            "first_logged_dev_f1": dev_f1_series[0] if dev_f1_series else None,
            "last_logged_dev_f1": dev_f1_series[-1] if dev_f1_series else None,
            "best_logged_dev_f1": max(dev_f1_series) if dev_f1_series else None,
            "step_of_best_logged_dev_f1": _step_of_best(history, "dev_f1", mode="max"),
            "best_logged_dev_em": max(_get_series(history, "dev_em")) if _get_series(history, "dev_em") else None,
            "step_of_best_logged_dev_em": _step_of_best(history, "dev_em", mode="max"),
            "best_logged_dev_loss": min(dev_loss_series) if dev_loss_series else None,
            "step_of_best_logged_dev_loss": _step_of_best(history, "dev_loss", mode="min"),
            "first_step_dev_f1_ge_1": _first_step_meeting(history, "dev_f1", 1.0),
            "first_step_dev_f1_ge_3": _first_step_meeting(history, "dev_f1", 3.0),
            "dev_f1_std": _safe_std(dev_f1_series),
            "dev_loss_std": _safe_std(dev_loss_series),
            "history_path": os.path.join(cond_dir, "history.json"),
            "checkpoint_path": train_result.get("best_ckpt_path", os.path.join(save_dir, "best_model.pt")),
            "last_checkpoint_path": train_result.get("ckpt_path", os.path.join(save_dir, "model.pt")),
        }
        _write_json(os.path.join(cond_dir, "summary.json"), results[init_name])

    comparison_rows = []
    for init_name, item in results.items():
        comparison_rows.append(
            {
                "condition": init_name,
                "best_f1_during_training": item["best_f1_during_training"],
                "best_em_during_training": item["best_em_during_training"],
                "best_step_during_training": item["best_step_during_training"],
                "official_eval_f1": item["official_eval_f1"],
                "official_eval_em": item["official_eval_em"],
                "official_eval_loss": item["official_eval_loss"],
                "best_logged_dev_f1": item["best_logged_dev_f1"],
                "step_of_best_logged_dev_f1": item["step_of_best_logged_dev_f1"],
                "first_step_dev_f1_ge_1": item["first_step_dev_f1_ge_1"],
                "first_step_dev_f1_ge_3": item["first_step_dev_f1_ge_3"],
                "dev_f1_std": item["dev_f1_std"],
                "dev_loss_std": item["dev_loss_std"],
            }
        )

    comparison_rows = sorted(
        comparison_rows,
        key=lambda x: (-float("-inf") if x["official_eval_f1"] is None else -x["official_eval_f1"])
    )

    summary_payload: Dict[str, Any] = dict(results)
    if len(comparison_rows) >= 2:
        summary_payload["_ranking"] = {
            "winner_by_official_eval_f1": comparison_rows[0]["condition"],
            "ordered_conditions": [row["condition"] for row in comparison_rows],
        }

    _write_csv(os.path.join(output_root, "comparison.csv"), comparison_rows)
    _write_json(os.path.join(output_root, "histories.json"), histories)
    _write_json(os.path.join(output_root, "summary.json"), summary_payload)

    print_table("Experiment 3 Results", comparison_rows, DEFAULT_RESULT_COLUMNS)

    if plot_results:
        plot_experiment3_results(output_root=output_root, histories=histories)

    print(f"\nSaved experiment outputs to {output_root}")
    return {
        "summary": summary_payload,
        "histories": histories,
        "comparison_rows": comparison_rows,
        "output_root": output_root,
    }


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
