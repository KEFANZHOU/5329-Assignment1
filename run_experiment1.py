import csv
import json
import os
from statistics import pstdev
from typing import Dict, Any, List, Optional

from TrainTools.train import train
from EvaluateTools.evaluate import evaluate
from experiment_report_utils import (
    DEFAULT_RESULT_COLUMNS,
    plot_standard_history_bundle,
    print_table,
    read_json,
)


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
    out = []
    for row in hist:
        if key in row and row[key] is not None:
            out.append(row[key])
    return out


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


def run_experiment1_norm_in_assignment(
    output_root: str = "exp_outputs/experiment1_norm",
    train_npz: str = "_data/train.npz",
    dev_npz: str = "_data/dev.npz",
    word_emb_json: str = "_data/word_emb.json",
    char_emb_json: str = "_data/char_emb.json",
    train_eval_json: str = "_data/train_eval.json",
    dev_eval_json: str = "_data/dev_eval.json",
    num_steps: int = 60000,
    checkpoint: int = 200,
    batch_size: int = 8,
    seed: int = 42,
    optimizer_name: str = "sgd",
    scheduler_name: str = "none",
    loss_name: str = "qa_nll",
    norm_groups: int = 8,
    plot_results: bool = False,
) -> Dict[str, Any]:
    """
    Experiment 1:
    Compare LayerNorm vs GroupNorm under the same training recipe used in Experiment 2.
    """
    output_root = _mkdir(output_root)

    experiment_spec = {
        "title": "Experiment 1: Effect of Normalization Strategy",
        "based_on": "same training-and-evaluation workflow as run_experiment2.py",
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
    _write_json(os.path.join(output_root, "experiment_spec.json"), experiment_spec)

    conditions = {
        "layer_norm": {"norm_name": "layer_norm", "norm_groups": norm_groups},
        "group_norm": {"norm_name": "group_norm", "norm_groups": norm_groups},
    }

    all_histories: Dict[str, List[Dict[str, Any]]] = {}
    summary: Dict[str, Dict[str, Any]] = {}

    for cond_name, cond_cfg in conditions.items():
        print("\n" + "=" * 80)
        print(f"Running condition: {cond_name}")
        print("=" * 80)

        cond_dir = _mkdir(os.path.join(output_root, cond_name))
        save_dir = _mkdir(os.path.join(cond_dir, "checkpoints"))
        train_log_dir = _mkdir(os.path.join(cond_dir, "train_logs"))
        eval_log_dir = _mkdir(os.path.join(cond_dir, "eval_logs"))

        train_result = train(
            train_npz=train_npz,
            dev_npz=dev_npz,
            word_emb_json=word_emb_json,
            char_emb_json=char_emb_json,
            train_eval_json=train_eval_json,
            dev_eval_json=dev_eval_json,
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
            norm_name=cond_cfg["norm_name"],
            norm_groups=cond_cfg["norm_groups"],
        )

        history = train_result.get("history", [])
        all_histories[cond_name] = history

        _write_json(os.path.join(cond_dir, "history.json"), history)
        _write_csv(os.path.join(cond_dir, "history.csv"), history)
        _write_json(os.path.join(cond_dir, "train_return.json"), train_result)

        print(f"\nRunning official evaluation for normalization '{cond_name}'...")
        eval_result = run_official_evaluation(
            train_metrics=train_result,
            save_dir=save_dir,
            log_dir=eval_log_dir,
        )

        dev_f1_series = _get_series(history, "dev_f1")
        dev_loss_series = _get_series(history, "dev_loss")
        train_loss_series = _get_series(history, "train_loss")

        summary_item = {
            "condition": cond_name,
            "norm_name": cond_cfg["norm_name"],
            "norm_groups": cond_cfg["norm_groups"],

            "best_f1_during_training": float(train_result["best_f1"]) if "best_f1" in train_result else None,
            "best_em_during_training": float(train_result["best_em"]) if "best_em" in train_result else None,
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

            "first_step_dev_f1_ge_1": _first_step_meeting(history, "dev_f1", 1.0),
            "first_step_dev_f1_ge_3": _first_step_meeting(history, "dev_f1", 3.0),

            "best_logged_dev_em": max(_get_series(history, "dev_em")) if _get_series(history, "dev_em") else None,
            "step_of_best_logged_dev_em": _step_of_best(history, "dev_em", mode="max"),

            "best_logged_dev_loss": min(dev_loss_series) if dev_loss_series else None,
            "step_of_best_logged_dev_loss": _step_of_best(history, "dev_loss", mode="min"),

            "dev_f1_std": _safe_std(dev_f1_series),
            "dev_loss_std": _safe_std(dev_loss_series),

            "history_path": os.path.join(cond_dir, "history.json"),
            "checkpoint_path": train_result.get("best_ckpt_path", os.path.join(save_dir, "best_model.pt")),
            "last_checkpoint_path": train_result.get("ckpt_path", os.path.join(save_dir, "model.pt")),
        }

        summary[cond_name] = summary_item
        _write_json(os.path.join(cond_dir, "summary.json"), summary_item)

    comparison_rows = []
    for cond_name, item in summary.items():
        comparison_rows.append(
            {
                "condition": cond_name,
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

    _write_csv(os.path.join(output_root, "comparison.csv"), comparison_rows)
    _write_json(os.path.join(output_root, "histories.json"), all_histories)
    _write_json(os.path.join(output_root, "summary.json"), summary)

    if len(comparison_rows) >= 2:
        winner = comparison_rows[0]["condition"]
        summary["_ranking"] = {
            "winner_by_official_eval_f1": winner,
            "ordered_conditions": [row["condition"] for row in comparison_rows],
        }
        _write_json(os.path.join(output_root, "summary.json"), summary)

    print_table(
        "Experiment 1 Results",
        comparison_rows,
        DEFAULT_RESULT_COLUMNS,
    )

    if plot_results:
        plot_experiment1_results(output_root=output_root, histories=all_histories)

    print("\nExperiment 1 finished.")
    print(json.dumps(summary, indent=2))
    return {
        "summary": summary,
        "histories": all_histories,
        "output_root": output_root,
        "comparison_rows": comparison_rows,
    }


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
