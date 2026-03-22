import csv
import json
import os
from statistics import pstdev
from typing import Any, Callable, Dict, List, Optional

from EvaluateTools.evaluate import evaluate
from TrainTools.train import train
from experiment_report_utils import DEFAULT_RESULT_COLUMNS, plot_standard_history_bundle, print_table


ConditionConfig = Dict[str, Any]
SummaryExtraFn = Callable[
    [str, ConditionConfig, Dict[str, Any], Dict[str, Any], Dict[str, float], List[Dict[str, Any]], str, str],
    Dict[str, Any],
]


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


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _aggregate_histories(histories: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    per_step: Dict[int, Dict[str, List[float]]] = {}
    for history in histories:
        for row in history:
            step = row.get("step")
            if step is None:
                continue
            step_bucket = per_step.setdefault(int(step), {})
            for key, value in row.items():
                if key == "step" or not _is_numeric(value):
                    continue
                step_bucket.setdefault(key, []).append(float(value))

    aggregated = []
    for step in sorted(per_step):
        row: Dict[str, Any] = {"step": step}
        for key, values in per_step[step].items():
            if values:
                row[key] = float(sum(values) / len(values))
        aggregated.append(row)
    return aggregated


def _aggregate_seed_summaries(
    condition_name: str,
    seed_summaries: List[Dict[str, Any]],
    cond_dir: str,
) -> Dict[str, Any]:
    aggregate = dict(seed_summaries[0])

    numeric_keys = set()
    for item in seed_summaries:
        for key, value in item.items():
            if key != "seed" and _is_numeric(value):
                numeric_keys.add(key)

    for key in numeric_keys:
        values = [float(item[key]) for item in seed_summaries if key in item and _is_numeric(item[key])]
        if values:
            aggregate[key] = float(sum(values) / len(values))

    for key in [
        "official_eval_f1",
        "official_eval_em",
        "best_logged_dev_f1",
        "best_step_during_training",
        "last_logged_grad_norm",
    ]:
        values = [float(item[key]) for item in seed_summaries if key in item and _is_numeric(item[key])]
        aggregate[f"{key}_seed_std"] = _safe_std(values)

    aggregate["condition"] = condition_name
    aggregate["seed_count"] = len(seed_summaries)
    aggregate["seeds"] = [item["seed"] for item in seed_summaries]
    aggregate.pop("seed", None)
    aggregate.pop("seed_dir", None)
    aggregate["history_path"] = os.path.join(cond_dir, "history.json")
    aggregate["seed_histories_path"] = os.path.join(cond_dir, "seed_histories.json")
    aggregate["seed_summaries_path"] = os.path.join(cond_dir, "seed_summaries.json")
    aggregate["checkpoint_path"] = None
    aggregate["last_checkpoint_path"] = None
    aggregate["seed_checkpoint_paths"] = {
        str(item["seed"]): item.get("checkpoint_path") for item in seed_summaries
    }
    aggregate["seed_last_checkpoint_paths"] = {
        str(item["seed"]): item.get("last_checkpoint_path") for item in seed_summaries
    }
    return aggregate


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


def _build_summary_item(
    condition_name: str,
    history: List[Dict[str, Any]],
    train_result: Dict[str, Any],
    eval_result: Dict[str, float],
    cond_dir: str,
    save_dir: str,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dev_f1_series = _get_series(history, "dev_f1")
    dev_loss_series = _get_series(history, "dev_loss")
    train_loss_series = _get_series(history, "train_loss")
    dev_em_series = _get_series(history, "dev_em")
    grad_norm_series = _get_series(history, "grad_norm")

    summary_item = {
        "condition": condition_name,
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
        "best_logged_dev_em": max(dev_em_series) if dev_em_series else None,
        "step_of_best_logged_dev_em": _step_of_best(history, "dev_em", mode="max"),
        "best_logged_dev_loss": min(dev_loss_series) if dev_loss_series else None,
        "step_of_best_logged_dev_loss": _step_of_best(history, "dev_loss", mode="min"),
        "first_step_dev_f1_ge_1": _first_step_meeting(history, "dev_f1", 1.0),
        "first_step_dev_f1_ge_3": _first_step_meeting(history, "dev_f1", 3.0),
        "dev_f1_std": _safe_std(dev_f1_series),
        "dev_loss_std": _safe_std(dev_loss_series),
        "first_logged_grad_norm": grad_norm_series[0] if grad_norm_series else None,
        "last_logged_grad_norm": grad_norm_series[-1] if grad_norm_series else None,
        "min_logged_grad_norm": min(grad_norm_series) if grad_norm_series else None,
        "max_logged_grad_norm": max(grad_norm_series) if grad_norm_series else None,
        "grad_norm_std": _safe_std(grad_norm_series),
        "history_path": os.path.join(cond_dir, "history.json"),
        "checkpoint_path": train_result.get("best_ckpt_path", os.path.join(save_dir, "best_model.pt")),
        "last_checkpoint_path": train_result.get("ckpt_path", os.path.join(save_dir, "model.pt")),
    }
    if extra_fields:
        summary_item.update(extra_fields)
    return summary_item


def _build_comparison_rows(summary_items: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    comparison_rows = []
    for item in summary_items.values():
        comparison_rows.append(
            {
                "condition": item["condition"],
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
                "last_logged_grad_norm": item["last_logged_grad_norm"],
                "min_logged_grad_norm": item["min_logged_grad_norm"],
                "max_logged_grad_norm": item["max_logged_grad_norm"],
                "grad_norm_std": item["grad_norm_std"],
                "seed_count": item.get("seed_count"),
                "official_eval_f1_seed_std": item.get("official_eval_f1_seed_std"),
                "official_eval_em_seed_std": item.get("official_eval_em_seed_std"),
                "best_logged_dev_f1_seed_std": item.get("best_logged_dev_f1_seed_std"),
            }
        )

    return sorted(
        comparison_rows,
        key=lambda x: (-float("-inf") if x["official_eval_f1"] is None else -x["official_eval_f1"]),
    )


def run_experiment_suite(
    *,
    title: str,
    output_root: str,
    experiment_spec: Dict[str, Any],
    conditions: Dict[str, ConditionConfig],
    base_train_kwargs: Dict[str, Any],
    seeds: Optional[List[int]] = None,
    plot_results: bool = False,
    result_table_title: Optional[str] = None,
    bundle_filename: Optional[str] = None,
    summary_extra_fn: Optional[SummaryExtraFn] = None,
) -> Dict[str, Any]:
    output_root = _mkdir(output_root)
    run_seeds = list(seeds) if seeds is not None else [int(base_train_kwargs.get("seed", 42))]
    if not run_seeds:
        raise ValueError("seeds must contain at least one value")
    _write_json(os.path.join(output_root, "experiment_spec.json"), experiment_spec)

    all_histories: Dict[str, List[Dict[str, Any]]] = {}
    all_seed_histories: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    summary_items: Dict[str, Dict[str, Any]] = {}

    for condition_name, condition_kwargs in conditions.items():
        print("\n" + "=" * 80)
        print(f"Running condition: {condition_name}")
        print("=" * 80)

        cond_dir = _mkdir(os.path.join(output_root, condition_name))
        if len(run_seeds) == 1:
            seed = run_seeds[0]
            save_dir = _mkdir(os.path.join(cond_dir, "checkpoints"))
            train_log_dir = _mkdir(os.path.join(cond_dir, "train_logs"))
            eval_log_dir = _mkdir(os.path.join(cond_dir, "official_eval"))

            train_kwargs = dict(base_train_kwargs)
            train_kwargs.update(condition_kwargs)
            train_kwargs["seed"] = seed
            train_result = train(
                save_dir=save_dir,
                log_dir=train_log_dir,
                ckpt_name="model.pt",
                **train_kwargs,
            )

            history = train_result.get("history", [])
            all_histories[condition_name] = history

            _write_json(os.path.join(cond_dir, "history.json"), history)
            _write_csv(os.path.join(cond_dir, "history.csv"), history)
            _write_json(os.path.join(cond_dir, "train_return.json"), train_result)

            print(f"\nRunning official evaluation for condition '{condition_name}'...")
            eval_result = run_official_evaluation(
                train_metrics=train_result,
                save_dir=save_dir,
                log_dir=eval_log_dir,
            )
            _write_json(os.path.join(eval_log_dir, "metrics.json"), eval_result)

            extra_fields = None
            if summary_extra_fn is not None:
                extra_fields = summary_extra_fn(
                    condition_name,
                    condition_kwargs,
                    train_kwargs,
                    train_result,
                    eval_result,
                    history,
                    cond_dir,
                    save_dir,
                )

            summary_item = _build_summary_item(
                condition_name=condition_name,
                history=history,
                train_result=train_result,
                eval_result=eval_result,
                cond_dir=cond_dir,
                save_dir=save_dir,
                extra_fields=extra_fields,
            )
            summary_items[condition_name] = summary_item
            _write_json(os.path.join(cond_dir, "summary.json"), summary_item)
            continue

        seed_histories: Dict[str, List[Dict[str, Any]]] = {}
        seed_summaries: List[Dict[str, Any]] = []

        for seed in run_seeds:
            seed_name = f"seed_{seed}"
            seed_dir = _mkdir(os.path.join(cond_dir, seed_name))
            save_dir = _mkdir(os.path.join(seed_dir, "checkpoints"))
            train_log_dir = _mkdir(os.path.join(seed_dir, "train_logs"))
            eval_log_dir = _mkdir(os.path.join(seed_dir, "official_eval"))

            print(f"\nRunning seed {seed} for condition '{condition_name}'...")
            train_kwargs = dict(base_train_kwargs)
            train_kwargs.update(condition_kwargs)
            train_kwargs["seed"] = seed
            train_result = train(
                save_dir=save_dir,
                log_dir=train_log_dir,
                ckpt_name="model.pt",
                **train_kwargs,
            )

            history = train_result.get("history", [])
            seed_histories[seed_name] = history

            _write_json(os.path.join(seed_dir, "history.json"), history)
            _write_csv(os.path.join(seed_dir, "history.csv"), history)
            _write_json(os.path.join(seed_dir, "train_return.json"), train_result)

            print(f"\nRunning official evaluation for condition '{condition_name}' (seed {seed})...")
            eval_result = run_official_evaluation(
                train_metrics=train_result,
                save_dir=save_dir,
                log_dir=eval_log_dir,
            )
            _write_json(os.path.join(eval_log_dir, "metrics.json"), eval_result)

            extra_fields = None
            if summary_extra_fn is not None:
                extra_fields = summary_extra_fn(
                    condition_name,
                    condition_kwargs,
                    train_kwargs,
                    train_result,
                    eval_result,
                    history,
                    seed_dir,
                    save_dir,
                )

            seed_summary = _build_summary_item(
                condition_name=condition_name,
                history=history,
                train_result=train_result,
                eval_result=eval_result,
                cond_dir=seed_dir,
                save_dir=save_dir,
                extra_fields=extra_fields,
            )
            seed_summary["seed"] = seed
            seed_summary["seed_dir"] = seed_dir
            seed_summaries.append(seed_summary)
            _write_json(os.path.join(seed_dir, "summary.json"), seed_summary)

        aggregated_history = _aggregate_histories(list(seed_histories.values()))
        all_histories[condition_name] = aggregated_history
        all_seed_histories[condition_name] = seed_histories

        _write_json(os.path.join(cond_dir, "history.json"), aggregated_history)
        _write_csv(os.path.join(cond_dir, "history.csv"), aggregated_history)
        _write_json(os.path.join(cond_dir, "seed_histories.json"), seed_histories)
        _write_json(os.path.join(cond_dir, "seed_summaries.json"), seed_summaries)

        summary_item = _aggregate_seed_summaries(condition_name, seed_summaries, cond_dir)
        summary_items[condition_name] = summary_item
        _write_json(os.path.join(cond_dir, "summary.json"), summary_item)

    comparison_rows = _build_comparison_rows(summary_items)

    summary_payload: Dict[str, Any] = dict(summary_items)
    if len(comparison_rows) >= 2:
        summary_payload["_ranking"] = {
            "winner_by_official_eval_f1": comparison_rows[0]["condition"],
            "ordered_conditions": [row["condition"] for row in comparison_rows],
        }

    _write_csv(os.path.join(output_root, "comparison.csv"), comparison_rows)
    _write_json(os.path.join(output_root, "histories.json"), all_histories)
    if all_seed_histories:
        _write_json(os.path.join(output_root, "seed_histories.json"), all_seed_histories)
    _write_json(os.path.join(output_root, "summary.json"), summary_payload)
    if bundle_filename is not None:
        bundle_payload = {
            "summary": summary_payload,
            "histories": all_histories,
            "comparison_rows": comparison_rows,
        }
        if all_seed_histories:
            bundle_payload["seed_histories"] = all_seed_histories
        _write_json(os.path.join(output_root, bundle_filename), bundle_payload)

    print_table(result_table_title or f"{title} Results", comparison_rows, DEFAULT_RESULT_COLUMNS)

    if plot_results:
        plot_standard_history_bundle(output_root=output_root, histories=all_histories)

    print(f"Saved experiment outputs to {output_root}")
    return {
        "summary": summary_payload,
        "histories": all_histories,
        "comparison_rows": comparison_rows,
        "output_root": output_root,
        "seed_histories": all_seed_histories if all_seed_histories else None,
    }
