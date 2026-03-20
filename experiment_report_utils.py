import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


DEFAULT_RESULT_COLUMNS = [
    "condition",
    "best_f1_during_training",
    "best_em_during_training",
    "best_step_during_training",
    "official_eval_f1",
    "official_eval_em",
    "official_eval_loss",
    "best_logged_dev_f1",
]


def read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(title: str, rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
    if not rows:
        print(f"\n{title}\n(no data)")
        return

    if columns is None:
        columns = list(rows[0].keys())

    headers = {col: col.replace("_", " ") for col in columns}
    widths = {
        col: max(len(headers[col]), max(len(_format_cell(row.get(col))) for row in rows))
        for col in columns
    }

    def fmt_row(row: Dict[str, Any]) -> str:
        return " | ".join(_format_cell(row.get(col)).ljust(widths[col]) for col in columns)

    header_line = " | ".join(headers[col].ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)

    print(f"\n{title}")
    print(header_line)
    print(separator)
    for row in rows:
        print(fmt_row(row))


def _get_xy(hist: List[Dict[str, Any]], key: str):
    x, y = [], []
    for row in hist:
        if "step" in row and key in row and row[key] is not None:
            x.append(row["step"])
            y.append(row[key])
    return x, y


def plot_metric(
    histories: Dict[str, List[Dict[str, Any]]],
    key: str,
    title: str,
    out_path: str,
) -> None:
    plotted = False
    plt.figure(figsize=(7, 4))
    for cond, hist in histories.items():
        x, y = _get_xy(hist, key)
        if not x:
            continue
        plt.plot(x, y, label=cond)
        plotted = True

    if plotted:
        plt.xlabel("Training step")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.show()
    plt.close()


def plot_gap(histories: Dict[str, List[Dict[str, Any]]], out_path: str) -> None:
    plotted = False
    plt.figure(figsize=(7, 4))
    for cond, hist in histories.items():
        x, y = [], []
        for row in hist:
            if (
                "step" in row
                and "train_f1" in row and row["train_f1"] is not None
                and "dev_f1" in row and row["dev_f1"] is not None
            ):
                x.append(row["step"])
                y.append(row["train_f1"] - row["dev_f1"])
        if not x:
            continue
        plt.plot(x, y, label=cond)
        plotted = True

    if plotted:
        plt.xlabel("Training step")
        plt.ylabel("Train F1 - Dev F1")
        plt.title("Generalization Gap")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.show()
    plt.close()


def plot_standard_history_bundle(
    output_root: str,
    histories: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> None:
    if histories is None:
        histories = read_json(os.path.join(output_root, "histories.json"))

    plot_metric(histories, "train_loss", "Train Loss", os.path.join(output_root, "train_loss.png"))
    plot_metric(histories, "dev_loss", "Dev Loss", os.path.join(output_root, "dev_loss.png"))
    plot_metric(histories, "dev_f1", "Dev F1", os.path.join(output_root, "dev_f1.png"))
    plot_metric(histories, "dev_em", "Dev EM", os.path.join(output_root, "dev_em.png"))
    plot_gap(histories, os.path.join(output_root, "f1_gap.png"))
