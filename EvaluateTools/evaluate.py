"""
evaluate.py — Evaluation entry point for QANet / Assignment 1.

Usage (from Assignment1.ipynb):
    from EvaluateTools.evaluate import evaluate
    metrics = evaluate()
    metrics = evaluate(save_dir="_model", ckpt_name="model.pt")

Returns
-------
dict with keys: f1, exact_match, loss
"""

import argparse
import os

import torch
import ujson as json

from Data import SQuADDataset, load_dev_eval, load_word_char_mats
from Losses import losses
from Models import QANet
from EvaluateTools.eval_utils import run_eval


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    # ── Data paths ────────────────────────────────────────────────────────────
    dev_npz:        str   = "_data/dev.npz",
    word_emb_json:  str   = "_data/word_emb.json",
    char_emb_json:  str   = "_data/char_emb.json",
    dev_eval_json:  str   = "_data/dev_eval.json",
    save_dir:       str   = "_model",
    log_dir:        str   = "_log",
    ckpt_name:      str   = "model.pt",

    # ── Eval settings ─────────────────────────────────────────────────────────
    batch_size:         int   = 8,
    test_num_batches:   int   = -1,       # -1 = full dev set
    loss_name:          str   = "qa_nll",

    # ── Model architecture (must match the checkpoint) ────────────────────────
    para_limit:     int   = 400,
    ques_limit:     int   = 50,
    char_limit:     int   = 16,
    d_model:        int   = 96,
    num_heads:      int   = 8,
    glove_dim:      int   = 300,
    char_dim:       int   = 64,
    dropout:        float = 0.1,
    dropout_char:   float = 0.05,
    pretrained_char: bool = False,
    norm_name:      str   = "layer_norm",
    norm_groups:    int   = 8,
    activation:     str   = "relu",
    init_name:      str   = "kaiming",
    use_batch_norm: bool  = False,
) -> dict:
    """Evaluate a saved QANet checkpoint on the SQuAD v1.1 dev set.

    Parameters
    ----------
    dev_npz:
        Path to the preprocessed dev record file.
    word_emb_json, char_emb_json:
        Paths to the embedding matrices produced by ``preprocess()``.
    dev_eval_json:
        Path to the dev evaluation metadata (contexts, gold answers).
    save_dir:
        Directory containing the checkpoint file.
    log_dir:
        Directory where ``answers.json`` will be written.
    ckpt_name:
        Filename of the checkpoint inside ``save_dir``.
    batch_size:
        Number of examples per batch.
    test_num_batches:
        Number of batches to evaluate (-1 = entire dev set).
    loss_name:
        Loss function key from the registry (default ``"qa_nll"``).
    para_limit, ques_limit, char_limit, d_model, num_heads,
    glove_dim, char_dim, dropout, dropout_char, pretrained_char,
    norm_name, norm_groups, activation, init_name, use_batch_norm:
        Model / architecture parameters. When the checkpoint contains a
        saved ``config``, those values are used by default so evaluation
        rebuilds the same model that was trained.

    Returns
    -------
    dict
        ``{"f1": float, "exact_match": float, "loss": float}``
    """
    os.makedirs(log_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    ckpt_cfg = ckpt.get("config", {})

    resolved_loss_name = loss_name if loss_name != "qa_nll" else ckpt_cfg.get("loss_name", loss_name)
    if resolved_loss_name not in losses:
        raise ValueError(f"Unknown loss '{resolved_loss_name}'. Available: {list(losses.keys())}")

    default_values = {
        "dev_npz": "_data/dev.npz",
        "word_emb_json": "_data/word_emb.json",
        "char_emb_json": "_data/char_emb.json",
        "dev_eval_json": "_data/dev_eval.json",
        "para_limit": 400,
        "ques_limit": 50,
        "char_limit": 16,
        "d_model": 96,
        "num_heads": 8,
        "glove_dim": 300,
        "char_dim": 64,
        "dropout": 0.1,
        "dropout_char": 0.05,
        "pretrained_char": False,
        "norm_name": "layer_norm",
        "norm_groups": 8,
        "activation": "relu",
        "init_name": "kaiming",
        "use_batch_norm": False,
    }

    def resolve(name, value):
        if value != default_values[name]:
            return value
        return ckpt_cfg.get(name, value)

    # Prefer the checkpoint config so evaluation reconstructs the trained model.
    args = argparse.Namespace(
        dev_npz=resolve("dev_npz", dev_npz),
        word_emb_json=resolve("word_emb_json", word_emb_json),
        char_emb_json=resolve("char_emb_json", char_emb_json),
        dev_eval_json=resolve("dev_eval_json", dev_eval_json),
        para_limit=resolve("para_limit", para_limit),
        ques_limit=resolve("ques_limit", ques_limit),
        char_limit=resolve("char_limit", char_limit),
        d_model=resolve("d_model", d_model),
        num_heads=resolve("num_heads", num_heads),
        glove_dim=resolve("glove_dim", glove_dim),
        char_dim=resolve("char_dim", char_dim),
        dropout=resolve("dropout", dropout),
        dropout_char=resolve("dropout_char", dropout_char),
        pretrained_char=resolve("pretrained_char", pretrained_char),
        norm_name=resolve("norm_name", norm_name),
        norm_groups=resolve("norm_groups", norm_groups),
        activation=resolve("activation", activation),
        init_name=resolve("init_name", init_name),
        use_batch_norm=resolve("use_batch_norm", use_batch_norm),
    )

    word_mat, char_mat = load_word_char_mats(args)
    model = QANet(word_mat, char_mat, args).to(DEVICE)

    dev_eval = load_dev_eval(args)
    dev_dataset = SQuADDataset(args.dev_npz)

    model.load_state_dict(ckpt["model_state"])

    metrics, ans = run_eval(
        model, dev_dataset, dev_eval,
        num_batches=test_num_batches,
        batch_size=batch_size,
        use_random_batches=False,
        device=DEVICE,
        loss_fn=losses[resolved_loss_name],
    )

    with open(os.path.join(log_dir, "answers.json"), "w") as f:
        json.dump(ans, f)

    print("TEST  loss {loss:.6f}  F1 {f1:.6f}  EM {exact_match:.6f}".format(**metrics))
    return {"f1": metrics["f1"], "exact_match": metrics["exact_match"], "loss": metrics["loss"]}
