import argparse
import json
import os

from EvaluateTools.evaluate import evaluate
from Tools.download import download_mini
from Tools.preproc import preprocess
from TrainTools.train import train
from run_experiment2 import run_experiment2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the Assignment 1 notebook workflow from a single script."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the mini-dataset download step.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip final evaluation.",
    )
    parser.add_argument(
        "--run-experiment2",
        action="store_true",
        help="Run the scheduler comparison experiment after the main pipeline.",
    )
    parser.add_argument(
        "--data-dir",
        default="_data",
        help="Directory containing raw and preprocessed data.",
    )
    parser.add_argument(
        "--save-dir",
        default="_model",
        help="Directory for training checkpoints.",
    )
    parser.add_argument(
        "--log-dir",
        default="_log",
        help="Directory for logs and predictions.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=60000,
        help="Number of training steps for the main training run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the main training run.",
    )
    return parser


def files_exist(paths):
    return all(os.path.exists(path) for path in paths)


def main():
    args = build_parser().parse_args()

    train_file = f"{args.data_dir}/squad/train-mini.json"
    dev_file = f"{args.data_dir}/squad/dev-v1.1.json"
    glove_file = f"{args.data_dir}/glove/glove.mini.txt"

    train_npz = f"{args.data_dir}/train.npz"
    dev_npz = f"{args.data_dir}/dev.npz"
    word_emb_json = f"{args.data_dir}/word_emb.json"
    char_emb_json = f"{args.data_dir}/char_emb.json"
    train_eval_json = f"{args.data_dir}/train_eval.json"
    dev_eval_json = f"{args.data_dir}/dev_eval.json"

    raw_data_files = [train_file, dev_file, glove_file]
    preprocessed_files = [
        train_npz,
        dev_npz,
        word_emb_json,
        char_emb_json,
        train_eval_json,
        dev_eval_json,
    ]

    training_results = None
    eval_metrics = None

    should_download = not args.skip_download and not files_exist(raw_data_files)
    should_preprocess = not args.skip_preprocess and not files_exist(preprocessed_files)

    if should_download:
        print("\n=== Step 1: Download Mini Dataset ===")
        download_mini(data_dir=args.data_dir)
    elif args.skip_download:
        print("\n=== Step 1: Download Mini Dataset (skipped by flag) ===")
    else:
        print("\n=== Step 1: Download Mini Dataset (already available) ===")

    if should_preprocess:
        print("\n=== Step 2: Preprocess Data ===")
        preprocess(
            train_file=train_file,
            dev_file=dev_file,
            glove_word_file=glove_file,
            target_dir=args.data_dir,
            para_limit=400,
            ques_limit=50,
        )
    elif args.skip_preprocess:
        print("\n=== Step 2: Preprocess Data (skipped by flag) ===")
    else:
        print("\n=== Step 2: Preprocess Data (already available) ===")

    if not args.skip_train:
        print("\n=== Step 3: Train Model ===")
        training_results = train(
            train_npz=train_npz,
            dev_npz=dev_npz,
            word_emb_json=word_emb_json,
            char_emb_json=char_emb_json,
            train_eval_json=train_eval_json,
            dev_eval_json=dev_eval_json,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            optimizer_name="sgd",
            scheduler_name="none",
            loss_name="qa_nll",
        )
        print(
            "Best checkpoint: "
            f"step={training_results['best_step']}  "
            f"F1={training_results['best_f1']:.4f}  "
            f"EM={training_results['best_em']:.4f}"
        )

    if not args.skip_eval:
        print("\n=== Step 4: Evaluate Best Checkpoint ===")
        ckpt_name = "best_model.pt"
        if training_results is not None:
            ckpt_name = training_results["best_ckpt_path"].split("/")[-1]

        eval_metrics = evaluate(
            dev_npz=dev_npz,
            word_emb_json=word_emb_json,
            char_emb_json=char_emb_json,
            dev_eval_json=dev_eval_json,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            ckpt_name=ckpt_name,
            batch_size=args.batch_size,
        )
        print(
            f"Official Eval  F1={eval_metrics['f1']:.4f}  "
            f"EM={eval_metrics['exact_match']:.4f}  "
            f"Loss={eval_metrics['loss']:.6f}"
        )

    if args.run_experiment2:
        print("\n=== Step 5: Run Experiment 2 ===")
        run_experiment2()

    summary = {
        "training_results": training_results,
        "evaluation_metrics": eval_metrics,
    }
    print("\n=== Pipeline Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
