import os
import json
import matplotlib.pyplot as plt
from EvaluateTools.evaluate import evaluate
from TrainTools.train import train

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Effect of Learning-Rate Scheduling on Late-Stage Optimization
# ─────────────────────────────────────────────────────────────────────────────


def run_official_evaluation(train_metrics, save_dir, log_dir):
    config = train_metrics["config"]
    ckpt_name = os.path.basename(train_metrics["best_ckpt_path"])

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
    )

def run_experiment2():
    # Experimental conditions
    schedulers_to_test = ["none", "step", "cosine"]
    
    # Controlled Variables
    num_steps = 60000        # keep the same training budget, adjust if it's too long
    seed = 42
    optimizer = "sgd_momentum"
    
    results = {}
    histories = {}
    
    for s in schedulers_to_test:
        print(f"\n=======================================================")
        print(f" Starting Experiment Group: {optimizer} + scheduler '{s}' ")
        print(f"=======================================================\n")
        
        # We save models to separate folders to avoid overriding
        current_save_dir = f"_model_exp2_{s}"
        current_log_dir = f"_log_exp2_{s}"
        current_eval_log_dir = os.path.join(current_log_dir, "official_eval")

        # Run training
        train_metrics = train(
            save_dir=current_save_dir,
            log_dir=current_log_dir,
            optimizer_name=optimizer,
            scheduler_name=s,
            num_steps=num_steps,
            seed=seed
            # All other parameters are left as default from train.py
        )

        print(f"\nRunning official evaluation for scheduler '{s}'...")
        eval_metrics = run_official_evaluation(
            train_metrics=train_metrics,
            save_dir=current_save_dir,
            log_dir=current_eval_log_dir,
        )

        results[s] = {
            "best_f1": train_metrics["best_f1"],
            "best_em": train_metrics["best_em"],
            "best_step": train_metrics["best_step"],
            "final_eval_f1": eval_metrics["f1"],
            "final_eval_em": eval_metrics["exact_match"],
            "final_eval_loss": eval_metrics["loss"],
            "ckpt_path": train_metrics["ckpt_path"],
            "best_ckpt_path": train_metrics["best_ckpt_path"],
        }
        histories[s] = train_metrics["history"]

        # Save individual experiment history config just in case
        with open(os.path.join(current_log_dir, "history.json"), "w") as f:
            json.dump(train_metrics["history"], f, indent=2)

        with open(os.path.join(current_eval_log_dir, "metrics.json"), "w") as f:
            json.dump(eval_metrics, f, indent=2)
            
    # Save combined results
    print("\nTraining completed.")
    print("Aggregate Results:")
    for s, res in results.items():
        print(
            f"  {s}: Best Step = {res['best_step']}, "
            f"Best F1 = {res['best_f1']:.4f}, Best EM = {res['best_em']:.4f}, "
            f"Official Eval F1 = {res['final_eval_f1']:.4f}, "
            f"Official Eval EM = {res['final_eval_em']:.4f}"
        )
        
    with open("experiment2_results.json", "w") as f:
        json.dump({
            "summary": results,
            "histories": histories
        }, f, indent=2)
        
    # Visualizations
    plot_results(histories)

def plot_results(histories):
    print("Generating plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Extract steps
    # We assume 'none' history has the same steps as others for x-axis
    for s, hist in histories.items():
        steps = [h["step"] for h in hist]
        train_loss = [h["train_loss"] for h in hist]
        dev_f1 = [h["dev_f1"] for h in hist]
        
        axes[0].plot(steps, train_loss, label=s)
        axes[1].plot(steps, dev_f1, label=s)
        
        # Gap analysis: train EM vs Dev EM, or train F1 vs Dev F1
        train_f1 = [h["train_f1"] for h in hist]
        gap = [tr - dev for tr, dev in zip(train_f1, dev_f1)]
        axes[2].plot(steps, gap, label=s)
        
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    axes[1].set_title("Dev F1")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()

    axes[2].set_title("Train - Dev F1 Gap")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Gap (Overfitting)")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("experiment2_curves.png")
    print("Saved plots to experiment2_curves.png")

if __name__ == "__main__":
    run_experiment2()
