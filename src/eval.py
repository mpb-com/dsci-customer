import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def evaluate_models(df: pd.DataFrame) -> None:
    models = [
        col.replace("_p_alive", "") for col in df.columns if col.endswith("_p_alive")
    ]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. P_alive distributions
    axes[0, 0].hist([df[f"{model}_p_alive"] for model in models], bins=10, label=models)
    axes[0, 0].set_title("P_alive Distributions")
    axes[0, 0].set_xlabel("Probability Alive")
    axes[0, 0].legend()

    # 2. Status counts
    status_data = []
    for model in models:
        status_counts = df[f"{model}_status"].value_counts()
        status_data.append(
            [status_counts.get(status, 0) for status in ["Active", "Lapsing", "Lost"]]
        )

    x = np.arange(3)
    width = 0.1
    for i, model in enumerate(models):
        axes[0, 1].bar(x + i * width, status_data[i], width, label=model)
    axes[0, 1].set_title("Customer Status Distribution")
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(["Active", "Lapsing", "Lost"])
    axes[0, 1].legend()

    # 3. Future transactions vs frequency
    for model in models:
        axes[1, 0].scatter(
            df["frequency"],
            df[f"{model}_future_transactions"],
            alpha=0.1,
            s=1,
            label=model,
        )
    axes[1, 0].set_title("Future Transactions vs Frequency")
    axes[1, 0].set_xlabel("Frequency")
    axes[1, 0].set_ylabel("Future Transactions")
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 50)

    # 4. P_alive vs days_since_last
    for model in models:
        axes[1, 1].scatter(
            df["days_since_last"], df[f"{model}_p_alive"], alpha=0.1, s=1, label=model
        )
    axes[1, 1].set_title("P_alive vs Days Since Last")
    axes[1, 1].set_xlabel("Days Since Last Transaction")
    axes[1, 1].set_ylabel("Probability Alive")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Individual P_alive vs days_since_last plots
    n_models = len(models)
    n_cols = (n_models + 1) // 2  # Ceiling division for columns
    _, axes2 = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    axes2 = axes2.flatten() if n_models > 1 else [axes2]

    for i, model in enumerate(models):
        axes2[i].scatter(
            df["days_since_last"], df[f"{model}_p_alive"], alpha=0.1, s=1, color=f"C{i}"
        )
        axes2[i].set_title(f"{model} P_alive vs Days Since Last")
        axes2[i].set_xlabel("Days Since Last Transaction")
        axes2[i].set_ylabel("Probability Alive")
        axes2[i].set_ylim(0, 1.05)

    # Hide unused subplots
    for i in range(n_models, len(axes2)):
        axes2[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    summary_stats = []
    for model in models:
        stats = {
            "Model": model,
            "Avg P_alive": df[f"{model}_p_alive"].mean(),
            "P_alive Std": df[f"{model}_p_alive"].std(),
            "Active %": (df[f"{model}_status"] == "Active").mean() * 100,
            "Lapsing %": (df[f"{model}_status"] == "Lapsing").mean() * 100,
            "Lost %": (df[f"{model}_status"] == "Lost").mean() * 100,
            "Avg Future Txns": df[f"{model}_future_transactions"].mean(),
        }
        summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)

    return summary_df
