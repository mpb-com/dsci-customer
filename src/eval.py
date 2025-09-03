import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    brier_score_loss,
)
from sksurv.metrics import concordance_index_censored


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def poisson_deviance(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.maximum(y_pred, 1e-10)
    return 2 * np.mean(y_pred - y_true * np.log(y_pred))


class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_p_alive(self, y_true, y_prob, model_name):
        from sklearn.metrics import roc_curve, auc
        from sklearn.calibration import calibration_curve
        
        # Calculate ROC metrics
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Expected Calibration Error (ECE)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10
        )
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        metrics = {
            "log_loss": log_loss(y_true, y_prob),
            "brier_score": brier_score_loss(y_true, y_prob),
            "auc": roc_auc,
            "ece": ece,
            "concordance_index": concordance_index_censored(
                y_true.astype(bool), np.ones_like(y_true), -y_prob
            )[0],
        }

        self.results[f"{model_name}_p_alive"] = metrics
        return metrics

    def evaluate_future_transactions(self, y_true, y_pred, model_name):
        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mape": mape(y_true, y_pred),
            "poisson_deviance": poisson_deviance(y_true, y_pred),
        }

        self.results[f"{model_name}_future_txns"] = metrics
        return metrics

    def evaluate_model(self, df, model_name, y_true_alive=None, y_true_txns=None):
        results = {}

        if f"{model_name}_p_alive" in df.columns and y_true_alive is not None:
            results["p_alive"] = self.evaluate_p_alive(
                y_true_alive, df[f"{model_name}_p_alive"], model_name
            )

        if (
            f"{model_name}_future_transactions" in df.columns
            and y_true_txns is not None
        ):
            results["future_txns"] = self.evaluate_future_transactions(
                y_true_txns, df[f"{model_name}_future_transactions"], model_name
            )

        return results

    def compare_models(self, df, y_true_alive=None, y_true_txns=None):
        models = [
            col.replace("_p_alive", "")
            for col in df.columns
            if col.endswith("_p_alive")
        ]
        return {
            model: self.evaluate_model(df, model, y_true_alive, y_true_txns)
            for model in models
        }

    def summary_table(self):
        summary_data = []

        for key, metrics in self.results.items():
            if "_p_alive" in key:
                model_name = key.replace("_p_alive", "")
                row = {"Model": model_name, "Metric_Type": "P_Alive", **metrics}
                summary_data.append(row)
            elif "_future_txns" in key:
                model_name = key.replace("_future_txns", "")
                row = {"Model": model_name, "Metric_Type": "Future_Txns", **metrics}
                summary_data.append(row)

        return pd.DataFrame(summary_data)

    def plot_results(self, df, y_true_alive=None, y_true_txns=None):
        """Plot evaluation results"""
        models = [
            col.replace("_p_alive", "")
            for col in df.columns
            if col.endswith("_p_alive")
        ]

        plt.figure(figsize=(12, 8))

        # P_alive distributions
        plt.subplot(2, 2, 1)
        for model in models:
            plt.hist(df[f"{model}_p_alive"], bins=20, alpha=0.6, label=model)
        plt.title("P_alive Distributions")
        plt.legend()

        # Future transactions vs actual
        if y_true_txns is not None:
            plt.subplot(2, 2, 2)
            for model in models:
                plt.scatter(
                    y_true_txns,
                    df[f"{model}_future_transactions"],
                    alpha=0.3,
                    s=1,
                    label=model,
                )
            plt.plot([0, max(y_true_txns)], [0, max(y_true_txns)], "k--", alpha=0.5)
            plt.xlabel("Actual Transactions")
            plt.ylabel("Predicted Transactions")
            plt.title("Predicted vs Actual")
            plt.legend()

        # Metrics comparison
        if self.results:
            summary = self.summary_table()

            # Concordance Index comparison
            plt.subplot(2, 2, 3)
            p_alive_data = summary[summary["Metric_Type"] == "P_Alive"]
            if not p_alive_data.empty and "log_loss" in p_alive_data.columns:
                plt.bar(p_alive_data["Model"], p_alive_data["log_loss"])
                plt.title("Log Loss")
                plt.xticks(rotation=45)

            # MAE comparison
            plt.subplot(2, 2, 4)
            txns_data = summary[summary["Metric_Type"] == "Future_Txns"]
            if not txns_data.empty and "mae" in txns_data.columns:
                plt.bar(txns_data["Model"], txns_data["mae"])
                plt.title("MAE Comparison")
                plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_calibration_purchases(self, df, y_true_txns, model_name):
        """Plot calibration like lifetimes package - line plot with actual vs predicted"""

        if f"{model_name}_future_transactions" not in df.columns:
            return

        # Use frequency as calibration period purchases
        calibration_txns = df["frequency"].values
        holdout_actual = df["y_true_txns"].values  # Use ground truth from dataframe
        holdout_predicted = df[f"{model_name}_future_transactions"].values

        # Create bins based on frequency values with minimum sample size
        min_bin_size = 50  # Minimum customers per bin
        max_freq = min(calibration_txns.max(), 20)  # Cap at 20 to avoid too many bins

        # Create bins with sufficient sample sizes
        binned_calibration = []
        binned_holdout_actual = []
        binned_holdout_predicted = []
        bin_counts = []

        for freq in range(int(max_freq) + 1):
            mask = calibration_txns == freq
            count = mask.sum()

            if count >= min_bin_size:
                binned_calibration.append(freq)
                binned_holdout_actual.append(holdout_actual[mask].mean())
                binned_holdout_predicted.append(holdout_predicted[mask].mean())
                bin_counts.append(count)

        # If we don't have enough individual frequency bins, group them
        if len(binned_calibration) < 5:
            binned_calibration = []
            binned_holdout_actual = []
            binned_holdout_predicted = []
            bin_counts = []

            # Group frequencies: 0, 1, 2-3, 4-6, 7+
            freq_groups = [(0, 0), (1, 1), (2, 3), (4, 6), (7, max_freq)]

            for min_f, max_f in freq_groups:
                mask = (calibration_txns >= min_f) & (calibration_txns <= max_f)
                count = mask.sum()

                if count >= min_bin_size:
                    binned_calibration.append(calibration_txns[mask].mean())
                    binned_holdout_actual.append(holdout_actual[mask].mean())
                    binned_holdout_predicted.append(holdout_predicted[mask].mean())
                    bin_counts.append(count)

        # Plot
        plt.figure(figsize=(10, 6))

        plt.plot(
            binned_calibration,
            binned_holdout_actual,
            "o-",
            label="Actual Holdout",
            linewidth=2,
            markersize=8,
        )
        plt.plot(
            binned_calibration,
            binned_holdout_predicted,
            "s-",
            label=f"Predicted Holdout ({model_name})",
            linewidth=2,
            markersize=8,
        )

        plt.xlabel("Purchases in Calibration Period")
        plt.ylabel("Purchases in Holdout Period")
        plt.title(f"Calibration vs Holdout - {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add sample size annotations
        for x, y, count in zip(binned_calibration, binned_holdout_actual, bin_counts):
            plt.annotate(
                f"n={count}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        plt.tight_layout()
        plt.show()


    def plot_all_calibrations(self, df, y_true_txns):
        """Plot calibration and ROC curves for all models in separate 2x2 grids"""
        models = [
            col.replace("_future_transactions", "")
            for col in df.columns
            if col.endswith("_future_transactions")
        ]

        if not models:
            print("No models found for plotting")
            return

        # Calculate grid dimensions - always 2 rows, scale columns
        n_models = len(models)
        n_cols = (n_models + 1) // 2  # Ceiling division for columns

        # Plot 1: Transaction Calibration plots
        _, axes1 = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
        axes1 = axes1.flatten() if n_models > 1 else [axes1]

        for i, model in enumerate(models):
            if f"{model}_future_transactions" in df.columns:
                self._plot_transaction_calibration(df, model, axes1[i])
            else:
                axes1[i].text(
                    0.5,
                    0.5,
                    f"No transaction data for {model}",
                    ha="center",
                    va="center",
                    transform=axes1[i].transAxes,
                )
                axes1[i].set_title(f"Transaction Calibration - {model} (No Data)")

        # Hide unused subplots
        for i in range(n_models, len(axes1)):
            axes1[i].set_visible(False)

        plt.suptitle("Transaction Calibration Plots", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()

        # Plot 2: ROC curves
        _, axes2 = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
        axes2 = axes2.flatten() if n_models > 1 else [axes2]

        for i, model in enumerate(models):
            if f"{model}_p_alive" in df.columns:
                self._plot_roc_curve(df, model, axes2[i])
            else:
                axes2[i].text(
                    0.5,
                    0.5,
                    f"No p_alive data for {model}",
                    ha="center",
                    va="center",
                    transform=axes2[i].transAxes,
                )
                axes2[i].set_title(f"ROC Curve - {model} (No Data)")

        # Hide unused subplots
        for i in range(n_models, len(axes2)):
            axes2[i].set_visible(False)

        plt.suptitle("ROC Curves", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()

    def plot_model_calibrations(self, df, y_true_txns, model_name):
        """Plot both transaction and probability calibrations for a single model in 2-row grid"""

        if f"{model_name}_future_transactions" not in df.columns:
            return

        # Create 2x1 subplot grid
        _, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Top plot: Future transactions calibration
        self._plot_transaction_calibration(df, model_name, axes[0])

        # Bottom plot: Probability alive calibration
        if f"{model_name}_p_alive" in df.columns:
            self._plot_probability_calibration(df, model_name, axes[1])
        else:
            axes[1].text(
                0.5,
                0.5,
                f"No p_alive predictions for {model_name}",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_title(f"P_alive Calibration - {model_name} (No Data)")

        plt.tight_layout()
        plt.show()

    def _plot_transaction_calibration(self, df, model_name, ax):
        """Plot transaction count calibration on given axis"""

        # Use frequency as calibration period purchases
        calibration_txns = df["frequency"].values
        holdout_actual = df["y_true_txns"].values
        holdout_predicted = df[f"{model_name}_future_transactions"].values

        # Create bins with sufficient sample sizes
        (
            binned_calibration,
            binned_holdout_actual,
            binned_holdout_predicted,
            bin_counts,
        ) = self._create_calibration_bins(
            calibration_txns, holdout_actual, holdout_predicted
        )

        # Plot on provided axis
        ax.plot(
            binned_calibration,
            binned_holdout_actual,
            "o-",
            label="Actual Holdout",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            binned_calibration,
            binned_holdout_predicted,
            "s-",
            label="Predicted Holdout",
            linewidth=2,
            markersize=8,
        )

        ax.set_xlabel("Purchases in Calibration Period")
        ax.set_ylabel("Purchases in Holdout Period")
        ax.set_title(f"Transaction Calibration - {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add sample size annotations
        for x, y, count in zip(binned_calibration, binned_holdout_actual, bin_counts):
            ax.annotate(
                f"n={count}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    def _plot_roc_curve(self, df, model_name, ax):
        """Plot ROC curve for p_alive predictions"""

        actual_alive = df["y_true_alive"].values
        predicted_prob = df[f"{model_name}_p_alive"].values

        # Calculate ROC curve
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(actual_alive, predicted_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")

        # Plot random classifier line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"P_alive ROC Curve - {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def _create_calibration_bins(
        self, calibration_values, actual_values, predicted_values
    ):
        """Create calibration bins with sufficient sample sizes"""
        min_bin_size = 50
        max_freq = min(calibration_values.max(), 20)

        binned_calibration = []
        binned_actual = []
        binned_predicted = []
        bin_counts = []

        for freq in range(int(max_freq) + 1):
            mask = calibration_values == freq
            count = mask.sum()

            if count >= min_bin_size:
                binned_calibration.append(freq)
                binned_actual.append(actual_values[mask].mean())
                binned_predicted.append(predicted_values[mask].mean())
                bin_counts.append(count)

        # If we don't have enough individual frequency bins, group them
        if len(binned_calibration) < 5:
            binned_calibration = []
            binned_actual = []
            binned_predicted = []
            bin_counts = []

            freq_groups = [(0, 0), (1, 1), (2, 3), (4, 6), (7, max_freq)]

            for min_f, max_f in freq_groups:
                mask = (calibration_values >= min_f) & (calibration_values <= max_f)
                count = mask.sum()

                if count >= min_bin_size:
                    binned_calibration.append(calibration_values[mask].mean())
                    binned_actual.append(actual_values[mask].mean())
                    binned_predicted.append(predicted_values[mask].mean())
                    bin_counts.append(count)

        return binned_calibration, binned_actual, binned_predicted, bin_counts


def evaluate_models(df: pd.DataFrame) -> None:
    models = [
        col.replace("_p_alive", "") for col in df.columns if col.endswith("_p_alive")
    ]

    # Create subplots
    _, axes = plt.subplots(2, 2, figsize=(15, 12))

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
