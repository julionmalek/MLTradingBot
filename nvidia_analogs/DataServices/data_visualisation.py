import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from pathlib import Path

# ─── Setup Directories ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Parameters ─────────────────────────────────────────────────────
CSV_DIR = DATA_DIR / "AnalogFullBacktests"
LATEST_YEARS = 2  # Number of years to keep
SIGNAL_THRESHOLD = 0.02  # Only act if avg_analog_ret > 3%

# ─── Loaders ────────────────────────────────────────────────────────
def load_csv_files(directory):
    dataframes = {}
    for file in directory.glob("*_analog_backtest.csv"):
        df = pd.read_csv(file, parse_dates=["date"], index_col="date")
        df.index = df.index.normalize()
        dataframes[file.stem] = df
    return dataframes


def filter_last_years(df, years=2):
    cutoff = df.index.max() - pd.DateOffset(years=years)
    return df[df.index >= cutoff]


def apply_signal_threshold(df, threshold=SIGNAL_THRESHOLD):
    return df[df['avg_analog_ret'] >= threshold]

# ─── Analysis ───────────────────────────────────────────────────────
def directionality_test(df):
    correct_direction = (np.sign(df['avg_analog_ret']) == np.sign(df['actual_ret']))
    direction_accuracy = correct_direction.mean()
    direction_diff = (df['avg_analog_ret'] - df['actual_ret']).abs()

    print(f"Direction Agreement Rate: {direction_accuracy:.2%}")
    print(f"Mean Absolute Difference: {direction_diff.mean():.4f}")

    return direction_accuracy, direction_diff.mean()


def plot_avg_analog_vs_actual(df, symbol):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['avg_analog_ret'], label='Avg Analog Return')
    plt.plot(df.index, df['actual_ret'], label='Actual Return')
    plt.title(f'{symbol} - Avg Analog Return vs Actual Return (Last {LATEST_YEARS} Years)')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid()
    plt.show()


def rolling_correlation_analysis(df, window_size=90):
    rolling_corr = df['avg_analog_ret'].rolling(window=window_size).corr(df['actual_ret'])
    return rolling_corr


def plot_rolling_correlation(rolling_corr, symbol):
    plt.figure(figsize=(14, 6))
    rolling_corr.plot()
    plt.title(f'{symbol} - Rolling Correlation (Last {LATEST_YEARS} Years)')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.grid()
    plt.show()


def scatter_plot_correlation(df, symbol):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='avg_analog_ret', y='actual_ret', data=df)
    plt.title(f'{symbol} - Scatter Plot (Last {LATEST_YEARS} Years)')
    plt.xlabel('Avg Analog Return')
    plt.ylabel('Actual Return')
    plt.grid()
    plt.show()


def plot_confusion(df, symbol):
    y_true = np.sign(df['actual_ret'])
    y_pred = np.sign(df['avg_analog_ret'])
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0, -1])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Up', 'Flat', 'Down'])
    disp.plot(cmap='Blues')
    plt.title(f'{symbol} - Direction Prediction Confusion Matrix')
    plt.grid(False)
    plt.show()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=[1, 0, -1], target_names=['Up', 'Flat', 'Down'], zero_division=0))


# ─── Main ───────────────────────────────────────────────────────────
def main():
    dfs = load_csv_files(CSV_DIR)
    results = []

    for symbol, df in dfs.items():
        print(f"\nAnalyzing {symbol}")

        if 'avg_analog_ret' in df.columns and 'actual_ret' in df.columns:
            df_filtered = filter_last_years(df, years=LATEST_YEARS)
            df_filtered = apply_signal_threshold(df_filtered)

            if df_filtered.empty:
                print(f"⚠️  No signals passed the threshold for {symbol}")
                continue

            #plot_avg_analog_vs_actual(df_filtered, symbol)
            rolling_corr = rolling_correlation_analysis(df_filtered)
            #plot_rolling_correlation(rolling_corr, symbol)
            #scatter_plot_correlation(df_filtered, symbol)
            direction_accuracy, mean_abs_diff = directionality_test(df_filtered)
            #plot_confusion(df_filtered, symbol)

            results.append({
                "Symbol": symbol,
                "Direction_Agreement_Rate": direction_accuracy,
                "Mean_Absolute_Difference": mean_abs_diff,
                "Num_Samples": len(df_filtered)
            })
        else:
            print(f"⚠️  Missing required columns in {symbol}")

    if results:
        results_df = pd.DataFrame(results)
        results_path = RESULTS_DIR / "directionality_summary.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n✔️ Summary results saved to {results_path}")


if __name__ == "__main__":
    main()