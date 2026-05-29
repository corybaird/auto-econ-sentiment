"""Generate figures and tables for the lexical arXiv paper draft.

This script intentionally lives outside ``reports/overleaf`` because that
directory is a separate git repository synced to Overleaf. The script writes
finished manuscript assets into ``reports/overleaf/figures`` so Overleaf can
render them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/sentiment/cb_speeches/sentiment_all_results.parquet.gzip"
OVERLEAF_FIGURES = PROJECT_ROOT / "reports/overleaf/figures"

FIGURE_DPI = 300

DICT_LABELS = {
    "correa": "Correa",
    "hubert": "Hubert-Labondance",
    "lm": "Loughran-McDonald",
    "hiv": "General Inquirer",
    "bn": "Bennani-Neuenkirch",
    "ap": "Apel-Blix Grimaldi",
}

METHOD_LABELS = {
    "posneg": "Pos-neg normalized",
    "allwords": "All-words normalized",
}

NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]


def load_sentiment_results(path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load the paper sentiment output table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected sentiment output at {path}. Run the CBS speeches pipeline first."
        )
    return pd.read_parquet(path)


def sentiment_columns(df: pd.DataFrame) -> list[str]:
    """Return final sentiment score columns, excluding matched-word metadata."""
    return [
        col
        for col in df.columns
        if "_sentiment_" in col and pd.api.types.is_numeric_dtype(df[col])
    ]


def parse_sentiment_column(col: str) -> tuple[str, str]:
    """Parse dictionary and aggregation method from a sentiment column."""
    dictionary, rest = col.split("_sentiment_", maxsplit=1)
    method = rest.replace("_stem", "")
    return DICT_LABELS.get(dictionary, dictionary), METHOD_LABELS.get(method, method)


def clean_axis(ax: plt.Axes) -> None:
    """Apply consistent paper figure axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.7)
    ax.set_axisbelow(True)


def save_pdf(fig: plt.Figure, path: Path) -> None:
    """Save a tight PDF figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)


def shade_nber_recessions(ax: plt.Axes, label_first: bool = True) -> None:
    """Shade NBER-dated US recessions on a datetime x-axis."""
    for idx, (start, end) in enumerate(NBER_RECESSIONS):
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color="#b8b8b8",
            alpha=0.28,
            linewidth=0,
            label="NBER recession" if label_first and idx == 0 else None,
            zorder=0,
        )


def make_pipeline_architecture(output_dir: Path = OVERLEAF_FIGURES) -> None:
    """Create the package workflow diagram."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="paper", font_scale=1.1)

    stages = [
        ("YAML / Python API", "inputs, cleaning rules,\ndictionaries"),
        ("Load Text", "CSV or parquet\ntext + dates"),
        ("Clean Text", "normalize, tokenize,\nstem when requested"),
        ("Lexical Scoring", "multiple dictionaries\nand methods"),
        ("Export Results", "scores, matched words,\nreproducible tables"),
    ]

    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_color = "#f3f6f8"
    edge_color = "#2f4858"
    arrow_color = "#5b6770"
    title_color = "#102a43"
    text_color = "#334e68"

    for idx, (title, subtitle) in enumerate(stages):
        x = idx + 0.08
        rect = plt.Rectangle(
            (x, 0.25),
            0.78,
            0.5,
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.39,
            0.57,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            x + 0.39,
            0.41,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
        )
        if idx < len(stages) - 1:
            ax.annotate(
                "",
                xy=(idx + 1.02, 0.5),
                xytext=(idx + 0.88, 0.5),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=arrow_color),
            )

    save_pdf(fig, output_dir / "pipeline_architecture.pdf")


def make_sentiment_distributions(df: pd.DataFrame, output_dir: Path = OVERLEAF_FIGURES) -> None:
    """Create the dictionary-method sentiment distribution figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = sentiment_columns(df)
    if not cols:
        raise ValueError("No sentiment columns found.")

    plot_df = df[cols].melt(var_name="measure", value_name="sentiment").dropna()
    parsed = plot_df["measure"].apply(parse_sentiment_column)
    plot_df["dictionary"] = parsed.apply(lambda item: item[0])
    plot_df["method"] = parsed.apply(lambda item: item[1])

    order = [DICT_LABELS[key] for key in ["correa", "hubert", "lm", "hiv", "bn", "ap"]]
    palette = {
        "Pos-neg normalized": "#2a6f97",
        "All-words normalized": "#c77d2b",
    }

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    grid = sns.FacetGrid(
        plot_df,
        col="method",
        sharex=False,
        sharey=False,
        height=4.0,
        aspect=1.1,
        despine=False,
    )
    grid.map_dataframe(
        sns.violinplot,
        x="sentiment",
        y="dictionary",
        order=order,
        hue="method",
        palette=palette,
        inner="quartile",
        cut=0,
        linewidth=0.8,
        legend=False,
    )
    grid.set_axis_labels("Sentiment score", "")
    grid.set_titles("{col_name}")
    for ax in grid.axes.flat:
        ax.axvline(1.0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.8)
        clean_axis(ax)
        ax.grid(True, axis="x", color="#e8e8e8", linewidth=0.7)
    grid.figure.suptitle("Lexical sentiment distributions vary by dictionary and aggregation method", y=1.03)
    save_pdf(grid.figure, output_dir / "sentiment_distributions.pdf")


def make_correlation_heatmap(df: pd.DataFrame, output_dir: Path = OVERLEAF_FIGURES) -> None:
    """Create the pairwise dictionary-method correlation heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = sentiment_columns(df)
    if not cols:
        raise ValueError("No sentiment columns found.")
    labels = {
        col: f"{parse_sentiment_column(col)[0]}\n{parse_sentiment_column(col)[1].replace(' normalized', '')}"
        for col in cols
    }
    corr = df[cols].corr().rename(index=labels, columns=labels)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.set_theme(style="white", context="paper", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(10.5, 9))
    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9.5, "fontweight": "bold"},
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Correlation", "shrink": 0.75},
    )
    ax.set_title("Pairwise correlations across lexical sentiment measures", pad=14)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    save_pdf(fig, output_dir / "sentiment_correlation_heatmap.pdf")


def make_rolling_time_series(df: pd.DataFrame, output_dir: Path = OVERLEAF_FIGURES) -> None:
    """Create the rolling central-bank sentiment time-series figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = sentiment_columns(df)
    if not cols:
        raise ValueError("No sentiment columns found.")

    required_cols = ["date", "CentralBank"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for rolling time series: {missing}")

    selected = [
        "correa_sentiment_allwords",
        "hubert_sentiment_allwords",
        "lm_sentiment_allwords",
        "hiv_sentiment_allwords",
        "bn_sentiment_allwords_stem",
        "ap_sentiment_allwords_stem",
    ]
    selected = [col for col in selected if col in df.columns]

    fed = df[df["CentralBank"].eq("Board_of_Governors_of_the_Federal_Reserve")].copy()
    if fed.empty:
        fed = df.copy()
        title_subject = "all central banks"
    else:
        title_subject = "Board of Governors of the Federal Reserve"

    fed["date"] = pd.to_datetime(fed["date"], errors="coerce")
    fed = fed.dropna(subset=["date"])
    monthly = fed.set_index("date")[selected].resample("ME").mean()
    rolling = monthly.rolling(window=6, min_periods=3).mean()

    labels = {col: parse_sentiment_column(col)[0] for col in selected}
    colors = ["#245c7a", "#9b3d2e", "#4f7f3f", "#6d597a", "#b7791f", "#2c7a7b"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    shade_nber_recessions(ax)
    for color, col in zip(colors, selected):
        ax.plot(
            rolling.index,
            rolling[col],
            label=labels[col],
            linewidth=1.9,
            color=color,
            alpha=0.95,
        )

    ax.axhline(1.0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_title(f"Rolling all-words sentiment over time: {title_subject}")
    ax.set_xlabel("")
    ax.set_ylabel("Six-month rolling mean all-words sentiment")
    ax.legend(ncol=2, frameon=False, loc="best")
    clean_axis(ax)
    fig.autofmt_xdate()
    save_pdf(fig, output_dir / "sentiment_rolling_time_series.pdf")


def make_rolling_change_time_series(df: pd.DataFrame, output_dir: Path = OVERLEAF_FIGURES) -> None:
    """Create a rolling first-difference sentiment time-series figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    required_cols = ["date", "CentralBank"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for rolling changes: {missing}")

    selected = [
        "correa_sentiment_allwords",
        "hubert_sentiment_allwords",
        "lm_sentiment_allwords",
        "hiv_sentiment_allwords",
        "bn_sentiment_allwords_stem",
        "ap_sentiment_allwords_stem",
    ]
    selected = [col for col in selected if col in df.columns]

    fed = df[df["CentralBank"].eq("Board_of_Governors_of_the_Federal_Reserve")].copy()
    if fed.empty:
        fed = df.copy()
        title_subject = "all central banks"
    else:
        title_subject = "Board of Governors of the Federal Reserve"

    fed["date"] = pd.to_datetime(fed["date"], errors="coerce")
    fed = fed.dropna(subset=["date"])
    monthly = fed.set_index("date")[selected].resample("ME").mean()
    rolling_changes = monthly.diff().rolling(window=6, min_periods=3).mean()

    labels = {col: parse_sentiment_column(col)[0] for col in selected}
    colors = ["#245c7a", "#9b3d2e", "#4f7f3f", "#6d597a", "#b7791f", "#2c7a7b"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    shade_nber_recessions(ax)
    for color, col in zip(colors, selected):
        ax.plot(
            rolling_changes.index,
            rolling_changes[col],
            label=labels[col],
            linewidth=1.8,
            color=color,
            alpha=0.95,
        )

    ax.axhline(0.0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_title(f"Rolling changes in all-words sentiment: {title_subject}")
    ax.set_xlabel("")
    ax.set_ylabel("Six-month rolling mean monthly change")
    ax.legend(ncol=2, frameon=False, loc="best")
    clean_axis(ax)
    fig.autofmt_xdate()
    save_pdf(fig, output_dir / "sentiment_rolling_changes_time_series.pdf")


def make_change_correlation_heatmap(df: pd.DataFrame, output_dir: Path = OVERLEAF_FIGURES) -> None:
    """Create a triangular heatmap for correlations in monthly sentiment changes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = sentiment_columns(df)
    if not cols:
        raise ValueError("No sentiment columns found.")
    if "date" not in df.columns:
        raise ValueError("Missing date column for change correlation heatmap.")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    monthly = work.set_index("date")[cols].resample("ME").mean()
    changes = monthly.diff().dropna(how="all")

    labels = {
        col: f"{parse_sentiment_column(col)[0]}\n{parse_sentiment_column(col)[1].replace(' normalized', '')}"
        for col in cols
    }
    corr = changes.corr().rename(index=labels, columns=labels)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.set_theme(style="white", context="paper", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(10.5, 9))
    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9.5, "fontweight": "bold"},
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Correlation of monthly changes", "shrink": 0.75},
    )
    ax.set_title("Pairwise correlations in monthly changes across lexical sentiment measures", pad=14)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    save_pdf(fig, output_dir / "sentiment_change_correlation_heatmap.pdf")


def main() -> None:
    df = load_sentiment_results()
    make_pipeline_architecture()
    make_sentiment_distributions(df)
    make_correlation_heatmap(df)
    make_rolling_time_series(df)
    make_rolling_change_time_series(df)
    make_change_correlation_heatmap(df)


if __name__ == "__main__":
    main()
