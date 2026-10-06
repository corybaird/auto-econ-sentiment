"""LaTeX building blocks shared by the paper's tables."""

from __future__ import annotations


def tabular(align: str, header: list[str], rows: list[list[str]], footer: list[str] | None = None) -> str:
    """A booktabs tabular with one header row and optional footer lines after a midrule."""
    body = "\n    ".join(" & ".join(row) + " \\\\" for row in rows)
    if footer:
        body += "\n    \\midrule\n    " + "\n    ".join(footer)
    return (
        f"\\begin{{tabular}}{{{align}}}\n"
        "    \\toprule\n"
        f"    {' & '.join(header).strip()} \\\\\n"
        "    \\midrule\n"
        f"    {body}\n"
        "    \\bottomrule\n"
        "\\end{tabular}\n"
    )


def integer(value: float) -> str:
    return f"{int(value):,}"


def texttt(text: str) -> str:
    return f"\\texttt{{{text}}}"
