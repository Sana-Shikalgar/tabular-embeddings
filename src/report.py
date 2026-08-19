"""Output-artifact helpers shared across notebooks: append timestamped EDA
findings to a single running markdown log (reports/eda_log.md), and save
figures at a consistent resolution.

Each notebook keeps its figures in its own reports/figures/<NN>/
subdirectory (e.g. reports/figures/00/ for 00_tmdb_eda.ipynb), not a
shared directory -- savefig's figures_dir has no default for this reason:
every caller must say which notebook's folder a figure belongs in. The
usual pattern is to bind that once per notebook, right after import:

    from functools import partial
    from src.config import FIGURES_DIR
    FIGURES_DIR = FIGURES_DIR / "00"
    savefig = partial(savefig, figures_dir=FIGURES_DIR)

so every subsequent savefig(fig, filename) call in that notebook lands in
the right subdirectory without repeating figures_dir at each call site.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.config import REPORTS_DIR

EDA_LOG_PATH = REPORTS_DIR / "eda_log.md"


def savefig(fig, filename, figures_dir: Path) -> None:
    """Saves a matplotlib figure to figures_dir at 300dpi, creating the
    directory if needed. figures_dir is required, not defaulted -- see
    this module's own docstring for the per-notebook subdirectory
    convention (reports/figures/<NN>/) and the partial-binding pattern
    that keeps individual call sites from having to repeat it."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")


def log_findings(
    title: str,
    findings: dict,
    figures: list[str] | None = None,
    log_path: Path = EDA_LOG_PATH,
) -> None:
    """Append a timestamped section to the EDA log.

    Args:
        title: Section heading, e.g. "TMDB missingness overview".
        findings: Key findings as a dict; rendered as bullet points.
        figures: Optional paths (e.g. under reports/figures/<NN>/) to reference.
        log_path: Log file to append to. Defaults to reports/eda_log.md.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"## {title} ({timestamp})", ""]

    for key, value in findings.items():
        lines.append(f"- **{key}**: {value}")

    if figures:
        lines.append("")
        lines.append("Figures:")
        for fig in figures:
            lines.append(f"- `{fig}`")

    lines.append("")
    lines.append("---")
    lines.append("")

    is_new = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("# EDA Log\n\n")
        f.write("\n".join(lines) + "\n")
