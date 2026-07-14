"""Append timestamped EDA findings to a single running markdown log
(reports/eda_log.md) so results accumulate in one place across notebooks.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Anchored to this file's location (not the process cwd), so this resolves
# correctly regardless of where a notebook kernel or script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]
EDA_LOG_PATH = REPO_ROOT / "reports" / "eda_log.md"


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
        figures: Optional paths (e.g. under reports/figures/) to reference.
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
