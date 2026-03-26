"""
Anti-Gravity Financial Agentic System
Visualization — Spider / Radar Chart with 3-region background coloring.
"""

import io
import math
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap


# ── Colors ─────────────────────────────────────────────────────────
RED = "#FF4B4B"
YELLOW = "#FFEB3B"
GREEN = "#4CAF50"

ZONE_COLORS = {
    "red":    (RED,    0.15),   # 0-4
    "yellow": (YELLOW, 0.12),   # 4-8
    "green":  (GREEN,  0.10),   # 8-10
}

FILL_COLOR = "#00BCD4"
LINE_COLOR = "#00E5FF"
BG_COLOR = "#0D1117"
GRID_COLOR = "#21262D"
LABEL_COLOR = "#C9D1D9"


def create_spider_chart(
    scores: dict,
    title: str = "Financial Health Radar",
    save_path: str | None = None,
) -> str:
    """
    Create a radar/spider chart with 3-region background coloring.
    
    Args:
        scores: Dict with keys matching the 6 financial dimensions.
        title: Chart title.
        save_path: If provided, save to this path. Otherwise, auto-generate.
    
    Returns:
        Absolute path to the saved PNG file.
    """
    categories = list(scores.keys())
    values = list(scores.values())
    N = len(categories)

    # Compute angles for each axis
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # close the loop
    values += values[:1]

    # ── Figure setup ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # ── Draw colored background zones ──────────────────────────────
    theta_fill = np.linspace(0, 2 * np.pi, 100)

    # Green zone: 8-10
    ax.fill_between(
        theta_fill, 8, 10,
        color=GREEN, alpha=ZONE_COLORS["green"][1], zorder=0,
    )
    # Yellow zone: 4-8
    ax.fill_between(
        theta_fill, 4, 8,
        color=YELLOW, alpha=ZONE_COLORS["yellow"][1], zorder=0,
    )
    # Red zone: 0-4
    ax.fill_between(
        theta_fill, 0, 4,
        color=RED, alpha=ZONE_COLORS["red"][1], zorder=0,
    )

    # ── Zone labels ────────────────────────────────────────────
    ax.text(
        math.pi / 4, 2, "DANGER", fontsize=8, fontweight="bold",
        color=RED, alpha=0.7, ha="center", va="center",
    )
    ax.text(
        math.pi / 4, 6, "CAUTION", fontsize=8, fontweight="bold",
        color=YELLOW, alpha=0.7, ha="center", va="center",
    )
    ax.text(
        math.pi / 4, 9, "STRONG", fontsize=8, fontweight="bold",
        color=GREEN, alpha=0.7, ha="center", va="center",
    )

    # ── Grid styling ───────────────────────────────────────────────
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color=GRID_COLOR, fontsize=8)
    ax.set_xticks(angles[:-1])

    # Capitalize labels nicely
    pretty_labels = [c.replace("_", " ").title() for c in categories]
    ax.set_xticklabels(pretty_labels, color=LABEL_COLOR, fontsize=11, fontweight="bold")

    # Grid lines
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.spines["polar"].set_linewidth(0.5)

    # ── Plot data ──────────────────────────────────────────────────
    ax.plot(angles, values, color=LINE_COLOR, linewidth=2.5, linestyle="-", zorder=3)
    ax.fill(angles, values, color=FILL_COLOR, alpha=0.25, zorder=2)

    # Score dots
    for angle, value in zip(angles[:-1], values[:-1]):
        color = RED if value < 4 else (YELLOW if value < 8 else GREEN)
        ax.plot(angle, value, "o", color=color, markersize=10, zorder=4,
                markeredgecolor="white", markeredgewidth=1.5)
        # Score label next to dot
        ax.text(
            angle, value + 0.6, f"{value:.1f}", fontsize=9, fontweight="bold",
            color="white", ha="center", va="center", zorder=5,
        )

    # ── Title ──────────────────────────────────────────────────────
    ax.set_title(
        title, fontsize=16, fontweight="bold", color="white",
        pad=25, y=1.05,
    )

    # ── Overall score badge ────────────────────────────────────────
    avg_score = sum(values[:-1]) / N
    badge_color = RED if avg_score < 4 else (YELLOW if avg_score < 8 else GREEN)
    fig.text(
        0.5, 0.02,
        f"Overall Score: {avg_score:.1f} / 10",
        fontsize=14, fontweight="bold", color=badge_color,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_COLOR, edgecolor=badge_color, linewidth=2),
    )

    plt.tight_layout()

    # ── Save ───────────────────────────────────────────────────────
    if save_path is None:
        out_dir = pathlib.Path(__file__).parent / "output"
        out_dir.mkdir(exist_ok=True)
        save_path = str(out_dir / "spider_chart.png")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    return save_path


def create_comparison_chart(
    current_scores: dict,
    projected_scores: dict,
    save_path: str | None = None,
) -> str:
    """
    Side-by-side radar chart: current vs projected scores.
    """
    categories = list(current_scores.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    current_vals = list(current_scores.values()) + [list(current_scores.values())[0]]
    projected_vals = list(projected_scores.values()) + [list(projected_scores.values())[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Background zones
    theta_fill = np.linspace(0, 2 * np.pi, 100)
    ax.fill_between(theta_fill, 8, 10, color=GREEN, alpha=0.10, zorder=0)
    ax.fill_between(theta_fill, 4, 8,  color=YELLOW, alpha=0.12, zorder=0)
    ax.fill_between(theta_fill, 0, 4,  color=RED, alpha=0.15, zorder=0)

    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color=GRID_COLOR, fontsize=8)
    ax.set_xticks(angles[:-1])
    pretty_labels = [c.replace("_", " ").title() for c in categories]
    ax.set_xticklabels(pretty_labels, color=LABEL_COLOR, fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    ax.spines["polar"].set_color(GRID_COLOR)

    # Current (dimmed)
    ax.plot(angles, current_vals, color="#FF6B6B", linewidth=1.5, linestyle="--", alpha=0.7, label="Current")
    ax.fill(angles, current_vals, color="#FF6B6B", alpha=0.08)

    # Projected (bright)
    ax.plot(angles, projected_vals, color=LINE_COLOR, linewidth=2.5, label="After Plan")
    ax.fill(angles, projected_vals, color=FILL_COLOR, alpha=0.25)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11,
              facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor="white")

    ax.set_title(
        "Current vs. Projected Scores", fontsize=16, fontweight="bold",
        color="white", pad=25, y=1.05,
    )

    plt.tight_layout()

    if save_path is None:
        out_dir = pathlib.Path(__file__).parent / "output"
        out_dir.mkdir(exist_ok=True)
        save_path = str(out_dir / "comparison_chart.png")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    return save_path


# ── CLI test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_scores = {
        "emergency": 3.5,
        "debt": 6.2,
        "insurance": 8.0,
        "retirement": 4.8,
        "tax": 5.5,
        "investment": 7.1,
    }
    path = create_spider_chart(sample_scores)
    print(f"✅ Spider chart saved to {path}")
