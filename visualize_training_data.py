"""
Training Data Visualization Utility
-----------------------------------
This script inspects the cached Waymo scenarios used for training and produces
rich visualizations that highlight what the dataset contains:

* Map geometry (lanes, road edges, crosswalks)
* All agent trajectories with past/future separation
* Ego agent highlighted and plotted after the ground-truth path
* Speed timelines for ego and surrounding agents
* Distribution plots / summary tables per-scenario

It also aggregates statistics across processed scenarios and saves both the
per-scenario figures and dataset-level summaries to disk.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import config
from waymo_tfexample_loader import WaymoDataset


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #

FPS = config.DATA_CONFIG["fps"]
PAST_STEPS = config.DATA_CONFIG["past_timesteps"]
CURRENT_STEPS = config.DATA_CONFIG["current_timesteps"]
FUTURE_STEPS = config.DATA_CONFIG["future_timesteps"]
TOTAL_STEPS = config.DATA_CONFIG["total_timesteps"]


# --------------------------------------------------------------------------- #
# Utility dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class ScenarioStats:
    scenario_id: str
    num_agents: int
    num_valid_agents: int
    duration_s: float
    ego_speed_mean: float
    ego_speed_max: float
    ego_acc_mean: float
    ego_acc_max: float
    avg_speed_all: float
    avg_acc_all: float
    map_feature_counts: Dict[str, int]


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

def get_dataset(max_scenarios: int) -> WaymoDataset:
    """Load the most relevant PKL dataset within the data root."""
    data_root = config.DATA_ROOT
    train_pkl = data_root / "training_processed.pkl"
    val_pkl = data_root / "validation_processed.pkl"

    if train_pkl.exists():
        source = train_pkl
    elif val_pkl.exists():
        source = val_pkl
    else:
        raise FileNotFoundError(
            f"No PKL data files found in {data_root}. Expected "
            f"{train_pkl.name} or {val_pkl.name}."
        )

    print(f"[INFO] Loading scenarios from {source}")
    dataset = WaymoDataset(pkl_file=str(source), max_scenarios=max_scenarios)
    print(f"[INFO] Loaded {len(dataset.scenarios)} scenarios.")
    return dataset


def as_array(points) -> Optional[np.ndarray]:
    if points is None:
        return None
    if isinstance(points, np.ndarray):
        return points
    try:
        arr = np.asarray(points)
        if arr.ndim >= 2:
            return arr
    except Exception:
        return None
    return None


def extract_map_feature_count(map_data: Optional[Dict]) -> Dict[str, int]:
    counts = {"lanes": 0, "road_edges": 0, "crosswalks": 0}
    if not isinstance(map_data, dict):
        return counts
    for key in counts.keys():
        value = map_data.get(key)
        if isinstance(value, Iterable):
            try:
                counts[key] = len(value)
            except Exception:
                counts[key] = 0
    return counts


def get_valid_agent_indices(positions: np.ndarray) -> List[int]:
    """Return indices of agents that have at least one non-zero position."""
    valid = []
    for idx in range(positions.shape[0]):
        if np.any(np.abs(positions[idx]) > 1e-6):
            valid.append(idx)
    return valid


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #

def plot_map(ax: plt.Axes, map_data: Optional[Dict]) -> None:
    """Plot map polylines if available."""
    if not isinstance(map_data, dict):
        ax.set_title("Map (not available)")
        return

    for key, style in [
        ("lanes", dict(color="#d0d0d0", linewidth=1.0, linestyle="--")),
        ("road_edges", dict(color="#aaaaaa", linewidth=1.5, linestyle="-")),
        ("crosswalks", dict(color="#f5b041", linewidth=1.2, linestyle=":")),
    ]:
        features = map_data.get(key, [])
        for feat in features:
            arr = as_array(feat)
            if arr is None or arr.ndim < 2 or arr.shape[1] < 2:
                continue
            ax.plot(arr[:, 0], arr[:, 1], **style, alpha=0.8)


def plot_trajectories(
    ax: plt.Axes,
    positions: np.ndarray,
    ego_idx: int,
    agent_indices: List[int],
) -> None:
    """Plot all agent trajectories with past/future segments distinguished."""
    cmap = plt.cm.get_cmap("tab20", len(agent_indices) + 1)
    time_history = np.arange(TOTAL_STEPS) / FPS

    for plot_idx, agent_idx in enumerate(agent_indices):
        pos = positions[agent_idx]
        if not np.any(np.abs(pos) > 1e-6):
            continue

        past = pos[:PAST_STEPS]
        future = pos[PAST_STEPS:]
        color = cmap(plot_idx % cmap.N)
        alpha = 0.9 if agent_idx == ego_idx else 0.45
        lw = 2.4 if agent_idx == ego_idx else 1.2
        label = "Ego" if agent_idx == ego_idx else f"A{agent_idx}"

        # Past trajectory (dashed)
        ax.plot(past[:, 0], past[:, 1], linestyle="--", linewidth=lw, color=color, alpha=alpha * 0.8)
        # Future trajectory
        ax.plot(future[:, 0], future[:, 1], linestyle="-", linewidth=lw, color=color, alpha=alpha)

        # Mark start, current, and final positions
        ax.scatter(past[0, 0], past[0, 1], s=25, color=color, alpha=alpha * 0.9)
        ax.scatter(past[-1, 0], past[-1, 1], s=35, marker="o", color=color, alpha=alpha * 0.9)
        ax.scatter(future[-1, 0], future[-1, 1], s=35, marker="x", color=color, alpha=alpha * 0.9)

        if agent_idx == ego_idx:
            ax.text(
                future[-1, 0],
                future[-1, 1],
                label,
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal")
    ax.set_title("Map + Agent Trajectories")

    # Fit view to data
    all_valid_positions = positions[agent_indices].reshape(-1, 2)
    non_zero_mask = np.any(np.abs(all_valid_positions) > 1e-6, axis=1)
    if np.any(non_zero_mask):
        valid_positions = all_valid_positions[non_zero_mask]
        min_xy = valid_positions.min(axis=0) - 10.0
        max_xy = valid_positions.max(axis=0) + 10.0
        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])

    # Indicate history vs future separation
    ax.axvline(0.0, color="#cccccc", linestyle=":", linewidth=1.0, alpha=0.5)


def plot_speed_profiles(
    ax: plt.Axes,
    speeds: np.ndarray,
    ego_idx: int,
    agent_indices: List[int],
) -> None:
    """Plot speed timelines for the ego and surrounding agents."""
    time = np.arange(TOTAL_STEPS) / FPS
    for agent_idx in agent_indices:
        speed = speeds[agent_idx]
        if not np.any(speed > 1e-6):
            continue
        color = "#e74c3c" if agent_idx == ego_idx else "#95a5a6"
        lw = 2.0 if agent_idx == ego_idx else 1.0
        alpha = 0.9 if agent_idx == ego_idx else 0.3
        label = "Ego speed" if agent_idx == ego_idx else None
        ax.plot(time, speed, color=color, linewidth=lw, alpha=alpha, label=label)

    history_end = PAST_STEPS / FPS
    ax.axvspan(0, history_end, color="#d6eaf8", alpha=0.25, label="History window")
    ax.axvline(history_end, color="#3498db", linestyle="--", linewidth=1.1, alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed Profiles")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")


def plot_distributions(
    ax: plt.Axes,
    speeds: np.ndarray,
    accelerations: np.ndarray,
    agent_indices: List[int],
    scenario_stats: ScenarioStats,
) -> None:
    """Visualize speed and acceleration distributions plus textual summary."""
    speed_samples = speeds[agent_indices].ravel()
    speed_samples = speed_samples[speed_samples > 1e-6]

    accel_samples = accelerations[agent_indices].ravel()
    accel_samples = accel_samples[np.abs(accel_samples) > 1e-6]

    ax.hist(speed_samples, bins=30, color="#3498db", alpha=0.65, label="Speed (m/s)")
    ax.hist(
        accel_samples,
        bins=30,
        color="#e67e22",
        alpha=0.45,
        label="Acceleration (m/s²)",
    )
    ax.set_title("Distribution Snapshot")
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.3)

    text_lines = [
        f"Scenario: {scenario_stats.scenario_id}",
        f"Agents (valid / total): {scenario_stats.num_valid_agents} / {scenario_stats.num_agents}",
        f"Duration: {scenario_stats.duration_s:.1f}s @ {FPS}Hz",
        f"Ego speed μ/σ/max: {scenario_stats.ego_speed_mean:.2f} / "
        f"{scenario_stats.ego_speed_max - scenario_stats.ego_speed_mean:.2f} / "
        f"{scenario_stats.ego_speed_max:.2f}",
        f"Ego accel μ/max: {scenario_stats.ego_acc_mean:.2f} / {scenario_stats.ego_acc_max:.2f}",
        f"Avg speed (all agents): {scenario_stats.avg_speed_all:.2f}",
        f"Avg accel (all agents): {scenario_stats.avg_acc_all:.2f}",
        "Map features:",
        f"  Lanes: {scenario_stats.map_feature_counts.get('lanes', 0)}",
        f"  Road edges: {scenario_stats.map_feature_counts.get('road_edges', 0)}",
        f"  Crosswalks: {scenario_stats.map_feature_counts.get('crosswalks', 0)}",
    ]

    ax.annotate(
        "\n".join(text_lines),
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#34495e", alpha=0.85),
    )


# --------------------------------------------------------------------------- #
# Per-scenario visualization
# --------------------------------------------------------------------------- #

def visualize_scenario(
    scenario: Dict,
    output_dir: Path,
) -> ScenarioStats:
    """Create visualization for one scenario and return computed statistics."""
    scenario_id = scenario.get("scenario_id", "unknown")
    trajectories = scenario["trajectories"]  # [agents, steps, 7]
    map_data = scenario.get("map_data", None)
    ego_id = scenario.get("ego_agent_id", 0)

    positions = trajectories[:, :, :2]
    velocities = trajectories[:, :, 3:5]
    accelerations = trajectories[:, :, 5:7]

    speeds = np.linalg.norm(velocities, axis=-1)
    accel_mags = np.linalg.norm(accelerations, axis=-1)

    agent_indices = get_valid_agent_indices(positions)
    if ego_id not in agent_indices and agent_indices:
        ego_id = agent_indices[0]

    map_feature_counts = extract_map_feature_count(map_data)
    duration_s = TOTAL_STEPS / FPS

    # Ego stats
    ego_speed = speeds[ego_id]
    ego_acc = accel_mags[ego_id]
    ego_speed_valid = ego_speed[ego_speed > 1e-6]
    ego_acc_valid = ego_acc[np.abs(ego_acc) > 1e-6]

    # All agent stats
    all_speed_valid = speeds[agent_indices][speeds[agent_indices] > 1e-6]
    all_acc_valid = accel_mags[agent_indices][np.abs(accel_mags[agent_indices]) > 1e-6]

    scenario_stats = ScenarioStats(
        scenario_id=scenario_id,
        num_agents=trajectories.shape[0],
        num_valid_agents=len(agent_indices),
        duration_s=duration_s,
        ego_speed_mean=float(np.mean(ego_speed_valid)) if ego_speed_valid.size else 0.0,
        ego_speed_max=float(np.max(ego_speed_valid)) if ego_speed_valid.size else 0.0,
        ego_acc_mean=float(np.mean(ego_acc_valid)) if ego_acc_valid.size else 0.0,
        ego_acc_max=float(np.max(np.abs(ego_acc_valid))) if ego_acc_valid.size else 0.0,
        avg_speed_all=float(np.mean(all_speed_valid)) if all_speed_valid.size else 0.0,
        avg_acc_all=float(np.mean(np.abs(all_acc_valid))) if all_acc_valid.size else 0.0,
        map_feature_counts=map_feature_counts,
    )

    # Build figure
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.5))
    ax_map, ax_speed, ax_hist = axes

    plot_map(ax_map, map_data)
    plot_trajectories(ax_map, positions, ego_id, agent_indices)
    plot_speed_profiles(ax_speed, speeds, ego_id, agent_indices)
    plot_distributions(ax_hist, speeds, accel_mags, agent_indices, scenario_stats)

    fig.suptitle(f"Training Scenario: {scenario_id}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / f"{scenario_id}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[INFO] Saved scenario visualization to {out_path}")

    return scenario_stats


# --------------------------------------------------------------------------- #
# Aggregated statistics & visualization
# --------------------------------------------------------------------------- #

def save_aggregate_summary(
    scenario_stats: List[ScenarioStats],
    aggregate_dir: Path,
    all_speeds: np.ndarray,
    all_accels: np.ndarray,
) -> None:
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    # Summary histogram plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_speed_hist, ax_acc_hist, ax_bar = axes

    ax_speed_hist.hist(all_speeds, bins=60, color="#3498db", alpha=0.7)
    ax_speed_hist.set_title("Global Speed Distribution")
    ax_speed_hist.set_xlabel("Speed (m/s)")
    ax_speed_hist.set_ylabel("Count")
    ax_speed_hist.grid(True, linestyle=":", alpha=0.3)

    ax_acc_hist.hist(all_accels, bins=60, color="#e67e22", alpha=0.7)
    ax_acc_hist.set_title("Global Acceleration Distribution")
    ax_acc_hist.set_xlabel("Acceleration (m/s²)")
    ax_acc_hist.set_ylabel("Count")
    ax_acc_hist.grid(True, linestyle=":", alpha=0.3)

    valid_agents = [stats.num_valid_agents for stats in scenario_stats]
    ax_bar.bar(range(len(valid_agents)), valid_agents, color="#2ecc71", alpha=0.7)
    ax_bar.set_title("Valid Agents per Scenario")
    ax_bar.set_xlabel("Scenario Index")
    ax_bar.set_ylabel("Valid Agents (#)")
    ax_bar.grid(True, linestyle=":", alpha=0.3)

    fig.suptitle("Dataset Aggregate Statistics", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(aggregate_dir / "aggregate_statistics.png", dpi=200)
    plt.close(fig)

    # Save textual summary
    summary_lines = [
        f"Total scenarios processed: {len(scenario_stats)}",
        f"Total valid agents observed: {sum(stat.num_valid_agents for stat in scenario_stats)}",
        f"Global speed mean/std/max: "
        f"{np.mean(all_speeds):.2f} / {np.std(all_speeds):.2f} / {np.max(all_speeds):.2f}",
        f"Global acceleration mean/std/max: "
        f"{np.mean(all_accels):.2f} / {np.std(all_accels):.2f} / {np.max(all_accels):.2f}",
    ]
    summary_lines.append("Per-scenario highlights:")
    for stats in scenario_stats:
        summary_lines.append(
            f"  • {stats.scenario_id}: valid agents={stats.num_valid_agents}, "
            f"ego speed μ={stats.ego_speed_mean:.2f} m/s, "
            f"ego max speed={stats.ego_speed_max:.2f} m/s"
        )

    summary_path = aggregate_dir / "aggregate_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"[INFO] Aggregate summary saved to {summary_path}")


# --------------------------------------------------------------------------- #
# Main execution
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training/validation Waymo scenarios.")
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=5,
        help="Number of scenarios to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training_data_viz"),
        help="Directory to store generated figures.",
    )
    args = parser.parse_args()

    dataset = get_dataset(args.num_scenarios)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_stats: List[ScenarioStats] = []
    global_speeds: List[np.ndarray] = []
    global_accels: List[np.ndarray] = []

    for scenario in dataset.scenarios[: args.num_scenarios]:
        stats = visualize_scenario(scenario, output_dir)
        scenario_stats.append(stats)

        trajectories = scenario["trajectories"]
        speeds = np.linalg.norm(trajectories[:, :, 3:5], axis=-1)
        accelerations = np.linalg.norm(trajectories[:, :, 5:7], axis=-1)
        valid_agents = get_valid_agent_indices(trajectories[:, :, :2])

        if valid_agents:
            global_speeds.append(speeds[valid_agents].ravel())
            global_accels.append(accelerations[valid_agents].ravel())

    if scenario_stats:
        all_speed_values = np.concatenate(global_speeds)
        all_acc_values = np.concatenate(global_accels)
        save_aggregate_summary(scenario_stats, output_dir, all_speed_values, all_acc_values)
    else:
        print("[WARN] No scenarios were processed; aggregate summary skipped.")


if __name__ == "__main__":
    main()


