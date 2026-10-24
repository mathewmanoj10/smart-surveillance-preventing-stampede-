import math

import torch


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    return value if math.isfinite(value) else default


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def risk_level(risk_percent, low=35.0, high=65.0):
    """Same thresholds as the client (risk on a 0-100 scale)."""
    risk_percent = max(0.0, min(100.0, safe_float(risk_percent)))

    if risk_percent >= high:
        return "HIGH"

    if risk_percent >= low:
        return "MEDIUM"

    return "LOW"


# ============================================================
# 1. FEDERATED AVERAGING OF MODEL WEIGHTS (FedAvg)
# ============================================================

def federated_average(local_states, sample_counts=None):
    """
    Weighted average of model state_dicts.

    local_states : list of state_dicts (tensors)
    sample_counts: list of ints (e.g. frames processed per client)
    Returns a state_dict that can be passed to load_state_dict().
    """

    if not local_states:
        raise ValueError("No local model states were provided.")

    if sample_counts is None:
        sample_counts = [1] * len(local_states)

    if len(sample_counts) != len(local_states):
        raise ValueError("sample_counts and local_states must match.")

    counts = [max(1, safe_int(c, 1)) for c in sample_counts]
    total = float(sum(counts))

    print("\n[SERVER] FedAvg started.")
    print(f"[SERVER] Clients: {len(local_states)} | total weight: {int(total)}")

    averaged = {}

    for key in local_states[0]:
        acc = None

        for state, count in zip(local_states, counts):
            part = state[key].float() * (count / total)
            acc = part if acc is None else acc + part

        averaged[key] = acc

    print("[SERVER] FedAvg completed.")
    return averaged


# ============================================================
# 2. AGGREGATION OF CAMERA RISK STATISTICS
# ============================================================

def aggregate_risk(local_updates):
    """
    Combine per-camera risk summaries (from LocalClient.get_risk_update()).

    A stampede is local, so the alert uses the WORST camera (max_risk),
    not the average. The weighted averages are for reporting.
    """

    if not local_updates:
        raise ValueError("No local risk updates were provided.")

    weights = [max(1, safe_int(u.get("frames", 1), 1)) for u in local_updates]
    total = float(sum(weights))

    keys = [
        "avg_people", "avg_density", "avg_speed",
        "avg_direction_variance", "avg_risk",
    ]

    global_update = {
        key: sum(
            safe_float(u.get(key)) * w / total
            for u, w in zip(local_updates, weights)
        )
        for key in keys
    }

    worst = max(local_updates, key=lambda u: safe_float(u.get("max_risk")))

    global_update["max_risk"] = safe_float(worst.get("max_risk"))
    global_update["worst_camera"] = worst.get("client_id", "unknown")
    global_update["frames"] = sum(weights)
    global_update["num_clients"] = len(local_updates)

    return global_update


def print_risk_report(local_updates, global_update):
    print("\n" + "=" * 70)
    print("CAMERA RISK REPORT")
    print("=" * 70)

    for i, u in enumerate(local_updates):
        name = u.get("client_id", f"Camera {i + 1}")
        print(
            f"{name:<14} people={safe_float(u.get('avg_people')):6.1f} | "
            f"density={safe_float(u.get('avg_density')):5.2f}/m2 | "
            f"speed={safe_float(u.get('avg_speed')):4.2f} m/s | "
            f"dir.var={safe_float(u.get('avg_direction_variance')):4.2f} | "
            f"avg risk={safe_float(u.get('avg_risk')):5.1f} | "
            f"max risk={safe_float(u.get('max_risk')):5.1f}"
        )

    avg_risk = safe_float(global_update.get("avg_risk"))
    max_risk = safe_float(global_update.get("max_risk"))
    level = risk_level(max_risk)

    print("-" * 70)
    print(f"Global average risk : {avg_risk:.1f}/100")
    print(
        f"Worst camera        : {global_update.get('worst_camera')} "
        f"({max_risk:.1f}/100)"
    )
    print(f"ALERT LEVEL         : {level}")

    if level == "HIGH":
        print("ALERT: HIGH CROWD RISK DETECTED.")
    elif level == "MEDIUM":
        print("WARNING: MODERATE CROWD RISK DETECTED.")
    else:
        print("STATUS: LOW ESTIMATED CROWD RISK.")

    print("=" * 70)
    return level