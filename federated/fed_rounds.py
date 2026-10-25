"""
federated/fed_rounds.py

Multi-round federated learning (simulated on one computer).

Each ROUND:
    1. SERVER  sends the current global model to every client
    2. CLIENT  trains it locally on its OWN private features/labels
    3. CLIENT  sends back only the model weights (never video or features)
    4. SERVER  averages the weights (FedAvg, weighted by frames processed)
    5. The new global model is tested on each client's private data

Optional: each client's update can be clipped and noised before it is sent
(a DP-style protection). This is NOT a formal privacy guarantee unless you
also do privacy accounting.
"""

import contextlib
import csv
import io
import os

import torch

from federated.fed_avg import federated_average
from models.simple_model import SimpleDensityModel


def privatize_update(local_state, global_state, clip_norm, noise_multiplier):
    """
    Clips the client's update (local - global) to a maximum L2 norm and adds
    Gaussian noise scaled by clip_norm, then returns the protected weights.
    """
    delta = {
        k: local_state[k].float() - global_state[k].float()
        for k in local_state
    }

    norm = float(torch.sqrt(sum((d ** 2).sum() for d in delta.values())))
    scale = min(1.0, clip_norm / (norm + 1e-12))

    protected = {}

    for k, d in delta.items():
        d = d * scale

        if noise_multiplier > 0:
            d = d + torch.randn_like(d) * noise_multiplier * clip_norm

        protected[k] = global_state[k].float() + d

    return protected


def run_federated_rounds(
    clients,
    labels,
    rounds=10,
    local_epochs=5,
    clip_norm=None,
    noise_multiplier=0.0,
    log_path=os.path.join("outputs", "federated_rounds.csv"),
):
    """
    Returns (global_state, init_state, history).

    clients : list of LocalClient (already processed their videos)
    labels  : list of training labels, one per client (None = formula risk)
    """
    if not clients:
        raise ValueError("No clients provided.")

    if len(clients) < 2:
        print("[NOTE] Only 1 client: this is just local training. "
              "Add more cameras to see federation help.")

    init_state = {
        k: v.clone()
        for k, v in SimpleDensityModel(input_size=3).state_dict().items()
    }
    global_state = {k: v.clone() for k, v in init_state.items()}

    history = []

    print("\n" + "=" * 70)
    print(f"FEDERATED TRAINING: {len(clients)} clients, {rounds} rounds, "
          f"{local_epochs} local epochs each")
    if clip_norm:
        print(f"Update clipping: {clip_norm} | noise multiplier: "
              f"{noise_multiplier}")
    print("=" * 70)

    for r in range(1, rounds + 1):

        states, counts, local_losses = [], [], []

        for client, label in zip(clients, labels):

            # 1) server -> client: current global model
            client.set_model_weights(global_state)

            # 2) client trains on its own private data
            loss = client.simulate_local_training(
                epochs=local_epochs, target_label=label, verbose=False
            )

            if loss is None:
                continue

            # 3) client -> server: weights only
            state = client.get_model_weights()

            if clip_norm:
                state = privatize_update(
                    state, global_state, clip_norm, noise_multiplier
                )

            states.append(state)
            counts.append(client.frames_processed)
            local_losses.append(loss)

        if not states:
            print("No client produced an update; stopping.")
            break

        # 4) server aggregates (silence the per-call prints)
        with contextlib.redirect_stdout(io.StringIO()):
            global_state = federated_average(states, counts)

        # 5) test the new global model on each client's private data
        evals = {}

        for client, label in zip(clients, labels):
            e = client.evaluate_state(global_state, label)

            if e is not None:
                evals[client.client_id] = e

        mean_eval = sum(evals.values()) / len(evals) if evals else float("nan")
        mean_local = sum(local_losses) / len(local_losses)

        print(
            f"Round {r:02d}/{rounds} | local loss {mean_local:.4f} | "
            f"global MSE on client data {mean_eval:.4f}"
        )

        for name, e in evals.items():
            history.append({
                "round": r,
                "client": name,
                "global_mse_on_client": round(e, 6),
                "mean_local_loss": round(mean_local, 6),
            })

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "client", "global_mse_on_client",
                        "mean_local_loss"],
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"\nRound-by-round results saved: {log_path}")

    return global_state, init_state, history


def compare_local_vs_global(clients, labels, init_state, global_state, epochs):
    """
    Shows WHY federation helps: trains each client ALONE (same start, same
    total epochs), then tests every model on every client's private data.
    Lower MSE is better.
    """
    models = {}

    for client, label in zip(clients, labels):
        client.set_model_weights(init_state)
        client.simulate_local_training(
            epochs=epochs, target_label=label, verbose=False
        )
        models[f"{client.client_id} only"] = client.get_model_weights()

    models["GLOBAL (federated)"] = global_state

    names = [c.client_id for c in clients]
    width = max(len(n) for n in models) + 2

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (MSE on each client's own data, lower = better)")
    print("=" * 70)
    print(f"{'Model':<{width}}" + "".join(f"{n:>14}" for n in names)
          + f"{'MEAN':>12}")

    results = {}

    for model_name, state in models.items():
        scores = []

        for client, label in zip(clients, labels):
            e = client.evaluate_state(state, label)
            scores.append(float("nan") if e is None else e)

        valid = [s for s in scores if s == s]
        mean = sum(valid) / len(valid) if valid else float("nan")
        results[model_name] = mean

        print(f"{model_name:<{width}}"
              + "".join(f"{s:>14.4f}" for s in scores)
              + f"{mean:>12.4f}")

    print("=" * 70)

    return results