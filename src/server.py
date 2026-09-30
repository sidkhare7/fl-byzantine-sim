"""Flower ServerApp: FedAvg aggregation and centralised evaluation."""

import csv
import os

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.data import load_test_data
from src.model import SmallCNN, evaluate, get_weights, set_weights

CSV_FIELDS = [
    "attack",
    "byzantine_fraction",
    "num_clients",
    "partition",
    "alpha",
    "seed",
    "round",
    "loss",
    "accuracy",
]


def init_csv(path):
    """Create the results file and write its header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()


def append_row(path, row):
    """Append one round of results, flushing so partial runs are still usable."""
    with open(path, "a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS).writerow(row)


def build_server_app(cfg):
    """Return a ServerApp configured for one experiment.

    The server evaluates the aggregated model on the full CIFAR-10 test set
    after every round. Evaluation is centralised rather than federated, so the
    reported accuracy is never influenced by the Byzantine clients themselves.
    """
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        torch.manual_seed(cfg["seed"])
        torch.set_num_threads(cfg["server_threads"])

        model = SmallCNN()
        test_loader = load_test_data(cfg["data_root"])
        init_csv(cfg["out"])

        def evaluate_global(server_round, arrays):
            """Score the aggregated model and record one CSV row."""
            set_weights(model, arrays.to_numpy_ndarrays())
            loss, accuracy = evaluate(model, test_loader)

            append_row(
                cfg["out"],
                {
                    "attack": cfg["attack"],
                    "byzantine_fraction": cfg["byzantine_fraction"],
                    "num_clients": cfg["num_clients"],
                    "partition": cfg["partition"],
                    "alpha": cfg["alpha"],
                    "seed": cfg["seed"],
                    "round": server_round,
                    "loss": round(loss, 6),
                    "accuracy": round(accuracy, 6),
                },
            )
            print(
                f"round {server_round:>3}  loss {loss:.4f}  accuracy {accuracy:.4f}",
                flush=True,
            )
            return MetricRecord({"loss": loss, "accuracy": accuracy})

        # Config forwarded to every client. Flower adds "server-round" itself.
        train_config = ConfigRecord(
            {
                "num-clients": cfg["num_clients"],
                "attack": cfg["attack"],
                "byzantine-fraction": cfg["byzantine_fraction"],
                "partition": cfg["partition"],
                "alpha": cfg["alpha"],
                "seed": cfg["seed"],
                "data-root": cfg["data_root"],
                "batch-size": cfg["batch_size"],
                "local-epochs": cfg["local_epochs"],
                "lr": cfg["lr"],
                "gaussian-sigma": cfg["gaussian_sigma"],
                "sign-flip-scale": cfg["sign_flip_scale"],
            }
        )

        # Plain FedAvg: the server applies no robust aggregation, so the effect
        # of the attackers is visible in the accuracy curve.
        strategy = FedAvg(
            fraction_train=1.0,
            fraction_evaluate=0.0,
            min_train_nodes=cfg["num_clients"],
            min_evaluate_nodes=0,
            min_available_nodes=cfg["num_clients"],
        )

        strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(get_weights(model)),
            num_rounds=cfg["rounds"],
            train_config=train_config,
            evaluate_fn=evaluate_global,
        )

    return app
