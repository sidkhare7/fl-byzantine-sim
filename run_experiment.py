"""Run one federated learning experiment with a given Byzantine configuration.

Example:
    python run_experiment.py --attack sign_flip --byzantine-fraction 0.2 --rounds 20
"""

import argparse
import os
import random

import numpy as np
import torch
from flwr.simulation import run_simulation

from src.attacks import ATTACKS, byzantine_ids
from src.client import app as client_app
from src.server import build_server_app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure how Byzantine clients degrade FedAvg on CIFAR-10."
    )
    parser.add_argument("--attack", choices=ATTACKS, default="none",
                        help="Byzantine behaviour. 'none' gives the honest baseline.")
    parser.add_argument("--byzantine-fraction", type=float, default=0.0,
                        help="Fraction of clients that are Byzantine, for example 0.2.")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of federated rounds.")
    parser.add_argument("--num-clients", type=int, default=10,
                        help="Total number of clients in the federation.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for partitioning, initialisation and attacks.")
    parser.add_argument("--partition", choices=["iid", "dirichlet"], default="iid",
                        help="How CIFAR-10 is split across clients.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration. Lower means more label skew.")
    parser.add_argument("--local-epochs", type=int, default=1,
                        help="Local epochs per client per round.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Client SGD learning rate.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Client batch size.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0,
                        help="Standard deviation for the Gaussian noise attack.")
    parser.add_argument("--sign-flip-scale", type=float, default=3.0,
                        help="Multiplier applied to the negated update in sign flip.")
    parser.add_argument("--data-root", default="data",
                        help="Directory for the CIFAR-10 download.")
    parser.add_argument("--results-dir", default="results",
                        help="Directory for the per-round CSV output.")
    parser.add_argument("--out", default=None,
                        help="Explicit CSV path. Defaults to a name built from the config.")
    parser.add_argument("--num-cpus", type=int, default=2,
                        help="CPUs reserved per concurrent client in the Ray backend.")
    return parser.parse_args()


def default_out_path(args):
    """Build a result filename that identifies the configuration."""
    fraction = f"{args.byzantine_fraction:.2f}".replace(".", "p")
    parts = [args.attack, f"byz{fraction}", args.partition, f"seed{args.seed}"]
    return os.path.join(args.results_dir, "_".join(parts) + ".csv")


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = args.out or default_out_path(args)
    attackers = sorted(byzantine_ids(args.num_clients, args.byzantine_fraction))

    # An attack with no attackers is the honest baseline, and vice versa. Warn
    # rather than fail, since 0.0 is a legitimate point on the sweep.
    if args.attack != "none" and not attackers:
        print(f"warning: attack '{args.attack}' selected but byzantine-fraction "
              f"{args.byzantine_fraction} yields no attackers")
    if args.attack == "none" and attackers:
        print(f"warning: byzantine-fraction {args.byzantine_fraction} selected but "
              f"attack is 'none', so all clients behave honestly")

    print(f"attack={args.attack} fraction={args.byzantine_fraction} "
          f"clients={args.num_clients} byzantine={attackers}")
    print(f"partition={args.partition} rounds={args.rounds} seed={args.seed}")
    print(f"writing {out}")

    cfg = {
        "attack": args.attack,
        "byzantine_fraction": args.byzantine_fraction,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "seed": args.seed,
        "partition": args.partition,
        "alpha": args.alpha,
        "local_epochs": args.local_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "gaussian_sigma": args.gaussian_sigma,
        "sign_flip_scale": args.sign_flip_scale,
        "data_root": args.data_root,
        "out": out,
        "server_threads": max(1, os.cpu_count() // 2),
    }

    run_simulation(
        server_app=build_server_app(cfg),
        client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config={"client_resources": {"num_cpus": args.num_cpus, "num_gpus": 0.0}},
    )

    print(f"done, results in {out}")


if __name__ == "__main__":
    main()
