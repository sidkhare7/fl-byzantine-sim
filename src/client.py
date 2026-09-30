"""Flower ClientApp: honest and Byzantine client behaviour."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.attacks import apply_update_attack, byzantine_ids
from src.data import load_client_data
from src.model import SmallCNN, get_weights, set_weights, train_local

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Run one round of local training, corrupting the result if Byzantine.

    Every client receives the same config, so the decision to attack is made
    locally from the client's own partition id. This mirrors the real setting,
    where the server cannot tell which clients are adversarial.
    """
    # Each Ray worker gets its own process, so limit threads to avoid
    # oversubscribing the CPU when several clients train concurrently.
    torch.set_num_threads(1)

    config = msg.content["config"]
    partition_id = context.node_config["partition-id"]
    num_clients = int(config["num-clients"])

    attack = str(config["attack"])
    attackers = byzantine_ids(num_clients, float(config["byzantine-fraction"]))
    is_byzantine = partition_id in attackers

    # A label flipping client trains on corrupted labels but is otherwise
    # ordinary, so the flip is applied when building its DataLoader.
    flip = is_byzantine and attack == "label_flip"

    loader = load_client_data(
        partition_id=partition_id,
        num_clients=num_clients,
        partition=str(config["partition"]),
        alpha=float(config["alpha"]),
        seed=int(config["seed"]),
        root=str(config["data-root"]),
        batch_size=int(config["batch-size"]),
        flip=flip,
    )

    model = SmallCNN()
    set_weights(model, msg.content["arrays"].to_numpy_ndarrays())

    loss = train_local(
        model,
        loader,
        epochs=int(config["local-epochs"]),
        lr=float(config["lr"]),
    )

    update = get_weights(model)
    if is_byzantine:
        # Vary the noise per client and per round so attackers do not all
        # send the same corrupted update.
        attack_seed = int(config["seed"]) * 100003 + partition_id * 997 + int(config["server-round"])
        update = apply_update_attack(
            attack,
            update,
            sigma=float(config["gaussian-sigma"]),
            scale=float(config["sign-flip-scale"]),
            seed=attack_seed,
        )

    metrics = MetricRecord(
        {
            "num-examples": len(loader.dataset),
            "train-loss": loss,
            "byzantine": int(is_byzantine),
        }
    )
    content = RecordDict({"arrays": ArrayRecord(update), "metrics": metrics})
    return Message(content=content, reply_to=msg)
