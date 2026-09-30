"""CIFAR-10 loading and partitioning across federated clients."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

# CIFAR-10 channel statistics, used to normalise batches during training.
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

NUM_CLASSES = 10

# Each Ray worker runs in its own process and would otherwise re-read CIFAR-10
# from disk on every round. Cache the raw arrays per process instead.
_CACHE = {}


def _load_raw(root, train):
    """Return (images uint8 NCHW, labels int64) for one CIFAR-10 split."""
    key = ("train" if train else "test", root)
    if key not in _CACHE:
        dataset = CIFAR10(root=root, train=train, download=True)
        # dataset.data is a uint8 array in NHWC order.
        images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous()
        labels = torch.tensor(dataset.targets, dtype=torch.int64)
        _CACHE[key] = (images, labels)
    return _CACHE[key]


def normalise(images_uint8):
    """Scale a uint8 image batch to normalised float32.

    Images are kept as uint8 in memory and converted per batch. This keeps the
    cached dataset around 150 MB instead of 600 MB per worker process.
    """
    x = images_uint8.float().div_(255.0)
    return (x - CIFAR10_MEAN) / CIFAR10_STD


def partition_iid(num_samples, num_clients, seed):
    """Split indices uniformly at random into equal shards."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    return [np.sort(shard) for shard in np.array_split(indices, num_clients)]


def partition_dirichlet(labels, num_clients, alpha, seed):
    """Split indices by drawing per-class proportions from Dir(alpha).

    Smaller alpha gives more label skew across clients. This is the standard
    non-IID construction used in the federated learning literature.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    shards = [[] for _ in range(num_clients)]

    for class_id in range(NUM_CLASSES):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Cut points convert proportions into contiguous slices of this class.
        cuts = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        for client_id, part in enumerate(np.split(class_indices, cuts)):
            shards[client_id].extend(part.tolist())

    out = []
    for shard in shards:
        shard = np.array(sorted(shard), dtype=np.int64)
        rng.shuffle(shard)
        out.append(shard)
    return out


def get_partitions(num_clients, partition, alpha, seed, root):
    """Return the index shard assigned to every client."""
    _, labels = _load_raw(root, train=True)
    if partition == "iid":
        return partition_iid(len(labels), num_clients, seed)
    if partition == "dirichlet":
        return partition_dirichlet(labels.numpy(), num_clients, alpha, seed)
    raise ValueError(f"unknown partition scheme: {partition}")


def flip_labels(labels):
    """Map every label y to 9 - y.

    A deterministic permutation with no fixed points, so a client training on
    flipped labels learns a consistently wrong mapping rather than noise.
    """
    return (NUM_CLASSES - 1) - labels


def load_client_data(partition_id, num_clients, partition, alpha, seed, root,
                     batch_size, flip=False):
    """Build the training DataLoader for one client."""
    images, labels = _load_raw(root, train=True)
    shards = get_partitions(num_clients, partition, alpha, seed, root)
    shard = torch.from_numpy(shards[partition_id])

    client_images = images[shard]
    client_labels = labels[shard]
    if flip:
        client_labels = flip_labels(client_labels)

    dataset = TensorDataset(client_images, client_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def load_test_data(root, batch_size=512):
    """Build the DataLoader for the held-out test set used by the server."""
    images, labels = _load_raw(root, train=False)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
