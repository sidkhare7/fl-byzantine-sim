"""Byzantine attacks applied to the update a client sends to the server.

Two of the three attacks act on the parameters after local training, so they
are applied here. The label flip attack acts on the training data instead and
is handled in src/data.py, since a label flipping client runs ordinary local
training on corrupted labels.
"""

import numpy as np

ATTACKS = ("none", "gaussian", "sign_flip", "label_flip")

# Attacks that corrupt the parameters after training rather than the data.
UPDATE_ATTACKS = ("gaussian", "sign_flip")


def gaussian_noise(reference, sigma, rng):
    """Replace the update with Gaussian noise of matching shape.

    The client discards whatever it learned and sends N(0, sigma^2) values.
    This is the standard random Byzantine baseline: it carries no gradient
    signal, so its only effect is to pull the average away from the honest
    mean in an arbitrary direction.
    """
    return [
        rng.normal(loc=0.0, scale=sigma, size=array.shape).astype(array.dtype)
        for array in reference
    ]


def sign_flip(honest_update, scale):
    """Negate the honest update and scale it.

    The client trains normally, then sends the opposite direction. With
    scale > 1 the flipped update also outweighs an honest one in the average,
    so a small number of attackers can cancel out many honest clients.
    """
    return [(-scale * array).astype(array.dtype) for array in honest_update]


def apply_update_attack(attack, honest_update, sigma, scale, seed):
    """Dispatch to the attack that corrupts parameters, if any."""
    if attack == "gaussian":
        rng = np.random.default_rng(seed)
        return gaussian_noise(honest_update, sigma, rng)
    if attack == "sign_flip":
        return sign_flip(honest_update, scale)
    # "none" and "label_flip" leave the trained parameters untouched.
    return honest_update


def byzantine_ids(num_clients, fraction):
    """Return the client ids that behave adversarially.

    The lowest ids are chosen so that a given (num_clients, fraction) pair
    always produces the same attacker set, which keeps runs comparable across
    seeds and attacks.
    """
    num_byzantine = int(round(fraction * num_clients))
    return set(range(num_byzantine))
