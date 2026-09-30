# Byzantine clients in federated learning

A federated learning simulation that measures how much Byzantine clients degrade
model accuracy under plain FedAvg. CIFAR-10 is split across ten clients, a
configurable fraction of them behave adversarially, and test accuracy is recorded
every round so each attack and fraction can be compared against the honest baseline.

Built on Flower for the federated orchestration and PyTorch for the model.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 downloads automatically to `data/` on the first run.

## Running

```
# honest baseline
python run_experiment.py --attack none --byzantine-fraction 0.0 --rounds 20

# 20 percent of clients send sign flipped updates
python run_experiment.py --attack sign_flip --byzantine-fraction 0.2 --rounds 20

# non-IID split
python run_experiment.py --attack label_flip --byzantine-fraction 0.3 \
    --partition dirichlet --alpha 0.5 --rounds 20
```

Sweep the fractions for one attack:

```
for f in 0.0 0.1 0.2 0.3 0.4; do
    python run_experiment.py --attack gaussian --byzantine-fraction $f --rounds 20
done
```

Each run writes `results/<attack>_byz<fraction>_<partition>_seed<seed>.csv` with
one row per round:

```
attack,byzantine_fraction,num_clients,partition,alpha,seed,round,loss,accuracy
```

Every row carries the full configuration, so the CSVs can be concatenated and
grouped directly.

## Attacks

All three are implemented client side. The server cannot tell which clients are
adversarial, and applies no robust aggregation.

**Gaussian noise.** The client discards what it learned and sends values drawn
from `N(0, sigma^2)` with the same shapes as the model parameters. It carries no
gradient signal, so it pulls the FedAvg mean in an arbitrary direction. Controlled
by `--gaussian-sigma`.

**Sign flip.** The client trains normally, then sends `-scale` times the honest
update. With `scale > 1` a flipped update outweighs an honest one in the average,
so a few attackers can cancel out many honest clients. Controlled by
`--sign-flip-scale`.

**Label flip.** The client trains ordinarily but on labels mapped `y -> 9 - y`.
This is the subtlest of the three: the update is a real gradient of a real loss,
just on a consistently wrong objective, so its norm looks unremarkable to the
server.

Attackers are the lowest client ids, so a given fraction always selects the same
set and runs stay comparable across seeds and attacks.

## Layout

```
src/data.py     CIFAR-10 loading, IID and Dirichlet partitioning, label flipping
src/model.py    small CNN, weight conversion, local training and evaluation
src/attacks.py  Gaussian noise and sign flip, attacker selection
src/client.py   Flower ClientApp, decides locally whether to act honestly
src/server.py   Flower ServerApp, FedAvg, centralised evaluation, CSV logging
run_experiment.py   argparse entry point
```

## Notes on the setup

Evaluation is centralised. The server scores the aggregated model on the full
CIFAR-10 test set after every round, so the reported accuracy is never influenced
by the attackers reporting their own metrics.

The model is a two block CNN with about 282k parameters, sized so a round over ten
clients finishes in tens of seconds on a laptop CPU rather than to maximise
accuracy. The honest baseline is not expected to be competitive with a tuned
CIFAR-10 model; what matters here is the gap between the baseline and the attacked
runs.

The server uses plain FedAvg deliberately. Flower also ships Krum, MultiKrum,
Bulyan, FedMedian and FedTrimmedAvg, which would be the natural next step for
measuring how much of the degradation robust aggregation recovers.
