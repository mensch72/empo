# Plasticity-Loss Diagnostics (Phase 2)

Phase 2 computes the robot policy against a **non-stationary** target: the
intrinsic reward `U_r` shifts as the mutual-dependency loop (equations 4–9)
converges, and the human-model networks (`V_h^e`, `X_h`) keep changing during
warm-up. Training neural networks on a moving target for many steps can cause
**plasticity loss** — the gradual erosion of a network's ability to keep
learning. Symptoms include neurons that stop firing, representations that
collapse to a low-dimensional subspace, and parameter norms that grow without
bound.

These diagnostics give cheap, forward-pass-only signals to detect plasticity
loss early. They are **disabled by default** and add one extra forward pass
plus a small SVD over a probe batch when enabled.

- Shared implementation: [src/empo/learning_based/plasticity_diagnostics.py](../src/empo/learning_based/plasticity_diagnostics.py)
- DQN-path wrapper: [src/empo/learning_based/phase2/diagnostics.py](../src/empo/learning_based/phase2/diagnostics.py)
- PPO-path wrapper: [src/empo/learning_based/phase2_ppo/diagnostics.py](../src/empo/learning_based/phase2_ppo/diagnostics.py)

## Enabling the diagnostics

Set the interval to a positive number of `training_step`s. `0` disables them.

**DQN path** (`Phase2Config`, used by bushworld / MultiGrid):

```python
config = Phase2Config(
    plasticity_diagnostics_interval=1000,   # measure every 1000 training_steps (0 = off)
    plasticity_diagnostics_batch_size=256,  # states sampled from the replay buffer to probe
    plasticity_dormant_tau=0.025,           # dormant-neuron threshold (see below)
)
```

**PPO path** (`PPOPhase2Config`):

```python
config = PPOPhase2Config(
    plasticity_diagnostics_interval=50,     # measure every 50 PPO iterations (0 = off)
    plasticity_diagnostics_batch_size=256,
    plasticity_dormant_tau=0.025,
)
```

Metrics are written to TensorBoard. Requires a TensorBoard writer to be active
(`tensorboard_dir` set); otherwise measurement is skipped.

The `examples/bushworld/bushworld_compare.py` script exposes this directly:

```bash
python examples/bushworld/bushworld_compare.py --method neural \
    --plasticity-diagnostics-interval 500
```

(TensorBoard logs are written under `<output-dir>/tensorboard`.)


## Where the metrics appear

All metrics live under the **`Plasticity/`** namespace in TensorBoard.

- **DQN path:** grouped per network, e.g. `Plasticity/q_r/...` and
  `Plasticity/v_r/...` (the robot value network is only probed when
  `v_r_use_network=True`). Lookup-table networks (no neural layers) are skipped.
- **PPO path:** a single actor-critic, e.g. `Plasticity/dormant_frac/...`.

The x-axis is the training-step counter (`training_step_count` for DQN, the
auxiliary training step for PPO).

## The metrics

### 1. Dormant / dead neurons

`dormant_frac/<layer>`, `dead_frac/<layer>`, and the aggregates
`dormant_frac/overall`, `dead_frac/overall`.

For each ReLU layer we compute every unit's **mean absolute activation** over
the probe batch, then normalise by the layer mean:

$$
s_i \;=\; \frac{\mathbb{E}_x\,\lvert a_i(x)\rvert}{\frac{1}{H}\sum_{j=1}^{H}\mathbb{E}_x\,\lvert a_j(x)\rvert}
$$

- A unit is **`tau`-dormant** when $s_i \le \tau$ (`plasticity_dormant_tau`,
  default `0.025`). This is the criterion from Sokar et al. (2023), *"The
  Dormant Neuron Phenomenon in Deep Reinforcement Learning."*
- A unit is **dead** when its mean absolute activation is (numerically) zero —
  it never fires on the batch. `dead_frac` uses `tau = 0`, so
  `dead_frac ≤ dormant_frac` always.

**How to read it.** Values are fractions in `[0, 1]`.

- Low and roughly flat (e.g. `dormant_frac/overall` < ~0.1) → healthy.
- A **rising** dormant/dead fraction over training is the clearest sign of
  plasticity loss: capacity is being lost as more units switch off.
- `dead_frac` climbing toward `dormant_frac` means dormant units are becoming
  permanently dead (unrecoverable without resets).

**What to do.** Lower the learning rate, add/keep normalisation, reduce the
target-shift rate (e.g. longer freeze intervals), or apply a plasticity
intervention (ReDo-style neuron resets, weight decay, periodic re-init).

### 2. Effective rank of the shared representation

`effective_rank/srank` and `effective_rank/erank`, computed from the state
encoder's output (the shared feature vector feeding the head). Let
$\sigma_1 \ge \sigma_2 \ge \dots$ be the singular values of the **mean-centred**
feature matrix (batch × feature_dim), and $p_i = \sigma_i / \sum_j \sigma_j$.

- **`srank`** (Kumar et al., 2020): the smallest $k$ whose top-$k$ singular
  values retain ≥ 99% of the total singular-value mass ($1-\delta$, $\delta=0.01$).
  An integer, bounded by `feature_dim`.
- **`erank`** (Roy & Vetterli, 2007): $\exp\big(-\sum_i p_i \log p_i\big)$, the
  exponential of the singular-value entropy. A smooth real number, also bounded
  by `feature_dim`.

**How to read it.** Both measure how many dimensions the representation
*actually* uses.

- Higher is generally better (the network spreads information across many
  dimensions).
- A **collapsing** (steadily decreasing) effective rank indicates
  representation collapse — a common companion to plasticity loss, where the
  encoder maps many distinct states onto a shrinking subspace.
- A value near `1` means near-total collapse (almost all states share one
  feature direction).

Watch the **trend**, not the absolute number: the natural scale depends on
`feature_dim` and the task. A sudden drop that coincides with a warm-up stage
transition or a learning-rate change is worth investigating.

### 3. Weight-norm growth

`weight_norm/total` and per-submodule norms (e.g. `weight_norm/q_head`,
`weight_norm/state_encoder` on the DQN path; `weight_norm/encoder`,
`weight_norm/actor`, `weight_norm/critic` on the PPO path). Each is the L2 norm
of the (grad-tracked) parameters.

**How to read it.**

- Slow growth that plateaus → normal.
- **Unbounded, monotonic growth** is a hallmark of plasticity loss: large
  weights push ReLU pre-activations into saturated regions, which in turn feeds
  the dormant-neuron problem above. Correlate a rising `weight_norm/*` with a
  rising `dormant_frac/*` and a falling `effective_rank/*` — together they are a
  strong plasticity-loss signature.

## Reading them together

| Signal | Healthy | Plasticity loss |
|---|---|---|
| `dormant_frac/overall`, `dead_frac/overall` | low, flat | rising |
| `effective_rank/{srank,erank}` | stable / high | falling toward 1 |
| `weight_norm/total` | plateaus | grows without bound |

A convincing diagnosis usually shows **all three** moving together. Any single
metric can move for benign reasons (e.g. effective rank dipping briefly at a
warm-up stage boundary), so treat them as corroborating evidence and always
read the **trend over training**, not a single snapshot.

## Notes and caveats

- Measurement runs under `torch.no_grad()` and restores each probed network's
  `train`/`eval` mode, so it does not affect optimisation.
- The probe batch is sampled from the replay buffer (DQN) or the current
  rollout states (PPO); metrics reflect the current data distribution.
- Cost is one forward pass + one SVD per measurement — keep the interval large
  (hundreds–thousands of steps) for long runs.
- Lookup-table (non-neural) networks are skipped automatically; these
  diagnostics only apply in neural mode.

## References

- G. Sokar, R. Agarwal, P. S. Castro, U. Evci. *The Dormant Neuron Phenomenon
  in Deep Reinforcement Learning.* ICML 2023.
- A. Kumar, R. Agarwal, D. Ghosh, S. Levine. *Implicit Under-Parameterization
  Inhibits Data-Efficient Deep Reinforcement Learning.* ICLR 2021 (srank).
- O. Roy, M. Vetterli. *The Effective Rank: A Measure of Effective
  Dimensionality.* EUSIPCO 2007 (erank).
