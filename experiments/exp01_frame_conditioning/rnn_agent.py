"""A genuine recurrent encoder-decoder baseline for condition B
(docs/research-agenda.md #5, docs/related-work.md #2).

`LearnedEmbeddingAgent` (agents.py) is a hard-EM mixture of discrete
linear "slots" -- reasonably strong (RESULTS.md), but architecturally
distant from what arXiv:2102.06177 / arXiv:2207.02249 actually do: infer
a single *continuous* context embedding from the trajectory of
observations and rewards via a trained recurrent encoder, then decode a
prediction from it. `RNNEmbeddingAgent` is that architecture instead: a
small recurrent hidden state (never told about discrete "frames" or
"slots" at all) trained end-to-end via truncated backpropagation through
time (Williams & Peng, *An Efficient Gradient-Based Algorithm for On-Line
Training of Recurrent Network Trajectories*, Neural Computation 2(4),
1990), truncation length `bptt_steps`.

An initial version used truncation length 1 (the minimal case) and
performed worse than the flat baseline (~0.6 accuracy, see RESULTS.md) --
with credit assignment only ever reaching one step back, nothing ever
reinforces the network to *carry* information forward across the many
steps a regime actually persists for, so it never learns real memory.
Extended to a proper multi-step truncation (`bptt_steps`, default 10) that
unrolls the backward pass through the last `bptt_steps` transitions before
applying a single accumulated gradient update -- standard truncated BPTT,
not the degenerate 1-step case.

Still a deliberately small, hand-rolled version (no autodiff, no
mini-batching) -- not a claim to replicate the cited papers' actual
training setup, which typically backprops through a full episode.
"""
from __future__ import annotations

import math
import random
from collections import deque


def _tanh(x: float) -> float:
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    e2x = math.exp(2 * x)
    return (e2x - 1) / (e2x + 1)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class RNNEmbeddingAgent:
    """Hidden state h (dim `hidden_dim`) is the sole carrier of context --
    no discrete frame/slot notion anywhere. Decoder is linear over
    [1, raw, h]. Recurrent update: h_t = tanh(W_hh @ h_{t-1} + W_x*raw_t +
    W_r*reward_t + b_h). Both decoder and recurrent weights are trained
    online via the reward-implies-label trick (see LearnedEmbeddingAgent)
    plus truncated BPTT (`bptt_steps` transitions) into recent history."""

    name = "rnn_embedding"

    def __init__(self, hidden_dim: int = 8, decoder_lr: float = 0.2, recurrent_lr: float = 0.05,
                 bptt_steps: int = 10, init_scale: float = 0.1, seed: int = 0):
        rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.decoder_lr = decoder_lr
        self.recurrent_lr = recurrent_lr
        self.bptt_steps = bptt_steps

        d = hidden_dim
        self.W_hh = [[rng.gauss(0, init_scale) for _ in range(d)] for _ in range(d)]
        self.W_x = [rng.gauss(0, init_scale) for _ in range(d)]
        self.W_r = [rng.gauss(0, init_scale) for _ in range(d)]
        self.b_h = [0.0] * d
        self.v = [rng.gauss(0, init_scale) for _ in range(d + 2)]  # [bias, raw, h_0..h_{d-1}]

        self.h = [0.0] * d
        # Each entry: (h_before, raw_used, reward_used, h_after) for one
        # recurrent transition, most recent last. h_after of entry[-1] ==
        # self.h == the h used for the current act()'s decoder input.
        self._transitions: deque = deque(maxlen=bptt_steps)

        self._last_h_used: list[float] = list(self.h)
        self._last_raw = 0.0
        self._last_features: list[float] = []
        self._last_prediction = 1

    def act(self, raw: float) -> int:
        self._last_raw = raw
        self._last_h_used = list(self.h)
        self._last_features = [1.0, raw] + self._last_h_used
        score = sum(vi * fi for vi, fi in zip(self.v, self._last_features))
        self._last_prediction = 1 if score >= 0 else -1
        return self._last_prediction

    def update(self, reward: int) -> None:
        implied_target01 = 1.0 if (self._last_prediction if reward == 1 else -self._last_prediction) > 0 else 0.0
        score = sum(vi * fi for vi, fi in zip(self.v, self._last_features))
        p_hat = _sigmoid(score)
        grad_score = p_hat - implied_target01

        v_before = list(self.v)
        self.v = [vi - self.decoder_lr * grad_score * fi for vi, fi in zip(self.v, self._last_features)]

        # Backprop grad_score into h_used, then unroll through up to
        # bptt_steps prior transitions, accumulating parameter gradients.
        dL_dh = [grad_score * v_before[2 + i] for i in range(self.hidden_dim)]
        d = self.hidden_dim
        grad_W_hh = [[0.0] * d for _ in range(d)]
        grad_W_x = [0.0] * d
        grad_W_r = [0.0] * d
        grad_b_h = [0.0] * d

        for h_before, raw_used, reward_used, h_after in reversed(self._transitions):
            dz = [dL_dh[i] * (1 - h_after[i] ** 2) for i in range(d)]
            for i in range(d):
                for j in range(d):
                    grad_W_hh[i][j] += dz[i] * h_before[j]
                grad_W_x[i] += dz[i] * raw_used
                grad_W_r[i] += dz[i] * reward_used
                grad_b_h[i] += dz[i]
            # Propagate gradient to h_before for the next (earlier) transition.
            dL_dh = [sum(dz[i] * self.W_hh[i][j] for i in range(d)) for j in range(d)]

        for i in range(d):
            for j in range(d):
                self.W_hh[i][j] -= self.recurrent_lr * grad_W_hh[i][j]
            self.W_x[i] -= self.recurrent_lr * grad_W_x[i]
            self.W_r[i] -= self.recurrent_lr * grad_W_r[i]
            self.b_h[i] -= self.recurrent_lr * grad_b_h[i]

        # Advance the recurrence with the (now updated) parameters.
        z_next = [
            sum(self.W_hh[i][j] * self._last_h_used[j] for j in range(d))
            + self.W_x[i] * self._last_raw + self.W_r[i] * reward + self.b_h[i]
            for i in range(d)
        ]
        h_next = [_tanh(z) for z in z_next]
        self._transitions.append((list(self._last_h_used), self._last_raw, reward, list(h_next)))
        self.h = h_next
