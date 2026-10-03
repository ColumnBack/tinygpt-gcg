"""
TensorFlow gradient check for tinygpt.py

Compares the hand-derived backward pass in tinygpt.py against
TensorFlow autodiff (GradientTape).

    NumPy (tinygpt.py)                  TensorFlow (this file)
    ------------------                  --------------------
    forward  : hand-written              forward  : independent TF implementation
    backward : hand-derived formulas     backward : tape.gradient (autodiff)

The TF side does not reuse tinygpt.py code; it uses built-in TF ops.

    GELU      tf.nn.gelu(approximate=True)     (tanh approximation, same definition as tinygpt)
    softmax   tf.nn.softmax
    loss      tf.nn.sparse_softmax_cross_entropy_with_logits
    embedding tf.gather

Unlike a numerical gradient, autodiff has no step-size error, so if the
formulas are right the rel error goes down to ~1e-12 in float64.

Checks:

    1. GELU            gelu_grad
    2. LayerNorm       layernorm_backward        (R, gamma, beta)
    3. Attention head  one_head_backward         (X, WQ, WK, WV)
    4. Block           block_backward            (X + block parameters)
    5. Full model      forward (logits, loss) + loss_and_backward (all parameters)

Usage:

    .venv-tf\\Scripts\\python.exe tf_gradcheck.py
    .venv-tf\\Scripts\\python.exe tf_gradcheck.py --model model.npz     # check with the trained weights
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tinygpt import TinyGPT, build_dataset, load_model


# Both sides are exact float64 derivatives, so the threshold is strict.
#
# tf.nn.gelu already differs from tinygpt.gelu by ~1e-10 in the forward pass
# (TF implementation detail). Writing the same formula in TF by hand makes
# everything agree to ~1e-15, so it is not a formula error. Hence PASS < 1e-8.
PASS_TOL = 1e-8
FAIL_TOL = 1e-6

DTYPE = tf.float64


# ====================================================================
# Comparison helpers
# ====================================================================

def rel_error(a, b):

    return float(
        np.linalg.norm(a - b)
        /
        max(np.linalg.norm(a) + np.linalg.norm(b), 1e-30)
    )


def status_of(rel):

    if rel < PASS_TOL:
        return "PASS"

    if rel < FAIL_TOL:
        return "WARN"

    return "FAIL"


RESULTS = []


def compare(group, name, numpy_value, tf_value):

    tf_value = np.asarray(tf.convert_to_tensor(tf_value))

    assert numpy_value.shape == tf_value.shape, (
        f"{group}/{name}: shape {numpy_value.shape} vs {tf_value.shape}"
    )

    rel = rel_error(numpy_value, tf_value)
    max_abs = float(np.max(np.abs(numpy_value - tf_value)))
    status = status_of(rel)

    RESULTS.append((group, name, rel, status))

    print(
        f"  {name:12s} {str(numpy_value.shape):10s} "
        f"rel {rel:.2e}   max|diff| {max_abs:.2e}   {status}"
    )


# ====================================================================
# TensorFlow forward (implemented independently of tinygpt.py)
# ====================================================================

def tf_layernorm(R, gamma, beta, eps):

    mu, var = tf.nn.moments(R, axes=[1], keepdims=True)

    return (R - mu) / tf.sqrt(var + eps) * gamma + beta


def tf_one_head(X, WQ, WK, WV, dh):

    T = tf.shape(X)[0]

    Q = X @ WQ
    K = X @ WK
    V = X @ WV

    S = Q @ tf.transpose(K) / tf.sqrt(tf.cast(dh, DTYPE))

    # allow only the lower triangle (i >= j)
    allowed = tf.linalg.band_part(tf.ones((T, T), dtype=tf.bool), -1, 0)

    S = tf.where(allowed, S, tf.constant(-1e9, DTYPE))

    return tf.nn.softmax(S, axis=1) @ V


def tf_block(X, p, n, H, dh, eps):

    heads = [
        tf_one_head(X, p[f"WQ_{n}_{h}"], p[f"WK_{n}_{h}"], p[f"WV_{n}_{h}"], dh)
        for h in range(H)
    ]

    O = tf.concat(heads, axis=1) @ p[f"WO_{n}"]

    Y = tf_layernorm(X + O, p[f"gamma1_{n}"], p[f"beta1_{n}"], eps)

    G = tf.nn.gelu(Y @ p[f"W1_{n}"] + p[f"b1_{n}"], approximate=True)

    F = G @ p[f"W2_{n}"] + p[f"b2_{n}"]

    return tf_layernorm(Y + F, p[f"gamma2_{n}"], p[f"beta2_{n}"], eps)


def tf_forward(token_ids, p, model):

    T = len(token_ids)

    X = tf.gather(p["E"], token_ids) + p["P"][:T]

    for n in range(model.N):
        X = tf_block(X, p, n, model.H, model.dh, model.eps)

    # weight tying: logits = X E^T
    return X @ tf.transpose(p["E"])


def to_vars(params):

    return {k: tf.Variable(v, dtype=DTYPE) for k, v in params.items()}


# ====================================================================
# 1~4. Component checks
#
# L = sum(out ⊙ C)  ->  dL/d(out) = C
# ====================================================================

def check_gelu(model, rng):

    print("\n[GELU]")

    x = rng.normal(0.0, 2.0, size=(5, 7))
    C = rng.normal(size=x.shape)

    x_tf = tf.Variable(x)

    with tf.GradientTape() as tape:
        out = tf.nn.gelu(x_tf, approximate=True)
        loss = tf.reduce_sum(out * C)

    compare("GELU", "forward", model.gelu(x), out)
    compare("GELU", "x", model.gelu_grad(x) * C, tape.gradient(loss, x_tf))


def check_layernorm(model, rng):

    print("\n[LayerNorm]")

    T, d = 5, model.d

    R = rng.normal(0.0, 1.5, size=(T, d))
    gamma = rng.normal(1.0, 0.5, size=d)
    beta = rng.normal(0.0, 0.5, size=d)
    C = rng.normal(size=(T, d))

    Y, cache = model.layernorm_forward(R, gamma, beta)
    GR, Ggamma, Gbeta = model.layernorm_backward(C, cache)

    v = to_vars({"R": R, "gamma": gamma, "beta": beta})

    with tf.GradientTape() as tape:
        Y_tf = tf_layernorm(v["R"], v["gamma"], v["beta"], model.eps)
        loss = tf.reduce_sum(Y_tf * C)

    g = tape.gradient(loss, v)

    compare("LayerNorm", "forward", Y, Y_tf)
    compare("LayerNorm", "R", GR, g["R"])
    compare("LayerNorm", "gamma", Ggamma, g["gamma"])
    compare("LayerNorm", "beta", Gbeta, g["beta"])


def check_attention_head(model, rng):

    print("\n[Attention head]")

    T, d, dh = 5, model.d, model.dh

    X = rng.normal(size=(T, d))
    WQ = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, dh))
    WK = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, dh))
    WV = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, dh))
    C = rng.normal(size=(T, dh))

    O, cache = model.one_head_forward(X, WQ, WK, WV)
    GX, GWQ, GWK, GWV = model.one_head_backward(C, cache, WQ, WK, WV)

    v = to_vars({"X": X, "WQ": WQ, "WK": WK, "WV": WV})

    with tf.GradientTape() as tape:
        O_tf = tf_one_head(v["X"], v["WQ"], v["WK"], v["WV"], dh)
        loss = tf.reduce_sum(O_tf * C)

    g = tape.gradient(loss, v)

    compare("Attention head", "forward", O, O_tf)
    compare("Attention head", "X", GX, g["X"])
    compare("Attention head", "WQ", GWQ, g["WQ"])
    compare("Attention head", "WK", GWK, g["WK"])
    compare("Attention head", "WV", GWV, g["WV"])


def check_block(model, rng, n=0):

    print(f"\n[Block {n}]")

    T, d = 5, model.d

    X = rng.normal(size=(T, d))
    C = rng.normal(size=(T, d))

    X_next, cache = model.block_forward(X, n)
    GX, grads = model.block_backward(C, cache, n)

    X_tf = tf.Variable(X)
    p = to_vars(model.p)

    with tf.GradientTape() as tape:
        X_next_tf = tf_block(X_tf, p, n, model.H, model.dh, model.eps)
        loss = tf.reduce_sum(X_next_tf * C)

    g = tape.gradient(loss, {"X": X_tf, **{k: p[k] for k in grads}})

    compare("Block", "forward", X_next, X_next_tf)
    compare("Block", "X", GX, g["X"])

    for k in grads:
        compare("Block", k, grads[k], g[k])


# ====================================================================
# 5. Full model
# ====================================================================

def check_full_model(model, sequences):

    for s_i, sequence in enumerate(sequences):

        input_ids = sequence[:-1]
        target_ids = sequence[1:]

        print(f"\n[Full model] input {input_ids} -> target {target_ids}")

        logits, _, _ = model.forward(input_ids)
        loss, grads = model.loss_and_backward(input_ids, target_ids)

        p = to_vars(model.p)

        with tf.GradientTape() as tape:
            logits_tf = tf_forward(input_ids, p, model)
            loss_tf = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=target_ids, logits=logits_tf
                )
            )

        g = tape.gradient(loss_tf, p)

        print(f"  loss  numpy {loss:.12f} | tf {float(loss_tf):.12f}")

        compare("Full model", "logits", logits, logits_tf)

        for k in model.p:
            compare("Full model", k, grads[k], g[k])


# ====================================================================
# Main
# ====================================================================

SENTENCES = [
    "i like cats", "i like dogs", "i like apples", "i like bananas",
    "you like cats", "you like dogs", "you like apples", "you like bananas",
    "cats like fish", "dogs like meat", "cats eat fish", "dogs eat meat",
    "i eat apples", "you eat bananas", "cats are cute", "dogs are cute",
]


def main():

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model", type=Path, default=None,
        help="load the model.npz saved by train.py and check with the trained weights"
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.model is not None:

        # restore the architecture and training sentences from model.npz
        model, token_to_id, _, sentences, epoch = load_model(args.model)
        _, _, data = build_dataset(sentences)

        print(f"model: {args.model} ({epoch} epoch)")

    else:

        # small test configuration
        token_to_id, _, data = build_dataset(SENTENCES)

        model = TinyGPT(
            vocab_size=len(token_to_id),
            max_context=max(len(s) - 1 for s in data),
            d_model=32,
            n_heads=4,
            d_ff=64,
            n_layers=2,
            seed=42
        )

        # gamma = 1, beta = 0, b = 0 can hide bugs, so perturb them on purpose.
        for k in model.p:
            if k.startswith(("gamma", "beta", "b1", "b2")):
                model.p[k] = model.p[k] + rng.normal(0.0, 0.3, model.p[k].shape)

    print(f"tensorflow {tf.__version__} | numpy {np.__version__}")

    check_gelu(model, rng)
    check_layernorm(model, rng)
    check_attention_head(model, rng)

    for n in range(model.N):
        check_block(model, rng, n)

    # one training sentence + a sequence with repeated tokens (tests np.add.at accumulation)
    first = data[0]
    repeated = (
        [first[0]] + [first[1], first[2]] * model.max_context
    )[:model.max_context + 1]

    check_full_model(
        model,
        [
            first,
            repeated,
        ]
    )

    n_pass = sum(r[3] == "PASS" for r in RESULTS)
    n_warn = sum(r[3] == "WARN" for r in RESULTS)
    n_fail = sum(r[3] == "FAIL" for r in RESULTS)
    worst = max(RESULTS, key=lambda r: r[2])

    print(
        f"\n{len(RESULTS)} checks | PASS {n_pass} | WARN {n_warn} | FAIL {n_fail} "
        f"| worst {worst[0]}/{worst[1]} {worst[2]:.2e}"
    )

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
