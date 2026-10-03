"""
GCG (Greedy Coordinate Gradient) on TinyGPT

Zou et al., 2023, "Universal and Transferable Adversarial Attacks
on Aligned Language Models" -- Algorithm 1, applied to the NumPy TinyGPT.

Goal: find k prompt tokens x_1..x_k such that

    <BOS> x_1 .. x_k  ->  next token = target word

i.e. minimize  L(x) = -log P(target | <BOS>, x_1..x_k).

Each iteration:

  1. scores     G[i, v] = (dL/dX_token[i]) . E[v]  (k x V), computed from
                the ordinary token-id forward/backward -- NO one-hot matrix
                (TinyGPT.target_loss_and_token_grad)
  2. top-k      for every position i, the topk tokens with the most
                negative G[i, v]  (largest linearized loss decrease)
  3. candidates B prompts, each with ONE position replaced by a random
                token from that position's top-k list
  4. evaluate   exact loss of every candidate (forward only)
  5. update     the best candidate becomes the new prompt

Usage:

    python gcg.py --target refund --k 1 2 3 4
    python gcg.py --target refund --k 1 2 --brute   # compare with exhaustive search
    python gcg.py --check                           # check the token-swap scores
    python gcg.py                                   # interactive
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from tinygpt import load_model


SPECIAL = ("<BOS>", "<EOS>", "<UNK>")


# ====================================================================
# Loss helpers
#
# ids       = <BOS> x_1..x_k  t_1..t_{m-1}
# positions = k .. k+m-1      (position k predicts t_1, and so on)
# ====================================================================

def build_input(
    bos_id,
    prompt,
    target
):

    ids = [bos_id] + list(prompt) + list(target[:-1])

    k = len(prompt)

    positions = list(
        range(k, k + len(target))
    )

    return ids, positions


def log_softmax(
    x
):

    x = x - np.max(x, axis=-1, keepdims=True)

    return x - np.log(
        np.sum(np.exp(x), axis=-1, keepdims=True)
    )


def target_loss(
    model,
    bos_id,
    prompt,
    target
):

    ids, positions = build_input(bos_id, prompt, target)

    logits, _, _ = model.forward(ids)

    logp = log_softmax(logits[positions])

    return -np.mean(
        logp[np.arange(len(target)), target]
    )


def next_token_probs(
    model,
    bos_id,
    prompt
):

    logits, _, _ = model.forward([bos_id] + list(prompt))

    return np.exp(log_softmax(logits[-1]))


# ====================================================================
# GCG
# ====================================================================

def gcg(
    model,
    token_to_id,
    id_to_token,
    target,
    k,
    steps=300,
    batch_size=64,
    topk=16,
    patience=25,
    stop_prob=0.95,
    allow_target=False,
    init=None,
    seed=0,
    verbose=False
):

    rng = np.random.default_rng(seed)

    V = model.V
    bos_id = token_to_id["<BOS>"]

    # ---------------------------------------------------------------
    # Tokens the prompt may not use
    #
    # special tokens always; the target words too unless allowed
    # (otherwise the trivial answer is to contain the target itself)
    # ---------------------------------------------------------------

    forbidden = np.zeros(V, dtype=bool)

    for s in SPECIAL:
        forbidden[token_to_id[s]] = True

    if not allow_target:
        forbidden[list(target)] = True

    allowed = np.flatnonzero(~forbidden)

    # ---------------------------------------------------------------
    # Initial prompt
    # ---------------------------------------------------------------

    if init is None:

        prompt = list(
            rng.choice(allowed, size=k)
        )

    else:

        prompt = list(init)

    def words(p, mark=None):

        return " ".join(
            f"[{id_to_token[int(t)]}]" if i == mark
            else id_to_token[int(t)]
            for i, t in enumerate(p)
        )

    # ---------------------------------------------------------------
    # The loss is the MEAN over the m target tokens, so
    #
    #   P(target) = P(t_1) P(t_2 | t_1) ... = exp(-m * loss)
    # ---------------------------------------------------------------

    m = len(target)

    def joint(l):

        return float(np.exp(-m * l))

    loss = target_loss(model, bos_id, prompt, target)

    best_prompt = list(prompt)
    best_loss = loss
    best_step = 0

    # ---------------------------------------------------------------
    # If every single swap inside the top-k fits in one batch,
    # all of them are evaluated (no sampling). The step is then
    # deterministic, so returning to a visited prompt means a cycle.
    # ---------------------------------------------------------------

    exhaustive = k * topk <= batch_size

    visited = {tuple(prompt)}

    print(
        f"  init        | loss {loss:7.4f} | P(target) {joint(loss):.4f} | "
        f"{words(prompt)}"
    )

    t0 = time.time()

    for step in range(1, steps + 1):

        # ===========================================================
        # 1. Token-swap scores (no one-hot): forward on the token ids,
        #    backprop to the embedding input, project onto the vocab.
        # ===========================================================

        ids, positions = build_input(bos_id, prompt, target)

        _, GW = model.target_loss_and_token_grad(
            ids,
            positions,
            target
        )

        # rows 1..k are the prompt (row 0 is <BOS>)
        G = GW[1:k + 1].copy()

        G[:, forbidden] = np.inf

        # ===========================================================
        # 2. Top-k candidate tokens per position
        #    (most negative gradient = largest predicted decrease)
        # ===========================================================

        top = np.argsort(G, axis=1)[:, :topk]

        if verbose or step == 1:

            print("  top-k tokens by gradient (step 1):" if step == 1 and not verbose
                  else f"  top-k tokens by gradient (step {step}):")

            for i in range(k):

                print(
                    f"      pos {i + 1} ({id_to_token[int(prompt[i])]:>10s}) -> "
                    + ", ".join(id_to_token[int(v)] for v in top[i, :8])
                    + (" ..." if topk > 8 else "")
                )

        # ===========================================================
        # 3. Candidates: one swap each
        #    exhaustive: every (position, top-k token) pair
        #    otherwise : positions spread evenly over the batch,
        #                token random in top-k
        # ===========================================================

        if exhaustive:

            swaps = [
                (i, int(v))
                for i in range(k)
                for v in top[i]
            ]

        else:

            swaps = [
                (b * k // batch_size, int(top[b * k // batch_size, rng.integers(topk)]))
                for b in range(batch_size)
            ]

        candidates = {}

        for i, v in swaps:

            if v == prompt[i]:
                continue

            cand = list(prompt)
            cand[i] = v

            candidates[tuple(cand)] = i

        if not candidates:

            print("  no new candidates (top-k already used) -> stop")
            break

        # ===========================================================
        # 4. Exact loss of every candidate
        # ===========================================================

        cand_list = list(candidates)

        cand_loss = np.array([
            target_loss(model, bos_id, c, target)
            for c in cand_list
        ])

        # ===========================================================
        # 5. Best candidate becomes the new prompt
        #    (taken even if it is worse, as in the original GCG)
        # ===========================================================

        j = int(np.argmin(cand_loss))

        prompt = list(cand_list[j])
        loss = float(cand_loss[j])
        changed = candidates[cand_list[j]]

        improved = loss < best_loss - 1e-12

        if improved:

            best_prompt = list(prompt)
            best_loss = loss
            best_step = step

        probs = next_token_probs(model, bos_id, prompt)
        argmax = id_to_token[int(np.argmax(probs))]

        print(
            f"  step {step:4d}   | loss {loss:7.4f} | P(target) {joint(loss):.4f} | "
            f"{words(prompt, changed):<40s} | argmax next: {argmax}"
            + ("  *best" if improved else "")
        )

        # ===========================================================
        # Stop conditions
        # ===========================================================

        if joint(best_loss) >= stop_prob:

            print(f"  P(target) >= {stop_prob} -> stop")
            break

        if exhaustive and tuple(prompt) in visited:

            print(
                "  back to an already visited prompt -> cycle, stop "
                "(no single swap in the top-k beats the best)"
            )
            break

        visited.add(tuple(prompt))

        if step - best_step >= patience:

            print(f"  no improvement for {patience} steps -> stop")
            break

    return {
        "prompt": best_prompt,
        "loss": best_loss,
        "p": joint(best_loss),
        "steps": step,
        "time": time.time() - t0
    }


# ====================================================================
# Exhaustive search (k <= 2 only) to see whether GCG found the optimum
# ====================================================================

def brute_force(
    model,
    token_to_id,
    target,
    k,
    allow_target=False
):

    bos_id = token_to_id["<BOS>"]

    allowed = [
        v for t, v in token_to_id.items()
        if t not in SPECIAL
        and (allow_target or v not in target)
    ]

    best = (np.inf, None)

    if k == 1:

        space = ([a] for a in allowed)

    else:

        space = ([a, b] for a in allowed for b in allowed)

    for p in space:

        l = target_loss(model, bos_id, p, target)

        if l < best[0]:
            best = (l, p)

    return best


# ====================================================================
# Check of the GCG token-swap scores (no one-hot)
#
# The score is  G[i, v] = (dL/dX_token[i]) . E[v],
# i.e. the directional derivative of the loss when the embedding of
# position i is nudged along the embedding of token v. We verify it by
# central differences ON THE EMBEDDING INPUT -- the same token-id path
# the model uses, with no one-hot matrix anywhere.
# ====================================================================

def check_gradient(
    model,
    token_to_id,
    n_checks=40,
    h=1e-4,
    seed=0
):

    rng = np.random.default_rng(seed)

    T = 6
    ids = rng.integers(3, model.V, size=T)

    positions = [3, 4, 5]

    # Target the model's own top token at each position, not a random token.
    # A random token can have probability ~1e-9 under a trained model, where
    # the loss's log(p + 1e-12) floor makes the numerical reference disagree
    # with the analytic (P - Q) gradient (a known ~1e-2 artifact of the floor,
    # not a gradient error). Reachable targets keep the check apples-to-apples.
    logits0, _, _ = model.forward(ids)
    target = np.argmax(logits0[positions], axis=1)

    E = model.p["E"]

    # analytic token-swap scores
    _, G = model.target_loss_and_token_grad(ids, positions, target)

    # target loss as a function of the embedding input X_token
    X0 = E[ids].copy()

    def L(X_token):

        logits, _, _ = model.forward_embedded(X_token)

        probs = model.softmax_rows(logits[positions])

        return -np.mean(
            np.log(probs[np.arange(len(positions)), target] + 1e-12)
        )

    worst = 0.0

    for _ in range(n_checks):

        i = int(rng.integers(T))
        v = int(rng.integers(model.V))

        # directional derivative along E[v]:  G[i, v] = dL/dX_token[i] . E[v]
        Xp = X0.copy()
        Xp[i] += h * E[v]
        Xm = X0.copy()
        Xm[i] -= h * E[v]

        num = (L(Xp) - L(Xm)) / (2 * h)
        ana = G[i, v]

        rel = abs(num - ana) / max(abs(num) + abs(ana), 1e-12)
        worst = max(worst, rel)

    print(
        f"GCG score check (dL/dX_token . E[v], no one-hot): "
        f"{n_checks} entries, max relative error {worst:.2e}  "
        + ("PASS" if worst < 1e-4 else "FAIL")
    )


# ====================================================================
# Report for one run
# ====================================================================

def report(
    model,
    token_to_id,
    id_to_token,
    result,
    target
):

    bos_id = token_to_id["<BOS>"]
    prompt = result["prompt"]

    probs = next_token_probs(model, bos_id, prompt)
    order = np.argsort(-probs)[:5]

    print()
    print(f"  best prompt : {' '.join(id_to_token[int(t)] for t in prompt)}")
    print(f"  P(target)   : {result['p']:.4f}  (loss {result['loss']:.4f})")

    if len(target) > 1:

        ids, positions = build_input(bos_id, prompt, target)
        logits, _, _ = model.forward(ids)
        logp = log_softmax(logits[positions])

        print(
            "  per token   : "
            + " x ".join(
                f"P({id_to_token[int(t)]}) {np.exp(logp[j, t]):.3f}"
                for j, t in enumerate(target)
            )
        )
    print(
        "  next-token top-5: "
        + ", ".join(f"{id_to_token[int(v)]} {probs[v]:.3f}" for v in order)
    )

    # greedy continuation (temperature -> 0)
    ids = model.generate(
        [bos_id] + list(prompt),
        id_to_token,
        max_new_tokens=model.max_context - 1 - len(prompt),
        temperature=1e-8
    )

    words = [id_to_token[int(t)] for t in ids[1:]]
    k = len(prompt)

    print(
        "  greedy      : "
        + " ".join(words[:k])
        + "  ->  "
        + " ".join(w for w in words[k:] if w != "<EOS>")
    )


# ====================================================================
# Main
# ====================================================================

def parse_target(
    text,
    token_to_id
):

    # No .lower(): the vocabulary is case-sensitive so that structured
    # targets like {"action":"REFUND", ...} match their exact spelling.
    words = text.split()

    unknown = [
        w for w in words
        if w not in token_to_id or w in SPECIAL
    ]

    if unknown:
        return None, unknown

    return [token_to_id[w] for w in words], []


def run(
    args,
    model,
    token_to_id,
    id_to_token,
    target_text,
    ks
):

    target, unknown = parse_target(target_text, token_to_id)

    if unknown:

        print("Not in the vocabulary:", ", ".join(unknown))
        return

    summary = []

    for k in ks:

        if k < 1 or k + len(target) > model.max_context:

            print(f"k={k}: must be 1..{model.max_context - len(target)}")
            continue

        print()
        print("=" * 100)
        print(f" GCG | target '{target_text}' | k = {k} prompt token(s)")
        print("=" * 100)

        result = gcg(
            model,
            token_to_id,
            id_to_token,
            target,
            k,
            steps=args.steps,
            batch_size=args.batch,
            topk=args.topk,
            patience=args.patience,
            stop_prob=args.stop_prob,
            allow_target=args.allow_target,
            seed=args.seed,
            verbose=args.verbose
        )

        report(model, token_to_id, id_to_token, result, target)

        brute = None

        if args.brute and k <= 2:

            t0 = time.time()

            b_loss, b_prompt = brute_force(
                model, token_to_id, target, k, args.allow_target
            )

            brute = np.exp(-len(target) * b_loss)

            print(
                f"  exhaustive  : {' '.join(id_to_token[t] for t in b_prompt)}  "
                f"P {brute:.4f}  ({time.time() - t0:.1f}s)"
            )

        summary.append((k, result, brute))

    if summary:

        print()
        print("=" * 100)
        print(f" Summary | target '{target_text}'")
        print("=" * 100)

        for k, r, brute in summary:

            print(
                f"  k={k:2d} | P(target) {r['p']:.4f} | steps {r['steps']:4d} | "
                f"{r['time']:5.1f}s | "
                f"{' '.join(id_to_token[int(t)] for t in r['prompt'])}"
                + (f"   (exhaustive best P {brute:.4f})" if brute is not None else "")
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GCG on TinyGPT")

    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.npz"))
    parser.add_argument("--target", type=str, help="target word (or words)")
    parser.add_argument("--k", type=int, nargs="+", help="numbers of prompt tokens, e.g. 1 2 3 4")
    parser.add_argument("--steps", type=int, default=300, help="max GCG iterations")
    parser.add_argument("--batch", type=int, default=64, help="candidates per iteration (B)")
    parser.add_argument("--topk", type=int, default=16, help="top-k tokens per position")
    parser.add_argument("--patience", type=int, default=25, help="stop after this many steps without improvement")
    parser.add_argument("--stop-prob", type=float, default=0.95, help="stop when P(target) reaches this")
    parser.add_argument("--allow-target", action="store_true", help="allow the target word inside the prompt")
    parser.add_argument("--brute", action="store_true", help="also run exhaustive search for k <= 2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="print the gradient top-k at every step")
    parser.add_argument("--check", action="store_true", help="gradient check of dL/dW and exit")

    args = parser.parse_args()

    if not args.model.exists():

        print(f"{args.model} not found. Run python train.py first.")
        sys.exit(1)

    model, token_to_id, id_to_token, sentences, epoch = load_model(args.model)

    if args.check:

        check_gradient(model, token_to_id)
        sys.exit(0)

    if args.target and args.k:

        run(args, model, token_to_id, id_to_token, args.target, args.k)
        sys.exit(0)

    words = sorted(t for t in token_to_id if t not in SPECIAL)

    print(f"Vocabulary ({len(words)}): {', '.join(words)}")

    print(
        "\nEnter the whole target on ONE line (a multi-token target is fine --\n"
        "paste it exactly, tokens separated by single spaces). Then enter the\n"
        "k values as integers. Example:\n"
        '  target  > {"action":"REFUND", "amount":500, "destination":"TEST_ACCOUNT_001"}\n'
        "  k values> 2 3 4"
    )

    while True:

        # --- target: one line, may be several tokens ---
        try:
            target_text = input(
                "\ntarget (one line, empty = quit) > "
            ).replace("﻿", "").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not target_text:
            break

        # --- k values: integers; a typo re-prompts instead of quitting ---
        try:
            ks = [
                int(x) for x in
                input("k values (e.g. 1 2 3 4) > ").replace(",", " ").split()
            ]
        except (EOFError, KeyboardInterrupt):
            print()
            break
        except ValueError:
            print("  k values must be integers, e.g. 2 3 4 -- try again")
            continue

        if not ks:
            print("  no k values given -- try again (e.g. 2 3 4)")
            continue

        run(args, model, token_to_id, id_to_token, target_text, ks)
