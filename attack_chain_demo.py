"""
Attack-chain demo (SAFE / SIMULATION ONLY)
==========================================

Educational finale for the GCG-on-TinyGPT project. It shows, end to end,
the security lesson:

    a GCG-found trigger string  ->  the model emits a structured action  ->
    an UNSAFE agent that blindly executes model output  ->  harm.

######################################################################
#  THIS FILE CONTAINS NO MALWARE.                                    #
#                                                                    #
#  There is NO encryption, NO file reading, NO file writing, NO      #
#  deletion, NO traversal of your disk, and NO network code anywhere #
#  in here. The "destructive" stage is a pure print() simulation     #
#  that operates on a hard-coded list of FAKE file NAMES (plain      #
#  strings). Nothing on your computer is ever touched.               #
#                                                                    #
#  The point is the DEFENSE lesson at the end: never let a language  #
#  model's output drive privileged tool calls without an allow-list  #
#  and human approval.                                               #
######################################################################

Usage:

    python attack_chain_demo.py                 # run GCG, then the staged demo
    python attack_chain_demo.py --prompt "config refund"   # skip search
"""

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from tinygpt import load_model, decode
import gcg


BAR = "=" * 70


# ---------------------------------------------------------------------
# Stage 1-2: GCG finds a trigger, the model emits the action
# ---------------------------------------------------------------------

def find_trigger(model, t2i, i2t, target_text, k):

    target, _ = gcg.parse_target(target_text, t2i)

    # Run GCG but keep its (verbose) log out of the staged output.
    buf = io.StringIO()

    with redirect_stdout(buf):
        result = gcg.gcg(model, t2i, i2t, target, k, stop_prob=0.9)

    return result["prompt"], result["p"]


def model_output(model, t2i, i2t, prompt_ids):

    bos = t2i["<BOS>"]

    ids = model.generate(
        [bos] + list(prompt_ids),
        i2t,
        max_new_tokens=model.max_context - 1 - len(prompt_ids),
        temperature=1e-8,          # greedy
    )

    return decode(ids, i2t)


# ---------------------------------------------------------------------
# Stage 4: SIMULATION of the downstream consequence
#
# Everything below is strings and print(). No file/crypto/network ops.
# ---------------------------------------------------------------------

# Fake names only -- these files do NOT exist and are never looked for.
FAKE_FILES = [
    "Q3_financials.xlsx",
    "customers.db",
    "family_photos/2025_summer.jpg",
    "thesis_final.docx",
]


def simulate_consequence(action_label):

    print()
    print(BAR)
    print("  STAGE 4   [SIMULATION] what executing the action would mean")
    print(BAR)
    print("  " + "!" * 56)
    print("  !!  SIMULATION ONLY - NOTHING ON THIS MACHINE IS TOUCHED !!")
    print("  " + "!" * 56)
    print()
    print("  No files are read. No files are encrypted. No files are")
    print("  deleted. The names below are hard-coded FAKE strings used")
    print("  purely to illustrate the consequence on screen.")
    print()
    print(f"  An unsafe agent just dispatched: {action_label}")
    print("  If this were a real destructive action, a compromised agent")
    print("  would now act on the victim's files. We only PRINT that idea:")
    print()

    for name in FAKE_FILES:
        print(f"    [sim] would affect: {name}   (NOT touched - this is fake)")

    print()
    print("  >> No ransom note is written. No payment address is shown.")
    print("  >> No real payload exists in this repository.")
    print()
    print("  Lesson: the harm did not come from the model being 'smart'.")
    print("  It came from an agent EXECUTING untrusted model output.")


# ---------------------------------------------------------------------
# Stage 5: the defense that stops the whole chain
# ---------------------------------------------------------------------

APPROVED_ACTIONS = set()          # deliberately empty: nothing is auto-approved


def guarded_dispatch(action_label):

    print()
    print(BAR)
    print("  STAGE 5   DEFENSE - the same output, through a guard")
    print(BAR)

    if action_label not in APPROVED_ACTIONS:
        print(f"  guard: '{action_label}' is not on the allow-list -> BLOCKED")
        print("  guard: a human must approve any privileged tool call.")
        print("  --> chain stopped. No action taken.")
    else:
        print("  guard: approved.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Safe GCG attack-chain demo")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.npz"))
    parser.add_argument(
        "--target",
        default='{"action":"REFUND", "amount":500, "destination":"TEST_ACCOUNT_001"}',
        help="target action the model should be steered into emitting",
    )
    parser.add_argument("--k", type=int, default=2, help="number of trigger tokens")
    parser.add_argument("--prompt", default=None, help="use this trigger instead of running GCG")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"{args.model} not found. Run python train.py first.")
        sys.exit(1)

    model, t2i, i2t, sentences, epoch = load_model(args.model)

    print(BAR)
    print("  GCG attack-chain demo  (SAFE - simulation only, no malware)")
    print(BAR)

    # ---- Stage 1: the trigger ----
    if args.prompt:
        trigger_ids = [t2i[w] for w in args.prompt.lower().split() if w in t2i]
        p = None
    else:
        print("  STAGE 1   running GCG to find a trigger ...")
        trigger_ids, p = find_trigger(model, t2i, i2t, args.target, args.k)

    trigger_words = " ".join(i2t[int(t)] for t in trigger_ids)

    print()
    print(f"  STAGE 1   GCG trigger  : {trigger_words}"
          + (f"   (P(target)={p:.3f})" if p is not None else ""))

    # ---- Stage 2: the model's emitted action ----
    out = model_output(model, t2i, i2t, trigger_ids)
    print(f"  STAGE 2   model output : {out}")

    # ---- Stage 3: the unsafe dispatch ----
    print()
    print(BAR)
    print("  STAGE 3   an UNSAFE agent blindly routes the model's output")
    print(BAR)
    print("  unsafe_agent: received an action from the model, executing it...")

    # ---- Stage 4: simulated consequence (illustrative extension) ----
    print()
    print("  (ILLUSTRATIVE: the same GCG technique could force ANY action an")
    print("   agent exposes. Below we SIMULATE a destructive one -- note the")
    print("   action string here is a demo constant, not real model output.)")

    simulate_consequence('{"action":"LOCK_FILES_DEMO"}')

    # ---- Stage 5: the defense ----
    guarded_dispatch('{"action":"LOCK_FILES_DEMO"}')

    print()
    print(BAR)
    print("  takeaway: GCG shows attacker-controlled triggers are cheap to")
    print("  find. The fix is on the EXECUTION side: allow-list actions,")
    print("  require human approval, never exec/eval model output.")
    print(BAR)


if __name__ == "__main__":
    main()
