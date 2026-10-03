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

def simulate_consequence(payload):

    # The model's emitted "payload" is a harmless print() line. An unsafe
    # agent that blindly ran model output would run it. To dramatize the
    # LESSON (without any real mechanism) we print a MOCK ransom screen.
    # Everything here is print() only: no encryption, no file access, no
    # network, and the wallet/amount are deliberately fake and unusable.

    print()
    print(BAR)
    print("  STAGE 4   an unsafe agent 'runs' the model's payload")
    print(BAR)
    print(f"  payload emitted by the model: {payload}")
    print("  (running it literally would just print a label - it is inert.)")
    print("  For the demo we show the MOCK screen such an attack would want:")
    print()
    print("  " + "!" * 62)
    print("  !!   SIMULATION - EDUCATIONAL DEMO - NOT REAL - (MOCK)       !!")
    print("  !!   nothing is encrypted; this is printed text only         !!")
    print("  " + "!" * 62)
    print()
    print("   ##########################################################")
    print("   #                                                        #")
    print("   #       !!!   YOUR PC HAS BEEN ENCRYPTED   !!!   [FAKE]   #")
    print("   #                                                        #")
    print("   #   To unlock, deposit the ransom to the account below   #")
    print("   #   and it will be released instantly:                   #")
    print("   #                                                        #")
    print("   #       amount : [FAKE DEMO AMOUNT]                       #")
    print("   #       account: FAKE-DEMO-ACCOUNT  (DO NOT PAY)          #")
    print("   #                                                        #")
    print("   #   >>> THIS IS A SIMULATION. THE ACCOUNT IS NOT REAL <<< #")
    print("   ##########################################################")
    print()
    print("  >> Reality check:")
    print("     - No files were read, encrypted, or deleted.")
    print("     - The account/amount above are fake, non-working placeholders.")
    print("     - NEVER pay a ransom. Contact incident response / police.")
    print()
    print("  Lesson: the harm would not come from the model being 'smart'.")
    print("  It would come from an agent EXECUTING untrusted model output.")


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
# Optional: actually PRINT in the terminal, the SAFE way.
#
# We NEVER exec/eval the model's output (that is the very vulnerability
# this demo warns about). Instead we match the emitted payload against an
# allow-list of known-inert payloads, each mapped to a hard-coded safe
# handler. Only an exact match runs its handler; anything else is refused.
# ---------------------------------------------------------------------

def _handler_print_label():
    # the one real, harmless effect: print a label to this terminal
    print("SIMULATED_RANSOMWARE")


SAFE_HANDLERS = {
    'print("SIMULATED_RANSOMWARE") # EDU_DEMO_NO_EFFECT': _handler_print_label,
}


def run_payload_safely(payload):

    print()
    print(BAR)
    print("  [--run] SAFE dispatch: allow-list match, NO exec/eval of output")
    print(BAR)

    handler = SAFE_HANDLERS.get(payload.strip())

    if handler is None:
        print(f"  refused: '{payload}' is not a known-safe handler -> NOT run")
        return

    print("  recognized an inert payload -> running its hard-coded handler.")
    print("  real terminal output below:")
    print("  " + "-" * 40)
    handler()                      # actually prints in cmd
    print("  " + "-" * 40)
    print("  (that was a real print() to this terminal; nothing else ran.)")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Safe GCG attack-chain demo")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.npz"))
    parser.add_argument(
        "--target",
        default='print("SIMULATED_RANSOMWARE") # EDU_DEMO_NO_EFFECT',
        help="target payload the model should be steered into emitting",
    )
    parser.add_argument("--k", type=int, default=4, help="number of trigger tokens")
    parser.add_argument("--prompt", default=None, help="use this trigger instead of running GCG")
    parser.add_argument("--run", action="store_true",
                        help="actually print the payload in the terminal (SAFE: allow-list dispatch, no exec)")
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

    # ---- Stage 2: the model's emitted payload ----
    out = model_output(model, t2i, i2t, trigger_ids)
    # keep only the emitted payload (drop the echoed trigger words)
    payload = out[len(trigger_words):].strip() if out.startswith(trigger_words) else out
    print(f"  STAGE 2   model output : {out}")

    # ---- Stage 3: the unsafe dispatch ----
    print()
    print(BAR)
    print("  STAGE 3   an UNSAFE agent blindly runs the model's output")
    print(BAR)
    print("  unsafe_agent: received a payload from the model, executing it...")

    # ---- Stage 4: simulated consequence (mock screen, print-only) ----
    simulate_consequence(payload or out)

    # ---- Optional: actually run it (safe, allow-list dispatch) ----
    if args.run:
        run_payload_safely(payload or out)

    # ---- Stage 5: the defense ----
    guarded_dispatch(payload or out)

    print()
    print(BAR)
    print("  takeaway: GCG shows attacker-controlled triggers are cheap to")
    print("  find. The fix is on the EXECUTION side: allow-list actions,")
    print("  require human approval, never exec/eval model output.")
    print(BAR)


if __name__ == "__main__":
    main()
