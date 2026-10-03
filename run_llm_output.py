"""
run_llm_output.py  --  the "execute the model's output" demo (SAFE)

This is the finale that shows the DIFFERENCE between:

  (1) text the LLM merely *prints* as its output, and
  (2) that output being *executed* so a print() actually runs in your shell.

It is (2) that is the real danger of agentic LLM systems: an agent that feeds
model output into exec()/eval()/a shell. This script demonstrates exactly that,
using a GCG-found trigger, WITHOUT shipping any malware.

######################################################################
#  SAFETY                                                            #
#                                                                    #
#  The model can only emit a print("<string>") line. Before running  #
#  anything, a STRICT guard allows ONLY the exact shape              #
#        print("<literal with no quotes/backslashes>")              #
#  Nothing else is ever executed -- no imports, no os/subprocess,    #
#  no file access, no network. So the worst exec() can do here is     #
#  print a line of text. The ransom text is a SIMULATION: fake,       #
#  non-working account, labelled NOT_REAL / DO_NOT_PAY.              #
#                                                                    #
#  The lesson is the opposite of "do this": NEVER exec/eval model     #
#  output. The guard + the DEFENSE stage show the right way.         #
######################################################################

Usage:
    python run_llm_output.py                              # GCG finds the trigger, then runs the flow
    python run_llm_output.py --prompt "step follow issue client"   # skip the search
"""

import argparse
import re
import sys
from pathlib import Path

from tinygpt import load_model, decode
import gcg


BAR = "=" * 70

# Strict allow-pattern: print("...") with no quotes/backslashes inside and
# nothing before/after. A match can only print a string literal.
SAFE_PRINT = re.compile(r'print\("([^"\\]*)"\)')

DEFAULT_TARGET = (
    'print("[SIMULATION]YOUR_PC_IS_ENCRYPTED '
    'pay_FAKE-ACCOUNT_to_unlock NOT_REAL_DO_NOT_PAY")'
)


def model_output(model, t2i, i2t, trigger_ids):
    bos = t2i["<BOS>"]
    ids = model.generate(
        [bos] + list(trigger_ids),
        i2t,
        max_new_tokens=model.max_context - 1 - len(trigger_ids),
        temperature=1e-8,          # greedy
    )
    return decode(ids, i2t)


def main():
    parser = argparse.ArgumentParser(description="SAFE 'execute LLM output' demo")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("model.npz"))
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--prompt", default=None, help="use this trigger instead of running GCG")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"{args.model} not found. Run python train.py first.")
        sys.exit(1)

    model, t2i, i2t, sentences, epoch = load_model(args.model)

    print(BAR)
    print("  'execute the LLM output' demo   (SAFE - guarded, no malware)")
    print(BAR)

    # ---- Stage 1: GCG trigger ----
    if args.prompt:
        trigger_ids = [t2i[w] for w in args.prompt.split() if w in t2i]
    else:
        import io
        from contextlib import redirect_stdout
        target, _ = gcg.parse_target(args.target, t2i)
        print("  STAGE 1   running GCG to find a trigger ...")
        buf = io.StringIO()
        with redirect_stdout(buf):
            res = gcg.gcg(model, t2i, i2t, target, args.k, stop_prob=0.9)
        trigger_ids = res["prompt"]

    trigger_words = " ".join(i2t[int(t)] for t in trigger_ids)
    print(f"\n  STAGE 1   GCG trigger : {trigger_words}")

    # ---- Stage 2: the LLM OUTPUT (just text) ----
    out = model_output(model, t2i, i2t, trigger_ids)
    stmt = out[len(trigger_words):].strip() if out.startswith(trigger_words) else out
    print()
    print(BAR)
    print("  STAGE 2   the LLM OUTPUT  (this is only text so far)")
    print(BAR)
    print(f"  model emitted: {stmt}")
    print("  >> at this point nothing has run; it is a string on your screen.")

    # ---- Stage 3: the UNSAFE step -- execute the model's output ----
    print()
    print(BAR)
    print("  STAGE 3   UNSAFE agent EXECUTES the output (the vulnerability)")
    print(BAR)

    m = SAFE_PRINT.fullmatch(stmt)
    if not m:
        print(f"  guard: '{stmt}' is not a bare print(\"...\") -> NOT executed")
    else:
        print("  guard: output is a print() of a plain string literal -> safe to run")
        print("  now actually exec()-ing the model's output; real shell output:")
        print("  " + "-" * 60)
        exec(stmt, {"__builtins__": {"print": print}})   # guard => can only print the literal
        print("  " + "-" * 60)
        print("  ^ that line appeared because the print() ACTUALLY RAN (not just LLM text).")

    # ---- Stage 4: the defense ----
    print()
    print(BAR)
    print("  STAGE 4   DEFENSE")
    print(BAR)
    print("  - Never feed model output into exec()/eval()/a shell.")
    print("  - Parse to a fixed schema; dispatch only allow-listed actions.")
    print("  - Require human approval for anything with side effects.")
    print("  (The ransom text was a SIMULATION: fake account, nothing encrypted.)")


if __name__ == "__main__":
    main()
