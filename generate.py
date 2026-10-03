"""
TinyGPT inference

Loads the model.npz saved by train.py and,
when you type one or two words, generates the rest of the sentence.

No training happens here. The architecture and vocabulary are restored from model.npz.

Usage:

    python generate.py
    python generate.py --temperature 1.0 --samples 5

Training: train.py
"""

import argparse
import sys
from pathlib import Path

from tinygpt import (
    decode,
    load_model
)


# ====================================================================
# Interactive generation
#
# When the user types one or two words,
#
#   <BOS> + input words
#
# is used as the prompt and the sentence is continued.
# ====================================================================

def show_corpus(
    sentences,
    token_to_id
):

    print()
    print("===== Training sentences =====")

    for i, sentence in enumerate(sentences, 1):

        print(
            f"{i:3d}. {sentence}"
        )

    # ---------------------------------------------------------------
    # Sentence-starting words
    #
    # The input is appended after <BOS>, so words that start
    # a sentence give the most natural results.
    # ---------------------------------------------------------------

    first_words = sorted(
        {
            sentence.lower().split()[0]
            for sentence in sentences
        }
    )

    words = sorted(
        token
        for token in token_to_id
        if not token.startswith("<")
    )

    print()
    print(
        "Sentence-starting words:",
        ", ".join(first_words)
    )

    print(
        f"All words ({len(words)}):",
        ", ".join(words)
    )


def interactive(
    model,
    token_to_id,
    id_to_token,
    sentences,
    n_samples=3,
    temperature=0.7
):

    known_sentences = {
        sentence
        for sentence in sentences
    }

    print()
    print(
        "Type one or two words to generate a sentence. "
        "(e.g. the cat / my / we)  Quit: q or an empty line"
    )

    while True:

        try:

            # Strip the BOM that piped input may carry
            text = input("\n> ").replace("﻿", "").strip().lower()

        except (EOFError, KeyboardInterrupt):

            print()
            break

        if text in ("", "q", "quit", "exit"):
            break

        words = text.split()

        # ------------------------------------------------------------
        # Unknown words become <UNK>, but <UNK> never appears in training,
        # so it cannot produce a meaningful sentence. Report them instead.
        # ------------------------------------------------------------

        unknown = [
            w
            for w in words
            if w not in token_to_id
            or w.startswith("<")
        ]

        if unknown:

            print(
                "  Not in the training sentences:",
                ", ".join(unknown)
            )

            continue

        prompt_ids = (
            [token_to_id["<BOS>"]]
            +
            [token_to_id[w] for w in words]
        )

        outputs = []

        for _ in range(n_samples):

            generated_ids = model.generate(
                prompt_ids,
                id_to_token,
                max_new_tokens=model.max_context,
                temperature=temperature
            )

            # Only the model's CONTINUATION, not the words we fed in.
            # generated_ids = prompt_ids + newly generated ids, so slice
            # off the prompt to keep the input and output separate.
            output = decode(
                generated_ids[len(prompt_ids):],
                id_to_token
            )

            if output not in outputs:
                outputs.append(output)

        # ------------------------------------------------------------
        # Show the INPUT (what you typed) and the OUTPUT (what the
        # model generated) on separate lines, so they are not confused.
        # ------------------------------------------------------------

        print(f"  input  (you typed) : {' '.join(words)}")

        for output in outputs:

            full = (
                (" ".join(words) + " " + output).strip().lower()
            )

            tag = (
                "training sentence"
                if full in known_sentences
                else "new combination"
            )

            print(
                f"  output (model)     : {output}   [{tag}]"
            )


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TinyGPT inference"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("model.npz"),
        help="model file to load (default: model.npz)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="lower = closer to the training sentences (near-greedy), higher = more varied"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="number of outputs to generate per input (default 1: one answer)"
    )

    args = parser.parse_args()

    if not args.model.exists():

        print(
            f"{args.model} not found. "
            "Run python train.py first."
        )

        sys.exit(1)

    (
        model,
        token_to_id,
        id_to_token,
        sentences,
        epoch
    ) = load_model(
        args.model
    )

    parameter_count = sum(
        parameter.size
        for parameter in model.p.values()
    )

    print(
        f"model: {args.model.name} | trained {epoch} epochs | "
        f"vocab {len(token_to_id)} | parameters {parameter_count}"
    )

    show_corpus(
        sentences,
        token_to_id
    )

    interactive(
        model,
        token_to_id,
        id_to_token,
        sentences,
        n_samples=args.samples,
        temperature=args.temperature
    )
