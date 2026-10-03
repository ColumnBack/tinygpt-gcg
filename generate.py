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

        results = []

        for _ in range(n_samples):

            generated_ids = model.generate(
                prompt_ids,
                id_to_token,
                max_new_tokens=model.max_context,
                temperature=temperature
            )

            sentence = decode(
                generated_ids,
                id_to_token
            )

            if sentence not in results:
                results.append(sentence)

        for sentence in results:

            tag = (
                "training sentence"
                if sentence in known_sentences
                else "new combination"
            )

            print(
                f"  - {sentence}   [{tag}]"
            )

        # ------------------------------------------------------------
        # SAFE demo hook: if the model emitted the inert, code-shaped
        # SIMULATED_RANSOMWARE payload, show the MOCK (clearly fake)
        # ransom screen. This is print() only -- no exec/eval of the
        # output, no encryption, no file/network access. It just
        # dramatizes "a trigger made the model emit a dangerous-looking
        # payload." See attack_chain_demo.py.
        # ------------------------------------------------------------

        if any("SIMULATED_RANSOMWARE" in s for s in results):

            from attack_chain_demo import print_mock_ransom_banner

            print_mock_ransom_banner()


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
        default=0.7,
        help="lower = closer to the training sentences, higher = more varied"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="number of sentences to generate per input"
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
