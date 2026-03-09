from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse

from noetic_pawp.feature_flags import add_feature_flag_arguments, feature_flags_from_args
from pawp import PAWPTokenizer, review_alignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Execução demo do PAWP.")
    add_feature_flag_arguments(parser)
    args = parser.parse_args()
    feature_flags = feature_flags_from_args(args)

    corpus = [
        "karaokê linguística pronúncia computação multimodal",
        "wordpiece tokenizer fonética alinhamento",
    ]
    tokenizer = PAWPTokenizer()
    tokenizer.train_vocab(corpus)

    print({"feature_flags": feature_flags.to_dict()})
    words = ["karaokê", "linguística", "pronúncia", "computação", "multimodal"]
    for row in review_alignment(tokenizer, words, language="pt"):
        print(row)


if __name__ == "__main__":
    main()
