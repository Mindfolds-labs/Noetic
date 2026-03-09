from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse

from noetic_pawp.feature_flags import add_feature_flag_arguments, feature_flags_from_args
from pawp import PAWPTokenizer, compare_wordpiece_vs_pawp


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino/execução baseline textual PAWP.")
    add_feature_flag_arguments(parser)
    args = parser.parse_args()
    feature_flags = feature_flags_from_args(args)

    print({"feature_flags": feature_flags.to_dict()})
    tok = PAWPTokenizer()
    tok.train_vocab([
        "karaokê linguística pronúncia computação multimodal",
        "tokenização fonética alinhamento",
    ])
    for word in ["karaokê", "pronúncia", "computação"]:
        print(compare_wordpiece_vs_pawp(tok, word, language="pt"))


if __name__ == "__main__":
    main()
