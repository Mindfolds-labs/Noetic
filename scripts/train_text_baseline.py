from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from pawp import PAWPTokenizer, compare_wordpiece_vs_pawp


def main() -> None:
    tok = PAWPTokenizer()
    tok.train_vocab([
        "karaokê linguística pronúncia computação multimodal",
        "tokenização fonética alinhamento",
    ])
    for word in ["karaokê", "pronúncia", "computação"]:
        print(compare_wordpiece_vs_pawp(tok, word, language="pt"))


if __name__ == "__main__":
    main()
