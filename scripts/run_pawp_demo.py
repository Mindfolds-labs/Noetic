from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from pawp import PAWPTokenizer, review_alignment


def main() -> None:
    corpus = [
        "karaokê linguística pronúncia computação multimodal",
        "wordpiece tokenizer fonética alinhamento",
    ]
    tokenizer = PAWPTokenizer()
    tokenizer.train_vocab(corpus)

    words = ["karaokê", "linguística", "pronúncia", "computação", "multimodal"]
    for row in review_alignment(tokenizer, words, language="pt"):
        print(row)


if __name__ == "__main__":
    main()
