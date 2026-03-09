from __future__ import annotations

from noetic_pawp import PAWPTokenizer, review_alignment


def main() -> None:
    corpus = [
        "Karaokê e pronúncia em português",
        "Linguística computacional e multimodalidade",
        "Tokenização WordPiece com pista fonética",
    ]

    tokenizer = PAWPTokenizer()
    tokenizer.fit_vocab(corpus, min_freq=1)

    test_words = ["karaokê", "linguística", "pronúncia", "computação", "multimodal"]

    print("\nCOMPARE WORDPIECE VS PAWP:")
    for row in review_alignment(tokenizer, test_words, language="pt"):
        print(row)

    print("\nPontos de melhoria sugeridos:")
    print("1. Trocar pseudo-G2P por G2P real por língua.")
    print("2. Melhorar alinhamento para dígrafos e nasalização.")
    print("3. Separar baseline WordPiece e PAWP em testes automáticos.")
    print("4. Substituir cn mockado por feature real em fase futura.")


if __name__ == "__main__":
    main()
