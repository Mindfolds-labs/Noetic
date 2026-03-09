from noetic_pawp.retrieval import RetrievalSample, retrieval_at_k


def test_retrieval_at_k_for_concepts() -> None:
    samples = [
        RetrievalSample("hello", "en", "concept.greeting.hello"),
        RetrievalSample("olá", "pt", "concept.greeting.hello"),
        RetrievalSample("cafe\u0301", "pt", "concept.food.coffee"),
    ]

    r_at_1 = retrieval_at_k(samples, k=1)
    r_at_3 = retrieval_at_k(samples, k=3)

    assert r_at_1 >= 0.66
    assert r_at_3 >= r_at_1
