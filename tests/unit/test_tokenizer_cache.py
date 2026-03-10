from noetic_pawp.tokenizer import PAWPTokenizer


def test_wordpiece_tokenize_cache_hits():
    tok = PAWPTokenizer()
    tok.fit_vocab(["linguística computação fonética"], min_freq=1)
    tok.clear_caches()

    tok.encode("linguística", language="pt")
    tok.encode("linguística", language="pt")

    # segunda chamada deve ter batido no cache (sem erro e resultado idêntico)
    r1 = tok.encode("computação", language="pt")
    r2 = tok.encode("computação", language="pt")
    assert [t.wp_piece for t in r1] == [t.wp_piece for t in r2]


def test_clear_caches_does_not_break_encode():
    tok = PAWPTokenizer()
    tok.fit_vocab(["pronúncia"], min_freq=1)
    tok.encode("pronúncia", language="pt")
    tok.clear_caches()
    result = tok.encode("pronúncia", language="pt")
    assert result
