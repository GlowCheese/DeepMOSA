####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
# --------------------------


def test_register_special_tokens_single_token():
    tokenizer = RegexTokenizer()
    special_tokens = {"<|endoftext|>": 100257}
    tokenizer.register_special_tokens(special_tokens)
    assert tokenizer.special_tokens == {"<|endoftext|>": 100257}
    assert tokenizer.inverse_special_tokens == {100257: "<|endoftext|>"}


def test_register_special_tokens_multiple_tokens():
    tokenizer = RegexTokenizer()
    special_tokens = {"<|endoftext|>": 100257, "<|pad|>": 100258, "<|unk|>": 100259}
    tokenizer.register_special_tokens(special_tokens)
    assert tokenizer.special_tokens == {
        "<|endoftext|>": 100257,
        "<|pad|>": 100258,
        "<|unk|>": 100259,
    }
    assert tokenizer.inverse_special_tokens == {
        100257: "<|endoftext|>",
        100258: "<|pad|>",
        100259: "<|unk|>",
    }


def test_register_special_tokens_overwrites_previous():
    tokenizer = RegexTokenizer()
    first_tokens = {"<|endoftext|>": 100257}
    tokenizer.register_special_tokens(first_tokens)
    second_tokens = {"<|pad|>": 100258}
    tokenizer.register_special_tokens(second_tokens)
    assert tokenizer.special_tokens == {"<|pad|>": 100258}
    assert tokenizer.inverse_special_tokens == {100258: "<|pad|>"}


def test_register_special_tokens_empty_dict():
    tokenizer = RegexTokenizer()
    initial_tokens = {"<|endoftext|>": 100257}
    tokenizer.register_special_tokens(initial_tokens)
    tokenizer.register_special_tokens({})
    assert tokenizer.special_tokens == {}
    assert tokenizer.inverse_special_tokens == {}


def test_register_special_tokens_duplicate_values():
    tokenizer = RegexTokenizer()
    special_tokens = {"<|endoftext|>": 100257, "<|pad|>": 100257}
    tokenizer.register_special_tokens(special_tokens)
    assert tokenizer.special_tokens == {"<|endoftext|>": 100257, "<|pad|>": 100257}
    assert tokenizer.inverse_special_tokens == {100257: "<|pad|>"}


# LLM-generated content at query #2
# --------------------------


def test_decode_with_valid_ids():
    tokenizer = RegexTokenizer()
    tokenizer.vocab = {0: b"a", 1: b"b", 2: b"c"}
    tokenizer.inverse_special_tokens = {}
    ids = [0, 1, 2]
    result = tokenizer.decode(ids)
    assert result == "abc"


def test_decode_with_special_token():
    tokenizer = RegexTokenizer()
    tokenizer.vocab = {0: b"a", 1: b"b"}
    tokenizer.inverse_special_tokens = {100: "<|end|>"}
    ids = [0, 100, 1]
    result = tokenizer.decode(ids)
    assert result == "a<|end|>b"


def test_decode_with_invalid_token_id():
    tokenizer = RegexTokenizer()
    tokenizer.vocab = {0: b"a"}
    tokenizer.inverse_special_tokens = {}
    ids = [0, 999]
    try:
        tokenizer.decode(ids)
        assert False
    except ValueError as e:
        assert str(e) == "invalid token id: 999"


def test_decode_empty_ids():
    tokenizer = RegexTokenizer()
    tokenizer.vocab = {}
    tokenizer.inverse_special_tokens = {}
    ids = []
    result = tokenizer.decode(ids)
    assert result == ""


def test_decode_utf8_multibyte():
    tokenizer = RegexTokenizer()
    tokenizer.vocab = {0: b"\xe2\x82\xac", 1: b"\xf0\x9f\x98\x80"}
    tokenizer.inverse_special_tokens = {}
    ids = [0, 1]
    result = tokenizer.decode(ids)
    assert result == "€😀"


def test_decode_mixed_vocab_and_special():
    tokenizer = RegexTokenizer()
    tokenizer.vocab = {10: b"hello", 20: b" "}
    tokenizer.inverse_special_tokens = {30: "<|sep|>", 40: "<|end|>"}
    ids = [10, 20, 30, 20, 40]
    result = tokenizer.decode(ids)
    assert result == "hello <|sep|> <|end|>"


# LLM-generated content at query #3
# --------------------------


def test_encode_without_special_tokens():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    result = tokenizer.encode("hello world", allowed_special="none")
    expected = tokenizer.encode_ordinary("hello world")
    assert result == expected


def test_encode_with_special_tokens_all():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    tokenizer.register_special_tokens({"<|endoftext|>": 256})
    result = tokenizer.encode("hello<|endoftext|>world", allowed_special="all")
    expected = tokenizer.encode_ordinary("hello") + [256] + tokenizer.encode_ordinary("world")
    assert result == expected


def test_encode_with_special_tokens_none_raise_no_special():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    tokenizer.register_special_tokens({"<|endoftext|>": 256})
    result = tokenizer.encode("hello world", allowed_special="none_raise")
    expected = tokenizer.encode_ordinary("hello world")
    assert result == expected


def test_encode_with_special_tokens_custom_set():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    tokenizer.register_special_tokens({"<|endoftext|>": 256, "<|pad|>": 257})
    result = tokenizer.encode("hello<|endoftext|>world", allowed_special={"<|endoftext|>"})
    expected = tokenizer.encode_ordinary("hello") + [256] + tokenizer.encode_ordinary("world")
    assert result == expected


def test_encode_raises_on_invalid_allowed_special():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    try:
        tokenizer.encode("hello world", allowed_special="invalid")
        assert False
    except ValueError:
        assert True


def test_encode_raises_on_none_raise_with_special():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    tokenizer.register_special_tokens({"<|endoftext|>": 256})
    try:
        tokenizer.encode("hello<|endoftext|>world", allowed_special="none_raise")
        assert False
    except AssertionError:
        assert True


def test_encode_empty_text():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    result = tokenizer.encode("", allowed_special="none")
    assert result == []


def test_encode_only_special_token():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    tokenizer.register_special_tokens({"<|endoftext|>": 256})
    result = tokenizer.encode("<|endoftext|>", allowed_special="all")
    assert result == [256]


def test_encode_special_token_at_start():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    tokenizer.register_special_tokens({"<|endoftext|>": 256})
    result = tokenizer.encode("<|endoftext|>hello", allowed_special="all")
    expected = [256] + tokenizer.encode_ordinary("hello")
    assert result == expected


def test_encode_special_token_at_end():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    tokenizer.register_special_tokens({"<|endoftext|>": 256})
    result = tokenizer.encode("hello<|endoftext|>", allowed_special="all")
    expected = tokenizer.encode_ordinary("hello") + [256]
    assert result == expected


def test_encode_multiple_special_tokens():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    tokenizer.register_special_tokens({"<|endoftext|>": 256, "<|pad|>": 257})
    result = tokenizer.encode("<|endoftext|>hello<|pad|>world", allowed_special="all")
    expected = (
        [256] + tokenizer.encode_ordinary("hello") + [257] + tokenizer.encode_ordinary("world")
    )
    assert result == expected


def test_encode_special_tokens_not_in_allowed_set_ignored():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello world", 300, verbose=False)
    tokenizer.register_special_tokens({"<|endoftext|>": 256, "<|pad|>": 257})
    result = tokenizer.encode("hello<|endoftext|>world", allowed_special={"<|pad|>"})
    expected = tokenizer.encode_ordinary("hello<|endoftext|>world")
    assert result == expected


# LLM-generated content at query #4
# --------------------------


def test_train_basic():
    tokenizer = RegexTokenizer()
    text = "hello world"
    vocab_size = 260
    tokenizer.train(text, vocab_size, verbose=False)
    merges = tokenizer.merges
    vocab = tokenizer.vocab
    assert len(merges) == 4
    assert len(vocab) == 260
    assert all(isinstance(k, tuple) and len(k) == 2 for k in merges.keys())
    assert all(isinstance(v, int) for v in merges.values())
    assert all(isinstance(k, int) for k in vocab.keys())
    assert all(isinstance(v, bytes) for v in vocab.values())
    for idx in range(256):
        assert vocab[idx] == bytes([idx])
    for i, (pair, new_idx) in enumerate(merges.items()):
        assert new_idx == 256 + i
        assert vocab[new_idx] == vocab[pair[0]] + vocab[pair[1]]


def test_train_with_special_pattern():
    tokenizer = RegexTokenizer(pattern=r"\w+|\s+")
    text = "hello world"
    vocab_size = 258
    tokenizer.train(text, vocab_size, verbose=False)
    merges = tokenizer.merges
    vocab = tokenizer.vocab
    assert len(merges) == 2
    assert len(vocab) == 258


def test_train_empty_text():
    tokenizer = RegexTokenizer()
    text = ""
    vocab_size = 256
    tokenizer.train(text, vocab_size, verbose=False)
    merges = tokenizer.merges
    vocab = tokenizer.vocab
    assert len(merges) == 0
    assert len(vocab) == 256


def test_train_single_character():
    tokenizer = RegexTokenizer()
    text = "a"
    vocab_size = 257
    tokenizer.train(text, vocab_size, verbose=False)
    merges = tokenizer.merges
    vocab = tokenizer.vocab
    assert len(merges) == 1
    assert len(vocab) == 257


def test_train_repeated_characters():
    tokenizer = RegexTokenizer()
    text = "aa"
    vocab_size = 257
    tokenizer.train(text, vocab_size, verbose=False)
    merges = tokenizer.merges
    vocab = tokenizer.vocab
    assert len(merges) == 1
    assert (97, 97) in merges
    new_idx = merges[(97, 97)]
    assert new_idx == 256
    assert vocab[new_idx] == b"aa"


def test_train_verify_merge_order():
    tokenizer = RegexTokenizer()
    text = "abab"
    vocab_size = 258
    tokenizer.train(text, vocab_size, verbose=False)
    merges = tokenizer.merges
    first_pair = max(
        [(count, pair) for pair, count in get_stats(list(text.encode("utf-8")))], key=lambda x: x[0]
    )[1]
    assert first_pair in merges
    assert merges[first_pair] == 256


def test_train_vocab_size_256():
    tokenizer = RegexTokenizer()
    text = "hello"
    vocab_size = 256
    tokenizer.train(text, vocab_size, verbose=False)
    merges = tokenizer.merges
    vocab = tokenizer.vocab
    assert len(merges) == 0
    assert len(vocab) == 256


def test_train_vocab_size_large():
    tokenizer = RegexTokenizer()
    text = "hello world, this is a test."
    vocab_size = 300
    tokenizer.train(text, vocab_size, verbose=False)
    merges = tokenizer.merges
    vocab = tokenizer.vocab
    assert len(merges) == 44
    assert len(vocab) == 300


def test_train_ids_after_merging():
    tokenizer = RegexTokenizer()
    text = "aa bb aa bb"
    vocab_size = 258
    tokenizer.train(text, vocab_size, verbose=False)
    ids = tokenizer.encode_ordinary(text)
    assert len(ids) < len(list(text.encode("utf-8")))


def test_train_verbose_false():
    tokenizer = RegexTokenizer()
    text = "test"
    vocab_size = 258
    tokenizer.train(text, vocab_size, verbose=False)
    merges = tokenizer.merges
    assert len(merges) == 2


# LLM-generated content at query #5
# --------------------------


def test_decode_with_special_token():
    tokenizer = RegexTokenizer()
    tokenizer.vocab = {0: b"a", 1: b"b"}
    tokenizer.inverse_special_tokens = {100: "<|endoftext|>"}
    ids = [0, 100, 1]
    result = tokenizer.decode(ids)
    assert result == "a<|endoftext|>b"


# LLM-generated content at query #6
# --------------------------


def test_decode_with_special_token():
    tokenizer = RegexTokenizer()
    tokenizer.vocab = {0: b"a", 1: b"b"}
    tokenizer.inverse_special_tokens = {100: "<|endoftext|>"}
    ids = [0, 100, 1]
    result = tokenizer.decode(ids)
    assert result == "a<|endoftext|>b"


# LLM-generated content at query #7
# --------------------------


def test_decode_with_special_token():
    tokenizer = RegexTokenizer()
    tokenizer.vocab = {0: b"a", 1: b"b"}
    tokenizer.register_special_tokens({"<|endoftext|>": 2})
    result = tokenizer.decode([0, 2, 1])
    assert result == "a<|endoftext|>b"


# LLM-generated content at query #8
# --------------------------


def test_decode_with_special_token():
    tokenizer = RegexTokenizer()
    tokenizer.vocab = {0: b"a", 1: b"b"}
    tokenizer.inverse_special_tokens = {100: "<special>"}
    result = tokenizer.decode([0, 100, 1])
    assert result == "a<special>b"
