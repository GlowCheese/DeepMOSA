####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
# --------------------------


import minbpe.regex as module_0


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "<|endoftext|>"
    var_2 = 100257
    var_3 = {var_1: var_2}
    var_4 = var_0.register_special_tokens(var_3)
    var_5 = var_0.special_tokens
    var_6 = bool(var_0.special_tokens == {"<|endoftext|>": 100257})
    assert var_6 is True
    var_7 = var_0.inverse_special_tokens
    var_8 = bool(var_0.inverse_special_tokens == {100257: "<|endoftext|>"})
    assert var_8 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "<|endoftext|>"
    var_2 = "<|pad|>"
    var_3 = "<|unk|>"
    var_4 = 100257
    var_5 = 100258
    var_6 = 100259
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = var_0.register_special_tokens(var_7)
    var_9 = var_0.special_tokens
    var_10 = bool(
        var_0.special_tokens == {"<|endoftext|>": 100257, "<|pad|>": 100258, "<|unk|>": 100259}
    )
    assert var_10 is True
    var_11 = var_0.inverse_special_tokens
    var_12 = bool(
        var_0.inverse_special_tokens
        == {100257: "<|endoftext|>", 100258: "<|pad|>", 100259: "<|unk|>"}
    )
    assert var_12 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "<|endoftext|>"
    var_2 = 100257
    var_3 = {var_1: var_2}
    var_4 = var_0.register_special_tokens(var_3)
    var_5 = "<|pad|>"
    var_6 = 100258
    var_7 = {var_5: var_6}
    var_8 = var_0.register_special_tokens(var_7)
    var_9 = var_0.special_tokens
    var_10 = bool(var_0.special_tokens == {"<|pad|>": 100258})
    assert var_10 is True
    var_11 = var_0.inverse_special_tokens
    var_12 = bool(var_0.inverse_special_tokens == {100258: "<|pad|>"})
    assert var_12 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "<|endoftext|>"
    var_2 = 100257
    var_3 = {var_1: var_2}
    var_4 = var_0.register_special_tokens(var_3)
    var_5 = {}
    var_6 = var_0.register_special_tokens(var_5)
    var_7 = var_0.special_tokens
    var_8 = bool(var_0.special_tokens == {})
    assert var_8 is True
    var_9 = var_0.inverse_special_tokens
    var_10 = bool(var_0.inverse_special_tokens == {})
    assert var_10 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "<|endoftext|>"
    var_2 = "<|pad|>"
    var_3 = 100257
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = var_0.register_special_tokens(var_4)
    var_6 = var_0.special_tokens
    var_7 = bool(var_0.special_tokens == {"<|endoftext|>": 100257, "<|pad|>": 100257})
    assert var_7 is True
    var_8 = var_0.inverse_special_tokens
    var_9 = bool(var_0.inverse_special_tokens == {100257: "<|pad|>"})
    assert var_9 is True


# Parsed testcases at query #2
# --------------------------


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = b"a"
    var_5 = b"b"
    var_6 = b"c"
    var_7 = [var_1, var_2, var_3]
    var_8 = var_0.decode(var_7)
    assert var_8 == "abc"


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = 0
    var_2 = 1
    var_3 = b"a"
    var_4 = b"b"
    var_5 = 100
    var_6 = "<|end|>"
    var_7 = [var_1, var_5, var_2]
    var_8 = var_0.decode(var_7)
    assert var_8 == "a<|end|>b"


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = 0
    var_2 = b"a"
    var_3 = 999
    var_4 = [var_1, var_3]
    var_5 = var_0.decode(var_4)
    var_6 = bool(False)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = []
    var_2 = var_0.decode(var_1)
    assert var_2 == ""


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = 0
    var_2 = 1
    var_3 = b"\xe2\x82\xac"
    var_4 = b"\xf0\x9f\x98\x80"
    var_5 = [var_1, var_2]
    var_6 = var_0.decode(var_5)
    assert var_6 == "€😀"


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = 10
    var_2 = 20
    var_3 = b"hello"
    var_4 = b" "
    var_5 = 30
    var_6 = 40
    var_7 = "<|sep|>"
    var_8 = "<|end|>"
    var_9 = [var_1, var_2, var_5, var_2, var_6]
    var_10 = var_0.decode(var_9)
    assert var_10 == "hello <|sep|> <|end|>"


# Parsed testcases at query #3
# --------------------------


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "none"
    var_6 = var_0.encode(var_1, var_5)
    var_7 = var_0.encode_ordinary(var_1)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "<|endoftext|>"
    var_6 = 256
    var_7 = {var_5: var_6}
    var_8 = var_0.register_special_tokens(var_7)
    var_9 = "hello<|endoftext|>world"
    var_10 = "all"
    var_11 = var_0.encode(var_9, var_10)
    var_12 = "hello"
    var_13 = var_0.encode_ordinary(var_12)
    var_14 = [var_6]
    var_15 = var_13 + var_14
    var_16 = "world"
    var_17 = var_0.encode_ordinary(var_16)
    var_18 = var_15 + var_17
    var_19 = bool(var_11 == var_18)
    assert var_19 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "<|endoftext|>"
    var_6 = 256
    var_7 = {var_5: var_6}
    var_8 = var_0.register_special_tokens(var_7)
    var_9 = "none_raise"
    var_10 = var_0.encode(var_1, var_9)
    var_11 = var_0.encode_ordinary(var_1)
    var_12 = bool(var_10 == var_11)
    assert var_12 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "<|endoftext|>"
    var_6 = "<|pad|>"
    var_7 = 256
    var_8 = 257
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_0.register_special_tokens(var_9)
    var_11 = "hello<|endoftext|>world"
    var_12 = {var_5}
    var_13 = var_0.encode(var_11, var_12)
    var_14 = "hello"
    var_15 = var_0.encode_ordinary(var_14)
    var_16 = [var_7]
    var_17 = var_15 + var_16
    var_18 = "world"
    var_19 = var_0.encode_ordinary(var_18)
    var_20 = var_17 + var_19
    var_21 = bool(var_13 == var_20)
    assert var_21 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "hello world"
    var_6 = "invalid"
    var_7 = var_0.encode(var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "<|endoftext|>"
    var_6 = 256
    var_7 = {var_5: var_6}
    var_8 = var_0.register_special_tokens(var_7)
    var_9 = "hello<|endoftext|>world"
    var_10 = "none_raise"
    var_11 = var_0.encode(var_9, var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = ""
    var_6 = "none"
    var_7 = var_0.encode(var_5, var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "<|endoftext|>"
    var_6 = 256
    var_7 = {var_5: var_6}
    var_8 = var_0.register_special_tokens(var_7)
    var_9 = "all"
    var_10 = var_0.encode(var_5, var_9)
    var_11 = bool(var_10 == [256])
    assert var_11 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "<|endoftext|>"
    var_6 = 256
    var_7 = {var_5: var_6}
    var_8 = var_0.register_special_tokens(var_7)
    var_9 = "<|endoftext|>hello"
    var_10 = "all"
    var_11 = var_0.encode(var_9, var_10)
    var_12 = [var_6]
    var_13 = "hello"
    var_14 = var_0.encode_ordinary(var_13)
    var_15 = var_12 + var_14
    var_16 = bool(var_11 == var_15)
    assert var_16 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "<|endoftext|>"
    var_6 = 256
    var_7 = {var_5: var_6}
    var_8 = var_0.register_special_tokens(var_7)
    var_9 = "hello<|endoftext|>"
    var_10 = "all"
    var_11 = var_0.encode(var_9, var_10)
    var_12 = "hello"
    var_13 = var_0.encode_ordinary(var_12)
    var_14 = [var_6]
    var_15 = var_13 + var_14
    var_16 = bool(var_11 == var_15)
    assert var_16 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "<|endoftext|>"
    var_6 = "<|pad|>"
    var_7 = 256
    var_8 = 257
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_0.register_special_tokens(var_9)
    var_11 = "<|endoftext|>hello<|pad|>world"
    var_12 = "all"
    var_13 = var_0.encode(var_11, var_12)
    var_14 = [var_7]
    var_15 = "hello"
    var_16 = var_0.encode_ordinary(var_15)
    var_17 = var_14 + var_16
    var_18 = [var_8]
    var_19 = var_17 + var_18
    var_20 = "world"
    var_21 = var_0.encode_ordinary(var_20)
    var_22 = var_19 + var_21
    var_23 = bool(var_13 == var_22)
    assert var_23 is True


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = "<|endoftext|>"
    var_6 = "<|pad|>"
    var_7 = 256
    var_8 = 257
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_0.register_special_tokens(var_9)
    var_11 = "hello<|endoftext|>world"
    var_12 = {var_6}
    var_13 = var_0.encode(var_11, var_12)
    var_14 = var_0.encode_ordinary(var_11)
    var_15 = bool(var_13 == var_14)
    assert var_15 is True


# Parsed testcases at query #4
# --------------------------


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world"
    var_2 = 260
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = var_0.merges
    var_6 = var_0.vocab
    var_7 = len(var_5)
    assert var_7 == 4
    var_8 = len(var_6)
    assert var_8 == 260
    var_9 = 2
    var_10 = bytes(var_3)


def test_case_0():
    var_0 = "\\w+|\\s+"
    var_1 = module_0.RegexTokenizer(var_0)
    var_2 = "hello world"
    var_3 = 258
    var_4 = False
    var_5 = var_1.train(var_2, var_3, var_4)
    var_6 = var_1.merges
    var_7 = var_1.vocab
    var_8 = len(var_6)
    assert var_8 == 2
    var_9 = len(var_7)
    assert var_9 == 258


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = ""
    var_2 = 256
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = var_0.merges
    var_6 = var_0.vocab
    var_7 = len(var_5)
    assert var_7 == 0
    var_8 = len(var_6)
    assert var_8 == 256


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "a"
    var_2 = 257
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = var_0.merges
    var_6 = var_0.vocab
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = len(var_6)
    assert var_8 == 257


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "aa"
    var_2 = 257
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = var_0.merges
    var_6 = var_0.vocab
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = 97
    var_9 = 97
    var_10 = (var_8, var_9)
    var_11 = bool((97, 97) in var_5)
    assert var_11 is True
    var_12 = 97
    var_13 = var_5[var_12, var_12]
    assert var_13 == 256
    var_14 = var_6[var_13]
    assert var_14 == b"aa"


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "abab"
    var_2 = 258
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = var_0.merges
    var_6 = 1
    var_7 = "utf-8"
    var_8 = lambda x: x[var_3]


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello"
    var_2 = 256
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = var_0.merges
    var_6 = var_0.vocab
    var_7 = len(var_5)
    assert var_7 == 0
    var_8 = len(var_6)
    assert var_8 == 256


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "hello world, this is a test."
    var_2 = 300
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = var_0.merges
    var_6 = var_0.vocab
    var_7 = len(var_5)
    assert var_7 == 44
    var_8 = len(var_6)
    assert var_8 == 300


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "aa bb aa bb"
    var_2 = 258
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = var_0.encode_ordinary(var_1)
    var_6 = len(var_5)
    var_7 = "utf-8"


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = "test"
    var_2 = 258
    var_3 = False
    var_4 = var_0.train(var_1, var_2, var_3)
    var_5 = var_0.merges
    var_6 = len(var_5)
    assert var_6 == 2


# Parsed testcases at query #5
# --------------------------


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = 0
    var_2 = 1
    var_3 = b"a"
    var_4 = b"b"
    var_5 = 100
    var_6 = "<|endoftext|>"
    var_7 = [var_1, var_5, var_2]
    var_8 = var_0.decode(var_7)
    assert var_8 == "a<|endoftext|>b"


# Parsed testcases at query #6
# --------------------------


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = 0
    var_2 = 1
    var_3 = b"a"
    var_4 = b"b"
    var_5 = 100
    var_6 = "<|endoftext|>"
    var_7 = [var_1, var_5, var_2]
    var_8 = var_0.decode(var_7)
    assert var_8 == "a<|endoftext|>b"


# Parsed testcases at query #7
# --------------------------


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = 0
    var_2 = 1
    var_3 = b"a"
    var_4 = b"b"
    var_5 = "<|endoftext|>"
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = var_0.register_special_tokens(var_7)
    var_9 = [var_1, var_6, var_2]
    var_10 = var_0.decode(var_9)
    assert var_10 == "a<|endoftext|>b"


# Parsed testcases at query #8
# --------------------------


def test_case_0():
    var_0 = module_0.RegexTokenizer()
    var_1 = 0
    var_2 = 1
    var_3 = b"a"
    var_4 = b"b"
    var_5 = 100
    var_6 = "<special>"
    var_7 = [var_1, var_5, var_2]
    var_8 = var_0.decode(var_7)
    assert var_8 == "a<special>b"
