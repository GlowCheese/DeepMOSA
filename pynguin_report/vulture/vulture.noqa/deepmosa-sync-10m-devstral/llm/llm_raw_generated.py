####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_noqa_empty_code():
    assert parse_noqa([]) == defaultdict(set)

def test_parse_noqa_no_noqa_lines():
    code = ["print('hello')", "x = 1"]
    assert parse_noqa(code) == defaultdict(set)

def test_parse_noqa_single_all():
    code = ["x = 1  # noqa"]
    expected = defaultdict(set, {"all": {1}})
    assert parse_noqa(code) == expected

def test_parse_noqa_single_specific_code():
    code = ["x = 1  # noqa: E123"]
    expected = defaultdict(set, {"E123": {1}})
    assert parse_noqa(code) == expected

def test_parse_noqa_multiple_codes():
    code = ["x = 1  # noqa: E123, F456"]
    expected = defaultdict(set, {"E123": {1}, "F456": {1}})
    assert parse_noqa(code) == expected

def test_parse_noqa_multiple_lines():
    code = ["x = 1  # noqa: E123", "y = 2  # noqa: F456"]
    expected = defaultdict(set, {"E123": {1}, "F456": {2}})
    assert parse_noqa(code) == expected

def test_parse_noqa_mixed_all_and_specific():
    code = ["x = 1  # noqa", "y = 2  # noqa: E123"]
    expected = defaultdict(set, {"all": {1}, "E123": {2}})
    assert parse_noqa(code) == expected

def test_parse_noqa_code_mapping():
    code = ["x = 1  # noqa: W123"]
    expected = defaultdict(set, {"mapped_W123": {1}})
    with patch("__main__.NOQA_CODE_MAP", {"W123": "mapped_W123"}):
        assert parse_noqa(code) == expected


# LLM-generated content at query #2
#--------------------------

```python
def test_ignore_line_with_error_code_in_noqa_lines():
    noqa_lines = {"E123": {5, 10}, "all": {15}}
    assert ignore_line(noqa_lines, 5, "E123") == True

def test_ignore_line_with_lineno_in_all_noqa_lines():
    noqa_lines = {"E123": {5, 10}, "all": {15}}
    assert ignore_line(noqa_lines, 15, "E123") == True

def test_ignore_line_with_lineno_not_in_noqa_lines():
    noqa_lines = {"E123": {5, 10}, "all": {15}}
    assert ignore_line(noqa_lines, 20, "E123") == False

def test_ignore_line_with_empty_noqa_lines():
    noqa_lines = {"E123": set(), "all": set()}
    assert ignore_line(noqa_lines, 5, "E123") == False


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ignore_line_when_lineno_in_noqa_lines_for_error_code():
    noqa_lines = {"E123": {5, 10}, "all": {1, 2}}
    assert ignore_line(noqa_lines, 5, "E123") is True

def test_ignore_line_when_lineno_in_noqa_lines_for_all():
    noqa_lines = {"E123": {5, 10}, "all": {1, 2}}
    assert ignore_line(noqa_lines, 1, "E456") is True

def test_ignore_line_when_lineno_not_in_noqa_lines():
    noqa_lines = {"E123": {5, 10}, "all": {1, 2}}
    assert ignore_line(noqa_lines, 3, "E123") is False


# LLM-generated content at query #2
#--------------------------

```python
def test_parse_noqa_empty_code():
    assert parse_noqa([]) == defaultdict(set)

def test_parse_noqa_no_noqa_lines():
    code = ["print('hello')", "x = 1 + 2"]
    assert parse_noqa(code) == defaultdict(set)

def test_parse_noqa_single_all():
    code = ["x = 1  # noqa"]
    result = parse_noqa(code)
    assert result["all"] == {1}

def test_parse_noqa_single_specific_code():
    code = ["x = 1  # noqa: E123"]
    result = parse_noqa(code)
    assert result["E123"] == {1}

def test_parse_noqa_multiple_codes():
    code = ["x = 1  # noqa: E123, F456"]
    result = parse_noqa(code)
    assert result["E123"] == {1}
    assert result["F456"] == {1}

def test_parse_noqa_multiple_lines():
    code = [
        "x = 1  # noqa: E123",
        "y = 2  # noqa: F456",
        "z = 3  # noqa"
    ]
    result = parse_noqa(code)
    assert result["E123"] == {1}
    assert result["F456"] == {2}
    assert result["all"] == {3}

def test_parse_noqa_code_mapping():
    code = ["x = 1  # noqa: E123"]
    result = parse_noqa(code)
    assert result[NOQA_CODE_MAP.get("E123", "E123")] == {1}


