####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_parse_noqa_no_match():
    code = ["x = 1", "y = 2"]
    result = parse_noqa(code)
    assert result == {}

def test_parse_noqa_single_error_code():
    code = ["x = 1  # noqa: E301"]
    result = parse_noqa(code)
    assert result == {"E301": {1}}

def test_parse_noqa_multiple_error_codes():
    code = ["x = 1  # noqa: E301, E302"]
    result = parse_noqa(code)
    assert result == {"E301": {1}, "E302": {1}}

def test_parse_noqa_all_codes():
    code = ["x = 1  # noqa"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

def test_parse_noqa_multiple_lines():
    code = ["x = 1  # noqa: E301", "y = 2  # noqa: E302"]
    result = parse_noqa(code)
    assert result == {"E301": {1}, "E302": {2}}

def test_parse_noqa_same_code_multiple_lines():
    code = ["x = 1  # noqa: E301", "y = 2  # noqa: E301"]
    result = parse_noqa(code)
    assert result == {"E301": {1, 2}}

def test_parse_noqa_code_mapping():
    code = ["x = 1  # noqa: F401"]
    result = parse_noqa(code)
    assert result == {"F401": {1}}

def test_parse_noqa_no_comment():
    code = ["x = 1"]
    result = parse_noqa(code)
    assert result == {}

def test_parse_noqa_empty_code():
    code = []
    result = parse_noqa(code)
    assert result == {}

def test_parse_noqa_leading_trailing_spaces():
    code = ["  x = 1  # noqa: E301  "]
    result = parse_noqa(code)
    assert result == {"E301": {1}}
```


# LLM-generated content at query #2
#--------------------------

def test_ignore_line_with_matching_error_code():
    noqa_lines = {"E501": {1, 2, 3}, "all": set()}
    assert ignore_line(noqa_lines, 1, "E501") == True

def test_ignore_line_with_wildcard_all():
    noqa_lines = {"E501": set(), "all": {5, 6}}
    assert ignore_line(noqa_lines, 6, "W292") == True

def test_ignore_line_not_annotated():
    noqa_lines = {"E501": {1, 2}, "all": set()}
    assert ignore_line(noqa_lines, 3, "E501") == False

def test_ignore_line_missing_from_both():
    noqa_lines = {"E501": {10}, "all": {20}}
    assert ignore_line(noqa_lines, 15, "F401") == False

def test_ignore_line_empty_sets():
    noqa_lines = {"E501": set(), "all": set()}
    assert ignore_line(noqa_lines, 1, "E501") == False


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_ignore_line_with_exact_error_code():
    noqa_lines = {"E123": {1, 2, 3}, "all": set()}
    assert ignore_line(noqa_lines, 1, "E123") == True

def test_ignore_line_with_all_code():
    noqa_lines = {"E123": set(), "all": {1, 2, 3}}
    assert ignore_line(noqa_lines, 2, "E456") == True

def test_ignore_line_not_in_noqa_lines():
    noqa_lines = {"E123": {1}, "all": set()}
    assert ignore_line(noqa_lines, 2, "E123") == False

def test_ignore_line_not_in_any():
    noqa_lines = {"E123": set(), "all": set()}
    assert ignore_line(noqa_lines, 3, "E789") == False

def test_ignore_line_with_multiple_lines():
    noqa_lines = {"E111": {10, 20}, "all": {30}}
    assert ignore_line(noqa_lines, 10, "E111") == True

def test_ignore_line_with_all_priority():
    noqa_lines = {"E123": {1}, "all": {1}}
    assert ignore_line(noqa_lines, 1, "E123") == True

def test_ignore_line_empty_noqa_lines():
    noqa_lines = {"E123": set(), "all": set()}
    assert ignore_line(noqa_lines, 5, "E123") == False
```


# LLM-generated content at query #2
#--------------------------

```
def test_parse_noqa_returns_empty_dict_when_no_noqa_found():
    code = "x = 1\ny = 2\n"
    result = parse_noqa(code)
    assert result == {}

def test_parse_noqa_adds_line_to_all_category_when_no_specific_code():
    code = "x = 1  # noqa\ny = 2\n"
    result = parse_noqa(code)
    assert result == {"all": {1}}

def test_parse_noqa_parses_single_error_code():
    code = "x = 1  # noqa: E501\ny = 2\n"
    result = parse_noqa(code)
    assert result == {"E501": {1}}

def test_parse_noqa_parses_multiple_error_codes():
    code = "x = 1  # noqa: E501, W503\ny = 2\n"
    result = parse_noqa(code)
    assert result == {"E501": {1}, "W503": {1}}

def test_parse_noqa_handles_multiple_lines_with_noqa():
    code = "x = 1  # noqa: E501\ny = 2  # noqa: W503\n"
    result = parse_noqa(code)
    assert result == {"E501": {1}, "W503": {2}}

def test_parse_noqa_accumulates_same_error_code_on_different_lines():
    code = "x = 1  # noqa: E501\ny = 2  # noqa: E501\n"
    result = parse_noqa(code)
    assert result == {"E501": {1, 2}}

def test_parse_noqa_applies_code_mapping():
    code = "x = 1  # noqa: F401\n"
    result = parse_noqa(code)
    assert result == {"F401": {1}}

def test_parse_noqa_handles_empty_code_string_after_colon():
    code = "x = 1  # noqa:\ny = 2\n"
    result = parse_noqa(code)
    assert result == {"all": {1}}


# LLM-generated content at query #3
#--------------------------

```
def test_parse_error_codes_all_codes():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "all")
    result = _parse_error_codes(match)
    assert result == ["all"]

def test_parse_error_codes_single_code():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "E001")
    result = _parse_error_codes(match)
    assert result == ["E001"]

def test_parse_error_codes_multiple_codes():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "E001, E002, E003")
    result = _parse_error_codes(match)
    assert result == ["E001", "E002", "E003"]

def test_parse_error_codes_with_spaces():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "  E001  ,  E002  ")
    result = _parse_error_codes(match)
    assert result == ["E001", "E002"]

def test_parse_error_codes_empty_string():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "")
    result = _parse_error_codes(match)
    assert result == ["all"]

def test_parse_error_codes_none():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, None)
    result = _parse_error_codes(match)
    assert result == ["all"]
```


# LLM-generated content at query #4
#--------------------------

```
def test_parse_noqa_with_no_noqa_comments():
    code = "x = 1\ny = 2"
    result = parse_noqa(code)
    assert result == defaultdict(set)

def test_parse_noqa_with_single_error_code():
    code = "x = 1  # noqa: E501"
    result = parse_noqa(code)
    assert result == defaultdict(set, {"E501": {1}})

def test_parse_noqa_with_multiple_error_codes():
    code = "x = 1  # noqa: E501, W601"
    result = parse_noqa(code)
    assert result == defaultdict(set, {"E501": {1}, "W601": {1}})

def test_parse_noqa_with_all_error_codes():
    code = "x = 1  # noqa"
    result = parse_noqa(code)
    assert result == defaultdict(set, {"all": {1}})

def test_parse_noqa_with_multiple_lines():
    code = "x = 1  # noqa: E501\ny = 2  # noqa: W601"
    result = parse_noqa(code)
    assert result == defaultdict(set, {"E501": {1}, "W601": {2}})

def test_parse_noqa_with_mapped_error_code():
    code = "x = 1  # noqa: E501"
    # Assuming NOQA_CODE_MAP maps "E501" to something else, but for simplicity test with unmapped
    result = parse_noqa(code)
    assert result == defaultdict(set, {"E501": {1}})


# LLM-generated content at query #5
#--------------------------

```
def test_parse_noqa_returns_non_empty_dict_when_match_found():
    code = ["import os  # noqa"]
    result = parse_noqa(code)
    assert result


