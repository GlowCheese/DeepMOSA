####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_parse_noqa_single_line_single_error_code():
    code = "some code  # noqa: E123"
    result = parse_noqa(code)
    assert result == {"E123": {1}}

def test_parse_noqa_single_line_multiple_error_codes():
    code = "some code  # noqa: E123, W456, F789"
    result = parse_noqa(code)
    assert result == {"E123": {1}, "W456": {1}, "F789": {1}}

def test_parse_noqa_single_line_no_error_code():
    code = "some code  # noqa"
    result = parse_noqa(code)
    assert result == {"all": {1}}

def test_parse_noqa_multiple_lines():
    code = "line1  # noqa: E123\nline2  # noqa: W456\nline3"
    result = parse_noqa(code)
    assert result == {"E123": {1}, "W456": {2}}

def test_parse_noqa_same_error_code_on_multiple_lines():
    code = "line1  # noqa: E123\nline2  # noqa: E123"
    result = parse_noqa(code)
    assert result == {"E123": {1, 2}}

def test_parse_noqa_no_matches():
    code = "line1\nline2\nline3"
    result = parse_noqa(code)
    assert result == {}

def test_parse_noqa_with_code_mapping():
    code = "some code  # noqa: I001"
    # Assuming NOQA_CODE_MAP maps "I001" to something else
    # This test may need adjustment based on actual mapping
    result = parse_noqa(code)
    # For this test, we assume the mapping results in a different key
    mapped_key = NOQA_CODE_MAP.get("I001", "I001")
    assert result == {mapped_key: {1}}

def test_parse_noqa_empty_string():
    code = ""
    result = parse_noqa(code)
    assert result == {}

def test_parse_noqa_line_with_leading_spaces():
    code = "  # noqa: E123"
    result = parse_noqa(code)
    assert result == {"E123": {1}}

def test_parse_noqa_line_with_trailing_spaces():
    code = "some code  # noqa: E123   "
    result = parse_noqa(code)
    assert result == {"E123": {1}}

def test_parse_noqa_case_insensitivity():
    code = "some code  # NOQA: E123"
    result = parse_noqa(code)
    assert result == {"E123": {1}}

def test_parse_noqa_multiple_codes_with_whitespace():
    code = "some code  # noqa: E123, W456"
    result = parse_noqa(code)
    assert result == {"E123": {1}, "W456": {1}}

def test_parse_noqa_only_comment():
    code = "# noqa: E123"
    result = parse_noqa(code)
    assert result == {"E123": {1}}
```


# LLM-generated content at query #2
#--------------------------

def test_parse_error_codes_with_none():
    class MockMatch:
        def groupdict(self):
            return {"codes": None}
    result = _parse_error_codes(MockMatch())
    assert result == ["all"]

def test_parse_error_codes_with_single_code():
    class MockMatch:
        def groupdict(self):
            return {"codes": "E001"}
    result = _parse_error_codes(MockMatch())
    assert result == ["E001"]

def test_parse_error_codes_with_multiple_codes():
    class MockMatch:
        def groupdict(self):
            return {"codes": "E001, E002, E003"}
    result = _parse_error_codes(MockMatch())
    assert result == ["E001", "E002", "E003"]

def test_parse_error_codes_with_spaces():
    class MockMatch:
        def groupdict(self):
            return {"codes": "  E001 , E002 "}
    result = _parse_error_codes(MockMatch())
    assert result == ["E001", "E002"]

def test_parse_error_codes_with_empty_string():
    class MockMatch:
        def groupdict(self):
            return {"codes": ""}
    result = _parse_error_codes(MockMatch())
    assert result == ["all"]


# LLM-generated content at query #3
#--------------------------

```
def test_parse_error_codes_with_codes_specified():
    import re
    pattern = r"(?P<codes>error1,error2)"
    match = re.match(pattern, "error1,error2")
    result = _parse_error_codes(match)
    assert result == ["error1", "error2"]
```


# LLM-generated content at query #4
#--------------------------

```
def test_ignore_line_with_matching_error_code():
    noqa_lines = {"E501": {3, 5}, "all": set()}
    assert ignore_line(noqa_lines, 3, "E501") == True

def test_ignore_line_with_matching_all():
    noqa_lines = {"E501": set(), "all": {7, 9}}
    assert ignore_line(noqa_lines, 9, "W123") == True

def test_ignore_line_with_no_match():
    noqa_lines = {"E501": {1}, "all": {2}}
    assert ignore_line(noqa_lines, 3, "E501") == False

def test_ignore_line_with_empty_sets():
    noqa_lines = {"E501": set(), "all": set()}
    assert ignore_line(noqa_lines, 5, "E501") == False

def test_ignore_line_with_lineno_in_both_sets():
    noqa_lines = {"E501": {10}, "all": {10}}
    assert ignore_line(noqa_lines, 10, "E501") == True

def test_ignore_line_with_different_error_code_in_all():
    noqa_lines = {"E501": set(), "all": {15}}
    assert ignore_line(noqa_lines, 15, "F401") == True
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_parse_error_codes_with_multiple_codes():
    import re
    match = re.match(r"(?P<codes>.*)", "code1, code2, code3")
    result = _parse_error_codes(match)
    assert result == ["code1", "code2", "code3"]

def test_parse_error_codes_with_single_code():
    import re
    match = re.match(r"(?P<codes>.*)", "single")
    result = _parse_error_codes(match)
    assert result == ["single"]

def test_parse_error_codes_with_no_code():
    import re
    match = re.match(r"(?P<codes>.*)", "")
    result = _parse_error_codes(match)
    assert result == ["all"]

def test_parse_error_codes_with_whitespace_codes():
    import re
    match = re.match(r"(?P<codes>.*)", "  a , b  , c  ")
    result = _parse_error_codes(match)
    assert result == ["a", "b", "c"]

def test_parse_error_codes_with_all_keyword():
    import re
    match = re.match(r"(?P<codes>.*)", "all")
    result = _parse_error_codes(match)
    assert result == ["all"]


# LLM-generated content at query #2
#--------------------------

```
def test_parse_noqa_with_no_noqa_comments():
    code = "x = 1\ny = 2\nz = 3"
    result = parse_noqa(code)
    assert result == {}

def test_parse_noqa_with_noqa_comment_no_codes():
    code = "x = 1  # noqa\ny = 2"
    result = parse_noqa(code)
    assert result == {"all": {1}}

def test_parse_noqa_with_specific_code():
    code = "x = 1  # noqa: E501"
    result = parse_noqa(code)
    assert result == {"E501": {1}}

def test_parse_noqa_with_multiple_codes():
    code = "x = 1  # noqa: E501, E302"
    result = parse_noqa(code)
    assert result == {"E501": {1}, "E302": {1}}

def test_parse_noqa_with_multiple_lines():
    code = "x = 1  # noqa: E501\ny = 2  # noqa: E302"
    result = parse_noqa(code)
    assert result == {"E501": {1}, "E302": {2}}

def test_parse_noqa_with_same_code_on_multiple_lines():
    code = "x = 1  # noqa: E501\ny = 2  # noqa: E501"
    result = parse_noqa(code)
    assert result == {"E501": {1, 2}}

def test_parse_noqa_with_code_mapping():
    code = "x = 1  # noqa: F401"
    result = parse_noqa(code)
    assert result == {"E501": {1}}

def test_parse_noqa_with_noqa_comment_whitespace():
    code = "x = 1  #   noqa   "
    result = parse_noqa(code)
    assert result == {"all": {1}}

def test_parse_noqa_with_noqa_comment_at_end():
    code = "x = 1; y = 2  # noqa: E501"
    result = parse_noqa(code)
    assert result == {"E501": {1}}

def test_parse_noqa_with_empty_codes():
    code = "x = 1  # noqa:"
    result = parse_noqa(code)
    assert result == {"all": {1}}
```


# LLM-generated content at query #3
#--------------------------

def test_ignore_line_returns_true_when_lineno_in_error_code_set():
    noqa_lines = {"E501": {1, 2}, "all": set()}
    assert ignore_line(noqa_lines, 1, "E501") == True

def test_ignore_line_returns_true_when_lineno_in_all_set():
    noqa_lines = {"E501": set(), "all": {3}}
    assert ignore_line(noqa_lines, 3, "E501") == True

def test_ignore_line_returns_false_when_lineno_not_in_any_set():
    noqa_lines = {"E501": {1}, "all": {2}}
    assert ignore_line(noqa_lines, 3, "E501") == False

def test_ignore_line_returns_true_when_lineno_in_both_sets():
    noqa_lines = {"E501": {1}, "all": {1}}
    assert ignore_line(noqa_lines, 1, "E501") == True

def test_ignore_line_returns_false_with_empty_sets():
    noqa_lines = {"E501": set(), "all": set()}
    assert ignore_line(noqa_lines, 1, "E501") == False


# LLM-generated content at query #4
#--------------------------

def test_parse_noqa_predicate_true():
    code = ["# noqa"]
    result = parse_noqa(code)
    assert isinstance(result, dict)


