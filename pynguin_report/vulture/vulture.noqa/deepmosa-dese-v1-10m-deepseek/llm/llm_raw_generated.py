####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_noqa_with_no_noqa_comments():
    code = "x = 1\ny = 2\n"
    result = parse_noqa(code)
    assert result == defaultdict(set)

def test_parse_noqa_with_simple_noqa_comment():
    code = "x = 1  # noqa\n"
    result = parse_noqa(code)
    assert result == {"all": {1}}

def test_parse_noqa_with_specific_error_code():
    code = "x = 1  # noqa: E501\n"
    result = parse_noqa(code)
    assert result == {"E501": {1}}

def test_parse_noqa_with_multiple_error_codes():
    code = "x = 1  # noqa: E501, W601\n"
    result = parse_noqa(code)
    assert result == {"E501": {1}, "W601": {1}}

def test_parse_noqa_with_multiple_lines():
    code = "x = 1  # noqa: E501\ny = 2  # noqa: W601\n"
    result = parse_noqa(code)
    assert result == {"E501": {1}, "W601": {2}}

def test_parse_noqa_with_mapped_error_code():
    code = "x = 1  # noqa: F401\n"
    result = parse_noqa(code)
    assert result == {"F401": {1}}

def test_parse_noqa_with_all_category():
    code = "x = 1  # noqa\n"
    result = parse_noqa(code)
    assert result == {"all": {1}}

def test_parse_noqa_with_multiple_occurrences_same_line():
    code = "x = 1  # noqa: E501, E501\n"
    result = parse_noqa(code)
    assert result == {"E501": {1}}

def test_parse_noqa_with_empty_code():
    code = "x = 1  # noqa:\n"
    result = parse_noqa(code)
    assert result == {"all": {1}}
```


# LLM-generated content at query #2
#--------------------------

```
def test_parse_noqa_predicate_true():
    code = ["# noqa"]
    result = parse_noqa(code)
    assert result is not None
```


# LLM-generated content at query #3
#--------------------------

```
def test_ignore_line_in_noqa_lines_for_specific_error_code():
    noqa_lines = {"E101": {1, 2, 3}, "all": set()}
    assert ignore_line(noqa_lines, 2, "E101") == True

def test_ignore_line_not_in_noqa_lines_for_specific_error_code():
    noqa_lines = {"E101": {1, 3}, "all": set()}
    assert ignore_line(noqa_lines, 2, "E101") == False

def test_ignore_line_in_noqa_lines_for_all():
    noqa_lines = {"E101": set(), "all": {1, 2, 3}}
    assert ignore_line(noqa_lines, 2, "E101") == True

def test_ignore_line_not_in_noqa_lines_for_all():
    noqa_lines = {"E101": set(), "all": {1, 3}}
    assert ignore_line(noqa_lines, 2, "E101") == False

def test_ignore_line_in_both():
    noqa_lines = {"E101": {2}, "all": {2}}
    assert ignore_line(noqa_lines, 2, "E101") == True

def test_ignore_line_in_neither():
    noqa_lines = {"E101": set(), "all": set()}
    assert ignore_line(noqa_lines, 2, "E101") == False

def test_ignore_line_different_error_code():
    noqa_lines = {"E201": {2}, "all": set()}
    assert ignore_line(noqa_lines, 2, "E101") == False
```


# LLM-generated content at query #4
#--------------------------

```
def test_parse_error_codes_returns_all_when_codes_is_none():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "")
    assert _parse_error_codes(match) == ["all"]

def test_parse_error_codes_returns_all_when_codes_is_empty_string():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "")
    assert _parse_error_codes(match) == ["all"]

def test_parse_error_codes_single_code():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "E001")
    assert _parse_error_codes(match) == ["E001"]

def test_parse_error_codes_multiple_codes():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "E001, E002, E003")
    assert _parse_error_codes(match) == ["E001", "E002", "E003"]

def test_parse_error_codes_handles_whitespace():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "  E001  ,  E002  ")
    assert _parse_error_codes(match) == ["E001", "E002"]

def test_parse_error_codes_handles_all_keyword():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "all")
    assert _parse_error_codes(match) == ["all"]

def test_parse_error_codes_handles_newlines():
    import re
    pattern = r"(?P<codes>.*)"
    match = re.match(pattern, "E001\n,E002")
    assert _parse_error_codes(match) == ["E001", "E002"]
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_parse_error_codes_with_specific_codes():
    import re
    pattern = re.compile(r'(?P<codes>[^"]*)')
    match = pattern.match("123, 456, 789")
    result = _parse_error_codes(match)
    assert result == ["123", "456", "789"]

def test_parse_error_codes_with_all():
    import re
    pattern = re.compile(r'(?P<codes>[^"]*)')
    match = pattern.match("")
    result = _parse_error_codes(match)
    assert result == ["all"]

def test_parse_error_codes_with_single_code():
    import re
    pattern = re.compile(r'(?P<codes>[^"]*)')
    match = pattern.match("ERROR")
    result = _parse_error_codes(match)
    assert result == ["ERROR"]

def test_parse_error_codes_with_whitespace():
    import re
    pattern = re.compile(r'(?P<codes>[^"]*)')
    match = pattern.match(" 100 , 200 ")
    result = _parse_error_codes(match)
    assert result == ["100", "200"]

def test_parse_error_codes_with_none_match():
    import re
    class MockMatch:
        def groupdict(self):
            return {"codes": None}
    match = MockMatch()
    result = _parse_error_codes(match)
    assert result == ["all"]
```


# LLM-generated content at query #2
#--------------------------

```
def test_parse_noqa_returns_defaultdict_with_sets():
    code = "# noqa\n"
    result = parse_noqa(code)
    assert isinstance(result, defaultdict)

def test_parse_noqa_all_codes():
    code = "import os  # noqa\n"
    result = parse_noqa(code)
    assert "all" in result
    assert 1 in result["all"]

def test_parse_noqa_specific_code():
    code = "import os  # noqa: F401\n"
    result = parse_noqa(code)
    assert "F401" in result
    assert 1 in result["F401"]

def test_parse_noqa_multiple_codes():
    code = "import os  # noqa: F401, E302\n"
    result = parse_noqa(code)
    assert "F401" in result
    assert "E302" in result
    assert 1 in result["F401"]
    assert 1 in result["E302"]

def test_parse_noqa_multiple_lines():
    code = "# noqa: F401\na = 1  # noqa: E302\n"
    result = parse_noqa(code)
    assert 1 in result["F401"]
    assert 2 in result["E302"]

def test_parse_noqa_no_noqa():
    code = "import os\n"
    result = parse_noqa(code)
    assert len(result) == 0

def test_parse_noqa_empty_string():
    code = ""
    result = parse_noqa(code)
    assert len(result) == 0

def test_parse_noqa_maps_code():
    code = "# noqa: E501\n"
    result = parse_noqa(code)
    assert "E501" in result
    assert 1 in result["E501"]

def test_parse_noqa_unknown_code():
    code = "# noqa: X123\n"
    result = parse_noqa(code)
    assert "X123" in result
    assert 1 in result["X123"]
```


# LLM-generated content at query #3
#--------------------------

```
def test_parse_noqa_with_match():
    code = ["x = 1  # noqa"]
    result = parse_noqa(code)
    assert len(result) > 0
```


# LLM-generated content at query #4
#--------------------------

```python
def test_ignore_line_error_code_match():
    noqa_lines = {"E501": {3, 5}, "all": {7}}
    assert ignore_line(noqa_lines, 3, "E501") == True

def test_ignore_line_no_error_code_match():
    noqa_lines = {"E501": {3, 5}, "all": {7}}
    assert ignore_line(noqa_lines, 4, "E501") == False

def test_ignore_line_all_code_match():
    noqa_lines = {"E501": {3, 5}, "all": {7}}
    assert ignore_line(noqa_lines, 7, "E501") == True

def test_ignore_line_error_code_match_not_all():
    noqa_lines = {"E501": {3, 5}, "all": {7}}
    assert ignore_line(noqa_lines, 5, "W001") == False

def test_ignore_line_empty_noqa_lines():
    noqa_lines = {"E501": set(), "all": set()}
    assert ignore_line(noqa_lines, 1, "E501") == False

def test_ignore_line_lineno_in_both():
    noqa_lines = {"E501": {2}, "all": {2}}
    assert ignore_line(noqa_lines, 2, "E501") == True

def test_ignore_line_different_error_code():
    noqa_lines = {"E501": {3}, "all": set()}
    assert ignore_line(noqa_lines, 3, "W001") == False
```


