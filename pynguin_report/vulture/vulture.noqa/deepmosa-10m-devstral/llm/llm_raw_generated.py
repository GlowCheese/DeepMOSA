####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_noqa_with_single_error_code():
    code = ["x = 1  # noqa: E123"]
    result = parse_noqa(code)
    assert result == {"E123": {1}}

def test_parse_noqa_with_multiple_error_codes():
    code = ["x = 1  # noqa: E123, F456"]
    result = parse_noqa(code)
    assert result == {"E123": {1}, "F456": {1}}

def test_parse_noqa_with_all_error_codes():
    code = ["x = 1  # noqa"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

def test_parse_noqa_with_mixed_error_codes():
    code = ["x = 1  # noqa: E123", "y = 2  # noqa: F456, G789"]
    result = parse_noqa(code)
    assert result == {"E123": {1}, "F456": {2}, "G789": {2}}

def test_parse_noqa_with_no_noqa_comments():
    code = ["x = 1", "y = 2"]
    result = parse_noqa(code)
    assert result == {}

def test_parse_noqa_with_empty_line():
    code = ["", "x = 1  # noqa: E123"]
    result = parse_noqa(code)
    assert result == {"E123": {2}}

def test_parse_noqa_with_mapped_error_code():
    code = ["x = 1  # noqa: W123"]
    result = parse_noqa(code)
    assert result == {NOQA_CODE_MAP["W123"]: {1}}


# LLM-generated content at query #2
#--------------------------

```python
def test_ignore_line_when_lineno_in_noqa_lines_for_error_code():
    noqa_lines = {"E123": {5, 10}, "all": {1, 2}}
    assert ignore_line(noqa_lines, 5, "E123") == True

def test_ignore_line_when_lineno_in_noqa_lines_all():
    noqa_lines = {"E123": {5, 10}, "all": {1, 2}}
    assert ignore_line(noqa_lines, 1, "E123") == True

def test_ignore_line_when_lineno_not_in_noqa_lines():
    noqa_lines = {"E123": {5, 10}, "all": {1, 2}}
    assert ignore_line(noqa_lines, 3, "E123") == False

def test_ignore_line_when_noqa_lines_empty():
    noqa_lines = {"E123": set(), "all": set()}
    assert ignore_line(noqa_lines, 5, "E123") == False


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_error_codes_with_no_codes():
    import re
    match = re.match(r"(?P<codes>)", "")
    assert _parse_error_codes(match) == ["all"]

def test_parse_error_codes_with_single_code():
    import re
    match = re.match(r"(?P<codes>E001)", "E001")
    assert _parse_error_codes(match) == ["E001"]

def test_parse_error_codes_with_multiple_codes():
    import re
    match = re.match(r"(?P<codes>E001,E002,E003)", "E001,E002,E003")
    assert _parse_error_codes(match) == ["E001", "E002", "E003"]

def test_parse_error_codes_with_whitespace():
    import re
    match = re.match(r"(?P<codes> E001 , E002 , E003 )", " E001 , E002 , E003 ")
    assert _parse_error_codes(match) == ["E001", "E002", "E003"]


# LLM-generated content at query #2
#--------------------------

```python
def test_ignore_line_when_lineno_in_error_code_noqa_lines():
    noqa_lines = {"E123": {5, 10}, "all": {2, 7}}
    assert ignore_line(noqa_lines, 5, "E123") is True

def test_ignore_line_when_lineno_in_all_noqa_lines():
    noqa_lines = {"E123": {5, 10}, "all": {2, 7}}
    assert ignore_line(noqa_lines, 2, "E123") is True

def test_ignore_line_when_lineno_not_in_noqa_lines():
    noqa_lines = {"E123": {5, 10}, "all": {2, 7}}
    assert ignore_line(noqa_lines, 3, "E123") is False

def test_ignore_line_when_error_code_not_in_noqa_lines():
    noqa_lines = {"E123": {5, 10}, "all": {2, 7}}
    assert ignore_line(noqa_lines, 5, "E456") is False

def test_ignore_line_with_empty_noqa_lines():
    noqa_lines = {"E123": set(), "all": set()}
    assert ignore_line(noqa_lines, 5, "E123") is False


# LLM-generated content at query #3
#--------------------------

```python
def test_parse_noqa_with_no_error_codes():
    code = ["x = 1  # noqa", "y = 2  # noqa"]
    result = parse_noqa(code)
    assert result == {"all": {1, 2}}

def test_parse_noqa_with_single_error_code():
    code = ["x = 1  # noqa: F401", "y = 2  # noqa: F841"]
    result = parse_noqa(code)
    assert result == {"F401": {1}, "F841": {2}}

def test_parse_noqa_with_multiple_error_codes():
    code = ["x = 1  # noqa: F401, F841", "y = 2  # noqa: E711, E712"]
    result = parse_noqa(code)
    assert result == {"F401": {1}, "F841": {1}, "E711": {2}, "E712": {2}}

def test_parse_noqa_with_mixed_error_codes():
    code = ["x = 1  # noqa", "y = 2  # noqa: F401", "z = 3  # noqa: F841, E711"]
    result = parse_noqa(code)
    assert result == {"all": {1}, "F401": {2}, "F841": {3}, "E711": {3}}

def test_parse_noqa_with_no_matches():
    code = ["x = 1", "y = 2"]
    result = parse_noqa(code)
    assert result == {}

def test_parse_noqa_with_code_mapping():
    code = ["x = 1  # noqa: F", "y = 2  # noqa: E"]
    result = parse_noqa(code)
    assert result == {"F": {1}, "E": {2}}


