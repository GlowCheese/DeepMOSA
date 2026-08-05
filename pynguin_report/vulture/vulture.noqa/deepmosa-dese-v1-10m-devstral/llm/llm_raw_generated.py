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

def test_parse_noqa_with_code_mapping():
    code = ["x = 1  # noqa: E123"]
    NOQA_CODE_MAP = {"E123": "E999"}
    expected = defaultdict(set, {"E999": {1}})
    assert parse_noqa(code) == expected


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_error_codes_with_no_codes():
    match = type('Match', (), {'groupdict': lambda: {'codes': None}})
    assert _parse_error_codes(match) == ['all']

def test_parse_error_codes_with_single_code():
    match = type('Match', (), {'groupdict': lambda: {'codes': 'E001'}})
    assert _parse_error_codes(match) == ['E001']

def test_parse_error_codes_with_multiple_codes():
    match = type('Match', (), {'groupdict': lambda: {'codes': 'E001, E002, E003'}})
    assert _parse_error_codes(match) == ['E001', 'E002', 'E003']

def test_parse_error_codes_with_whitespace():
    match = type('Match', (), {'groupdict': lambda: {'codes': '  E001  ,  E002  '}})
    assert _parse_error_codes(match) == ['E001', 'E002']


# LLM-generated content at query #2
#--------------------------

```python
def test_ignore_line_with_specific_error_code():
    noqa_lines = {"E123": {5, 10}, "all": {3}}
    assert ignore_line(noqa_lines, 5, "E123") == True
    assert ignore_line(noqa_lines, 10, "E123") == True
    assert ignore_line(noqa_lines, 7, "E123") == False

def test_ignore_line_with_all_error_codes():
    noqa_lines = {"E123": {5, 10}, "all": {3, 8}}
    assert ignore_line(noqa_lines, 3, "E123") == True
    assert ignore_line(noqa_lines, 8, "E456") == True
    assert ignore_line(noqa_lines, 5, "E456") == False

def test_ignore_line_with_empty_noqa_lines():
    noqa_lines = {"E123": set(), "all": set()}
    assert ignore_line(noqa_lines, 1, "E123") == False
    assert ignore_line(noqa_lines, 2, "E456") == False

def test_ignore_line_with_missing_error_code():
    noqa_lines = {"E123": {5}, "all": {3}}
    assert ignore_line(noqa_lines, 5, "E456") == False
    assert ignore_line(noqa_lines, 3, "E456") == True


# LLM-generated content at query #3
#--------------------------

```python
def test_parse_error_codes_with_empty_codes():
    match = type('Match', (), {'groupdict': lambda: {'codes': ''}})
    assert _parse_error_codes(match) == ['all']


# LLM-generated content at query #4
#--------------------------

```python
def test_parse_noqa_with_single_error_code():
    code = ["# noqa: E123"]
    result = parse_noqa(code)
    assert result == {"E123": {1}}

def test_parse_noqa_with_multiple_error_codes():
    code = ["# noqa: E123, F456"]
    result = parse_noqa(code)
    assert result == {"E123": {1}, "F456": {1}}

def test_parse_noqa_with_all_error_codes():
    code = ["# noqa"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

def test_parse_noqa_with_multiple_lines():
    code = ["# noqa: E123", "x = 1  # noqa: F456", "y = 2"]
    result = parse_noqa(code)
    assert result == {"E123": {1}, "F456": {2}}

def test_parse_noqa_with_mapped_error_code():
    code = ["# noqa: W123"]
    result = parse_noqa(code)
    assert result == {"W123": {1}}

def test_parse_noqa_with_empty_code():
    code = []
    result = parse_noqa(code)
    assert result == {}

def test_parse_noqa_with_no_noqa_comment():
    code = ["x = 1", "y = 2"]
    result = parse_noqa(code)
    assert result == {}


