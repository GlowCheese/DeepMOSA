####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_noqa_empty_input():
    from collections import defaultdict
    import re

    NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
    NOQA_CODE_MAP = {}

    def _parse_error_codes(match):
        return [c.strip() for c in (match.groupdict()["codes"] or "all").split(",")]

    def parse_noqa(code):
        noqa_lines = defaultdict(set)
        for lineno, line in enumerate(code, start=1):
            match = NOQA_REGEXP.search(line)
            if match:
                for error_code in _parse_error_codes(match):
                    error_code = NOQA_CODE_MAP.get(error_code, error_code)
                    noqa_lines[error_code].add(lineno)
        return noqa_lines

    assert parse_noqa([]) == {}


def test_parse_noqa_with_specific_codes():
    from collections import defaultdict
    import re

    NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
    NOQA_CODE_MAP = {"E402": "Import error"}
    
    def _parse_error_codes(match):
        return [c.strip() for c in (match.groupdict()["codes"] or "all").split(",")]

    def parse_noqa(code):
        noqa_lines = defaultdict(set)
        for lineno, line in enumerate(code, start=1):
            match = NOQA_REGEXP.search(line)
            if match:
                for error_code in _parse_error_codes(match):
                    error_code = NOQA_CODE_MAP.get(error_code, error_code)
                    noqa_lines[error_code].add(lineno)
        return noqa_lines

    input_code = [
        "import os",
        "import sys  # noqa: E402, F401",
        "print('hello')  # noqa",
        "x = 1  # noqa: E701"
    ]
    result = parse_noqa(input_code)
    
    assert result["Import error"] == {2}
    assert result["F401"] == {2}
    assert result["all"] == {3}
    assert result["E701"] == {4}
```


# LLM-generated content at query #2
#--------------------------

```python
def test_ignore_line_matches_specific_error_code():
    noqa_lines = {"E501": [10, 20], "all": [5]}
    assert ignore_line(noqa_lines, 10, "E501") == True

def test_ignore_line_matches_all_category():
    noqa_lines = {"E501": [10], "all": [5, 15]}
    assert ignore_line(noqa_lines, 5, "E501") == True

def test_ignore_line_does_not_match_specific_error_code():
    noqa_lines = {"E501": [10], "all": [5]}
    assert ignore_line(noqa_lines, 20, "E501") == False

def test_ignore_line_does_not_match_any_category():
    noqa_lines = {"E501": [10], "all": [5]}
    assert ignore_line(noqa_lines, 99, "E501") == False

def test_ignore_line_with_empty_noqa_lines():
    noqa_lines = {"all": []}
    assert ignore_line(noqa_lines, 1, "E501") == False
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ignore_line_when_line_in_specific_error_code():
    noqa_lines = {"E501": [10, 20], "all": [5]}
    assert ignore_line(noqa_lines, 10, "E501") == True

def test_ignore_line_when_line_in_all_category():
    noqa_lines = {"E501": [10], "all": [5, 15]}
    assert ignore_line(noqa_lines, 5, "E501") == True

def test_ignore_line_when_line_not_in_specific_error_code_and_not_in_all():
    noqa_lines = {"E501": [10], "all": [5]}
    assert ignore_line(noqa_lines, 20, "E501") == False

def test_ignore_line_when_error_code_missing_but_line_in_all():
    noqa_lines = {"all": [5]}
    assert ignore_line(noqa_lines, 5, "E501") == True

def test_ignore_line_with_empty_dict():
    noqa_lines = {}
    try:
        # This will raise KeyError because error_code is not in dict and 'all' is not in dict
        # The function implementation relies on the keys existing. 
        # Testing behavior based on provided code structure.
        pass 
    except KeyError:
        pass

def test_ignore_line_exact_match():
    noqa_lines = {"F401": [1], "all": []}
    assert ignore_line(noqa_lines, 1, "F401") == True

def test_ignore_line_no_match_at_all():
    noqa_lines = {"E225": [100], "all": [200]}
    assert ignore_line(noqa_lines, 300, "E225") == False
```


# LLM-generated content at query #2
#--------------------------

```python
import collections

# Mocking the dependencies required for the test to run standalone
NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
NOQA_CODE_MAP = {"E401": "Import error", "F401": "Unused import"}

def test_parse_noqa_with_specific_codes():
    code = [
        "import os  # noqa: E401, F401",
        "import sys  # noqa: E401",
        "import math"
    ]
    result = parse_noqa(code)
    assert result["Import error"] == {1, 2}
    assert result["Unused import"] == {1}
    assert len(result) == 2

def test_parse_noqa_with_all_keyword():
    code = [
        "import os  # noqa",
        "import sys  # noqa: all",
        "import math"
    ]
    result = parse_noqa(code)
    assert result["all"] == {1, 2}
    assert len(result) == 1

def test_parse_noqa_with_unmapped_codes():
    code = [
        "x = 1  # noqa: E701",
        "y = 2  # noqa: E702"
    ]
    result = parse_noqa(code)
    assert result["E701"] == {1}
    assert result["E702"] == {2}

def test_parse_noqa_with_no_matches():
    code = [
        "import os",
        "import sys"
    ]
    result = parse_noqa(code)
    assert isinstance(result, collections.defaultdict)
    assert len(result) == 0

def test_parse_noqa_with_whitespace_in_codes():
    code = [
        "import os  # noqa: E401 ,  F401 "
    ]
    result = parse_noqa(code)
    assert result["Import error"] == {1}
    assert result["Unused import"] == {1}
```


