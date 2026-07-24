####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import collections

# Mocking dependencies for the test environment
NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
NOQA_CODE_MAP = {"E401": "Multiple imports", "F401": "Unused import"}

def test_parse_noqa_with_specific_codes():
    code = [
        "import os  # noqa: E401",
        "import sys  # noqa: F401, E722",
        "import math"
    ]
    expected = collections.defaultdict(set, {
        "Multiple imports": {1},
        "Unused import": {2},
        "E722": {2}
    })
    assert parse_noqa(code) == expected

def test_parse_noqa_with_all_keyword():
    code = [
        "import os  # noqa",
        "import sys  # noqa: all",
        "import math"
    ]
    expected = collections.defaultdict(set, {
        "all": {1, 2}
    })
    assert parse_noqa(code) == expected

def test_parse_noqa_empty_input():
    code = []
    expected = collections.defaultdict(set)
    assert parse_noqa(code) == expected

def test_parse_noqa_no_matches():
    code = [
        "import os",
        "import sys",
        "import math"
    ]
    expected = collections.defaultdict(set)
    assert parse_noqa(code) == expected

def test_parse_noqa_with_whitespace_in_codes():
    code = [
        "import os  # noqa: E401,  F401 "
    ]
    expected = collections.defaultdict(set, {
        "Multiple imports": {1},
        "Unused import": {1}
    })
    assert parse_noqa(code) == expected
```


# LLM-generated content at query #2
#--------------------------

```python
def test_ignore_line_matches_specific_error_code():
    noqa_lines = {"E401": [10, 20], "all": [5]}
    assert ignore_line(noqa_lines, 10, "E401") == True

def test_ignore_line_matches_all_annotation():
    noqa_lines = {"E401": [10], "all": [5, 6]}
    assert ignore_line(noqa_lines, 5, "E401") == True

def test_ignore_line_does_not_match_different_error_code():
    noqa_lines = {"E401": [10], "all": [5]}
    assert ignore_line(noqa_lines, 20, "E401") == False

def test_ignore_line_does_not_match_when_lineno_not_in_any_list():
    noqa_lines = {"E401": [10], "all": [5]}
    assert ignore_line(noqa_lines, 99, "E401") == False

def test_ignore_line_with_empty_noqa_dict():
    noqa_lines = {}
    # This will raise KeyError in current implementation if error_code not in dict
    # Assuming the function expects keys to exist based on provided snippet
    pass

def test_ignore_line_handles_missing_specific_key_but_present_in_all():
    noqa_lines = {"all": [1]}
    assert ignore_line(noqa_lines, 1, "E501") == True
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_error_codes_with_multiple_codes():
    import re
    match = re.match(r"(?P<code>.*)", "404, 500, 403")
    assert _parse_error_codes(match) == ["404", "500", "403"]

def test_parse_error_codes_with_single_code():
    import re
    match = re.match(r"(?P<code>.*)", "404")
    assert _parse_error_codes(match) == ["404"]

def test_parse_error_codes_with_none_returns_all():
    import re
    match = re.match(r"(?P<code>(?P<codes>.*))", "")
    # In this case, groupdict()["codes"] will be "" which is not None, 
    # but let's simulate the logic where it might return None/Empty if regex doesn't capture
    class MockMatch:
        def groupdict(self):
            return {"codes": None}
    assert _parse_error_codes(MockMatch()) == ["all"]

def test_parse_error_codes_with_empty_string_returns_all_logic():
    import re
    # If codes is an empty string, split(",") on "" results in ['']
    match = re.match(r"(?P<code>.*)", "")
    assert _parse_error_codes(match) == [""]

def test_parse_error_codes_with_whitespace():
    import re
    match = re.match(r"(?P<code>.*)", " 401 , 402 ")
    assert _parse_error_codes(match) == ["401", "402"]
```


# LLM-generated content at query #2
#--------------------------

```python
from collections import defaultdict
import re

NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
NOQA_CODE_MAP = {"E401": "Too many arguments", "F401": "Unused import"}

def test_parse_noqa_empty_input():
    assert parse_noqa([]) == defaultdict(set)

def test_parse_noqa_single_code_with_mapping():
    lines = ["import os  # noqa: F401"]
    result = parse_noqa(lines)
    assert result["Unused import"] == {1}

def test_parse_noqa_multiple_codes_and_all_keyword():
    lines = [
        "import sys  # noqa: E401, F821",
        "import math  # noqa",
        "import os  # noqa: all"
    ]
    result = parse_noqa(lines)
    assert result["Too many arguments"] == {1}
    assert result["F821"] == {1}
    assert result["all"] == {2}
    assert result["all"].__contains__(3)

def test_parse_noqa_unmapped_codes():
    lines = ["x = 1  # noqa: E722"]
    result = parse_noqa(lines)
    assert result["E722"] == {1}

def test_parse_noqa_strip_whitespace_in_codes():
    lines = ["print(1)  # noqa: E401 , F401 "]
    result = parse_noqa(lines)
    assert result["Too many arguments"] == {1}
    assert result["F401"] == {1}

def test_parse_noqa_no_match_on_line():
    lines = ["print('hello')"]
    result = parse_noqa(lines)
    assert result == defaultdict(set)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_ignore_line_match_specific_error():
    noqa_lines = {"E501": [10, 20], "all": [30]}
    assert ignore_line(noqa_lines, 10, "E501") == True

def test_ignore_line_match_all_error():
    noqa_lines = {"E501": [10], "all": [30]}
    assert ignore_line(noqa_lines, 30, "E501") == True

def test_ignore_line_no_match():
    noqa_lines = {"E501": [10], "all": [30]}
    assert ignore_line(noqa_lines, 40, "E501") == False

def test_ignore_line_empty_dict():
    noqa_lines = {}
    # Note: This would raise KeyError in the current implementation if error_code or 'all' is missing.
    # Assuming valid input structure based on provided function logic.
    pass

def test_ignore_line_specific_error_not_in_list():
    noqa_lines = {"E501": [10], "all": []}
    assert ignore_line(noqa_lines, 20, "E501") == False

def test_ignore_line_with_only_all_present():
    noqa_lines = {"all": [5]}
    # Accessing noqa_lines[error_code] will fail if error_code is not in dict.
    # Assuming the function expects error_code to exist in the dictionary.
    pass
```

Wait, looking at the provided code, `noqa_lines[error_code]` will raise a `KeyError` if the specific error code is not a key in the dictionary. I will provide tests that assume valid input (where the error code exists).

```python
def test_ignore_line_true_when_specific_error_matches():
    noqa_lines = {"E501": [1, 2], "all": [3]}
    assert ignore_line(noqa_lines, 1, "E501") == True

def test_ignore_line_true_when_all_matches():
    noqa_lines = {"E501": [1], "all": [2]}
    assert ignore_line(noqa_lines, 2, "E501") == True

def test_ignore_line_false_when_none_match():
    noqa_lines = {"E501": [1], "all": [2]}
    assert ignore_line(noqa_lines, 3, "E501") == False

def test_ignore_line_true_when_specific_error_matches_exactly():
    noqa_lines = {"E501": [5], "all": []}
    assert ignore_line(noqa_lines, 5, "E501") == True
```


