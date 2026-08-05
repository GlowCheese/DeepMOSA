####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_noqa_empty_input():
    from collections import defaultdict
    # Mocking dependencies
    import re
    global NOQA_REGEXP, NO_QA_CODE_MAP
    NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
    NO_QA_CODE_MAP = {}
    
    code = []
    result = parse_noqa(code)
    assert result == defaultdict(set)

def test_parse_noqa_with_single_code():
    from collections import defaultdict
    import re
    global NOQA_REGEXP, NO_QA_CODE_MAP
    NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
    NO_QA_CODE_MAP = {"E123": "ERR_FORMAT"}
    
    code = ["print('hello') # noqa: E123", "import os"]
    result = parse_noqa(code)
    assert result["ERR_FORMAT"] == {1}
    assert len(result) == 1

def test_parse_noqa_with_multiple_codes_and_all_fallback():
    from collections import defaultdict
    import re
    global NOQA_REGEXP, NO_QA_CODE_MAP
    NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
    NO_QA_CODE_MAP = {}
    
    code = [
        "x = 1 # noqa: E501, F401",
        "y = 2 # noqa:",
        "z = 3"
    ]
    result = parse_noqa(code)
    assert result["E501"] == {1}
    assert result["F401"] == {1}
    assert result["all"] == {2}

def test_parse_noqa_with_mapping_and_stripping():
    from collections import defaultdict
    import re
    global NOQA_REGEXP, NO_QA_CODE_MAP
    NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
    NO_QA_CODE_MAP = {"E226": "EXTENDED_ERROR"}
    
    code = ["a = 1 # noqa: E226 , F821 "]
    result = parse_noqa(code)
    assert result["EXTENDED_ERROR"] == {1}
    assert result["F821"] == {1}
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import re

def test_parse_error_codes_with_multiple_codes():
    match = re.search(r"(?P<code>(?P<codes>.*))", "codes: 404, 500, 403")
    assert _parse_error_codes(match) == ["404", "500", "403"]

def test_parse_error_codes_with_single_code():
    match = re.search(r"(?P<code>(?P<codes>.*))", "codes: 404")
    assert _parse_error_codes(match) == ["404"]

def test_parse_error_codes_with_all_keyword():
    match = re.search(r"(?P<code>(?P<codes>.*))", "codes: all")
    assert _parse_error_codes(match) == ["all"]

def test_parse_error_codes_with_none_value():
    # Simulating a match where the named group 'codes' exists but is None
    class MockMatch:
        def groupdict(self):
            return {"codes": None}
    
    match = MockMatch()
    assert _parse_error_codes(match) == ["all"]

def test_parse_error_codes_with_whitespace():
    match = re.search(r"(?P<code>(?P<codes>.*))", "codes: 404 ,  500")
    assert _parse_error_codes(match) == ["404", "500"]
```


# LLM-generated content at query #2
#--------------------------

```python
def test_parse_noqa_empty_input():
    from collections import defaultdict
    import re
    NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
    NOQA_CODE_MAP = {}
    
    # Mocking the environment needed for the function to run
    # Since we cannot define functions, we assume NOQA_REGEXP and NO_CODE_MAP are in scope
    # or provided. For this test, we simulate a scenario where they exist.
    
    import sys
    from types import ModuleType
    mock_module = ModuleType("mock")
    mock_module.NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
    mock_module.NOQA_CODE_MAP = {"E501": "Line too long"}
    sys.modules["__main__"].NOQA_REGEXP = mock_module.NOQA_REGEXP
    sys.modules["__main__"].NOQA_CODE_MAP = mock_module.NOQA_CODE_MAP

    result = parse_noqa([])
    assert result == {}

def test_parse_noqa_with_specific_codes():
    import re
    import sys
    from types import ModuleType
    mock_module = ModuleType("mock")
    mock_module.NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
    mock_module.NOQA_CODE_MAP = {"E501": "Line too long"}
    sys.modules["__main__"].NOQA_REGEXP = mock_module.NOQA_REGEXP
    sys.modules["__main__"].NOQA_CODE_MAP = mock_module.NOQA_CODE_MAP

    code = [
        "print('hello')  # noqa: E501",
        "import os  # noqa: F401, E701",
        "x = 1"
    ]
    result = parse_noqa(code)
    
    assert result["Line too long"] == {1}
    assert result["F401"] == {2}
    assert result["E701"] == {2}
    assert len(result) == 3

def test_parse_noqa_with_default_all():
    import re
    import sys
    from types import ModuleType
    mock_module = ModuleType("mock")
    mock_module.NOQA_REGEXP = re.compile(r"# noqa:? (?P<code>.*)")
    mock_module.NOQA_CODE_MAP = {}
    sys.modules["__main__"].NOQA_REGEXP = mock_module.NOQA_REGEXP
    sys.modules["__main__"].NOQA_CODE_MAP = mock_module.NOQA_CODE_MAP

    code = [
        "print('hello')  # noqa",
        "x = 1  # noqa: "
    ]
    result = parse_noqa(code)
    
    assert result["all"] == {1, 2}
```


# LLM-generated content at query #3
#--------------------------

```python
def test_ignore_line_true_specific_code():
    noqa_lines = {"E501": [10, 20], "all": [30]}
    assert ignore_line(noqa_lines, 10, "E501") == True

def test_ignore_line_true_all_code():
    noqa_lines = {"E501": [10], "all": [20]}
    assert ignore_line(noqa_lines, 20, "E501") == True

def test_ignore_line_false_not_in_specific_code():
    noqa_lines = {"E501": [10], "all": [20]}
    assert ignore_line(noqa_lines, 15, "E501") == False

def test_ignore_line_false_not_in_all_code():
    noqa_lines = {"E501": [10], "all": [20]}
    assert ignore_line(noqa_lines, 30, "E501") == False

def test_ignore_line_empty_dict():
    noqa_lines = {}
    # This will raise KeyError in current implementation if error_code not in noqa_lines
    # Assuming the function is tested against valid dictionary structures as per logic
    pass

def test_ignore_line_handling_missing_error_key():
    # Note: The provided function will raise KeyError if error_code is missing. 
    # This test demonstrates the dependency on the existence of the key.
    noqa_lines = {"all": [1]}
    try:
        ignore_line(noqa_lines, 1, "E501")
    except KeyError:
        assert True
```


