####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_noqa():
    # Test with a simple noqa comment without codes
    code = ["x = 1  # noqa"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test with specific error codes
    code = ["x = 1  # noqa: F401", "y = 2  # noqa: F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {2}}

    # Test with multiple codes on the same line
    code = ["x = 1  # noqa: F401, F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test with case-insensitive noqa
    code = ["x = 1  # NoQA: F401"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}

    # Test with multiple lines
    code = [
        "import os  # noqa: F401",
        "x = 1",
        "y = 2  # noqa",
        "z = 3  # noqa: F841",
    ]
    result = parse_noqa(code)
    assert result == {
        "V104": {1},
        "all": {3},
        "V107": {4},
    }

    # Test with codes that are not in the mapping
    code = ["x = 1  # noqa: E501"]
    result = parse_noqa(code)
    assert result == {"E501": {1}}

    # Test with spaces between codes
    code = ["x = 1  # noqa: F401 W123"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "W123": {1}}

    # Test with no noqa comments
    code = ["x = 1", "y = 2"]
    result = parse_noqa(code)
    assert result == {}

    # Test empty code
    code = []
    result = parse_noqa(code)
    assert result == {}
```


# LLM-generated content at query #2
#--------------------------

```python
def test_parse_noqa():
    # Test basic noqa comment without codes
    code = [
        "import unused_module  # noqa",
        "x = 1",
        "y = 2  # noqa: F401",
        "z = 3  # noqa: F841",
        "w = 4  # noqa: F401,F841",
        "v = 5  # NoQA: F401",
        "u = 6  # noqa: E501",
    ]
    result = parse_noqa(code)
    
    # Line 1 should be in "all" category
    assert 1 in result["all"]
    
    # Line 3 should be in "F401" (mapped to V104)
    assert 3 in result["V104"]
    assert 3 not in result["all"]
    
    # Line 4 should be in "F841" (mapped to V107)
    assert 4 in result["V107"]
    assert 4 not in result["all"]
    
    # Line 5 should be in both V104 and V107
    assert 5 in result["V104"]
    assert 5 in result["V107"]
    
    # Line 6 should be in V104 (case-insensitive NoQA)
    assert 6 in result["V104"]
    
    # Line 7 should have E501 code
    assert 7 in result["E501"]
    
    # Test no noqa comments
    code_no_noqa = [
        "import module",
        "x = 1",
    ]
    result_empty = parse_noqa(code_no_noqa)
    assert result_empty == defaultdict(set)
    
    # Test multiple codes with spaces and commas
    code_mixed = [
        "a = 1  # noqa: F401 F841",
        "b = 2  # noqa: F401,F841",
        "c = 3  # noqa: F401, F841",
    ]
    result_mixed = parse_noqa(code_mixed)
    
    for line_num in [1, 2, 3]:
        assert line_num in result_mixed["V104"]
        assert line_num in result_mixed["V107"]
    
    # Test code with no line numbers
    code_empty = []
    result_empty_code = parse_noqa(code_empty)
    assert result_empty_code == defaultdict(set)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_parse_noqa():
    # Test with no noqa comments
    code = ["import os", "def foo():", "    return 1"]
    result = parse_noqa(code)
    assert result == defaultdict(set)

    # Test with simple noqa comment
    code = ["import os  # noqa"]
    result = parse_noqa(code)
    assert result["all"] == {1}

    # Test with specific error code
    code = ["import os  # noqa: F401"]
    result = parse_noqa(code)
    assert result["V104"] == {1}

    # Test with multiple error codes
    code = ["import os  # noqa: F401,F841"]
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {1}

    # Test with uppercase NoQA
    code = ["import os  # NoQA: F401"]
    result = parse_noqa(code)
    assert result["V104"] == {1}

    # Test with multiple lines
    code = ["import os  # noqa", "x = 1  # noqa: F841", "y = 2"]
    result = parse_noqa(code)
    assert result["all"] == {1}
    assert result["V107"] == {2}

    # Test with spaces after colon
    code = ["import os  # noqa: F401, F841"]
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {1}

    # Test with unknown error code (not in map)
    code = ["import os  # noqa: E501"]
    result = parse_noqa(code)
    assert result["E501"] == {1}

    # Test with comma and space separated codes
    code = ["import os  # noqa: F401, F841"]
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {1}
```


# LLM-generated content at query #4
#--------------------------

```python
def test_ignore_line():
    # Test case 1: Line with specific error code
    code = ["x = 1  # noqa: F401"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 1, "F401") == False

    # Test case 2: Line with "all" noqa
    code = ["x = 1  # noqa"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 1, "F401") == True
    assert ignore_line(noqa_lines, 1, "E123") == True

    # Test case 3: Multiple error codes on same line
    code = ["x = 1  # noqa: F401, F841"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 1, "V107") == True
    assert ignore_line(noqa_lines, 1, "E123") == False

    # Test case 4: Line without noqa
    code = ["x = 1"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == False
    assert ignore_line(noqa_lines, 1, "all") == False

    # Test case 5: Multiple lines with different noqa annotations
    code = [
        "import os  # noqa: F401",
        "y = 2  # noqa",
        "z = 3",
    ]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 2, "V104") == True
    assert ignore_line(noqa_lines, 2, "E123") == True
    assert ignore_line(noqa_lines, 3, "V104") == False

    # Test case 6: Case-insensitive noqa
    code = ["x = 1  # NoQA: F401"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True

    # Test case 7: Error code with whitespace variations
    code = ["x = 1  # noqa: F401, F841,W451"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 1, "V107") == True
    assert ignore_line(noqa_lines, 1, "W451") == True

    # Test case 8: Non-mapped error code remains as-is
    code = ["x = 1  # noqa: E123"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E123") == True
    assert ignore_line(noqa_lines, 1, "V104") == False
```


# LLM-generated content at query #5
#--------------------------

```python
def test_parse_noqa():
    # Test with no noqa comments
    code = ["x = 1", "y = 2"]
    result = parse_noqa(code)
    assert result == {}

    # Test with plain # noqa
    code = ["x = 1  # noqa", "y = 2"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test with specific error codes
    code = ["x = 1  # noqa: F401", "y = 2  # noqa: F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {2}}

    # Test with multiple codes on one line
    code = ["x = 1  # noqa: F401,F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test with spaces after commas
    code = ["x = 1  # noqa: F401, F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test with case-insensitive noqa
    code = ["x = 1  # NoQA: F401"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}

    # Test with unknown error codes (not in map)
    code = ["x = 1  # noqa: E501"]
    result = parse_noqa(code)
    assert result == {"E501": {1}}

    # Test multiple lines with same code
    code = ["x = 1  # noqa: F401", "y = 2  # noqa: F401"]
    result = parse_noqa(code)
    assert result == {"V104": {1, 2}}

    # Test mixed codes on different lines
    code = ["x = 1  # noqa: F401", "y = 2  # noqa: F841", "z = 3"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {2}}

    # Test with whitespace around noqa
    code = ["x = 1  #  noqa", "y = 2  #noqa"]
    result = parse_noqa(code)
    assert result == {"all": {1, 2}}

    # Test empty code list
    code = []
    result = parse_noqa(code)
    assert result == {}
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_parse_noqa():
    # Test with no noqa comments
    code = [
        "import os",
        "import sys",
        "print('hello')"
    ]
    result = parse_noqa(code)
    assert result == {}

    # Test with simple # noqa on specific line
    code = [
        "import os  # noqa",
        "import sys",
        "print('hello')"
    ]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test with specific error code
    code = [
        "import os  # noqa: F401",
        "import sys",
        "print('hello')"
    ]
    result = parse_noqa(code)
    assert result == {"V104": {1}}  # F401 maps to V104

    # Test with multiple error codes
    code = [
        "import os  # noqa: F401,F841",
        "import sys",
        "var = 1  # noqa: F841"
    ]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1, 3}}

    # Test with case insensitive noqa
    code = [
        "import os  # NoQA: F401",
        "print('hello')"
    ]
    result = parse_noqa(code)
    assert result == {"V104": {1}}

    # Test with whitespace variations
    code = [
        "import os  # noqa: F401 , F841",
        "print('hello')"
    ]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test with multiple lines having same error code
    code = [
        "import os  # noqa: F401",
        "import sys  # noqa: F401",
        "print('hello')"
    ]
    result = parse_noqa(code)
    assert result == {"V104": {1, 2}}

    # Test with unknown error code (not in mapping)
    code = [
        "import os  # noqa: E501",
        "print('hello')"
    ]
    result = parse_noqa(code)
    assert result == {"E501": {1}}

    # Test with mixed known and unknown codes
    code = [
        "import os  # noqa: F401,E501",
        "print('hello')"
    ]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "E501": {1}}

    # Test with noqa on multiple lines
    code = [
        "import os  # noqa",
        "import sys",
        "import json  # noqa",
        "print('hello')"
    ]
    result = parse_noqa(code)
    assert result == {"all": {1, 3}}

    # Test with complex code including other comments
    code = [
        "# This is a comment",
        "import os  # noqa: F401  This is a comment",
        "x = 1  # noqa",
        "y = 2"
    ]
    result = parse_noqa(code)
    assert result == {"V104": {2}, "all": {3}}
```


# LLM-generated content at query #2
#--------------------------

```python
def test_parse_noqa():
    # Test basic noqa without codes
    code = ["x = 1  # noqa", "y = 2"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test with specific codes
    code = ["import os  # noqa: F401", "import sys  # noqa: W123"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "W123": {2}}

    # Test with multiple codes on same line
    code = ["x = 1  # noqa: E123,W451,F921"]
    result = parse_noqa(code)
    assert result == {"E123": {1}, "W451": {1}, "F921": {1}}

    # Test case insensitivity
    code = ["x = 1  # NoQA: E123"]
    result = parse_noqa(code)
    assert result == {"E123": {1}}

    # Test with spaces in codes
    code = ["x = 1  # noqa: E123, W451"]
    result = parse_noqa(code)
    assert result == {"E123": {1}, "W451": {1}}

    # Test code mapping
    code = ["import unused  # noqa: F401", "y = 1  # noqa: F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {2}}

    # Test multiple lines with same error code
    code = ["x = 1  # noqa: E123", "y = 2  # noqa: E123"]
    result = parse_noqa(code)
    assert result == {"E123": {1, 2}}

    # Test no noqa comments
    code = ["x = 1", "y = 2"]
    result = parse_noqa(code)
    assert result == {}

    # Test empty code
    code = []
    result = parse_noqa(code)
    assert result == {}
```


# LLM-generated content at query #3
#--------------------------

```python
def test_parse_noqa():
    # Test empty code
    code = []
    result = parse_noqa(code)
    assert result == {}

    # Test no noqa comments
    code = ["x = 1", "y = 2"]
    result = parse_noqa(code)
    assert result == {}

    # Test basic noqa without codes
    code = ["x = 1  # noqa"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test noqa with specific code
    code = ["x = 1  # noqa: F401"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}  # F401 maps to V104

    # Test noqa with multiple codes
    code = ["x = 1  # noqa: F401,F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test noqa with code not in map
    code = ["x = 1  # noqa: E123"]
    result = parse_noqa(code)
    assert result == {"E123": {1}}

    # Test noqa with case variations
    code = ["x = 1  # NoQA: E123"]
    result = parse_noqa(code)
    assert result == {"E123": {1}}

    # Test multiple lines with noqa
    code = ["x = 1  # noqa", "y = 2  # noqa: F401", "z = 3"]
    result = parse_noqa(code)
    assert result == {"all": {1}, "V104": {2}}

    # Test noqa with codes separated by spaces and commas
    code = ["x = 1  # noqa: F401,F841 E123"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}, "E123": {1}}

    # Test same code on multiple lines
    code = ["x = 1  # noqa: E123", "y = 2  # noqa: E123"]
    result = parse_noqa(code)
    assert result == {"E123": {1, 2}}

    # Test noqa with no space after colon
    code = ["x = 1  # noqa:E123"]
    result = parse_noqa(code)
    assert result == {"E123": {1}}

    # Test multiple noqa on same line
    code = ["x = 1  # noqa: E123  # noqa: F401"]
    result = parse_noqa(code)
    assert result == {"E123": {1}, "V104": {1}}

    # Test with codes that have different formats
    code = ["x = 1  # noqa: E123, W451, F921"]
    result = parse_noqa(code)
    assert result == {"E123": {1}, "W451": {1}, "F921": {1}}
```


# LLM-generated content at query #4
#--------------------------

```python
def test_ignore_line():
    # Test case 1: Line with specific error code match
    code = ["x = 1  # noqa: F401"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 1, "F401") == False
    
    # Test case 2: Line with "all" catch-all
    code = ["y = 2  # noqa"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 1, "V107") == True
    assert ignore_line(noqa_lines, 1, "AnyCode") == True
    
    # Test case 3: Line not annotated
    code = ["z = 3"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == False
    
    # Test case 4: Multiple error codes on same line
    code = ["a = 1  # noqa: F401, F841"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 1, "V107") == True
    assert ignore_line(noqa_lines, 1, "F401") == False
    assert ignore_line(noqa_lines, 1, "E123") == False
    
    # Test case 5: Different lines with different codes
    code = [
        "import os  # noqa: F401",
        "unused_var = 5  # noqa: F841",
    ]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 1, "V107") == False
    assert ignore_line(noqa_lines, 2, "V107") == True
    assert ignore_line(noqa_lines, 2, "V104") == False
    
    # Test case 6: Case-insensitive noqa
    code = ["import sys  # NoQA: F401"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    
    # Test case 7: Line number not in any set
    code = ["x = 1  # noqa: F401"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 2, "V104") == False
    
    # Test case 8: Whitespace in codes
    code = ["x = 1  # noqa: F401 , F841"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 1, "V107") == True
    
    # Test case 9: Empty lines and edge cases
    code = ["", "x = 1  # noqa", ""]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == False
    assert ignore_line(noqa_lines, 2, "V104") == True
    assert ignore_line(noqa_lines, 3, "V104") == False
    
    # Test case 10: Multiple lines with same code
    code = ["x = 1  # noqa: F401", "y = 2  # noqa: F401"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 2, "V104") == True
```


# LLM-generated content at query #5
#--------------------------

```python
def test_parse_noqa():
    # Test with no noqa comments
    code = ["x = 1\n", "y = 2\n"]
    result = parse_noqa(code)
    assert len(result) == 0

    # Test with generic noqa comment
    code = ["x = 1  # noqa\n", "y = 2\n"]
    result = parse_noqa(code)
    assert result["all"] == {1}

    # Test with specific error codes
    code = ["x = 1  # noqa: F401\n", "y = 2  # noqa: F841\n"]
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {2}

    # Test with multiple codes in one line
    code = ["x = 1  # noqa: F401,F841\n", "y = 2\n"]
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {1}

    # Test with codes separated by spaces
    code = ["x = 1  # noqa: F401 F841\n", "y = 2\n"]
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {1}

    # Test with unknown code (not in map)
    code = ["x = 1  # noqa: E501\n", "y = 2\n"]
    result = parse_noqa(code)
    assert result["E501"] == {1}

    # Test with mixed known and unknown codes
    code = ["x = 1  # noqa: F401,E501\n", "y = 2\n"]
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["E501"] == {1}

    # Test with multiline codes
    code = ["x = 1  # noqa: F401,\n", "    F841\n", "y = 2\n"]
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {1}

    # Test with case-insensitive noqa
    code = ["x = 1  # NoQA: F401\n", "y = 2\n"]
    result = parse_noqa(code)
    assert result["V104"] == {1}

    # Test with same code on multiple lines
    code = ["x = 1  # noqa: F401\n", "y = 2  # noqa: F401\n"]
    result = parse_noqa(code)
    assert result["V104"] == {1, 2}

    # Test with mixed codes and "all"
    code = ["x = 1  # noqa\n", "y = 2  # noqa: F401\n"]
    result = parse_noqa(code)
    assert result["all"] == {1}
    assert result["V104"] == {2}
```


