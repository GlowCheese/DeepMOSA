####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ignore_line():
    # Test 1: Line is in noqa_lines for specific error code
    noqa_lines = {"V104": {1, 3, 5}, "all": set()}
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 2, "V104") == False
    
    # Test 2: Line is in "all" category
    noqa_lines = {"V104": {1}, "all": {2, 4}}
    assert ignore_line(noqa_lines, 2, "V107") == True
    assert ignore_line(noqa_lines, 3, "V107") == False
    
    # Test 3: Line is in both specific error code and "all"
    noqa_lines = {"V104": {1}, "all": {1}}
    assert ignore_line(noqa_lines, 1, "V104") == True
    
    # Test 4: Line is not in any category
    noqa_lines = {"V104": {1}, "all": set()}
    assert ignore_line(noqa_lines, 2, "V104") == False
    assert ignore_line(noqa_lines, 1, "V107") == False
    
    # Test 5: Empty noqa_lines
    noqa_lines = {"all": set()}
    assert ignore_line(noqa_lines, 1, "F401") == False
    
    # Test 6: Only "all" category with multiple lines
    noqa_lines = {"all": {1, 2, 3}}
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 4, "V104") == False
    
    # Test 7: Multiple error codes
    noqa_lines = {"V104": {1}, "V107": {2}, "all": set()}
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 2, "V107") == True
    assert ignore_line(noqa_lines, 1, "V107") == False


# LLM-generated content at query #2
#--------------------------

```python
def test_parse_noqa():
    # Test empty code
    assert parse_noqa("") == defaultdict(set)
    
    # Test code without noqa comments
    code = "x = 1\ny = 2\nz = 3"
    result = parse_noqa(code)
    assert len(result) == 0
    
    # Test simple noqa without specific error codes
    code = "import os  # noqa"
    result = parse_noqa(code)
    assert "all" in result
    assert 1 in result["all"]
    
    # Test noqa with single error code
    code = "import os  # noqa: F401"
    result = parse_noqa(code)
    assert "V104" in result  # Mapped from F401
    assert 1 in result["V104"]
    
    # Test noqa with multiple error codes
    code = "x = 1  # noqa: F401, F841"
    result = parse_noqa(code)
    assert "V104" in result
    assert "V107" in result
    assert 1 in result["V104"]
    assert 1 in result["V107"]
    
    # Test noqa with unknown error code (not in map)
    code = "x = 1  # noqa: E123"
    result = parse_noqa(code)
    assert "E123" in result
    assert 1 in result["E123"]
    
    # Test multiple lines with different noqa comments
    code = """import os  # noqa: F401
x = 1  # noqa
y = 2  # noqa: F841"""
    result = parse_noqa(code)
    assert "V104" in result
    assert "all" in result
    assert "V107" in result
    assert 1 in result["V104"]
    assert 2 in result["all"]
    assert 3 in result["V107"]
    
    # Test case-insensitive noqa
    code = "import os  # NoQA: F401"
    result = parse_noqa(code)
    assert "V104" in result
    assert 1 in result["V104"]
    
    # Test noqa with spaces in codes
    code = "x = 1  # noqa: F401, F841"
    result = parse_noqa(code)
    assert "V104" in result
    assert "V107" in result
    
    # Test noqa with different separator (comma or space)
    code = "x = 1  # noqa: E123 E124"
    result = parse_noqa(code)
    assert "E123" in result
    assert "E124" in result
    assert 1 in result["E123"]
    assert 1 in result["E124"]
    
    # Test multiple lines with same error code
    code = """import os  # noqa: F401
import sys  # noqa: F401"""
    result = parse_noqa(code)
    assert "V104" in result
    assert 1 in result["V104"]
    assert 2 in result["V104"]
```


# LLM-generated content at query #3
#--------------------------

```python
def test_parse_noqa():
    # Test 1: Line without any noqa comment
    code = ["x = 1\n", "y = 2\n"]
    result = parse_noqa(code)
    assert result == {}, f"Expected empty dict, got {result}"

    # Test 2: Line with simple noqa comment (no error codes)
    code = ["import os  # noqa\n"]
    result = parse_noqa(code)
    assert result == {"all": {1}}, f"Expected {{'all': {{1}}}}, got {result}"

    # Test 3: Line with noqa and specific error codes
    code = ["import os  # noqa: F401\n"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}, f"Expected {{'V104': {{1}}}}, got {result}"

    # Test 4: Line with noqa and multiple error codes
    code = ["x = 1  # noqa: F401, F841\n"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}, f"Expected {{'V104': {{1}}, 'V107': {{1}}}}, got {result}"

    # Test 5: Line with noqa and multiple error codes (space separated)
    code = ["x = 1  # noqa: F401 F841\n"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}, f"Expected {{'V104': {{1}}, 'V107': {{1}}}}, got {result}"

    # Test 6: Multiple lines with noqa comments
    code = [
        "import os  # noqa: F401\n",
        "y = 1  # noqa\n",
        "import sys  # noqa: F401\n"
    ]
    result = parse_noqa(code)
    assert result == {"V104": {1, 3}, "all": {2}}, f"Expected {{'V104': {{1, 3}}, 'all': {{2}}}}, got {result}"

    # Test 7: Line with noqa and error code that is not in the map
    code = ["x = 1  # noqa: E501\n"]
    result = parse_noqa(code)
    assert result == {"E501": {1}}, f"Expected {{'E501': {{1}}}}, got {result}"

    # Test 8: Case insensitive noqa
    code = ["import os  # NoQA: F401\n"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}, f"Expected {{'V104': {{1}}}}, got {result}"

    # Test 9: Empty code
    code = []
    result = parse_noqa(code)
    assert result == {}, f"Expected empty dict, got {result}"

    # Test 10: Line with noqa and mixed case error codes
    code = ["x = 1  # noqa: f401\n"]
    result = parse_noqa(code)
    assert result == {"f401": {1}}, f"Expected {{'f401': {{1}}}}, got {result}"
```


# LLM-generated content at query #4
#--------------------------

```python
def test_ignore_line():
    # Test 1: Line with specific error code noqa
    code = ["x = 1  # noqa: F401"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == True
    assert ignore_line(noqa_lines, 1, "V104") == True  # Mapped code
    
    # Test 2: Line with "all" noqa
    code = ["x = 1  # noqa"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == True
    assert ignore_line(noqa_lines, 1, "XYZ") == True
    
    # Test 3: Line without noqa
    code = ["x = 1"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == False
    
    # Test 4: Line with different error code
    code = ["x = 1  # noqa: E123"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == False
    assert ignore_line(noqa_lines, 1, "E123") == True
    
    # Test 5: Multiple lines with different noqa
    code = [
        "import os  # noqa: F401",
        "y = 1  # noqa",
        "z = 1",
    ]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == True
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 2, "F401") == True  # "all" category
    assert ignore_line(noqa_lines, 3, "F401") == False
    
    # Test 6: Line number not in noqa_lines
    code = ["x = 1  # noqa: F401"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 2, "F401") == False
    
    # Test 7: Empty noqa set
    noqa_lines = defaultdict(set)
    assert ignore_line(noqa_lines, 1, "F401") == False


# LLM-generated content at query #5
#--------------------------

```python
def test_parse_noqa():
    # Test empty code
    assert parse_noqa("") == defaultdict(set)
    
    # Test code without noqa comments
    code = "x = 1\ny = 2\nz = 3"
    result = parse_noqa(code)
    assert result == defaultdict(set)
    
    # Test single noqa without specific codes
    code = "import os  # noqa"
    result = parse_noqa(code)
    assert result["all"] == {1}
    
    # Test noqa with specific code
    code = "import os  # noqa: F401"
    result = parse_noqa(code)
    assert result["V104"] == {1}  # F401 maps to V104
    
    # Test noqa with multiple codes
    code = "import os  # noqa: F401,F841"
    result = parse_noqa(code)
    assert result["V104"] == {1}  # F401 maps to V104
    assert result["V107"] == {1}  # F841 maps to V107
    
    # Test noqa with case-insensitive matching
    code = "import os  # NOQA: F401"
    result = parse_noqa(code)
    assert result["V104"] == {1}
    
    code = "import os  # NoQA: F401"
    result = parse_noqa(code)
    assert result["V104"] == {1}
    
    # Test multiple lines with noqa
    code = "import os  # noqa: F401\nx = 1  # noqa\nimport sys  # noqa: F841"
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["all"] == {2}
    assert result["V107"] == {3}
    
    # Test error code that doesn't map to anything
    code = "x = 1  # noqa: E501"
    result = parse_noqa(code)
    assert result["E501"] == {1}
    assert "all" not in result
    
    # Test multiple codes on same line
    code = "x = 1  # noqa: F401,F841"
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {1}
    
    # Test codes with spaces after comma
    code = "x = 1  # noqa: F401, F841"
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {1}
    
    # Test code with mixed format
    code = "x = 1  # noqa: F401,F841\nimport os  # noqa"
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {1}
    assert result["all"] == {2}
    
    # Test line numbers are correct
    code = "line1  # noqa\nline2\nline3  # noqa: F401"
    result = parse_noqa(code)
    assert result["all"] == {1}
    assert result["V104"] == {3}
    assert 2 not in result["all"]
    assert 2 not in result["V104"]
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ignore_line():
    # Test basic noqa for specific error code
    code = "x = 1  # noqa: F401"
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == True
    assert ignore_line(noqa_lines, 1, "V104") == True  # Mapped from F401
    
    # Test noqa for "all" error codes
    code = "x = 1  # noqa"
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == True
    assert ignore_line(noqa_lines, 1, "E123") == True
    
    # Test noqa for different line than the one with annotation
    code = "x = 1\n# noqa: F401"
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == False
    assert ignore_line(noqa_lines, 2, "F401") == True
    
    # Test noqa with multiple error codes
    code = "x = 1  # noqa: F401, E123"
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == True
    assert ignore_line(noqa_lines, 1, "E123") == True
    assert ignore_line(noqa_lines, 1, "W451") == False
    
    # Test noqa case insensitivity
    code = "x = 1  # NoQA: F401"
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == True
    
    # Test when line is not in any noqa list
    code = "x = 1"
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == False
    
    # Test error code mapping
    code = "x = 1  # noqa: F841"
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F841") == True
    assert ignore_line(noqa_lines, 1, "V107") == True  # Mapped from F841
    
    # Test multiple lines with different noqa
    code = """x = 1  # noqa: F401
y = 2  # noqa: E123
z = 3  # noqa"""
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == True
    assert ignore_line(noqa_lines, 1, "E123") == False
    assert ignore_line(noqa_lines, 2, "F401") == False
    assert ignore_line(noqa_lines, 2, "E123") == True
    assert ignore_line(noqa_lines, 3, "F401") == True  # "all" covers everything
    assert ignore_line(noqa_lines, 3, "E123") == True  # "all" covers everything
    
    # Test with noqa: all
    code = "x = 1  # noqa: all"
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == True
    assert ignore_line(noqa_lines, 1, "E123") == True
    assert ignore_line(noqa_lines, 1, "W451") == True
    
    # Test empty code
    code = ""
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "F401") == False
```


# LLM-generated content at query #2
#--------------------------

```python
def test_ignore_line():
    # Test basic noqa comment
    code = ["x = 1  # noqa"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E123") == True
    
    # Test noqa: all
    code = ["x = 1  # noqa: all"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E123") == True
    
    # Test specific error code
    code = ["x = 1  # noqa: E123"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E123") == True
    assert ignore_line(noqa_lines, 1, "W456") == False
    
    # Test multiple error codes
    code = ["x = 1  # noqa: E123,W456"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E123") == True
    assert ignore_line(noqa_lines, 1, "W456") == True
    assert ignore_line(noqa_lines, 1, "F789") == False
    
    # Test line without noqa
    code = ["x = 1"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E123") == False
    
    # Test different line number
    code = ["", "y = 2  # noqa: E456"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E456") == False
    assert ignore_line(noqa_lines, 2, "E456") == True
    
    # Test noqa mapping (F401 -> V104)
    code = ["import os  # noqa: F401"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "V104") == True
    assert ignore_line(noqa_lines, 1, "F401") == False
    
    # Test case insensitivity
    code = ["x = 1  # NoQA"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E123") == True
    
    # Test empty code
    code = []
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E123") == False
    
    # Test multiple lines with noqa
    code = ["x = 1  # noqa: E111", "y = 2  # noqa: E222"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E111") == True
    assert ignore_line(noqa_lines, 1, "E222") == False
    assert ignore_line(noqa_lines, 2, "E222") == True
    assert ignore_line(noqa_lines, 2, "E111") == False
    
    # Test noqa with all and specific codes
    code = ["x = 1  # noqa: all", "y = 2  # noqa: E333"]
    noqa_lines = parse_noqa(code)
    assert ignore_line(noqa_lines, 1, "E111") == True
    assert ignore_line(noqa_lines, 1, "E333") == True
    assert ignore_line(noqa_lines, 2, "E333") == True
    assert ignore_line(noqa_lines, 2, "E111") == False


# LLM-generated content at query #3
#--------------------------

```python
def test_parse_noqa():
    # Test 1: No noqa comments
    code = ["x = 1", "y = 2"]
    result = parse_noqa(code)
    assert result == {}, f"Expected empty dict, got {result}"

    # Test 2: Simple noqa comment without codes
    code = ["x = 1  # noqa"]
    result = parse_noqa(code)
    assert result == {"all": {1}}, f"Expected all on line 1, got {result}"

    # Test 3: Noqa with specific code
    code = ["x = 1  # noqa: F401"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}, f"Expected V104 on line 1, got {result}"

    # Test 4: Noqa with multiple codes
    code = ["x = 1  # noqa: F401, E501"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "E501": {1}}, f"Expected V104 and E501 on line 1, got {result}"

    # Test 5: Noqa with multiple codes using different separators
    code = ["x = 1  # noqa: F401 E501"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "E501": {1}}, f"Expected V104 and E501 on line 1, got {result}"

    # Test 6: Multiple lines with noqa
    code = ["x = 1  # noqa: F401", "y = 2  # noqa: F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {2}}, f"Expected V104 on line 1 and V107 on line 2, got {result}"

    # Test 7: Multiple lines with the same noqa code
    code = ["x = 1  # noqa: F401", "y = 2  # noqa: F401"]
    result = parse_noqa(code)
    assert result == {"V104": {1, 2}}, f"Expected V104 on lines 1 and 2, got {result}"

    # Test 8: Case insensitive noqa
    code = ["x = 1  # NoQA: F401"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}, f"Expected V104 on line 1, got {result}"

    # Test 9: Noqa with unknown code (not in NOQA_CODE_MAP)
    code = ["x = 1  # noqa: W123"]
    result = parse_noqa(code)
    assert result == {"W123": {1}}, f"Expected W123 on line 1, got {result}"

    # Test 10: Mixed cases - some codes mapped, some not
    code = ["x = 1  # noqa: F401, W123"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "W123": {1}}, f"Expected V104 and W123 on line 1, got {result}"

    # Test 11: Noqa with "all" and specific codes on different lines
    code = ["x = 1  # noqa", "y = 2  # noqa: F401"]
    result = parse_noqa(code)
    assert result == {"all": {1}, "V104": {2}}, f"Expected all on line 1 and V104 on line 2, got {result}"

    # Test 12: Empty code
    code = []
    result = parse_noqa(code)
    assert result == {}, f"Expected empty dict for empty code, got {result}"


# LLM-generated content at query #4
#--------------------------

```python
def test_parse_noqa():
    # Test 1: Empty code
    code = []
    result = parse_noqa(code)
    assert result == defaultdict(set, {})

    # Test 2: No noqa comments
    code = ["x = 1", "y = 2"]
    result = parse_noqa(code)
    assert result == defaultdict(set, {})

    # Test 3: Simple noqa without codes
    code = ["x = 1  # noqa"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test 4: Noqa with specific code
    code = ["x = 1  # noqa: F401"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}  # F401 maps to V104

    # Test 5: Noqa with multiple codes
    code = ["x = 1  # noqa: F401, F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test 6: Noqa with unmapped code
    code = ["x = 1  # noqa: E123"]
    result = parse_noqa(code)
    assert result == {"E123": {1}}

    # Test 7: Multiple lines with noqa
    code = ["x = 1  # noqa: F401", "y = 2  # noqa: F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {2}}

    # Test 8: Same code on multiple lines
    code = ["x = 1  # noqa: F401", "y = 2  # noqa: F401"]
    result = parse_noqa(code)
    assert result == {"V104": {1, 2}}

    # Test 9: Case insensitive noqa
    code = ["x = 1  # NoQA: F401"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}

    # Test 10: Noqa with all codes
    code = ["x = 1  # noqa: all"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test 11: Mixed noqa types
    code = ["x = 1  # noqa", "y = 2  # noqa: F401"]
    result = parse_noqa(code)
    assert result == {"all": {1}, "V104": {2}}

    # Test 12: Noqa with extra whitespace
    code = ["x = 1  # noqa: F401 , F841 "]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test 13: Multiple codes on same line
    code = ["x = 1  # noqa: E123, F401"]
    result = parse_noqa(code)
    assert result == {"E123": {1}, "V104": {1}}


# LLM-generated content at query #5
#--------------------------

```python
def test_parse_noqa():
    # Test 1: Empty code
    code = []
    result = parse_noqa(code)
    assert result == {}

    # Test 2: No noqa comments
    code = [
        "import os",
        "x = 1",
        "print(x)"
    ]
    result = parse_noqa(code)
    assert result == {}

    # Test 3: Simple noqa without codes
    code = [
        "import os  # noqa",
        "x = 1",
    ]
    result = parse_noqa(code)
    assert "all" in result
    assert 1 in result["all"]

    # Test 4: noqa with specific code
    code = [
        "import os  # noqa: F401",
        "x = 1",
    ]
    result = parse_noqa(code)
    assert "V104" in result  # F401 maps to V104
    assert 1 in result["V104"]

    # Test 5: Multiple codes in one comment
    code = [
        "import os  # noqa: F401, F841",
    ]
    result = parse_noqa(code)
    assert "V104" in result  # F401 maps to V104
    assert "V107" in result  # F841 maps to V107
    assert 1 in result["V104"]
    assert 1 in result["V107"]

    # Test 6: Multiple lines with noqa
    code = [
        "import os  # noqa: F401",
        "x = 1",
        "y = 2  # noqa: F841",
    ]
    result = parse_noqa(code)
    assert "V104" in result
    assert "V107" in result
    assert 1 in result["V104"]
    assert 3 in result["V107"]

    # Test 7: Case insensitive noqa
    code = [
        "import os  # NOQA: f401",
    ]
    result = parse_noqa(code)
    assert "V104" in result
    assert 1 in result["V104"]

    # Test 8: Noqa with "all" category
    code = [
        "import os  # noqa: all",
    ]
    result = parse_noqa(code)
    assert "all" in result
    assert 1 in result["all"]

    # Test 9: Multiple noqa on same line
    code = [
        "import os  # noqa: F401  # noqa: F841",
    ]
    result = parse_noqa(code)
    assert "V104" in result
    assert "V107" in result
    assert 1 in result["V104"]
    assert 1 in result["V107"]

    # Test 10: Code without mapping (not in NOQA_CODE_MAP)
    code = [
        "x = 1  # noqa: E999",
    ]
    result = parse_noqa(code)
    assert "E999" in result
    assert 1 in result["E999"]

    # Test 11: Mixed mapped and unmapped codes
    code = [
        "import os  # noqa: F401, E999",
    ]
    result = parse_noqa(code)
    assert "V104" in result
    assert "E999" in result
    assert 1 in result["V104"]
    assert 1 in result["E999"]
```


