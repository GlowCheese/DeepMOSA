####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Case 1: Empty input
    assert parse_noqa([]) == {}

    # Case 2: Simple # noqa without specific codes (should map to 'all')
    code_simple = ["import os  # noqa"]
    result_simple = parse_noqa(code_simple)
    assert "all" in result_simple
    assert 1 in result_simple["all"]

    # Case 3: # noqa with specific codes and case insensitivity
    code_codes = [
        "import sys  # noqa: E123",
        "import os  # NOQA: W451, F921",
        "import math  # noqa:F841"  # Testing mapping F841 -> V107
    ]
    result_codes = parse_noqa(code_codes)
    assert 1 in result_codes["E123"]
    assert 2 in result_codes["W451"]
    assert 2 in result_codes["F921"]
    assert 3 in result_codes["V107"]

    # Case 4: Multiple codes on different lines and comma spacing
    code_complex = [
        "import a  # noqa: E1, E2",
        "import b  # noqa: E2, E3",
        "import c  # noqa"
    ]
    result_complex = parse_noqa(code_complex)
    assert 1 in result_complex["E1"]
    assert 1 in result_complex["E2"]
    assert 2 in result_complex["E2"]
    assert 2 in result_complex["E3"]
    assert 3 in result_complex["all"]

    # Case 5: Line with no noqa should not be in the dict
    code_no_match = ["import os", "print('hello')"]
    assert parse_noqa(code_no_match) == {}

    # Case 6: Testing mapping for F401 -> V104
    code_mapping = ["import sys  # noqa: F401"]
    result_map = parse_noqa(code_mapping)
    assert "V104" in result_map
    assert 1 in result_map["V104"]

def test_ignore_line():
    noqa_lines = {
        "E123": {1},
        "all": {2, 5}
    }
    # Exact match
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Match via 'all'
    assert ignore_line(noqa_lines, 2, "F401") is True
    # No match
    assert ignore_line(noqa_lines, 3, "E123") is False
    # Line exists but for different error code
    assert ignore_line(noqa_lines, 1, "W451") is False
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test case 1: Empty input
    assert parse_noqa([]) == {}

    # Test case 2: Basic # noqa without codes (should map to 'all')
    code_basic = ["import os  # noqa"]
    result_basic = parse_noqa(code_basic)
    assert "all" in result_basic
    assert 1 in result_basic["all"]

    # Test case 3: # noqa with specific codes
    code_specific = ["import sys  # noqa: E123, W451"]
    result_specific = parse_noqa(code_specific)
    assert "E123" in result_specific
    assert "W451" in result_specific
    assert 1 in result_specific["E123"]
    assert 1 in result_specific["W451"]

    # Test case 4: Mapping of F-codes to V-codes via NOQA_CODE_MAP
    code_mapping = ["import os  # noqa: F401, F841"]
    result_mapping = parse_noqa(code_mapping)
    assert "V104" in result_mapping
    assert "V107" in result_mapping
    assert 1 in result_mapping["V104"]
    assert 1 in result_mapping["V107"]
    assert "F401" not in result_mapping

    # Test case 5: Multiple lines and mixed formats
    code_mixed = [
        "import math  # noqa",            # Line 1: all
        "import sys  # noqa: E123",       # Line 2: E123
        "import os  # NoQA: W451, F401",  # Line 3: W451, V104
        "print('hello')",                 # Line 4: nothing
    ]
    result_mixed = parse_noqa(code_mixed)
    assert 1 in result_mixed["all"]
    assert 2 in result_mixed["E123"]
    assert 3 in result_mixed["W451"]
    assert 3 in result_mixed["V104"]
    assert "F401" not in result_mixed
    assert len(result_mixed) == 4  # all, E123, W451, V104

    # Test case 6: Case insensitivity check
    code_caps = ["import os  # NOQA: e123"]
    result_caps = parse_noqa(code_caps)
    # Note: The regex uses IGNORECASE, but the split/strip logic preserves original casing from match group
    # Depending on how re.IGNORECASE interacts with the capture group content in the provided regex
    assert "e123" in result_caps or "E123" in result_caps

def test_ignore_line():
    noqa_lines = {
        "E123": {2},
        "all": {1, 5}
    }
    # Line is explicitly in code list
    assert ignore_line(noqa_lines, 2, "E123") is True
    # Line is in 'all' list
    assert ignore_line(noqa_lines, 1, "F401") is True
    # Line is not in code list and not in 'all'
    assert ignore_line(noqa_lines, 2, "W451") is False
    # Line is not in the file at all
    assert ignore_line(noqa_lines, 10, "E123") is False
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test case 1: Empty input
    assert parse_noqa([]) == {}

    # Test case 2: Standard # noqa (all)
    code1 = ["import os  # noqa"]
    result1 = parse_noqa(code1)
    assert result1["all"] == {1}

    # Test case 3: Specific error codes
    code2 = ["import sys  # noqa: E123, W451"]
    result2 = parse_noqa(code2)
    assert "E123" in result2
    assert "W451" in result2
    assert result2["E123"] == {1}
    assert result2["W451"] == {1}

    # Test case 4: Mapping of F codes to V codes
    code3 = ["import os  # noqa: F401, F841"]
    result3 = parse_noqa(code3)
    assert "V104" in result3
    assert "V107" in result3
    assert result3["V104"] == {1}
    assert result3["V107"] == {1}

    # Test case 5: Multiple lines and mixed formats
    code4 = [
        "import os  # noqa",           # Line 1: all
        "import sys  # noqa: E123",    # Line 2: E123
        "import math  # NoQA: F401",   # Line 3: V104 (mapped)
        "import re"                     # Line 4: no match
    ]
    result4 = parse_noqa(code4)
    assert result4["all"] == {1}
    assert result4["E123"] == {2}
    assert result4["V104"] == {3}
    assert len(result4) == 3

    # Test case 6: Case insensitivity and spacing
    code5 = ["import os  # NOQA:  E777  "]
    result5 = parse_noqa(code5)
    assert "E777" in result5
    assert result5["E777"] == {1}

def test_ignore_line():
    noqa_lines = {"E123": {1}, "all": {2}, "V104": {3}}
    
    # Match specific code
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Match 'all' category
    assert ignore_line(noqa_lines, 2, "E501") is True
    # No match for specific code
    assert ignore_line(noqa_lines, 3, "E123") is False
    # No match at all
    assert ignore_line(noqa_lines, 4, "E123") is False
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_ignore_line():
    # Test case 1: Line is explicitly ignored by a specific error code
    noqa_lines = {"E123": {5, 10}, "F401": {2}}
    assert ignore_line(noqa_lines, 5, "E123") is True
    assert ignore_line(noqa_lines, 10, "E123") is True
    assert ignore_line(noqa_lines, 2, "F401") is True

    # Test case 2: Line is ignored by the "all" category
    noqa_lines_with_all = {"all": {7}, "E123": {5}}
    assert ignore_line(noqa_lines_with_all, 7, "E123") is True
    assert ignore_line(noqa_lines_with_all, 7, "F841") is True

    # Test case 3: Line is not ignored by the specific error code and not in "all"
    noqa_lines = {"E123": {5}, "all": {1}}
    assert ignore_line(noqa_lines, 6, "E123") is False
    assert ignore_line(noqa_lines, 2, "E123") is False

    # Test case 4: Error code does not exist in the dictionary at all
    noqa_lines = {"all": {1}}
    assert ignore_line(noqa_lines, 1, "NONEXISTENT") is True
    assert ignore_line(noqa_lines, 2, "NONEXISTENT") is False

    # Test case 5: Empty dictionary
    assert ignore_line({}, 1, "E123") is False
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test case 1: Empty input
    assert parse_noqa([]) == {}

    # Test case 2: Simple # noqa without specific codes (should map to 'all')
    code1 = ["import os  # noqa"]
    result1 = parse_noqa(code1)
    assert "all" in result1
    assert 1 in result1["all"]

    # Test case 3: # noqa with specific codes (case insensitive, space handling)
    code2 = ["import sys  # noqa: E123", "import math  # noqa: W451, F921"]
    result2 = parse_noqa(code2)
    assert 1 in result2["E123"]
    assert 2 in result2["W451"]
    assert 2 in result2["F921"]

    # Test case 4: Mapping of flake8 codes to custom codes (F401 -> V104)
    code3 = ["import os  # noqa: F401"]
    result3 = parse_noqa(code3)
    assert "V104" in result3
    assert 1 in result3["V104"]
    assert "F401" not in result3

    # Test case 5: Mixed lines (some with noqa, some without)
    code4 = [
        "import os",
        "import sys  # noqa: E123",
        "print('hello')",
        "import math  # noqa"
    ]
    result4 = parse_noqa(code4)
    assert 2 in result4["E123"]
    assert 4 in result4["all"]
    assert "all" in result4
    assert 1 not in result4["all"]

    # Test case 6: Multiple codes on one line and trailing commas/spaces
    code5 = ["import os  # noqa: E123,  E456 , F841"]
    result5 = parse_noqa(code5)
    assert 1 in result5["E121"] # Check mapping logic if any, but here checking split
    assert 1 in result5["E123"]
    assert 1 in result5["E456"]
    assert 1 in result5["V107"]  # F841 maps to V107

def test_ignore_line():
    noqa_lines = {"E123": {1, 2}, "all": {3}, "V104": {4}}
    
    # Test exact match
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Test 'all' catch-all
    assert ignore_line(noqa_lines, 3, "E999") is True
    # Test no match
    assert ignore_line(noqa_lines, 5, "E123") is False
    # Test line exists in 'all' but not specific error code dict
    assert ignore_line(noqa_lines, 3, "F401") is True
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test empty input
    assert parse_noqa([]) == {}

    # Test basic # noqa without codes (should map to 'all')
    code1 = ["import os  # noqa"]
    result1 = parse_noqa(code1)
    assert result1["all"] == {1}

    # Test # noqa with single code
    code2 = ["import os  # noqa: E123"]
    result2 = parse_noqa(code2)
    assert "E123" in result2
    assert result2["E123"] == {1}

    # Test # noqa with multiple codes (comma separated)
    code3 = ["import os  # noqa: E123, W451"]
    result3 = parse_noqa(code3)
    assert result3["E123"] == {1}
    assert result3["W451"] == {1}

    # Test case insensitivity
    code4 = ["import os  # NOQA: E123"]
    result4 = parse_noqa(code4)
    assert result4["E123"] == {1}

    # Test code mapping (F401 -> V104, F841 -> V107)
    code5 = ["import os  # noqa: F401, F841"]
    result5 = parse_noqa(code5)
    assert "V104" in result5
    assert "V107" in result5
    assert result5["V106"] == set() # Should not exist

    # Test multiple lines and mixed content
    code6 = [
        "import os  # noqa: E123",      # Line 1: E123
        "import sys  # noqa",           # Line 2: all
        "import math  # noqa: F401",    # Line 3: V104
        "print(x)  # No noqa here",     # Line 4: nothing
        "import re  # noqa: E123, F841" # Line 5: E123, V107
    ]
    result6 = parse_noqa(code6)
    assert result6["E123"] == {1, 5}
    assert result6["all"] == {2}
    assert result6["V104"] == {3}
    assert result6["V107"] == {5}

    # Test ignore_line utility logic within the context of parsed results
    noqa_lines = result6
    assert ignore_line(noqa_lines, 1, "E123") is True
    assert ignore_line(noqa_lines, 5, "E123") is True
    assert ignore_line(noqa_lines, 2, "any_code") is True  # because line 2 is 'all'
    assert ignore_line(noqa_lines, 4, "E123") is False     # Line 4 has no noqa
    assert ignore_line(noqa_lines, 1, "V104") is False     # Line 1 is E123, not V104
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test case 1: Empty input
    assert parse_noqa([]) == {}

    # Test case 2: Simple # noqa without codes (should map to 'all')
    code2 = ["import os  # noqa"]
    expected2 = {"all": {1}}
    assert parse_noqa(code2) == expected2

    # Test case 3: # noqa with single code
    code3 = ["import os  # noqa: E123"]
    expected3 = {"E123": {1}}
    assert parse_noqa(code3) == expected3

    # Test case 4: # noqa with multiple codes (comma separated)
    code4 = ["import os, sys  # noqa: E123, W451"]
    expected4 = {"E123": {1}, "W451": {1}}
    assert parse_noqa(code4) == expected4

    # Test case 5: Case insensitivity (NoQA)
    code5 = ["import os  # NoQA: E123"]
    expected5 = {"E123": {1}}
    assert parse_noqa(code5) == expected5

    # Test case 6: Code mapping (F401 -> V104, F841 -> V107)
    code6 = [
        "import os  # noqa: F401",
        "x = 1  # noqa: F841"
    ]
    expected6 = {"V104": {1}, "V107": {2}}
    assert parse_noqa(code6) == expected6

    # Test case 7: Multiple lines with mixed annotations
    code7 = [
        "import os  # noqa",                # line 1: all
        "import sys  # noqa: E123",         # line 2: E123
        "import math  # noqa: F401, E501",   # line 3: V104, E501
        "print(x)  # No error here"          # line 4: nothing
    ]
    expected7 = {
        "all": {1},
        "E123": {2},
        "V104": {3},
        "E501": {3}
    }
    assert parse_noqa(code7) == expected7

def test_ignore_line():
    noqa_lines = {"E123": {2}, "all": {1}, "V104": {3}}
    
    # Line in specific error category
    assert ignore_line(noqa_lines, 2, "E123") is True
    # Line in 'all' category
    assert ignore_line(noqa_lines, 1, "F888") is True
    # Line not annotated for specific error
    assert ignore_line(noqa_lines, 3, "E123") is False
    # Line not annotated at all
    assert ignore_line(noqa_lines, 4, "E123") is False
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test Case 1: Empty input
    assert parse_noqa([]) == {}

    # Test Case 2: Basic # noqa without specific codes (should map to 'all')
    code_basic = ["import os  # noqa"]
    result_basic = parse_noqa(code_basic)
    assert "all" in result_basic
    assert 1 in result_basic["all"]

    # Test Case 3: # noqa with specific codes
    code_specific = ["import sys  # noqa: E123, W451"]
    result_specific = parse_noqa(code_specific)
    assert "E123" in result_specific
    assert "W451" in result_specific
    assert 1 in result_specific["E123"]
    assert 1 in result_specific["W451"]

    # Test Case 4: Mapping of flake8 codes to custom codes (F401 -> V104)
    code_mapping = ["import os  # noqa: F401"]
    result_mapping = parse_noqa(code_mapping)
    assert "V104" in result_mapping
    assert "F401" not in result_mapping
    assert 1 in result_mapping["V104"]

    # Test Case 5: Multiple lines and multiple codes per line
    code_multi = [
        "import os  # noqa: E123",         # Line 1
        "import sys  # noqa",              # Line 2 (all)
        "import math  # noqa: F841, E501", # Line 3 (F841 -> V107)
        "print('hello') # no other match"  # Line 4
    ]
    result_multi = parse_noqa(code_multi)
    assert 1 in result_multi["E123"]
    assert 2 in result_multi["all"]
    assert 3 in result_multi["V107"]
    assert 3 in result_multi["E501"]
    assert 4 not in result_multi

    # Test Case 6: Case insensitivity and whitespace handling
    code_formatting = ["import os  # NOQA:  E123 ,  W451 "]
    result_formatting = parse_noqa(code_formatting)
    assert "E123" in result_formatting
    assert "W451" in result_formatting
    assert 1 in result_formatting["E123"]

def test_ignore_line():
    noqa_lines = {
        "E123": {1},
        "V104": {3},
        "all": {2}
    }
    # Matches specific code
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Matches 'all' category
    assert ignore_line(noqa_lines, 2, "F401") is True
    # Line exists but not for this error code
    assert ignore_line(noqa_lines, 3, "E123") is False
    # Line does not exist in dict at all
    assert ignore_line(noqa_lines, 5, "E123") is False
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_ignore_line():
    # Case 1: Line is explicitly ignored by a specific error code
    noqa_lines = {"E123": {10}, "F401": {5}}
    assert ignore_line(noqa_lines, 10, "E123") is True
    assert ignore_line(noqa_lines, 11, "E123") is False

    # Case 2: Line is ignored by the "all" category
    noqa_lines = {"all": {20}, "E123": {5}}
    assert ignore_line(noqa_lines, 20, "E123") is True
    assert ignore_line(noqa_lines, 20, "F401") is True

    # Case 3: Line is not ignored by specific code or "all"
    noqa_lines = {"E123": {10}, "all": {20}}
    assert ignore_line(noqa_lines, 15, "E123") is False
    assert ignore_line(noqa_lines, 15, "F401") is False

    # Case 4: Error code does not exist in the dictionary at all
    noqa_lines = {"all": {1}}
    assert ignore_line(noqa_lines, 2, "W291") is False

    # Case 5: Empty noqa_lines
    assert ignore_line({}, 1, "E123") is False
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test case 1: Empty input
    assert parse_noqa([]) == {}

    # Test case 2: Basic # noqa without specific codes (should map to 'all')
    code_all = ["import os  # noqa"]
    result_all = parse_noqa(code_all)
    assert "all" in result_all
    assert 1 in result_all["all"]

    # Test case 3: # noqa with single code (unmapped)
    code_single = ["import sys  # noqa: E123"]
    result_single = parse_noqa(code_single)
    assert "E123" in result_single
    assert 1 in result_single["E123"]

    # Test case 4: # noqa with multiple codes and spaces/commas
    code_multi = ["import os  # noqa: E123, W451, F921"]
    result_multi = parse_noqa(code_multi)
    assert "E123" in result_multi
    assert "W451" in result_multi
    assert "F921" in result_multi
    assert 1 in result_multi["E123"]
    assert 1 in result_multi["W451"]

    # Test case 5: Mapping of F-codes to V-codes
    code_mapping = ["import os  # noqa: F401, F841"]
    result_mapped = parse_noqa(code_mapping)
    assert "V104" in result_mapped
    assert "V107" in result_mapped
    assert "F401" not in result_mapped
    assert "F841" not in result_mapped

    # Test case 6: Case insensitivity
    code_caps = ["import os  # NOQA: e123"]
    result_caps = parse_noqa(code_caps)
    assert "E123" in result_caps # Note: regex captures group, but logic depends on string content. 
    # Based on regex definition r"(?P<codes>([A-Z]+[0-9]+...))", it expects uppercase.
    # If the regex is case insensitive (re.IGNORECASE), the match works, 
    # but the codes themselves are extracted as found in the string.

    # Test case 7: Multiple lines with different annotations
    code_multi_line = [
        "import os  # noqa: E123",      # Line 1
        "import sys  # noqa",           # Line 2
        "import math  # noqa: F401",    # Line 3 -> V104
        "import numpy  # noqa: W451"    # Line 4
    ]
    result_multi_line = parse_noqa(code_multi_line)
    assert 1 in result_multi_line["E123"]
    assert 2 in result_multi_line["all"]
    assert 3 in result_multi_line["V104"]
    assert 4 in result_multi_line["W451"]

    # Test case 8: No noqa present in some lines
    code_mixed = ["import os", "import sys  # noqa: E123"]
    result_mixed = parse_noqa(code_mixed)
    assert len(result_mixed) == 1
    assert 2 in result_mixed["E123"]
    assert 1 not in result_mixed.get("all", set())

def test_ignore_line():
    noqa_lines = {
        "E123": {1},
        "V104": {3},
        "all": {2}
    }
    # Exact match
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Match via 'all'
    assert ignore_line(noqa_lines, 2, "E402") is True
    # No match
    assert ignore_line(noqa_lines, 1, "V104") is False
    assert ignore_line(noqa_lines, 3, "E123") is False
    assert ignore_line(noqa_lines, 4, "E123") is False
```


