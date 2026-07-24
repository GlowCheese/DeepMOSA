####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_ignore_line():
    # Test case 1: Line is explicitly ignored by specific error code
    noqa_lines = {"E123": {5, 10}, "F401": {2}}
    assert ignore_line(noqa_lines, 5, "E123") is True
    assert ignore_line(noqa_lines, 10, "E123") is True
    assert ignore_lines_with_specific_code = ignore_line(noqa_lines, 5, "E123")

    # Test case 2: Line is ignored by 'all' category
    noqa_lines_all = {"all": {1, 8}}
    assert ignore_line(noqa_lines_all, 1, "E123") is True
    assert ignore_line(noqa_lines_all, 8, "F401") is True

    # Test case 3: Line is NOT ignored (not in specific code and not in 'all')
    noqa_lines_none = {"E123": {5}, "all": {1}}
    assert ignore_line(noqa_lines_none, 2, "E123") is False
    assert ignore_line(noqa_lines_none, 3, "F401") is False

    # Test case 4: Error code does not exist in dictionary at all
    noqa_lines_empty = {}
    assert ignore_line(noqa_lines_empty, 5, "E123") is False

    # Test case 5: Verification of mapping (F401 should be treated as V104)
    # This tests the integration with how parse_noqa populates the dict
    code = [
        "import os  # noqa: F401",
        "import sys  # noqa"
    ]
    parsed = parse_noqa(code)
    # Line 1 should be ignored for V104 because of mapping
    assert ignore_line(parsed, 1, "V104") is True
    # Line 2 should be ignored for 'all'
    assert ignore_line(parsed, 2, "E999") is True
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test empty input
    assert parse_noqa([]) == {}

    # Test line with just # noqa (should map to 'all')
    code1 = ["import os  # noqa"]
    result1 = parse_noqa(code1)
    assert 1 in result1["all"]

    # Test line with specific codes
    code2 = ["import sys  # noqa: E123, W451"]
    result2 = parse_noqa(code2)
    assert 1 in result2["E123"]
    assert 1 in result2["W451"]

    # Test case insensitivity and mapping (F401 -> V104)
    code3 = ["import os  # NOQA: F401"]
    result3 = parse_noqa(code3)
    assert 1 in result3["V104"]

    # Test multiple lines and mixed content
    code4 = [
        "import os  # noqa: E123",      # Line 1: E123
        "import sys  # noqa",           # Line 2: all
        "import math  # noqa: F841",    # Line 3: V841 -> V107
        "print('hello')  # no error",   # Line 4: nothing
    ]
    result4 = parse_noqa(code4)
    assert result4["E123"] == {1}
    assert result4["all"] == {2}
    assert result4["V107"] == {3}
    assert len(result4) == 3

    # Test comma separation with extra whitespace
    code5 = ["import os  # noqa: E123,   E456,F789"]
    result5 = parse_noqa(code5)
    assert 1 in result5["E123"]
    assert 1 in result5["E456"]
    assert 1 in result5["F789"]

def test_ignore_line():
    noqa_lines = {
        "E123": {1},
        "all": {2},
        "V104": {3}
    }
    # Match specific code
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Match 'all' category
    assert ignore_line(noqa_lines, 2, "E501") is True
    # No match for specific code
    assert ignore_line(noqa_lines, 3, "E123") is False
    # No match at all
    assert ignore_line(noqa_lines, 4, "E123") is False
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Case 1: Empty input
    assert parse_noqa([]) == {}

    # Case 2: Simple # noqa without specific codes (should map to 'all')
    code_all = ["import os  # noqa", "import sys"]
    result_all = parse_noqa(code_all)
    assert result_all["all"] == {1}
    assert 2 not in result_all["all"]

    # Case 3: # noqa with specific error codes
    code_codes = ["import os  # noqa: E123", "import sys  # noqa: W451, F921"]
    result_codes = parse_noqa(code_codes)
    assert result_codes["E123"] == {1}
    assert result_codes["W451"] == {2}
    assert result_codes["F921"] == {2}

    # Case 4: Testing NOQA_CODE_MAP remapping (F401 -> V104)
    code_remap = ["import os  # noqa: F401"]
    result_remap = parse_noqa(code_remap)
    assert "V104" in result_remap
    assert "F401" not in result_remap
    assert result_remap["V104"] == {1}

    # Case 5: Mixed case and whitespace handling
    code_mixed = [
        "import os  # NoQA: E123",
        "import sys  # noqa:  E456 , F888 "
    ]
    result_mixed = parse_noqa(code_mixed)
    assert result_mixed["E123"] == {1}
    assert result_mixed["E456"] == {2}
    assert result_mixed["F888"] == {2}

    # Case 6: Line with no noqa should not be in dict
    code_none = ["import os", "import sys"]
    assert parse_noqa(code_none) == {}

def test_ignore_line():
    noqa_lines = {"E123": {1}, "V104": {5}, "all": {10}}
    
    # Match specific code
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Match 'all' category
    assert ignore_line(noqa_lines, 10, "E456") is True
    # No match
    assert ignore_line(noqa_lines, 2, "E123") is False
    assert ignore_line(noqa_lines, 5, "E123") is False
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_ignore_line():
    # Case 1: Line is explicitly ignored by a specific error code
    noqa_lines = {"E123": {5, 10}, "F401": {2}}
    assert ignore_line(noqa_lines, 5, "E123") is True
    assert ignore_line(noqa_lines, 10, "E123") is True
    assert ignore_line(noqa_lines, 2, "F401") is True
    
    # Case 2: Line is ignored by the "all" category
    noqa_lines_all = {"all": {7, 8}, "E123": {5}}
    assert ignore_line(noqa_lines_all, 7, "E123") is True
    assert ignore_line(noqa_lines_all, 8, "F401") is True
    
    # Case 3: Line is not ignored for a specific code, but is ignored by "all"
    noqa_lines_mixed = {"E123": {5}, "all": {1}}
    assert ignore_line(noqa_lines_mixed, 1, "E123") is True
    
    # Case 4: Line is not in the dictionary at all (not ignored)
    noqa_lines_empty = {"E123": {5}}
    assert ignore_line(noqa_lines_empty, 6, "E123") is False
    assert ignore_line(noqa_lines_empty, 5, "F401") is False
    
    # Case 5: Empty dictionary (nothing ignored)
    assert ignore_line({}, 1, "E123") is False

def test_parse_noqa_integration():
    code = [
        "import os  # noqa: F401",        # Line 1: Should map to V104
        "x = 1  # noqa",                   # Line 2: Should be 'all'
        "print(x)  # noqa: E123, W451",   # Line 3: Specific codes
        "y = 2",                          # Line 4: No noqa
    ]
    result = parse_noqa(code)
    
    assert 1 in result["V104"]
    assert 2 in result["all"]
    assert 3 in result["E123"]
    assert 3 in result["W451"]
    assert 4 not in result["all"]
    assert 4 not in result["E123"]

def test_parse_error_codes_logic():
    # Testing the regex and mapping logic via parse_noqa
    code = ["# noqa: F841, E501"]
    result = parse_noqa(code)
    assert "V107" in result # F841 maps to V107
    assert "E501" in result
    assert 1 in result["V107"]
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test 1: Empty input
    assert parse_noqa([]) == {}

    # Test 2: Basic # noqa (all category)
    code_all = ["import os  # noqa"]
    result_all = parse_noqa(code_all)
    assert 1 in result_all["all"]
    assert len(result_all) == 1

    # Test 3: Specific error codes
    code_specific = ["import sys  # noqa: E123, W451"]
    result_specific = parse_noqa(code_specific)
    assert 1 in result_specific["E123"]
    assert 1 in result_specific["W451"]

    # Test 4: Mapping of F-codes to V-codes
    code_mapped = ["import os  # noqa: F401, F841"]
    result_mapped = parse_noqa(code_mapped)
    assert 1 in result_mapped["V104"]
    assert 1 in result_mapped["V107"]
    assert "F401" not in result_mapped

    # Test 5: Case insensitivity and spacing
    code_mixed = [
        "import os  # NOQA: E123",
        "import sys  # noqa:   E461, E701 "
    ]
    result_mixed = parse_noqa(code_mixed)
    assert 1 in result_mixed["E123"]
    assert 2 in result_mixed["E461"]
    assert 2 in result_mixed["E701"]

    # Test 6: Multiple lines and multiple codes per line
    code_multi = [
        "import os  # noqa: E123",
        "import sys  # noqa",
        "import math # noqa: F401"
    ]
    result_multi = parse_noqa(code_multi)
    assert 1 in result_multi["E123"]
    assert 2 in result_multi["all"]
    assert 3 in result_multi["V104"]

    # Test 7: Line without noqa should not be in results
    code_clean = ["import os", "import sys"]
    assert parse_noqa(code_clean) == {}

def test_ignore_line():
    noqa_lines = {
        "E123": {1},
        "all": {2},
        "V104": {3}
    }
    # Match specific code
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Match 'all' category
    assert ignore_line(noqa_lines, 2, "E501") is True
    # No match
    assert ignore_line(noqa_lines, 4, "E123") is False
    # Match mapped code line
    assert ignore_line(noqa_lines, 3, "F401") is True
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_ignore_line():
    # Test case 1: Line is explicitly ignored by specific error code
    noqa_lines = {"E123": {5, 10}, "F401": {2}}
    assert ignore_line(noqa_lines, 5, "E123") is True
    assert ignore_line(noqa_lines, 10, "E123") is True
    assert ignore_line(noqa_lines, 7, "E123") is False

    # Test case 2: Line is ignored by "all" category
    noqa_lines = {"all": {1, 3}, "E123": {5}}
    assert ignore_line(noqa_lines, 1, "E123") is True
    assert ignore_line(noqa_lines, 3, "F888") is True
    assert ignore_line(noqa_lines, 5, "E123") is True
    assert ignore_line(noqa_lines, 2, "E123") is False

    # Test case 3: Line and code are not in the dictionary at all
    noqa_lines = {"E123": {5}}
    assert ignore_line(noqa_lines, 6, "E123") is False
    assert ignore_line(noqa_lines, 5, "F401") is False

    # Test case 4: Empty dictionary
    assert ignore_line({}, 1, "E123") is False

    # Test case 5: Verifying mapping logic via parse_noqa integration (contextual check)
    code = [
        "import os  # noqa",             # Line 1: all
        "import sys  # noqa: F401",      # Line 2: F401 -> V104
        "x = 1  # noqa: E123, W451",     # Line 3: E122, W451
    ]
    parsed = parse_noqa(code)
    # Line 1 is in 'all'
    assert ignore_line(parsed, 1, "E123") is True
    # Line 2 is F401 which maps to V104
    assert ignore_line(parsed, 2, "V104") is True
    assert ignore_line(parsed, 2, "F401") is False
    # Line 3
    assert ignore_line(parsed, 3, "E123") is True
    assert ignore_line(parsed, 3, "W451") is True
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_ignore_line():
    # Setup cases
    noqa_lines = {
        "E123": {5, 10},
        "F401": {2},
        "all": {7}
    }

    # Test case: Exact error code match
    assert ignore_line(noqa_lines, 5, "E123") is True
    assert ignore_line(noqa_lines, 10, "E123") is True
    
    # Test case: Error code not in dict, but line exists in 'all'
    assert ignore_line(noqa_lines, 7, "E501") is True
    
    # Test case: Error code match via mapping (F401 -> V104)
    # Note: parse_noqa handles the mapping, but ignore_line checks the keys provided.
    # If we pass F401 directly and it's in the dict:
    assert ignore_line(noqa_lines, 2, "F401") is True

    # Test case: Line number does not exist for specific error code
    assert ignore_line(noqa_lines, 1, "E123") is False
    
    # Test case: Line number does not exist in 'all' or specific code
    assert ignore_line(noqa_lines, 8, "E123") is False
    assert ignore_line(noqa_lines, 9, "F401") is False

    # Test case: Empty dictionary
    assert ignore_line({}, 1, "E123") is False

    # Test case: Error code exists in dict but line not present
    assert ignore_line({"E123": {1}}, 2, "E123") is False
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test case 1: Empty input
    assert parse_noqa([]) == {}

    # Test case 2: Basic # noqa without specific codes (should map to 'all')
    code_basic = [
        "import os  # noqa",
        "import sys"
    ]
    result_basic = parse_noqa(code_basic)
    assert "all" in result_basic
    assert 1 in result_basic["all"]

    # Test case 3: # noqa with specific codes (single)
    code_single = [
        "import os  # noqa: E123",
        "import sys"
    ]
    result_single = parse_noqa(code_single)
    assert "E123" in result_single
    assert 1 in result_single["E123"]
    assert "all" not in result_single

    # Test case 4: # noqa with multiple codes (comma separated, spaces/no spaces)
    code_multi = [
        "import os  # noqa: E123, W451, F921",
        "import sys  # noqa:E722"
    ]
    result_multi = parse_noqa(code_multi)
    assert 1 in result_multi["E123"]
    assert 1 in result_multi["W451"]
    assert 1 in result_multi["F921"]
    assert 2 in result_multi["E722"]

    # Test case 5: Case insensitivity and NOQA_CODE_MAP mapping
    code_mapping = [
        "import os  # NOQA: F401",  # Should map to V104
        "import sys  # noqa: F841"   # Should map to V107
    ]
    result_mapping = parse_noqa(code_mapping)
    assert "V104" in result_mapping
    assert 1 in result_mapping["V104"]
    assert "V107" in result_mapping
    assert 2 in result_mapping["V107"]

    # Test case 6: Mixed 'all' and specific codes
    code_mixed = [
        "import os  # noqa",         # Adds to 'all'
        "import sys  # noqa: E123"   # Adds to 'E123'
    ]
    result_mixed = parse_noqa(code_mixed)
    assert 1 in result_mixed["all"]
    assert 2 in result_mixed["E123"]

    # Test case 7: Line with no noqa at all
    code_none = [
        "import os",
        "import sys"
    ]
    assert parse_noqa(code_none) == {}

def test_ignore_line():
    noqa_lines = {
        "E123": {1},
        "all": {2}
    }
    # Match specific code
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Match 'all' category
    assert ignore_line(noqa_lines, 2, "F401") is True
    # No match for specific code
    assert ignore_line(noqa_lines, 3, "E123") is False
    # No match at all
    assert ignore_line(noqa_lines, 5, "E999") is False
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test Case 1: Empty input
    assert parse_noqa([]) == {}

    # Test Case 2: Basic # noqa without codes (should map to 'all')
    code = [
        "import os  # noqa",
        "import sys"
    ]
    expected = {"all": {1}}
    assert parse_noqa(code) == expected

    # Test Case 3: # noqa with specific codes and case insensitivity
    code = [
        "import os  # noqa: E123",
        "import sys  # NOQA: W451",
        "import math  # noqa: F401, E501"
    ]
    # Note: F401 should be mapped to V104 per NOQA_CODE_MAP
    expected = {
        "E123": {1},
        "W451": {2},
        "V104": {3},
        "E501": {3}
    }
    assert parse_noqa(code) == expected

    # Test Case 4: Comma separated codes with whitespace
    code = [
        "x = 1  # noqa: E123,  W451, F841"
    ]
    # Note: F841 should be mapped to V107 per NOQA_CODE_MAP
    expected = {
        "E123": {1},
        "W451": {1},
        "V107": {1}
    }
    assert parse_noqa(code) == expected

    # Test Case 5: Multiple lines with same error code
    code = [
        "import a  # noqa: E123",
        "import b  # noqa: E123",
        "import c"
    ]
    expected = {"E123": {1, 2}}
    assert parse_noqa(code) == expected

    # Test Case 6: Lines without any noqa pattern should not be in dict
    code = [
        "print('hello')",
        "x = 10"
    ]
    assert parse_noqa(code) == {}

def test_ignore_line():
    noqa_lines = {"E123": {1}, "all": {2}, "V104": {3}}
    
    # Line matches specific code
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Line matches 'all'
    assert ignore_line(noqa_lines, 2, "F401") is True
    # Line does not match specific code and not in all
    assert ignore_line(noqa_lines, 3, "E123") is False
    # Line exists nowhere
    assert ignore_line(noqa_lines, 4, "E123") is False
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_parse_noqa():
    # Test case 1: Empty input
    assert parse_noqa([]) == {}

    # Test case 2: Simple # noqa without specific codes (should map to 'all')
    code2 = [
        "import os  # noqa",
        "import sys  # NoQA",
    ]
    expected2 = {"all": {1, 2}}
    assert parse_noqa(code2) == expected2

    # Test case 3: # noqa with specific codes
    code3 = [
        "import os  # noqa: E123",
        "import sys  # noqa: E123, W451",
    ]
    expected3 = {"E123": {1, 2}, "W451": {2}}
    assert parse_noqa(code3) == expected3

    # Test case 4: Mapping of F-codes to V-codes via NOQA_CODE_MAP
    code4 = [
        "import os  # noqa: F401",
        "x = 1     # noqa: F841",
    ]
    expected4 = {"V104": {1}, "V107": {2}}
    assert parse_noqa(code4) == expected4

    # Test case 5: Mixed content (lines without noqa, lines with different codes)
    code5 = [
        "import os",                 # Line 1: No noqa
        "import sys  # noqa: E123",  # Line 2: E123
        "import math  # noqa",       # Line 3: all
        "import re    # noqa: F401", # Line 4: V104
    ]
    expected5 = {
        "E123": {2},
        "all": {3},
        "V104": {4}
    }
    assert parse_noqa(code5) == expected5

    # Test case 6: Case insensitivity and whitespace in codes
    code6 = [
        "import os  # NOQA: E123,   E456",
    ]
    expected6 = {"E123": {1}, "E456": {1}}
    assert parse_noqa(code6) == expected6

def test_ignore_line():
    noqa_lines = {"E123": {1}, "all": {2}, "V104": {3}}
    
    # Matches specific code
    assert ignore_line(noqa_lines, 1, "E123") is True
    # Matches 'all' category
    assert ignore_line(noqa_lines, 2, "E501") is True
    # Does not match any
    assert ignore_line(noqa_lines, 4, "E123") is False
    # Code exists but line number does not
    assert ignore_line(noqa_lines, 5, "V104") is False
```


