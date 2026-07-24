####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ignore_line():
    # Test case 1: Line is in the error_code set
    noqa_lines = {"V104": {1, 2, 3}, "all": set()}
    assert ignore_line(noqa_lines, 1, "V104") is True
    assert ignore_line(noqa_lines, 2, "V104") is True
    assert ignore_line(noqa_lines, 3, "V104") is True
    assert ignore_line(noqa_lines, 4, "V104") is False

    # Test case 2: Line is in the "all" set
    noqa_lines = {"V104": set(), "all": {1, 2, 3}}
    assert ignore_line(noqa_lines, 1, "V104") is True
    assert ignore_line(noqa_lines, 2, "V104") is True
    assert ignore_line(noqa_lines, 3, "V104") is True
    assert ignore_line(noqa_lines, 4, "V104") is False

    # Test case 3: Line is in both error_code and "all" sets
    noqa_lines = {"V104": {1, 2}, "all": {2, 3}}
    assert ignore_line(noqa_lines, 1, "V104") is True
    assert ignore_line(noqa_lines, 2, "V104") is True
    assert ignore_line(noqa_lines, 3, "V104") is True
    assert ignore_line(noqa_lines, 4, "V104") is False

    # Test case 4: Line is not in any set
    noqa_lines = {"V104": {1, 2}, "all": {3, 4}}
    assert ignore_line(noqa_lines, 5, "V104") is False

    # Test case 5: Different error code
    noqa_lines = {"V107": {1, 2}, "all": {3}}
    assert ignore_line(noqa_lines, 1, "V104") is False
    assert ignore_line(noqa_lines, 1, "V107") is True
    assert ignore_line(noqa_lines, 3, "V104") is True


# LLM-generated content at query #2
#--------------------------

```python
def test_parse_noqa():
    # Test with no noqa comments
    code = ["print('hello')", "x = 1"]
    assert parse_noqa(code) == {"all": set()}

    # Test with simple noqa comment
    code = ["print('hello')  # noqa", "x = 1"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test with specific error codes
    code = ["print('hello')  # noqa: F401", "x = 1  # noqa: F841"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {2}}

    # Test with multiple error codes
    code = ["print('hello')  # noqa: F401, F841", "x = 1"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test with mixed case noqa
    code = ["print('hello')  # NoQa: F401", "x = 1  # NOQA"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "all": {2}}

    # Test with noqa in middle of line
    code = ["x = 1  # some comment  # noqa", "y = 2"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test with multiple lines having noqa
    code = [
        "print('hello')  # noqa",
        "x = 1  # noqa: F401",
        "y = 2  # noqa: F841, F401",
        "z = 3"
    ]
    result = parse_noqa(code)
    assert result == {"all": {1}, "V104": {2, 3}, "V107": {3}}

    # Test with unknown error code (should remain unchanged)
    code = ["print('hello')  # noqa: E123", "x = 1"]
    result = parse_noqa(code)
    assert result == {"E123": {1}}


# LLM-generated content at query #3
#--------------------------

```python
def test_parse_noqa():
    # Test case 1: No noqa comments
    code = ["print('hello')", "x = 1"]
    assert parse_noqa(code) == {"all": set()}

    # Test case 2: Simple noqa comment
    code = ["x = 1  # noqa", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test case 3: Noqa with specific error codes
    code = ["x = 1  # noqa: F401, F841", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test case 4: Noqa with mixed case
    code = ["x = 1  # NoQa: f401", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}

    # Test case 5: Multiple noqa comments
    code = ["x = 1  # noqa", "y = 2  # noqa: F841", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"all": {1}, "V107": {2}}

    # Test case 6: Noqa with multiple error codes
    code = ["x = 1  # noqa: F401, W123, F841", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "W123": {1}, "V107": {1}}

    # Test case 7: Noqa with trailing comma
    code = ["x = 1  # noqa: F401,", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}

    # Test case 8: Noqa with spaces after comma
    code = ["x = 1  # noqa: F401, F841", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test case 9: Noqa with no error codes (should default to "all")
    code = ["x = 1  # noqa:", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"all": {1}}


# LLM-generated content at query #4
#--------------------------

```python
def test_ignore_line():
    # Test when line is ignored for specific error code
    noqa_lines = {"V104": {1, 2}, "V107": {3}}
    assert ignore_line(noqa_lines, 1, "V104") is True
    assert ignore_line(noqa_lines, 2, "V104") is True
    assert ignore_line(noqa_lines, 3, "V107") is True

    # Test when line is not ignored for specific error code
    assert ignore_line(noqa_lines, 1, "V107") is False
    assert ignore_line(noqa_lines, 4, "V104") is False

    # Test when line is ignored for all error codes
    noqa_lines_all = {"all": {1, 2}, "V104": {3}}
    assert ignore_line(noqa_lines_all, 1, "V104") is True
    assert ignore_line(noqa_lines_all, 1, "V107") is True
    assert ignore_line(noqa_lines_all, 2, "V104") is True

    # Test when line is not ignored even with "all" present
    assert ignore_line(noqa_lines_all, 4, "V104") is False

    # Test with empty noqa_lines
    assert ignore_line({}, 1, "V104") is False


# LLM-generated content at query #5
#--------------------------

```python
def test_parse_noqa():
    # Test with no noqa comments
    code = ["print('hello')", "x = 1"]
    assert parse_noqa(code) == {"all": set()}

    # Test with simple noqa comment
    code = ["x = 1  # noqa", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test with noqa comment with specific codes
    code = ["x = 1  # noqa: F401, F841", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test with mixed case noqa comment
    code = ["x = 1  # NoQA: f401", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"F401": {1}}

    # Test with multiple noqa comments
    code = [
        "x = 1  # noqa",
        "y = 2  # noqa: F401",
        "print('hello')  # noqa: F841, E123"
    ]
    result = parse_noqa(code)
    assert result == {
        "all": {1},
        "V104": {2},
        "V107": {3},
        "E123": {3}
    }

    # Test with noqa comment in the middle of the line
    code = ["x = 1 + 2  # noqa: F401", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"V104": {1}}

    # Test with multiple codes in one noqa comment
    code = ["x = 1  # noqa: F401, F841, E123", "print('hello')"]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}, "E123": {1}}


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_ignore_line():
    # Test case 1: Line is ignored with specific error code
    noqa_lines = {"V104": {1, 2}, "all": set()}
    assert ignore_line(noqa_lines, 1, "V104") is True
    assert ignore_line(noqa_lines, 2, "V104") is True
    assert ignore_line(noqa_lines, 3, "V104") is False

    # Test case 2: Line is ignored with "all" error code
    noqa_lines = {"all": {1, 2}, "V107": set()}
    assert ignore_line(noqa_lines, 1, "V107") is True
    assert ignore_line(noqa_lines, 2, "V107") is True
    assert ignore_line(noqa_lines, 3, "V107") is False

    # Test case 3: Line is not ignored
    noqa_lines = {"V104": {1}, "all": {2}}
    assert ignore_line(noqa_lines, 3, "V104") is False
    assert ignore_line(noqa_lines, 3, "V107") is False

    # Test case 4: Empty noqa_lines
    noqa_lines = defaultdict(set)
    assert ignore_line(noqa_lines, 1, "V104") is False
    assert ignore_line(noqa_lines, 1, "all") is False


# LLM-generated content at query #2
#--------------------------

```python
def test_ignore_line():
    # Test case 1: Line is ignored with specific error code
    noqa_lines = {"V104": {1, 3}, "V107": {2}}
    assert ignore_line(noqa_lines, 1, "V104") is True
    assert ignore_line(noqa_lines, 2, "V107") is True
    assert ignore_line(noqa_lines, 3, "V104") is True

    # Test case 2: Line is ignored with "all" error code
    noqa_lines = {"all": {1, 2, 3}}
    assert ignore_line(noqa_lines, 1, "V104") is True
    assert ignore_line(noqa_lines, 2, "V107") is True
    assert ignore_line(noqa_lines, 3, "V104") is True

    # Test case 3: Line is not ignored
    noqa_lines = {"V104": {1, 3}, "V107": {2}}
    assert ignore_line(noqa_lines, 4, "V104") is False
    assert ignore_line(noqa_lines, 2, "V104") is False
    assert ignore_line(noqa_lines, 1, "V107") is False

    # Test case 4: Empty noqa_lines
    noqa_lines = defaultdict(set)
    assert ignore_line(noqa_lines, 1, "V104") is False
    assert ignore_line(noqa_lines, 2, "V107") is False


# LLM-generated content at query #3
#--------------------------

```python
def test_parse_noqa():
    # Test with no noqa comments
    code = ["print('hello')", "x = 1"]
    assert parse_noqa(code) == {"all": set()}

    # Test with simple # noqa
    code = ["x = 1  # noqa", "print('hello')"]
    result = parse_noqa(code)
    assert result["all"] == {1}

    # Test with specific error codes
    code = ["x = 1  # noqa: F401", "print('hello')  # noqa: F841, E123"]
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {2}
    assert result["E123"] == {2}

    # Test with mixed case
    code = ["x = 1  # NOQA", "print('hello')  # NoQa: f401"]
    result = parse_noqa(code)
    assert result["all"] == {1}
    assert result["V104"] == {2}

    # Test with multiple codes and whitespace
    code = ["x = 1  # noqa: F401, F841", "print('hello')  # noqa: E123, W451"]
    result = parse_noqa(code)
    assert result["V104"] == {1}
    assert result["V107"] == {1}
    assert result["E123"] == {2}
    assert result["W451"] == {2}

    # Test with no codes specified (should default to "all")
    code = ["x = 1  # noqa:", "print('hello')  # NOQA:"]
    result = parse_noqa(code)
    assert result["all"] == {1, 2}

    # Test with multiple lines having same error code
    code = [
        "x = 1  # noqa: F401",
        "y = 2  # noqa: F401",
        "print('hello')  # noqa: F841"
    ]
    result = parse_noqa(code)
    assert result["V104"] == {1, 2}
    assert result["V107"] == {3}

    # Test with noqa in the middle of the line
    code = ["x = 1 + 2  # noqa: F401", "print('hello')"]
    result = parse_noqa(code)
    assert result["V104"] == {1}

    # Test with empty code
    assert parse_noqa([]) == {"all": set()}


# LLM-generated content at query #4
#--------------------------

```python
def test_parse_noqa():
    # Test case 1: No noqa comments
    code = [
        "print('hello')",
        "x = 1",
    ]
    assert parse_noqa(code) == {"all": set()}

    # Test case 2: Simple noqa comment
    code = [
        "print('hello')  # noqa",
        "x = 1",
    ]
    result = parse_noqa(code)
    assert result == {"all": {1}}

    # Test case 3: Noqa with specific error code
    code = [
        "print('hello')  # noqa: F401",
        "x = 1",
    ]
    result = parse_noqa(code)
    assert result == {"V104": {1}}

    # Test case 4: Noqa with multiple error codes
    code = [
        "print('hello')  # noqa: F401, F841",
        "x = 1",
    ]
    result = parse_noqa(code)
    assert result == {"V104": {1}, "V107": {1}}

    # Test case 5: Noqa with mixed case
    code = [
        "print('hello')  # NoQA: F401",
        "x = 1",
    ]
    result = parse_noqa(code)
    assert result == {"V104": {1}}

    # Test case 6: Multiple noqa comments
    code = [
        "print('hello')  # noqa",
        "x = 1  # noqa: F841",
        "y = 2  # noqa: F401, F841",
    ]
    result = parse_noqa(code)
    assert result == {"all": {1}, "V107": {2, 3}, "V104": {3}}

    # Test case 7: Noqa with unknown error code
    code = [
        "print('hello')  # noqa: E123",
        "x = 1",
    ]
    result = parse_noqa(code)
    assert result == {"E123": {1}}


# LLM-generated content at query #5
#--------------------------

```python
def test_parse_noqa():
    # Test empty code
    assert parse_noqa("") == defaultdict(set)

    # Test code with no noqa comments
    code = """
def foo():
    pass
"""
    assert parse_noqa(code) == defaultdict(set)

    # Test code with a single noqa comment
    code = """
def foo():  # noqa
    pass
"""
    result = parse_noqa(code)
    assert result["all"] == {2}

    # Test code with noqa comment and specific error codes
    code = """
def foo():  # noqa: F401, F841
    pass
"""
    result = parse_noqa(code)
    assert result["V104"] == {2}
    assert result["V107"] == {2}

    # Test code with multiple noqa comments
    code = """
def foo():  # noqa
    pass

def bar():  # noqa: E123
    pass
"""
    result = parse_noqa(code)
    assert result["all"] == {2}
    assert result["E123"] == {5}

    # Test case insensitivity
    code = """
def foo():  # NOQA: f401
    pass
"""
    result = parse_noqa(code)
    assert result["V104"] == {2}

    # Test mixed case and spacing
    code = """
def foo():  # NoQa: F401,  F841
    pass
"""
    result = parse_noqa(code)
    assert result["V104"] == {2}
    assert result["V107"] == {2}

    # Test noqa with trailing comma
    code = """
def foo():  # noqa: F401,
    pass
"""
    result = parse_noqa(code)
    assert result["V104"] == {2}


