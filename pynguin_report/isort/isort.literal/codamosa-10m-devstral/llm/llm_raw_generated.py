####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_assignment():
    # Test assignments sort_type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2\n"

    # Test dict sort_type
    code = "my_dict = {'b': 2, 'a': 1}\n"
    assert assignment(code, "dict", ".py") == "my_dict = {'a': 1, 'b': 2}\n"

    # Test list sort_type
    code = "my_list = [3, 1, 2]\n"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]\n"

    # Test unique-list sort_type
    code = "my_list = [3, 1, 2, 2]\n"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]\n"

    # Test set sort_type
    code = "my_set = {3, 1, 2}\n"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}\n"

    # Test tuple sort_type
    code = "my_tuple = (3, 1, 2)\n"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)\n"

    # Test unique-tuple sort_type
    code = "my_tuple = (3, 1, 2, 2)\n"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)\n"

    # Test invalid sort_type
    try:
        assignment("my_var = [1, 2, 3]\n", "invalid_type", ".py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal parsing
    try:
        assignment("my_var = invalid_literal\n", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_var = [1, 2, 3]\n", "dict", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test with custom config
    config = Config(line_length=40)
    code = "my_dict = {'b': 2, 'a': 1}\n"
    assert assignment(code, "dict", ".py", config) == "my_dict = {'a': 1, 'b': 2}\n"


# LLM-generated content at query #2
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    assert assignment(code, "dict", "py") == "my_dict = {1: 'a', 2: 'b', 3: 'c'}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_list = {3, 1, 2}"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #3
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    assert assignment(code, "dict", "py") == "my_dict = {1: 'a', 2: 'b', 3: 'c'}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_dict = [3, 1, 2]"
    try:
        assignment(code, "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #4
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", "py")
    except ValueError as e:
        assert str(e) == (
            "Trying to sort using an undefined sort_type. "
            "Defined sort types are assignments, dict, list, unique-list, set, tuple, unique-tuple."
        )

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
    except LiteralParsingFailure as e:
        assert str(e) == "Unable to parse literal: [3, 1, invalid]"

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}", "list", "py")
    except LiteralSortTypeMismatch as e:
        assert str(e) == "Literal type: <class 'dict'> does not match expected type: <class 'list'>"

    # Test formatting function
    config = Config(formatting_function=lambda code, extension, config: code.upper())
    assert assignment("my_list = [3, 1, 2]", "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #5
#--------------------------

```python
def test_assignments():
    # Test basic assignment sorting
    code = "b = 2\na = 1"
    assert assignments(code) == "a = 1\nb = 2"

    # Test with empty lines
    code = "b = 2\n\na = 1"
    assert assignments(code) == "a = 1\n\nb = 2"

    # Test with multiple assignments
    code = "z = 26\na = 1\nm = 13"
    assert assignments(code) == "a = 1\nm = 13\nz = 26"

    # Test with values containing equals signs
    code = "b = 'a = 1'\na = 2"
    assert assignments(code) == "a = 2\nb = 'a = 1'"

    # Test invalid format
    try:
        assignments("invalid code")
        assert False, "Should have raised AssignmentsFormatMismatch"
    except AssignmentsFormatMismatch:
        pass

    # Test empty code
    assert assignments("") == ""
    assert assignments("\n") == "\n"


# LLM-generated content at query #6
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2"

    # Test dict sort type
    code = "my_dict = {'a': 2, 'b': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 2, 'b': 1}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "d = {'a': 3, 'b': 1, 'c': 2}"
    assert assignment(code, "dict", ".py") == "d = {'b': 1, 'c': 2, 'a': 3}"

    # Test list sort type
    code = "l = [3, 1, 2]"
    assert assignment(code, "list", ".py") == "l = [1, 2, 3]"

    # Test unique-list sort type
    code = "ul = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", ".py") == "ul = [1, 2, 3]"

    # Test set sort type
    code = "s = {3, 1, 2}"
    assert assignment(code, "set", ".py") == "s = {1, 2, 3}"

    # Test tuple sort type
    code = "t = (3, 1, 2)"
    assert assignment(code, "tuple", ".py") == "t = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "ut = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", ".py") == "ut = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("x = 1", "invalid", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("x = invalid", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("x = {'a': 1}", "list", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test with trailing whitespace
    code = "l = [3, 1, 2]  \n"
    assert assignment(code, "list", ".py") == "l = [1, 2, 3]  \n"

    # Test with custom config
    config = Config(line_length=50)
    code = "d = {'a': 3, 'b': 1, 'c': 2}"
    assert assignment(code, "dict", ".py", config) == "d = {'b': 1, 'c': 2, 'a': 3}"


# LLM-generated content at query #8
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    assert assignment(code, "dict", "py") == "my_dict = {'b': 1, 'c': 2, 'a': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 1]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 1)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda x, y, z: x.upper())
    assert assignment("my_list = [3, 1, 2]", "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #9
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [2, 1]"
    assert assignment(code, "list", "py") == "my_list = [1, 2]"

    # Test unique-list sort type
    code = "my_list = [2, 1, 2]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2]"

    # Test set sort type
    code = "my_set = {2, 1}"
    assert assignment(code, "set", "py") == "my_set = {1, 2}"

    # Test tuple sort type
    code = "my_tuple = (2, 1)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2)"

    # Test unique-tuple sort type
    code = "my_tuple = (2, 1, 2)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2)"

    # Test invalid sort type
    try:
        assignment("x = 1", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("x = invalid", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("x = {'a': 1}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda x, y, z: x.upper())
    assert assignment("x = [2, 1]", "list", "py", config) == "X = [1, 2]"

    # Test trailing whitespace
    code = "x = [2, 1]   \n"
    assert assignment(code, "list", "py") == "x = [1, 2]   \n"


# LLM-generated content at query #10
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "z = 1\na = 2\nb = 3"
    assert assignment(code, "assignments", "py") == "a = 2\nb = 3\nz = 1"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2, 'c': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    code = "my_list = [3, 1, 2"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]    "
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]    "


# LLM-generated content at query #12
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    expected = "a = 1\nb = 2\nc = 3"
    assert assignment(code, "assignments", "py") == expected

    # Test dict sort type
    code = "my_dict = {1: 'a', 2: 'b'}"
    expected = "my_dict = {1: 'a', 2: 'b'}"
    assert assignment(code, "dict", "py") == expected

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    expected = "my_list = [1, 2, 3]"
    assert assignment(code, "list", "py") == expected

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    expected = "my_list = [1, 2, 3]"
    assert assignment(code, "unique-list", "py") == expected

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    expected = "my_set = {1, 2, 3}"
    assert assignment(code, "set", "py") == expected

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    expected = "my_tuple = (1, 2, 3)"
    assert assignment(code, "tuple", "py") == expected

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    expected = "my_tuple = (1, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == expected

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid_type", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = [3, 1, 2]", "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2, 'c': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid_type", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1, 'c': 3}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    assert assignment("my_list = [3, 1, 2]", "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]  \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]  \n"


# LLM-generated content at query #14
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    assert assignment(code, "dict", ".py") == "my_dict = {'b': 2, 'a': 1, 'c': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1, 'c': 3}", "list", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    assert assignment("my_list = [3, 1, 2]", "list", ".py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   "
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]   "


# LLM-generated content at query #15
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    assert assignment(code, "dict", "py") == "my_dict = {'b': 1, 'c': 2, 'a': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    assert assignment("my_list = [3, 1, 2]", "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]  \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]  \n"


# LLM-generated content at query #16
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}\n"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}\n"

    # Test list sort type
    code = "my_list = [3, 1, 2]\n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]\n"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]\n"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]\n"

    # Test set sort type
    code = "my_set = {3, 1, 2}\n"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}\n"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)\n"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)\n"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)\n"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)\n"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]\n", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]\n", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}\n", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test with custom config
    config = Config(line_length=50)
    code = "my_dict = {'b': 2, 'a': 1}\n"
    assert assignment(code, "dict", "py", config) == "my_dict = {'a': 1, 'b': 2}\n"


# LLM-generated content at query #17
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", ".py") == "my_dict = {'b': 2, 'a': 1}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, 2", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}", "list", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test with custom config
    config = Config(line_length=50)
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", ".py", config) == "my_list = [1, 2, 3]"


# LLM-generated content at query #18
#--------------------------

```python
def test_assignment():
    # Test assignments sort_type
    code = "b = 2\na = 1"
    assert assignment(code, "assignments") == "a = 1\nb = 2"

    # Test dict sort_type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort_type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list") == "my_list = [1, 2, 3]"

    # Test unique-list sort_type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list") == "my_list = [1, 2, 3]"

    # Test set sort_type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set") == "my_set = {1, 2, 3}"

    # Test tuple sort_type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort_type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple") == "my_tuple = (1, 2, 3)"

    # Test invalid sort_type
    try:
        assignment("my_list = [3, 1, 2]", "invalid_type")
    except ValueError as e:
        assert str(e) == (
            "Trying to sort using an undefined sort_type. "
            "Defined sort types are assignments, dict, list, unique-list, set, tuple, unique-tuple."
        )

    # Test invalid literal parsing
    try:
        assignment("my_list = [3, 1, invalid]", "list")
    except LiteralParsingFailure as e:
        assert e.code == "my_list = [3, 1, invalid]"
        assert isinstance(e.error, SyntaxError)

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}", "list")
    except LiteralSortTypeMismatch as e:
        assert e.actual_type == dict
        assert e.expected_type == list


# LLM-generated content at query #19
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [2, 1, 3]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [2, 1, 3, 2]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {2, 1, 3}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (2, 1, 3)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (2, 1, 3, 2)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [2, 1, 3]", "invalid_type", "py")
    except ValueError as e:
        assert str(e) == (
            "Trying to sort using an undefined sort_type. "
            "Defined sort types are assignments, dict, list, unique-list, set, tuple, unique-tuple."
        )

    # Test invalid literal parsing
    try:
        assignment("my_list = [2, 1, invalid]", "list", "py")
    except LiteralParsingFailure as e:
        assert str(e) == "Unable to parse literal: my_list = [2, 1, invalid]"

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}", "list", "py")
    except LiteralSortTypeMismatch as e:
        assert str(e) == "Expected type: <class 'list'>, got: <class 'dict'>"

    # Test with custom config
    config = Config(line_length=40)
    code = "my_list = [2, 1, 3, 4, 5]"
    assert assignment(code, "list", "py", config) == "my_list = [1, 2, 3, 4, 5]"

    # Test with trailing whitespace
    code = "my_list = [2, 1, 3]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #20
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}\n"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}\n"

    # Test list sort type
    code = "my_list = [3, 1, 2]\n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]\n"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]\n"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]\n"

    # Test set sort type
    code = "my_set = {3, 1, 2}\n"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}\n"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)\n"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)\n"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)\n"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)\n"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]\n", "invalid_type", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]\n", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}\n", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]\n"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]\n"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]  \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]  \n"


# LLM-generated content at query #21
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {2: 'b', 1: 'a', 3: 'c'}"
    assert assignment(code, "dict", "py") == "my_dict = {2: 'b', 1: 'a', 3: 'c'}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_dict = [3, 1, 2]"
    try:
        assignment(code, "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]  \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]  \n"


# LLM-generated content at query #22
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 3, 'b': 1, 'c': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 1, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 1, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_dict = [3, 1, 2]"
    try:
        assignment(code, "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}\n"
    assert assignment(code, "dict", ".py") == "my_dict = {'b': 2, 'a': 1}\n"

    # Test list sort type
    code = "my_list = [3, 1, 2]\n"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]\n"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]\n"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]\n"

    # Test set sort type
    code = "my_set = {3, 1, 2}\n"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}\n"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)\n"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)\n"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)\n"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)\n"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]\n", "invalid_type", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]\n", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]\n", "dict", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    assert assignment("my_list = [3, 1, 2]\n", "list", ".py", config) == "MY_LIST = [1, 2, 3]\n"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]    \n"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]    \n"


# LLM-generated content at query #24
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 3, 'b': 1, 'c': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   "
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   "


# LLM-generated content at query #25
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("x = 1", "invalid_type", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    try:
        assignment("x = invalid_literal", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("x = [1, 2, 3]", "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    assert assignment("x = [3, 1, 2]", "list", "py", config) == "X = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "x = [3, 1, 2]  \n"
    assert assignment(code, "list", "py") == "x = [1, 2, 3]  \n"


# LLM-generated content at query #26
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_dict = {'b': 2, 'a': 1}"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #27
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid_type", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal parsing
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]  \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]  \n"


# LLM-generated content at query #28
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    assert assignment(code, "dict", "py") == "my_dict = {1: 'a', 2: 'b', 3: 'c'}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, extension, config: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #29
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2, 'c': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1, 'c': 3}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   "
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   "


# LLM-generated content at query #30
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    assert assignment(code, "dict", "py") == "my_dict = {'b': 2, 'a': 1, 'c': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   "
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   "


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", ".py") == "my_dict = {'b': 2, 'a': 1}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}", "list", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("x = 1", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("x = invalid", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("x = {'a': 1}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda x, _, __: x.upper())
    assert assignment("x = [3, 1, 2]", "list", "py", config) == "X = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "x = [3, 1, 2]  \n"
    assert assignment(code, "list", "py") == "x = [1, 2, 3]  \n"


# LLM-generated content at query #3
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2"

    # Test dict sort type
    code = "my_dict = {'a': 2, 'b': 1}"
    assert assignment(code, "dict", ".py") == "my_dict = {'a': 2, 'b': 1}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, 2", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2"

    # Test dict sort type
    code = "my_dict = {'a': 2, 'b': 1}"
    assert assignment(code, "dict", ".py") == "my_dict = {'a': 2, 'b': 1}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, 2", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'a': 2, 'b': 1}", "list", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("x = 1", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("x = invalid_literal", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("x = {'a': 1}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda x, y, z: x.upper())
    assert assignment("x = [3, 1, 2]", "list", "py", config) == "X = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "x = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "x = [1, 2, 3]   \n"


# LLM-generated content at query #6
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", ".py") == "my_dict = {'b': 2, 'a': 1}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid_type", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}", "list", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("x = 1", "invalid_type", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    try:
        assignment("x = invalid_literal", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("x = {'a': 1}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test with custom config
    config = Config(line_length=50)
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py", config) == "my_dict = {'a': 1, 'b': 2}"


# LLM-generated content at query #8
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}\n"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 3, 'b': 1, 'c': 2}\n"

    # Test list sort type
    code = "my_list = [3, 1, 2]\n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]\n"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]\n"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]\n"

    # Test set sort type
    code = "my_set = {3, 1, 2}\n"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}\n"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)\n"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)\n"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)\n"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)\n"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]\n", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]\n", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]\n", "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2, 'c': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, 2", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    assert assignment("my_list = [3, 1, 2]", "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #10
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1"
    assert assignment(code, "assignments") == "a = 1\nb = 2"

    # Test dict sort type
    code = "my_dict = {1: 3, 2: 1}"
    assert assignment(code, "dict") == "my_dict = {1: 3, 2: 1}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid_type")
    except ValueError as e:
        assert str(e) == (
            "Trying to sort using an undefined sort_type. "
            "Defined sort types are dict, list, unique-list, set, tuple, unique-tuple."
        )

    # Test invalid literal
    code = "my_list = [3, 1, 2"
    try:
        assignment(code, "list")
    except LiteralParsingFailure as e:
        assert str(e) == "Failed to parse literal: [3, 1, 2"

    # Test type mismatch
    code = "my_dict = {1: 3, 2: 1}"
    try:
        assignment(code, "list")
    except LiteralSortTypeMismatch as e:
        assert str(e) == "Expected type <class 'list'> but got <class 'dict'>"


# LLM-generated content at query #11
#--------------------------

```python
def test_assignment():
    # Test assignments sort_type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort_type
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    assert assignment(code, "dict", "py") == "my_dict = {'b': 1, 'c': 2, 'a': 3}"

    # Test list sort_type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort_type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort_type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort_type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort_type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort_type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid_type", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [2, 1]"
    assert assignment(code, "list", "py") == "my_list = [1, 2]"

    # Test unique-list sort type
    code = "my_list = [2, 1, 2]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2]"

    # Test set sort type
    code = "my_set = {2, 1}"
    assert assignment(code, "set", "py") == "my_set = {1, 2}"

    # Test tuple sort type
    code = "my_tuple = (2, 1)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2)"

    # Test unique-tuple sort type
    code = "my_tuple = (2, 1, 2)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2)"

    # Test invalid sort type
    try:
        assignment("x = 1", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("x = invalid", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("x = {'a': 1}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #13
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}\n"
    assert assignment(code, "dict", ".py") == "my_dict = {'a': 1, 'b': 2}\n"

    # Test list sort type
    code = "my_list = [3, 1, 2]\n"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]\n"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]\n"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]\n"

    # Test set sort type
    code = "my_set = {3, 1, 2}\n"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}\n"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)\n"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)\n"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)\n"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)\n"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]\n"
    try:
        assignment(code, "invalid_type", ".py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    code = "my_list = [3, 1, invalid]\n"
    try:
        assignment(code, "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_dict = {'b': 2, 'a': 1}\n"
    try:
        assignment(code, "list", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]\n"
    assert assignment(code, "list", ".py", config) == "MY_LIST = [1, 2, 3]\n"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]    \n"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]    \n"


# LLM-generated content at query #14
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_var = [1, 2, 3]", "invalid_type", "py")
    except ValueError as e:
        assert str(e) == (
            "Trying to sort using an undefined sort_type. "
            "Defined sort types are assignments, dict, list, unique-list, set, tuple, unique-tuple."
        )

    # Test invalid literal parsing
    try:
        assignment("my_var = invalid_literal", "list", "py")
    except LiteralParsingFailure as e:
        assert str(e) == "Unable to parse literal: invalid_literal"

    # Test type mismatch
    try:
        assignment("my_var = [1, 2, 3]", "dict", "py")
    except LiteralSortTypeMismatch as e:
        assert str(e) == "Expected type: <class 'dict'>, got: <class 'list'>"

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #15
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    assert assignment(code, "dict", ".py") == "my_dict = {'b': 1, 'c': 2, 'a': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 1, 3]"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 1, 3)"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    assert assignment("my_list = [3, 1, 2]", "list", ".py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #16
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    assert assignment(code, "dict", "py") == "my_dict = {'b': 1, 'c': 2, 'a': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid_type", "py")
        assert False, "Expected ValueError for invalid sort type"
    except ValueError:
        pass

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure for invalid literal"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralSortTypeMismatch for type mismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]    "
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]    "


# LLM-generated content at query #17
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "z = 1\na = 2\nb = 3"
    assert assignment(code, "assignments", "py") == "a = 2\nb = 3\nz = 1"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    code = "my_list = [3, 1, 2"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   "
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   "


# LLM-generated content at query #18
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2, 'c': 3}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1, 'c': 3}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #19
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid_type", "py")
    except ValueError as e:
        assert str(e) == (
            "Trying to sort using an undefined sort_type. "
            "Defined sort types are assignments, dict, list, unique-list, set, tuple, unique-tuple."
        )

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, 2", "list", "py")
    except LiteralParsingFailure as e:
        assert str(e) == "Failed to parse literal: [3, 1, 2"

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}", "list", "py")
    except LiteralSortTypeMismatch as e:
        assert str(e) == "Expected type <class 'list'> but found <class 'dict'>."


# LLM-generated content at query #20
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {1: 3, 2: 2, 3: 1}"
    assert assignment(code, "dict", "py") == "my_dict = {1: 3, 2: 2, 3: 1}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", "py")
    except ValueError as e:
        assert str(e) == (
            "Trying to sort using an undefined sort_type. "
            "Defined sort types are assignments, dict, list, unique-list, set, tuple, unique-tuple."
        )

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, 2", "list", "py")
    except LiteralParsingFailure as e:
        assert str(e) == "Unable to parse literal: [3, 1, 2"

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", "py")
    except LiteralSortTypeMismatch as e:
        assert str(e) == "Expected type: <class 'dict'> but got <class 'list'>"


# LLM-generated content at query #21
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2"

    # Test dict sort type
    code = "my_dict = {'a': 2, 'b': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 2, 'b': 1}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid", "py")
    except ValueError as e:
        assert str(e) == (
            "Trying to sort using an undefined sort_type. "
            "Defined sort types are assignments, dict, list, unique-list, set, tuple, unique-tuple."
        )

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
    except LiteralParsingFailure as e:
        assert str(e) == "my_list = [3, 1, invalid]"

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", "py")
    except LiteralSortTypeMismatch as e:
        assert str(e) == "Trying to sort a <class 'list'> as a <class 'dict'>."


# LLM-generated content at query #22
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'a': 2, 'b': 1}\n"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 2, 'b': 1}\n"

    # Test list sort type
    code = "my_list = [3, 1, 2]\n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]\n"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]\n"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]\n"

    # Test set sort type
    code = "my_set = {3, 1, 2}\n"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}\n"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)\n"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)\n"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)\n"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)\n"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]\n"
    try:
        assignment(code, "invalid_type", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    code = "my_list = invalid_literal\n"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_dict = {'a': 2, 'b': 1}\n"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test with trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"

    # Test with custom config
    config = Config(line_length=40)
    code = "my_list = [3, 1, 2]\n"
    assert assignment(code, "list", "py", config) == "my_list = [1, 2, 3]\n"


# LLM-generated content at query #23
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'a': 2, 'b': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 2, 'b': 1}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid_type", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", ".py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {'a': 3, 'b': 1, 'c': 2}"
    assert assignment(code, "dict", ".py") == "my_dict = {'a': 3, 'b': 1, 'c': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 1, 3]"
    assert assignment(code, "unique-list", ".py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", ".py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 1, 3)"
    assert assignment(code, "unique-tuple", ".py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid_type", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    assert assignment("my_list = [3, 1, 2]", "list", ".py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", ".py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #25
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\nc = 3"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\nc = 3"

    # Test dict sort type
    code = "my_dict = {3: 'c', 1: 'a', 2: 'b'}"
    assert assignment(code, "dict", "py") == "my_dict = {1: 'a', 2: 'b', 3: 'c'}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 1, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 1, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal
    code = "my_list = [3, 1, invalid]"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    code = "my_dict = [3, 1, 2]"
    try:
        assignment(code, "dict", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace
    code = "my_list = [3, 1, 2]   \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]   \n"


# LLM-generated content at query #26
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_var = 1", "invalid_type", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal parsing
    try:
        assignment("my_var = invalid_literal", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_var = {'a': 1}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test with custom config
    config = Config(line_length=40)
    code = "my_dict = {'b': 2, 'a': 1, 'c': 3}"
    assert assignment(code, "dict", "py", config) == "my_dict = {'a': 1, 'b': 2, 'c': 3}"


# LLM-generated content at query #27
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid_type", "py")
    except ValueError as e:
        assert str(e) == (
            "Trying to sort using an undefined sort_type. "
            "Defined sort types are assignments, dict, list, unique-list, set, tuple, unique-tuple."
        )

    # Test invalid literal parsing
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
    except LiteralParsingFailure as e:
        assert str(e) == "my_list = [3, 1, invalid]"

    # Test type mismatch
    try:
        assignment("my_list = [3, 1, 2]", "dict", "py")
    except LiteralSortTypeMismatch as e:
        assert str(e) == "Expected type <class 'dict'> but got <class 'list'>"

    # Test formatting function
    config = Config(formatting_function=lambda code, extension, config: code.upper())
    assert assignment("my_list = [3, 1, 2]", "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]  \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]  \n"


# LLM-generated content at query #28
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "x = {'a': 2, 'b': 1}\n"
    assert assignment(code, "dict", "py") == "x = {'a': 2, 'b': 1}\n"

    # Test list sort type
    code = "y = [3, 1, 2]\n"
    assert assignment(code, "list", "py") == "y = [1, 2, 3]\n"

    # Test unique-list sort type
    code = "z = [1, 2, 2, 3]\n"
    assert assignment(code, "unique-list", "py") == "z = [1, 2, 3]\n"

    # Test set sort type
    code = "s = {3, 1, 2}\n"
    assert assignment(code, "set", "py") == "s = {1, 2, 3}\n"

    # Test tuple sort type
    code = "t = (3, 1, 2)\n"
    assert assignment(code, "tuple", "py") == "t = (1, 2, 3)\n"

    # Test unique-tuple sort type
    code = "u = (1, 2, 2, 3)\n"
    assert assignment(code, "unique-tuple", "py") == "u = (1, 2, 3)\n"

    # Test invalid sort type
    try:
        assignment("x = 1", "invalid", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test invalid literal
    try:
        assignment("x = invalid", "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("x = {'a': 1}", "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    assert assignment("x = [3, 1, 2]", "list", "py", config) == "X = [1, 2, 3]"


# LLM-generated content at query #29
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", ".py")
    assert result == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", ".py")
    assert result == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", ".py")
    assert result == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    result = assignment(code, "unique-list", ".py")
    assert result == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    result = assignment(code, "set", ".py")
    assert result == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    result = assignment(code, "tuple", ".py")
    assert result == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    result = assignment(code, "unique-tuple", ".py")
    assert result == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_var = 1", "invalid_type", ".py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)

    # Test invalid literal parsing
    try:
        assignment("my_var = invalid_literal", "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

    # Test type mismatch
    try:
        assignment("my_var = {'a': 1}", "list", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_assignment():
    # Test assignments sort type
    code = "b = 2\na = 1\n"
    assert assignment(code, "assignments", "py") == "a = 1\nb = 2\n"

    # Test dict sort type
    code = "my_dict = {'b': 2, 'a': 1}"
    assert assignment(code, "dict", "py") == "my_dict = {'a': 1, 'b': 2}"

    # Test list sort type
    code = "my_list = [3, 1, 2]"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]"

    # Test unique-list sort type
    code = "my_list = [3, 1, 2, 2, 3]"
    assert assignment(code, "unique-list", "py") == "my_list = [1, 2, 3]"

    # Test set sort type
    code = "my_set = {3, 1, 2}"
    assert assignment(code, "set", "py") == "my_set = {1, 2, 3}"

    # Test tuple sort type
    code = "my_tuple = (3, 1, 2)"
    assert assignment(code, "tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test unique-tuple sort type
    code = "my_tuple = (3, 1, 2, 2, 3)"
    assert assignment(code, "unique-tuple", "py") == "my_tuple = (1, 2, 3)"

    # Test invalid sort type
    try:
        assignment("my_list = [3, 1, 2]", "invalid_type", "py")
    except ValueError as e:
        assert str(e) == (
            "Trying to sort using an undefined sort_type. "
            "Defined sort types are dict, list, unique-list, set, tuple, unique-tuple."
        )

    # Test invalid literal
    try:
        assignment("my_list = [3, 1, invalid]", "list", "py")
    except LiteralParsingFailure as e:
        assert str(e) == "Unable to parse literal: [3, 1, invalid]"

    # Test type mismatch
    try:
        assignment("my_dict = {'b': 2, 'a': 1}", "list", "py")
    except LiteralSortTypeMismatch as e:
        assert str(e) == "Expected type: <class 'list'>, got: <class 'dict'>"

    # Test formatting function
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    assert assignment("my_list = [3, 1, 2]", "list", "py", config) == "MY_LIST = [1, 2, 3]"

    # Test trailing whitespace preservation
    code = "my_list = [3, 1, 2]  \n"
    assert assignment(code, "list", "py") == "my_list = [1, 2, 3]  \n"


