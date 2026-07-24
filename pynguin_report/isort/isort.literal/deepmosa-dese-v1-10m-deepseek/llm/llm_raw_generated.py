####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_list_sorting():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    assert _list([3, 1, 2], printer) == "[1, 2, 3]"
    assert _list(["c", "a", "b"], printer) == "['a', 'b', 'c']"
    assert _list([], printer) == "[]"


# LLM-generated content at query #2
#--------------------------

```python
def test__set_with_empty_set():
    config = Config(line_length=88)
    printer = ISortPrettyPrinter(config)
    result = _set(set(), printer)
    assert result == "{}"

def test__set_with_single_element():
    config = Config(line_length=88)
    printer = ISortPrettyPrinter(config)
    result = _set({1}, printer)
    assert result == "{1}"

def test__set_with_multiple_elements():
    config = Config(line_length=88)
    printer = ISortPrettyPrinter(config)
    result = _set({3, 2, 1}, printer)
    assert result == "{1, 2, 3}"

def test__set_with_string_elements():
    config = Config(line_length=88)
    printer = ISortPrettyPrinter(config)
    result = _set({"c", "b", "a"}, printer)
    assert result == "{'a', 'b', 'c'}"

def test__set_with_mixed_types():
    config = Config(line_length=88)
    printer = ISortPrettyPrinter(config)
    result = _set({1, "a", True}, printer)
    assert result == "{1, True, 'a'}"


# LLM-generated content at query #3
#--------------------------

```python
def test_assignments_with_multiple_lines():
    code = "a = 1\nb = 2\nc = 3"
    result = assignments(code)
    assert result == "a = 1b = 2c = 3"

def test_assignments_with_empty_lines():
    code = "a = 1\n\nb = 2\n\nc = 3"
    result = assignments(code)
    assert result == "a = 1b = 2c = 3"

def test_assignments_with_trailing_spaces():
    code = "a = 1 \nb = 2 \nc = 3 "
    result = assignments(code)
    assert result == "a = 1 b = 2 c = 3 "

def test_assignments_with_missing_assignment():
    try:
        code = "a = 1\nb\nc = 3"
        assignments(code)
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignments_with_single_line():
    code = "a = 1"
    result = assignments(code)
    assert result == "a = 1"

def test_assignments_with_sorted_output():
    code = "c = 3\na = 1\nb = 2"
    result = assignments(code)
    assert result == "a = 1b = 2c = 3"


# LLM-generated content at query #4
#--------------------------

```python
def test_assignment_with_valid_assignments_sort_type():
    code = "a = 1\nb = 2\nc = 3"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\nc = 3"

def test_assignment_with_empty_code():
    code = ""
    result = assignment(code, "assignments", "py")
    assert result == ""

def test_assignment_with_invalid_sort_type():
    code = "a = 1"
    try:
        assignment(code, "invalid_sort_type", "py")
        assert False
    except ValueError:
        assert True

def test_assignment_with_literal_parsing_failure():
    code = "a = invalid_literal"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_with_literal_sort_type_mismatch():
    code = "a = 1"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_with_valid_sort_type_and_formatting():
    code = "a = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "a = [1, 2, 3]"

def test_assignment_with_trailing_whitespace():
    code = "a = [3, 1, 2]   \n"
    result = assignment(code, "list", "py")
    assert result == "a = [1, 2, 3]   \n"

def test_assignment_with_formatting_function():
    code = "a = [3, 1, 2]"
    config = Config(line_length=80, formatting_function=lambda x, _, __: x.upper())
    result = assignment(code, "list", "py", config)
    assert result == "A = [1, 2, 3]"


# LLM-generated content at query #5
#--------------------------

```python
def test__dict():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    input_dict = {'b': 2, 'a': 1}
    expected_output = "{'a': 1, 'b': 2}"
    assert _dict(input_dict, printer) == expected_output


# LLM-generated content at query #6
#--------------------------

```
def test_tuple_with_empty_tuple():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    assert _tuple((), printer) == "()"

def test_tuple_with_single_element():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    assert _tuple((1,), printer) == "(1,)"

def test_tuple_with_multiple_elements():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    assert _tuple((3, 1, 2), printer) == "(1, 2, 3)"

def test_tuple_with_strings():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    assert _tuple(("b", "a", "c"), printer) == "('a', 'b', 'c')"

def test_tuple_with_mixed_types():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    assert _tuple((3, "a", 1, "b"), printer) == "(1, 3, 'a', 'b')"


# LLM-generated content at query #7
#--------------------------

```python
def test_assignment_with_formatting_function():
    config = Config(line_length=88, formatting_function=lambda x, y, z: x.upper())
    code = "var = [3, 1, 2]"
    sorted_code = assignment(code, "lists", ".py", config)
    assert "VAR = [1, 2, 3]" in sorted_code


# LLM-generated content at query #8
#--------------------------

```python
def test_unique_list():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    value = [3, 1, 2, 2, 3]
    result = _unique_list(value, printer)
    assert result == "[1, 2, 3]"


# LLM-generated content at query #9
#--------------------------

```python
def test_assignment_with_invalid_literal_raises_literal_parsing_failure():
    invalid_code = "x = invalid_literal"
    expected_error = LiteralParsingFailure
    try:
        assignment(invalid_code, "list", ".py")
    except expected_error as error:
        assert error.code == invalid_code
    else:
        assert False, "Expected LiteralParsingFailure to be raised"


# LLM-generated content at query #10
#--------------------------

```python
def test__unique_tuple():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _unique_tuple((3, 1, 2, 1, 2), printer)
    assert result == "(1, 2, 3)"


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_18_evaluates_to_false():
    config = Config(line_length=80)
    code = "x = 'invalid_literal'"
    sort_type = "invalid_type"
    extension = "py"
    try:
        assignment(code, sort_type, extension, config)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #12
#--------------------------

```python
def test_literal_parsing_failure():
    config = Config(line_length=80)
    code = "x = invalid_literal"
    try:
        assignment(code, "int", "py", config)
    except LiteralParsingFailure:
        pass
    else:
        assert False, "Expected LiteralParsingFailure"


# LLM-generated content at query #13
#--------------------------

```python
def test_assignment_with_valid_assignment():
    code = "x = 1"
    result = assignment(code, "assignments", ".py")
    assert result == "x = 1"

def test_assignment_with_multiple_assignments():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", ".py")
    assert result == "a = 1\nb = 2"

def test_assignment_with_empty_lines():
    code = "b = 2\n\na = 1"
    result = assignment(code, "assignments", ".py")
    assert result == "a = 1\nb = 2"

def test_assignment_with_invalid_format():
    code = "invalid line"
    try:
        assignment(code, "assignments", ".py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_with_unsupported_sort_type():
    code = "x = 1"
    try:
        assignment(code, "invalid_type", ".py")
        assert False
    except ValueError:
        assert True

def test_assignment_with_literal_parsing_failure():
    code = "x = invalid_literal"
    try:
        assignment(code, "some_type", ".py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_with_type_mismatch():
    code = "x = 'string'"
    try:
        assignment(code, "some_type", ".py")
        assert False
    except LiteralSortTypeMismatch:
        assert True


# LLM-generated content at query #14
#--------------------------

```
def test_assignment_applies_formatting_function_when_present():
    config = Config(formatting_function=lambda x, y, z: x.upper())
    result = assignment("x = 1", "numbers", "py", config)
    assert result == "X = 1"

def test_assignment_does_not_apply_formatting_function_when_absent():
    config = Config(formatting_function=None)
    result = assignment("x = 1", "numbers", "py", config)
    assert result == "x = 1"


# LLM-generated content at query #15
#--------------------------

```python
def test_assignment_with_formatting_function():
    config = Config(line_length=80, formatting_function=lambda code, ext, cfg: code.upper())
    code = "var = [3, 1, 2]"
    sort_type = "list"
    extension = "py"
    result = assignment(code, sort_type, extension, config)
    assert "VAR = [1, 2, 3]" in result


# LLM-generated content at query #16
#--------------------------

```python
def test_assignment_with_formatting_function():
    config = Config(line_length=88, formatting_function=lambda code, ext, cfg: code.upper())
    code = "x = [2, 1, 3]"
    sorted_code = assignment(code, "assignments", "py", config)
    assert sorted_code == "X = [2, 1, 3]"

def test_assignment_without_formatting_function():
    config = Config(line_length=88, formatting_function=None)
    code = "x = [2, 1, 3]"
    sorted_code = assignment(code, "assignments", "py", config)
    assert sorted_code == "x = [1, 2, 3]"


# LLM-generated content at query #17
#--------------------------

```python
def test_assignment_raises_literal_parsing_failure():
    code = "invalid_literal = this_will_fail"
    sort_type = "assignments"
    with pytest.raises(LiteralParsingFailure):
        assignment(code, sort_type, "py")


# LLM-generated content at query #18
#--------------------------

```python
def test_assignment_with_formatting_function():
    config = Config(formatting_function=lambda code, extension, config: code.upper())
    code = "x = [3, 1, 2]"
    sort_type = "lists"
    extension = ".py"
    result = assignment(code, sort_type, extension, config)
    assert "X = [1, 2, 3]" in result


# LLM-generated content at query #19
#--------------------------

```
def test_assignment_with_valid_literal():
    code = "x = [2, 1, 3]"
    sort_type = "lists"
    extension = "py"
    result = assignment(code, sort_type, extension)
    assert "x = [1, 2, 3]" in result

def test_assignment_with_invalid_literal():
    code = "x = invalid_literal"
    sort_type = "lists"
    extension = "py"
    try:
        assignment(code, sort_type, extension)
    except LiteralParsingFailure:
        pass
    else:
        assert False, "Expected LiteralParsingFailure"


# LLM-generated content at query #20
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", ".py")
    assert result == "a = 1\nb = 2"

def test_assignment_with_unsupported_sort_type():
    try:
        assignment("a = 1", "unsupported", ".py")
    except ValueError:
        pass
    else:
        assert False

def test_assignment_with_invalid_literal_parsing():
    try:
        assignment("a = invalid", "list", ".py")
    except LiteralParsingFailure:
        pass
    else:
        assert False

def test_assignment_with_literal_sort_type_mismatch():
    try:
        assignment("a = 1", "dict", ".py")
    except LiteralSortTypeMismatch:
        pass
    else:
        assert False

def test_assignment_with_valid_list_sort():
    code = "a = [3, 1, 2]"
    result = assignment(code, "list", ".py")
    assert result == "a = [1, 2, 3]"

def test_assignment_with_valid_dict_sort():
    code = "a = {'c': 3, 'a': 1}"
    result = assignment(code, "dict", ".py")
    assert result == "a = {'a': 1, 'c': 3}"

def test_assignment_with_valid_set_sort():
    code = "a = {3, 1, 2}"
    result = assignment(code, "set", ".py")
    assert result == "a = {1, 2, 3}"

def test_assignment_with_empty_code():
    result = assignment("", "assignments", ".py")
    assert result == ""

def test_assignment_with_whitespace_only_code():
    result = assignment("   \n   ", "assignments", ".py")
    assert result == ""

def test_assignment_with_invalid_assignment_format():
    try:
        assignment("a 1", "assignments", ".py")
    except AssignmentsFormatMismatch:
        pass
    else:
        assert False


# LLM-generated content at query #21
#--------------------------

```python
def test_assignment_with_formatting_function():
    config = Config(formatting_function=lambda x, y, z: x.upper(), line_length=80)
    result = assignment("x = [2, 1]", "sequences", "py", config)
    assert result == "X = [1, 2]"

def test_assignment_without_formatting_function():
    config = Config(formatting_function=None, line_length=80)
    result = assignment("x = [2, 1]", "sequences", "py", config)
    assert result == "x = [1, 2]"


# LLM-generated content at query #22
#--------------------------

```python
def test_assignment_with_valid_literal():
    result = assignment("x = [3, 1, 2]", "lists", ".py")
    assert result == "x = [1, 2, 3]"

def test_assignment_with_valid_dict_literal():
    result = assignment("x = {'b': 2, 'a': 1}", "dicts", ".py")
    assert result == "x = {'a': 1, 'b': 2}"

def test_assignment_with_valid_set_literal():
    result = assignment("x = {3, 1, 2}", "sets", ".py")
    assert result == "x = {1, 2, 3}"

def test_assignment_with_valid_tuple_literal():
    result = assignment("x = (3, 1, 2)", "tuples", ".py")
    assert result == "x = (1, 2, 3)"


# LLM-generated content at query #23
#--------------------------

```python
def test_assignment_with_invalid_literal():
    code = "x = invalid_literal"
    sort_type = "list"
    extension = "py"
    config = Config(line_length=88)
    try:
        assignment(code, sort_type, extension, config)
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass


# LLM-generated content at query #24
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_with_invalid_sort_type():
    code = "a = 1"
    try:
        assignment(code, "invalid_sort_type", "py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_assignment_with_literal_parsing_failure():
    code = "a = invalid_literal"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

def test_assignment_with_literal_sort_type_mismatch():
    code = "a = 1"
    try:
        assignment(code, "list", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

def test_assignment_with_valid_list_sort_type():
    code = "a = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "a = [1, 2, 3]"

def test_assignment_with_formatting_function():
    code = "a = [3, 1, 2]"
    config = Config(formatting_function=lambda x, y, z: x.upper())
    result = assignment(code, "list", "py", config)
    assert result == "A = [1, 2, 3]"

def test_assignment_preserves_trailing_whitespace():
    code = "a = [3, 1, 2]  \n"
    result = assignment(code, "list", "py")
    assert result == "a = [1, 2, 3]  \n"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_assignment_valid_assignments():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_invalid_assignments_format():
    code = "b = 2\n a 1"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_valid_literal_sort():
    code = "a = [2, 1]"
    result = assignment(code, "list", "py")
    assert result == "a = [1, 2]"

def test_assignment_invalid_sort_type():
    code = "a = [2, 1]"
    try:
        assignment(code, "invalid_sort", "py")
        assert False
    except ValueError:
        assert True

def test_assignment_literal_parsing_failure():
    code = "a = invalid_literal"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_literal_sort_type_mismatch():
    code = "a = [2, 1]"
    try:
        assignment(code, "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True


# LLM-generated content at query #2
#--------------------------

```
def test_dict_sorting():
    test_data = {"b": 2, "a": 1, "c": 3}
    expected = "{'a': 1, 'b': 2, 'c': 3}"
    config = Config(line_length=88)
    printer = ISortPrettyPrinter(config)
    result = _dict(test_data, printer)
    assert result == expected

def test_dict_empty():
    test_data = {}
    expected = "{}"
    config = Config(line_length=88)
    printer = ISortPrettyPrinter(config)
    result = _dict(test_data, printer)
    assert result == expected

def test_dict_single_item():
    test_data = {"a": 1}
    expected = "{'a': 1}"
    config = Config(line_length=88)
    printer = ISortPrettyPrinter(config)
    result = _dict(test_data, printer)
    assert result == expected

def test_dict_already_sorted():
    test_data = {"a": 1, "b": 2, "c": 3}
    expected = "{'a': 1, 'b': 2, 'c': 3}"
    config = Config(line_length=88)
    printer = ISortPrettyPrinter(config)
    result = _dict(test_data, printer)
    assert result == expected


# LLM-generated content at query #3
#--------------------------

```python
def test__unique_tuple():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _unique_tuple((3, 1, 2, 2, 1), printer)
    assert result == "(1, 2, 3)"


# LLM-generated content at query #4
#--------------------------

```python
def test_unique_list_with_duplicates():
    printer = ISortPrettyPrinter(Config(line_length=88))
    result = _unique_list([3, 1, 2, 3, 1], printer)
    assert result == "[1, 2, 3]"

def test_unique_list_without_duplicates():
    printer = ISortPrettyPrinter(Config(line_length=88))
    result = _unique_list([3, 1, 2], printer)
    assert result == "[1, 2, 3]"

def test_unique_list_empty():
    printer = ISortPrettyPrinter(Config(line_length=88))
    result = _unique_list([], printer)
    assert result == "[]"


# LLM-generated content at query #5
#--------------------------

```python
def test__set_with_empty_set():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set(set(), printer)
    assert result == "{}"

def test__set_with_non_empty_set():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({3, 1, 2}, printer)
    assert result == "{1, 2, 3}"

def test__set_with_non_empty_set_and_custom_line_length():
    config = Config(line_length=10)
    printer = ISortPrettyPrinter(config)
    result = _set({3, 1, 2}, printer)
    assert result == "{1, 2, 3}"


# LLM-generated content at query #6
#--------------------------

```python
def test_assignment_with_formatting_function():
    class Config:
        def __init__(self, formatting_function, line_length):
            self.formatting_function = formatting_function
            self.line_length = line_length

    def formatting_function(code, extension, config):
        return code.upper()

    config = Config(formatting_function=formatting_function, line_length=80)
    code = "x = 1"
    sort_type = "assignments"
    extension = "py"
    result = assignment(code, sort_type, extension, config)
    assert "X = 1" in result


# LLM-generated content at query #7
#--------------------------

```python
def test_assignment_raises_literal_parsing_failure():
    code = "x = invalid_literal"
    sort_type = "list"
    extension = "py"
    config = DEFAULT_CONFIG
    try:
        assignment(code, sort_type, extension, config)
    except Exception as e:
        assert isinstance(e, LiteralParsingFailure)


# LLM-generated content at query #8
#--------------------------

```
def test__tuple_sorts_and_formats_tuple():
    from unittest.mock import Mock
    mock_printer = Mock()
    mock_printer.pformat.return_value = "sorted_result"
    input_tuple = (3, 1, 2)
    result = _tuple(input_tuple, mock_printer)
    assert mock_printer.pformat.call_args[0][0] == (1, 2, 3)
    assert result == "sorted_result"

def test__tuple_empty_tuple():
    from unittest.mock import Mock
    mock_printer = Mock()
    mock_printer.pformat.return_value = "empty_result"
    input_tuple = tuple()
    result = _tuple(input_tuple, mock_printer)
    assert mock_printer.pformat.call_args[0][0] == tuple()
    assert result == "empty_result"

def test__tuple_single_element():
    from unittest.mock import Mock
    mock_printer = Mock()
    mock_printer.pformat.return_value = "single_result"
    input_tuple = (42,)
    result = _tuple(input_tuple, mock_printer)
    assert mock_printer.pformat.call_args[0][0] == (42,)
    assert result == "single_result"


# LLM-generated content at query #9
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_with_invalid_sort_type():
    code = "b = 2\na = 1\n"
    try:
        assignment(code, "invalid_sort_type", "py")
    except ValueError as e:
        assert str(e) == "Trying to sort using an undefined sort_type. Defined sort types are assignments."

def test_assignment_with_literal_parsing_failure():
    code = "a = invalid_literal"
    try:
        assignment(code, "assignments", "py")
    except LiteralParsingFailure:
        pass

def test_assignment_with_literal_sort_type_mismatch():
    code = "a = [1, 2, 3]"
    try:
        assignment(code, "assignments", "py")
    except LiteralSortTypeMismatch:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test__tuple():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    value = (3, 1, 2)
    result = _tuple(value, printer)
    assert result == "(1, 2, 3)"


# LLM-generated content at query #11
#--------------------------

```python
def test_literal_sort_type_mismatch():
    config = Config(line_length=88)
    code = "x = [3, 1, 2]"
    sort_type = "dict"
    try:
        assignment(code, sort_type, ".py", config)
        assert False
    except LiteralSortTypeMismatch:
        assert True


# LLM-generated content at query #12
#--------------------------

```python
def test_assignment_with_invalid_literal():
    config = Config(line_length=80, formatting_function=None)
    assignment("x = invalid_literal", "strings", "py", config)


# LLM-generated content at query #13
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    input_code = "b = 2\na = 1"
    expected_output = "a = 1\nb = 2"
    assert assignment(input_code, "assignments", ".py") == expected_output

def test_assignment_with_invalid_sort_type():
    input_code = "a = 1"
    try:
        assignment(input_code, "invalid_type", ".py")
        assert False
    except ValueError:
        assert True

def test_assignment_with_literal_parsing_failure():
    input_code = "a = invalid_literal"
    try:
        assignment(input_code, "int", ".py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_with_type_mismatch():
    input_code = "a = 'string'"
    try:
        assignment(input_code, "int", ".py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_with_formatting():
    input_code = "a = {'b': 2, 'a': 1}"
    expected_output = "a = {'a': 1, 'b': 2}"
    assert assignment(input_code, "dict", ".py") == expected_output

def test_assignment_with_empty_input():
    input_code = ""
    try:
        assignment(input_code, "assignments", ".py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_with_single_line():
    input_code = "a = 1"
    assert assignment(input_code, "int", ".py") == "a = 1"


# LLM-generated content at query #14
#--------------------------

```python
def test_formatting_function_applied_when_present():
    class Config:
        def __init__(self, formatting_function):
            self.formatting_function = formatting_function
            self.line_length = 80

    def dummy_formatting_function(code, extension, config):
        return code.upper()

    config = Config(dummy_formatting_function)
    code = "x = 1"
    sorted_code = assignment(code, "assignments", ".py", config)
    assert sorted_code == "X = 1"


# LLM-generated content at query #15
#--------------------------

```python
def test_assignment_sorts_assignments():
    code = "b = 2\na = 1"
    sorted_code = assignment(code, "assignments", ".py")
    assert sorted_code == "a = 1b = 2"

def test_assignment_raises_error_for_invalid_sort_type():
    code = "a = 1"
    try:
        assignment(code, "invalid_sort_type", ".py")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_assignment_raises_error_for_invalid_literal():
    code = "a = invalid_literal"
    try:
        assignment(code, "list", ".py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass

def test_assignment_raises_error_for_type_mismatch():
    code = "a = 1"
    try:
        assignment(code, "dict", ".py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch:
        pass

def test_assignment_sorts_list():
    code = "a = [2, 1]"
    sorted_code = assignment(code, "list", ".py")
    assert sorted_code == "a = [1, 2]"

def test_assignment_sorts_dict():
    code = "a = {'b': 2, 'a': 1}"
    sorted_code = assignment(code, "dict", ".py")
    assert sorted_code == "a = {'a': 1, 'b': 2}"

def test_assignment_preserves_whitespace():
    code = "a = [2, 1]  \n"
    sorted_code = assignment(code, "list", ".py")
    assert sorted_code == "a = [1, 2]  \n"

def test_assignment_applies_formatting_function():
    code = "a = [2, 1]"
    config = Config(formatting_function=lambda x, y, z: x.upper())
    sorted_code = assignment(code, "list", ".py", config)
    assert sorted_code == "A = [1, 2]"


# LLM-generated content at query #16
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "b = 2\na = 1\n"
    sorted_code = assignment(code, "assignments", "py")
    assert sorted_code == "a = 1\nb = 2\n"

def test_assignment_with_invalid_sort_type():
    code = "a = 1\n"
    try:
        assignment(code, "invalid_sort_type", "py")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Trying to sort using an undefined sort_type. Defined sort types are assignments."

def test_assignment_with_literal_parsing_failure():
    code = "a = invalid_literal\n"
    try:
        assignment(code, "assignments", "py")
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure as e:
        assert str(e) == "Failed to parse literal in code: a = invalid_literal\n"

def test_assignment_with_literal_sort_type_mismatch():
    code = "a = 1\n"
    try:
        assignment(code, "assignments", "py")
        assert False, "Expected LiteralSortTypeMismatch"
    except LiteralSortTypeMismatch as e:
        assert str(e) == "Expected type <class 'str'>, but got <class 'int'>."

def test_assignment_with_formatting_function():
    code = "a = 1\n"
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    sorted_code = assignment(code, "assignments", "py", config)
    assert sorted_code == "A = 1\n"


# LLM-generated content at query #17
#--------------------------

```python
def test_assignment_with_invalid_literal():
    code = "invalid_literal = not_a_valid_literal"
    sort_type = "int"
    extension = "py"
    try:
        assignment(code, sort_type, extension)
    except LiteralParsingFailure:
        pass
    else:
        assert False, "Expected LiteralParsingFailure to be raised"


# LLM-generated content at query #18
#--------------------------

```
def test_assignment_applies_formatting_function_when_present():
    config = Config(formatting_function=lambda x, y, z: x.upper())
    result = assignment("x = 1", "numbers", "py", config)
    assert "X = 1" in result

def test_assignment_does_not_apply_formatting_function_when_absent():
    config = Config(formatting_function=None)
    result = assignment("x = 1", "numbers", "py", config)
    assert "x = 1" in result


# LLM-generated content at query #19
#--------------------------

```
def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1"
    expected = "a = 1\nb = 2"
    assert assignment(code, "assignments", ".py") == expected


def test_assignment_raises_format_mismatch():
    code = "invalid line"
    try:
        assignment(code, "assignments", ".py")
        assert False
    except AssignmentsFormatMismatch:
        assert True


def test_assignment_raises_undefined_sort_type():
    code = "a = 1"
    try:
        assignment(code, "invalid_type", ".py")
        assert False
    except ValueError:
        assert True


def test_assignment_raises_literal_parsing_failure():
    code = "a = invalid"
    try:
        assignment(code, "dict", ".py")
        assert False
    except LiteralParsingFailure:
        assert True


def test_assignment_raises_type_mismatch():
    code = "a = 1"
    try:
        assignment(code, "dict", ".py")
        assert False
    except LiteralSortTypeMismatch:
        assert True


def test_assignment_with_formatting_function():
    code = "a = {'b': 2, 'a': 1}"
    config = Config(formatting_function=lambda x, _, __: x.upper())
    expected = "A = {'A': 1, 'B': 2}"
    assert assignment(code, "dict", ".py", config) == expected


def test_assignment_preserves_trailing_whitespace():
    code = "a = {'b': 2, 'a': 1}  \n"
    expected = "a = {'a': 1, 'b': 2}  \n"
    assert assignment(code, "dict", ".py") == expected


# LLM-generated content at query #20
#--------------------------

```python
def test_assignment_with_valid_literal():
    code = "x = 42"
    sort_type = "int"
    extension = "py"
    config = Config()
    assignment(code, sort_type, extension, config)


# LLM-generated content at query #21
#--------------------------

```python
def test_config_formatting_function_evaluates_to_true():
    config = Config()
    config.formatting_function = lambda x, y, z: x
    code = "var = [2, 1]"
    sort_type = "lists"
    extension = ".py"
    result = assignment(code, sort_type, extension, config)
    assert result == "var = [1, 2]"


# LLM-generated content at query #22
#--------------------------

```python
def test_assignment_sorts_assignments():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_raises_format_mismatch_for_invalid_assignment():
    code = "a + 2"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_raises_value_error_for_undefined_sort_type():
    code = "a = [2, 1]"
    try:
        assignment(code, "invalid_sort_type", "py")
        assert False
    except ValueError:
        assert True

def test_assignment_raises_literal_parsing_failure_for_invalid_literal():
    code = "a = invalid_literal"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_raises_sort_type_mismatch_for_invalid_type():
    code = "a = {'key': 'value'}"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True


# LLM-generated content at query #23
#--------------------------

```python
def test_formatting_function_applied_when_config_has_formatting_function():
    config = Config(formatting_function=lambda code, extension, config: code.upper())
    code = "x = [3, 1, 2]"
    sorted_code = assignment(code, "lists", ".py", config)
    assert sorted_code == "X = [1, 2, 3]"


