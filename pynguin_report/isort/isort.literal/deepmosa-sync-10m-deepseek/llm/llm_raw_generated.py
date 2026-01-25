####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_unique_tuple_sorts_and_removes_duplicates():
    printer = ISortPrettyPrinter(Config())
    result = _unique_tuple((3, 1, 2, 1), printer)
    assert result == "(1, 2, 3)"

def test_unique_tuple_with_empty_tuple():
    printer = ISortPrettyPrinter(Config())
    result = _unique_tuple((), printer)
    assert result == "()"

def test_unique_tuple_with_single_element():
    printer = ISortPrettyPrinter(Config())
    result = _unique_tuple((5,), printer)
    assert result == "(5,)"


# LLM-generated content at query #2
#--------------------------

def test_assignment_assignments():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_empty_lines():
    code = "b = 2\n\n\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_keep_trailing_newline():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_format_mismatch():
    code = "b = 2\na 1\n"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_list_sort():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]"

def test_assignment_dict_sort():
    code = "x = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", "py")
    assert result == "x = {'a': 1, 'b': 2}"

def test_assignment_set_sort():
    code = "x = {3, 1, 2}"
    result = assignment(code, "set", "py")
    assert result == "x = {1, 2, 3}"

def test_assignment_tuple_sort():
    code = "x = (3, 1, 2)"
    result = assignment(code, "tuple", "py")
    assert result == "x = (1, 2, 3)"

def test_assignment_undefined_sort_type():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "undefined", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_literal_parsing_failure():
    code = "x = [1, 2"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_type_mismatch():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_with_formatting_function():
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "x = [1, 2, 3]"
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2, 3]"

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   \n"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]   \n"

def test_assignment_compact_printer():
    config = Config(line_length=10, compact=True)
    code = "x = [1, 2, 3, 4, 5]"
    result = assignment(code, "list", "py", config)
    assert result == "x = [1, 2, 3, 4, 5]"


# LLM-generated content at query #3
#--------------------------

def test_assignment_raises_literal_sort_type_mismatch_when_type_mismatch():
    import ast
    from isort.pretty_printer import ISortPrettyPrinter
    from isort.exceptions import LiteralSortTypeMismatch
    from isort.type_checking import Config
    config = Config()
    type_mapping = {"list": (list, lambda x, p: p.pformat(sorted(x)))}
    code = "my_var = {1, 2, 3}"
    variable_name, literal = code.split("=")
    variable_name = variable_name.strip()
    literal = literal.lstrip()
    value = ast.literal_eval(literal)
    expected_type, sort_function = type_mapping["list"]
    assert type(value) is not expected_type


# LLM-generated content at query #4
#--------------------------

def test_unique_list_sorts_and_removes_duplicates():
    printer = ISortPrettyPrinter(Config(line_length=80))
    result = _unique_list([3, 1, 2, 3, 1], printer)
    assert result == '[1, 2, 3]'

def test_unique_list_with_empty_list():
    printer = ISortPrettyPrinter(Config(line_length=80))
    result = _unique_list([], printer)
    assert result == '[]'

def test_unique_list_with_single_element():
    printer = ISortPrettyPrinter(Config(line_length=80))
    result = _unique_list([5], printer)
    assert result == '[5]'

def test_unique_list_with_strings():
    printer = ISortPrettyPrinter(Config(line_length=80))
    result = _unique_list(['banana', 'apple', 'apple', 'cherry'], printer)
    assert result == "['apple', 'banana', 'cherry']"

def test_unique_list_with_mixed_types_that_are_sortable():
    printer = ISortPrettyPrinter(Config(line_length=80))
    result = _unique_list([3, 1.5, 2, 1.5], printer)
    assert result == '[1.5, 2, 3]'


# LLM-generated content at query #5
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    code = "x = invalid_literal"
    sort_type = "list"
    extension = "py"
    config = Config()
    try:
        assignment(code, sort_type, extension, config)
        assert False
    except LiteralParsingFailure:
        assert True


# LLM-generated content at query #6
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = None
    try:
        result = assignment("x = invalid", "dict", ".py")
    except Exception as e:
        assert isinstance(e, LiteralParsingFailure)
    assert result is None


# LLM-generated content at query #7
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_with_empty_lines():
    code = "b = 2\n\n\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_raises_format_mismatch():
    try:
        assignment("invalid line", "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_list_sort_type():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]"

def test_assignment_dict_sort_type():
    code = "x = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", "py")
    assert result == "x = {'a': 1, 'b': 2}"

def test_assignment_set_sort_type():
    code = "x = {3, 1, 2}"
    result = assignment(code, "set", "py")
    assert result == "x = {1, 2, 3}"

def test_assignment_tuple_sort_type():
    code = "x = (3, 1, 2)"
    result = assignment(code, "tuple", "py")
    assert result == "x = (1, 2, 3)"

def test_assignment_raises_value_error_for_undefined_sort_type():
    try:
        assignment("x = [1, 2, 3]", "undefined", "py")
        assert False
    except ValueError:
        assert True

def test_assignment_raises_literal_parsing_failure():
    try:
        assignment("x = invalid_literal", "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_raises_literal_sort_type_mismatch():
    try:
        assignment("x = [1, 2, 3]", "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   "
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]   "

def test_assignment_with_formatting_function():
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2, 3]"


# LLM-generated content at query #8
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = None
    try:
        result = assignment("x = invalid", "list", "py")
    except Exception as e:
        result = e
    assert isinstance(result, LiteralParsingFailure)


# LLM-generated content at query #9
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    code = "x = invalid_literal"
    sort_type = "list"
    extension = "py"
    config = Config()
    try:
        assignment(code, sort_type, extension, config)
        assert False
    except LiteralParsingFailure:
        assert True


# LLM-generated content at query #10
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_with_empty_lines():
    code = "b = 2\n\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_raises_format_mismatch():
    try:
        assignment("invalid line", "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_list_sort_type():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]"

def test_assignment_dict_sort_type():
    code = "x = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", "py")
    assert result == "x = {'a': 1, 'b': 2}"

def test_assignment_tuple_sort_type():
    code = "x = (3, 1, 2)"
    result = assignment(code, "tuple", "py")
    assert result == "x = (1, 2, 3)"

def test_assignment_set_sort_type():
    code = "x = {3, 1, 2}"
    result = assignment(code, "set", "py")
    assert result == "x = {1, 2, 3}"

def test_assignment_raises_undefined_sort_type():
    try:
        assignment("x = [1, 2]", "invalid", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_raises_literal_parsing_failure():
    try:
        assignment("x = invalid_literal", "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_raises_literal_sort_type_mismatch():
    try:
        assignment("x = [1, 2]", "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   "
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]   "

def test_assignment_with_formatting_function():
    config = Config()
    config.formatting_function = lambda code, ext, cfg: code.upper()
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2, 3]"

def test_assignment_with_custom_line_length():
    config = Config(line_length=10)
    code = "x = [3, 1, 2, 4, 5, 6]"
    result = assignment(code, "list", "py", config)
    assert result == "x = [1, 2, 3, 4, 5, 6]"


# LLM-generated content at query #11
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    expected = "a = 1\nb = 2"
    assert result == expected

def test_assignment_assignments_sort_type_with_trailing_whitespace():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    expected = "a = 1\nb = 2\n"
    assert result == expected

def test_assignment_assignments_sort_type_empty_lines():
    code = "b = 2\n\na = 1"
    result = assignment(code, "assignments", "py")
    expected = "a = 1\nb = 2"
    assert result == expected

def test_assignment_assignments_sort_type_single_assignment():
    code = "a = 1"
    result = assignment(code, "assignments", "py")
    expected = "a = 1"
    assert result == expected

def test_assignment_assignments_sort_type_no_equals_raises():
    code = "invalid line"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_undefined_sort_type_raises():
    code = "x = [1, 2]"
    try:
        assignment(code, "undefined_type", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_list_sort_type():
    code = "x = [3, 1, 2]"
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    result = assignment(code, "list", "py")
    expected = "x = [1, 2, 3]"
    assert result == expected

def test_assignment_dict_sort_type():
    code = "x = {'b': 2, 'a': 1}"
    type_mapping["dict"] = (dict, lambda v, p: p.pformat(dict(sorted(v.items()))))
    result = assignment(code, "dict", "py")
    expected = "x = {'a': 1, 'b': 2}"
    assert result == expected

def test_assignment_type_mismatch_raises():
    code = "x = 'not a list'"
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_literal_parsing_failure_raises():
    code = "x = invalid_literal"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_with_formatting_function():
    code = "x = [3, 1, 2]"
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    result = assignment(code, "list", "py", config)
    expected = "X = [1, 2, 3]"
    assert result == expected

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   "
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    result = assignment(code, "list", "py")
    expected = "x = [1, 2, 3]   "
    assert result == expected

def test_assignment_with_compact_printer():
    code = "x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
    config = Config(line_length=20, compact=True)
    type_mapping["list"] = (list, lambda v, p: p.pformat(v))
    result = assignment(code, "list", "py", config)
    assert len(result.splitlines()) > 1


# LLM-generated content at query #12
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = None
    try:
        result = assignment("x = invalid", "lists", "py", Config())
    except Exception as e:
        assert isinstance(e, LiteralParsingFailure)
    assert result is None


# LLM-generated content at query #13
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_sort_type_with_trailing_whitespace():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_empty_lines():
    code = "b = 2\n\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_sort_type_raises_format_mismatch():
    code = "b = 2\ninvalid_line"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_list_sort_type():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]"

def test_assignment_dict_sort_type():
    code = "x = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", "py")
    assert result == "x = {'a': 1, 'b': 2}"

def test_assignment_set_sort_type():
    code = "x = {3, 1, 2}"
    result = assignment(code, "set", "py")
    assert result == "x = {1, 2, 3}"

def test_assignment_tuple_sort_type():
    code = "x = (3, 1, 2)"
    result = assignment(code, "tuple", "py")
    assert result == "x = (1, 2, 3)"

def test_assignment_raises_undefined_sort_type():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "invalid", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_raises_literal_parsing_failure():
    code = "x = invalid_literal"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_raises_literal_sort_type_mismatch():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_with_formatting_function():
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "x = [1, 2, 3]"
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2, 3]"

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   "
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]   "

def test_assignment_with_compact_printer():
    config = Config(line_length=10, compact=True)
    code = "x = [1, 2, 3, 4, 5]"
    result = assignment(code, "list", "py", config)
    assert result == "x = [1, 2, 3, 4, 5]"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_sort_type_with_empty_lines():
    code = "b = 2\n\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_sort_type_single_assignment():
    code = "a = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1"

def test_assignment_assignments_sort_type_raises_format_mismatch():
    try:
        assignment("invalid line", "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_valid_sort_type_list():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]"

def test_assignment_valid_sort_type_dict():
    code = "x = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", "py")
    assert result == "x = {'a': 1, 'b': 2}"

def test_assignment_raises_undefined_sort_type():
    try:
        assignment("x = [1, 2]", "invalid", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_raises_literal_parsing_failure():
    try:
        assignment("x = invalid_literal", "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_raises_literal_sort_type_mismatch():
    try:
        assignment("x = [1, 2]", "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   "
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]   "

def test_assignment_applies_formatting_function():
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2, 3]"


# LLM-generated content at query #2
#--------------------------

def test_tuple_sorts_and_formats_correctly():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    test_tuple = (3, 1, 2)
    result = _tuple(test_tuple, printer)
    expected = printer.pformat((1, 2, 3))
    assert result == expected

def test_tuple_with_single_element():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    test_tuple = (5,)
    result = _tuple(test_tuple, printer)
    expected = printer.pformat((5,))
    assert result == expected

def test_tuple_with_duplicates():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    test_tuple = (2, 1, 2, 3, 1)
    result = _tuple(test_tuple, printer)
    expected = printer.pformat((1, 1, 2, 2, 3))
    assert result == expected

def test_empty_tuple():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    test_tuple = ()
    result = _tuple(test_tuple, printer)
    expected = printer.pformat(())
    assert result == expected

def test_tuple_with_strings():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    test_tuple = ("banana", "apple", "cherry")
    result = _tuple(test_tuple, printer)
    expected = printer.pformat(("apple", "banana", "cherry"))
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    code = "my_var = invalid_literal"
    sort_type = "list"
    extension = "py"
    config = Config()
    try:
        assignment(code, sort_type, extension, config)
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass


# LLM-generated content at query #4
#--------------------------

def test_assignment_raises_literal_sort_type_mismatch_when_type_mismatch():
    import ast
    from typing import Any, Tuple, Type
    from unittest.mock import Mock
    type_mapping = {"list": (list, lambda x, p: p.pformat(sorted(x)))}
    expected_type, sort_function = type_mapping["list"]
    value = "not_a_list"
    assert type(value) is not expected_type


# LLM-generated content at query #5
#--------------------------

def test_unique_tuple_sorts_and_removes_duplicates():
    printer = ISortPrettyPrinter(Config())
    result = _unique_tuple((3, 1, 2, 1, 3), printer)
    assert result == "(1, 2, 3)"

def test_unique_tuple_with_empty_tuple():
    printer = ISortPrettyPrinter(Config())
    result = _unique_tuple((), printer)
    assert result == "()"

def test_unique_tuple_with_single_element():
    printer = ISortPrettyPrinter(Config())
    result = _unique_tuple((5,), printer)
    assert result == "(5,)"

def test_unique_tuple_with_already_sorted_unique_elements():
    printer = ISortPrettyPrinter(Config())
    result = _unique_tuple((10, 20, 30), printer)
    assert result == "(10, 20, 30)"

def test_unique_tuple_with_strings():
    printer = ISortPrettyPrinter(Config())
    result = _unique_tuple(("banana", "apple", "cherry", "apple"), printer)
    assert result == "('apple', 'banana', 'cherry')"

def test_unique_tuple_with_mixed_types_that_are_sortable():
    printer = ISortPrettyPrinter(Config())
    result = _unique_tuple((3.5, 1, 2.0), printer)
    assert result == "(1, 2.0, 3.5)"


# LLM-generated content at query #6
#--------------------------

def test_assignment_raises_literal_sort_type_mismatch_when_type_mismatch():
    config = Config()
    code = "x = [1, 2, 3]"
    sort_type = "dict"
    extension = "py"
    try:
        assignment(code, sort_type, extension, config)
        assert False
    except LiteralSortTypeMismatch as e:
        assert True


# LLM-generated content at query #7
#--------------------------

def test_set_with_empty_set():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set(set(), printer)
    assert result == "{}"

def test_set_with_integers():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({3, 1, 2}, printer)
    assert result == "{1, 2, 3}"

def test_set_with_strings():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({"c", "a", "b"}, printer)
    assert result == "{'a', 'b', 'c'}"

def test_set_with_mixed_types():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({2, "a", 1}, printer)
    assert result == "{1, 2, 'a'}"

def test_set_with_duplicates():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({2, 1, 2, 1}, printer)
    assert result == "{1, 2}"

def test_set_with_nested_sets():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({frozenset({2, 1}), frozenset({4, 3})}, printer)
    assert result == "{frozenset({1, 2}), frozenset({3, 4})}"

def test_set_with_line_length_limit():
    config = Config(line_length=20)
    printer = ISortPrettyPrinter(config)
    result = _set({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, printer)
    assert result == "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}"

def test_set_with_single_element():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({42}, printer)
    assert result == "{42}"


# LLM-generated content at query #8
#--------------------------

def test_unique_list_sorts_and_removes_duplicates():
    printer = ISortPrettyPrinter(Config())
    result = _unique_list([3, 1, 2, 3, 1], printer)
    assert result == "[1, 2, 3]"

def test_unique_list_with_empty_list():
    printer = ISortPrettyPrinter(Config())
    result = _unique_list([], printer)
    assert result == "[]"

def test_unique_list_with_single_element():
    printer = ISortPrettyPrinter(Config())
    result = _unique_list([5], printer)
    assert result == "[5]"

def test_unique_list_with_strings():
    printer = ISortPrettyPrinter(Config())
    result = _unique_list(["banana", "apple", "cherry", "apple"], printer)
    assert result == "['apple', 'banana', 'cherry']"

def test_unique_list_with_mixed_types_raises_error():
    printer = ISortPrettyPrinter(Config())
    try:
        _unique_list([1, "a", 2], printer)
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #9
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_sort_type_with_empty_lines():
    code = "b = 2\n\n\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_sort_type_keepends():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_sort_type_no_equals_raises():
    code = "b  2\na = 1"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_list_sort_type():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]"

def test_assignment_dict_sort_type():
    code = "x = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", "py")
    assert result == "x = {'a': 1, 'b': 2}"

def test_assignment_set_sort_type():
    code = "x = {3, 1, 2}"
    result = assignment(code, "set", "py")
    assert result == "x = {1, 2, 3}"

def test_assignment_tuple_sort_type():
    code = "x = (3, 1, 2)"
    result = assignment(code, "tuple", "py")
    assert result == "x = (1, 2, 3)"

def test_assignment_undefined_sort_type_raises():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "undefined", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_literal_parsing_failure_raises():
    code = "x = not_a_literal"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_type_mismatch_raises():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_with_formatting_function():
    code = "x = [3, 1, 2]"
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2, 3]"

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   "
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]   "

def test_assignment_with_compact_printer():
    code = "x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
    config = Config(line_length=20, compact=True)
    result = assignment(code, "list", "py", config)
    assert "[" in result and "]" in result and "x = " in result


# LLM-generated content at query #10
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = None
    try:
        result = assignment("x = invalid", "lists", "py", Config())
    except Exception as e:
        result = e
    assert isinstance(result, LiteralParsingFailure)


# LLM-generated content at query #11
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    code = "my_var = invalid_literal"
    sort_type = "list"
    extension = "py"
    config = Config()
    try:
        assignment(code, sort_type, extension, config)
        assert False
    except LiteralParsingFailure:
        assert True


# LLM-generated content at query #12
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = assignment("x = invalid", "lists", ".py")


# LLM-generated content at query #13
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_with_empty_lines():
    code = "b = 2\n\n\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_raises_format_mismatch():
    code = "b = 2\na 1\n"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_list_sort_type():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]"

def test_assignment_dict_sort_type():
    code = "x = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", "py")
    assert result == "x = {'a': 1, 'b': 2}"

def test_assignment_undefined_sort_type_raises_value_error():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "undefined", "py")
        assert False
    except ValueError:
        assert True

def test_assignment_literal_parsing_failure():
    code = "x = [1, 2,"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_type_mismatch():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_with_formatting_function():
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2, 3]"

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   \n"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]   \n"


# LLM-generated content at query #14
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_with_trailing_whitespace():
    code = "b = 2\na = 1\n\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n\n"

def test_assignment_assignments_sort_type_empty_lines_ignored():
    code = "\nb = 2\n\na = 1\n\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n\n"

def test_assignment_assignments_sort_type_single_assignment():
    code = "a = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\n"

def test_assignment_assignments_sort_type_no_equals_raises():
    code = "a 1\n"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_assignments_sort_type_multiple_equals():
    code = "a = b = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = b = 1\n"

def test_assignment_undefined_sort_type_raises():
    code = "a = [2, 1]"
    try:
        assignment(code, "undefined", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_list_sort_type():
    code = "a = [2, 1]"
    result = assignment(code, "list", "py")
    assert result == "a = [1, 2]"

def test_assignment_list_sort_type_with_formatting():
    config = Config(line_length=10, formatting_function=lambda x, y, z: x)
    code = "a = [2, 1]"
    result = assignment(code, "list", "py", config)
    assert result == "a = [1, 2]"

def test_assignment_tuple_sort_type():
    code = "a = (2, 1)"
    result = assignment(code, "tuple", "py")
    assert result == "a = (1, 2)"

def test_assignment_set_sort_type():
    code = "a = {2, 1}"
    result = assignment(code, "set", "py")
    assert result == "a = {1, 2}"

def test_assignment_dict_sort_type():
    code = "a = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", "py")
    assert result == "a = {'a': 1, 'b': 2}"

def test_assignment_literal_parsing_failure_raises():
    code = "a = [1, 2"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_literal_sort_type_mismatch_raises():
    code = "a = [1, 2]"
    try:
        assignment(code, "tuple", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_preserves_trailing_whitespace():
    code = "a = [2, 1]   "
    result = assignment(code, "list", "py")
    assert result == "a = [1, 2]   "

def test_assignment_with_formatting_function():
    def formatter(code, ext, conf):
        return code.upper()
    config = Config(formatting_function=formatter)
    code = "a = [2, 1]"
    result = assignment(code, "list", "py", config)
    assert result == "A = [1, 2]"


# LLM-generated content at query #15
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_with_empty_lines():
    code = "b = 2\n\n\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_raises_format_mismatch():
    try:
        assignment("invalid line", "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_list_sort_type():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]"

def test_assignment_dict_sort_type():
    code = "x = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", "py")
    assert result == "x = {'a': 1, 'b': 2}"

def test_assignment_set_sort_type():
    code = "x = {3, 1, 2}"
    result = assignment(code, "set", "py")
    assert result == "x = {1, 2, 3}"

def test_assignment_tuple_sort_type():
    code = "x = (3, 1, 2)"
    result = assignment(code, "tuple", "py")
    assert result == "x = (1, 2, 3)"

def test_assignment_raises_value_error_for_undefined_sort_type():
    try:
        assignment("x = [1, 2, 3]", "undefined", "py")
        assert False
    except ValueError:
        assert True

def test_assignment_raises_literal_parsing_failure():
    try:
        assignment("x = invalid_literal", "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_raises_literal_sort_type_mismatch():
    try:
        assignment("x = [1, 2, 3]", "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   \n"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]   \n"

def test_assignment_with_formatting_function():
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2, 3]"


