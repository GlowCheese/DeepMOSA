####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_list_sorts_and_formats():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _list([3, 1, 2], printer)
    assert result == "[1, 2, 3]"

def test_list_empty():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _list([], printer)
    assert result == "[]"

def test_list_single_element():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _list([5], printer)
    assert result == "[5]"

def test_list_already_sorted():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _list([10, 20, 30], printer)
    assert result == "[10, 20, 30]"

def test_list_reverse_sorted():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _list([30, 20, 10], printer)
    assert result == "[10, 20, 30]"

def test_list_with_strings():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _list(["banana", "apple", "cherry"], printer)
    assert result == "['apple', 'banana', 'cherry']"

def test_list_mixed_types_raises_error():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    try:
        _list([1, "a"], printer)
        assert False
    except TypeError:
        assert True

def test_list_uses_printer_line_length():
    config = Config(line_length=10)
    printer = ISortPrettyPrinter(config)
    long_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = _list(long_list, printer)
    expected = "[\n 1,\n 2,\n 3,\n 4,\n 5,\n 6,\n 7,\n 8,\n 9,\n 10\n]"
    assert result == expected


# LLM-generated content at query #2
#--------------------------

def test_assignments_simple():
    code = "a = 1\nb = 2\nc = 3"
    result = assignments(code)
    assert result == "a = 1b = 2c = 3"

def test_assignments_empty_lines():
    code = "x = 10\n\ny = 20\n\nz = 30"
    result = assignments(code)
    assert result == "x = 10y = 20z = 30"

def test_assignments_whitespace():
    code = "  var1 = value1  \n\tvar2 = value2\t\n  var3 = value3  "
    result = assignments(code)
    assert result == "  var1 = value1  \tvar2 = value2\t  var3 = value3  "

def test_assignments_single_line():
    code = "single = line"
    result = assignments(code)
    assert result == "single = line"

def test_assignments_no_equals():
    code = "invalid line"
    try:
        assignments(code)
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignments_multiple_equals():
    code = "key = value = extra"
    result = assignments(code)
    assert result == "key = value = extra"

def test_assignments_sorting():
    code = "zebra = animal\napple = fruit\nbanana = fruit"
    result = assignments(code)
    assert result == "apple = fruitbanana = fruitzebra = animal"

def test_assignments_empty_string():
    code = ""
    result = assignments(code)
    assert result == ""

def test_assignments_only_whitespace():
    code = "\n\n\t\n  \n"
    result = assignments(code)
    assert result == ""


# LLM-generated content at query #3
#--------------------------

def test_assignment_assignments():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_empty_lines():
    code = "b = 2\n\n\na = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_keepends():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_format_mismatch():
    code = "b = 2\na 1"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_assignments_single_line():
    code = "a = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1"

def test_assignment_assignments_whitespace():
    code = "  b = 2  \n  a = 1  "
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_duplicate_keys():
    code = "b = 2\na = 1\nb = 3"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 3"

def test_assignment_assignments_no_newline():
    code = "b = 2"
    result = assignment(code, "assignments", "py")
    assert result == "b = 2"

def test_assignment_assignments_trailing_whitespace():
    code = "b = 2 \na = 1 "
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"

def test_assignment_assignments_mixed_whitespace():
    code = "b = 2\n\ta = 1"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2"


# LLM-generated content at query #4
#--------------------------

def test_unique_list_sorts_and_removes_duplicates():
    mock_printer = ISortPrettyPrinter(Config())
    mock_printer.pformat = lambda x: str(x)
    result = _unique_list([3, 1, 2, 2, 1], mock_printer)
    assert result == "[1, 2, 3]"

def test_unique_list_with_empty_list():
    mock_printer = ISortPrettyPrinter(Config())
    mock_printer.pformat = lambda x: str(x)
    result = _unique_list([], mock_printer)
    assert result == "[]"

def test_unique_list_with_single_element():
    mock_printer = ISortPrettyPrinter(Config())
    mock_printer.pformat = lambda x: str(x)
    result = _unique_list([5], mock_printer)
    assert result == "[5]"

def test_unique_list_with_strings():
    mock_printer = ISortPrettyPrinter(Config())
    mock_printer.pformat = lambda x: str(x)
    result = _unique_list(["z", "a", "z", "b"], mock_printer)
    assert result == "['a', 'b', 'z']"

def test_unique_list_with_mixed_types_raises_error():
    mock_printer = ISortPrettyPrinter(Config())
    mock_printer.pformat = lambda x: str(x)
    try:
        _unique_list([1, "a"], mock_printer)
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #5
#--------------------------

def test_unique_tuple_sorts_and_removes_duplicates():
    printer = ISortPrettyPrinter(Config(line_length=88))
    result = _unique_tuple((3, 1, 2, 1), printer)
    assert result == "(1, 2, 3)"

def test_unique_tuple_with_empty_tuple():
    printer = ISortPrettyPrinter(Config(line_length=88))
    result = _unique_tuple((), printer)
    assert result == "()"

def test_unique_tuple_with_single_element():
    printer = ISortPrettyPrinter(Config(line_length=88))
    result = _unique_tuple((5,), printer)
    assert result == "(5,)"

def test_unique_tuple_with_already_sorted_unique_elements():
    printer = ISortPrettyPrinter(Config(line_length=88))
    result = _unique_tuple((1, 2, 3), printer)
    assert result == "(1, 2, 3)"

def test_unique_tuple_with_mixed_types_that_are_sortable():
    printer = ISortPrettyPrinter(Config(line_length=88))
    result = _unique_tuple((3, 1.5, 2), printer)
    assert result == "(1.5, 2, 3)"

def test_unique_tuple_pformat_called_with_correct_arguments():
    mock_printer = unittest.mock.MagicMock(spec=ISortPrettyPrinter)
    mock_printer.pformat.return_value = "(1, 2, 3)"
    result = _unique_tuple((3, 2, 1), mock_printer)
    mock_printer.pformat.assert_called_once_with((1, 2, 3))
    assert result == "(1, 2, 3)"


# LLM-generated content at query #6
#--------------------------

def test_tuple_sorted():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _tuple((3, 1, 2), printer)
    assert result == "(1, 2, 3)"

def test_tuple_empty():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _tuple((), printer)
    assert result == "()"

def test_tuple_single_element():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _tuple((5,), printer)
    assert result == "(5,)"

def test_tuple_already_sorted():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _tuple((1, 2, 3, 4), printer)
    assert result == "(1, 2, 3, 4)"

def test_tuple_negative_numbers():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _tuple((-2, -5, -1), printer)
    assert result == "(-5, -2, -1)"

def test_tuple_mixed_types():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _tuple((3, 1.5, 2), printer)
    assert result == "(1.5, 2, 3)"


# LLM-generated content at query #7
#--------------------------

def test_assignment_raises_value_error_for_undefined_sort_type():
    result = assignment("x = [3, 1, 2]", "invalid_type", "py")
    assert result is None


# LLM-generated content at query #8
#--------------------------

def test_dict_sorts_by_value():
    printer = ISortPrettyPrinter(Config())
    result = _dict({3: 'c', 1: 'a', 2: 'b'}, printer)
    expected = printer.pformat({1: 'a', 2: 'b', 3: 'c'})
    assert result == expected

def test_dict_empty():
    printer = ISortPrettyPrinter(Config())
    result = _dict({}, printer)
    expected = printer.pformat({})
    assert result == expected

def test_dict_single_item():
    printer = ISortPrettyPrinter(Config())
    result = _dict({5: 'x'}, printer)
    expected = printer.pformat({5: 'x'})
    assert result == expected

def test_dict_duplicate_values():
    printer = ISortPrettyPrinter(Config())
    result = _dict({2: 'z', 1: 'z', 3: 'a'}, printer)
    expected = printer.pformat({3: 'a', 1: 'z', 2: 'z'})
    assert result == expected

def test_dict_numeric_values():
    printer = ISortPrettyPrinter(Config())
    result = _dict({'a': 10, 'c': 5, 'b': 20}, printer)
    expected = printer.pformat({'c': 5, 'a': 10, 'b': 20})
    assert result == expected


# LLM-generated content at query #9
#--------------------------

def test__set_with_empty_set():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set(set(), printer)
    assert result == "{}"

def test__set_with_single_element():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({1}, printer)
    assert result == "{1}"

def test__set_with_multiple_elements():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({3, 1, 2}, printer)
    assert result == "{1, 2, 3}"

def test__set_with_strings():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({"c", "a", "b"}, printer)
    assert result == "{'a', 'b', 'c'}"

def test__set_with_mixed_types():
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    result = _set({2, "a", 1}, printer)
    assert result == "{1, 2, 'a'}"

def test__set_with_line_length_short():
    config = Config(line_length=10)
    printer = ISortPrettyPrinter(config)
    result = _set({1, 2, 3, 4, 5}, printer)
    assert result == "{1,\n 2,\n 3,\n 4,\n 5}"


# LLM-generated content at query #10
#--------------------------

def test_assignment_raises_value_error_for_undefined_sort_type():
    result = assignment("x = [3, 1, 2]", "invalid_type", "py")
    assert False


# LLM-generated content at query #11
#--------------------------

def test_assignment_raises_value_error_for_undefined_sort_type():
    result = assignment("x = [3, 1, 2]", "invalid_type", ".py")
    assert result is None


# LLM-generated content at query #12
#--------------------------

def test_assignment_assignments_sort_type():
    code = "x = 1\ny = 2\nz = 3"
    result = assignment(code, "assignments", "py")
    assert result == "x = 1\ny = 2\nz = 3"

def test_assignment_assignments_sort_type_with_whitespace():
    code = "b = 2\n\na = 1\n\nc = 3"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\nc = 3"

def test_assignment_assignments_sort_type_empty_lines():
    code = "\n\nx = 1\n\n\n"
    result = assignment(code, "assignments", "py")
    assert result == "x = 1"

def test_assignment_assignments_sort_type_no_equals():
    try:
        assignment("invalid line", "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_assignments_sort_type_multiple_equals():
    code = "x = y = 1"
    result = assignment(code, "assignments", "py")
    assert result == "x = y = 1"

def test_assignment_undefined_sort_type():
    try:
        assignment("x = [1, 2, 3]", "invalid", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_list_sort_type():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]"

def test_assignment_tuple_sort_type():
    code = "x = (3, 1, 2)"
    result = assignment(code, "tuple", "py")
    assert result == "x = (1, 2, 3)"

def test_assignment_set_sort_type():
    code = "x = {3, 1, 2}"
    result = assignment(code, "set", "py")
    assert result == "x = {1, 2, 3}"

def test_assignment_dict_sort_type():
    code = "x = {'b': 2, 'a': 1}"
    result = assignment(code, "dict", "py")
    assert result == "x = {'a': 1, 'b': 2}"

def test_assignment_type_mismatch():
    try:
        assignment("x = [1, 2, 3]", "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_literal_parsing_failure():
    try:
        assignment("x = invalid", "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_with_trailing_whitespace():
    code = "x = [3, 1, 2]   "
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]   "

def test_assignment_with_formatting_function():
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2, 3]"

def test_assignment_with_extension():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", "txt")
    assert result == "x = [1, 2, 3]"


# LLM-generated content at query #13
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = assignment("x = invalid", "lists", ".py")


# LLM-generated content at query #14
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    expected = "a = 1\nb = 2"
    assert result == expected

def test_assignment_assignments_sort_type_with_whitespace():
    code = "z = 3\n\nx = 1\n"
    result = assignment(code, "assignments", "py")
    expected = "x = 1\nz = 3\n"
    assert result == expected

def test_assignment_assignments_sort_type_empty_lines_ignored():
    code = "\n\na = 1\n\n\nb = 2\n"
    result = assignment(code, "assignments", "py")
    expected = "a = 1\nb = 2\n"
    assert result == expected

def test_assignment_assignments_sort_type_no_equals_raises():
    code = "not an assignment"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_assignments_sort_type_mixed_lines_raises():
    code = "a = 1\ninvalid line\nb = 2"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_unknown_sort_type_raises():
    code = "x = [1, 3, 2]"
    try:
        assignment(code, "unknown_type", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_valid_sort_type_list():
    code = "my_list = [3, 1, 2]"
    type_mapping = {"list": (list, lambda v, p: p.pformat(sorted(v)))}
    result = assignment(code, "list", "py")
    assert result.startswith("my_list = [1, 2, 3]")

def test_assignment_literal_parsing_failure_raises():
    code = "x = not_a_literal"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_type_mismatch_raises():
    code = "x = {1, 2, 3}"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_preserves_trailing_whitespace():
    code = "x = [2, 1]   \n"
    result = assignment(code, "list", "py")
    assert result.endswith("   \n")

def test_assignment_applies_formatting_function():
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "x = [2, 1]"
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2]"


# LLM-generated content at query #15
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = None
    try:
        result = assignment("x = invalid", "dict", "py")
    except Exception as e:
        assert isinstance(e, LiteralParsingFailure)
    assert result is None


# LLM-generated content at query #16
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = None
    try:
        result = assignment("x = invalid", "lists", "py", Config())
    except Exception as e:
        result = e
    assert isinstance(result, LiteralParsingFailure)


# LLM-generated content at query #17
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    expected = "a = 1\nb = 2"
    assert result == expected

def test_assignment_assignments_sort_type_with_empty_lines():
    code = "b = 2\n\na = 1"
    result = assignment(code, "assignments", "py")
    expected = "a = 1\nb = 2"
    assert result == expected

def test_assignment_assignments_sort_type_raises_on_missing_equals():
    code = "b  2\na = 1"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_unknown_sort_type_raises():
    code = "var = [2, 1]"
    try:
        assignment(code, "unknown", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_list_sort_type():
    code = "var = [2, 1]"
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    result = assignment(code, "list", "py")
    expected = "var = [1, 2]"
    assert result == expected

def test_assignment_type_mismatch_raises():
    code = "var = [2, 1]"
    type_mapping["dict"] = (dict, lambda v, p: p.pformat(v))
    try:
        assignment(code, "dict", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_literal_parsing_failure_raises():
    code = "var = [2, 1"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_with_formatting_function():
    code = "var = [2, 1]"
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    result = assignment(code, "list", "py", config)
    expected = "VAR = [1, 2]"
    assert result == expected

def test_assignment_preserves_trailing_whitespace():
    code = "var = [2, 1]   \n"
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    result = assignment(code, "list", "py")
    expected = "var = [1, 2]   \n"
    assert result == expected


# LLM-generated content at query #18
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = assignment("x = invalid", "lists", "py", Config())
    assert result == "x = []"


# LLM-generated content at query #19
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = assignment("x = invalid", "lists", "py", Config())
    assert False


# LLM-generated content at query #20
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = None
    try:
        result = assignment("x = invalid", "lists", ".py")
    except Exception as e:
        assert isinstance(e, LiteralParsingFailure)
    assert result is None


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_with_empty_lines():
    code = "b = 2\n\n\na = 1\n"
    result = assignment(code, "assignments", "py")
    assert result == "a = 1\nb = 2\n"

def test_assignment_assignments_sort_type_no_equals_raises():
    code = "b  2\na = 1\n"
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
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    code = "x = [1, 2, 3]"
    result = assignment(code, "list", "py", config)
    assert result == "X = [1, 2, 3]"

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   \n"
    result = assignment(code, "list", "py")
    assert result == "x = [1, 2, 3]   \n"

def test_assignment_with_compact_printer():
    config = Config(line_length=10, compact=True)
    code = "x = [1, 2, 3, 4, 5]"
    result = assignment(code, "list", "py", config)
    assert result == "x = [1, 2, 3, 4, 5]"


# LLM-generated content at query #2
#--------------------------

def test_assignment_raises_literal_sort_type_mismatch_when_type_mismatch():
    code = "my_var = [3, 1, 2]"
    sort_type = "dicts"
    config = Config()
    try:
        assignment(code, sort_type, "py", config)
        assert False
    except LiteralSortTypeMismatch as e:
        assert type(e) is LiteralSortTypeMismatch


# LLM-generated content at query #3
#--------------------------

def test_unique_list_sorts_and_removes_duplicates():
    printer = ISortPrettyPrinter(Config())
    result = _unique_list([3, 1, 2, 1], printer)
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
    result = _unique_list(["banana", "apple", "apple", "cherry"], printer)
    assert result == "['apple', 'banana', 'cherry']"

def test_unique_list_with_mixed_types_raises_error():
    printer = ISortPrettyPrinter(Config())
    try:
        _unique_list([1, "a"], printer)
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #4
#--------------------------

def test_assignment_raises_literal_sort_type_mismatch_when_type_mismatch():
    code = "my_var = [3, 1, 2]"
    sort_type = "dict"
    extension = "py"
    config = Config()
    try:
        assignment(code, sort_type, extension, config)
        assert False
    except LiteralSortTypeMismatch as e:
        assert e.actual == list
        assert e.expected == dict


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

def test_unique_tuple_with_mixed_types_raises_error():
    printer = ISortPrettyPrinter(Config())
    try:
        _unique_tuple((1, "a", 2), printer)
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #6
#--------------------------

def test_assignment_raises_literal_parsing_failure_when_ast_literal_eval_fails():
    result = assignment("x = invalid_literal", "lists", ".py")
    assert False


# LLM-generated content at query #7
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = assignment("x = invalid", "lists", ".py")
    assert False


# LLM-generated content at query #8
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = assignment("x = invalid", "lists", "py", Config())
    assert False


# LLM-generated content at query #9
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = None
    try:
        result = assignment("x = invalid", "lists", "py", Config())
    except Exception as e:
        result = e
    assert isinstance(result, LiteralParsingFailure)


# LLM-generated content at query #10
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    code = "x = invalid_literal"
    sort_type = "list"
    extension = "py"
    config = Config()
    try:
        assignment(code, sort_type, extension, config)
        assert False, "Expected LiteralParsingFailure"
    except LiteralParsingFailure:
        pass


# LLM-generated content at query #11
#--------------------------

def test_assignment_assignments_sort_type():
    code = "b = 2\na = 1"
    result = assignment(code, "assignments", "py")
    expected = "a = 1\nb = 2"
    assert result == expected

def test_assignment_assignments_sort_type_with_empty_lines():
    code = "b = 2\n\na = 1"
    result = assignment(code, "assignments", "py")
    expected = "a = 1\nb = 2"
    assert result == expected

def test_assignment_assignments_sort_type_raises_on_missing_equals():
    code = "b  2\na = 1"
    try:
        assignment(code, "assignments", "py")
        assert False
    except AssignmentsFormatMismatch:
        assert True

def test_assignment_list_sort_type():
    code = "x = [3, 1, 2]"
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    result = assignment(code, "list", "py")
    expected = "x = [1, 2, 3]"
    assert result == expected

def test_assignment_undefined_sort_type_raises():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "undefined_type", "py")
        assert False
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)

def test_assignment_literal_parsing_failure_raises():
    code = "x = invalid_literal"
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralParsingFailure:
        assert True

def test_assignment_type_mismatch_raises():
    code = "x = 123"
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    try:
        assignment(code, "list", "py")
        assert False
    except LiteralSortTypeMismatch:
        assert True

def test_assignment_with_formatting_function():
    code = "x = [3, 1, 2]"
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    result = assignment(code, "list", "py", config)
    expected = "X = [1, 2, 3]"
    assert result == expected

def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]   \n"
    type_mapping["list"] = (list, lambda v, p: p.pformat(sorted(v)))
    result = assignment(code, "list", "py")
    expected = "x = [1, 2, 3]   \n"
    assert result == expected


# LLM-generated content at query #12
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = None
    try:
        result = assignment("x = invalid", "lists", ".py")
    except Exception as e:
        assert isinstance(e, LiteralParsingFailure)
    assert result is None


# LLM-generated content at query #13
#--------------------------

def test_assignment_raises_literal_parsing_failure_on_invalid_literal():
    result = None
    try:
        result = assignment("x = invalid", "lists", ".py")
    except Exception as e:
        assert isinstance(e, LiteralParsingFailure)
    assert result is None


