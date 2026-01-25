####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_list_registers_and_formats_sorted_list():
    from unittest.mock import Mock
    
    test_list = [3, 1, 2]
    mock_printer = Mock(spec=ISortPrettyPrinter)
    mock_printer.pformat.return_value = "[1, 2, 3]"
    
    result = _list(test_list, mock_printer)
    
    assert result == "[1, 2, 3]"
    mock_printer.pformat.assert_called_once()
    call_args = mock_printer.pformat.call_args[0][0]
    assert call_args == [1, 2, 3]


def test_list_with_empty_list():
    from unittest.mock import Mock
    
    test_list = []
    mock_printer = Mock(spec=ISortPrettyPrinter)
    mock_printer.pformat.return_value = "[]"
    
    result = _list(test_list, mock_printer)
    
    assert result == "[]"
    mock_printer.pformat.assert_called_once_with([])


def test_list_with_strings():
    from unittest.mock import Mock
    
    test_list = ["zebra", "apple", "banana"]
    mock_printer = Mock(spec=ISortPrettyPrinter)
    mock_printer.pformat.return_value = "['apple', 'banana', 'zebra']"
    
    result = _list(test_list, mock_printer)
    
    assert result == "['apple', 'banana', 'zebra']"
    call_args = mock_printer.pformat.call_args[0][0]
    assert call_args == ["apple", "banana", "zebra"]


def test_list_with_mixed_comparable_types():
    from unittest.mock import Mock
    
    test_list = [3.5, 1, 2.1]
    mock_printer = Mock(spec=ISortPrettyPrinter)
    mock_printer.pformat.return_value = "[1, 2.1, 3.5]"
    
    result = _list(test_list, mock_printer)
    
    assert result == "[1, 2.1, 3.5]"
    call_args = mock_printer.pformat.call_args[0][0]
    assert call_args == [1, 2.1, 3.5]


# LLM-generated content at query #2
#--------------------------

```python
def test_dict_sorts_by_value():
    from pprint import PrettyPrinter
    
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    test_dict = {'a': 3, 'b': 1, 'c': 2}
    result = _dict(test_dict, printer)
    
    assert "'b': 1" in result
    assert "'c': 2" in result
    assert "'a': 3" in result
    assert result.index("'b': 1") < result.index("'c': 2") < result.index("'a': 3")


def test_dict_empty():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    test_dict = {}
    result = _dict(test_dict, printer)
    
    assert result == '{}'


def test_dict_single_item():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    test_dict = {'key': 5}
    result = _dict(test_dict, printer)
    
    assert 'key' in result
    assert '5' in result


def test_dict_numeric_values():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    test_dict = {'x': 100, 'y': 50, 'z': 75}
    result = _dict(test_dict, printer)
    
    assert result.index("'y': 50") < result.index("'z': 75") < result.index("'x': 100")


def test_dict_string_values():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    test_dict = {'a': 'zebra', 'b': 'apple', 'c': 'mango'}
    result = _dict(test_dict, printer)
    
    assert "'b': 'apple'" in result
    assert "'c': 'mango'" in result
    assert "'a': 'zebra'" in result
    assert result.index("'b': 'apple'") < result.index("'c': 'mango'") < result.index("'a': 'zebra'")


# LLM-generated content at query #3
#--------------------------

```python
def test_dict_sorts_by_values():
    from pprint import PrettyPrinter
    
    config_mock = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config_mock)
    
    test_dict = {'a': 3, 'b': 1, 'c': 2}
    result = _dict(test_dict, printer)
    
    assert "'b': 1" in result
    assert "'c': 2" in result
    assert "'a': 3" in result
    assert result.index("'b': 1") < result.index("'c': 2") < result.index("'a': 3")


def test_dict_empty():
    from pprint import PrettyPrinter
    
    config_mock = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config_mock)
    
    test_dict = {}
    result = _dict(test_dict, printer)
    
    assert result == "{}"


def test_dict_single_item():
    from pprint import PrettyPrinter
    
    config_mock = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config_mock)
    
    test_dict = {'key': 'value'}
    result = _dict(test_dict, printer)
    
    assert "'key': 'value'" in result


def test_dict_with_numeric_values():
    from pprint import PrettyPrinter
    
    config_mock = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config_mock)
    
    test_dict = {'z': 100, 'a': 50, 'b': 75}
    result = _dict(test_dict, printer)
    
    assert result.index("'a': 50") < result.index("'b': 75") < result.index("'z': 100")


def test_dict_with_string_values():
    from pprint import PrettyPrinter
    
    config_mock = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config_mock)
    
    test_dict = {'x': 'zebra', 'y': 'apple', 'z': 'monkey'}
    result = _dict(test_dict, printer)
    
    assert result.index("'y': 'apple'") < result.index("'z': 'monkey'") < result.index("'x': 'zebra'")


# LLM-generated content at query #4
#--------------------------

```python
def test_set_empty():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _set(set(), printer)
    assert result == "{}"


def test_set_single_element():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _set({1}, printer)
    assert result == "{1}"


def test_set_multiple_elements():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _set({3, 1, 2}, printer)
    assert result == "{1, 2, 3}"


def test_set_strings():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _set({"c", "a", "b"}, printer)
    assert result == "{'a', 'b', 'c'}"


def test_set_mixed_types():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _set({3, 1, 2}, printer)
    assert "{1, 2, 3}" == result


def test_set_sorted_order():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _set({5, 2, 8, 1}, printer)
    assert result == "{1, 2, 5, 8}"


# LLM-generated content at query #5
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "a = 1\nb = 2\n"
    result = assignment(code, "assignments", ".py")
    assert "a = 1" in result
    assert "b = 2" in result


def test_assignment_with_list_sort_type():
    from isort.stdlibs.all import all as all_stdlibs
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", ".py")
    assert "my_list = [1, 2, 3]" in result


def test_assignment_with_tuple_sort_type():
    code = "my_tuple = (3, 1, 2)"
    result = assignment(code, "tuple", ".py")
    assert "my_tuple = (1, 2, 3)" in result


def test_assignment_with_set_sort_type():
    code = "my_set = {3, 1, 2}"
    result = assignment(code, "set", ".py")
    assert "my_set = {1, 2, 3}" in result


def test_assignment_with_dict_sort_type():
    code = "my_dict = {'c': 1, 'a': 2, 'b': 3}"
    result = assignment(code, "dict", ".py")
    assert "my_dict" in result
    assert "'a'" in result


def test_assignment_invalid_sort_type():
    code = "my_var = [1, 2, 3]"
    try:
        assignment(code, "invalid_type", ".py")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)


def test_assignment_type_mismatch():
    code = "my_var = [1, 2, 3]"
    try:
        assignment(code, "dict", ".py")
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception as e:
        assert "LiteralSortTypeMismatch" in type(e).__name__


def test_assignment_invalid_literal():
    code = "my_var = invalid_literal"
    try:
        assignment(code, "list", ".py")
        assert False, "Should raise LiteralParsingFailure"
    except Exception as e:
        assert "LiteralParsingFailure" in type(e).__name__


def test_assignment_preserves_trailing_whitespace():
    code = "my_list = [3, 1, 2]  \n"
    result = assignment(code, "list", ".py")
    assert result.endswith("  \n")


def test_assignment_with_custom_config():
    from isort.settings import Config
    config = Config(line_length=80)
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", ".py", config)
    assert "my_list = [1, 2, 3]" in result


# LLM-generated content at query #6
#--------------------------

```python
def test_unique_list_removes_duplicates_and_sorts():
    from isort.stdlibs.py310 import all as isort_all
    from pprint import PrettyPrinter
    
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list([3, 1, 2, 1, 3], printer)
    assert "[1, 2, 3]" == result


def test_unique_list_empty_list():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list([], printer)
    assert "[]" == result


def test_unique_list_single_element():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list([42], printer)
    assert "[42]" == result


def test_unique_list_strings():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list(['c', 'a', 'b', 'a'], printer)
    assert "['a', 'b', 'c']" == result


def test_unique_list_already_sorted_and_unique():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list([1, 2, 3], printer)
    assert "[1, 2, 3]" == result


# LLM-generated content at query #7
#--------------------------

```python
def test_assignment_with_formatting_function():
    from isort.settings import Config
    
    def mock_formatter(code, extension, config):
        return code.upper()
    
    custom_config = Config(line_length=80, formatting_function=mock_formatter)
    
    assert custom_config.formatting_function is not None
    assert custom_config.formatting_function("test", "py", custom_config) == "TEST"


# LLM-generated content at query #8
#--------------------------

```python
def test_assignment_with_formatting_function():
    from isort.settings import Config
    
    def mock_formatting_function(code, extension, config):
        return code.upper()
    
    config = Config(line_length=80, formatting_function=mock_formatting_function)
    
    assert config.formatting_function is not None
    assert config.formatting_function("test", "py", config) == "TEST"


# LLM-generated content at query #9
#--------------------------

```python
def test_unique_list():
    from pprint import PrettyPrinter
    from isort.settings import Config
    
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list([3, 1, 2, 1, 3], printer)
    assert result == "[1, 2, 3]"


def test_unique_list_empty():
    from pprint import PrettyPrinter
    from isort.settings import Config
    
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list([], printer)
    assert result == "[]"


def test_unique_list_strings():
    from pprint import PrettyPrinter
    from isort.settings import Config
    
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list(["c", "a", "b", "a"], printer)
    assert result == "['a', 'b', 'c']"


def test_unique_list_single_element():
    from pprint import PrettyPrinter
    from isort.settings import Config
    
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list([42], printer)
    assert result == "[42]"


def test_unique_list_all_unique():
    from pprint import PrettyPrinter
    from isort.settings import Config
    
    config = Config(line_length=80)
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list([5, 3, 1], printer)
    assert result == "[1, 3, 5]"


# LLM-generated content at query #10
#--------------------------

```python
def test_assignment_with_valid_literal_does_not_raise_exception():
    from pprint import PrettyPrinter
    import ast
    
    config = type('Config', (), {'line_length': 80})()
    code = "my_var = [1, 2, 3]"
    sort_type = "assignments"
    extension = ".py"
    
    variable_name, literal = code.split("=")
    variable_name = variable_name.strip()
    literal = literal.lstrip()
    
    exception_raised = False
    try:
        value = ast.literal_eval(literal)
    except Exception as error:
        exception_raised = True
    
    assert exception_raised is False


# LLM-generated content at query #11
#--------------------------

```python
def test_formatting_function_is_called_when_config_has_it():
    from isort.settings import Config
    
    formatting_function_called = []
    
    def mock_formatting_function(code, extension, config):
        formatting_function_called.append(True)
        return code
    
    config = Config(formatting_function=mock_formatting_function)
    
    assert config.formatting_function is not None
    assert config.formatting_function == mock_formatting_function


# LLM-generated content at query #12
#--------------------------

```python
def test_assignment_literal_eval_succeeds():
    from isort.settings import Config
    
    config = Config()
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = "py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert isinstance(result, str)
    assert "my_list" in result


# LLM-generated content at query #13
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "x = 1\ny = 2\nz = 3"
    result = assignment(code, "assignments", ".py")
    assert "x = 1" in result
    assert "y = 2" in result
    assert "z = 3" in result


def test_assignment_with_list_sort_type():
    from isort.settings import Config
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", ".py", Config())
    assert "my_list = [1, 2, 3]" in result


def test_assignment_with_tuple_sort_type():
    from isort.settings import Config
    code = "my_tuple = (3, 1, 2)"
    result = assignment(code, "tuple", ".py", Config())
    assert "my_tuple = (1, 2, 3)" in result


def test_assignment_with_set_sort_type():
    from isort.settings import Config
    code = "my_set = {3, 1, 2}"
    result = assignment(code, "set", ".py", Config())
    assert "{1, 2, 3}" in result


def test_assignment_with_invalid_sort_type():
    from isort.settings import Config
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "invalid_type", ".py", Config())
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)


def test_assignment_with_invalid_literal():
    from isort.settings import Config
    code = "x = [1, 2,"
    try:
        assignment(code, "list", ".py", Config())
        assert False, "Should raise LiteralParsingFailure"
    except Exception:
        pass


def test_assignment_with_type_mismatch():
    from isort.settings import Config
    code = "x = (1, 2, 3)"
    try:
        assignment(code, "list", ".py", Config())
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception:
        pass


def test_assignment_preserves_trailing_whitespace():
    from isort.settings import Config
    code = "my_list = [3, 1, 2]  \n"
    result = assignment(code, "list", ".py", Config())
    assert result.endswith("  \n")


def test_assignment_with_spaces_around_equals():
    from isort.settings import Config
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", ".py", Config())
    assert "my_list = " in result


def test_assignment_dict_sort_type():
    from isort.settings import Config
    code = "my_dict = {'z': 1, 'a': 2, 'm': 3}"
    result = assignment(code, "dict", ".py", Config())
    assert "my_dict = " in result


# LLM-generated content at query #14
#--------------------------

```python
def test_assignment_literal_eval_succeeds():
    from isort.stdlibs.all import all as all_stdlibs
    
    config = Config(line_length=80)
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = "py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert isinstance(result, str)
    assert "my_list" in result


# LLM-generated content at query #15
#--------------------------

```python
def test_unique_tuple_removes_duplicates_and_sorts():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _unique_tuple((3, 1, 2, 1, 3), printer)
    assert result == "(1, 2, 3)"


def test_unique_tuple_with_empty_tuple():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _unique_tuple((), printer)
    assert result == "()"


def test_unique_tuple_with_single_element():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _unique_tuple((5,), printer)
    assert result == "(5,)"


def test_unique_tuple_with_strings():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _unique_tuple(("c", "a", "b", "a"), printer)
    assert result == "('a', 'b', 'c')"


def test_unique_tuple_already_sorted_and_unique():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _unique_tuple((1, 2, 3), printer)
    assert result == "(1, 2, 3)"


def test_unique_tuple_all_duplicates():
    config = Config()
    printer = ISortPrettyPrinter(config)
    result = _unique_tuple((5, 5, 5, 5), printer)
    assert result == "(5,)"


# LLM-generated content at query #16
#--------------------------

```python
def test_assignment_with_formatting_function():
    from pathlib import Path
    from isort.config import Config
    from isort.stdlibs.all import all as all_stdlibs
    
    def mock_formatting_function(code: str, extension: str, config: Config) -> str:
        return code + "\n"
    
    config = Config(
        line_length=88,
        formatting_function=mock_formatting_function
    )
    
    assert config.formatting_function is not None
    assert config.formatting_function == mock_formatting_function


# LLM-generated content at query #17
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "a = 1\nb = 2\n"
    result = assignment(code, "assignments", ".py")
    assert "a = " in result
    assert "b = " in result


def test_assignment_with_invalid_sort_type():
    code = "x = [3, 1, 2]"
    try:
        assignment(code, "invalid_sort_type", ".py")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)


def test_assignment_with_list_sort_type():
    from isort import Config
    code = "x = [3, 1, 2]"
    config = Config()
    # Assuming "list" is a valid sort_type in type_mapping
    try:
        result = assignment(code, "list", ".py", config)
        assert "x = " in result
    except (KeyError, ValueError):
        pass


def test_assignment_with_dict_sort_type():
    from isort import Config
    code = "d = {'b': 2, 'a': 1}"
    config = Config()
    try:
        result = assignment(code, "dict", ".py", config)
        assert "d = " in result
    except (KeyError, ValueError):
        pass


def test_assignment_with_invalid_literal():
    from isort import Config
    code = "x = [1, 2, invalid]"
    config = Config()
    try:
        assignment(code, "list", ".py", config)
        assert False, "Should raise LiteralParsingFailure"
    except Exception:
        pass


def test_assignment_with_type_mismatch():
    from isort import Config
    code = "x = 'string'"
    config = Config()
    try:
        assignment(code, "list", ".py", config)
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception:
        pass


def test_assignment_preserves_trailing_whitespace():
    from isort import Config
    code = "x = [3, 1, 2]  \n"
    config = Config()
    try:
        result = assignment(code, "list", ".py", config)
        assert result.endswith("  \n") or result.endswith("  ")
    except (KeyError, ValueError):
        pass


def test_assignment_with_formatting_function():
    from isort import Config
    def mock_formatter(code, ext, cfg):
        return code.upper()
    
    config = Config(formatting_function=mock_formatter)
    code = "x = [3, 1, 2]"
    try:
        result = assignment(code, "list", ".py", config)
        assert len(result) > 0
    except (KeyError, ValueError):
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "a = 1\nb = 2\n"
    result = assignment(code, "assignments", ".py")
    assert "a = 1" in result
    assert "b = 2" in result


def test_assignment_with_invalid_sort_type():
    code = "x = [3, 1, 2]"
    try:
        assignment(code, "invalid_type", ".py")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Defined sort types are" in str(e)


def test_assignment_with_list_sort_type():
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", ".py")
    assert "x = " in result
    assert "[1, 2, 3]" in result


def test_assignment_with_tuple_sort_type():
    code = "y = (3, 1, 2)"
    result = assignment(code, "tuple", ".py")
    assert "y = " in result


def test_assignment_with_set_sort_type():
    code = "z = {3, 1, 2}"
    result = assignment(code, "set", ".py")
    assert "z = " in result


def test_assignment_with_dict_sort_type():
    code = "d = {'c': 3, 'a': 1, 'b': 2}"
    result = assignment(code, "dict", ".py")
    assert "d = " in result


def test_assignment_with_invalid_literal():
    code = "x = invalid_syntax"
    try:
        assignment(code, "list", ".py")
        assert False, "Should raise LiteralParsingFailure"
    except Exception:
        pass


def test_assignment_with_type_mismatch():
    code = "x = (1, 2, 3)"
    try:
        assignment(code, "list", ".py")
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception:
        pass


def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]  \n"
    result = assignment(code, "list", ".py")
    assert result.endswith("  \n")


def test_assignment_with_custom_config():
    from isort import Config
    config = Config(line_length=80)
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", ".py", config)
    assert "x = " in result


# LLM-generated content at query #19
#--------------------------

```python
def test_assignment_with_valid_literal_no_exception():
    from isort.stdlibs.all import all as stdlib_all
    from isort.settings import Config
    
    config = Config()
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = "py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert isinstance(result, str)
    assert "my_list" in result


# LLM-generated content at query #20
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "a = 1\nb = 2\n"
    result = assignment(code, "assignments", ".py")
    assert "a = " in result
    assert "b = " in result


def test_assignment_with_undefined_sort_type():
    code = "x = [3, 1, 2]"
    try:
        assignment(code, "undefined_sort", ".py")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)


def test_assignment_with_list_sort_type():
    code = "x = [3, 1, 2]"
    from isort.stdlibs.all import all as stdlib_all
    result = assignment(code, "list", ".py")
    assert "x = " in result
    assert "[" in result and "]" in result


def test_assignment_with_tuple_sort_type():
    code = "x = (3, 1, 2)"
    result = assignment(code, "tuple", ".py")
    assert "x = " in result
    assert "(" in result and ")" in result


def test_assignment_with_set_sort_type():
    code = "x = {3, 1, 2}"
    result = assignment(code, "set", ".py")
    assert "x = " in result
    assert "{" in result and "}" in result


def test_assignment_with_dict_sort_type():
    code = "x = {'c': 3, 'a': 1, 'b': 2}"
    result = assignment(code, "dict", ".py")
    assert "x = " in result
    assert "{" in result and "}" in result


def test_assignment_with_wrong_literal_type():
    code = "x = [3, 1, 2]"
    try:
        assignment(code, "dict", ".py")
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception as e:
        assert "LiteralSortTypeMismatch" in type(e).__name__


def test_assignment_with_invalid_literal():
    code = "x = [3, 1, invalid_var]"
    try:
        assignment(code, "list", ".py")
        assert False, "Should raise LiteralParsingFailure"
    except Exception as e:
        assert "LiteralParsingFailure" in type(e).__name__


def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]  \n"
    result = assignment(code, "list", ".py")
    assert result.endswith("  \n")


def test_assignment_with_empty_list():
    code = "x = []"
    result = assignment(code, "list", ".py")
    assert "x = " in result
    assert "[]" in result


def test_assignment_with_empty_dict():
    code = "x = {}"
    result = assignment(code, "dict", ".py")
    assert "x = " in result
    assert "{}" in result


# LLM-generated content at query #21
#--------------------------

```python
def test_assignment_exception_not_raised_for_valid_literal():
    from isort.stdlibs.all import all as all_stdlibs
    
    config = Config()
    code = "x = [3, 1, 2]"
    sort_type = "list"
    extension = ".py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert result is not None
    assert "x =" in result


# LLM-generated content at query #22
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "x = 1\ny = 2\n"
    result = assignment(code, "assignments", "py")
    assert "x = 1\n" in result
    assert "y = 2\n" in result


def test_assignment_with_list_sort_type():
    from isort.settings import Config
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", "py", Config())
    assert "my_list = [1, 2, 3]" in result


def test_assignment_with_tuple_sort_type():
    from isort.settings import Config
    code = "my_tuple = (3, 1, 2)"
    result = assignment(code, "tuple", "py", Config())
    assert "my_tuple = (1, 2, 3)" in result


def test_assignment_with_set_sort_type():
    from isort.settings import Config
    code = "my_set = {3, 1, 2}"
    result = assignment(code, "set", "py", Config())
    assert "my_set = " in result


def test_assignment_with_invalid_sort_type():
    from isort.settings import Config
    code = "x = 1"
    try:
        assignment(code, "invalid_type", "py", Config())
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Undefined sort_type" in str(e) or "undefined sort_type" in str(e)


def test_assignment_with_invalid_literal():
    from isort.settings import Config
    code = "x = invalid_syntax_here"
    try:
        assignment(code, "list", "py", Config())
        assert False, "Should raise LiteralParsingFailure"
    except Exception:
        pass


def test_assignment_with_type_mismatch():
    from isort.settings import Config
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "set", "py", Config())
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception:
        pass


def test_assignment_preserves_trailing_whitespace():
    from isort.settings import Config
    code = "my_list = [3, 1, 2]  \n"
    result = assignment(code, "list", "py", Config())
    assert result.endswith("  \n")


def test_assignment_with_dict_sort_type():
    from isort.settings import Config
    code = "my_dict = {'z': 1, 'a': 2}"
    result = assignment(code, "dict", "py", Config())
    assert "my_dict = " in result


# LLM-generated content at query #23
#--------------------------

```python
def test_assignment_literal_eval_succeeds():
    from isort.stdlibs.all import all as all_stdlibs
    
    config = Config()
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = ".py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert result is not None
    assert "my_list" in result


# LLM-generated content at query #24
#--------------------------

```python
def test_assignment_with_formatting_function():
    from isort.settings import Config
    
    def mock_formatting_function(code, extension, config):
        return code.upper()
    
    config = Config(line_length=80, formatting_function=mock_formatting_function)
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", "py", config)
    
    assert config.formatting_function is not None
    assert result == result.upper() or result.rstrip() == result.upper().rstrip()


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "x = 1\ny = 2\n"
    result = assignment(code, "assignments", ".py")
    assert "x = 1" in result
    assert "y = 2" in result


def test_assignment_with_list_sort_type():
    from isort.stdlibs.all import all as stdlib_all
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", ".py")
    assert "my_list = " in result
    assert "[" in result and "]" in result


def test_assignment_with_tuple_sort_type():
    code = "my_tuple = (3, 1, 2)"
    result = assignment(code, "tuple", ".py")
    assert "my_tuple = " in result
    assert "(" in result and ")" in result


def test_assignment_with_set_sort_type():
    code = "my_set = {3, 1, 2}"
    result = assignment(code, "set", ".py")
    assert "my_set = " in result
    assert "{" in result and "}" in result


def test_assignment_with_dict_sort_type():
    code = "my_dict = {'c': 3, 'a': 1, 'b': 2}"
    result = assignment(code, "dict", ".py")
    assert "my_dict = " in result
    assert "{" in result and "}" in result


def test_assignment_with_undefined_sort_type():
    code = "x = 1"
    try:
        assignment(code, "undefined_type", ".py")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)


def test_assignment_with_invalid_literal():
    code = "x = invalid_syntax!!!"
    try:
        assignment(code, "list", ".py")
        assert False, "Should raise LiteralParsingFailure"
    except Exception as e:
        assert "LiteralParsingFailure" in type(e).__name__


def test_assignment_with_type_mismatch():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "dict", ".py")
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception as e:
        assert "LiteralSortTypeMismatch" in type(e).__name__


def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]  \n"
    result = assignment(code, "list", ".py")
    assert result.endswith("  \n") or result.endswith("  ")


def test_assignment_with_custom_config():
    from isort import Config
    custom_config = Config(line_length=120)
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", ".py", custom_config)
    assert "x = " in result


# LLM-generated content at query #2
#--------------------------

```python
def test_unique_tuple_removes_duplicates_and_sorts():
    from isort.stdlibs.all import all as all_stdlibs
    
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    value = (3, 1, 2, 1, 3)
    result = _unique_tuple(value, printer)
    
    assert result == "(1, 2, 3)"


def test_unique_tuple_with_empty_tuple():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    value = ()
    result = _unique_tuple(value, printer)
    
    assert result == "()"


def test_unique_tuple_with_single_element():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    value = (5,)
    result = _unique_tuple(value, printer)
    
    assert result == "(5,)"


def test_unique_tuple_with_strings():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    value = ("zebra", "apple", "zebra", "banana")
    result = _unique_tuple(value, printer)
    
    assert result == "('apple', 'banana', 'zebra')"


def test_unique_tuple_already_sorted_and_unique():
    config = type('Config', (), {'line_length': 80})()
    printer = ISortPrettyPrinter(config)
    
    value = (1, 2, 3, 4)
    result = _unique_tuple(value, printer)
    
    assert result == "(1, 2, 3, 4)"


# LLM-generated content at query #3
#--------------------------

```python
def test_assignment_with_formatting_function():
    from isort.settings import Config
    
    def mock_formatter(code, extension, config):
        return code.upper()
    
    config = Config(line_length=80, formatting_function=mock_formatter)
    
    assert config.formatting_function is not None
    assert config.formatting_function("test", "py", config) == "TEST"


# LLM-generated content at query #4
#--------------------------

```python
def test_unique_tuple_removes_duplicates_and_sorts():
    from pprint import PrettyPrinter
    
    class Config:
        line_length = 80
    
    class ISortPrettyPrinter(PrettyPrinter):
        def __init__(self, config):
            super().__init__(width=config.line_length, compact=True)
    
    config = Config()
    printer = ISortPrettyPrinter(config)
    
    value = (3, 1, 2, 1, 3)
    result = _unique_tuple(value, printer)
    expected = printer.pformat((1, 2, 3))
    
    assert result == expected


def test_unique_tuple_with_empty_tuple():
    from pprint import PrettyPrinter
    
    class Config:
        line_length = 80
    
    class ISortPrettyPrinter(PrettyPrinter):
        def __init__(self, config):
            super().__init__(width=config.line_length, compact=True)
    
    config = Config()
    printer = ISortPrettyPrinter(config)
    
    value = ()
    result = _unique_tuple(value, printer)
    expected = printer.pformat(())
    
    assert result == expected


def test_unique_tuple_with_single_element():
    from pprint import PrettyPrinter
    
    class Config:
        line_length = 80
    
    class ISortPrettyPrinter(PrettyPrinter):
        def __init__(self, config):
            super().__init__(width=config.line_length, compact=True)
    
    config = Config()
    printer = ISortPrettyPrinter(config)
    
    value = (5,)
    result = _unique_tuple(value, printer)
    expected = printer.pformat((5,))
    
    assert result == expected


def test_unique_tuple_with_strings():
    from pprint import PrettyPrinter
    
    class Config:
        line_length = 80
    
    class ISortPrettyPrinter(PrettyPrinter):
        def __init__(self, config):
            super().__init__(width=config.line_length, compact=True)
    
    config = Config()
    printer = ISortPrettyPrinter(config)
    
    value = ("c", "a", "b", "a")
    result = _unique_tuple(value, printer)
    expected = printer.pformat(("a", "b", "c"))
    
    assert result == expected


def test_unique_tuple_with_all_duplicates():
    from pprint import PrettyPrinter
    
    class Config:
        line_length = 80
    
    class ISortPrettyPrinter(PrettyPrinter):
        def __init__(self, config):
            super().__init__(width=config.line_length, compact=True)
    
    config = Config()
    printer = ISortPrettyPrinter(config)
    
    value = (2, 2, 2, 2)
    result = _unique_tuple(value, printer)
    expected = printer.pformat((2,))
    
    assert result == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_formatting_function_predicate_evaluates_to_true():
    from isort.settings import Config
    
    def mock_formatting_function(code, extension, config):
        return code + "\n"
    
    config = Config(formatting_function=mock_formatting_function)
    
    assert config.formatting_function is not None
    assert config.formatting_function == mock_formatting_function
    assert callable(config.formatting_function)


# LLM-generated content at query #6
#--------------------------

```python
def test_assignment_literal_eval_succeeds():
    from isort.settings import Config
    
    config = Config()
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = ".py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert isinstance(result, str)
    assert "my_list" in result


# LLM-generated content at query #7
#--------------------------

```python
def test_assignment_literal_eval_succeeds():
    from isort.stdlibs.all import all as all_stdlibs
    from isort.settings import Config
    
    config = Config()
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = ".py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert isinstance(result, str)
    assert "my_list" in result


# LLM-generated content at query #8
#--------------------------

```python
def test_unique_list():
    from isort.settings import Config
    
    config = Config()
    printer = ISortPrettyPrinter(config)
    
    result = _unique_list([3, 1, 2, 1, 3], printer)
    assert result == "[1, 2, 3]"
    
    result = _unique_list([], printer)
    assert result == "[]"
    
    result = _unique_list([1], printer)
    assert result == "[1]"
    
    result = _unique_list(['c', 'a', 'b', 'a'], printer)
    assert result == "['a', 'b', 'c']"
    
    result = _unique_list([1, 2, 3, 2, 1], printer)
    assert result == "[1, 2, 3]"


# LLM-generated content at query #9
#--------------------------

```python
def test_assignment_literal_parsing_succeeds():
    from isort.stdlibs.all import all as all_stdlibs
    from isort.settings import Config
    
    config = Config()
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = "py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert isinstance(result, str)
    assert "my_list" in result


# LLM-generated content at query #10
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "a = 1\nb = 2\n"
    result = assignment(code, "assignments", "py")
    assert "a = " in result
    assert "b = " in result


def test_assignment_with_list_sort_type():
    from isort.stdlibs.py310 import all as all_modules
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", "py")
    assert "my_list = " in result


def test_assignment_with_tuple_sort_type():
    code = "my_tuple = (3, 1, 2)"
    result = assignment(code, "tuple", "py")
    assert "my_tuple = " in result


def test_assignment_with_dict_sort_type():
    code = "my_dict = {'c': 3, 'a': 1, 'b': 2}"
    result = assignment(code, "dict", "py")
    assert "my_dict = " in result


def test_assignment_invalid_sort_type():
    code = "x = 1"
    try:
        assignment(code, "invalid_type", "py")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)


def test_assignment_literal_parsing_failure():
    code = "x = invalid_literal"
    try:
        assignment(code, "list", "py")
        assert False, "Should raise LiteralParsingFailure"
    except Exception:
        pass


def test_assignment_sort_type_mismatch():
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "dict", "py")
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception:
        pass


def test_assignment_preserves_trailing_whitespace():
    code = "x = [3, 1, 2]  \n"
    result = assignment(code, "list", "py")
    assert result.endswith("  \n")


def test_assignment_with_config():
    from isort.settings import Config
    code = "x = [3, 1, 2]"
    config = Config(line_length=80)
    result = assignment(code, "list", "py", config)
    assert "x = " in result


# LLM-generated content at query #11
#--------------------------

```python
def test_assignment_with_formatting_function():
    from isort.settings import Config
    
    def mock_formatting_function(code, extension, config):
        return code.upper()
    
    config = Config(line_length=80, formatting_function=mock_formatting_function)
    code = "my_list = [3, 1, 2]"
    sort_type = "literal_list"
    extension = ".py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert config.formatting_function is not None
    assert callable(config.formatting_function)


# LLM-generated content at query #12
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "x = 1\ny = 2\n"
    result = assignment(code, "assignments", ".py")
    assert "x = 1" in result
    assert "y = 2" in result


def test_assignment_with_list_sort_type():
    from isort.settings import Config
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", ".py", Config())
    assert "my_list = [1, 2, 3]" in result


def test_assignment_with_tuple_sort_type():
    from isort.settings import Config
    code = "my_tuple = (3, 1, 2)"
    result = assignment(code, "tuple", ".py", Config())
    assert "my_tuple = (1, 2, 3)" in result


def test_assignment_with_set_sort_type():
    from isort.settings import Config
    code = "my_set = {3, 1, 2}"
    result = assignment(code, "set", ".py", Config())
    assert "my_set = {1, 2, 3}" in result


def test_assignment_with_undefined_sort_type():
    from isort.settings import Config
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "undefined_type", ".py", Config())
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Trying to sort using an undefined sort_type" in str(e)


def test_assignment_with_mismatched_type():
    from isort.settings import Config
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "tuple", ".py", Config())
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception:
        pass


def test_assignment_with_invalid_literal():
    from isort.settings import Config
    code = "x = invalid_code"
    try:
        assignment(code, "list", ".py", Config())
        assert False, "Should raise LiteralParsingFailure"
    except Exception:
        pass


def test_assignment_preserves_trailing_whitespace():
    from isort.settings import Config
    code = "x = [3, 1, 2]  \n"
    result = assignment(code, "list", ".py", Config())
    assert result.endswith("  \n")


def test_assignment_with_formatting_function():
    from isort.settings import Config
    def mock_formatter(code, ext, cfg):
        return code.upper()
    
    config = Config(formatting_function=mock_formatter)
    code = "x = [3, 1, 2]"
    result = assignment(code, "list", ".py", config)
    assert result.isupper() or "X = " in result


# LLM-generated content at query #13
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "x = 1\ny = 2\nz = 3\n"
    result = assignment(code, "assignments", ".py")
    assert "x = 1" in result
    assert "y = 2" in result
    assert "z = 3" in result


def test_assignment_with_list_sort_type():
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", ".py")
    assert "my_list = " in result
    assert "[1, 2, 3]" in result


def test_assignment_with_tuple_sort_type():
    code = "my_tuple = (3, 1, 2)"
    result = assignment(code, "tuple", ".py")
    assert "my_tuple = " in result
    assert "(1, 2, 3)" in result


def test_assignment_with_set_sort_type():
    code = "my_set = {3, 1, 2}"
    result = assignment(code, "set", ".py")
    assert "my_set = " in result


def test_assignment_with_dict_sort_type():
    code = "my_dict = {'c': 3, 'a': 1, 'b': 2}"
    result = assignment(code, "dict", ".py")
    assert "my_dict = " in result


def test_assignment_preserves_trailing_whitespace():
    code = "my_list = [3, 1, 2]  \n"
    result = assignment(code, "list", ".py")
    assert result.endswith("  \n")


def test_assignment_with_invalid_sort_type():
    code = "my_list = [3, 1, 2]"
    try:
        assignment(code, "invalid_type", ".py")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)


def test_assignment_with_type_mismatch():
    code = "my_list = (3, 1, 2)"
    try:
        assignment(code, "list", ".py")
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception:
        pass


def test_assignment_with_invalid_literal():
    code = "my_var = not_a_valid_literal"
    try:
        assignment(code, "list", ".py")
        assert False, "Should raise LiteralParsingFailure"
    except Exception:
        pass


def test_assignment_with_custom_config():
    from isort.settings import Config
    code = "my_list = [3, 1, 2]"
    custom_config = Config(line_length=80)
    result = assignment(code, "list", ".py", custom_config)
    assert "my_list = " in result


def test_assignment_strips_variable_name():
    code = "  my_var  = [3, 1, 2]"
    result = assignment(code, "list", ".py")
    assert "my_var = " in result


# LLM-generated content at query #14
#--------------------------

```python
def test_assignment_literal_eval_succeeds():
    from isort.stdlibs.all import all as all_stdlibs
    
    config = type('Config', (), {'line_length': 88, 'formatting_function': None})()
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = ".py"
    
    type_mapping = {
        "list": (list, lambda x, p: str(sorted(x)))
    }
    
    variable_name, literal = code.split("=")
    variable_name = variable_name.strip()
    literal = literal.lstrip()
    
    import ast
    try:
        value = ast.literal_eval(literal)
        exception_raised = False
    except Exception as error:
        exception_raised = True
    
    assert exception_raised is False


# LLM-generated content at query #15
#--------------------------

```python
def test_assignment_literal_eval_succeeds():
    """Test that line 18 predicate evaluates to False when ast.literal_eval succeeds"""
    from isort.stdlibs.all import all as isort_all
    from isort.settings import Config
    
    config = Config()
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = ".py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert isinstance(result, str)
    assert "my_list" in result


# LLM-generated content at query #16
#--------------------------

```python
def test_assignment_applies_formatting_function_when_config_has_it():
    from isort.settings import Config
    
    def mock_formatting_function(code, extension, config):
        return code.upper()
    
    config = Config(formatting_function=mock_formatting_function)
    result = assignment("x = [3, 1, 2]", "list", "py", config)
    
    assert result == result.upper() or result.rstrip() == result.rstrip().upper()


# LLM-generated content at query #17
#--------------------------

```python
def test_assignment_with_assignments_sort_type():
    code = "x = 1\ny = 2\nz = 3\n"
    result = assignment(code, "assignments", ".py")
    assert "x = 1" in result
    assert "y = 2" in result
    assert "z = 3" in result


def test_assignment_with_list_sort_type():
    from isort.settings import Config
    code = "my_list = [3, 1, 2]"
    result = assignment(code, "list", ".py", Config())
    assert "my_list = [1, 2, 3]" in result


def test_assignment_with_tuple_sort_type():
    from isort.settings import Config
    code = "my_tuple = (3, 1, 2)"
    result = assignment(code, "tuple", ".py", Config())
    assert "my_tuple = (1, 2, 3)" in result


def test_assignment_with_set_sort_type():
    from isort.settings import Config
    code = "my_set = {3, 1, 2}"
    result = assignment(code, "set", ".py", Config())
    assert "my_set = " in result
    assert "1" in result
    assert "2" in result
    assert "3" in result


def test_assignment_with_invalid_sort_type():
    from isort.settings import Config
    code = "x = [1, 2, 3]"
    try:
        assignment(code, "invalid_type", ".py", Config())
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "undefined sort_type" in str(e)


def test_assignment_with_invalid_literal():
    from isort.settings import Config
    code = "x = [1, 2, invalid]"
    try:
        assignment(code, "list", ".py", Config())
        assert False, "Should raise LiteralParsingFailure"
    except Exception:
        pass


def test_assignment_with_type_mismatch():
    from isort.settings import Config
    code = "x = (1, 2, 3)"
    try:
        assignment(code, "list", ".py", Config())
        assert False, "Should raise LiteralSortTypeMismatch"
    except Exception:
        pass


def test_assignment_preserves_trailing_whitespace():
    from isort.settings import Config
    code = "x = [3, 1, 2]  \n"
    result = assignment(code, "list", ".py", Config())
    assert result.endswith("  \n")


def test_assignment_with_complex_list():
    from isort.settings import Config
    code = "items = [5, 2, 8, 1, 9]"
    result = assignment(code, "list", ".py", Config())
    assert "items = [1, 2, 5, 8, 9]" in result


# LLM-generated content at query #18
#--------------------------

```python
def test_assignment_literal_eval_succeeds():
    from isort.stdlibs.all import all as all_stdlibs
    from isort.settings import Config
    
    config = Config()
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = ".py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert isinstance(result, str)
    assert "my_list" in result


# LLM-generated content at query #19
#--------------------------

```python
def test_assignment_with_formatting_function():
    from isort.settings import Config
    
    def mock_formatting_function(code, extension, config):
        return code.upper()
    
    config = Config(line_length=80, formatting_function=mock_formatting_function)
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = ".py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert "MY_LIST" in result


# LLM-generated content at query #20
#--------------------------

```python
def test_assignment_predicate_line_18_evaluates_to_false():
    from pprint import PrettyPrinter
    
    config = Config()
    code = "my_list = [3, 1, 2]"
    sort_type = "list"
    extension = ".py"
    
    result = assignment(code, sort_type, extension, config)
    
    assert isinstance(result, str)
    assert "my_list" in result


