# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.wrap_modes as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'invalid_name'
    module_0.from_string(var_0)

def test_case_1():
    pass

def test_case_2():
    var_0 = 'invalid_name'
    var_1 = module_0.formatter_from_string(var_0)

def test_case_3():
    var_0 = '1'
    var_1 = module_0.from_string(var_0)
    assert var_1 == module_0.WrapModes.VERTICAL
    assert module_0.WrapModes.GRID == module_0.WrapModes.GRID
    assert module_0.WrapModes.VERTICAL == module_0.WrapModes.VERTICAL
    assert module_0.WrapModes.HANGING_INDENT == module_0.WrapModes.HANGING_INDENT
    assert module_0.WrapModes.VERTICAL_HANGING_INDENT == module_0.WrapModes.VERTICAL_HANGING_INDENT
    assert module_0.WrapModes.VERTICAL_GRID == module_0.WrapModes.VERTICAL_GRID
    assert module_0.WrapModes.VERTICAL_GRID_GROUPED == module_0.WrapModes.VERTICAL_GRID_GROUPED
    assert module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA == module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA
    assert module_0.WrapModes.NOQA == module_0.WrapModes.NOQA
    assert module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET == module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET
    assert module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT == module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT
    assert module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    assert module_0.WrapModes.BACKSLASH_GRID == module_0.WrapModes.BACKSLASH_GRID
    var_2 = '#'
    var_3 = [var_2]
    var_4 = None
    var_5 = 1873
    var_6 = True
    var_7 = module_0._wrap_mode_interface(var_2, var_3, var_0, var_4, var_5, var_4, var_4, var_4, var_4, var_6)
    assert var_7 == ''

def test_case_4():
    with pytest.raises(NotImplementedError):
        module_0.vertical_grid_grouped_no_comma()

def test_case_5():
    var_0 = 'hello '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'hello \\'

def test_case_6():
    var_0 = ''
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == ' \\'