# Check out: https://github.com/GlowCheese/deepmosa
import isort.wrap_modes as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = ''
    module_0.from_string(var_0)

def test_case_1():
    with pytest.raises(NotImplementedError):
        module_0.vertical_grid_grouped_no_comma()

def test_case_2():
    var_0 = ''
    var_1 = module_0.formatter_from_string(var_0)

def test_case_3():
    var_0 = 'GRID'
    var_1 = '1'
    var_2 = module_0.from_string(var_1)
    assert var_2 == module_0.WrapModes.VERTICAL
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
    var_3 = '10'
    var_4 = module_0.from_string(var_3)
    assert var_4 == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    var_5 = module_0.from_string(var_0)
    assert var_5 == module_0.WrapModes.GRID