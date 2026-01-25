# Check out: https://github.com/GlowCheese/deepmosa
import isort.wrap_modes as module_0
import pytest


def test_case_0():
    pass

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.formatter_from_string(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.from_string(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'VERTICAL'
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
    var_2 = 'HANGING_INDENT'
    var_3 = module_0.from_string(var_2)
    assert var_3 == module_0.WrapModes.HANGING_INDENT
    var_4 = 'VERTICAL_HANGING_INDENT'
    var_5 = module_0.from_string(var_4)
    assert var_5 == module_0.WrapModes.VERTICAL_HANGING_INDENT
    var_6 = 'VERTICAL_GRID'
    var_7 = module_0.from_string(var_6)
    assert var_7 == module_0.WrapModes.VERTICAL_GRID
    var_8 = 'VERTICAL_GRID_GROUPED'
    var_9 = module_0.from_string(var_8)
    assert var_9 == module_0.WrapModes.VERTICAL_GRID_GROUPED
    var_10 = 'NOQA'
    var_11 = module_0.from_string(var_10)
    assert var_11 == module_0.WrapModes.NOQA
    var_12 = 'VERTICAL_HANGING_INDENT_BRACKET'
    var_13 = module_0.from_string(var_12)
    assert var_13 == module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET
    var_14 = 'VERTICAL_PREFIX_FROM_MODULE_IMPORT'
    var_15 = module_0.from_string(var_14)
    assert var_15 == module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT
    var_16 = 'HANGING_INDENT_WITH_PARENTHESES'
    var_17 = module_0.from_string(var_16)
    assert var_17 == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    var_18 = 'BACKSLASH_GRID'
    var_19 = module_0.from_string(var_18)
    assert var_19 == module_0.WrapModes.BACKSLASH_GRID
    var_20 = '0'
    var_21 = module_0.from_string(var_20)
    assert var_21 == module_0.WrapModes.GRID
    var_22 = '1'
    var_23 = module_0.from_string(var_22)
    assert var_23 == module_0.WrapModes.VERTICAL
    var_24 = '?'
    module_0.from_string(var_24)