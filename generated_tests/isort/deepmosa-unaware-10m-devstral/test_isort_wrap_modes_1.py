# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.wrap_modes as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '~'
    module_0.from_string(var_0)

def test_case_1():
    pass

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = 'E3/4zBWys0*o1<\rp'
    var_2 = module_0.formatter_from_string(var_1)
    module_0.from_string(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'GRID'
    var_1 = module_0.from_string(var_0)
    assert var_1 == module_0.WrapModes.GRID
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
    var_2 = 'VERTICAL'
    var_3 = module_0.from_string(var_2)
    assert var_3 == module_0.WrapModes.VERTICAL
    var_4 = module_0.from_string(var_2)
    assert var_4 == module_0.WrapModes.VERTICAL
    var_5 = 'VERTICAL_HANGING_INDENT'
    var_6 = module_0.from_string(var_5)
    assert var_6 == module_0.WrapModes.VERTICAL_HANGING_INDENT
    var_7 = 'VERTICAL_GRID'
    var_8 = module_0.from_string(var_7)
    assert var_8 == module_0.WrapModes.VERTICAL_GRID
    var_9 = 'VERTICAL_GRID_GROUPED'
    var_10 = module_0.from_string(var_9)
    assert var_10 == module_0.WrapModes.VERTICAL_GRID_GROUPED
    var_11 = 'NOQA'
    var_12 = module_0.from_string(var_11)
    assert var_12 == module_0.WrapModes.NOQA
    var_13 = 'VERTICAL_HANGING_INDENT_BRACKET'
    var_14 = module_0.from_string(var_13)
    assert var_14 == module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET
    var_15 = 'VERTICAL_PREFIX_FROM_MODULE_IMPORT'
    var_16 = module_0.from_string(var_15)
    assert var_16 == module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT
    var_17 = 'HANGING_INDENT_WITH_PARENTHESES'
    var_18 = module_0.from_string(var_17)
    assert var_18 == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    var_19 = 'BACKSLASH_GRID'
    var_20 = module_0.from_string(var_19)
    assert var_20 == module_0.WrapModes.BACKSLASH_GRID
    var_21 = '0'
    var_22 = module_0.from_string(var_21)
    assert var_22 == module_0.WrapModes.GRID
    var_23 = '1'
    var_24 = module_0.from_string(var_23)
    assert var_24 == module_0.WrapModes.VERTICAL
    var_25 = '2'
    var_26 = module_0.from_string(var_25)
    assert var_26 == module_0.WrapModes.HANGING_INDENT
    var_27 = 'invalid'
    module_0.from_string(var_27)
    assert var_28 is None