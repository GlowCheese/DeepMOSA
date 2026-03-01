# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.wrap_modes as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '0}cr#{YK`.nU]dnJM'
    module_0.from_string(var_0)

def test_case_1():
    pass

def test_case_2():
    var_0 = '0}cr#{YK`.nU]dnJM'
    var_1 = module_0.formatter_from_string(var_0)

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
    var_4 = 'HANGING_INDENT'
    var_5 = module_0.from_string(var_4)
    assert var_5 == module_0.WrapModes.HANGING_INDENT
    var_6 = '0'
    var_7 = module_0.from_string(var_6)
    assert var_7 == module_0.WrapModes.GRID
    var_8 = '1'
    var_9 = module_0.from_string(var_8)
    assert var_9 == module_0.WrapModes.VERTICAL
    var_10 = '2'
    var_11 = module_0.from_string(var_10)
    assert var_11 == module_0.WrapModes.HANGING_INDENT
    var_12 = 'invalid'
    module_0.from_string(var_12)
    assert var_13 is None