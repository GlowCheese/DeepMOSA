# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.wrap_modes as module_0

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
    var_6 = 'VERTICAL_HANGING_INDENT'
    var_7 = module_0.from_string(var_6)
    assert var_7 == module_0.WrapModes.VERTICAL_HANGING_INDENT
    var_8 = 'VERTICAL_GRID'
    var_9 = module_0.from_string(var_8)
    assert var_9 == module_0.WrapModes.VERTICAL_GRID
    var_10 = 'VERTICAL_GRID_GROUPED'
    var_11 = module_0.from_string(var_10)
    assert var_11 == module_0.WrapModes.VERTICAL_GRID_GROUPED
    var_12 = 'NOQA'
    var_13 = module_0.from_string(var_12)
    assert var_13 == module_0.WrapModes.NOQA
    var_14 = 'VERTICAL_HANGING_INDENT_BRACKET'
    var_15 = module_0.from_string(var_14)
    assert var_15 == module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET
    var_16 = 'VERTICAL_PREFIX_FROM_MODULE_IMPORT'
    var_17 = module_0.from_string(var_16)
    assert var_17 == module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT
    var_18 = 'HANGING_INDENT_WITH_PARENTHESES'
    var_19 = module_0.from_string(var_18)
    assert var_19 == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    var_20 = 'BACKSLASH_GRID'
    var_21 = module_0.from_string(var_20)
    assert var_21 == module_0.WrapModes.BACKSLASH_GRID
    var_22 = '0'
    var_23 = module_0.from_string(var_22)
    assert var_23 == module_0.WrapModes.GRID
    var_24 = '1'
    var_25 = module_0.from_string(var_24)
    assert var_25 == module_0.WrapModes.VERTICAL
    var_26 = '2'
    var_27 = module_0.from_string(var_26)
    assert var_27 == module_0.WrapModes.HANGING_INDENT
    var_28 = '3'
    var_29 = module_0.from_string(var_28)
    assert var_29 == module_0.WrapModes.VERTICAL_HANGING_INDENT
    var_30 = '4'
    var_31 = module_0.from_string(var_30)
    assert var_31 == module_0.WrapModes.VERTICAL_GRID
    var_32 = '5'
    var_33 = module_0.from_string(var_32)
    assert var_33 == module_0.WrapModes.VERTICAL_GRID_GROUPED
    var_34 = '6'
    var_35 = module_0.from_string(var_34)
    assert var_35 == module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA
    var_36 = '7'
    var_37 = module_0.from_string(var_36)
    assert var_37 == module_0.WrapModes.NOQA
    var_38 = '8'
    var_39 = module_0.from_string(var_38)
    assert var_39 == module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET
    var_40 = '9'
    var_41 = module_0.from_string(var_40)
    assert var_41 == module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT
    var_42 = '10'
    var_43 = module_0.from_string(var_42)
    assert var_43 == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    var_44 = 'grid'
    module_0.from_string(var_44)