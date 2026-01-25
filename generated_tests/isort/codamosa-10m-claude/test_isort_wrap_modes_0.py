# Check out: https://github.com/GlowCheese/deepmosa
import isort.wrap_modes as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.from_string(var_0)

def test_case_1():
    pass

def test_case_2():
    var_0 = ''
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
    var_6 = 'VERTICAL_HANGING_INDENT'
    var_7 = module_0.from_string(var_6)
    assert var_7 == module_0.WrapModes.VERTICAL_HANGING_INDENT
    var_8 = module_0.from_string(var_2)
    assert var_8 == module_0.WrapModes.VERTICAL
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
    var_27 = '3'
    var_28 = module_0.from_string(var_27)
    assert var_28 == module_0.WrapModes.VERTICAL_HANGING_INDENT
    var_29 = '4'
    var_30 = module_0.from_string(var_29)
    assert var_30 == module_0.WrapModes.VERTICAL_GRID
    var_31 = '5'
    var_32 = module_0.from_string(var_31)
    assert var_32 == module_0.WrapModes.VERTICAL_GRID_GROUPED
    var_33 = '6'
    var_34 = module_0.from_string(var_33)
    assert var_34 == module_0.WrapModes.VERTICAL_GRID_GROUPED_NO_COMMA
    var_35 = '7'
    var_36 = module_0.from_string(var_35)
    assert var_36 == module_0.WrapModes.NOQA
    var_37 = '8'
    var_38 = module_0.from_string(var_37)
    assert var_38 == module_0.WrapModes.VERTICAL_HANGING_INDENT_BRACKET
    var_39 = '9'
    var_40 = module_0.from_string(var_39)
    assert var_40 == module_0.WrapModes.VERTICAL_PREFIX_FROM_MODULE_IMPORT
    var_41 = '10'
    var_42 = module_0.from_string(var_41)
    assert var_42 == module_0.WrapModes.HANGING_INDENT_WITH_PARENTHESES
    var_43 = '11'
    var_44 = module_0.from_string(var_43)
    assert var_44 == module_0.WrapModes.BACKSLASH_GRID
    var_45 = 'grid'
    module_0.from_string(var_45)