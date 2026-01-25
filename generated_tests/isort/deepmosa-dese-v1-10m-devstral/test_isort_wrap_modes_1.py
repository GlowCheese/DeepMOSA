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

def test_case_3():
    var_0 = 'test '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test \\'

def test_case_4():
    var_0 = 'test'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test \\'

def test_case_5():
    var_0 = None
    var_1 = 'a,[> ;9<D=~{$e(}'
    var_2 = []
    var_3 = module_0._wrap_mode_interface(var_0, var_0, var_0, var_1, var_0, var_2, var_0, var_0, var_0, var_0)
    assert var_3 == ''

def test_case_6():
    with pytest.raises(NotImplementedError):
        module_0.vertical_grid_grouped_no_comma()