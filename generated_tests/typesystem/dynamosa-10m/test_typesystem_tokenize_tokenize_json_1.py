# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_1
import typesystem.tokenize.tokenize_json as module_0


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'Dhe%'
    module_0.validate_json(var_0, var_0)

def test_case_1():
    var_0 = b'\x8d'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.tokenize_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0._TokenizingDecoder()

def test_case_4():
    var_0 = b'\x829\x95+\x89\xd2\x80\xeb\xec\xdd;'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'fb?Ivrq\ni&'
    module_0.validate_json(var_0, var_0)

def test_case_6():
    var_0 = '['
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_7():
    var_0 = b'"\xdf6\xc9\r\xf9\x93\xa5\xcas${\xf4\xb9\x15p\x07'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_8():
    var_0 = '\ntZS'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_9():
    var_0 = b'n'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_10():
    var_0 = '{T)n*xtEjnq]p'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '{\r)n*xtj6nM'
    module_0.validate_json(var_0, var_0)

def test_case_12():
    var_0 = '{\r}@)Vv*xdtj66M'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_13():
    var_0 = b'\xfb4E\x94\xda\xf1\xde8\x18\x8d>\nlA#Y\x13\x88\x02\xf0'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_14():
    var_0 = '{\r"4}@)Vv*xdtj66M'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_15():
    var_0 = '72.0-'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_16():
    var_0 = '{"W~X:k&}.pI9y.N$'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)