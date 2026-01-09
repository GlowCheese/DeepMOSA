# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_1
import typesystem.tokenize.tokenize_json as module_0


def test_case_0():
    var_0 = '9#'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = b'\x80\xa5{\xc4\x94\x15\xb8\x10\xf7\x11\xf1\x7f\x88'
    module_0.validate_json(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.tokenize_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0._TokenizingDecoder()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.validate_json(var_0, var_0)

def test_case_5():
    var_0 = 'k<uW_K\n=Z`2j`C\t*p'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_6():
    var_0 = '\t'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_7():
    var_0 = 'f#'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '[wW:U>H\r'
    module_0.validate_json(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'nNH?G5a5'
    module_0.validate_json(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'tIbglG9f,M?o@*H-41'
    module_0.validate_json(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '["^%Coz2HD(W'
    module_0.validate_json(var_0, var_0)

def test_case_12():
    var_0 = '['
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_13():
    var_0 = b'{\xf7\xe1}[\x88\xfe'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_14():
    var_0 = b'\x80\xa5{\n\xc4\x94\x15\xb8\x10\xf7\x11\xf1\x7f\x88'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = b'\x80\xa5{"\x94\x15\xb8\x10\xf7\x11\xf1\x7f\x88'
    module_0.validate_json(var_0, var_0)

def test_case_16():
    var_0 = '4E1"\tS)g=6;sQ'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)

def test_case_17():
    var_0 = '3.0\t#ENB]'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_json(var_0)