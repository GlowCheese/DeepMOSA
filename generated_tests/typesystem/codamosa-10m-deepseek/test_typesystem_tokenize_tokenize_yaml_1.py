# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_1
import typesystem.tokenize.tokenize_yaml as module_0


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '*!]qAP=['
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = b'G\xd0\x1e'
    module_0.tokenize_yaml(var_0)

def test_case_3():
    var_0 = b''
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.validate_yaml(var_0, var_0)

def test_case_5():
    var_0 = b'\xf2\x88J*\xe9\xd8[#A\xc7\xec\xd3\xc2\xf8'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '8'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = b'\xc9S\x82\xa2'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = '!'
    module_0.validate_yaml(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '-'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '.6'
    module_0.validate_yaml(var_0, var_0)

def test_case_10():
    var_0 = '\n    name: John\n    age: 30\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_3 = '\n    person:\n      name: John\n      age: 30\n      hobbies:\n        - reading\n        - swimming\n    '
    var_4 = module_0.tokenize_yaml(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_5 = '\n    name: John\n    age: 30\n    height: 1.75\n    is_student: false\n    address: null\n    '
    var_6 = module_0.tokenize_yaml(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_7 = 'All test cases passed!'
    var_8 = print(var_7)