# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_1
import typesystem.tokenize.tokenize_yaml as module_0


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.tokenize_yaml(var_0)

def test_case_1():
    var_0 = '\n    name: John Doe\n    age: : 30\n    '
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = "!la+3\rsB4zs'Al"
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = b'\xae\xae[\x10o\x9e\xb0\x1e>\x93\xd9\xfb\xea\xee'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '\t'
    module_0.validate_yaml(var_0, var_0)

def test_case_5():
    var_0 = b'\xac\xbf)K\xef\x9d\xdc\xe6\xa4\x8b\xcb\xc9'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_6():
    var_0 = '\n    name: John Doe\n    age: 30\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '~'
    module_0.validate_yaml(var_0, var_0)

def test_case_8():
    var_0 = '\n    price: 9.99\n    '
    var_1 = 'All test cases passed!'
    var_2 = print(var_1)
    var_3 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

def test_case_9():
    var_0 = b'\xa7-\xbe\x95'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

def test_case_10():
    var_0 = '\n    active: true\n    '
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2 = '\n    price: 9.99\n    '
    var_3 = 'All test cases passed!'
    var_4 = print(var_3)
    var_5 = module_0.tokenize_yaml(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'