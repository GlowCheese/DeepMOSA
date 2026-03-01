# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = b'G\xd0\x1e'
    module_0.tokenize_yaml(var_0)

def test_case_3():
    var_0 = '7YAn'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_4():
    var_0 = ''
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_5():
    var_0 = b'42'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_6():
    var_0 = 'inv!lid: yaml:Gcontent:!['
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'key: null'
    var_1 = 'key'
    module_0.validate_yaml(var_0, var_1)

def test_case_8():
    var_0 = 'invalid: yaml: content: ['
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_9():
    var_0 = '[1, 2, 3]'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_2 = var_1.value
    var_3 = bool(var_1.value == [1, 2, 3])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '[1, 2, 3]'
    var_5 = 1
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_2)
    var_8 = 9
    var_9 = module_1.Position(var_5, var_8, var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no == 1
    assert var_9.column_no == 9
    assert var_9.char_index is False
    var_10 = bool(var_1.end == var_9)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'int: 42\nfloat: 3.14\nbool: true\nnull: null'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'int': 42, 'float': 3.14, 'bool': True, 'null': None})
    var_4 = 'float'
    var_5 = [var_4]
    var_6 = var_1.lookup(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_7 = var_1.lookup(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = 'null'
    var_9 = [var_8]
    var_1.lookup(var_9)