# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1
import typesystem.fields as module_2

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
    var_0 = b'\xe9\x97\x1e'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '7:'
    var_1 = None
    module_0.validate_yaml(var_0, var_1)

def test_case_6():
    var_0 = 'Lj8FDS;GPIw:'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

def test_case_7():
    var_0 = b'\x83-'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

def test_case_8():
    var_0 = '%JxF4]'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_9():
    var_0 = module_2.String()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.String'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format is None
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = module_2.Integer()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Integer'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.minimum is None
    assert var_1.maximum is None
    assert var_1.exclusive_minimum is None
    assert var_1.exclusive_maximum is None
    assert var_1.multiple_of is None
    assert var_1.precision is None
    assert var_1.coerce_types is True
    var_2 = module_2.String()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.String'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.allow_blank is False
    assert var_2.trim_whitespace is True
    assert var_2.max_length is None
    assert var_2.min_length is None
    assert var_2.format is None
    assert var_2.coerce_types is True
    assert var_2.pattern is None
    assert var_2.pattern_regex is None
    var_3 = '\n    name: John Doe\n    age: [unclosed list\n    '
    var_4 = '\n    name: John Doe\n    age: not_a_number\n    active: true\n    '
    var_5 = module_0.tokenize_yaml(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_3)

def test_case_10():
    var_0 = 'hvllo'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = '123'
    var_3 = module_0.tokenize_yaml(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_4 = '45.67'
    var_5 = module_0.tokenize_yaml(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_6 = 'true'
    var_7 = module_0.tokenize_yaml(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = 'null'
    var_9 = module_0.tokenize_yaml(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_10 = '- item1\n- item2'
    var_11 = module_0.tokenize_yaml(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'
    var_12 = var_11.value
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'key: value\nfoo: bar'
    var_15 = module_0.tokenize_yaml(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_16 = 'parent:\n  child: 123'
    var_17 = module_0.tokenize_yaml(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_18 = 'parent'
    var_19 = var_17.value[var_18]
    var_20 = b'name: test'
    var_21 = module_0.tokenize_yaml(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_22 = 'key: : value'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_22)