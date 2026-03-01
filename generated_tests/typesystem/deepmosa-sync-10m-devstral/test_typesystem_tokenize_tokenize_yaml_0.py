# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1
import typesystem.fields as module_2
import typesystem.schemas as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = "!la+3\rsB4zs'Al"
    module_0.validate_yaml(var_0, var_0)

def test_case_2():
    var_0 = 'invalid: yaml: content: [unclosed'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'qws'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = b'\xa3yv\xcaf\x97$\xb2\xb0\xde\x10\x94\xdd'
    module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = b'\xdd'
    var_1 = None
    module_0.validate_yaml(var_0, var_1)

def test_case_6():
    var_0 = '\rjDw`}JRZm)]P&O4wO:'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

def test_case_7():
    var_0 = '{a: [1, 2], b: {c: 3}}'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': [1, 2], 'b': {'c': 3}})
    assert var_3 is True
    var_4 = 'a'
    var_5 = 0
    var_6 = [var_4, var_5]
    var_7 = var_1.lookup(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_8 = var_7.value
    assert var_8 == 1
    var_9 = 1
    var_10 = [var_4, var_9]
    var_11 = var_1.lookup(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_12 = var_11.value
    assert var_12 == 2
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_13, var_14]
    var_16 = var_1.lookup(var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_17 = var_16.value
    assert var_17 == 3
    var_18 = [var_13, var_14]
    var_19 = var_1.lookup_key(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_20 = var_19.value
    assert var_20 == 'c'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '7'
    var_1 = 'key'
    var_2 = {}
    var_3 = module_2.String(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.String'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.allow_blank is False
    assert var_3.trim_whitespace is True
    assert var_3.max_length is None
    assert var_3.min_length is None
    assert var_3.format is None
    assert var_3.coerce_types is True
    assert var_3.pattern is None
    assert var_3.pattern_regex is None
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_3.Schema(var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['key']
    assert module_3.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    module_0.validate_yaml(var_0, var_6)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 1
    assert var_7.column_no == 1
    assert var_7.char_index == 0
    var_8 = bool(var_1.start == var_7)
    assert var_8 is True
    var_2.__contains__(var_2)

def test_case_10():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 0
    var_5 = module_1.Position(var_1, var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.line_no).__module__}.{type(var_5.line_no).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    assert f'{type(var_5.column_no).__module__}.{type(var_5.column_no).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    assert var_5.char_index == 0
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    var_8 = 4
    var_9 = -3
    var_10 = module_1.Position(var_8, var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 4
    assert var_10.column_no == 4
    assert var_10.char_index == -3
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)