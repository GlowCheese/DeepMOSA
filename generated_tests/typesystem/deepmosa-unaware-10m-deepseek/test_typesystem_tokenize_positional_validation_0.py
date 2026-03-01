# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.positional_validation as module_0
import typesystem.fields as module_1
import typesystem.tokenize.tokens as module_2
import typesystem.base as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.validate_with_positions(token=var_0, validator=var_0)

def test_case_1():
    var_0 = 1573
    var_1 = module_1.String(max_length=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.String'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.allow_blank is False
    assert var_1.trim_whitespace is True
    assert var_1.max_length == 1573
    assert var_1.min_length is None
    assert var_1.format is None
    assert var_1.coerce_types is True
    assert var_1.pattern is None
    assert var_1.pattern_regex is None
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_2 = module_2.ScalarToken(var_0, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    with pytest.raises(module_3.ValidationError):
        module_0.validate_with_positions(token=var_2, validator=var_1)