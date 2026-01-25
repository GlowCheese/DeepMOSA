# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.positional_validation as module_0
import typesystem.schemas as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.validate_with_positions(token=var_0, validator=var_0)

def test_case_1():
    pass

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = {}
    var_1 = module_1.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = var_1.validate_or_error(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    module_0.validate_with_positions(token=var_2, validator=var_1)