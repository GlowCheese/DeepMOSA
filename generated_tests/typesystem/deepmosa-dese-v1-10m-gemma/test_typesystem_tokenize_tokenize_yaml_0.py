# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1
import yaml.constructor as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.tokenize_yaml(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = "!la+3\rsB4zs'Al"
    module_0.validate_yaml(var_0, var_0)

def test_case_2():
    var_0 = 'g.\ncHT^\tM;GC{aZ/nb'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_3():
    var_0 = b'Vix\xa2_g\x96bZ\xc5A.'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_4():
    var_0 = ''
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

def test_case_5():
    var_0 = '123'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

def test_case_6():
    var_0 = b'name: tester'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'

def test_case_7():
    var_0 = '-'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '7.'
    var_1 = module_2.SafeConstructor()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'yaml.constructor.SafeConstructor'
    assert var_1.constructed_objects == {}
    assert var_1.recursive_objects == {}
    assert var_1.state_generators == []
    assert var_1.deep_construct is False
    assert module_2.SafeConstructor.bool_values == {'yes': True, 'no': False, 'true': True, 'false': False, 'on': True, 'off': False}
    assert module_2.SafeConstructor.inf_value == pytest.approx(1e309, abs=0.01, rel=0.01)
    assert f'{type(module_2.SafeConstructor.timestamp_regexp).__module__}.{type(module_2.SafeConstructor.timestamp_regexp).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.SafeConstructor.yaml_constructors).__module__}.{type(module_2.SafeConstructor.yaml_constructors).__qualname__}' == 'builtins.dict'
    assert len(module_2.SafeConstructor.yaml_constructors) == 13
    var_2 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'
    var_3 = None
    module_0.tokenize_yaml(var_3)

def test_case_9():
    var_0 = '- item1\n- item2'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ListToken'

def test_case_10():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'