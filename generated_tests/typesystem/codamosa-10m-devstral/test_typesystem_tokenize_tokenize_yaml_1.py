# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.tokenize.tokenize_yaml as module_0
import typesystem.base as module_1
import yaml.loader as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.tokenize_yaml(var_0)

def test_case_1():
    var_0 = '{P:Ms>'
    with pytest.raises(module_1.ParseError):
        module_0.tokenize_yaml(var_0)

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
    var_0 = b'\xe1\xe9Lh}\xf0\xf5'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.ScalarToken'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '2'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '?'
    var_1 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_2 = module_2.SafeLoader(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'yaml.loader.SafeLoader'
    assert var_2.name == '<unicode string>'
    assert var_2.stream is None
    assert var_2.stream_pointer == 0
    assert var_2.eof is True
    assert var_2.buffer == '?\x00'
    assert var_2.pointer == 0
    assert var_2.raw_buffer is None
    assert var_2.raw_decode is None
    assert var_2.encoding is None
    assert var_2.index == 0
    assert var_2.line == 0
    assert var_2.column == 0
    assert var_2.done is False
    assert var_2.flow_level == 0
    assert f'{type(var_2.tokens).__module__}.{type(var_2.tokens).__qualname__}' == 'builtins.list'
    assert len(var_2.tokens) == 1
    assert var_2.tokens_taken == 0
    assert var_2.indent == -1
    assert var_2.indents == []
    assert var_2.allow_simple_key is True
    assert var_2.possible_simple_keys == {}
    assert var_2.current_event is None
    assert var_2.yaml_version is None
    assert var_2.tag_handles == {}
    assert var_2.states == []
    assert var_2.marks == []
    assert var_2.anchors == {}
    assert var_2.constructed_objects == {}
    assert var_2.recursive_objects == {}
    assert var_2.state_generators == []
    assert var_2.deep_construct is False
    assert var_2.resolver_exact_paths == []
    assert var_2.resolver_prefix_paths == []
    var_3 = module_0.tokenize_yaml(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.tokenize.tokens.DictToken'
    var_4 = [var_3]
    var_3.lookup(var_4)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '-'
    module_0.validate_yaml(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '.6'
    module_0.validate_yaml(var_0, var_0)