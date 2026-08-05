# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import json as module_0
import src.itsdangerous._json as module_1

def test_case_0():
    var_0 = None
    with pytest.raises(TypeError):
        module_0.loads(var_0, object_hook=var_0, parse_float=var_0, object_pairs_hook=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1._CompactJSON()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous._json._CompactJSON'
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    var_1 = None
    var_0.loads(var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1._CompactJSON()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'src.itsdangerous._json._CompactJSON'
    assert f'{type(module_1.annotations).__module__}.{type(module_1.annotations).__qualname__}' == '__future__._Feature'
    assert module_1.annotations.optional == (3, 7, 0, 'beta', 1)
    assert module_1.annotations.mandatory == (3, 11, 0, 'alpha', 0)
    assert module_1.annotations.compiler_flag == 16777216
    var_1 = None
    var_2 = 'r4)x((Y@\x0cjKOs\x0b.~Bh'
    var_3 = 'm++[i'
    var_4 = 't8xLt5\\q%KNYW55'
    var_5 = {var_2: var_1, var_2: var_1, var_3: var_1, var_4: var_1}
    var_0.dumps(var_1, **var_5)