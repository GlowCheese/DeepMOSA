# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.types as module_0

def test_case_0():
    var_0 = module_0._MissingSeed()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Timestamp).__module__}.{type(module_0.Timestamp).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.Keywords).__module__}.{type(module_0.Keywords).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.Number).__module__}.{type(module_0.Number).__qualname__}' == 'types.UnionType'
    var_1 = module_0._MissingSeed()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.types._MissingSeed'