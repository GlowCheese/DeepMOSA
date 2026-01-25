# Check out: https://github.com/GlowCheese/deepmosa
import dataclasses as module_0


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = module_0.dataclass(var_1, init=var_1, unsafe_hash=var_1, match_args=var_1)
    assert f'{type(module_0.MISSING).__module__}.{type(module_0.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_0.KW_ONLY).__module__}.{type(module_0.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_3 = var_2.__eq__(var_0)