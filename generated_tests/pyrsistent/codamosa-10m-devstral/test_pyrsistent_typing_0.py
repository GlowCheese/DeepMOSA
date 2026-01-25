# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent.typing as module_0

def test_case_0():
    var_0 = module_0.PSetEvolver()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent.typing.PSetEvolver'
    assert f'{type(module_0.absolute_import).__module__}.{type(module_0.absolute_import).__qualname__}' == '__future__._Feature'
    assert module_0.absolute_import.optional == (2, 5, 0, 'alpha', 1)
    assert module_0.absolute_import.mandatory == (3, 0, 0, 'alpha', 0)
    assert module_0.absolute_import.compiler_flag == 262144
    assert f'{type(module_0.T).__module__}.{type(module_0.T).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.T_co).__module__}.{type(module_0.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.KT).__module__}.{type(module_0.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT).__module__}.{type(module_0.VT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_0.VT_co).__module__}.{type(module_0.VT_co).__qualname__}' == 'typing.TypeVar'