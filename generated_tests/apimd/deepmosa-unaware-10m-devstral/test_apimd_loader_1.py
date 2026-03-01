# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.loader as module_0

def test_case_0():
    var_0 = 'F*:.[}n"'
    var_1 = None
    var_2 = module_0.loader(var_0, var_0, var_1, var_1, var_1)
    assert var_2 == '\n'
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert module_0.sep == '/'
    assert module_0.EXTENSION_SUFFIXES == ['.cpython-310-x86_64-linux-gnu.so', '.abi3.so', '.so']
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP561_SUFFIX == '-stubs'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.gen_api(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.walk_packages(var_0, var_0)
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert module_0.sep == '/'
    assert module_0.EXTENSION_SUFFIXES == ['.cpython-310-x86_64-linux-gnu.so', '.abi3.so', '.so']
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP561_SUFFIX == '-stubs'
    var_2 = None
    var_3 = ';LP6z}czV3W\x0c$'
    var_4 = False
    var_5 = module_0.loader(var_3, var_3, var_2, var_0, var_4)
    assert var_5 == '\n'
    module_0.gen_api(var_0, var_4, dry=var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ''
    var_1 = True
    module_0.loader(var_0, var_0, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = 'l<7?a5f:\x0biu[g9f{u='
    var_2 = module_0.walk_packages(var_0, var_0)
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert module_0.sep == '/'
    assert module_0.EXTENSION_SUFFIXES == ['.cpython-310-x86_64-linux-gnu.so', '.abi3.so', '.so']
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP561_SUFFIX == '-stubs'
    var_3 = ''
    var_4 = module_0.loader(var_1, var_3, var_0, var_0, var_0)
    assert var_4 == '\n'
    var_5 = '^z7,Bis@.1n5!:73[L'
    var_6 = module_0.walk_packages(var_5, var_0)
    var_7 = '5=$[A[>Cy$@3cJ'
    var_8 = 'u40\r^3dPL},9K9AVz'
    var_9 = {var_7: var_1, var_8: var_1}
    module_0.gen_api(var_9, var_8)