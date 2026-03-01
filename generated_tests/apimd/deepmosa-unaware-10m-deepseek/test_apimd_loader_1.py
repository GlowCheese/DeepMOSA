# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.loader as module_0

def test_case_0():
    var_0 = 'JK{TO)6)\x0bFImT((pIcs'
    var_1 = module_0.loader(var_0, var_0, var_0, var_0, var_0)
    assert var_1 == '**Table of contents:**\n\n\n'
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
    var_1 = True
    var_2 = 'mX@)".se/\tR\t'
    var_3 = module_0.loader(var_2, var_2, var_1, var_0, var_0)
    assert var_3 == '\n'
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
    var_4 = {var_3: var_3}
    var_5 = False
    module_0.gen_api(var_4, var_3, link=var_1, toc=var_5, dry=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ''
    module_0.loader(var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
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
    var_3 = 'l<7?a5f:\x0biu[g9f{u='
    var_4 = module_0.walk_packages(var_0, var_2)
    var_5 = ''
    var_6 = module_0.loader(var_3, var_5, var_0, var_0, var_0)
    assert var_6 == '\n'
    var_7 = '^z7,Bis@.1n5!:73[L'
    var_8 = module_0.walk_packages(var_7, var_2)
    var_9 = '5=$[A[>Cy$@3cJ'
    var_10 = 'u40\r^3dPL},9K9AVz'
    var_11 = {var_9: var_3, var_10: var_3}
    module_0.gen_api(var_11, var_10)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = False
    var_1 = ''
    module_0.loader(var_1, var_1, var_0, var_0, var_0)