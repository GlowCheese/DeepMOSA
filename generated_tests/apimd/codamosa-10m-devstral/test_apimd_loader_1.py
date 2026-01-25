# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.loader as module_0

def test_case_0():
    var_0 = '.'
    var_1 = 'N"i"S!!9@LFc}H@\t'
    var_2 = False
    var_3 = None
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_3)
    assert var_4 == '\n'
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
    var_0 = '[3(!!21R|FwnLgz@Ix#'
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
    var_2 = 'x_SO'
    var_3 = 'wS*_X>"?\'mZK8>('
    var_4 = module_0.walk_packages(var_2, var_3)
    var_5 = None
    var_6 = 'Ipp17igLG(2jW.o'
    module_0.gen_api(var_5, var_3, prefix=var_6)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '.'
    var_1 = None
    module_0.loader(var_0, var_0, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = ''
    var_1 = None
    var_2 = True
    module_0.loader(var_0, var_0, var_2, var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '.'
    var_1 = 'N"i"S!!9@LFc}H@\t'
    var_2 = None
    var_3 = -3967
    module_0.gen_api(var_1, prefix=var_0, level=var_3, dry=var_2)