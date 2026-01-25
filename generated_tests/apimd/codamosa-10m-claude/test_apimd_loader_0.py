# Check out: https://github.com/GlowCheese/deepmosa
import apimd.loader as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = 'v=(2M|JZ(!I+#FNy'
    var_2 = False
    var_3 = module_0.loader(var_1, var_1, var_0, var_2, var_0)
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
    module_0.gen_api(var_0, var_0, level=var_0, toc=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.gen_api(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = 'v=(2M|JZ(!I+#FNy'
    var_2 = False
    var_3 = module_0.loader(var_1, var_1, var_0, var_2, var_0)
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
    var_4 = {}
    module_0.gen_api(var_4, var_1, toc=var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ''
    var_1 = True
    var_2 = None
    module_0.loader(var_0, var_0, var_1, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '"P'
    var_1 = ''
    var_2 = -217
    var_3 = None
    var_4 = module_0.loader(var_0, var_1, var_0, var_2, var_3)
    assert var_4 == '\n'
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 1070
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
    var_5 = module_0.walk_packages(var_3, var_3)
    var_6 = module_0.walk_packages(var_3, var_3)
    module_0.gen_api(var_3)