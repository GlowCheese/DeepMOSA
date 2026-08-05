# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.loader as module_0

def test_case_0():
    var_0 = '9f<h\x0csr4RH|'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.gen_api(var_1, var_1)
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
    module_0.gen_api(var_0)

def test_case_3():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
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
def test_case_4():
    var_0 = '9f<h\x0csr4RH|'
    module_0.gen_api(var_0, var_0)

def test_case_5():
    var_0 = '9f<h\x0csr4RH|'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_0.gen_api(var_1, var_1)
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
def test_case_6():
    var_0 = None
    var_1 = 'dr4/u"},\'Uwj}i@x'
    var_2 = ')R2Bh\\1-'
    var_3 = 'DTil'
    var_4 = {var_1: var_1, var_2: var_2, var_1: var_3}
    module_0.gen_api(var_4, var_2, link=var_0, level=var_0, toc=var_4)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = 'uKJV))_ '
    var_2 = module_0.walk_packages(var_1, var_0)
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
    var_3 = module_0.walk_packages(var_0, var_0)
    var_4 = 'm0,e;Ga 58Y#51'
    var_5 = '?'
    var_6 = module_0.walk_packages(var_5, var_3)
    var_7 = '!|B9Ch&GIB;X;5'
    var_8 = 'oMd(2'
    var_9 = {var_4: var_8}
    var_10 = module_0.gen_api(var_9, toc=var_8, dry=var_0)
    var_11 = module_0.loader(var_4, var_7, var_0, var_0, var_0)
    assert var_11 == '\n'
    module_0.gen_api(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = ''
    var_2 = '!6YQ3#H|mK'
    var_3 = ':XI'
    var_4 = 'T<%Wd}Q;lE 9P+G'
    var_5 = module_0.walk_packages(var_0, var_4)
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
    var_6 = {var_1: var_1, var_1: var_2, var_3: var_3}
    var_7 = True
    var_8 = True
    var_9 = module_0.gen_api(var_6, var_2, link=var_7, level=var_7, toc=var_8, dry=var_7)
    var_10 = module_0.walk_packages(var_0, var_1)
    module_0.loader(var_0, var_0, var_0, var_0, var_8)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = 'ZqqQ'
    module_0._write(var_1, var_0)