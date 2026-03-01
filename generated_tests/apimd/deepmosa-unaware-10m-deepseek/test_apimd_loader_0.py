# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.loader as module_0
import posixpath as module_1

def test_case_0():
    var_0 = ';'
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
    var_1 = ''
    var_2 = ' B9'
    var_3 = module_0.loader(var_2, var_1, var_0, var_0, var_0)
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
    module_0.gen_api(var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ''
    module_0.loader(var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = ''
    var_1 = True
    module_0.loader(var_0, var_0, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = ''
    var_2 = True
    var_3 = ' BQ'
    var_4 = module_0.loader(var_3, var_1, var_0, var_0, var_0)
    assert var_4 == '\n'
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 3050
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
    module_0.loader(var_1, var_1, var_0, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'VgHBK<T5|\x0c'
    var_1 = ':9j+="c1i\x0bD)b"cWElB'
    var_2 = module_0.walk_packages(var_0, var_1)
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 3050
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
    var_3 = module_1.abspath(var_1)
    assert var_3 == '/workspace/:9j+="c1i\x0bD)b"cWElB'
    assert module_1.curdir == '.'
    assert module_1.pardir == '..'
    assert module_1.extsep == '.'
    assert module_1.sep == '/'
    assert module_1.pathsep == ':'
    assert module_1.defpath == '/bin:/usr/bin'
    assert module_1.altsep is None
    assert module_1.devnull == '/dev/null'
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    assert module_1.supports_unicode_filenames is False
    var_4 = module_1.dirname(var_3)
    assert var_4 == '/workspace'
    var_5 = True
    module_0.gen_api(var_4, prefix=var_4, link=var_4, toc=var_5)