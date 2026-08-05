# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.loader as module_0
import genericpath as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = ''
    var_1 = 'R2VFC>4/eJ~'
    var_2 = {var_0: var_1, var_1: var_0}
    var_3 = True
    module_0.gen_api(var_2, toc=var_3)

def test_case_1():
    var_0 = ''
    var_1 = module_0.loader(var_0, var_0, var_0, var_0, var_0)
    assert var_1 == '\n'
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
def test_case_2():
    var_0 = None
    module_0.gen_api(var_0, toc=var_0)

def test_case_3():
    var_0 = 'U{(C%?Rm'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.gen_api(var_1, prefix=var_0, dry=var_0)
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
    var_0 = None
    var_1 = '*,,jc4WJjhxK-vtw'
    var_2 = 'X'
    var_3 = "!/'bnp#W*|"
    var_4 = 'O\tk2I^JJ=Cu#z7='
    var_5 = {var_1: var_1, var_2: var_2, var_3: var_2, var_4: var_3}
    var_6 = 'O[b~%;='
    var_7 = module_0.gen_api(var_5, var_0, prefix=var_6)
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
    var_8 = module_1.isfile(var_2)
    assert var_8 is False
    assert f'{type(module_1.ALLOW_MISSING).__module__}.{type(module_1.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    module_0.gen_api(var_5, var_1, prefix=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '\nut8wKzM~ `aL^Z'
    var_1 = 'r8h_1B1-FA'
    var_2 = "o6`&L#67c}`'ZvaD"
    var_3 = {var_0: var_0, var_0: var_0, var_1: var_1, var_2: var_0}
    var_4 = module_0.gen_api(var_3, prefix=var_1, dry=var_0)
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 2102
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
    var_5 = module_0.gen_api(var_3, prefix=var_2)
    var_6 = None
    var_7 = module_0.walk_packages(var_6, var_0)
    var_8 = True
    var_9 = module_0.gen_api(var_3, prefix=var_1, toc=var_8, dry=var_8)
    var_10 = 'o'
    module_0._write(var_3, var_10)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'test_empty.txt'
    var_1 = None
    module_0._write(var_0, var_1)

def test_case_7():
    var_0 = '/'
    var_1 = None
    var_2 = module_0.loader(var_0, var_0, var_0, var_0, var_1)
    assert var_2 == '\n'
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 2102
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
def test_case_8():
    var_0 = '/'
    var_1 = None
    var_2 = ''
    var_3 = True
    module_0.loader(var_2, var_0, var_1, var_1, var_3)