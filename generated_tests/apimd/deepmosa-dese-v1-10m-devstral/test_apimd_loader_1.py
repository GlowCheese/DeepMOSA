# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.loader as module_0
import posixpath as module_1
import apimd.parser as module_2

def test_case_0():
    var_0 = 'os'
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
    var_0 = ' -Fwk_>bJS9T\\)*dpqv'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''
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
    var_2 = module_1.abspath(var_1)
    assert var_2 == '/workspace'
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
    var_3 = {var_0: var_0, var_1: var_2, var_1: var_1}
    module_0.gen_api(var_3, var_2, prefix=var_2)

def test_case_3():
    pass

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = ''
    module_0.loader(var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'x-FIE_>bJS9T\\)*N<pqv'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 6796
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
    var_2 = module_1.abspath(var_1)
    assert var_2 == '/workspace'
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
    var_3 = {var_0: var_0, var_1: var_0, var_1: var_1}
    module_0.gen_api(var_3, prefix=var_2, level=var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = ''
    var_1 = None
    var_2 = 958
    module_0.loader(var_0, var_0, var_1, var_2, var_1)

def test_case_7():
    var_0 = 'nonexistent_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 6796
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

def test_case_8():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 6796
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

def test_case_9():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = module_2.Parser()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc == {}
    assert var_2.docstring == {}
    assert var_2.imp == {}
    assert var_2.root == {}
    assert var_2.alias == {}
    assert var_2.const == {}
    assert f'{type(module_2.logger).__module__}.{type(module_2.logger).__qualname__}' == 'logging.RootLogger'
    assert module_2.logger.filters == []
    assert module_2.logger.name == 'root'
    assert module_2.logger.level == 10
    assert module_2.logger.parent is None
    assert module_2.logger.propagate is True
    assert f'{type(module_2.logger.handlers).__module__}.{type(module_2.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_2.logger.handlers) == 2
    assert module_2.logger.disabled is False
    assert module_2.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_2.ANY == 'Any'
    assert module_2.Parser.link is True
    assert module_2.Parser.b_level == 1
    assert module_2.Parser.toc is False
    assert f'{type(module_2.Parser.new).__module__}.{type(module_2.Parser.new).__qualname__}' == 'builtins.method'
    var_3 = module_0._load_module(var_0, var_1, var_2)
    assert var_3 is False
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 6796
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
def test_case_10():
    var_0 = None
    var_1 = False
    var_2 = module_2.Parser(var_0, toc=var_1, alias=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is None
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc == {}
    assert var_2.docstring == {}
    assert var_2.imp == {}
    assert var_2.root == {}
    assert var_2.alias is False
    assert var_2.const == {}
    assert f'{type(module_2.logger).__module__}.{type(module_2.logger).__qualname__}' == 'logging.RootLogger'
    assert module_2.logger.filters == []
    assert module_2.logger.name == 'root'
    assert module_2.logger.level == 10
    assert module_2.logger.parent is None
    assert module_2.logger.propagate is True
    assert f'{type(module_2.logger.handlers).__module__}.{type(module_2.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_2.logger.handlers) == 2
    assert module_2.logger.disabled is False
    assert module_2.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_2.ANY == 'Any'
    assert module_2.Parser.link is True
    assert module_2.Parser.b_level == 1
    assert module_2.Parser.toc is False
    assert f'{type(module_2.Parser.new).__module__}.{type(module_2.Parser.new).__qualname__}' == 'builtins.method'
    module_0._load_module(var_0, var_0, var_0)

def test_case_11():
    var_0 = 'n-FwE_>bJS9T\\)*dpqv'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 6796
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
    var_2 = 'E=OIOa{oN'
    var_3 = module_1.abspath(var_1)
    assert var_3 == '/workspace'
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
    var_4 = {var_0: var_0, var_2: var_3, var_1: var_2}
    var_5 = None
    var_6 = module_0.gen_api(var_4, prefix=var_3, level=var_5)