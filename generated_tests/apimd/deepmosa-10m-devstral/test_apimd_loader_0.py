# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.loader as module_0
import apimd.parser as module_1
import posixpath as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'test_pkg'
    var_1 = True
    var_2 = module_0.loader(var_0, var_0, var_1, var_1, var_1)
    assert var_2 == '**Table of contents:**\n\n\n'
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
    module_0._site_path(var_2)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.gen_api(var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'sys'
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
    var_2 = module_0.walk_packages(var_1, var_1)
    var_3 = 'UfmX>)#'
    var_4 = {var_3: var_3}
    var_5 = False
    module_0.gen_api(var_4, var_1, level=var_5)

def test_case_3():
    var_0 = bool(not '.py' == '.py')

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = ''
    var_1 = True
    module_0.loader(var_0, var_0, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'sys'
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
    var_2 = None
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    assert var_4 == '\n'
    var_5 = {}
    module_0.gen_api(var_5, link=var_2)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'nonexistent_package'
    module_0._site_path(var_0)
    assert var_1 == ''

def test_case_7():
    var_0 = 'sys'
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

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_1.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_1.logger).__module__}.{type(module_1.logger).__qualname__}' == 'logging.RootLogger'
    assert module_1.logger.filters == []
    assert module_1.logger.name == 'root'
    assert module_1.logger.level == 10
    assert module_1.logger.parent is None
    assert module_1.logger.propagate is True
    assert f'{type(module_1.logger.handlers).__module__}.{type(module_1.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_1.logger.handlers) == 2
    assert module_1.logger.disabled is False
    assert module_1.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_1.ANY == 'Any'
    assert module_1.Parser.link is True
    assert module_1.Parser.b_level == 1
    assert module_1.Parser.toc is False
    assert f'{type(module_1.Parser.new).__module__}.{type(module_1.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'nonexistent_module'
    var_2 = 'path/to/nonexistent_module.py'
    module_0._load_module(var_1, var_2, var_0)
    assert var_3 is False

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'nonexistent_package'
    var_1 = None
    var_2 = module_2.dirname(var_0)
    assert var_2 == ''
    assert module_2.curdir == '.'
    assert module_2.pardir == '..'
    assert module_2.extsep == '.'
    assert module_2.sep == '/'
    assert module_2.pathsep == ':'
    assert module_2.defpath == '/bin:/usr/bin'
    assert module_2.altsep is None
    assert module_2.devnull == '/dev/null'
    assert f'{type(module_2.ALLOW_MISSING).__module__}.{type(module_2.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    assert module_2.supports_unicode_filenames is False
    module_0._load_module(var_1, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_1.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_1.logger).__module__}.{type(module_1.logger).__qualname__}' == 'logging.RootLogger'
    assert module_1.logger.filters == []
    assert module_1.logger.name == 'root'
    assert module_1.logger.level == 10
    assert module_1.logger.parent is None
    assert module_1.logger.propagate is True
    assert f'{type(module_1.logger.handlers).__module__}.{type(module_1.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_1.logger.handlers) == 2
    assert module_1.logger.disabled is False
    assert module_1.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_1.ANY == 'Any'
    assert module_1.Parser.link is True
    assert module_1.Parser.b_level == 1
    assert module_1.Parser.toc is False
    assert f'{type(module_1.Parser.new).__module__}.{type(module_1.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'os.path'
    var_2 = 'path/to/os/path.py'
    module_0._load_module(var_1, var_2, var_0)
    assert var_3 is True