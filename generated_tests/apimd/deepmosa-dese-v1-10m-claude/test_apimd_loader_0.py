# Check out: https://github.com/GlowCheese/deepmosa
import posixpath as module_1

import apimd.loader as module_0
import apimd.parser as module_2
import pytest


def test_case_0():
    var_0 = 'r$+(SEoQ(=BLyLOp'
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
    var_0 = ''
    var_1 = {var_0: var_0}
    var_2 = '/tmp/test_docs'
    var_3 = False
    module_0.gen_api(var_1, var_3, prefix=var_2, link=var_3, level=var_3, toc=var_3, dry=var_3)

def test_case_3():
    pass

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = ''
    module_0.loader(var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = ''
    var_1 = False
    module_0.loader(var_0, var_0, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = ''
    var_1 = module_1.abspath(var_0)
    assert var_1 == '/workspace'
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
    var_2 = None
    var_3 = False
    var_4 = module_0.loader(var_1, var_1, var_3, var_3, var_2)
    assert var_4 == '\n'
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 12514
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
    module_0.gen_api(var_2, var_2, prefix=var_1, dry=var_2)

def test_case_7():
    var_0 = 'test_pkg'
    var_1 = module_2.Parser()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is True
    assert var_1.b_level == 1
    assert var_1.toc is False
    assert var_1.level == {}
    assert var_1.doc == {}
    assert var_1.docstring == {}
    assert var_1.imp == {}
    assert var_1.root == {}
    assert var_1.alias == {}
    assert var_1.const == {}
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
    var_2 = 'test_pkg.test_module'
    var_3 = module_0._load_module(var_2, var_0, var_1)
    assert var_3 is False
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
    var_0 = 'test_module.py'
    var_1 = 0
    var_2 = '|q so-uZu !:8(OB'
    var_3 = -2156
    var_4 = {var_2: var_1, var_0: var_3}
    var_5 = None
    var_6 = module_2.Parser(level=var_4, imp=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {'|q so-uZu !:8(OB': 0, 'test_module.py': -2156}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp is None
    assert var_6.root == {}
    assert var_6.alias == {}
    assert var_6.const == {}
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
    var_7 = "$c$o'eVEQ/p=K"
    module_0._load_module(var_5, var_7, var_1)

def test_case_9():
    var_0 = 'nonexistent_package_xyz_12345'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 12514
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

def test_case_10():
    var_0 = 'importlib'
    var_1 = module_0._site_path(var_0)
    assert var_1 == '/usr/local/lib/python3.10'
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 12514
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

def test_case_11():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 12514
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

def test_case_12():
    var_0 = 'Test _load_module successfully loads a module.'
    var_1 = None
    var_2 = '^Pt\x0c{m'
    var_3 = '70CX\no\tF\t6%c\x0c'
    var_4 = 'CKPg)"'
    var_5 = {var_2, var_3, var_4}
    var_6 = '05]Mb!H6'
    var_7 = 'OJ^(2chFn$7J`'
    var_8 = {var_0: var_5, var_2: var_5, var_6: var_5, var_7: var_5}
    var_9 = module_2.Parser(doc=var_1, imp=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc is False
    assert var_9.level == {}
    assert var_9.doc is None
    assert var_9.docstring == {}
    assert var_9.imp == {'Test _load_module successfully loads a module.': {'CKPg)"', '70CX\no\tF\t6%c\x0c', '^Pt\x0c{m'}, '^Pt\x0c{m': {'CKPg)"', '70CX\no\tF\t6%c\x0c', '^Pt\x0c{m'}, '05]Mb!H6': {'CKPg)"', '70CX\no\tF\t6%c\x0c', '^Pt\x0c{m'}, 'OJ^(2chFn$7J`': {'CKPg)"', '70CX\no\tF\t6%c\x0c', '^Pt\x0c{m'}}
    assert var_9.root == {}
    assert var_9.alias == {}
    assert var_9.const == {}
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
    var_10 = 'apimd.loader.spec_from_file_location'
    var_11 = '/nonexistent/path.py'
    var_12 = '}@aUb]{jP18X~<nTR'
    var_13 = module_0._load_module(var_10, var_12, var_11)
    assert var_13 is False
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 12514
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
def test_case_13():
    var_0 = module_2.Parser()
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
    var_1 = 'sys'
    var_2 = '/nonexistent/path.py'
    module_0._load_module(var_1, var_2, var_0)
    assert var_3 is False

def test_case_14():
    var_0 = {}
    var_1 = None
    var_2 = '/tmp/test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_0, var_1, prefix=var_2, link=var_3, level=var_3, toc=var_3, dry=var_3)
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 12514
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

def test_case_15():
    var_0 = 'm!j(Vy7g'
    var_1 = False
    var_2 = {var_0: var_0}
    var_3 = None
    var_4 = '/tmp/test_docs'
    var_5 = False
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_1, level=var_1, toc=var_5, dry=var_1)
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 12514
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

def test_case_16():
    var_0 = 'm!j(Vy7g'
    var_1 = True
    var_2 = {var_0: var_0}
    var_3 = None
    var_4 = '/tmp/test_docs'
    var_5 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_1, level=var_1, toc=var_1, dry=var_1)
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

def test_case_17():
    var_0 = 'm!j(Vy7g'
    var_1 = False
    var_2 = {var_0: var_0}
    var_3 = None
    var_4 = 'Y#GGn-p,wL&'
    var_5 = '/tmp/test_docs'
    var_6 = 'kpI'
    var_7 = module_0.walk_packages(var_6, var_3)
    assert f'{type(module_0.sys_path).__module__}.{type(module_0.sys_path).__qualname__}' == 'builtins.list'
    assert len(module_0.sys_path) == 12514
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
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_5, link=var_8, level=var_8, toc=var_8, dry=var_1)
    var_10 = module_0.walk_packages(var_3, var_4)