# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.parser as module_0
import ast as module_1
import inspect as module_2
import dataclasses as module_3

def test_case_0():
    var_0 = 'cTaR|U;:w:FlDp9}42'
    var_1 = module_0.is_magic(var_0)
    assert var_1 is False
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_1():
    var_0 = 'q.;]aHlmdH'
    var_1 = module_0.esc_underscore(var_0)
    assert var_1 == 'q.;]aHlmdH'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '{nq'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    module_0.table(*var_0, items=var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.table(items=var_0)

def test_case_4():
    var_0 = 'N|T?l3.NwD,x;XDu`(?'
    var_1 = module_0.const_type(var_0)
    assert var_1 == 'Any'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_5():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = 'GQ'
    var_2 = None
    var_3 = 898
    var_4 = module_0.Parser(var_2, var_3, imp=var_3, const=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is None
    assert var_4.b_level == 898
    assert var_4.toc is False
    assert var_4.level == {}
    assert var_4.doc == {}
    assert var_4.docstring == {}
    assert var_4.imp == 898
    assert var_4.root == {}
    assert var_4.alias == {}
    assert var_4.const is None
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4.parse(var_0, var_1)

def test_case_7():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n': 0, '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n.Color': 0}
    assert var_0.doc == {'\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| RED |\n| GREEN |\n| BLUE |\n\n'}
    assert var_0.imp == {'\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n': {*()}}
    assert var_0.root == {'\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n': '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n', '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n.Color': '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'}
    assert var_0.alias == {'\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n.Enum': 'enum.Enum'}

def test_case_8():
    var_0 = "'l0+zz:fx2#KXK7?t]\x0c"
    var_1 = module_0.parent(var_0)
    assert var_1 == "'l0+zz:fx2#KXK7?t]\x0c"
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_9():
    var_0 = None
    var_1 = module_0.Resolver(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Resolver'
    assert var_1.root is None
    assert var_1.alias is None
    assert var_1.self_ty == ''
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_10():
    var_0 = 'Z-Me1c~=R'
    var_1 = module_0.code(var_0)
    assert var_1 == '`Z-Me1c~=R`'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_11():
    var_0 = ''
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'vz:Qk'
    var_2 = module_0.code(var_1)
    assert var_2 == '`vz:Qk`'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'`vz:Qk`': 0}
    assert var_0.doc == {'`vz:Qk`': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'`vz:Qk`': {*()}}
    assert var_0.root == {'`vz:Qk`': '`vz:Qk`'}
    var_4 = 'test_modul.EmptyClhss'
    var_5 = '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_6 = 'H!'
    var_7 = var_0.parse(var_6, var_1)
    assert var_0.level == {'`vz:Qk`': 0, 'H!': 0}
    assert var_0.doc == {'`vz:Qk`': '## Module `{}`\n<a id="{}"></a>\n\n', 'H!': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'`vz:Qk`': {*()}, 'H!': {*()}}
    assert var_0.root == {'`vz:Qk`': '`vz:Qk`', 'H!': 'H!'}
    var_8 = var_0.compile()
    assert var_8 == '\n'
    var_9 = var_0.parse(var_1, var_5)
    assert var_0.level == {'`vz:Qk`': 0, 'H!': 0, 'vz:Qk': 0, 'vz:Qk.Color': 0}
    assert var_0.doc == {'`vz:Qk`': '## Module `{}`\n<a id="{}"></a>\n\n', 'H!': '## Module `{}`\n<a id="{}"></a>\n\n', 'vz:Qk': '## Module `{}`\n<a id="{}"></a>\n\n', 'vz:Qk.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| RED |\n| GREEN |\n| BLUE |\n\n'}
    assert var_0.imp == {'`vz:Qk`': {*()}, 'H!': {*()}, 'vz:Qk': {*()}}
    assert var_0.root == {'`vz:Qk`': '`vz:Qk`', 'H!': 'H!', 'vz:Qk': 'vz:Qk', 'vz:Qk.Color': 'vz:Qk'}
    assert var_0.alias == {'vz:Qk.Enum': 'enum.Enum'}
    var_10 = var_0.compile()
    assert var_10 == '## Module `vz:Qk`\n<a id="vz:qk"></a>\n\n### class Color\n\n*Full name:* `vz:Qk.Color`\n<a id="vz:qk-color"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| RED |\n| GREEN |\n| BLUE |\n'
    var_11 = True
    var_12 = -30
    var_13 = {var_8: var_12, var_1: var_11}
    var_14 = '<B)f-op\x0bsI%l(0Gq/<h'
    var_15 = ''
    var_16 = '0:l'
    var_17 = {var_2: var_4, var_14: var_5, var_14: var_15, var_1: var_16}
    var_18 = module_0.Parser(toc=var_3, level=var_13, doc=var_17, root=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'apimd.parser.Parser'
    assert var_18.link is True
    assert var_18.b_level == 1
    assert var_18.toc is None
    assert var_18.level == {'\n': -30, 'vz:Qk': True}
    assert var_18.doc == {'`vz:Qk`': 'test_modul.EmptyClhss', '<B)f-op\x0bsI%l(0Gq/<h': '', 'vz:Qk': '0:l'}
    assert var_18.docstring == {}
    assert var_18.imp == {}
    assert var_18.root == {'`vz:Qk`': 'test_modul.EmptyClhss', '<B)f-op\x0bsI%l(0Gq/<h': '', 'vz:Qk': '0:l'}
    assert var_18.alias == {}
    assert var_18.const == {}
    var_19 = module_0.doctest(var_6)
    assert var_19 == 'H!'
    var_20 = [var_3, var_9]
    var_21 = module_1.Dict(*var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'ast.Dict'
    assert var_21.keys is None
    assert var_21.values is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_22 = module_0.code(var_15)
    assert var_22 == ' '
    var_23 = module_0.const_type(var_9)
    assert var_23 == 'Any'
    var_0.load_docstring(var_9, var_3)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.Parser(doc=var_0, docstring=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is True
    assert var_1.b_level == 1
    assert var_1.toc is False
    assert var_1.level == {}
    assert var_1.doc is None
    assert var_1.docstring is None
    assert var_1.imp == {}
    assert var_1.root == {}
    assert var_1.alias == {}
    assert var_1.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1.is_public(var_0)

def test_case_14():
    var_0 = "'l0+zz:fx2#KXK7?t]\x0c"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "'l0+zz:fx2#KXK7?t]"
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_15():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': 1, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': 1, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': 1}
    assert var_0.doc == {'\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n| `y` | `str` |\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n'}

def test_case_16():
    var_0 = '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n'
    var_1 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_2 = var_1.parse(var_0, var_0)
    assert var_1.level == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n': 0, '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n.MyClass': 0}
    assert var_1.doc == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `CONSTANT` | `int` |\n| `name` | `str` |\n\n'}
    assert var_1.imp == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n': {*()}}
    assert var_1.root == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n': '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n', '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n.MyClass': '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n'}

def test_case_17():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_1.Set(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Set'
    assert var_2.elts is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = module_0.const_type(var_2)
    assert var_3 == 'set'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_18():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': 1, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': 1, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': 1}
    assert var_0.doc == {'\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n| `y` | `str` |\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n'}
    var_3 = var_0.load_docstring(var_1, var_2)

def test_case_19():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'qq=D'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'qq=D': 0}
    assert var_0.doc == {'qq=D': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'qq=D': {*()}}
    assert var_0.root == {'qq=D': 'qq=D'}
    assert var_0.alias == {'qq=D.qq': 'D'}

def test_case_20():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'v:Qk'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'v:Qk': 0}
    assert var_0.doc == {'v:Qk': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'v:Qk': {*()}}
    assert var_0.root == {'v:Qk': 'v:Qk'}

def test_case_21():
    var_0 = 'Z-Me91&~=R'
    var_1 = module_0.code(var_0)
    assert var_1 == '<code>Z-Me91&~=R</code>'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_22():
    var_0 = ',EyqB'
    var_1 = None
    var_2 = module_0.Parser(doc=var_1, docstring=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc is None
    assert var_2.docstring == ',EyqB'
    assert var_2.imp == {}
    assert var_2.root == {}
    assert var_2.alias == {}
    assert var_2.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_3 = []
    var_4 = var_2.class_api(var_0, var_0, var_3, var_0)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_1.Constant()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'ast.Constant'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    module_0.const_type(var_0)

def test_case_24():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n': 0, '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n.MyClass': 0}
    assert var_0.doc == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `CONSTANT` | `int` |\n| `name` | `str` |\n\n'}
    assert var_0.imp == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n': '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n', '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n.MyClass': '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n`\n<a id="\nclass myclass:\n    constant = 42\n    name = "test"\n    _hidden = tru\n"></a>\n\n### class MyClass\n\n*Full name:* `\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = Tru\n.MyClass`\n<a id="\nclass myclass:\n    constant = 42\n    name = "test"\n    _hidden = tru\n-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `CONSTANT` | `int` |\n| `name` | `str` |\n'

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = ',EyqB'
    var_1 = None
    var_2 = module_0.is_public_family(var_0)
    assert var_2 is True
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_3 = module_0.Parser(toc=var_2, const=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is True
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const is None
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4 = module_0.doctest(var_0)
    assert var_4 == ',EyqB'
    var_5 = module_0.Parser(doc=var_1, docstring=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is True
    assert var_5.b_level == 1
    assert var_5.toc is False
    assert var_5.level == {}
    assert var_5.doc is None
    assert var_5.docstring == ',EyqB'
    assert var_5.imp == {}
    assert var_5.root == {}
    assert var_5.alias == {}
    assert var_5.const == {}
    var_6 = var_5.globals(var_0, var_0)
    var_7 = []
    var_8 = var_5.class_api(var_0, var_0, var_7, var_0)
    var_9 = 'v4_xSUvJ\rb7\n%4r"$^'
    var_5.api(var_9, var_8)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = True
    var_1 = '?'
    var_2 = {var_1: var_1, var_1: var_1, var_1: var_1}
    var_3 = module_0.Parser(var_0, doc=var_1, alias=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == '?'
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {'?': '?'}
    assert var_3.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_3.compile()

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = True
    var_1 = 'wlUC'
    var_2 = '?'
    var_3 = {var_1: var_2, var_2: var_2, var_1: var_2}
    var_4 = module_0.Parser(var_0, doc=var_1, alias=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level == 1
    assert var_4.toc is False
    assert var_4.level == {}
    assert var_4.doc == 'wlUC'
    assert var_4.docstring == {}
    assert var_4.imp == {}
    assert var_4.root == {}
    assert var_4.alias == {'wlUC': '?', '?': '?'}
    assert var_4.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4.compile()

def test_case_28():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n': 1, '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass': 1, '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': 1}
    assert var_0.doc == {'\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `3` |\n| `y` | `sr` |\n\n', '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n': '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass': '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n'}
    var_3 = '['
    var_4 = var_0.load_docstring(var_3, var_2)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = '{nq'
    var_1 = None
    var_2 = module_0.Parser(doc=var_1, docstring=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc is None
    assert var_2.docstring is None
    assert var_2.imp == {}
    assert var_2.root == {}
    assert var_2.alias == {}
    assert var_2.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_3 = [var_1]
    var_4 = module_1.Attribute(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Attribute'
    assert var_4.value is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_2.resolve(var_0, var_4, var_0)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'vz:Qk'
    var_2 = module_0.code(var_1)
    assert var_2 == '`vz:Qk`'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'`vz:Qk`': 0}
    assert var_0.doc == {'`vz:Qk`': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'`vz:Qk`': {*()}}
    assert var_0.root == {'`vz:Qk`': '`vz:Qk`'}
    var_4 = '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_5 = 'H!'
    var_6 = var_0.parse(var_5, var_1)
    assert var_0.level == {'`vz:Qk`': 0, 'H!': 0}
    assert var_0.doc == {'`vz:Qk`': '## Module `{}`\n<a id="{}"></a>\n\n', 'H!': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'`vz:Qk`': {*()}, 'H!': {*()}}
    assert var_0.root == {'`vz:Qk`': '`vz:Qk`', 'H!': 'H!'}
    var_7 = var_0.parse(var_1, var_4)
    assert var_0.level == {'`vz:Qk`': 0, 'H!': 0, 'vz:Qk': 0, 'vz:Qk.Color': 0}
    assert var_0.doc == {'`vz:Qk`': '## Module `{}`\n<a id="{}"></a>\n\n', 'H!': '## Module `{}`\n<a id="{}"></a>\n\n', 'vz:Qk': '## Module `{}`\n<a id="{}"></a>\n\n', 'vz:Qk.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| RED |\n| GREEN |\n| BLUE |\n\n'}
    assert var_0.imp == {'`vz:Qk`': {*()}, 'H!': {*()}, 'vz:Qk': {*()}}
    assert var_0.root == {'`vz:Qk`': '`vz:Qk`', 'H!': 'H!', 'vz:Qk': 'vz:Qk', 'vz:Qk.Color': 'vz:Qk'}
    assert var_0.alias == {'vz:Qk.Enum': 'enum.Enum'}
    var_8 = var_0.__repr__()
    assert var_8 == 'Parser(link=True, b_level=1, toc=False, level={\'`vz:Qk`\': 0, \'H!\': 0, \'vz:Qk\': 0, \'vz:Qk.Color\': 0}, doc={\'`vz:Qk`\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'H!\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'vz:Qk\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'vz:Qk.Color\': \'### class Color\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Bases |\\n|:-----:|\\n| `enum.Enum` |\\n\\n| Enums |\\n|:-----:|\\n| RED |\\n| GREEN |\\n| BLUE |\\n\\n\'}, docstring={}, imp={\'`vz:Qk`\': set(), \'H!\': set(), \'vz:Qk\': set()}, root={\'`vz:Qk`\': \'`vz:Qk`\', \'H!\': \'H!\', \'vz:Qk\': \'vz:Qk\', \'vz:Qk.Color\': \'vz:Qk\'}, alias={\'vz:Qk.Enum\': \'enum.Enum\'}, const={})'
    var_9 = module_0.doctest(var_5)
    assert var_9 == 'H!'
    var_10 = module_0.const_type(var_9)
    assert var_10 == 'Any'
    var_11 = [var_3, var_7]
    var_12 = module_1.Dict(*var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Dict'
    assert var_12.keys is None
    assert var_12.values is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_13 = module_0.const_type(var_12)
    assert var_13 == 'dict'
    var_0.load_docstring(var_7, var_3)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = '/[Uel;C'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_2 = module_0.Parser(doc=var_1, docstring=var_1, imp=var_1, alias=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc is True
    assert var_2.docstring is True
    assert var_2.imp is True
    assert var_2.root == {}
    assert var_2.alias is True
    assert var_2.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_3 = []
    var_4 = var_2.class_api(var_0, var_0, var_3, var_0)
    var_5 = '|JrCh*a6\x0bn\\sT'
    var_6 = module_1.Try()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Try'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = [var_6, var_6, var_6, var_6, var_6]
    var_2.class_api(var_5, var_5, var_3, var_7)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = '{nq'
    var_1 = None
    var_2 = module_0.is_public_family(var_0)
    assert var_2 is True
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_3 = module_0.doctest(var_0)
    assert var_3 == '{nq'
    var_4 = module_0.Parser(doc=var_1, docstring=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level == 1
    assert var_4.toc is False
    assert var_4.level == {}
    assert var_4.doc is None
    assert var_4.docstring == '{nq'
    assert var_4.imp == {}
    assert var_4.root == {}
    assert var_4.alias == {}
    assert var_4.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_5 = var_4.globals(var_0, var_0)
    var_6 = []
    var_7 = module_1.If()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.If'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = [var_7]
    var_4.class_api(var_1, var_1, var_6, var_8)

def test_case_33():
    var_0 = 'gq(0?Fd~bavZ'
    var_1 = module_0.is_public_family(var_0)
    assert var_1 is True
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_2 = module_2.getdoc(var_1)
    assert var_2 == 'bool(x) -> bool\n\nReturns True when the argument x is true, False otherwise.\nThe builtins True and False are the only two instances of the class bool.\nThe class bool is a subclass of the class int, and cannot be subclassed.'
    assert f'{type(module_2.mod_dict).__module__}.{type(module_2.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_2.mod_dict) == 168
    assert module_2.k == 512
    assert module_2.v == 'ASYNC_GENERATOR'
    assert module_2.CO_OPTIMIZED == 1
    assert module_2.CO_NEWLOCALS == 2
    assert module_2.CO_VARARGS == 4
    assert module_2.CO_VARKEYWORDS == 8
    assert module_2.CO_NESTED == 16
    assert module_2.CO_GENERATOR == 32
    assert module_2.CO_NOFREE == 64
    assert module_2.CO_COROUTINE == 128
    assert module_2.CO_ITERABLE_COROUTINE == 256
    assert module_2.CO_ASYNC_GENERATOR == 512
    assert module_2.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_2.modulesbyfile == {}
    assert module_2.GEN_CREATED == 'GEN_CREATED'
    assert module_2.GEN_RUNNING == 'GEN_RUNNING'
    assert module_2.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_2.GEN_CLOSED == 'GEN_CLOSED'
    assert module_2.CORO_CREATED == 'CORO_CREATED'
    assert module_2.CORO_RUNNING == 'CORO_RUNNING'
    assert module_2.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_2.CORO_CLOSED == 'CORO_CLOSED'
    var_3 = [var_2]
    var_4 = module_1.Set(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Set'
    assert var_4.elts == 'bool(x) -> bool\n\nReturns True when the argument x is true, False otherwise.\nThe builtins True and False are the only two instances of the class bool.\nThe class bool is a subclass of the class int, and cannot be subclassed.'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = module_0.const_type(var_4)
    assert var_5 == 'set'

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = True
    var_2 = module_0.Parser(toc=var_1, docstring=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is True
    assert var_2.level == {}
    assert var_2.doc == {}
    assert var_2.docstring is None
    assert var_2.imp == {}
    assert var_2.root == {}
    assert var_2.alias == {}
    assert var_2.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_3 = var_2.compile()
    assert var_3 == '**Table of contents:**\n\n\n'
    var_4 = module_0.walk_body(var_0)
    var_5 = 'f\n'
    var_6 = 'CmcZtb['
    var_7 = 'i+UE9'
    var_8 = var_2.parse(var_7, var_5)
    assert var_2.level == {'i+UE9': 0}
    assert var_2.doc == {'i+UE9': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_2.imp == {'i+UE9': {*()}}
    assert var_2.root == {'i+UE9': 'i+UE9'}
    var_9 = '\n.]aaFjvf'
    var_10 = {var_6: var_9}
    var_11 = module_0.Parser(var_0, alias=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'apimd.parser.Parser'
    assert var_11.link is None
    assert var_11.b_level == 1
    assert var_11.toc is False
    assert var_11.level == {}
    assert var_11.doc == {}
    assert var_11.docstring == {}
    assert var_11.imp == {}
    assert var_11.root == {}
    assert var_11.alias == {'CmcZtb[': '\n.]aaFjvf'}
    assert var_11.const == {}
    var_2.is_public(var_5)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = "8!a':<ZL;[_\x0cY&U"
    var_1 = module_0.Parser(doc=var_0, docstring=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is True
    assert var_1.b_level == 1
    assert var_1.toc is False
    assert var_1.level == {}
    assert var_1.doc == "8!a':<ZL;[_\x0cY&U"
    assert var_1.docstring == "8!a':<ZL;[_\x0cY&U"
    assert var_1.imp == {}
    assert var_1.root == {}
    assert var_1.alias == {}
    assert var_1.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_2 = 'T\r2uqx'
    var_3 = []
    var_4 = var_1.class_api(var_2, var_2, var_3, var_2)
    var_5 = module_0.const_type(var_4)
    assert var_5 == 'Any'
    var_6 = var_1.__repr__()
    assert var_6 == 'Parser(link=True, b_level=1, toc=False, level={}, doc="8!a\':<ZL;[_\\x0cY&U", docstring="8!a\':<ZL;[_\\x0cY&U", imp={}, root={}, alias={}, const={})'
    var_7 = module_1.ImportFrom()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.ImportFrom'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_8 = ''
    var_9 = module_0.Resolver(var_8, var_6, var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Resolver'
    assert var_9.root == ''
    assert var_9.alias == 'Parser(link=True, b_level=1, toc=False, level={}, doc="8!a\':<ZL;[_\\x0cY&U", docstring="8!a\':<ZL;[_\\x0cY&U", imp={}, root={}, alias={}, const={})'
    assert var_9.self_ty == 'Parser(link=True, b_level=1, toc=False, level={}, doc="8!a\':<ZL;[_\\x0cY&U", docstring="8!a\':<ZL;[_\\x0cY&U", imp={}, root={}, alias={}, const={})'
    var_10 = var_1.imports(var_4, var_7)
    var_9.visit_Subscript(var_9)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'test_module'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'test_module': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module'}
    var_3 = module_0.Parser()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const == {}
    var_4 = module_0.Parser()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level == 1
    assert var_4.toc is False
    assert var_4.level == {}
    assert var_4.doc == {}
    assert var_4.docstring == {}
    assert var_4.imp == {}
    assert var_4.root == {}
    assert var_4.alias == {}
    assert var_4.const == {}
    var_5 = '\nclass MyClass:\n    x: int\n    y: str\n    _private: float\n'
    var_6 = var_4.parse(var_1, var_5)
    assert var_4.level == {'test_module': 0, 'test_module.MyClass': 0}
    assert var_4.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n| `y` | `str` |\n\n'}
    assert var_4.imp == {'test_module': {*()}}
    assert var_4.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module'}
    var_3.visit_Constant(var_3)

def test_case_37():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'test_module'
    var_2 = 'class EmptyClass:\n    pass'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module': 0, 'test_module.EmptyClass': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.EmptyClass': '### class EmptyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.EmptyClass': 'test_module'}
    var_4 = module_0.Parser()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level == 1
    assert var_4.toc is False
    assert var_4.level == {}
    assert var_4.doc == {}
    assert var_4.docstring == {}
    assert var_4.imp == {}
    assert var_4.root == {}
    assert var_4.alias == {}
    assert var_4.const == {}
    var_5 = 'class ChildClass(Base1, Base2):\n    pass'
    var_6 = var_4.parse(var_1, var_5)
    assert var_4.level == {'test_module': 0, 'test_module.ChildClass': 0}
    assert var_4.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.ChildClass': '### class ChildClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `Base1` |\n| `Base2` |\n\n'}
    assert var_4.imp == {'test_module': {*()}}
    assert var_4.root == {'test_module': 'test_module', 'test_module.ChildClass': 'test_module'}
    var_7 = module_0.Parser()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is True
    assert var_7.b_level == 1
    assert var_7.toc is False
    assert var_7.level == {}
    assert var_7.doc == {}
    assert var_7.docstring == {}
    assert var_7.imp == {}
    assert var_7.root == {}
    assert var_7.alias == {}
    assert var_7.const == {}
    var_8 = '\nclass MyClass:\n    x: int\n    y: str\n    _private: float\n'
    var_9 = var_7.parse(var_1, var_8)
    assert var_7.level == {'test_module': 0, 'test_module.MyClass': 0}
    assert var_7.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n| `y` | `str` |\n\n'}
    assert var_7.imp == {'test_module': {*()}}
    assert var_7.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module'}
    var_10 = module_0.Parser()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 1
    assert var_10.toc is False
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    var_11 = '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n'
    var_12 = var_10.parse(var_1, var_11)
    assert var_10.level == {'test_module': 0, 'test_module.MyClass': 0}
    assert var_10.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `CONSTANT` | `int` |\n| `name` | `str` |\n\n'}
    assert var_10.imp == {'test_module': {*()}}
    assert var_10.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module'}
    var_13 = module_0.Parser()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'apimd.parser.Parser'
    assert var_13.link is True
    assert var_13.b_level == 1
    assert var_13.toc is False
    assert var_13.level == {}
    assert var_13.doc == {}
    assert var_13.docstring == {}
    assert var_13.imp == {}
    assert var_13.root == {}
    assert var_13.alias == {}
    assert var_13.const == {}
    var_14 = '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_15 = var_13.parse(var_1, var_14)
    assert var_13.level == {'test_module': 0, 'test_module.Color': 0}
    assert var_13.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| RED |\n| GREEN |\n| BLUE |\n\n'}
    assert var_13.imp == {'test_module': {*()}}
    assert var_13.root == {'test_module': 'test_module', 'test_module.Color': 'test_module'}
    assert var_13.alias == {'test_module.Enum': 'enum.Enum'}
    var_16 = module_0.Parser()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'apimd.parser.Parser'
    assert var_16.link is True
    assert var_16.b_level == 1
    assert var_16.toc is False
    assert var_16.level == {}
    assert var_16.doc == {}
    assert var_16.docstring == {}
    assert var_16.imp == {}
    assert var_16.root == {}
    assert var_16.alias == {}
    assert var_16.const == {}
    var_17 = '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n'
    var_18 = var_16.parse(var_1, var_17)
    assert var_16.level == {'test_module': 0, 'test_module.MyClass': 0, 'test_module.MyClass.__init__': 0}
    assert var_16.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n| `y` | `str` |\n\n', 'test_module.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_16.imp == {'test_module': {*()}}
    assert var_16.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module', 'test_module.MyClass.__init__': 'test_module'}
    var_19 = module_0.Parser()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'apimd.parser.Parser'
    assert var_19.link is True
    assert var_19.b_level == 1
    assert var_19.toc is False
    assert var_19.level == {}
    assert var_19.doc == {}
    assert var_19.docstring == {}
    assert var_19.imp == {}
    assert var_19.root == {}
    assert var_19.alias == {}
    assert var_19.const == {}
    var_20 = '\nclass MyClass:\n    x = 42  # type: int\n    y = "hello"  # type: str\n'
    var_21 = var_19.parse(var_1, var_20)
    assert var_19.level == {'test_module': 0, 'test_module.MyClass': 0}
    assert var_19.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n| `y` | `str` |\n\n'}
    assert var_19.imp == {'test_module': {*()}}
    assert var_19.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module'}
    var_22 = '\nfrom enum import IntEnum\nclass MyEnum(IntEnum, BaseClass):\n    VALUE1 = 1\n    VALUE2 = 2\n'
    var_23 = var_16.parse(var_1, var_22)
    assert var_16.level == {'test_module': 0, 'test_module.MyClass': 0, 'test_module.MyClass.__init__': 0, 'test_module.MyEnum': 0}
    assert var_16.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n| `y` | `str` |\n\n', 'test_module.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'test_module.MyEnum': '### class MyEnum\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.IntEnum` |\n| `BaseClass` |\n\n| Enums |\n|:-----:|\n| VALUE1 |\n| VALUE2 |\n\n'}
    assert var_16.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module', 'test_module.MyClass.__init__': 'test_module', 'test_module.MyEnum': 'test_module'}
    assert var_16.alias == {'test_module.IntEnum': 'enum.IntEnum'}
    var_24 = module_0.Parser()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'apimd.parser.Parser'
    assert var_24.link is True
    assert var_24.b_level == 1
    assert var_24.toc is False
    assert var_24.level == {}
    assert var_24.doc == {}
    assert var_24.docstring == {}
    assert var_24.imp == {}
    assert var_24.root == {}
    assert var_24.alias == {}
    assert var_24.const == {}
    var_25 = '\nclass PrivateClass:\n    _private1: int\n    _private2: str\n    __very_private = True\n'
    var_26 = var_24.parse(var_1, var_25)
    assert var_24.level == {'test_module': 0, 'test_module.PrivateClass': 0}
    assert var_24.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.PrivateClass': '### class PrivateClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_24.imp == {'test_module': {*()}}
    assert var_24.root == {'test_module': 'test_module', 'test_module.PrivateClass': 'test_module'}

def test_case_38():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_1.Call(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Call'
    assert var_2.func is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = module_0.const_type(var_2)
    assert var_3 == 'Any'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_39():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n': 0, '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n.MyClass': 0}
    assert var_0.doc == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `CONSTANT` | `int` |\n| `name` | `str` |\n\n'}
    assert var_0.imp == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n': '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n', '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n.MyClass': '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n'}
    var_3 = var_0.is_public(var_1)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'test_module'
    var_2 = module_0.Parser()
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
    var_3 = module_0.Parser()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const == {}
    var_4 = var_3.parse(var_1, var_1)
    assert var_3.level == {'test_module': 0}
    assert var_3.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_3.imp == {'test_module': {*()}}
    assert var_3.root == {'test_module': 'test_module'}
    var_5 = [var_4, var_4, var_4, var_4]
    var_6 = module_1.AnnAssign(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.AnnAssign'
    assert var_6.target is None
    assert var_6.annotation is None
    assert var_6.value is None
    assert var_6.simple is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.AnnAssign.value is None
    var_7 = var_2.globals(var_3, var_6)
    var_0.imports(var_7, var_1)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'vz:Qk'
    var_2 = module_0.code(var_1)
    assert var_2 == '`vz:Qk`'
    var_3 = module_0.doctest(var_1)
    assert var_3 == 'vz:Qk'
    var_4 = var_0.parse(var_3, var_1)
    assert var_0.level == {'vz:Qk': 0}
    assert var_0.doc == {'vz:Qk': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'vz:Qk': {*()}}
    assert var_0.root == {'vz:Qk': 'vz:Qk'}
    var_5 = '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_6 = 'H!'
    var_7 = var_0.parse(var_6, var_3)
    assert var_0.level == {'vz:Qk': 0, 'H!': 0}
    assert var_0.doc == {'vz:Qk': '## Module `{}`\n<a id="{}"></a>\n\n', 'H!': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'vz:Qk': {*()}, 'H!': {*()}}
    assert var_0.root == {'vz:Qk': 'vz:Qk', 'H!': 'H!'}
    var_8 = var_0.compile()
    assert var_8 == '\n'
    var_0.parse(var_5, var_6)

def test_case_42():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'class EmptyClass:\n    pass'
    var_2 = var_0.compile()
    assert var_2 == '\n'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'\n': 0, '\n.EmptyClass': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.EmptyClass': '### class EmptyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n', '\n.EmptyClass': '\n'}
    var_4 = 'class ChildClass(Base1, Base2):\n    pass'
    var_5 = var_0.parse(var_4, var_4)
    assert var_0.level == {'\n': 0, '\n.EmptyClass': 0, 'class ChildClass(Base1, Base2):\n    pass': 0, 'class ChildClass(Base1, Base2):\n    pass.ChildClass': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.EmptyClass': '### class EmptyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class ChildClass(Base1, Base2):\n    pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class ChildClass(Base1, Base2):\n    pass.ChildClass': '### class ChildClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `Base1` |\n| `Base2` |\n\n'}
    assert var_0.imp == {'\n': {*()}, 'class ChildClass(Base1, Base2):\n    pass': {*()}}
    assert var_0.root == {'\n': '\n', '\n.EmptyClass': '\n', 'class ChildClass(Base1, Base2):\n    pass': 'class ChildClass(Base1, Base2):\n    pass', 'class ChildClass(Base1, Base2):\n    pass.ChildClass': 'class ChildClass(Base1, Base2):\n    pass'}
    var_6 = ''
    var_7 = module_0.Parser()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is True
    assert var_7.b_level == 1
    assert var_7.toc is False
    assert var_7.level == {}
    assert var_7.doc == {}
    assert var_7.docstring == {}
    assert var_7.imp == {}
    assert var_7.root == {}
    assert var_7.alias == {}
    assert var_7.const == {}
    var_8 = '3[YWh@Hvt'
    var_9 = '\nclass MyClass:\n    x: int\n    y: str\n    _private: float\n'
    var_10 = module_1.Load()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_11 = var_7.load_docstring(var_5, var_10)
    var_12 = var_7.parse(var_2, var_9)
    assert var_7.level == {'\n': 0, '\n.MyClass': 0}
    assert var_7.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n| `y` | `str` |\n\n'}
    assert var_7.imp == {'\n': {*()}}
    assert var_7.root == {'\n': '\n', '\n.MyClass': '\n'}
    var_13 = module_0.Parser()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'apimd.parser.Parser'
    assert var_13.link is True
    assert var_13.b_level == 1
    assert var_13.toc is False
    assert var_13.level == {}
    assert var_13.doc == {}
    assert var_13.docstring == {}
    assert var_13.imp == {}
    assert var_13.root == {}
    assert var_13.alias == {}
    assert var_13.const == {}
    var_14 = '\nclass MyClass:\n    CONSTANT = 42\n    name = "test"\n    _hidden = True\n'
    var_15 = var_13.parse(var_8, var_14)
    assert var_13.level == {'3[YWh@Hvt': 0, '3[YWh@Hvt.MyClass': 0}
    assert var_13.doc == {'3[YWh@Hvt': '## Module `{}`\n<a id="{}"></a>\n\n', '3[YWh@Hvt.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `CONSTANT` | `int` |\n| `name` | `str` |\n\n'}
    assert var_13.imp == {'3[YWh@Hvt': {*()}}
    assert var_13.root == {'3[YWh@Hvt': '3[YWh@Hvt', '3[YWh@Hvt.MyClass': '3[YWh@Hvt'}
    var_16 = '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_17 = var_13.parse(var_6, var_16)
    assert var_13.level == {'3[YWh@Hvt': 0, '3[YWh@Hvt.MyClass': 0, '': 0, 'Color': 0}
    assert var_13.doc == {'3[YWh@Hvt': '## Module `{}`\n<a id="{}"></a>\n\n', '3[YWh@Hvt.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `CONSTANT` | `int` |\n| `name` | `str` |\n\n', '': '## Module `{}`\n<a id="{}"></a>\n\n', 'Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `Enum` |\n\n| Members | Type |\n|:-------:|:----:|\n| `BLUE` | `int` |\n| `GREEN` | `int` |\n| `RED` | `int` |\n\n'}
    assert var_13.imp == {'3[YWh@Hvt': {*()}, '': {*()}}
    assert var_13.root == {'3[YWh@Hvt': '3[YWh@Hvt', '3[YWh@Hvt.MyClass': '3[YWh@Hvt', '': '', 'Color': ''}
    assert var_13.alias == {'Enum': 'enum.Enum'}
    var_18 = module_0.Parser(toc=var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'apimd.parser.Parser'
    assert var_18.link is True
    assert var_18.b_level == 1
    assert f'{type(var_18.toc).__module__}.{type(var_18.toc).__qualname__}' == 'apimd.parser.Parser'
    assert var_18.level == {}
    assert var_18.doc == {}
    assert var_18.docstring == {}
    assert var_18.imp == {}
    assert var_18.root == {}
    assert var_18.alias == {}
    assert var_18.const == {}
    var_19 = '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n'
    var_20 = var_0.parse(var_19, var_19)
    assert var_0.level == {'\n': 0, '\n.EmptyClass': 0, 'class ChildClass(Base1, Base2):\n    pass': 0, 'class ChildClass(Base1, Base2):\n    pass.ChildClass': 0, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': 1, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': 1, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': 1}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.EmptyClass': '### class EmptyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class ChildClass(Base1, Base2):\n    pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class ChildClass(Base1, Base2):\n    pass.ChildClass': '### class ChildClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `Base1` |\n| `Base2` |\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n| `y` | `str` |\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'\n': {*()}, 'class ChildClass(Base1, Base2):\n    pass': {*()}, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': {*()}}
    assert var_0.root == {'\n': '\n', '\n.EmptyClass': '\n', 'class ChildClass(Base1, Base2):\n    pass': 'class ChildClass(Base1, Base2):\n    pass', 'class ChildClass(Base1, Base2):\n    pass.ChildClass': 'class ChildClass(Base1, Base2):\n    pass', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n'}
    var_21 = module_0.const_type(var_20)
    assert var_21 == 'Any'
    var_22 = var_0.globals(var_15, var_9)
    var_23 = '9'
    var_24 = module_0.is_public_family(var_23)
    assert var_24 is True
    var_25 = var_13.is_public(var_6)
    assert var_25 is False

def test_case_43():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'test_module'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'test_module': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module'}
    var_3 = module_0.Parser()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const == {}
    var_4 = module_0.Parser()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level == 1
    assert var_4.toc is False
    assert var_4.level == {}
    assert var_4.doc == {}
    assert var_4.docstring == {}
    assert var_4.imp == {}
    assert var_4.root == {}
    assert var_4.alias == {}
    assert var_4.const == {}
    var_5 = var_4.parse(var_1, var_1)
    assert var_4.level == {'test_module': 0}
    assert var_4.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_4.imp == {'test_module': {*()}}
    assert var_4.root == {'test_module': 'test_module'}
    var_6 = [var_5, var_2, var_5, var_2]
    var_7 = module_1.AnnAssign(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.AnnAssign'
    assert var_7.target is None
    assert var_7.annotation is None
    assert var_7.value is None
    assert var_7.simple is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.AnnAssign.value is None
    var_8 = var_3.globals(var_4, var_7)
    var_9 = None
    var_10 = []
    var_11 = [var_7, var_7, var_7, var_7]
    var_12 = var_0.class_api(var_9, var_9, var_10, var_11)

def test_case_44():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n': 1, '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': 1, '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': 1}
    assert var_0.doc == {'\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `it` |\n| `y` | `str` |\n\n', '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n`\n<a id="\nclass myclass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self-x\n"></a>\n\n### class MyClass\n\n*Full name:* `\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass`\n<a id="\nclass myclass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self-x\n-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `it` |\n| `y` | `str` |\n'

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'class EmptyClass:\n    pass'
    var_2 = var_0.compile()
    assert var_2 == '\n'
    var_3 = module_2.getdoc(var_1)
    assert var_3 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert f'{type(module_2.mod_dict).__module__}.{type(module_2.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_2.mod_dict) == 168
    assert module_2.k == 512
    assert module_2.v == 'ASYNC_GENERATOR'
    assert module_2.CO_OPTIMIZED == 1
    assert module_2.CO_NEWLOCALS == 2
    assert module_2.CO_VARARGS == 4
    assert module_2.CO_VARKEYWORDS == 8
    assert module_2.CO_NESTED == 16
    assert module_2.CO_GENERATOR == 32
    assert module_2.CO_NOFREE == 64
    assert module_2.CO_COROUTINE == 128
    assert module_2.CO_ITERABLE_COROUTINE == 256
    assert module_2.CO_ASYNC_GENERATOR == 512
    assert module_2.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_2.modulesbyfile == {}
    assert module_2.GEN_CREATED == 'GEN_CREATED'
    assert module_2.GEN_RUNNING == 'GEN_RUNNING'
    assert module_2.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_2.GEN_CLOSED == 'GEN_CLOSED'
    assert module_2.CORO_CREATED == 'CORO_CREATED'
    assert module_2.CORO_RUNNING == 'CORO_RUNNING'
    assert module_2.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_2.CORO_CLOSED == 'CORO_CLOSED'
    var_4 = var_0.parse(var_3, var_1)
    assert var_0.level == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": 7, "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": 7}
    assert var_0.doc == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": '## Module `{}`\n<a id="{}"></a>\n\n', "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": '### class EmptyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": {*()}}
    assert var_0.root == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."}
    var_5 = 'test_modu:e.EmptyClass'
    var_6 = module_0.Parser()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp == {}
    assert var_6.root == {}
    assert var_6.alias == {}
    assert var_6.const == {}
    var_7 = module_0.const_type(var_3)
    assert var_7 == 'Any'
    var_8 = module_0.esc_underscore(var_2)
    assert var_8 == '\n'
    var_9 = '3[Y@Hvt'
    var_10 = module_1.Load()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_11 = var_6.load_docstring(var_4, var_10)
    var_12 = var_0.compile()
    assert var_12 == '\n'
    var_13 = module_0.Parser()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'apimd.parser.Parser'
    assert var_13.link is True
    assert var_13.b_level == 1
    assert var_13.toc is False
    assert var_13.level == {}
    assert var_13.doc == {}
    assert var_13.docstring == {}
    assert var_13.imp == {}
    assert var_13.root == {}
    assert var_13.alias == {}
    assert var_13.const == {}
    var_14 = var_13.parse(var_9, var_5)
    assert var_13.level == {'3[Y@Hvt': 0}
    assert var_13.doc == {'3[Y@Hvt': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_13.imp == {'3[Y@Hvt': {*()}}
    assert var_13.root == {'3[Y@Hvt': '3[Y@Hvt'}
    var_15 = module_0.const_type(var_3)
    assert var_15 == 'Any'
    var_16 = var_0.load_docstring(var_1, var_3)
    var_3.imports(var_8, var_16)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'class EmptyClass:\n    pass'
    var_2 = module_0.doctest(var_1)
    assert var_2 == 'class EmptyClass:\n    pass'
    var_3 = module_2.getdoc(var_1)
    assert var_3 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert f'{type(module_2.mod_dict).__module__}.{type(module_2.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_2.mod_dict) == 168
    assert module_2.k == 512
    assert module_2.v == 'ASYNC_GENERATOR'
    assert module_2.CO_OPTIMIZED == 1
    assert module_2.CO_NEWLOCALS == 2
    assert module_2.CO_VARARGS == 4
    assert module_2.CO_VARKEYWORDS == 8
    assert module_2.CO_NESTED == 16
    assert module_2.CO_GENERATOR == 32
    assert module_2.CO_NOFREE == 64
    assert module_2.CO_COROUTINE == 128
    assert module_2.CO_ITERABLE_COROUTINE == 256
    assert module_2.CO_ASYNC_GENERATOR == 512
    assert module_2.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_2.modulesbyfile == {}
    assert module_2.GEN_CREATED == 'GEN_CREATED'
    assert module_2.GEN_RUNNING == 'GEN_RUNNING'
    assert module_2.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_2.GEN_CLOSED == 'GEN_CLOSED'
    assert module_2.CORO_CREATED == 'CORO_CREATED'
    assert module_2.CORO_RUNNING == 'CORO_RUNNING'
    assert module_2.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_2.CORO_CLOSED == 'CORO_CLOSED'
    var_4 = var_0.parse(var_3, var_1)
    assert var_0.level == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": 7, "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": 7}
    assert var_0.doc == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": '## Module `{}`\n<a id="{}"></a>\n\n', "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": '### class EmptyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": {*()}}
    assert var_0.root == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."}
    var_5 = var_3.__repr__()
    assert var_5 == '"str(object=\'\') -> str\\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\\n\\nCreate a new string object from the given object. If encoding or\\nerrors is specified, then the object must expose a data buffer\\nthat will be decoded using the given encoding and error handler.\\nOtherwise, returns the result of object.__str__() (if defined)\\nor repr(object).\\nencoding defaults to sys.getdefaultencoding().\\nerrors defaults to \'strict\'."'
    var_6 = module_0.const_type(var_5)
    assert var_6 == 'Any'
    var_7 = var_0.parse(var_5, var_5)
    assert var_0.level == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": 7, "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": 7, '"str(object=\'\') -> str\\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\\n\\nCreate a new string object from the given object. If encoding or\\nerrors is specified, then the object must expose a data buffer\\nthat will be decoded using the given encoding and error handler.\\nOtherwise, returns the result of object.__str__() (if defined)\\nor repr(object).\\nencoding defaults to sys.getdefaultencoding().\\nerrors defaults to \'strict\'."': 7}
    assert var_0.doc == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": '## Module `{}`\n<a id="{}"></a>\n\n', "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": '### class EmptyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '"str(object=\'\') -> str\\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\\n\\nCreate a new string object from the given object. If encoding or\\nerrors is specified, then the object must expose a data buffer\\nthat will be decoded using the given encoding and error handler.\\nOtherwise, returns the result of object.__str__() (if defined)\\nor repr(object).\\nencoding defaults to sys.getdefaultencoding().\\nerrors defaults to \'strict\'."': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.docstring == {'"str(object=\'\') -> str\\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\\n\\nCreate a new string object from the given object. If encoding or\\nerrors is specified, then the object must expose a data buffer\\nthat will be decoded using the given encoding and error handler.\\nOtherwise, returns the result of object.__str__() (if defined)\\nor repr(object).\\nencoding defaults to sys.getdefaultencoding().\\nerrors defaults to \'strict\'."': "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."}
    assert var_0.imp == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": {*()}, '"str(object=\'\') -> str\\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\\n\\nCreate a new string object from the given object. If encoding or\\nerrors is specified, then the object must expose a data buffer\\nthat will be decoded using the given encoding and error handler.\\nOtherwise, returns the result of object.__str__() (if defined)\\nor repr(object).\\nencoding defaults to sys.getdefaultencoding().\\nerrors defaults to \'strict\'."': {*()}}
    assert var_0.root == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", '"str(object=\'\') -> str\\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\\n\\nCreate a new string object from the given object. If encoding or\\nerrors is specified, then the object must expose a data buffer\\nthat will be decoded using the given encoding and error handler.\\nOtherwise, returns the result of object.__str__() (if defined)\\nor repr(object).\\nencoding defaults to sys.getdefaultencoding().\\nerrors defaults to \'strict\'."': '"str(object=\'\') -> str\\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\\n\\nCreate a new string object from the given object. If encoding or\\nerrors is specified, then the object must expose a data buffer\\nthat will be decoded using the given encoding and error handler.\\nOtherwise, returns the result of object.__str__() (if defined)\\nor repr(object).\\nencoding defaults to sys.getdefaultencoding().\\nerrors defaults to \'strict\'."'}
    var_3.compile()

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'class EmptyClass:\n    pass'
    var_2 = module_0.doctest(var_1)
    assert var_2 == 'class EmptyClass:\n    pass'
    var_3 = module_2.getdoc(var_1)
    assert var_3 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert f'{type(module_2.mod_dict).__module__}.{type(module_2.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_2.mod_dict) == 168
    assert module_2.k == 512
    assert module_2.v == 'ASYNC_GENERATOR'
    assert module_2.CO_OPTIMIZED == 1
    assert module_2.CO_NEWLOCALS == 2
    assert module_2.CO_VARARGS == 4
    assert module_2.CO_VARKEYWORDS == 8
    assert module_2.CO_NESTED == 16
    assert module_2.CO_GENERATOR == 32
    assert module_2.CO_NOFREE == 64
    assert module_2.CO_COROUTINE == 128
    assert module_2.CO_ITERABLE_COROUTINE == 256
    assert module_2.CO_ASYNC_GENERATOR == 512
    assert module_2.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_2.modulesbyfile == {}
    assert module_2.GEN_CREATED == 'GEN_CREATED'
    assert module_2.GEN_RUNNING == 'GEN_RUNNING'
    assert module_2.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_2.GEN_CLOSED == 'GEN_CLOSED'
    assert module_2.CORO_CREATED == 'CORO_CREATED'
    assert module_2.CORO_RUNNING == 'CORO_RUNNING'
    assert module_2.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_2.CORO_CLOSED == 'CORO_CLOSED'
    var_4 = var_0.parse(var_3, var_1)
    assert var_0.level == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": 7, "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": 7}
    assert var_0.doc == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": '## Module `{}`\n<a id="{}"></a>\n\n', "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": '### class EmptyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": {*()}}
    assert var_0.root == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."}
    var_5 = 'test_module.EmptyClass'
    var_6 = module_0.Parser()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp == {}
    assert var_6.root == {}
    assert var_6.alias == {}
    assert var_6.const == {}
    var_7 = var_6.__repr__()
    assert var_7 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_8 = module_0.const_type(var_7)
    assert var_8 == 'Any'
    var_9 = module_0.esc_underscore(var_2)
    assert var_9 == 'class EmptyClass:\n    pass'
    var_10 = 'ZS\x0bjc~%$lRp2\r%$\npF'
    var_11 = module_0.is_public_family(var_5)
    assert var_11 is True
    var_12 = module_1.Load()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_13 = var_6.load_docstring(var_4, var_12)
    var_14 = var_6.compile()
    assert var_14 == '\n'
    var_15 = module_0.Parser()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'apimd.parser.Parser'
    assert var_15.link is True
    assert var_15.b_level == 1
    assert var_15.toc is False
    assert var_15.level == {}
    assert var_15.doc == {}
    assert var_15.docstring == {}
    assert var_15.imp == {}
    assert var_15.root == {}
    assert var_15.alias == {}
    assert var_15.const == {}
    var_16 = var_15.parse(var_10, var_5)
    assert var_15.level == {'ZS\x0bjc~%$lRp2\r%$\npF': 0}
    assert var_15.doc == {'ZS\x0bjc~%$lRp2\r%$\npF': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_15.imp == {'ZS\x0bjc~%$lRp2\r%$\npF': {*()}}
    assert var_15.root == {'ZS\x0bjc~%$lRp2\r%$\npF': 'ZS\x0bjc~%$lRp2\r%$\npF'}
    var_17 = 'test_module.MyClass'
    var_18 = '\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BL:E = 3\n'
    var_19 = var_15.parse(var_8, var_18)
    assert var_15.level == {'ZS\x0bjc~%$lRp2\r%$\npF': 0, 'Any': 0, 'Any.Color': 0}
    assert var_15.doc == {'ZS\x0bjc~%$lRp2\r%$\npF': '## Module `{}`\n<a id="{}"></a>\n\n', 'Any': '## Module `{}`\n<a id="{}"></a>\n\n', 'Any.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| RED |\n| GREEN |\n| BL |\n\n'}
    assert var_15.imp == {'ZS\x0bjc~%$lRp2\r%$\npF': {*()}, 'Any': {*()}}
    assert var_15.root == {'ZS\x0bjc~%$lRp2\r%$\npF': 'ZS\x0bjc~%$lRp2\r%$\npF', 'Any': 'Any', 'Any.Color': 'Any'}
    assert var_15.alias == {'Any.Enum': 'enum.Enum'}
    var_20 = var_15.compile()
    assert var_20 == '## Module `Any`\n<a id="any"></a>\n\n### class Color\n\n*Full name:* `Any.Color`\n<a id="any-color"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| RED |\n| GREEN |\n| BL |\n'
    var_21 = module_0.Parser(toc=var_0)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'apimd.parser.Parser'
    assert var_21.link is True
    assert var_21.b_level == 1
    assert f'{type(var_21.toc).__module__}.{type(var_21.toc).__qualname__}' == 'apimd.parser.Parser'
    assert var_21.level == {}
    assert var_21.doc == {}
    assert var_21.docstring == {}
    assert var_21.imp == {}
    assert var_21.root == {}
    assert var_21.alias == {}
    assert var_21.const == {}
    var_22 = '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n'
    var_23 = var_0.parse(var_22, var_22)
    assert var_0.level == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": 7, "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": 7, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': 1, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': 1, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': 1}
    assert var_0.doc == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": '## Module `{}`\n<a id="{}"></a>\n\n', "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": '### class EmptyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n| `y` | `str` |\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": {*()}, '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': {*()}}
    assert var_0.root == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..EmptyClass": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '\nclass MyClass:\n    x: int\n    y: str\n    \n    def __init__(self):\n        del self.x\n'}
    var_24 = module_0.const_type(var_23)
    assert var_24 == 'Any'
    var_25 = module_3.dataclass(match_args=var_23)
    assert f'{type(module_3.MISSING).__module__}.{type(module_3.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_3.KW_ONLY).__module__}.{type(module_3.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_26 = module_0.const_type(var_25)
    assert var_26 == 'Any'
    var_27 = var_0.load_docstring(var_17, var_25)
    var_6.imports(var_5, var_7)

def test_case_48():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = None
    var_2 = var_0.__eq__(var_0)
    assert var_2 is True
    var_3 = var_0.compile()
    assert var_3 == '\n'
    var_4 = '|Q*Qzi\x0c'
    var_5 = var_0.func_ann(var_4, var_2, has_self=var_1, cls_method=var_3)
    var_6 = module_0.code(var_3)
    assert var_6 == '`\n`'
    var_7 = None
    var_8 = var_0.compile()
    assert var_8 == '\n'
    var_9 = '<B)f-op\x0bsI%l(0Gq/<h'
    var_10 = None
    var_11 = module_0.Resolver(var_9, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'apimd.parser.Resolver'
    assert var_11.root == '<B)f-op\x0bsI%l(0Gq/<h'
    assert var_11.alias is None
    assert var_11.self_ty == ''
    var_12 = [var_6, var_7]
    var_13 = module_1.Assign(*var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Assign'
    assert var_13.targets == '`\n`'
    assert var_13.value is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Assign.type_comment is None
    var_14 = var_0.globals(var_7, var_13)
    var_15 = var_0.compile()
    assert var_15 == '\n'

def test_case_49():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'test_module'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'test_module': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module'}
    var_3 = True
    var_4 = 2
    var_5 = False
    var_6 = module_0.Parser(var_3, var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level == 2
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp == {}
    assert var_6.root == {}
    assert var_6.alias == {}
    assert var_6.const == {}
    var_7 = var_6.__repr__()
    var_8 = module_0.Parser()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is True
    assert var_8.b_level == 1
    assert var_8.toc is False
    assert var_8.level == {}
    assert var_8.doc == {}
    assert var_8.docstring == {}
    assert var_8.imp == {}
    assert var_8.root == {}
    assert var_8.alias == {}
    assert var_8.const == {}
    var_9 = 'module_with_decorators'
    var_10 = '\n@property\ndef computed_property():\n    """Property docstring."""\n    return 42\n\n@classmethod\n@staticmethod\ndef multi_decorated():\n    pass\n'
    var_11 = var_8.parse(var_9, var_10)
    assert var_8.level == {'module_with_decorators': 0, 'module_with_decorators.computed_property': 0, 'module_with_decorators.multi_decorated': 0}
    assert var_8.doc == {'module_with_decorators': '## Module `{}`\n<a id="{}"></a>\n\n', 'module_with_decorators.computed_property': '### computed_property()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@property` |\n\n| return |\n|:------:|\n| `Any` |\n\n', 'module_with_decorators.multi_decorated': '### multi_decorated()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_8.docstring == {'module_with_decorators.computed_property': 'Property docstring.'}
    assert var_8.imp == {'module_with_decorators': {*()}}
    assert var_8.root == {'module_with_decorators': 'module_with_decorators', 'module_with_decorators.computed_property': 'module_with_decorators', 'module_with_decorators.multi_decorated': 'module_with_decorators'}

def test_case_50():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'test_module'
    var_2 = '\ndef simple_function():\n    """Simple function docstring."""\n    pass\n\nasync def async_function():\n    """Async function docstring."""\n    pass\n\nclass TestClass:\n    """Test class docstring."""\n    def method(self):\n        pass\n    \n    @classmethod\n    def class_method(cls):\n        pass\n    \n    @staticmethod\n    def static_method():\n        pass\n    \n    class InnerClass:\n        """Inner class docstring."""\n        pass\n'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module': 0, 'test_module.simple_function': 0, 'test_module.async_function': 0, 'test_module.TestClass': 0, 'test_module.TestClass.method': 0, 'test_module.TestClass.class_method': 0, 'test_module.TestClass.static_method': 0, 'test_module.TestClass.InnerClass': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.simple_function': '### simple_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module.async_function': '### async async_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.TestClass.method': '#### TestClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'test_module.TestClass.class_method': '#### TestClass.class_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | return |\n|:---:|:------:|\n| `type[Self]` | `Any` |\n\n', 'test_module.TestClass.static_method': '#### TestClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module.TestClass.InnerClass': '#### class TestClass.InnerClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.docstring == {'test_module.simple_function': 'Simple function docstring.', 'test_module.async_function': 'Async function docstring.', 'test_module.TestClass': 'Test class docstring.', 'test_module.TestClass.InnerClass': 'Inner class docstring.'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.simple_function': 'test_module', 'test_module.async_function': 'test_module', 'test_module.TestClass': 'test_module', 'test_module.TestClass.method': 'test_module', 'test_module.TestClass.class_method': 'test_module', 'test_module.TestClass.static_method': 'test_module', 'test_module.TestClass.InnerClass': 'test_module'}
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = module_0.Parser(var_4, var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is True
    assert var_7.b_level == 2
    assert var_7.toc is False
    assert var_7.level == {}
    assert var_7.doc == {}
    assert var_7.docstring == {}
    assert var_7.imp == {}
    assert var_7.root == {}
    assert var_7.alias == {}
    assert var_7.const == {}
    var_8 = 'another_module'
    var_9 = '\nclass BaseClass:\n    pass\n\nclass DerivedClass(BaseClass):\n    """Derived class with base."""\n    pass\n'
    var_10 = var_7.parse(var_8, var_9)
    assert var_7.level == {'another_module': 0, 'another_module.BaseClass': 0, 'another_module.DerivedClass': 0}
    assert var_7.doc == {'another_module': '### Module `{}`\n<a id="{}"></a>\n\n', 'another_module.BaseClass': '#### class BaseClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'another_module.DerivedClass': '#### class DerivedClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `BaseClass` |\n\n'}
    assert var_7.docstring == {'another_module.DerivedClass': 'Derived class with base.'}
    assert var_7.imp == {'another_module': {*()}}
    assert var_7.root == {'another_module': 'another_module', 'another_module.BaseClass': 'another_module', 'another_module.DerivedClass': 'another_module'}
    var_11 = module_0.Parser()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'apimd.parser.Parser'
    assert var_11.link is True
    assert var_11.b_level == 1
    assert var_11.toc is False
    assert var_11.level == {}
    assert var_11.doc == {}
    assert var_11.docstring == {}
    assert var_11.imp == {}
    assert var_11.root == {}
    assert var_11.alias == {}
    assert var_11.const == {}
    var_12 = 'module_with_decorators'
    var_13 = '\n@property\ndef computed_property():\n    """Property docstring."""\n    return 42\n\n@classmethod\n@staticmethod\ndef multi_decorated():\n    pass\n'
    var_14 = var_11.parse(var_12, var_13)
    assert var_11.level == {'module_with_decorators': 0, 'module_with_decorators.computed_property': 0, 'module_with_decorators.multi_decorated': 0}
    assert var_11.doc == {'module_with_decorators': '## Module `{}`\n<a id="{}"></a>\n\n', 'module_with_decorators.computed_property': '### computed_property()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@property` |\n\n| return |\n|:------:|\n| `Any` |\n\n', 'module_with_decorators.multi_decorated': '### multi_decorated()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_11.docstring == {'module_with_decorators.computed_property': 'Property docstring.'}
    assert var_11.imp == {'module_with_decorators': {*()}}
    assert var_11.root == {'module_with_decorators': 'module_with_decorators', 'module_with_decorators.computed_property': 'module_with_decorators', 'module_with_decorators.multi_decorated': 'module_with_decorators'}

@pytest.mark.xfail(strict=True)
def test_case_51():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n'
    var_2 = 'test_module.Mylass'
    var_3 = None
    var_4 = module_0.Parser(var_3, toc=var_3, const=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is None
    assert var_4.b_level == 1
    assert var_4.toc is None
    assert var_4.level == {}
    assert var_4.doc == {}
    assert var_4.docstring == {}
    assert var_4.imp == {}
    assert var_4.root == {}
    assert var_4.alias == {}
    assert var_4.const is None
    var_5 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n': 1, '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': 1, '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': 1}
    assert var_0.doc == {'\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `it` |\n| `y` | `str` |\n\n', '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n': '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass': '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '\nclass MyClass:\n    x: it\n    y: str\n    \n    def __init__(self):\n        del self.x\n'}
    var_6 = var_4.parse(var_2, var_1)
    assert var_4.level == {'test_module.Mylass': 1, 'test_module.Mylass.MyClass': 1, 'test_module.Mylass.MyClass.__init__': 1}
    assert var_4.doc == {'test_module.Mylass': '## Module `{}`\n\n', 'test_module.Mylass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `it` |\n| `y` | `str` |\n\n', 'test_module.Mylass.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_4.imp == {'test_module.Mylass': {*()}}
    assert var_4.root == {'test_module.Mylass': 'test_module.Mylass', 'test_module.Mylass.MyClass': 'test_module.Mylass', 'test_module.Mylass.MyClass.__init__': 'test_module.Mylass'}
    var_4.compile()

def test_case_52():
    var_0 = 'This is a regular docstring.'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'This is a regular docstring.'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_2 = ">>> print('hello')"
    var_3 = module_0.doctest(var_2)
    assert var_3 == "```python\n>>> print('hello')\n```"
    var_4 = '>>> x = 1\n>>> print(x)'
    var_5 = module_0.doctest(var_4)
    assert var_5 == '```python\n>>> x = 1\n>>> print(x)\n```'
    var_6 = 'Some text.\n>>> code()\nMore text.'
    var_7 = module_0.doctest(var_6)
    assert var_7 == 'Some text.\n```python\n>>> code()\n```\nMore text.'
    var_8 = '>>> for i in range(3):\n...     print(i)'
    var_9 = module_0.doctest(var_8)
    assert var_9 == '```python\n>>> for i in range(3):\n```\n...     print(i)'
    var_10 = '>>> first()\nText\n>>> second()'
    var_11 = module_0.doctest(var_10)
    assert var_11 == '```python\n>>> first()\n```\nText\n```python\n>>> second()\n```'
    var_12 = ''
    var_13 = module_0.doctest(var_12)
    assert var_13 == ''
    var_14 = '   \n\t\n'
    var_15 = module_0.doctest(var_14)
    assert var_15 == '   \n\t'

@pytest.mark.xfail(strict=True)
def test_case_53():
    var_0 = 'test_module'
    var_1 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_2 = '\nclass MockClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    \n    @classmethod\n    def class_method(cls):\n        """Class method docstring."""\n        pass\n'
    var_3 = var_1.parse(var_0, var_2)
    assert var_1.level == {'test_module': 0, 'test_module.MockClass': 0, 'test_module.MockClass.method': 0, 'test_module.MockClass.class_method': 0}
    assert var_1.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MockClass': '### class MockClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.MockClass.method': '#### MockClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'test_module.MockClass.class_method': '#### MockClass.class_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | return |\n|:---:|:------:|\n| `type[Self]` | `Any` |\n\n'}
    assert var_1.docstring == {'test_module.MockClass': 'Class docstring.', 'test_module.MockClass.method': 'Method docstring.', 'test_module.MockClass.class_method': 'Class method docstring.'}
    assert var_1.imp == {'test_module': {*()}}
    assert var_1.root == {'test_module': 'test_module', 'test_module.MockClass': 'test_module', 'test_module.MockClass.method': 'test_module', 'test_module.MockClass.class_method': 'test_module'}
    var_4 = 'i'
    var_5 = None
    var_6 = var_1.load_docstring(var_4, var_5)
    var_7 = var_1.compile()
    assert var_7 == '## Module `test_module`\n<a id="test_module"></a>\n\n### class MockClass\n\n*Full name:* `test_module.MockClass`\n<a id="test_module-mockclass"></a>\n\nClass docstring.\n\n#### MockClass.class_method()\n\n*Full name:* `test_module.MockClass.class_method`\n<a id="test_module-mockclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | return |\n|:---:|:------:|\n| `type[Self]` | `Any` |\n\nClass method docstring.\n\n#### MockClass.method()\n\n*Full name:* `test_module.MockClass.method`\n<a id="test_module-mockclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\nMethod docstring.\n'
    var_8 = 'test_module.MyClass'
    var_9 = module_0.Parser(level=var_3, docstring=var_6, imp=var_6, alias=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc is False
    assert var_9.level is None
    assert var_9.doc == {}
    assert var_9.docstring is None
    assert var_9.imp is None
    assert var_9.root == {}
    assert var_9.alias is None
    assert var_9.const == {}
    var_10 = '_R;64"$ElJXh}#CO]"'
    var_9.parse(var_8, var_10)

def test_case_54():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_0.compile()
    assert var_2 == '\n'
    var_3 = var_0.func_ann(var_2, var_1, has_self=var_1, cls_method=var_2)
    var_4 = None
    var_5 = var_0.compile()
    assert var_5 == '\n'
    var_6 = '<B)f-op\x0bsI%l(0Gq/<h'
    var_7 = 'S'
    var_8 = None
    var_9 = module_0.Resolver(var_6, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Resolver'
    assert var_9.root == '<B)f-op\x0bsI%l(0Gq/<h'
    assert var_9.alias is None
    assert var_9.self_ty == ''
    var_10 = [var_7, var_4]
    var_11 = module_1.Assign(*var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Assign'
    assert var_11.targets == 'S'
    assert var_11.value is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Assign.type_comment is None
    var_12 = var_0.globals(var_4, var_11)
    var_13 = module_0.const_type(var_8)
    assert var_13 == 'Any'
    var_14 = module_1.List(*var_10)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.List'
    assert var_14.elts == 'S'
    assert var_14.ctx is None
    var_15 = module_0.const_type(var_14)
    assert var_15 == 'list'
    var_16 = var_0.load_docstring(var_1, var_8)

@pytest.mark.xfail(strict=True)
def test_case_55():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'test_modu:e.EmptyCla/s'
    var_2 = var_0.__repr__()
    assert var_2 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_3 = '3[Y@Hvt'
    var_4 = module_1.Load()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = module_0.Parser()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is True
    assert var_5.b_level == 1
    assert var_5.toc is False
    assert var_5.level == {}
    assert var_5.doc == {}
    assert var_5.docstring == {}
    assert var_5.imp == {}
    assert var_5.root == {}
    assert var_5.alias == {}
    assert var_5.const == {}
    var_6 = var_5.parse(var_3, var_1)
    assert var_5.level == {'3[Y@Hvt': 0}
    assert var_5.doc == {'3[Y@Hvt': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_5.imp == {'3[Y@Hvt': {*()}}
    assert var_5.root == {'3[Y@Hvt': '3[Y@Hvt'}
    var_7 = '\nfrom enu import Enum\nclass Clor(Enum):\n    RED = 1\n    HREEN = 2\n    BLUE = 3\n'
    var_8 = var_5.parse(var_2, var_7)
    assert var_5.level == {'3[Y@Hvt': 0, 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 0, 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={}).Clor': 0}
    assert var_5.doc == {'3[Y@Hvt': '## Module `{}`\n<a id="{}"></a>\n\n', 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': '## Module `{}`\n<a id="{}"></a>\n\n', 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={}).Clor': '### class Clor\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enu.Enum` |\n\n| Members | Type |\n|:-------:|:----:|\n| `BLUE` | `int` |\n| `HREEN` | `int` |\n| `RED` | `int` |\n\n'}
    assert var_5.imp == {'3[Y@Hvt': {*()}, 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': {*()}}
    assert var_5.root == {'3[Y@Hvt': '3[Y@Hvt', 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})', 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={}).Clor': 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'}
    assert var_5.alias == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={}).Enum': 'enu.Enum'}
    var_9 = var_5.__post_init__()
    var_10 = module_1.Import()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Import'
    var_0.imports(var_9, var_10)

@pytest.mark.xfail(strict=True)
def test_case_56():
    var_0 = module_0.Parser()
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n': 1, '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass': 1, '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': 1}
    assert var_0.doc == {'\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `3` |\n| `y` | `sr` |\n\n', '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n': '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass': '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n', '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n.MyClass.__init__': '\nclass MyClass:\n    x: 3\n    y: sr\n    \n    def __init__(self):\n        del self.x\n'}
    var_3 = var_0.__eq__(var_2)
    var_3.compile()