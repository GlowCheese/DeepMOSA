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
    var_0 = 'qEV'
    var_1 = module_0.esc_underscore(var_0)
    assert var_1 == 'qEV'
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

def test_case_3():
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

def test_case_4():
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
def test_case_5():
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

@pytest.mark.xfail(strict=True)
def test_case_6():
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
    var_0.imports(var_0, var_0)

def test_case_7():
    var_0 = ';&BF'
    var_1 = True
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == ';&BF'
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

def test_case_8():
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

def test_case_9():
    var_0 = "'l0+zz:fx2#KXK7?t]\x0c"
    var_1 = module_0.code(var_0)
    assert var_1 == "`'l0+zz:fx2#KXK7?t]\x0c`"
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
    var_0 = 'RZO'
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
def test_case_11():
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
    var_2 = var_0.compile()
    assert var_2 == '\n'
    var_3 = []
    var_4 = var_0.class_api(var_1, var_2, var_3, var_3)
    var_5 = module_0.doctest(var_2)
    assert var_5 == ''
    var_6 = module_0.code(var_5)
    assert var_6 == ' '
    var_7 = [var_6]
    var_0.class_api(var_2, var_0, var_7, var_7)

@pytest.mark.xfail(strict=True)
def test_case_12():
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
    var_2 = var_1.__repr__()
    assert var_2 == 'Parser(link=True, b_level=1, toc=False, level={}, doc=None, docstring=None, imp={}, root={}, alias={}, const={})'
    var_1.is_public(var_0)

def test_case_13():
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

def test_case_14():
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
    var_1 = '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'
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
    var_3 = var_2.parse(var_1, var_1)
    assert var_2.level == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': 3, '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': 3, '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': 3}
    assert var_2.doc == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\n'}
    assert var_2.docstring == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': 'Test module docstring.', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_2.imp == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_2.root == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'}

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
    var_1 = '_5\\'
    var_2 = var_0.compile()
    assert var_2 == '\n'
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
    var_4 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'
    var_5 = var_3.parse(var_1, var_4)
    assert var_3.level == {'_5\\': 0, '_5\\.public_func': 0, '_5\\._private_func': 0}
    assert var_3.doc == {'_5\\': '## Module `{}`\n<a id="{}"></a>\n\n', '_5\\.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '_5\\._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_3.docstring == {'_5\\': 'Test module docstring.'}
    assert var_3.imp == {'_5\\': {'_5\\.public_func'}}
    assert var_3.root == {'_5\\': '_5\\', '_5\\.public_func': '_5\\', '_5\\._private_func': '_5\\'}
    assert var_3.alias == {'_5\\.__all__': "['public_func']"}
    var_6 = var_3.compile()
    assert var_6 == '### public_func()\n\n*Full name:* `_5\\.public_func`\n<a id="_5\\-public_func"></a>\n\n| return |\n|:------:|\n| `None` |\n'

def test_case_16():
    var_0 = 'HVx'
    var_1 = module_0.table(items=var_0)
    assert var_1 == '||\n||\n| H |\n| V |\n| x |\n\n'
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
    var_2 = module_1.Set()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Set'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096

def test_case_17():
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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = var_0.load_docstring(var_1, var_1)

def test_case_18():
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
def test_case_19():
    var_0 = 'test_module'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'test_module'
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
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_3 = module_1.parse(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Module'
    assert f'{type(var_3.body).__module__}.{type(var_3.body).__qualname__}' == 'builtins.list'
    assert len(var_3.body) == 1
    assert var_3.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_4 = [var_3]
    var_2.class_api(var_0, var_2, var_4, var_4)

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
    var_1 = var_0.__repr__()
    assert var_1 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 0}
    assert var_0.doc == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': {*()}}
    assert var_0.root == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'}

def test_case_21():
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
    var_3 = var_2.globals(var_0, var_0)

def test_case_22():
    var_0 = ',|y5B'
    var_1 = None
    var_2 = module_0.code(var_0)
    assert var_2 == '<code>,&#124;y5B</code>'
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
    var_3 = module_0.Parser(doc=var_1, docstring=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc is None
    assert var_3.docstring == '<code>,&#124;y5B</code>'
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4 = []
    var_5 = var_3.class_api(var_0, var_0, var_4, var_0)

def test_case_23():
    var_0 = ',|y5B'
    var_1 = None
    var_2 = module_0.Parser(doc=var_1, docstring=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc is None
    assert var_2.docstring == ',|y5B'
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
def test_case_24():
    var_0 = module_1.Constant()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'ast.Constant'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    module_0.const_type(var_0)

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
    var_1 = var_0.__repr__()
    assert var_1 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 0}
    assert var_0.doc == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': {*()}}
    assert var_0.root == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'}
    var_3 = var_0.compile()
    assert var_3 == '\n'
    var_4 = module_2.getdoc(var_1)
    assert var_4 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
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
    var_5 = module_0.esc_underscore(var_4)
    assert var_5 == "str(object='') -> str\nstr(bytes\\_or\\_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.\\_\\_str\\_\\_() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    var_5.load_docstring(var_1, var_4)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = 'GQ'
    var_1 = False
    var_2 = ''
    var_3 = '/`uy.N'
    var_4 = {var_2: var_0, var_0: var_2, var_3: var_0}
    var_5 = module_0.Parser(var_1, doc=var_2, alias=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is False
    assert var_5.b_level == 1
    assert var_5.toc is False
    assert var_5.level == {}
    assert var_5.doc == ''
    assert var_5.docstring == {}
    assert var_5.imp == {}
    assert var_5.root == {}
    assert var_5.alias == {'': 'GQ', 'GQ': '', '/`uy.N': 'GQ'}
    assert var_5.const == {}
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
    var_5.compile()

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
    var_1 = var_0.__repr__()
    assert var_1 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 0}
    assert var_0.doc == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': {*()}}
    assert var_0.root == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'}
    var_3 = var_0.compile()
    assert var_3 == '\n'

def test_case_29():
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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n'}
    var_3 = var_0.compile()
    assert var_3 == '\n'
    var_4 = var_0.load_docstring(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 'HVx'
    var_1 = module_2.getdoc(var_0)
    assert var_1 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
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
    var_2 = module_0.Parser(doc=var_1, docstring=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert var_2.docstring == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
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
    var_3 = var_2.load_docstring(var_1, var_1)
    var_4 = module_0.const_type(var_1)
    assert var_4 == 'Any'
    var_5 = None
    var_6 = module_0.Parser(doc=var_1, docstring=var_5, root=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert var_6.docstring is None
    assert var_6.imp == {}
    assert var_6.root == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert var_6.alias == {}
    assert var_6.const == {}
    var_7 = module_0.Resolver(var_1, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Resolver'
    assert var_7.root == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert var_7.alias is None
    assert var_7.self_ty == ''
    var_2.imports(var_4, var_7)

@pytest.mark.xfail(strict=True)
def test_case_31():
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
    var_3 = module_0.Parser(doc=var_1, docstring=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc is None
    assert var_3.docstring == '{nq'
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4 = [var_1]
    var_5 = module_1.Attribute(*var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Attribute'
    assert var_5.value is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3.resolve(var_0, var_5, var_0)

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
    var_6 = 'kT|3'
    var_7 = []
    var_8 = var_4.class_api(var_0, var_6, var_7, var_6)
    var_9 = {}
    var_10 = module_1.Dict(**var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Dict'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    module_0.const_type(var_10)

def test_case_33():
    var_0 = []
    var_1 = module_0.walk_body(var_0)
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
    var_2 = list(var_1)
    var_3 = 'x'
    var_4 = module_1.Load()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = module_1.Name()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    var_6 = [var_5]
    var_7 = 1
    var_8 = module_1.Constant()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Constant'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_9 = module_1.Assign()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Assign'
    assert module_1.Assign.type_comment is None
    var_10 = 'print'
    var_11 = module_1.Load()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Load'
    var_12 = module_1.Name()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Name'
    var_13 = []
    var_14 = []
    var_15 = module_1.Call(*var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Call'
    var_16 = module_1.Expr()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Expr'
    var_17 = [var_9, var_16]
    var_18 = module_0.walk_body(var_17)
    var_19 = list(var_18)
    var_20 = True
    var_21 = module_1.Constant()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'ast.Constant'
    var_22 = 'y'
    var_23 = module_1.Load()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'ast.Load'
    var_24 = module_1.Name()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'ast.Name'
    var_25 = [var_24]
    var_26 = 2
    var_27 = module_1.Constant()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'ast.Constant'
    var_28 = module_1.Assign()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'ast.Assign'
    var_29 = [var_28]
    var_30 = 'exit'
    var_31 = module_1.Load()
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'ast.Load'
    var_32 = module_1.Name()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'ast.Name'
    var_33 = []
    var_34 = []
    var_35 = module_1.Call(*var_33)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'ast.Call'
    var_36 = module_1.Expr()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'ast.Expr'
    var_37 = [var_36]
    var_38 = module_1.If()
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'ast.If'
    var_39 = [var_38]
    var_40 = module_0.walk_body(var_39)
    with pytest.raises(AttributeError):
        var_41 = list(var_40)

def test_case_34():
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
    var_1 = []
    var_2 = module_1.ImportFrom(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.ImportFrom'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_3 = var_0.imports(var_2, var_2)
    var_4 = var_0.__post_init__()

def test_case_35():
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
    var_1 = var_0.__repr__()
    assert var_1 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 0}
    assert var_0.doc == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': {*()}}
    assert var_0.root == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'}
    var_3 = var_0.is_public(var_1)
    assert var_3 is False

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = var_0.compile()
    assert var_2 == '\n'
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
    assert var_3.level == {'\n': 0}
    assert var_3.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_3.imp == {'\n': {*()}}
    assert var_3.root == {'\n': '\n'}
    var_5 = var_3.compile()
    assert var_5 == '\n'
    var_6 = '\n"""Test module docstring."""\nfrom typing import List\nIntList = List[int]\nx: IntList = []\n'
    var_7 = var_3.parse(var_1, var_6)
    assert var_3.docstring == {'\n': 'Test module docstring.'}
    assert var_3.alias == {'\n.List': 'typing.List', '\n.IntList': 'List[int]', '\n.x': '[]'}

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = None
    var_3 = var_0.load_docstring(var_2, var_2)
    var_4 = module_2.getdoc(var_0)
    assert var_4 == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
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
    var_5 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n'}
    var_6 = module_0.is_public_family(var_4)
    assert var_6 is True
    var_7 = module_0.doctest(var_4)
    assert var_7 == 'AST parser.\n\nUsage:\n```python\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n```\n\nOr create with parameters:\n```python\n>>> p = Parser.new(link=True, level=1)\n```'
    var_8 = var_0.compile()
    assert var_8 == '\n'
    var_9 = module_0.Parser()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc is False
    assert var_9.level == {}
    assert var_9.doc == {}
    assert var_9.docstring == {}
    assert var_9.imp == {}
    assert var_9.root == {}
    assert var_9.alias == {}
    assert var_9.const == {}
    var_10 = 'BaseClass'
    var_11 = var_0.__post_init__()
    var_12 = var_9.globals(var_2, var_11)
    var_13 = var_9.__post_init__()
    var_14 = module_1.parse(var_10)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Module'
    assert f'{type(var_14.body).__module__}.{type(var_14.body).__qualname__}' == 'builtins.list'
    assert len(var_14.body) == 1
    assert var_14.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_15 = var_0.resolve(var_12, var_14)
    assert var_15 == 'BaseClass'
    var_16 = module_0.const_type(var_2)
    assert var_16 == 'Any'
    var_17 = var_0.is_public(var_1)
    assert var_17 is False

@pytest.mark.xfail(strict=True)
def test_case_38():
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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = [var_1, var_0, var_1]
    var_3 = None
    var_4 = var_0.compile()
    assert var_4 == '\n'
    var_5 = module_2.getdoc(var_4)
    assert var_5 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
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
    var_6 = module_0.const_type(var_3)
    assert var_6 == 'Any'
    var_7 = module_0.is_public_family(var_6)
    assert var_7 is True
    var_8 = var_0.load_docstring(var_4, var_4)
    var_9 = module_0.doctest(var_1)
    assert var_9 == ''
    var_10 = module_1.ImportFrom(*var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.ImportFrom'
    assert var_10.module == '\n'
    assert f'{type(var_10.names).__module__}.{type(var_10.names).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.level == '\n'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_11 = var_0.globals(var_1, var_3)
    var_12 = module_3.field()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'dataclasses.Field'
    assert f'{type(module_3.MISSING).__module__}.{type(module_3.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_3.KW_ONLY).__module__}.{type(module_3.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_3.Field.compare).__module__}.{type(module_3.Field.compare).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Field.default).__module__}.{type(module_3.Field.default).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Field.default_factory).__module__}.{type(module_3.Field.default_factory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Field.hash).__module__}.{type(module_3.Field.hash).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Field.init).__module__}.{type(module_3.Field.init).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Field.kw_only).__module__}.{type(module_3.Field.kw_only).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Field.metadata).__module__}.{type(module_3.Field.metadata).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Field.name).__module__}.{type(module_3.Field.name).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Field.repr).__module__}.{type(module_3.Field.repr).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_3.Field.type).__module__}.{type(module_3.Field.type).__qualname__}' == 'builtins.member_descriptor'
    var_13 = var_12.__eq__(var_3)
    var_0.imports(var_10, var_10)

@pytest.mark.xfail(strict=True)
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
    var_1 = var_0.__repr__()
    assert var_1 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_2 = module_0.code(var_1)
    assert var_2 == '`Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})`'
    var_3 = var_0.load_docstring(var_1, var_1)
    var_4 = module_0.table(items=var_1)
    assert var_4 == '||\n||\n| P |\n| a |\n| r |\n| s |\n| e |\n| r |\n| ( |\n| l |\n| i |\n| n |\n| k |\n| = |\n| T |\n| r |\n| u |\n| e |\n| , |\n|   |\n| b |\n| _ |\n| l |\n| e |\n| v |\n| e |\n| l |\n| = |\n| 1 |\n| , |\n|   |\n| t |\n| o |\n| c |\n| = |\n| F |\n| a |\n| l |\n| s |\n| e |\n| , |\n|   |\n| l |\n| e |\n| v |\n| e |\n| l |\n| = |\n| { |\n| } |\n| , |\n|   |\n| d |\n| o |\n| c |\n| = |\n| { |\n| } |\n| , |\n|   |\n| d |\n| o |\n| c |\n| s |\n| t |\n| r |\n| i |\n| n |\n| g |\n| = |\n| { |\n| } |\n| , |\n|   |\n| i |\n| m |\n| p |\n| = |\n| { |\n| } |\n| , |\n|   |\n| r |\n| o |\n| o |\n| t |\n| = |\n| { |\n| } |\n| , |\n|   |\n| a |\n| l |\n| i |\n| a |\n| s |\n| = |\n| { |\n| } |\n| , |\n|   |\n| c |\n| o |\n| n |\n| s |\n| t |\n| = |\n| { |\n| } |\n| ) |\n\n'
    var_5 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 0}
    assert var_0.doc == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': {*()}}
    assert var_0.root == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'}
    var_6 = module_0.esc_underscore(var_1)
    assert var_6 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_7 = module_0.is_public_family(var_2)
    assert var_7 is True
    var_8 = module_0.doctest(var_6)
    assert var_8 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_9 = module_1.ImportFrom()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.ImportFrom'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_10 = module_0.Parser(b_level=var_5, toc=var_7, docstring=var_1, imp=var_3, const=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level is None
    assert var_10.toc is True
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert var_10.imp is None
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_11 = var_10.compile()
    assert var_11 == '**Table of contents:**\n\n\n'
    var_12 = module_0.Parser(imp=var_1, alias=var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'apimd.parser.Parser'
    assert var_12.link is True
    assert var_12.b_level == 1
    assert var_12.toc is False
    assert var_12.level == {}
    assert var_12.doc == {}
    assert var_12.docstring == {}
    assert var_12.imp == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert var_12.root == {}
    assert var_12.alias == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert var_12.const == {}
    var_13 = module_2.getdoc(var_1)
    assert var_13 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
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
    var_13.load_docstring(var_2, var_1)

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
    var_1 = var_0.__repr__()
    assert var_1 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_2 = [var_1]
    var_3 = module_1.AnnAssign(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.AnnAssign'
    assert var_3.target == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.AnnAssign.value is None
    var_4 = var_0.globals(var_1, var_3)
    module_0.is_public_family(var_4)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = None
    var_3 = var_0.load_docstring(var_2, var_2)
    var_4 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n'}
    var_5 = var_0.compile()
    assert var_5 == '\n'
    var_6 = module_0.const_type(var_1)
    assert var_6 == 'Any'
    var_7 = 'IX/E]EkP"LCS%$>~E'
    var_8 = module_0.is_magic(var_7)
    assert var_8 is False
    var_9 = var_6.__repr__()
    assert var_9 == "'Any'"
    var_10 = module_1.parse(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Module'
    assert f'{type(var_10.body).__module__}.{type(var_10.body).__qualname__}' == 'builtins.list'
    assert len(var_10.body) == 1
    assert var_10.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_11 = var_0.resolve(var_9, var_10)
    assert var_11 == 'Any'
    var_12 = var_0.load_docstring(var_5, var_6)

@pytest.mark.xfail(strict=True)
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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = None
    var_3 = var_0.load_docstring(var_2, var_2)
    var_4 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n'}
    var_5 = var_0.compile()
    assert var_5 == '\n'
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
    var_7 = module_0.esc_underscore(var_5)
    assert var_7 == '\n'
    var_8 = var_7.__repr__()
    assert var_8 == "'\\n'"
    var_9 = module_1.parse(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Module'
    assert f'{type(var_9.body).__module__}.{type(var_9.body).__qualname__}' == 'builtins.list'
    assert len(var_9.body) == 1
    assert var_9.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_0.resolve(var_8, var_9)

@pytest.mark.xfail(strict=True)
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
    var_1 = module_0.const_type(var_0)
    assert var_1 == 'Any'
    var_2 = '@\x0crVM\x0cgVM'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = var_0.load_docstring(var_1, var_1)
    var_5 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Any': 0}
    assert var_0.doc == {'Any': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Any': {*()}}
    assert var_0.root == {'Any': 'Any'}
    var_6 = var_0.load_docstring(var_1, var_1)
    var_7 = var_0.__repr__()
    assert var_7 == 'Parser(link=True, b_level=1, toc=False, level={\'Any\': 0}, doc={\'Any\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={}, imp={\'Any\': set()}, root={\'Any\': \'Any\'}, alias={}, const={})'
    var_8 = module_0.doctest(var_1)
    assert var_8 == 'Any'
    var_9 = module_0.table(items=var_1)
    assert var_9 == '||\n||\n| A |\n| n |\n| y |\n\n'
    var_10 = var_0.compile()
    assert var_10 == '\n'
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
    var_12 = var_11.globals(var_1, var_8)
    var_13 = module_1.parse(var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Module'
    assert f'{type(var_13.body).__module__}.{type(var_13.body).__qualname__}' == 'builtins.list'
    assert len(var_13.body) == 1
    assert var_13.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_14 = var_0.resolve(var_12, var_13)
    assert var_14 == 'Parser(link=True, b_level=1, toc=False, level={Any: 0}, doc={Any: \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={}, imp={Any: set()}, root={Any: Any}, alias={}, const={})'
    var_15 = module_0.const_type(var_7)
    assert var_15 == 'Any'
    var_11.is_public(var_5)

@pytest.mark.xfail(strict=True)
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
    var_1 = module_0.const_type(var_0)
    assert var_1 == 'Any'
    var_2 = [var_0, var_1]
    var_3 = '@\x0crVM\x0cgVM'
    var_4 = var_0.load_docstring(var_1, var_1)
    var_5 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Any': 0}
    assert var_0.doc == {'Any': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Any': {*()}}
    assert var_0.root == {'Any': 'Any'}
    var_6 = var_0.load_docstring(var_3, var_1)
    var_7 = module_0.esc_underscore(var_1)
    assert var_7 == 'Any'
    var_8 = module_0.is_public_family(var_1)
    assert var_8 is True
    var_9 = 'b {'
    var_10 = None
    var_11 = var_0.__eq__(var_10)
    var_12 = var_11.__eq__(var_6)
    var_13 = 'kTnBrVIuUD*+\rn1`'
    var_14 = ':'
    var_15 = '<d2G7]>r#X`'
    var_16 = 'U2'
    var_17 = {var_13: var_14, var_14: var_15, var_9: var_16}
    var_18 = module_0.Resolver(var_9, var_17, var_7)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'apimd.parser.Resolver'
    assert var_18.root == 'b {'
    assert var_18.alias == {'kTnBrVIuUD*+\rn1`': ':', ':': '<d2G7]>r#X`', 'b {': 'U2'}
    assert var_18.self_ty == 'Any'
    var_19 = module_0.doctest(var_7)
    assert var_19 == 'Any'
    var_20 = module_0.table(items=var_1)
    assert var_20 == '||\n||\n| A |\n| n |\n| y |\n\n'
    var_21 = module_1.ImportFrom(*var_2)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'ast.ImportFrom'
    assert f'{type(var_21.module).__module__}.{type(var_21.module).__qualname__}' == 'apimd.parser.Parser'
    assert var_21.names == 'Any'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_22 = var_0.compile()
    assert var_22 == '\n'
    var_23 = module_0.Parser()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'apimd.parser.Parser'
    assert var_23.link is True
    assert var_23.b_level == 1
    assert var_23.toc is False
    assert var_23.level == {}
    assert var_23.doc == {}
    assert var_23.docstring == {}
    assert var_23.imp == {}
    assert var_23.root == {}
    assert var_23.alias == {}
    assert var_23.const == {}
    var_24 = module_1.parse(var_7)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'ast.Module'
    assert f'{type(var_24.body).__module__}.{type(var_24.body).__qualname__}' == 'builtins.list'
    assert len(var_24.body) == 1
    assert var_24.type_ignores == []
    var_25 = var_0.resolve(var_15, var_24)
    assert var_25 == 'Any'
    var_26 = module_0.const_type(var_21)
    assert var_26 == 'Any'
    var_27 = var_18.generic_visit(var_24)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'ast.Module'
    assert f'{type(var_27.body).__module__}.{type(var_27.body).__qualname__}' == 'builtins.list'
    assert len(var_27.body) == 1
    assert var_27.type_ignores == []
    var_23.imports(var_5, var_5)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = '9z?'
    var_1 = module_0.code(var_0)
    assert var_1 == '`9z?`'
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
    var_2 = None
    var_3 = module_1.Call()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Call'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_4 = module_0.Parser(b_level=var_2, toc=var_2, doc=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level is None
    assert var_4.toc is None
    assert var_4.level == {}
    assert var_4.doc is None
    assert var_4.docstring == {}
    assert var_4.imp == {}
    assert var_4.root == {}
    assert var_4.alias == {}
    assert var_4.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    module_0.const_type(var_3)

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
    var_1 = module_0.const_type(var_0)
    assert var_1 == 'Any'
    var_2 = 'i`[.a1ure*-Q5i'
    var_3 = module_1.Import()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Import'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_0.imports(var_2, var_3)

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
    var_1 = var_0.__repr__()
    assert var_1 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_2 = [var_1, var_1, var_1]
    var_3 = module_1.Subscript(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Subscript'
    assert var_3.value == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert var_3.slice == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert var_3.ctx == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_0.resolve(var_1, var_3)

@pytest.mark.xfail(strict=True)
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
    var_1 = module_0.const_type(var_0)
    assert var_1 == 'Any'
    var_2 = [var_0, var_1]
    var_3 = '@\x0crwM\x0cgV5'
    var_4 = module_0.is_public_family(var_3)
    assert var_4 is True
    var_5 = var_0.load_docstring(var_1, var_1)
    var_6 = module_0.table(items=var_1)
    assert var_6 == '||\n||\n| A |\n| n |\n| y |\n\n'
    var_7 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Any': 0}
    assert var_0.doc == {'Any': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Any': {*()}}
    assert var_0.root == {'Any': 'Any'}
    var_8 = var_0.__eq__(var_2)
    var_9 = var_0.load_docstring(var_1, var_1)
    var_10 = var_0.__post_init__()
    var_11 = module_0.esc_underscore(var_1)
    assert var_11 == 'Any'
    var_12 = 'kTnBrVIuUD*+\rn1`'
    var_13 = '[\\GsNF,#ad%4cKN),sz'
    var_14 = 'Pd2G.]>r#X`'
    var_15 = {var_12: var_11, var_13: var_14, var_11: var_1, var_1: var_11}
    var_16 = 'JIKd)-\x0b*i\x0cSUH)H;y'
    var_17 = module_0.Resolver(var_10, var_15, var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'apimd.parser.Resolver'
    assert var_17.root is None
    assert var_17.alias == {'kTnBrVIuUD*+\rn1`': 'Any', '[\\GsNF,#ad%4cKN),sz': 'Pd2G.]>r#X`', 'Any': 'Any'}
    assert var_17.self_ty == 'JIKd)-\x0b*i\x0cSUH)H;y'
    var_18 = module_0.table(items=var_1)
    assert var_18 == '||\n||\n| A |\n| n |\n| y |\n\n'
    var_19 = var_0.compile()
    assert var_19 == '\n'
    var_20 = module_0.Parser()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'apimd.parser.Parser'
    assert var_20.link is True
    assert var_20.b_level == 1
    assert var_20.toc is False
    assert var_20.level == {}
    assert var_20.doc == {}
    assert var_20.docstring == {}
    assert var_20.imp == {}
    assert var_20.root == {}
    assert var_20.alias == {}
    assert var_20.const == {}
    var_21 = 'uSe\r@:9^'
    var_22 = var_20.globals(var_21, var_8)
    var_23 = var_20.__repr__()
    assert var_23 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_24 = module_1.parse(var_11)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'ast.Module'
    assert f'{type(var_24.body).__module__}.{type(var_24.body).__qualname__}' == 'builtins.list'
    assert len(var_24.body) == 1
    assert var_24.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_25 = var_0.resolve(var_23, var_24)
    assert var_25 == 'Any'
    var_26 = False
    var_27 = var_17.visit(var_24)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'ast.Module'
    assert f'{type(var_27.body).__module__}.{type(var_27.body).__qualname__}' == 'builtins.list'
    assert len(var_27.body) == 1
    assert var_27.type_ignores == []
    var_28 = {var_1: var_24, var_11: var_27}
    var_29 = module_0.Parser(var_23, toc=var_26, level=var_7, root=var_27, const=var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'apimd.parser.Parser'
    assert var_29.link == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert var_29.b_level == 1
    assert var_29.toc is False
    assert var_29.level is None
    assert var_29.doc == {}
    assert var_29.docstring == {}
    assert var_29.imp == {}
    assert f'{type(var_29.root).__module__}.{type(var_29.root).__qualname__}' == 'ast.Module'
    assert var_29.alias == {}
    assert f'{type(var_29.const).__module__}.{type(var_29.const).__qualname__}' == 'builtins.dict'
    assert len(var_29.const) == 1
    var_30 = module_0.Parser(imp=var_6, alias=var_10)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'apimd.parser.Parser'
    assert var_30.link is True
    assert var_30.b_level == 1
    assert var_30.toc is False
    assert var_30.level == {}
    assert var_30.doc == {}
    assert var_30.docstring == {}
    assert var_30.imp == '||\n||\n| A |\n| n |\n| y |\n\n'
    assert var_30.root == {}
    assert var_30.alias is None
    assert var_30.const == {}
    module_1.NodeVisitor(**var_27)

@pytest.mark.xfail(strict=True)
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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n'}
    var_3 = var_0.__repr__()
    var_4 = var_0.compile()
    assert var_4 == '\n'
    var_5 = '\n"""Test module docstring."""\nfrom typin2 import List\nx: List[int] = []\n'
    var_6 = var_0.parse(var_1, var_5)
    assert var_0.docstring == {'\n': 'Test module docstring.'}
    assert var_0.alias == {'\n.List': 'typin2.List', '\n.x': '[]'}
    var_7 = None
    var_3.generic_visit(var_7)

@pytest.mark.xfail(strict=True)
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
    var_1 = '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'
    var_2 = var_0.compile()
    assert var_2 == '\n'
    var_3 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': 3, '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': 3, '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': 3}
    assert var_0.doc == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\n'}
    assert var_0.docstring == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': 'Test module docstring.', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_0.root == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'}
    var_4 = var_0.compile()
    assert var_4 == '## Module `\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n`\n<a id="\n"""test module docstring-"""\nclass myclass:\n    """class docstring-"""\n    x: int = 1\n    def method(self, a: str) -> none:\n        """method docstring-"""\n        pass\n"></a>\n\nTest module docstring.\n\n### class MyClass\n\n*Full name:* `\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass`\n<a id="\n"""test module docstring-"""\nclass myclass:\n    """class docstring-"""\n    x: int = 1\n    def method(self, a: str) -> none:\n        """method docstring-"""\n        pass\n-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method`\n<a id="\n"""test module docstring-"""\nclass myclass:\n    """class docstring-"""\n    x: int = 1\n    def method(self, a: str) -> none:\n        """method docstring-"""\n        pass\n-myclass-method"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\nMethod docstring.\n'
    var_5 = '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n'
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
    var_7 = '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n'
    var_8 = var_6.parse(var_5, var_7)
    assert var_6.level == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': 1, '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n._private_func': 1, '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass': 1, '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass.__init__': 1}
    assert var_6.doc == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_6.docstring == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': 'Test module docstring.'}
    assert var_6.imp == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': {*()}}
    assert var_6.root == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n._private_func': '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass': '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass.__init__': '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n'}
    var_9 = module_0.Parser()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc is False
    assert var_9.level == {}
    assert var_9.doc == {}
    assert var_9.docstring == {}
    assert var_9.imp == {}
    assert var_9.root == {}
    assert var_9.alias == {}
    assert var_9.const == {}
    var_10 = '\n"""Test module docstring."""\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_11 = var_9.parse(var_5, var_10)
    assert var_9.level == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': 1, '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.Color': 1}
    assert var_9.doc == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| RED |\n| GREEN |\n| BLUE |\n\n'}
    assert var_9.docstring == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': 'Test module docstring.'}
    assert var_9.imp == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': {*()}}
    assert var_9.root == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.Color': '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n'}
    assert var_9.alias == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.Enum': 'enum.Enum'}
    var_12 = var_6.__repr__()
    assert var_12 == 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n\': 1, \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n._private_func\': 1, \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n\': set()}, root={\'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n\': \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n\', \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n._private_func\': \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n\', \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n.MyClass\': \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n\', \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\nfrom typing import List\\nx: List[int] = []\\n\'}, alias={}, const={})'
    var_9.parse(var_12, var_4)

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
    var_2 = 'R+Z*Zk+EVyp\ni:jJQ'
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
    var_4 = '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'
    var_5 = var_3.parse(var_2, var_4)
    assert var_3.level == {'R+Z*Zk+EVyp\ni:jJQ': 0, 'R+Z*Zk+EVyp\ni:jJQ.func': 0}
    assert var_3.doc == {'R+Z*Zk+EVyp\ni:jJQ': '## Module `{}`\n<a id="{}"></a>\n\n', 'R+Z*Zk+EVyp\ni:jJQ.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_3.docstring == {'R+Z*Zk+EVyp\ni:jJQ': 'Test module docstring.'}
    assert var_3.imp == {'R+Z*Zk+EVyp\ni:jJQ': {*()}}
    assert var_3.root == {'R+Z*Zk+EVyp\ni:jJQ': 'R+Z*Zk+EVyp\ni:jJQ', 'R+Z*Zk+EVyp\ni:jJQ.CONST': 'R+Z*Zk+EVyp\ni:jJQ', 'R+Z*Zk+EVyp\ni:jJQ.func': 'R+Z*Zk+EVyp\ni:jJQ'}
    assert var_3.alias == {'R+Z*Zk+EVyp\ni:jJQ.CONST': '42'}
    assert var_3.const == {'R+Z*Zk+EVyp\ni:jJQ.CONST': 'int'}
    var_6 = var_3.compile()
    assert var_6 == '## Module `R+Z*Zk+EVyp\ni:jJQ`\n<a id="r+z*zk+evyp\ni:jjq"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONST` | `int` |\n\nTest module docstring.\n\n### func()\n\n*Full name:* `R+Z*Zk+EVyp\ni:jJQ.func`\n<a id="r+z*zk+evyp\ni:jjq-func"></a>\n\n| return |\n|:------:|\n| `None` |\n'
    var_7 = var_1.compile()
    assert var_7 == '\n'
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

def test_case_52():
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
    var_1 = '7\x0b\x0c\\CF]'
    var_2 = var_0.compile()
    assert var_2 == '\n'
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
    var_4 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'
    var_5 = var_3.parse(var_1, var_4)
    assert var_3.level == {'7\x0b\x0c\\CF]': 0, '7\x0b\x0c\\CF].public_func': 0, '7\x0b\x0c\\CF]._private_func': 0}
    assert var_3.doc == {'7\x0b\x0c\\CF]': '## Module `{}`\n<a id="{}"></a>\n\n', '7\x0b\x0c\\CF].public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '7\x0b\x0c\\CF]._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_3.docstring == {'7\x0b\x0c\\CF]': 'Test module docstring.'}
    assert var_3.imp == {'7\x0b\x0c\\CF]': {'7\x0b\x0c\\CF].public_func'}}
    assert var_3.root == {'7\x0b\x0c\\CF]': '7\x0b\x0c\\CF]', '7\x0b\x0c\\CF].public_func': '7\x0b\x0c\\CF]', '7\x0b\x0c\\CF]._private_func': '7\x0b\x0c\\CF]'}
    assert var_3.alias == {'7\x0b\x0c\\CF].__all__': "['public_func']"}
    var_6 = var_3.compile()
    assert var_6 == '## Module `7\x0b\x0c\\CF]`\n<a id="7\x0b\x0c\\cf]"></a>\n\nTest module docstring.\n\n### public_func()\n\n*Full name:* `7\x0b\x0c\\CF].public_func`\n<a id="7\x0b\x0c\\cf]-public_func"></a>\n\n| return |\n|:------:|\n| `None` |\n'
    var_7 = '\n"""Test module docstring."""\nfrom typing import List\nIntList = List[int]\nx: IntList = []\n'
    var_8 = var_3.parse(var_1, var_7)
    assert var_3.imp == {'7\x0b\x0c\\CF]': {*()}}
    assert var_3.alias == {'7\x0b\x0c\\CF].__all__': "['public_func']", '7\x0b\x0c\\CF].List': 'typing.List', '7\x0b\x0c\\CF].IntList': 'List[int]', '7\x0b\x0c\\CF].x': '[]'}

def test_case_53():
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
    var_1 = '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'
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
    var_3 = module_0.doctest(var_1)
    assert var_3 == '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass'
    var_4 = var_2.compile()
    assert var_4 == '\n'
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
    var_6 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'
    var_7 = '\n"""Test module docstring."""\nfrom enum import Enup\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_8 = var_5.parse(var_6, var_7)
    assert var_5.level == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': 1, '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': 1}
    assert var_5.doc == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `Enum` |\n\n| Members | Type |\n|:-------:|:----:|\n| `BLUE` | `int` |\n| `GREEN` | `int` |\n| `RED` | `int` |\n\n'}
    assert var_5.docstring == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': 'Test module docstring.'}
    assert var_5.imp == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': {*()}}
    assert var_5.root == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'}
    assert var_5.alias == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Enup': 'enum.Enup'}
    var_9 = module_0.Parser()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc is False
    assert var_9.level == {}
    assert var_9.doc == {}
    assert var_9.docstring == {}
    assert var_9.imp == {}
    assert var_9.root == {}
    assert var_9.alias == {}
    assert var_9.const == {}
    var_10 = var_8.__repr__()
    assert var_10 == 'None'

@pytest.mark.xfail(strict=True)
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
    var_1 = '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'
    var_2 = var_0.compile()
    assert var_2 == '\n'
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
    assert var_3.level == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': 3, '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': 3, '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': 3}
    assert var_3.doc == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\n'}
    assert var_3.docstring == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': 'Test module docstring.', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_3.imp == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_3.root == {'\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n': '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass': '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n', '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method': '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'}
    var_5 = var_3.compile()
    assert var_5 == '## Module `\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n`\n<a id="\n"""test module docstring-"""\nclass myclass:\n    """class docstring-"""\n    x: int = 1\n    def method(self, a: str) -> none:\n        """method docstring-"""\n        pass\n"></a>\n\nTest module docstring.\n\n### class MyClass\n\n*Full name:* `\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass`\n<a id="\n"""test module docstring-"""\nclass myclass:\n    """class docstring-"""\n    x: int = 1\n    def method(self, a: str) -> none:\n        """method docstring-"""\n        pass\n-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n.MyClass.method`\n<a id="\n"""test module docstring-"""\nclass myclass:\n    """class docstring-"""\n    x: int = 1\n    def method(self, a: str) -> none:\n        """method docstring-"""\n        pass\n-myclass-method"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\nMethod docstring.\n'
    var_6 = '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n'
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
    var_8 = '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n'
    var_9 = var_7.parse(var_6, var_8)
    assert var_7.level == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': 1, '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n._private_func': 1, '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass': 1, '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass.__init__': 1}
    assert var_7.doc == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_7.docstring == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': 'Test module docstring.'}
    assert var_7.imp == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': {*()}}
    assert var_7.root == {'\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n': '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n._private_func': '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass': '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n', '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass.__init__': '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n'}
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
    var_11 = var_7.compile()
    assert var_11 == '## Module `\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n`\n<a id="\n"""test module docstring-"""\nfrom typing import list\nx: list[int] = []\n"></a>\n\nTest module docstring.\n\n### class MyClass\n\n*Full name:* `\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n.MyClass`\n<a id="\n"""test module docstring-"""\nfrom typing import list\nx: list[int] = []\n-myclass"></a>\n'
    var_3.parse(var_11, var_5)

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
    var_1 = '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'
    var_2 = var_0.compile()
    assert var_2 == '\n'
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
    var_5 = '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n'
    var_6 = var_4.parse(var_5, var_5)
    assert var_4.level == {'\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n': 1, '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n._private_func': 1, '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n.MyClass': 1, '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n.MyClass.__init__': 1}
    assert var_4.doc == {'\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_4.docstring == {'\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n': 'Test module docstring.'}
    assert var_4.imp == {'\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n': {*()}}
    assert var_4.root == {'\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n': '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n', '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n._private_func': '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n', '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n.MyClass': '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n', '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n.MyClass.__init__': '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n'}
    var_7 = var_4.__repr__()
    assert var_7 == 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={})'
    var_8 = var_3.parse(var_7, var_1)
    assert var_3.level == {'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={})': 32, 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={}).MyClass': 32, 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={}).MyClass.method': 32}
    assert var_3.doc == {'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={})': '## Module `{}`\n<a id="{}"></a>\n\n', 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={}).MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n', 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={}).MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\n'}
    assert var_3.docstring == {'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={})': 'Test module docstring.', 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={}).MyClass': 'Class docstring.', 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={}).MyClass.method': 'Method docstring.'}
    assert var_3.imp == {'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={})': {*()}}
    assert var_3.root == {'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={})': 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={})', 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={}).MyClass': 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={})', 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={}).MyClass.method': 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': 1, \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': 1}, doc={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'### \\\\_private\\\\_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `None` |\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `None` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n._private_func\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\', \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n.MyClass.__init__\': \'\\n"""Test module docstring."""\\ndef _private_func() -> None:\\n    pass\\nclass MyClass:\\n    _private_attr: int = 1\\n    def __init__(self) -> None:\\n        pass\\n\'}, alias={}, const={})'}

def test_case_56():
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
    var_2 = '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'
    var_3 = var_1.parse(var_0, var_2)
    assert var_1.level == {'test_module': 0, 'test_module.MyClass': 0, 'test_module.MyClass.method': 0}
    assert var_1.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n', 'test_module.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\n'}
    assert var_1.docstring == {'test_module': 'Test module docstring.', 'test_module.MyClass': 'Class docstring.', 'test_module.MyClass.method': 'Method docstring.'}
    assert var_1.imp == {'test_module': {*()}}
    assert var_1.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module', 'test_module.MyClass.method': 'test_module'}
    var_4 = var_1.compile()
    assert var_4 == '## Module `test_module`\n<a id="test_module"></a>\n\nTest module docstring.\n\n### class MyClass\n\n*Full name:* `test_module.MyClass`\n<a id="test_module-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `test_module.MyClass.method`\n<a id="test_module-myclass-method"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\nMethod docstring.\n'
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
    var_6 = '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'
    var_7 = var_5.parse(var_0, var_6)
    assert var_5.level == {'test_module': 0, 'test_module.func': 0}
    assert var_5.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_5.docstring == {'test_module': 'Test module docstring.'}
    assert var_5.imp == {'test_module': {*()}}
    assert var_5.root == {'test_module': 'test_module', 'test_module.CONST': 'test_module', 'test_module.func': 'test_module'}
    assert var_5.alias == {'test_module.CONST': '42'}
    assert var_5.const == {'test_module.CONST': 'int'}
    var_8 = var_5.compile()
    assert var_8 == '## Module `test_module`\n<a id="test_module"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONST` | `int` |\n\nTest module docstring.\n\n### func()\n\n*Full name:* `test_module.func`\n<a id="test_module-func"></a>\n\n| return |\n|:------:|\n| `None` |\n'
    var_9 = module_0.Parser()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc is False
    assert var_9.level == {}
    assert var_9.doc == {}
    assert var_9.docstring == {}
    assert var_9.imp == {}
    assert var_9.root == {}
    assert var_9.alias == {}
    assert var_9.const == {}
    var_10 = '\n"""Test module docstring."""\nfrom typing import List\nx: List[int] = []\n'
    var_11 = var_9.parse(var_0, var_10)
    assert var_9.level == {'test_module': 0}
    assert var_9.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_9.docstring == {'test_module': 'Test module docstring.'}
    assert var_9.imp == {'test_module': {*()}}
    assert var_9.root == {'test_module': 'test_module'}
    assert var_9.alias == {'test_module.List': 'typing.List', 'test_module.x': '[]'}
    var_12 = var_9.compile()
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
    var_14 = '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n'
    var_15 = var_13.parse(var_0, var_14)
    assert var_13.level == {'test_module': 0, 'test_module._private_func': 0, 'test_module.MyClass': 0, 'test_module.MyClass.__init__': 0}
    assert var_13.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_13.docstring == {'test_module': 'Test module docstring.'}
    assert var_13.imp == {'test_module': {*()}}
    assert var_13.root == {'test_module': 'test_module', 'test_module._private_func': 'test_module', 'test_module.MyClass': 'test_module', 'test_module.MyClass.__init__': 'test_module'}
    var_16 = var_13.compile()
    assert var_16 == '## Module `test_module`\n<a id="test_module"></a>\n\nTest module docstring.\n\n### class MyClass\n\n*Full name:* `test_module.MyClass`\n<a id="test_module-myclass"></a>\n'
    var_17 = module_0.Parser()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'apimd.parser.Parser'
    assert var_17.link is True
    assert var_17.b_level == 1
    assert var_17.toc is False
    assert var_17.level == {}
    assert var_17.doc == {}
    assert var_17.docstring == {}
    assert var_17.imp == {}
    assert var_17.root == {}
    assert var_17.alias == {}
    assert var_17.const == {}
    var_18 = module_0.walk_body(var_3)
    var_19 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'
    var_20 = '9\tTTg]Jh.\\2NL'
    var_21 = var_17.parse(var_20, var_6)
    assert var_17.level == {'9\tTTg]Jh.\\2NL': 1, '9\tTTg]Jh.\\2NL.func': 1}
    assert var_17.doc == {'9\tTTg]Jh.\\2NL': '## Module `{}`\n<a id="{}"></a>\n\n', '9\tTTg]Jh.\\2NL.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_17.docstring == {'9\tTTg]Jh.\\2NL': 'Test module docstring.'}
    assert var_17.imp == {'9\tTTg]Jh.\\2NL': {*()}}
    assert var_17.root == {'9\tTTg]Jh.\\2NL': '9\tTTg]Jh.\\2NL', '9\tTTg]Jh.\\2NL.CONST': '9\tTTg]Jh.\\2NL', '9\tTTg]Jh.\\2NL.func': '9\tTTg]Jh.\\2NL'}
    assert var_17.alias == {'9\tTTg]Jh.\\2NL.CONST': '42'}
    assert var_17.const == {'9\tTTg]Jh.\\2NL.CONST': 'int'}
    var_22 = var_17.parse(var_0, var_19)
    assert var_17.level == {'9\tTTg]Jh.\\2NL': 1, '9\tTTg]Jh.\\2NL.func': 1, 'test_module': 0, 'test_module.public_func': 0, 'test_module._private_func': 0}
    assert var_17.doc == {'9\tTTg]Jh.\\2NL': '## Module `{}`\n<a id="{}"></a>\n\n', '9\tTTg]Jh.\\2NL.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', 'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', 'test_module._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_17.docstring == {'9\tTTg]Jh.\\2NL': 'Test module docstring.', 'test_module': 'Test module docstring.'}
    assert var_17.imp == {'9\tTTg]Jh.\\2NL': {*()}, 'test_module': {'test_module.public_func'}}
    assert var_17.root == {'9\tTTg]Jh.\\2NL': '9\tTTg]Jh.\\2NL', '9\tTTg]Jh.\\2NL.CONST': '9\tTTg]Jh.\\2NL', '9\tTTg]Jh.\\2NL.func': '9\tTTg]Jh.\\2NL', 'test_module': 'test_module', 'test_module.public_func': 'test_module', 'test_module._private_func': 'test_module'}
    assert var_17.alias == {'9\tTTg]Jh.\\2NL.CONST': '42', 'test_module.__all__': "['public_func']"}
    var_23 = var_17.compile()
    assert var_23 == '## Module `test_module`\n<a id="test_module"></a>\n\nTest module docstring.\n\n### public_func()\n\n*Full name:* `test_module.public_func`\n<a id="test_module-public_func"></a>\n\n| return |\n|:------:|\n| `None` |\n\n## Module `9\tTTg]Jh.\\2NL`\n<a id="9\tttg]jh-\\2nl"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONST` | `int` |\n\nTest module docstring.\n\n### func()\n\n*Full name:* `9\tTTg]Jh.\\2NL.func`\n<a id="9\tttg]jh-\\2nl-func"></a>\n\n| return |\n|:------:|\n| `None` |\n'
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
    var_25 = var_24.compile()
    assert var_25 == '\n'
    var_26 = module_0.Parser()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'apimd.parser.Parser'
    assert var_26.link is True
    assert var_26.b_level == 1
    assert var_26.toc is False
    assert var_26.level == {}
    assert var_26.doc == {}
    assert var_26.docstring == {}
    assert var_26.imp == {}
    assert var_26.root == {}
    assert var_26.alias == {}
    assert var_26.const == {}
    var_27 = '\n"""Test module docstring."""\nfrom typing import List\nIntList = List[int]\nx: IntList = []\n'
    var_28 = var_26.parse(var_0, var_27)
    assert var_26.level == {'test_module': 0}
    assert var_26.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_26.docstring == {'test_module': 'Test module docstring.'}
    assert var_26.imp == {'test_module': {*()}}
    assert var_26.root == {'test_module': 'test_module'}
    assert var_26.alias == {'test_module.List': 'typing.List', 'test_module.IntList': 'List[int]', 'test_module.x': '[]'}
    var_29 = var_26.compile()
    assert var_29 == '\n'

def test_case_57():
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
    var_2 = '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'
    var_3 = var_1.parse(var_0, var_2)
    assert var_1.level == {'test_module': 0, 'test_module.MyClass': 0, 'test_module.MyClass.method': 0}
    assert var_1.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n', 'test_module.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\n'}
    assert var_1.docstring == {'test_module': 'Test module docstring.', 'test_module.MyClass': 'Class docstring.', 'test_module.MyClass.method': 'Method docstring.'}
    assert var_1.imp == {'test_module': {*()}}
    assert var_1.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module', 'test_module.MyClass.method': 'test_module'}
    var_4 = '#'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True
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
    var_7 = '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'
    var_8 = var_6.parse(var_0, var_7)
    assert var_6.level == {'test_module': 0, 'test_module.func': 0}
    assert var_6.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_6.docstring == {'test_module': 'Test module docstring.'}
    assert var_6.imp == {'test_module': {*()}}
    assert var_6.root == {'test_module': 'test_module', 'test_module.CONST': 'test_module', 'test_module.func': 'test_module'}
    assert var_6.alias == {'test_module.CONST': '42'}
    assert var_6.const == {'test_module.CONST': 'int'}
    var_9 = module_0.Parser()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc is False
    assert var_9.level == {}
    assert var_9.doc == {}
    assert var_9.docstring == {}
    assert var_9.imp == {}
    assert var_9.root == {}
    assert var_9.alias == {}
    assert var_9.const == {}
    var_10 = '\n"""Test module docstring."""\nfrom typin2 import List\nx: List[int] = []\n'
    var_11 = var_9.parse(var_0, var_10)
    assert var_9.level == {'test_module': 0}
    assert var_9.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_9.docstring == {'test_module': 'Test module docstring.'}
    assert var_9.imp == {'test_module': {*()}}
    assert var_9.root == {'test_module': 'test_module'}
    assert var_9.alias == {'test_module.List': 'typin2.List', 'test_module.x': '[]'}
    var_12 = module_0.Parser()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'apimd.parser.Parser'
    assert var_12.link is True
    assert var_12.b_level == 1
    assert var_12.toc is False
    assert var_12.level == {}
    assert var_12.doc == {}
    assert var_12.docstring == {}
    assert var_12.imp == {}
    assert var_12.root == {}
    assert var_12.alias == {}
    assert var_12.const == {}
    var_13 = '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n'
    var_14 = var_12.compile()
    assert var_14 == '\n'
    var_15 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'
    var_16 = var_6.parse(var_0, var_15)
    assert var_6.level == {'test_module': 0, 'test_module.func': 0, 'test_module.public_func': 0, 'test_module._private_func': 0}
    assert var_6.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', 'test_module.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', 'test_module._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_6.imp == {'test_module': {'test_module.public_func'}}
    assert var_6.root == {'test_module': 'test_module', 'test_module.CONST': 'test_module', 'test_module.func': 'test_module', 'test_module.public_func': 'test_module', 'test_module._private_func': 'test_module'}
    assert var_6.alias == {'test_module.CONST': '42', 'test_module.__all__': "['public_func']"}
    var_17 = var_6.compile()
    assert var_17 == '## Module `test_module`\n<a id="test_module"></a>\n\nTest module docstring.\n\n### public_func()\n\n*Full name:* `test_module.public_func`\n<a id="test_module-public_func"></a>\n\n| return |\n|:------:|\n| `None` |\n'
    var_18 = module_0.Parser()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'apimd.parser.Parser'
    assert var_18.link is True
    assert var_18.b_level == 1
    assert var_18.toc is False
    assert var_18.level == {}
    assert var_18.doc == {}
    assert var_18.docstring == {}
    assert var_18.imp == {}
    assert var_18.root == {}
    assert var_18.alias == {}
    assert var_18.const == {}
    var_19 = var_18.parse(var_0, var_13)
    assert var_18.level == {'test_module': 0, 'test_module._private_func': 0, 'test_module.MyClass': 0, 'test_module.MyClass.__init__': 0}
    assert var_18.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_18.docstring == {'test_module': 'Test module docstring.'}
    assert var_18.imp == {'test_module': {*()}}
    assert var_18.root == {'test_module': 'test_module', 'test_module._private_func': 'test_module', 'test_module.MyClass': 'test_module', 'test_module.MyClass.__init__': 'test_module'}
    var_20 = var_18.compile()
    assert var_20 == '## Module `test_module`\n<a id="test_module"></a>\n\nTest module docstring.\n\n### class MyClass\n\n*Full name:* `test_module.MyClass`\n<a id="test_module-myclass"></a>\n'
    var_21 = module_0.Parser()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'apimd.parser.Parser'
    assert var_21.link is True
    assert var_21.b_level == 1
    assert var_21.toc is False
    assert var_21.level == {}
    assert var_21.doc == {}
    assert var_21.docstring == {}
    assert var_21.imp == {}
    assert var_21.root == {}
    assert var_21.alias == {}
    assert var_21.const == {}
    var_22 = var_1.__repr__()
    assert var_22 == 'Parser(link=True, b_level=1, toc=False, level={\'test_module\': 0, \'test_module.MyClass\': 0, \'test_module.MyClass.method\': 0}, doc={\'test_module\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'test_module.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Members | Type |\\n|:-------:|:----:|\\n| `x` | `int` |\\n\\n\', \'test_module.MyClass.method\': \'#### MyClass.method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | a | return |\\n|:----:|:---:|:------:|\\n| `Self` | `str` | `None` |\\n\\n\'}, docstring={\'test_module\': \'Test module docstring.\', \'test_module.MyClass\': \'Class docstring.\', \'test_module.MyClass.method\': \'Method docstring.\'}, imp={\'test_module\': set()}, root={\'test_module\': \'test_module\', \'test_module.MyClass\': \'test_module\', \'test_module.MyClass.method\': \'test_module\'}, alias={}, const={})'

def test_case_58():
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
    var_3 = var_0.compile()
    assert var_3 == '\n'
    var_4 = '\n"""Test module docstring."""\ndef func(a: int) -> str:\n    """Function docstring."""\n    return str(a)\n'
    var_5 = var_0.parse(var_1, var_4)
    assert var_0.level == {'test_module': 0, 'test_module.func': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | return |\n|:---:|:------:|\n| `int` | `str` |\n\n'}
    assert var_0.docstring == {'test_module': 'Test module docstring.', 'test_module.func': 'Function docstring.'}
    assert var_0.root == {'test_module': 'test_module', 'test_module.func': 'test_module'}
    var_6 = var_0.compile()
    assert var_6 == '## Module `test_module`\n<a id="test_module"></a>\n\nTest module docstring.\n\n### func()\n\n*Full name:* `test_module.func`\n<a id="test_module-func"></a>\n\n| a | return |\n|:---:|:------:|\n| `int` | `str` |\n\nFunction docstring.\n'
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
    var_8 = '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'
    var_9 = var_7.parse(var_1, var_8)
    assert var_7.level == {'test_module': 0, 'test_module.MyClass': 0, 'test_module.MyClass.method': 0}
    assert var_7.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n', 'test_module.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\n'}
    assert var_7.docstring == {'test_module': 'Test module docstring.', 'test_module.MyClass': 'Class docstring.', 'test_module.MyClass.method': 'Method docstring.'}
    assert var_7.imp == {'test_module': {*()}}
    assert var_7.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module', 'test_module.MyClass.method': 'test_module'}
    var_10 = 'R+Z*Zk+EVyp\ni:jJQ'
    var_11 = module_0.is_public_family(var_10)
    assert var_11 is True
    var_12 = module_0.Parser()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'apimd.parser.Parser'
    assert var_12.link is True
    assert var_12.b_level == 1
    assert var_12.toc is False
    assert var_12.level == {}
    assert var_12.doc == {}
    assert var_12.docstring == {}
    assert var_12.imp == {}
    assert var_12.root == {}
    assert var_12.alias == {}
    assert var_12.const == {}
    var_13 = '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'
    var_14 = var_12.parse(var_1, var_13)
    assert var_12.level == {'test_module': 0, 'test_module.func': 0}
    assert var_12.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_12.docstring == {'test_module': 'Test module docstring.'}
    assert var_12.imp == {'test_module': {*()}}
    assert var_12.root == {'test_module': 'test_module', 'test_module.CONST': 'test_module', 'test_module.func': 'test_module'}
    assert var_12.alias == {'test_module.CONST': '42'}
    assert var_12.const == {'test_module.CONST': 'int'}
    var_15 = var_12.compile()
    assert var_15 == '## Module `test_module`\n<a id="test_module"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONST` | `int` |\n\nTest module docstring.\n\n### func()\n\n*Full name:* `test_module.func`\n<a id="test_module-func"></a>\n\n| return |\n|:------:|\n| `None` |\n'
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
    var_17 = '\n"""Test module docstring."""\nfrom typin2 import List\nx: List[int] = []\n'
    var_18 = var_16.parse(var_1, var_17)
    assert var_16.level == {'test_module': 0}
    assert var_16.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_16.docstring == {'test_module': 'Test module docstring.'}
    assert var_16.imp == {'test_module': {*()}}
    assert var_16.root == {'test_module': 'test_module'}
    assert var_16.alias == {'test_module.List': 'typin2.List', 'test_module.x': '[]'}
    var_19 = var_16.compile()
    assert var_19 == '\n'
    var_20 = module_0.Parser()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'apimd.parser.Parser'
    assert var_20.link is True
    assert var_20.b_level == 1
    assert var_20.toc is False
    assert var_20.level == {}
    assert var_20.doc == {}
    assert var_20.docstring == {}
    assert var_20.imp == {}
    assert var_20.root == {}
    assert var_20.alias == {}
    assert var_20.const == {}
    var_21 = var_20.compile()
    assert var_21 == '\n'
    var_22 = module_0.Parser()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'apimd.parser.Parser'
    assert var_22.link is True
    assert var_22.b_level == 1
    assert var_22.toc is False
    assert var_22.level == {}
    assert var_22.doc == {}
    assert var_22.docstring == {}
    assert var_22.imp == {}
    assert var_22.root == {}
    assert var_22.alias == {}
    assert var_22.const == {}
    var_23 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'
    var_24 = var_22.parse(var_1, var_23)
    assert var_22.level == {'test_module': 0, 'test_module.public_func': 0, 'test_module._private_func': 0}
    assert var_22.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', 'test_module._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_22.docstring == {'test_module': 'Test module docstring.'}
    assert var_22.imp == {'test_module': {'test_module.public_func'}}
    assert var_22.root == {'test_module': 'test_module', 'test_module.public_func': 'test_module', 'test_module._private_func': 'test_module'}
    assert var_22.alias == {'test_module.__all__': "['public_func']"}
    var_25 = var_22.compile()
    assert var_25 == '## Module `test_module`\n<a id="test_module"></a>\n\nTest module docstring.\n\n### public_func()\n\n*Full name:* `test_module.public_func`\n<a id="test_module-public_func"></a>\n\n| return |\n|:------:|\n| `None` |\n'
    var_26 = module_0.Parser()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'apimd.parser.Parser'
    assert var_26.link is True
    assert var_26.b_level == 1
    assert var_26.toc is False
    assert var_26.level == {}
    assert var_26.doc == {}
    assert var_26.docstring == {}
    assert var_26.imp == {}
    assert var_26.root == {}
    assert var_26.alias == {}
    assert var_26.const == {}
    var_27 = var_26.parse(var_1, var_10)
    assert var_26.level == {'test_module': 0}
    assert var_26.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_26.imp == {'test_module': {*()}}
    assert var_26.root == {'test_module': 'test_module'}
    var_28 = 's:7\x0c/K$Tw&f7wf?e.MH'
    var_29 = var_22.load_docstring(var_28, var_5)
    var_30 = var_26.compile()
    assert var_30 == '\n'
    var_31 = module_0.Parser()
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'apimd.parser.Parser'
    assert var_31.link is True
    assert var_31.b_level == 1
    assert var_31.toc is False
    assert var_31.level == {}
    assert var_31.doc == {}
    assert var_31.docstring == {}
    assert var_31.imp == {}
    assert var_31.root == {}
    assert var_31.alias == {}
    assert var_31.const == {}
    var_32 = '\n"""Test module docstring."""\nfrom typing import List\nIntList = List[int]\nx: IntList = []\n'
    var_33 = var_31.parse(var_1, var_32)
    assert var_31.level == {'test_module': 0}
    assert var_31.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_31.docstring == {'test_module': 'Test module docstring.'}
    assert var_31.imp == {'test_module': {*()}}
    assert var_31.root == {'test_module': 'test_module'}
    assert var_31.alias == {'test_module.List': 'typing.List', 'test_module.IntList': 'List[int]', 'test_module.x': '[]'}
    var_34 = var_0.__repr__()
    assert var_34 == 'Parser(link=True, b_level=1, toc=False, level={\'test_module\': 0, \'test_module.func\': 0}, doc={\'test_module\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'test_module.func\': \'### func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| a | return |\\n|:---:|:------:|\\n| `int` | `str` |\\n\\n\'}, docstring={\'test_module\': \'Test module docstring.\', \'test_module.func\': \'Function docstring.\'}, imp={\'test_module\': set()}, root={\'test_module\': \'test_module\', \'test_module.func\': \'test_module\'}, alias={}, const={})'

@pytest.mark.xfail(strict=True)
def test_case_59():
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
    var_1 = var_0.compile()
    assert var_1 == '\n'
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
    var_3 = var_0.__repr__()
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
    var_5 = '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'
    var_6 = var_4.parse(var_1, var_5)
    assert var_4.level == {'\n': 0, '\n.func': 0}
    assert var_4.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_4.docstring == {'\n': 'Test module docstring.'}
    assert var_4.imp == {'\n': {*()}}
    assert var_4.root == {'\n': '\n', '\n.CONST': '\n', '\n.func': '\n'}
    assert var_4.alias == {'\n.CONST': '42'}
    assert var_4.const == {'\n.CONST': 'int'}
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
    var_8 = '\n"""Test module docstring."""\nfrom typin2 import List\nx: List[int] = []\n'
    var_9 = var_7.parse(var_1, var_8)
    assert var_7.level == {'\n': 0}
    assert var_7.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_7.docstring == {'\n': 'Test module docstring.'}
    assert var_7.imp == {'\n': {*()}}
    assert var_7.root == {'\n': '\n'}
    assert var_7.alias == {'\n.List': 'typin2.List', '\n.x': '[]'}
    var_10 = var_7.compile()
    assert var_10 == '\n'
    var_11 = var_2.__repr__()
    assert var_11 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_12 = module_2.getdoc(var_9)
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
    var_14 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'
    var_15 = var_13.parse(var_5, var_14)
    assert var_13.level == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 1, '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': 1, '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': 1}
    assert var_13.doc == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_13.docstring == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 'Test module docstring.'}
    assert var_13.imp == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func'}}
    assert var_13.root == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'}
    assert var_13.alias == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.__all__': "['public_func']"}
    var_16 = var_13.compile()
    assert var_16 == '## Module `\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n`\n<a id="\n"""test module docstring-"""\nconst = 42\ndef func() -> none:\n    pass\n"></a>\n\nTest module docstring.\n\n### public_func()\n\n*Full name:* `\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func`\n<a id="\n"""test module docstring-"""\nconst = 42\ndef func() -> none:\n    pass\n-public_func"></a>\n\n| return |\n|:------:|\n| `None` |\n'
    var_17 = module_0.Parser()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'apimd.parser.Parser'
    assert var_17.link is True
    assert var_17.b_level == 1
    assert var_17.toc is False
    assert var_17.level == {}
    assert var_17.doc == {}
    assert var_17.docstring == {}
    assert var_17.imp == {}
    assert var_17.root == {}
    assert var_17.alias == {}
    assert var_17.const == {}
    var_18 = '\n"""Test module docstring."""\nfrom enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_19 = var_17.parse(var_14, var_18)
    assert var_17.level == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': 1, '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': 1}
    assert var_17.doc == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| RED |\n| GREEN |\n| BLUE |\n\n'}
    assert var_17.docstring == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': 'Test module docstring.'}
    assert var_17.imp == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': {*()}}
    assert var_17.root == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'}
    assert var_17.alias == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Enum': 'enum.Enum'}
    var_20 = var_13.load_docstring(var_10, var_19)
    var_21 = module_0.Parser()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'apimd.parser.Parser'
    assert var_21.link is True
    assert var_21.b_level == 1
    assert var_21.toc is False
    assert var_21.level == {}
    assert var_21.doc == {}
    assert var_21.docstring == {}
    assert var_21.imp == {}
    assert var_21.root == {}
    assert var_21.alias == {}
    assert var_21.const == {}
    var_22 = '\n"""Test module docstring."""\nfrom typing import List\nInt,ist = List[int]\nx: IntLizt = []\n'
    var_23 = var_21.parse(var_5, var_22)
    assert var_21.level == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 1}
    assert var_21.doc == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_21.docstring == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 'Test module docstring.'}
    assert var_21.imp == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': {*()}}
    assert var_21.root == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'}
    assert var_21.alias == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.List': 'typing.List', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.x': '[]'}
    var_11.is_public(var_11)

def test_case_60():
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
    var_1 = 'R+Z*Zk+EVyp\ni:jJQ'
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
    var_3 = var_2.compile()
    assert var_3 == '\n'
    var_4 = var_2.compile()
    assert var_4 == '\n'
    var_5 = '\n"""Test module docstring."""\ndef _private_func() -> None:\n    pass\nclass MyClass:\n    _private_attr: int = 1\n    def __init__(self) -> None:\n        pass\n'
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
    var_7 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n'
    var_8 = var_6.compile()
    assert var_8 == '\n'
    var_9 = '\n"""Test module docstring."""\nfrom enum import Enup\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_10 = var_6.parse(var_7, var_9)
    assert var_6.level == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n': 1, '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n.Color': 1}
    assert var_6.doc == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `Enum` |\n\n| Members | Type |\n|:-------:|:----:|\n| `BLUE` | `int` |\n| `GREEN` | `int` |\n| `RED` | `int` |\n\n'}
    assert var_6.docstring == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n': 'Test module docstring.'}
    assert var_6.imp == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n': {*()}}
    assert var_6.root == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n': '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n.Color': '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n'}
    assert var_6.alias == {'\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n ^  pass\n.Enup': 'enum.Enup'}
    var_11 = var_0.load_docstring(var_1, var_10)
    var_12 = {}
    var_13 = module_0.Parser(var_11, docstring=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'apimd.parser.Parser'
    assert var_13.link is None
    assert var_13.b_level == 1
    assert var_13.toc is False
    assert var_13.level == {}
    assert var_13.doc == {}
    assert var_13.docstring == {}
    assert var_13.imp == {}
    assert var_13.root == {}
    assert var_13.alias == {}
    assert var_13.const == {}
    var_14 = var_6.__repr__()
    assert var_14 == 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n\': 1, \'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n.Color\': 1}, doc={\'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n.Color\': \'### class Color\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Bases |\\n|:-----:|\\n| `Enum` |\\n\\n| Members | Type |\\n|:-------:|:----:|\\n| `BLUE` | `int` |\\n| `GREEN` | `int` |\\n| `RED` | `int` |\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n\': \'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n\', \'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n.Color\': \'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n\'}, alias={\'\\n"""Test module docstring."""\\n__all__ = [\\\'public_func\\\']\\ndef public_func() -> None:\\n    pass\\ndef _private_func() -> None:\\n ^  pass\\n.Enup\': \'enum.Enup\'}, const={})'
    var_15 = 'v%Rr\\W@!W3'
    var_16 = var_13.parse(var_15, var_5)
    assert var_13.level == {'v%Rr\\W@!W3': 0, 'v%Rr\\W@!W3._private_func': 0, 'v%Rr\\W@!W3.MyClass': 0, 'v%Rr\\W@!W3.MyClass.__init__': 0}
    assert var_13.doc == {'v%Rr\\W@!W3': '## Module `{}`\n\n', 'v%Rr\\W@!W3._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n\n| return |\n|:------:|\n| `None` |\n\n', 'v%Rr\\W@!W3.MyClass': '### class MyClass\n\n*Full name:* `{}`\n\n', 'v%Rr\\W@!W3.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_13.docstring == {'v%Rr\\W@!W3': 'Test module docstring.'}
    assert var_13.imp == {'v%Rr\\W@!W3': {*()}}
    assert var_13.root == {'v%Rr\\W@!W3': 'v%Rr\\W@!W3', 'v%Rr\\W@!W3._private_func': 'v%Rr\\W@!W3', 'v%Rr\\W@!W3.MyClass': 'v%Rr\\W@!W3', 'v%Rr\\W@!W3.MyClass.__init__': 'v%Rr\\W@!W3'}
    var_17 = var_0.__repr__()
    assert var_17 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'

def test_case_61():
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
    var_1 = var_0.compile()
    assert var_1 == '\n'
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
    var_3 = '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'
    var_4 = var_2.parse(var_1, var_3)
    assert var_2.level == {'\n': 0, '\n.MyClass': 0, '\n.MyClass.method': 0}
    assert var_2.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n', '\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\n'}
    assert var_2.docstring == {'\n': 'Test module docstring.', '\n.MyClass': 'Class docstring.', '\n.MyClass.method': 'Method docstring.'}
    assert var_2.imp == {'\n': {*()}}
    assert var_2.root == {'\n': '\n', '\n.MyClass': '\n', '\n.MyClass.method': '\n'}
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
    var_6 = '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'
    var_7 = var_5.parse(var_1, var_6)
    assert var_5.level == {'\n': 0, '\n.func': 0}
    assert var_5.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_5.docstring == {'\n': 'Test module docstring.'}
    assert var_5.imp == {'\n': {*()}}
    assert var_5.root == {'\n': '\n', '\n.CONST': '\n', '\n.func': '\n'}
    assert var_5.alias == {'\n.CONST': '42'}
    assert var_5.const == {'\n.CONST': 'int'}
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
    var_9 = var_5.__eq__(var_4)
    var_10 = var_8.compile()
    assert var_10 == '\n'
    var_11 = '\n"""Test module docstring."""\nfrom typin2 import List\nx: List[int] = []\n'
    var_12 = var_8.parse(var_1, var_11)
    assert var_8.level == {'\n': 0}
    assert var_8.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_8.docstring == {'\n': 'Test module docstring.'}
    assert var_8.imp == {'\n': {*()}}
    assert var_8.root == {'\n': '\n'}
    assert var_8.alias == {'\n.List': 'typin2.List', '\n.x': '[]'}
    var_13 = var_8.compile()
    assert var_13 == '\n'
    var_14 = module_1.stmt()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.stmt'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.stmt.end_lineno is None
    assert module_1.stmt.end_col_offset is None
    var_15 = module_2.getdoc(var_12)
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
    var_16 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'
    var_17 = var_8.parse(var_6, var_16)
    assert var_8.level == {'\n': 0, '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 1, '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': 1, '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': 1}
    assert var_8.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_8.docstring == {'\n': 'Test module docstring.', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 'Test module docstring.'}
    assert var_8.imp == {'\n': {*()}, '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func'}}
    assert var_8.root == {'\n': '\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'}
    assert var_8.alias == {'\n.List': 'typin2.List', '\n.x': '[]', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.__all__': "['public_func']"}
    var_18 = var_2.compile()
    assert var_18 == '## Module `\n`\n<a id="\n"></a>\n\nTest module docstring.\n\n### class MyClass\n\n*Full name:* `\n.MyClass`\n<a id="\n-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\n.MyClass.method`\n<a id="\n-myclass-method"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\nMethod docstring.\n'
    var_19 = module_0.is_magic(var_10)
    assert var_19 is False
    var_20 = '\n"""Test module docstring."""\nfrom enum import Enup\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_21 = var_5.parse(var_16, var_20)
    assert var_5.level == {'\n': 0, '\n.func': 0, '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': 1, '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': 1}
    assert var_5.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `Enum` |\n\n| Members | Type |\n|:-------:|:----:|\n| `BLUE` | `int` |\n| `GREEN` | `int` |\n| `RED` | `int` |\n\n'}
    assert var_5.docstring == {'\n': 'Test module docstring.', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': 'Test module docstring.'}
    assert var_5.imp == {'\n': {*()}, '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': {*()}}
    assert var_5.root == {'\n': '\n', '\n.CONST': '\n', '\n.func': '\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'}
    assert var_5.alias == {'\n.CONST': '42', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Enup': 'enum.Enup'}
    var_22 = var_8.load_docstring(var_13, var_21)
    var_23 = module_0.Parser()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'apimd.parser.Parser'
    assert var_23.link is True
    assert var_23.b_level == 1
    assert var_23.toc is False
    assert var_23.level == {}
    assert var_23.doc == {}
    assert var_23.docstring == {}
    assert var_23.imp == {}
    assert var_23.root == {}
    assert var_23.alias == {}
    assert var_23.const == {}
    var_24 = '\n"""Test module docstring."""\nfrom typing import List\nIntList = List[int]\nx: IntList = []\n'
    var_25 = var_0.compile()
    assert var_25 == '\n'
    var_26 = var_23.parse(var_6, var_24)
    assert var_23.level == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 1}
    assert var_23.doc == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_23.docstring == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 'Test module docstring.'}
    assert var_23.imp == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': {*()}}
    assert var_23.root == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'}
    assert var_23.alias == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.List': 'typing.List', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.IntList': 'List[int]', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.x': '[]'}
    var_27 = var_9.__repr__()
    assert var_27 == 'NotImplemented'
    var_28 = var_5.parse(var_25, var_6)
    var_29 = var_23.__repr__()
    assert var_29 == 'Parser(link=True, b_level=1, toc=False, level={\'\\n"""Test module docstring."""\\nCONST = 42\\ndef func() -> None:\\n    pass\\n\': 1}, doc={\'\\n"""Test module docstring."""\\nCONST = 42\\ndef func() -> None:\\n    pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={\'\\n"""Test module docstring."""\\nCONST = 42\\ndef func() -> None:\\n    pass\\n\': \'Test module docstring.\'}, imp={\'\\n"""Test module docstring."""\\nCONST = 42\\ndef func() -> None:\\n    pass\\n\': set()}, root={\'\\n"""Test module docstring."""\\nCONST = 42\\ndef func() -> None:\\n    pass\\n\': \'\\n"""Test module docstring."""\\nCONST = 42\\ndef func() -> None:\\n    pass\\n\'}, alias={\'\\n"""Test module docstring."""\\nCONST = 42\\ndef func() -> None:\\n    pass\\n.List\': \'typing.List\', \'\\n"""Test module docstring."""\\nCONST = 42\\ndef func() -> None:\\n    pass\\n.IntList\': \'List[int]\', \'\\n"""Test module docstring."""\\nCONST = 42\\ndef func() -> None:\\n    pass\\n.x\': \'[]\'}, const={})'

def test_case_62():
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
    var_1 = var_0.compile()
    assert var_1 == '\n'
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
    var_3 = '\n"""Test module docstring."""\nclass MyClass:\n    """Class docstring."""\n    x: int = 1\n    def method(self, a: str) -> None:\n        """Method docstring."""\n        pass\n'
    var_4 = var_2.parse(var_1, var_3)
    assert var_2.level == {'\n': 0, '\n.MyClass': 0, '\n.MyClass.method': 0}
    assert var_2.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n', '\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | a | return |\n|:----:|:---:|:------:|\n| `Self` | `str` | `None` |\n\n'}
    assert var_2.docstring == {'\n': 'Test module docstring.', '\n.MyClass': 'Class docstring.', '\n.MyClass.method': 'Method docstring.'}
    assert var_2.imp == {'\n': {*()}}
    assert var_2.root == {'\n': '\n', '\n.MyClass': '\n', '\n.MyClass.method': '\n'}
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
    var_6 = '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'
    var_7 = '=2Km'
    var_8 = var_2.globals(var_7, var_4)
    var_9 = var_5.parse(var_1, var_6)
    assert var_5.level == {'\n': 0, '\n.func': 0}
    assert var_5.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_5.docstring == {'\n': 'Test module docstring.'}
    assert var_5.imp == {'\n': {*()}}
    assert var_5.root == {'\n': '\n', '\n.CONST': '\n', '\n.func': '\n'}
    assert var_5.alias == {'\n.CONST': '42'}
    assert var_5.const == {'\n.CONST': 'int'}
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
    var_11 = var_5.__eq__(var_4)
    var_12 = var_10.compile()
    assert var_12 == '\n'
    var_13 = '\n"""Test module docstring."""\nfrom typin2 import List\nx: List[int] = []\n'
    var_14 = var_10.parse(var_1, var_13)
    assert var_10.level == {'\n': 0}
    assert var_10.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_10.docstring == {'\n': 'Test module docstring.'}
    assert var_10.imp == {'\n': {*()}}
    assert var_10.root == {'\n': '\n'}
    assert var_10.alias == {'\n.List': 'typin2.List', '\n.x': '[]'}
    var_15 = var_10.compile()
    assert var_15 == '\n'
    var_16 = var_2.__repr__()
    assert var_16 == 'Parser(link=True, b_level=1, toc=False, level={\'\\n\': 0, \'\\n.MyClass\': 0, \'\\n.MyClass.method\': 0}, doc={\'\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Members | Type |\\n|:-------:|:----:|\\n| `x` | `int` |\\n\\n\', \'\\n.MyClass.method\': \'#### MyClass.method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | a | return |\\n|:----:|:---:|:------:|\\n| `Self` | `str` | `None` |\\n\\n\'}, docstring={\'\\n\': \'Test module docstring.\', \'\\n.MyClass\': \'Class docstring.\', \'\\n.MyClass.method\': \'Method docstring.\'}, imp={\'\\n\': set()}, root={\'\\n\': \'\\n\', \'\\n.MyClass\': \'\\n\', \'\\n.MyClass.method\': \'\\n\'}, alias={}, const={})'
    var_17 = module_2.getdoc(var_14)
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
    var_18 = module_0.Parser()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'apimd.parser.Parser'
    assert var_18.link is True
    assert var_18.b_level == 1
    assert var_18.toc is False
    assert var_18.level == {}
    assert var_18.doc == {}
    assert var_18.docstring == {}
    assert var_18.imp == {}
    assert var_18.root == {}
    assert var_18.alias == {}
    assert var_18.const == {}
    var_19 = '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'
    var_20 = var_18.parse(var_6, var_19)
    assert var_18.level == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 1, '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': 1, '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': 1}
    assert var_18.doc == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n'}
    assert var_18.docstring == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 'Test module docstring.'}
    assert var_18.imp == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func'}}
    assert var_18.root == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'}
    assert var_18.alias == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.__all__': "['public_func']"}
    var_21 = var_18.compile()
    assert var_21 == '## Module `\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n`\n<a id="\n"""test module docstring-"""\nconst = 42\ndef func() -> none:\n    pass\n"></a>\n\nTest module docstring.\n\n### public_func()\n\n*Full name:* `\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func`\n<a id="\n"""test module docstring-"""\nconst = 42\ndef func() -> none:\n    pass\n-public_func"></a>\n\n| return |\n|:------:|\n| `None` |\n'
    var_22 = var_11.__repr__()
    assert var_22 == 'NotImplemented'
    var_23 = '\n"""Test module docstring."""\nfrom enum import Enup\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n'
    var_24 = var_18.parse(var_19, var_23)
    assert var_18.level == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 1, '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': 1, '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': 1, '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': 1, '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': 1}
    assert var_18.doc == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `None` |\n\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': '### class Color\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `Enum` |\n\n| Members | Type |\n|:-------:|:----:|\n| `BLUE` | `int` |\n| `GREEN` | `int` |\n| `RED` | `int` |\n\n'}
    assert var_18.docstring == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 'Test module docstring.', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': 'Test module docstring.'}
    assert var_18.imp == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func'}, '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': {*()}}
    assert var_18.root == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.public_func': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n._private_func': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n': '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n', '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Color': '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n'}
    assert var_18.alias == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.__all__': "['public_func']", '\n"""Test module docstring."""\n__all__ = [\'public_func\']\ndef public_func() -> None:\n    pass\ndef _private_func() -> None:\n    pass\n.Enup': 'enum.Enup'}
    var_25 = var_18.load_docstring(var_15, var_24)
    var_26 = module_0.Parser()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'apimd.parser.Parser'
    assert var_26.link is True
    assert var_26.b_level == 1
    assert var_26.toc is False
    assert var_26.level == {}
    assert var_26.doc == {}
    assert var_26.docstring == {}
    assert var_26.imp == {}
    assert var_26.root == {}
    assert var_26.alias == {}
    assert var_26.const == {}
    var_27 = '\n"""Test module docstring."""\nfrom typing import List\nInt=ist = List[int]\nx: IntList = []\n'
    var_28 = var_26.parse(var_6, var_27)
    assert var_26.level == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 1}
    assert var_26.doc == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_26.docstring == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': 'Test module docstring.'}
    assert var_26.imp == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': {*()}}
    assert var_26.root == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n': '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n'}
    assert var_26.alias == {'\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.List': 'typing.List', '\n"""Test module docstring."""\nCONST = 42\ndef func() -> None:\n    pass\n.x': '[]'}
    var_29 = var_10.__repr__()
    assert var_29 == 'Parser(link=True, b_level=1, toc=False, level={\'\\n\': 0}, doc={\'\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={\'\\n\': \'Test module docstring.\'}, imp={\'\\n\': set()}, root={\'\\n\': \'\\n\'}, alias={\'\\n.List\': \'typin2.List\', \'\\n.x\': \'[]\'}, const={})'