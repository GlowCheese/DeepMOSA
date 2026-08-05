# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.parser as module_0
import ast as module_1
import inspect as module_2
import dataclasses as module_3

def test_case_0():
    var_0 = '_om:n__.module.sb'
    var_1 = module_0.is_public_family(var_0)
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
    var_0 = 'module.submodule.class'
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
def test_case_2():
    var_0 = '/1sw:'
    var_1 = module_0.code(var_0)
    assert var_1 == '`/1sw:`'
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
    module_1.parse(var_2, type_comments=var_2, feature_version=var_1)

def test_case_3():
    var_0 = 'sub'
    var_1 = module_0.table(items=var_0)
    assert var_1 == '||\n||\n| s |\n| u |\n| b |\n\n'
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
def test_case_4():
    var_0 = ''
    var_1 = module_0.code(var_0)
    assert var_1 == ' '
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
    var_3 = module_0.const_type(var_2)
    assert var_3 == 'Any'
    var_4 = "Fj\rhVa'Ie_?&N`"
    var_5 = {var_3: var_4}
    var_6 = module_0.Resolver(var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Resolver'
    assert var_6.root is None
    assert var_6.alias == {'Any': "Fj\rhVa'Ie_?&N`"}
    assert var_6.self_ty == ''
    var_7 = '_xe'
    var_8 = module_0.is_public_family(var_7)
    assert var_8 is False
    var_9 = 'tcgp2\x0bhid'
    var_10 = module_0.esc_underscore(var_9)
    assert var_10 == 'tcgp2\x0bhid'
    var_11 = module_2.getdoc(var_2)
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
    var_12 = True
    var_13 = True
    var_14 = module_0.Parser(var_12, toc=var_13, alias=var_2)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'apimd.parser.Parser'
    assert var_14.link is True
    assert var_14.b_level == 1
    assert var_14.toc is True
    assert var_14.level == {}
    assert var_14.doc == {}
    assert var_14.docstring == {}
    assert var_14.imp == {}
    assert var_14.root == {}
    assert var_14.alias is None
    assert var_14.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    module_0.doctest(var_2)

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
    var_1 = None
    var_2 = var_0.load_docstring(var_1, var_1)

def test_case_7():
    pass

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    module_0.parent(var_0)

def test_case_9():
    var_0 = 'L;Ih79D:9f&>'
    var_1 = module_0.Resolver(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Resolver'
    assert var_1.root == 'L;Ih79D:9f&>'
    assert var_1.alias == 'L;Ih79D:9f&>'
    assert var_1.self_ty == 'L;Ih79D:9f&>'
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
    var_2 = module_0.esc_underscore(var_0)
    assert var_2 == 'L;Ih79D:9f&>'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'GY>`"x~A/f}R'
    var_1 = None
    var_2 = 'L;Ih79D:9f&>'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'GY>`"x~A/f}R'
    assert var_3.alias is None
    assert var_3.self_ty == 'L;Ih79D:9f&>'
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
    var_4 = '6:pu.b=]p"rhP?Fs5'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = 'cX#Q\x0c)}&11nq'
    var_8 = module_0.code(var_7)
    assert var_8 == '<code>cX#Q\x0c)}&11nq</code>'
    module_1.parse(var_6, var_6, feature_version=var_6)

def test_case_11():
    var_0 = 'mGx\x0b\nO^&PiVn*%C5VR'
    var_1 = module_0.esc_underscore(var_0)
    assert var_1 == 'mGx\x0b\nO^&PiVn*%C5VR'
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

def test_case_12():
    var_0 = '=\x0b_v4y\t'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '=\n_v4y\t'
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

def test_case_13():
    var_0 = None
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

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '+4 O\nFx'
    var_1 = 'Fd`:s :sgX`k\r"n(b'
    var_2 = [var_0, var_0, var_1, var_0]
    var_3 = None
    module_0.table(*var_2, items=var_3)

@pytest.mark.xfail(strict=True)
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
    var_1 = 'TFI]0\tM\r=\x0b2JO"`D'
    var_0.is_public(var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
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
    var_2 = -671
    var_3 = module_0.Parser(var_0, var_2, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is None
    assert var_3.b_level == -671
    assert var_3.toc is None
    assert var_3.level is None
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4 = '&Bz'
    var_5 = var_3.globals(var_4, var_0)
    var_6 = var_3.__repr__()
    assert var_6 == 'Parser(link=None, b_level=-671, toc=None, level=None, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_7 = '<6gD(o\t$'
    var_8 = var_3.__eq__(var_6)
    var_9 = var_3.compile()
    assert var_9 == '\n'
    var_10 = '42Q~pCi?2MK5J}\n'
    var_11 = var_3.globals(var_10, var_5)
    var_12 = module_0.parent(var_6)
    assert var_12 == 'Parser(link=None, b_level=-671, toc=None, level=None, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_13 = var_3.__eq__(var_5)
    var_14 = '^}U3QOqn7\r8P'
    var_15 = module_0.Resolver(var_14, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'apimd.parser.Resolver'
    assert var_15.root == '^}U3QOqn7\r8P'
    assert var_15.alias == '^}U3QOqn7\r8P'
    assert var_15.self_ty == ''
    var_16 = ';5cX,\x0b"Q<E"Q$\n'
    var_17 = module_0.is_public_family(var_16)
    assert var_17 is True
    var_18 = module_0.esc_underscore(var_7)
    assert var_18 == '<6gD(o\t$'
    module_0.table(*var_6, items=var_13)

def test_case_17():
    var_0 = '__main__.module.sub'
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

def test_case_18():
    var_0 = ''
    var_1 = module_0.code(var_0)
    assert var_1 == ' '
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
    var_3 = module_0.const_type(var_2)
    assert var_3 == 'Any'
    var_4 = None
    var_5 = module_0.walk_body(var_4)
    var_6 = module_3.field(compare=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'dataclasses.Field'
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
    var_7 = '^}U3QOqn7\r8P'
    var_8 = module_0.is_public_family(var_7)
    assert var_8 is True
    var_9 = module_0.doctest(var_7)
    assert var_9 == '^}U3QOqn7\n8P'

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
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
    var_2 = module_0.Resolver(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Resolver'
    assert var_2.root == 'Any'
    assert var_2.alias is None
    assert var_2.self_ty == ''
    var_3 = module_1.Tuple()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Tuple'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    module_0.const_type(var_3)

@pytest.mark.xfail(strict=True)
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
    var_1 = None
    var_0.imports(var_1, var_0)

def test_case_21():
    var_0 = None
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
    var_2 = module_0.walk_body(var_0)
    var_3 = module_3.field(compare=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'dataclasses.Field'
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
    var_4 = var_3.__repr__()
    assert var_4 == 'Field(name=None,type=None,default=<dataclasses._MISSING_TYPE object at 0x7eb0391bd450>,default_factory=<dataclasses._MISSING_TYPE object at 0x7eb0391bd450>,init=True,repr=True,hash=None,compare=None,metadata=mappingproxy({}),kw_only=<dataclasses._MISSING_TYPE object at 0x7eb0391bd450>,_field_type=None)'
    var_5 = 'L;Ih79D:9f&>'
    var_6 = module_0.Resolver(var_0, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Resolver'
    assert var_6.root is None
    assert var_6.alias == 'Field(name=None,type=None,default=<dataclasses._MISSING_TYPE object at 0x7eb0391bd450>,default_factory=<dataclasses._MISSING_TYPE object at 0x7eb0391bd450>,init=True,repr=True,hash=None,compare=None,metadata=mappingproxy({}),kw_only=<dataclasses._MISSING_TYPE object at 0x7eb0391bd450>,_field_type=None)'
    assert var_6.self_ty == ''
    var_7 = '6:pu.b=]p"rhP?Fs5'
    var_8 = module_0.is_public_family(var_5)
    assert var_8 is True
    var_9 = module_0.esc_underscore(var_4)
    assert var_9 == 'Field(name=None,type=None,default=<dataclasses.\\_MISSING\\_TYPE object at 0x7eb0391bd450>,default\\_factory=<dataclasses.\\_MISSING\\_TYPE object at 0x7eb0391bd450>,init=True,repr=True,hash=None,compare=None,metadata=mappingproxy({}),kw\\_only=<dataclasses.\\_MISSING\\_TYPE object at 0x7eb0391bd450>,\\_field\\_type=None)'
    var_10 = module_0.doctest(var_7)
    assert var_10 == '6:pu.b=]p"rhP?Fs5'

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = 'Nc|AO:JOewl]'
    var_2 = 'E'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_1.Dict(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Dict'
    assert var_4.Nc|AO:JOewl] is None
    assert var_4.E is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    module_0.const_type(var_4)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = -671
    var_2 = module_0.Parser(var_0, var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is None
    assert var_2.b_level == -671
    assert var_2.toc is None
    assert var_2.level is None
    assert var_2.doc == {}
    assert var_2.docstring == {}
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
    var_3 = '&Bz'
    var_2.parse(var_3, var_3)

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
    var_1 = var_0.globals(var_0, var_0)
    var_2 = var_0.__repr__()
    assert var_2 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_3 = var_0.load_docstring(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
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
    var_2 = -671
    var_3 = module_0.Parser(var_0, var_2, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is None
    assert var_3.b_level == -671
    assert var_3.toc is None
    assert var_3.level is None
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4 = '&Bz'
    var_5 = var_3.globals(var_4, var_0)
    var_6 = module_3.field(default_factory=var_2, metadata=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'dataclasses.Field'
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
    var_7 = var_3.__eq__(var_0)
    var_8 = module_0.Parser(b_level=var_1, toc=var_5, imp=var_5, root=var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is True
    assert var_8.b_level == 'Any'
    assert var_8.toc is None
    assert var_8.level == {}
    assert var_8.doc == {}
    assert var_8.docstring == {}
    assert var_8.imp is None
    assert var_8.root == 'Any'
    assert var_8.alias == {}
    assert var_8.const == {}
    var_8.api(var_6, var_0)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_2, var_2: var_0}
    var_4 = module_1.Name(*var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Name'
    assert var_4.id is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = [var_4]
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_7}
    var_9 = module_1.Constant(*var_6, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Constant'
    assert var_9.value == 'value'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_10 = []
    var_11 = 'targets'
    var_12 = 'value'
    var_13 = {var_11: var_5, var_12: var_9, var_2: var_0}
    var_14 = module_1.Assign(*var_10, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Assign'
    assert f'{type(var_14.targets).__module__}.{type(var_14.targets).__qualname__}' == 'builtins.list'
    assert len(var_14.targets) == 1
    assert f'{type(var_14.value).__module__}.{type(var_14.value).__qualname__}' == 'ast.Constant'
    assert var_14.id is None
    assert module_1.Assign.type_comment is None
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
    var_15.globals(var_7, var_14)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = 'VG%'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'VG%'
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
    var_2 = module_0.code(var_0)
    assert var_2 == '`VG%`'
    var_3 = 'o}X=<aE_6tl#Oh~h[D:'
    var_4 = module_0.doctest(var_3)
    assert var_4 == 'o}X=<aE_6tl#Oh~h[D:'
    var_5 = module_1.Call()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Call'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    module_0.const_type(var_5)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = None
    var_1 = 'VG%'
    var_2 = -680
    var_3 = {var_1: var_2, var_1: var_2}
    var_4 = ''
    var_5 = 'j(Cj&e\t>'
    var_6 = '8S#6uk&'
    var_7 = '8^w4s8'
    var_8 = 'S\x0b!zvJ54JT]37'
    var_9 = {var_1: var_4, var_5: var_6, var_7: var_8}
    var_10 = module_0.Parser(level=var_3, docstring=var_9, alias=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 1
    assert var_10.toc is False
    assert var_10.level == {'VG%': -680}
    assert var_10.doc == {}
    assert var_10.docstring == {'VG%': '', 'j(Cj&e\t>': '8S#6uk&', '8^w4s8': 'S\x0b!zvJ54JT]37'}
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == 'VG%'
    assert var_10.const == {}
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
    var_11 = 'U%8(VaT'
    var_12 = module_2.getdoc(var_0)
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
    var_13 = var_10.globals(var_11, var_12)
    var_14 = module_3.field(init=var_12, metadata=var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'dataclasses.Field'
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
    var_15 = var_12.__repr__()
    assert var_15 == 'None'
    var_16 = module_0.Parser(level=var_14, doc=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'apimd.parser.Parser'
    assert var_16.link is True
    assert var_16.b_level == 1
    assert var_16.toc is False
    assert f'{type(var_16.level).__module__}.{type(var_16.level).__qualname__}' == 'dataclasses.Field'
    assert var_16.doc == 'None'
    assert var_16.docstring == {}
    assert var_16.imp == {}
    assert var_16.root == {}
    assert var_16.alias == {}
    assert var_16.const == {}
    var_16.compile()

def test_case_30():
    var_0 = None
    var_1 = -671
    var_2 = module_0.Parser(var_0, var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is None
    assert var_2.b_level == -671
    assert var_2.toc is None
    assert var_2.level is None
    assert var_2.doc == {}
    assert var_2.docstring == {}
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
    var_3 = '&Bz'
    var_4 = var_2.globals(var_3, var_0)
    var_5 = module_3.field(default_factory=var_1, metadata=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'dataclasses.Field'
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
    var_6 = var_2.__repr__()
    assert var_6 == 'Parser(link=None, b_level=-671, toc=None, level=None, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_7 = '<6D(o\t$'
    var_8 = module_0.Parser(b_level=var_6, toc=var_4, imp=var_4, root=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is True
    assert var_8.b_level == 'Parser(link=None, b_level=-671, toc=None, level=None, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert var_8.toc is None
    assert var_8.level == {}
    assert var_8.doc == {}
    assert var_8.docstring == {}
    assert var_8.imp is None
    assert var_8.root == 'Parser(link=None, b_level=-671, toc=None, level=None, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert var_8.alias == {}
    assert var_8.const == {}
    var_9 = var_2.compile()
    assert var_9 == '\n'
    var_10 = module_3.field(hash=var_0, compare=var_4, metadata=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'dataclasses.Field'
    var_11 = module_0.Resolver(var_7, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'apimd.parser.Resolver'
    assert var_11.root == '<6D(o\t$'
    assert f'{type(var_11.alias).__module__}.{type(var_11.alias).__qualname__}' == 'dataclasses.Field'
    assert var_11.self_ty == ''
    var_12 = ')Rna-|g'
    var_13 = module_0.esc_underscore(var_12)
    assert var_13 == ')Rna-|g'
    var_14 = module_2.getdoc(var_8)
    assert var_14 == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
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
    var_15 = module_0.Parser(var_10, toc=var_0, doc=var_6, root=var_5)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'apimd.parser.Parser'
    assert f'{type(var_15.link).__module__}.{type(var_15.link).__qualname__}' == 'dataclasses.Field'
    assert var_15.b_level == 1
    assert var_15.toc is None
    assert var_15.level == {}
    assert var_15.doc == 'Parser(link=None, b_level=-671, toc=None, level=None, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert var_15.docstring == {}
    assert var_15.imp == {}
    assert f'{type(var_15.root).__module__}.{type(var_15.root).__qualname__}' == 'dataclasses.Field'
    assert var_15.alias == {}
    assert var_15.const == {}
    var_16 = module_0.doctest(var_14)
    assert var_16 == 'AST parser.\n\nUsage:\n```python\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n```\n\nOr create with parameters:\n```python\n>>> p = Parser.new(link=True, level=1)\n```'

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
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
    var_2 = -674
    var_3 = module_0.Parser(var_0, var_2, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is None
    assert var_3.b_level == -674
    assert var_3.toc is None
    assert var_3.level is None
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4 = '&Bz'
    var_5 = var_3.globals(var_4, var_0)
    var_6 = module_3.field(default_factory=var_2, metadata=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'dataclasses.Field'
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
    var_7 = module_0.Parser(var_6, level=var_0, doc=var_6, imp=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert f'{type(var_7.link).__module__}.{type(var_7.link).__qualname__}' == 'dataclasses.Field'
    assert var_7.b_level == 1
    assert var_7.toc is False
    assert var_7.level is None
    assert f'{type(var_7.doc).__module__}.{type(var_7.doc).__qualname__}' == 'dataclasses.Field'
    assert var_7.docstring == {}
    assert f'{type(var_7.imp).__module__}.{type(var_7.imp).__qualname__}' == 'dataclasses.Field'
    assert var_7.root == {}
    assert var_7.alias == {}
    assert var_7.const == {}
    var_8 = var_3.compile()
    assert var_8 == '\n'
    var_9 = module_3.field(metadata=var_0, kw_only=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'dataclasses.Field'
    var_10 = '/f\\Y\x0b2h\r'
    var_7.api(var_9, var_5, prefix=var_10)

def test_case_32():
    var_0 = None
    var_1 = 'ctx'
    var_2 = 5
    var_3 = []
    var_4 = 'value'
    var_5 = {var_4: var_2}
    var_6 = module_1.Constant(*var_3, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Constant'
    assert var_6.value == 5
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_7 = 'i{t'
    var_8 = []
    var_9 = 'id'
    var_10 = {var_9: var_7, var_1: var_0}
    var_11 = module_1.Name(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Name'
    assert var_11.id == 'i{t'
    assert var_11.ctx is None
    var_12 = []
    var_13 = 'target'
    var_14 = 'value'
    var_15 = 'annotation'
    var_16 = {var_13: var_11, var_14: var_6, var_15: var_11}
    var_17 = module_1.AnnAssign(*var_12, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.AnnAssign'
    assert f'{type(var_17.target).__module__}.{type(var_17.target).__qualname__}' == 'ast.Name'
    assert f'{type(var_17.value).__module__}.{type(var_17.value).__qualname__}' == 'ast.Constant'
    assert f'{type(var_17.annotation).__module__}.{type(var_17.annotation).__qualname__}' == 'ast.Name'
    assert module_1.AnnAssign.value is None
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
    var_19 = var_18.globals(var_1, var_17)
    assert var_18.alias == {'ctx.i{t': '5'}
    var_20 = bool('pkg.VAL' in var_18.alias)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_1.Call(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Call'
    assert var_2.func is None
    assert var_2.args is None
    assert var_2.keywords is None
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
    var_4 = -625
    var_5 = ';'
    var_6 = ';e0\x0b_MW_YA;;"'
    var_7 = ' tM^hM,c$Z<$Sw6'
    var_8 = '9R:nQ\r)@e<z3E'
    var_9 = {var_3: var_3, var_5: var_6, var_5: var_7, var_7: var_8}
    var_10 = module_0.Parser(b_level=var_3, root=var_4, const=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 'Any'
    assert var_10.toc is False
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root == -625
    assert var_10.alias == {}
    assert var_10.const == {'Any': 'Any', ';': ' tM^hM,c$Z<$Sw6', ' tM^hM,c$Z<$Sw6': '9R:nQ\r)@e<z3E'}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_11 = 'IFC>iHe`&<j\r\n7ct,'
    var_12 = var_10.globals(var_11, var_0)
    var_13 = module_3.field(repr=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'dataclasses.Field'
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
    var_14 = var_13.__repr__()
    assert var_14 == 'Field(name=None,type=None,default=<dataclasses._MISSING_TYPE object at 0x7eb0391bd450>,default_factory=<dataclasses._MISSING_TYPE object at 0x7eb0391bd450>,init=True,repr=None,hash=None,compare=True,metadata=mappingproxy({}),kw_only=<dataclasses._MISSING_TYPE object at 0x7eb0391bd450>,_field_type=None)'
    var_15 = var_10.compile()
    assert var_15 == '\n'
    var_13.imports(var_12, var_13)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
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
    var_2 = -671
    var_3 = module_0.Parser(var_0, var_2, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is None
    assert var_3.b_level == -671
    assert var_3.toc is None
    assert var_3.level is None
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4 = '&Bz'
    var_5 = var_3.globals(var_4, var_0)
    var_6 = module_3.field(default_factory=var_2, metadata=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'dataclasses.Field'
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
    var_7 = var_3.__repr__()
    assert var_7 == 'Parser(link=None, b_level=-671, toc=None, level=None, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_8 = module_0.Parser(toc=var_6, imp=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is True
    assert var_8.b_level == 1
    assert f'{type(var_8.toc).__module__}.{type(var_8.toc).__qualname__}' == 'dataclasses.Field'
    assert var_8.level == {}
    assert var_8.doc == {}
    assert var_8.docstring == {}
    assert var_8.imp == 'Parser(link=None, b_level=-671, toc=None, level=None, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert var_8.root == {}
    assert var_8.alias == {}
    assert var_8.const == {}
    var_9 = var_8.compile()
    assert var_9 == '**Table of contents:**\n\n\n'
    var_6.visit(var_0)

@pytest.mark.xfail(strict=True)
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
    var_1 = 'pkg'
    var_2 = None
    var_3 = module_1.Tuple()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Tuple'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_4 = [var_3, var_3, var_3]
    var_5 = module_1.Try()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Try'
    var_6 = [var_5, var_5]
    var_0.class_api(var_2, var_1, var_4, var_6)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = 'q+w\x0c'
    var_1 = 'w{y%z.gKkCBV'
    var_2 = {var_1: var_0}
    var_3 = module_0.Parser(imp=var_0, alias=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == 'q+w\x0c'
    assert var_3.root == {}
    assert var_3.alias == {'w{y%z.gKkCBV': 'q+w\x0c'}
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
    var_4 = var_3.compile()
    assert var_4 == '\n'
    module_0.doctest(var_2)

@pytest.mark.xfail(strict=True)
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
    var_1 = var_0.load_docstring(var_0, var_0)
    var_0.parse(var_1, var_1)

def test_case_38():
    var_0 = 'os'
    var_1 = [var_0]
    var_2 = module_0._m(*var_1)
    assert var_2 == 'os'
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
    var_0 = module_0._m()
    assert var_0 == ''
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

def test_case_40():
    var_0 = ''
    var_1 = [var_0]
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
    var_3 = module_0._m(*var_1)
    assert var_3 == ''

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = '__all__'
    var_1 = None
    var_2 = module_0.const_type(var_0)
    assert var_2 == 'Any'
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
    var_3 = []
    var_4 = {var_0: var_0, var_2: var_1}
    var_5 = module_1.Name(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.Any is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = module_0.Parser(var_1, toc=var_1, level=var_4, imp=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is None
    assert var_7.b_level == 1
    assert var_7.toc is None
    assert var_7.level == {'__all__': '__all__', 'Any': None}
    assert var_7.doc == {}
    assert var_7.docstring == {}
    assert var_7.imp is None
    assert var_7.root == {}
    assert var_7.alias == {}
    assert var_7.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_8 = 'value'
    var_9 = {var_0: var_8, var_8: var_2}
    var_10 = module_1.Constant(*var_3, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == 'Any'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = [var_10, var_10, var_10, var_10]
    var_12 = 'elts'
    var_13 = 'P"I_X'
    var_14 = {var_12: var_11, var_13: var_1}
    var_15 = module_1.Tuple(*var_3, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_15.elts).__module__}.{type(var_15.elts).__qualname__}' == 'builtins.list'
    assert len(var_15.elts) == 4
    assert var_15.P"I_X is None
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_16 = []
    var_17 = 'targets'
    var_18 = 'value'
    var_19 = {var_17: var_6, var_18: var_15}
    var_20 = module_1.Assign(*var_16, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.Assign'
    assert f'{type(var_20.targets).__module__}.{type(var_20.targets).__qualname__}' == 'builtins.list'
    assert len(var_20.targets) == 1
    assert f'{type(var_20.value).__module__}.{type(var_20.value).__qualname__}' == 'ast.Tuple'
    assert module_1.Assign.type_comment is None
    var_21 = module_1.Tuple(**var_19)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_21.targets).__module__}.{type(var_21.targets).__qualname__}' == 'builtins.list'
    assert len(var_21.targets) == 1
    assert f'{type(var_21.value).__module__}.{type(var_21.value).__qualname__}' == 'ast.Tuple'
    var_7.globals(var_2, var_20)

def test_case_42():
    var_0 = 'pkg'
    var_1 = 'MyClass'
    var_2 = {var_1: var_1}
    var_3 = module_0.Resolver(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'pkg'
    assert var_3.alias == {'MyClass': 'MyClass'}
    assert var_3.self_ty == ''
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
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_0}
    var_7 = module_1.Constant(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 'pkg'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_8 = var_3.visit_Constant(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Name'
    assert var_8.id == 'pkg'
    assert f'{type(var_8.ctx).__module__}.{type(var_8.ctx).__qualname__}' == 'ast.Load'
    assert var_8.lineno == 1
    assert var_8.col_offset == 0
    assert var_8.end_lineno == 1
    assert var_8.end_col_offset == 3

def test_case_43():
    var_0 = None
    var_1 = []
    var_2 = 'id'
    var_3 = 'ctx'
    var_4 = {var_2: var_2, var_3: var_0}
    var_5 = module_1.Name(*var_1, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == 'id'
    assert var_5.ctx is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_8}
    var_10 = module_1.Constant(*var_7, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == 'value'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = []
    var_12 = 'targets'
    var_13 = 'value'
    var_14 = {var_12: var_6, var_13: var_10, var_3: var_0}
    var_15 = module_1.Assign(*var_11, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Assign'
    assert f'{type(var_15.targets).__module__}.{type(var_15.targets).__qualname__}' == 'builtins.list'
    assert len(var_15.targets) == 1
    assert f'{type(var_15.value).__module__}.{type(var_15.value).__qualname__}' == 'ast.Constant'
    assert var_15.ctx is None
    assert module_1.Assign.type_comment is None
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
    var_17 = 'pkg'
    var_18 = var_16.globals(var_17, var_15)
    assert var_16.alias == {'pkg.id': "'value'"}
    var_19 = bool('pkg.OTHER_CONST' in var_16.alias)
    assert var_19 is True
    with pytest.raises(KeyError):
        var_20 = var_16.alias['pkg.DISCARDED']
    assert var_20 is None

def test_case_44():
    var_0 = '__all__'
    var_1 = None
    var_2 = module_0.const_type(var_0)
    assert var_2 == 'Any'
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
    var_3 = []
    var_4 = 'id'
    var_5 = {var_4: var_0, var_2: var_1}
    var_6 = module_1.Name(*var_3, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Name'
    assert var_6.id == '__all__'
    assert var_6.Any is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = [var_6]
    var_8 = 'value'
    var_9 = ' c\nsfzs($,AXT}'
    var_10 = module_0.Parser(var_1, toc=var_1, level=var_5, imp=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is None
    assert var_10.b_level == 1
    assert var_10.toc is None
    assert var_10.level == {'id': '__all__', 'Any': None}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp is None
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_11 = 'value'
    var_12 = {var_0: var_8, var_11: var_2}
    var_13 = module_1.Constant(*var_3, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Constant'
    assert var_13.value == 'Any'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_14 = []
    var_15 = 'elts'
    var_16 = {var_15: var_14, var_9: var_1}
    var_17 = module_1.Tuple(*var_14, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Tuple'
    assert var_17.elts == []
    assert var_17. c
sfzs($,AXT} is None
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_18 = []
    var_19 = 'targets'
    var_20 = 'value'
    var_21 = {var_19: var_7, var_20: var_17}
    var_22 = module_1.Assign(*var_18, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Assign'
    assert f'{type(var_22.targets).__module__}.{type(var_22.targets).__qualname__}' == 'builtins.list'
    assert len(var_22.targets) == 1
    assert f'{type(var_22.value).__module__}.{type(var_22.value).__qualname__}' == 'ast.Tuple'
    assert module_1.Assign.type_comment is None
    var_23 = module_1.Tuple(**var_21)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_23.targets).__module__}.{type(var_23.targets).__qualname__}' == 'builtins.list'
    assert len(var_23.targets) == 1
    assert f'{type(var_23.value).__module__}.{type(var_23.value).__qualname__}' == 'ast.Tuple'
    var_24 = var_10.globals(var_2, var_22)
    assert var_10.alias == {'Any.__all__': '()'}

def test_case_45():
    var_0 = 'OTHER_CONST'
    var_1 = None
    var_2 = []
    var_3 = 'id'
    var_4 = 'ctx'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_1.Name(*var_2, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Name'
    assert var_6.id == 'OTHER_CONST'
    assert var_6.ctx is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = [var_6]
    var_8 = 'hello'
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Constant'
    assert var_12.value == 'hello'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_7, var_15: var_12, var_16: var_1}
    var_18 = module_1.Assign(*var_13, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Assign'
    assert f'{type(var_18.targets).__module__}.{type(var_18.targets).__qualname__}' == 'builtins.list'
    assert len(var_18.targets) == 1
    assert f'{type(var_18.value).__module__}.{type(var_18.value).__qualname__}' == 'ast.Constant'
    assert var_18.type_comment is None
    assert module_1.Assign.type_comment is None
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
    var_20 = 'pkg'
    var_21 = var_19.globals(var_20, var_18)
    assert var_19.root == {'pkg.OTHER_CONST': 'pkg'}
    assert var_19.alias == {'pkg.OTHER_CONST': "'hello'"}
    assert var_19.const == {'pkg.OTHER_CONST': 'str'}
    var_22 = bool('pkg.OTHER_CONST' in var_19.alias)
    assert var_22 is True
    with pytest.raises(KeyError):
        var_23 = var_19.alias['pkg.DISCARDED']
    assert var_23 is None

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = '__all__'
    var_1 = None
    var_2 = module_0.const_type(var_0)
    assert var_2 == 'Any'
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
    var_3 = []
    var_4 = 'id'
    var_5 = {var_4: var_0, var_2: var_1}
    var_6 = module_1.Name(*var_3, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Name'
    assert var_6.id == '__all__'
    assert var_6.Any is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = [var_6]
    var_8 = 'value'
    var_9 = module_0.Parser(var_1, toc=var_1, level=var_5, imp=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is None
    assert var_9.b_level == 1
    assert var_9.toc is None
    assert var_9.level == {'id': '__all__', 'Any': None}
    assert var_9.doc == {}
    assert var_9.docstring == {}
    assert var_9.imp is None
    assert var_9.root == {}
    assert var_9.alias == {}
    assert var_9.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_10 = 'value'
    var_11 = {var_0: var_4, var_10: var_2}
    var_12 = module_1.Constant(*var_3, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Constant'
    assert var_12.value == 'Any'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_13 = [var_12, var_12, var_12, var_12]
    var_14 = 'elts'
    var_15 = {var_14: var_13, var_8: var_1}
    var_16 = module_1.Tuple(*var_3, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_16.elts).__module__}.{type(var_16.elts).__qualname__}' == 'builtins.list'
    assert len(var_16.elts) == 4
    assert var_16.value is None
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_17 = []
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_7, var_19: var_16}
    var_21 = module_1.Assign(*var_17, **var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'ast.Assign'
    assert f'{type(var_21.targets).__module__}.{type(var_21.targets).__qualname__}' == 'builtins.list'
    assert len(var_21.targets) == 1
    assert f'{type(var_21.value).__module__}.{type(var_21.value).__qualname__}' == 'ast.Tuple'
    assert module_1.Assign.type_comment is None
    var_22 = module_1.Tuple(**var_20)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_22.targets).__module__}.{type(var_22.targets).__qualname__}' == 'builtins.list'
    assert len(var_22.targets) == 1
    assert f'{type(var_22.value).__module__}.{type(var_22.value).__qualname__}' == 'ast.Tuple'
    var_9.globals(var_2, var_21)

def test_case_47():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0._defaults(var_1)
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
    var_3 = list(var_2)
    var_4 = bool(var_3 == [' ', ' '])
    assert var_4 is True

def test_case_48():
    var_0 = None
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = module_0._defaults(var_2)
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
    with pytest.raises(AttributeError):
        var_4 = list(var_3)

def test_case_49():
    var_0 = 'VAL'
    var_1 = None
    var_2 = []
    var_3 = 'id'
    var_4 = 'ctx'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_1.Name(*var_2, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Name'
    assert var_6.id == 'VAL'
    assert var_6.ctx is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = 5
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Constant'
    assert var_11.value == 5
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_12 = 'int'
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_12, var_15: var_1}
    var_17 = module_1.Name(*var_13, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Name'
    assert var_17.id == 'int'
    assert var_17.ctx is None
    var_18 = []
    var_19 = 'target'
    var_20 = 'value'
    var_21 = 'annotation'
    var_22 = {var_19: var_6, var_21: var_11, var_20: var_6, var_21: var_17, var_19: var_18}
    var_23 = module_1.AnnAssign(*var_18, **var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'ast.AnnAssign'
    assert var_23.target == []
    assert f'{type(var_23.annotation).__module__}.{type(var_23.annotation).__qualname__}' == 'ast.Name'
    assert f'{type(var_23.value).__module__}.{type(var_23.value).__qualname__}' == 'ast.Name'
    assert module_1.AnnAssign.value is None
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
    var_25 = "!l'(|Pl/n\n,"
    var_26 = var_24.globals(var_25, var_23)
    var_27 = bool('pkg.VAL' in var_24.alias)
    assert var_27 is True
    with pytest.raises(KeyError):
        var_28 = var_24.alias['pkg.VAL']
    assert var_28 == '5'

def test_case_50():
    var_0 = '__all__'
    var_1 = None
    var_2 = []
    var_3 = 'id'
    var_4 = 'ctx'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_1.Name(*var_2, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Name'
    assert var_6.id == '__all__'
    assert var_6.ctx is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = 'sub'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Constant'
    assert var_11.value == 'sub'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_12 = 'other'
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.Constant(*var_13, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Constant'
    assert var_16.value == 'other'
    var_17 = [var_11, var_16]
    var_18 = []
    var_19 = 'elts'
    var_20 = 'ctx'
    var_21 = {var_19: var_17, var_20: var_1}
    var_22 = module_1.Tuple(*var_18, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_22.elts).__module__}.{type(var_22.elts).__qualname__}' == 'builtins.list'
    assert len(var_22.elts) == 2
    assert var_22.ctx is None
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_23 = []
    var_24 = 'targets'
    var_25 = 'value'
    var_26 = {var_24: var_8, var_25: var_22}
    var_27 = module_1.Assign(*var_23, **var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'ast.Assign'
    assert var_27.targets == []
    assert f'{type(var_27.value).__module__}.{type(var_27.value).__qualname__}' == 'ast.Tuple'
    assert module_1.Assign.type_comment is None
    var_28 = module_0.Parser()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'apimd.parser.Parser'
    assert var_28.link is True
    assert var_28.b_level == 1
    assert var_28.toc is False
    assert var_28.level == {}
    assert var_28.doc == {}
    assert var_28.docstring == {}
    assert var_28.imp == {}
    assert var_28.root == {}
    assert var_28.alias == {}
    assert var_28.const == {}
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
    var_29 = 'pkg'
    var_30 = set()
    var_31 = var_28.globals(var_29, var_27)
    with pytest.raises(KeyError):
        var_32 = bool('pkg.sub' in var_28.imp['pkg'])
    assert var_32 is True

def test_case_51():
    var_0 = '(I@~'
    var_1 = None
    var_2 = module_0.const_type(var_0)
    assert var_2 == 'Any'
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
    var_3 = []
    var_4 = 'id'
    var_5 = {var_4: var_0, var_2: var_1}
    var_6 = module_1.Name(*var_3, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Name'
    assert var_6.id == '(I@~'
    assert var_6.Any is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = [var_6]
    var_8 = ' c\nsfzs($,AXT}'
    var_9 = module_0.Parser(var_1, toc=var_1, level=var_5, imp=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is None
    assert var_9.b_level == 1
    assert var_9.toc is None
    assert var_9.level == {'id': '(I@~', 'Any': None}
    assert var_9.doc == {}
    assert var_9.docstring == {}
    assert var_9.imp is None
    assert var_9.root == {}
    assert var_9.alias == {}
    assert var_9.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_10 = var_9.__repr__()
    assert var_10 == "Parser(link=None, b_level=1, toc=None, level={'id': '(I@~', 'Any': None}, doc={}, docstring={}, imp=None, root={}, alias={}, const={})"
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_11, var_8: var_1}
    var_14 = module_1.Tuple(*var_11, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Tuple'
    assert var_14.elts == []
    assert var_14. c
sfzs($,AXT} is None
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_15 = []
    var_16 = 'targets'
    var_17 = 'value'
    var_18 = {var_16: var_7, var_17: var_14}
    var_19 = module_1.Assign(*var_15, **var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.Assign'
    assert f'{type(var_19.targets).__module__}.{type(var_19.targets).__qualname__}' == 'builtins.list'
    assert len(var_19.targets) == 1
    assert f'{type(var_19.value).__module__}.{type(var_19.value).__qualname__}' == 'ast.Tuple'
    assert module_1.Assign.type_comment is None
    var_20 = var_9.globals(var_2, var_19)
    assert var_9.root == {'Any.(I@~': 'Any'}
    assert var_9.alias == {'Any.(I@~': '()'}
    assert var_9.const == {'Any.(I@~': 'tuple'}

def test_case_52():
    var_0 = '__all__'
    var_1 = None
    var_2 = module_0.const_type(var_0)
    assert var_2 == 'Any'
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
    var_3 = []
    var_4 = [var_2]
    var_5 = 'b'
    var_6 = 'value'
    var_7 = ' c\nsfzs($,AXT}'
    var_8 = 4085
    var_9 = {var_7: var_8}
    var_10 = module_0.Parser(var_1, toc=var_1, level=var_9, imp=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is None
    assert var_10.b_level == 1
    assert var_10.toc is None
    assert var_10.level == {' c\nsfzs($,AXT}': 4085}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp is None
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_11 = {var_6: var_5}
    var_12 = module_1.Constant(*var_3, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Constant'
    assert var_12.value == 'b'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_13 = 'other'
    var_14 = 'value'
    var_15 = {var_0: var_5, var_14: var_13}
    var_16 = module_1.Constant(*var_3, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Constant'
    assert var_16.value == 'other'
    var_17 = [var_12, var_12, var_16]
    var_18 = []
    var_19 = 'elts'
    var_20 = 'ctx'
    var_21 = {var_19: var_17, var_20: var_1}
    var_22 = module_1.Tuple(*var_18, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_22.elts).__module__}.{type(var_22.elts).__qualname__}' == 'builtins.list'
    assert len(var_22.elts) == 3
    assert var_22.ctx is None
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_23 = []
    var_24 = 'targets'
    var_25 = 'value'
    var_26 = {var_24: var_4, var_25: var_22}
    var_27 = module_1.Assign(*var_23, **var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'ast.Assign'
    assert var_27.targets == ['Any']
    assert f'{type(var_27.value).__module__}.{type(var_27.value).__qualname__}' == 'ast.Tuple'
    assert module_1.Assign.type_comment is None
    var_28 = 'pkg'
    var_29 = set()
    var_30 = var_10.globals(var_28, var_27)

def test_case_53():
    var_0 = 'VAL'
    var_1 = None
    var_2 = []
    var_3 = 'id'
    var_4 = 'ctx'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_1.Name(*var_2, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Name'
    assert var_6.id == 'VAL'
    assert var_6.ctx is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = 5
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Constant'
    assert var_11.value == 5
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_12 = 'int'
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_12, var_15: var_1}
    var_17 = module_1.Name(*var_13, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Name'
    assert var_17.id == 'int'
    assert var_17.ctx is None
    var_18 = []
    var_19 = 'target'
    var_20 = 'annotation'
    var_21 = {var_19: var_6, var_20: var_17}
    var_22 = module_1.AnnAssign(*var_18, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.AnnAssign'
    assert f'{type(var_22.target).__module__}.{type(var_22.target).__qualname__}' == 'ast.Name'
    assert f'{type(var_22.annotation).__module__}.{type(var_22.annotation).__qualname__}' == 'ast.Name'
    assert module_1.AnnAssign.value is None
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
    var_24 = 'pkg'
    var_25 = var_23.globals(var_24, var_22)
    var_26 = bool('pkg.VAL' in var_23.alias)
    assert var_26 is True
    with pytest.raises(KeyError):
        var_27 = var_23.alias['pkg.VAL']
    assert var_27 == '5'

def test_case_54():
    var_0 = '__all__'
    var_1 = None
    var_2 = module_0.const_type(var_0)
    assert var_2 == 'Any'
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
    var_3 = []
    var_4 = 'id'
    var_5 = 'ctx'
    var_6 = 'value'
    var_7 = module_0.Parser(var_1, toc=var_1, level=var_2, imp=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is None
    assert var_7.b_level == 1
    assert var_7.toc is None
    assert var_7.level == 'Any'
    assert var_7.doc == {}
    assert var_7.docstring == {}
    assert var_7.imp is None
    assert var_7.root == {}
    assert var_7.alias == {}
    assert var_7.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_8 = {var_4: var_5, var_6: var_0}
    var_9 = module_3.field(init=var_8, compare=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'dataclasses.Field'
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
    var_10 = var_7.__repr__()
    assert var_10 == "Parser(link=None, b_level=1, toc=None, level='Any', doc={}, docstring={}, imp=None, root={}, alias={}, const={})"
    var_11 = var_7.class_api(var_1, var_1, var_3, var_10)
    with pytest.raises(TypeError):
        module_1.get_docstring(var_9, var_11)

def test_case_55():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Resolver'
    assert var_2.root == 'pkg'
    assert var_2.alias == {}
    assert var_2.self_ty == ''
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
    var_3 = 'pkg.Unknown'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 'pkg.Unknown'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_8 = var_2.visit_Constant(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Attribute'
    assert f'{type(var_8.value).__module__}.{type(var_8.value).__qualname__}' == 'ast.Name'
    assert var_8.attr == 'Unknown'
    assert f'{type(var_8.ctx).__module__}.{type(var_8.ctx).__qualname__}' == 'ast.Load'
    assert var_8.lineno == 1
    assert var_8.col_offset == 0
    assert var_8.end_lineno == 1
    assert var_8.end_col_offset == 11

@pytest.mark.xfail(strict=True)
def test_case_56():
    var_0 = '__all__'
    var_1 = None
    var_2 = []
    var_3 = {var_1: var_0, var_1: var_1}
    var_4 = 'sub'
    var_5 = ' c\nsfzs($,AXT}'
    var_6 = module_0.Parser(var_1, toc=var_1, level=var_3, imp=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is None
    assert var_6.b_level == 1
    assert var_6.toc is None
    assert var_6.level == {None: None}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp is None
    assert var_6.root == {}
    assert var_6.alias == {}
    assert var_6.const == {}
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
    var_7 = 'value'
    var_8 = {var_0: var_4, var_7: var_5}
    var_9 = module_1.Constant(*var_2, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Constant'
    assert var_9.value == ' c\nsfzs($,AXT}'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_10 = [var_9, var_9]
    var_11 = []
    var_12 = 'k'
    var_6.class_api(var_1, var_12, var_10, var_11)

def test_case_57():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Resolver'
    assert var_2.root == 'pkg'
    assert var_2.alias == {}
    assert var_2.self_ty == ''
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
    var_3 = 'invalid syntax @#$'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 'invalid syntax @#$'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_8 = var_2.visit_Constant(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Constant'
    assert var_8.value == 'invalid syntax @#$'
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

def test_case_58():
    var_0 = '__all__'
    var_1 = None
    var_2 = 'J.opWU2'
    var_3 = '-9@<*?'
    var_4 = '}&Pu6\x0bXGVvsdI)'
    var_5 = {var_0: var_0, var_2: var_2, var_3: var_3, var_2: var_4}
    var_6 = module_0.Parser(imp=var_0, root=var_1, const=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp == '__all__'
    assert var_6.root is None
    assert var_6.alias == {}
    assert var_6.const == {'__all__': '__all__', 'J.opWU2': '}&Pu6\x0bXGVvsdI)', '-9@<*?': '-9@<*?'}
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
    var_7 = var_6.compile()
    assert var_7 == '\n'
    var_8 = None
    var_9 = module_0.const_type(var_0)
    assert var_9 == 'Any'
    var_10 = []
    var_11 = 'id'
    var_12 = 'ctx'
    var_13 = {var_11: var_0, var_12: var_8}
    var_14 = module_1.Name(*var_10, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Name'
    assert var_14.id == '__all__'
    assert var_14.ctx is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_15 = [var_14]
    var_16 = 'sub'
    var_17 = 'value'
    var_18 = module_0.Parser(var_8, toc=var_8, level=var_13, imp=var_8)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'apimd.parser.Parser'
    assert var_18.link is None
    assert var_18.b_level == 1
    assert var_18.toc is None
    assert var_18.level == {'id': '__all__', 'ctx': None}
    assert var_18.doc == {}
    assert var_18.docstring == {}
    assert var_18.imp is None
    assert var_18.root == {}
    assert var_18.alias == {}
    assert var_18.const == {}
    var_19 = {var_17: var_16}
    var_20 = module_1.Constant(*var_10, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.Constant'
    assert var_20.value == 'sub'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_21 = 'other'
    var_22 = 'value'
    var_23 = {var_0: var_16, var_22: var_21}
    var_24 = module_1.Constant(*var_10, **var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'ast.Constant'
    assert var_24.value == 'other'
    var_25 = []
    var_26 = 'elts'
    var_27 = 'ctx'
    var_28 = {var_26: var_15, var_27: var_8}
    var_29 = module_1.Tuple(*var_25, **var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_29.elts).__module__}.{type(var_29.elts).__qualname__}' == 'builtins.list'
    assert len(var_29.elts) == 1
    assert var_29.ctx is None
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_30 = []
    var_31 = 'targets'
    var_32 = 'value'
    var_33 = {var_31: var_15, var_32: var_29}
    var_34 = module_1.Assign(*var_30, **var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'ast.Assign'
    assert f'{type(var_34.targets).__module__}.{type(var_34.targets).__qualname__}' == 'builtins.list'
    assert len(var_34.targets) == 1
    assert f'{type(var_34.value).__module__}.{type(var_34.value).__qualname__}' == 'ast.Tuple'
    assert module_1.Assign.type_comment is None
    var_35 = 'pkg'
    var_36 = set()
    var_37 = var_18.globals(var_35, var_34)
    assert var_18.alias == {'pkg.__all__': '(__all__,)'}

def test_case_59():
    var_0 = 'pkg'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Resolver'
    assert var_2.root == 'pkg'
    assert var_2.alias == {}
    assert var_2.self_ty == ''
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
    var_3 = 123
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 123
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_8 = var_2.visit_Constant(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Constant'
    assert var_8.value == 123
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

def test_case_60():
    var_0 = 'MY_CONST'
    var_1 = None
    var_2 = []
    var_3 = 'id'
    var_4 = 'ctx'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_1.Name(*var_2, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Name'
    assert var_6.id == 'MY_CONST'
    assert var_6.ctx is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = [var_6]
    var_8 = 10
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Constant'
    assert var_12.value == 10
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_13 = 'int'
    var_14 = []
    var_15 = 'targets'
    var_16 = 'value'
    var_17 = 'type_comment'
    var_18 = {var_15: var_7, var_16: var_12, var_17: var_13}
    var_19 = module_1.Assign(*var_14, **var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.Assign'
    assert f'{type(var_19.targets).__module__}.{type(var_19.targets).__qualname__}' == 'builtins.list'
    assert len(var_19.targets) == 1
    assert f'{type(var_19.value).__module__}.{type(var_19.value).__qualname__}' == 'ast.Constant'
    assert var_19.type_comment == 'int'
    assert module_1.Assign.type_comment is None
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
    var_21 = 'pkg'
    var_22 = var_20.globals(var_21, var_19)
    assert var_20.root == {'pkg.MY_CONST': 'pkg'}
    assert var_20.alias == {'pkg.MY_CONST': '10'}
    assert var_20.const == {'pkg.MY_CONST': 'int'}
    var_23 = var_20.alias['pkg.MY_CONST']
    assert var_23 == '10'