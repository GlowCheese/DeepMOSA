# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.parser as module_0
import dataclasses as module_1
import ast as module_2
import inspect as module_3

def test_case_0():
    var_0 = 'P&*&4D#'
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

def test_case_1():
    var_0 = 'aD]^'
    var_1 = module_0.code(var_0)
    assert var_1 == '`aD]^`'
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
    var_0 = None
    module_0.table(items=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.Parser(toc=var_0, alias=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is True
    assert var_1.b_level == 1
    assert var_1.toc is None
    assert var_1.level == {}
    assert var_1.doc == {}
    assert var_1.docstring == {}
    assert var_1.imp == {}
    assert var_1.root == {}
    assert var_1.alias is None
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
    assert var_2 == 'Parser(link=True, b_level=1, toc=None, level={}, doc={}, docstring={}, imp={}, root={}, alias=None, const={})'
    var_3 = module_0.Parser(var_2, var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 'Parser(link=True, b_level=1, toc=None, level={}, doc={}, docstring={}, imp={}, root={}, alias=None, const={})'
    assert var_3.toc == 'Parser(link=True, b_level=1, toc=None, level={}, doc={}, docstring={}, imp={}, root={}, alias=None, const={})'
    assert var_3.level == 'Parser(link=True, b_level=1, toc=None, level={}, doc={}, docstring={}, imp={}, root={}, alias=None, const={})'
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const == {}
    var_4 = '&Bz'
    var_5 = var_1.globals(var_4, var_3)
    var_6 = module_0.is_public_family(var_4)
    assert var_6 is True
    var_1.is_public(var_0)

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
    var_1 = 'from typing import List\nx: List[int] = []'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'from typing import List\nx: List[int] = []': 0}
    assert var_0.doc == {'from typing import List\nx: List[int] = []': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'from typing import List\nx: List[int] = []': {*()}}
    assert var_0.root == {'from typing import List\nx: List[int] = []': 'from typing import List\nx: List[int] = []'}
    assert var_0.alias == {'from typing import List\nx: List[int] = [].List': 'typing.List', 'from typing import List\nx: List[int] = [].x': '[]'}
    var_3 = var_0.load_docstring(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.parent(var_0)

def test_case_7():
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

def test_case_8():
    var_0 = 'mGx\x0b\nO^&PiVn*%C5VR'
    var_1 = module_0.code(var_0)
    assert var_1 == '<code>mGx\x0b\nO^&PiVn*%C5VR</code>'
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

def test_case_10():
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

def test_case_11():
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
    var_1 = "\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n": 0, "\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n.public_func": 0, "\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n._private_func": 0}
    assert var_0.doc == {"\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {"\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n": {"\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n.+ublic_func"}}
    assert var_0.root == {"\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n": "\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n", "\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n.public_func": "\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n", "\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n._private_func": "\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n"}
    assert var_0.alias == {"\n__all__ = ['+ublic_func']\ndef public_func(): pass\ndef _private_func(): pass\n.__all__": "['+ublic_func']"}

def test_case_13():
    var_0 = '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n'
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
    assert var_1.level == {'\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n': 0, '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n._private': 0, '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass': 0, '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass.__init__': 0}
    assert var_1.doc == {'\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n._private': '### _private()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_1.imp == {'\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n': {*()}}
    assert var_1.root == {'\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n': '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n._private': '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass': '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass.__init__': '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n'}
    var_3 = var_1.compile()
    assert var_3 == '## Module `\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n`\n<a id="\ndef _private(): pas\nclass myclass:\n    def __init__(self): pass\n"></a>\n\n### class MyClass\n\n*Full name:* `\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass`\n<a id="\ndef _private(): pas\nclass myclass:\n    def __init__(self): pass\n-myclass"></a>\n'

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
    var_1 = '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': 1, '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': 1}
    assert var_0.doc == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': '### async baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `float` | `str` |\n\n'}
    assert var_0.docstring == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': 'This is an async function.'}
    assert var_0.imp == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': {*()}}
    assert var_0.root == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n', '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n'}

def test_case_15():
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
    var_6 = module_1.field(compare=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'dataclasses.Field'
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_1.Field.compare).__module__}.{type(module_1.Field.compare).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.default).__module__}.{type(module_1.Field.default).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.default_factory).__module__}.{type(module_1.Field.default_factory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.hash).__module__}.{type(module_1.Field.hash).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.init).__module__}.{type(module_1.Field.init).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.kw_only).__module__}.{type(module_1.Field.kw_only).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.metadata).__module__}.{type(module_1.Field.metadata).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.name).__module__}.{type(module_1.Field.name).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.repr).__module__}.{type(module_1.Field.repr).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.type).__module__}.{type(module_1.Field.type).__qualname__}' == 'builtins.member_descriptor'
    var_7 = '^}U3QOqn7\r8P'
    var_8 = module_0.is_public_family(var_7)
    assert var_8 is True
    var_9 = module_0.doctest(var_7)
    assert var_9 == '^}U3QOqn7\n8P'

def test_case_16():
    var_0 = 'CqSTANT = 42'
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
    assert var_1.level == {'CqSTANT = 42': 0}
    assert var_1.doc == {'CqSTANT = 42': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.imp == {'CqSTANT = 42': {*()}}
    assert var_1.root == {'CqSTANT = 42': 'CqSTANT = 42'}
    assert var_1.alias == {'CqSTANT = 42.CqSTANT': '42'}

@pytest.mark.xfail(strict=True)
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
    var_1 = 'F!(>*4whooFDemhD!I'
    var_0.imports(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = 'Nc|AO:JOewl]'
    var_2 = 'E'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_2.Dict(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Dict'
    assert var_4.Nc|AO:JOewl] is None
    assert var_4.E is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    module_0.const_type(var_4)

@pytest.mark.xfail(strict=True)
def test_case_19():
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

def test_case_20():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_2.Call(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Call'
    assert var_2.func is None
    assert var_2.args is None
    assert var_2.keywords is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
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

def test_case_21():
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
    var_5 = module_1.field(default_factory=var_1, metadata=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'dataclasses.Field'
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_1.Field.compare).__module__}.{type(module_1.Field.compare).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.default).__module__}.{type(module_1.Field.default).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.default_factory).__module__}.{type(module_1.Field.default_factory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.hash).__module__}.{type(module_1.Field.hash).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.init).__module__}.{type(module_1.Field.init).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.kw_only).__module__}.{type(module_1.Field.kw_only).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.metadata).__module__}.{type(module_1.Field.metadata).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.name).__module__}.{type(module_1.Field.name).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.repr).__module__}.{type(module_1.Field.repr).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_1.Field.type).__module__}.{type(module_1.Field.type).__qualname__}' == 'builtins.member_descriptor'
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
    var_10 = module_1.field(hash=var_0, compare=var_4, metadata=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'dataclasses.Field'
    var_11 = module_0.Resolver(var_7, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'apimd.parser.Resolver'
    assert var_11.root == '<6D(o\t$'
    assert f'{type(var_11.alias).__module__}.{type(var_11.alias).__qualname__}' == 'dataclasses.Field'
    assert var_11.self_ty == ''
    var_12 = ')Rna-|g'
    var_13 = module_0.esc_underscore(var_12)
    assert var_13 == ')Rna-|g'
    var_14 = module_3.getdoc(var_8)
    assert var_14 == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
    assert f'{type(module_3.mod_dict).__module__}.{type(module_3.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_3.mod_dict) == 168
    assert module_3.k == 512
    assert module_3.v == 'ASYNC_GENERATOR'
    assert module_3.CO_OPTIMIZED == 1
    assert module_3.CO_NEWLOCALS == 2
    assert module_3.CO_VARARGS == 4
    assert module_3.CO_VARKEYWORDS == 8
    assert module_3.CO_NESTED == 16
    assert module_3.CO_GENERATOR == 32
    assert module_3.CO_NOFREE == 64
    assert module_3.CO_COROUTINE == 128
    assert module_3.CO_ITERABLE_COROUTINE == 256
    assert module_3.CO_ASYNC_GENERATOR == 512
    assert module_3.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_3.modulesbyfile == {}
    assert module_3.GEN_CREATED == 'GEN_CREATED'
    assert module_3.GEN_RUNNING == 'GEN_RUNNING'
    assert module_3.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_3.GEN_CLOSED == 'GEN_CLOSED'
    assert module_3.CORO_CREATED == 'CORO_CREATED'
    assert module_3.CORO_RUNNING == 'CORO_RUNNING'
    assert module_3.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_3.CORO_CLOSED == 'CORO_CLOSED'
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

def test_case_22():
    var_0 = '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n'
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
    assert var_1.level == {'\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n': 0, '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n._private': 0, '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass': 0, '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass.__init__': 0}
    assert var_1.doc == {'\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n._private': '### _private()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_1.imp == {'\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n': {*()}}
    assert var_1.root == {'\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n': '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n._private': '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass': '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n', '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n.MyClass.__init__': '\ndef _private(): pas\nclass MyClass:\n    def __init__(self): pass\n'}

def test_case_23():
    var_0 = "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n"
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
    var_2 = '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n'
    var_3 = var_1.parse(var_2, var_2)
    assert var_1.level == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': 1, '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': 1}
    assert var_1.doc == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': '### async baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `float` | `str` |\n\n'}
    assert var_1.docstring == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': 'This is an async function.'}
    assert var_1.imp == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': {*()}}
    assert var_1.root == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n', '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n'}
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
    var_5 = "\n@decorator1\n@decorator2\ndef decorated_func():\n    '''Decorated function.'''\n    pass\n"
    var_6 = var_4.parse(var_0, var_5)
    assert var_4.level == {"\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": 2, "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.decorated_func": 2}
    assert var_4.doc == {"\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.decorated_func": '### decorated_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@decorator1` |\n| `@decorator2` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_4.docstring == {"\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.decorated_func": 'Decorated function.'}
    assert var_4.imp == {"\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": {*()}}
    assert var_4.root == {"\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n", "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.decorated_func": "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n"}

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
    var_1 = 'from typing import List\nx: List[int] = []'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'from typing import List\nx: List[int] = []': 0}
    assert var_0.doc == {'from typing import List\nx: List[int] = []': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'from typing import List\nx: List[int] = []': {*()}}
    assert var_0.root == {'from typing import List\nx: List[int] = []': 'from typing import List\nx: List[int] = []'}
    assert var_0.alias == {'from typing import List\nx: List[int] = [].List': 'typing.List', 'from typing import List\nx: List[int] = [].x': '[]'}

def test_case_25():
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
    var_2 = 'from typing import List\nx: List[int] = []'
    var_3 = var_1.compile()
    assert var_3 == '\n'
    var_4 = var_1.compile()
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
    var_6 = '\nclass MyClass:\n    def method(self): pass\n'
    var_7 = var_5.parse(var_0, var_6)
    assert var_5.level == {'test_module': 0, 'test_module.MyClass': 0, 'test_module.MyClass.method': 0}
    assert var_5.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_5.imp == {'test_module': {*()}}
    assert var_5.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module', 'test_module.MyClass.method': 'test_module'}
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
    var_9 = 'kONSTANT = 42'
    var_10 = var_8.parse(var_0, var_9)
    assert var_8.level == {'test_module': 0}
    assert var_8.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_8.imp == {'test_module': {*()}}
    assert var_8.root == {'test_module': 'test_module'}
    assert var_8.alias == {'test_module.kONSTANT': '42'}
    var_11 = var_1.__repr__()
    assert var_11 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_12 = var_8.compile()
    assert var_12 == '\n'
    var_13 = var_5.parse(var_0, var_3)
    var_14 = module_0.const_type(var_7)
    assert var_14 == 'Any'
    var_15 = var_8.compile()
    assert var_15 == '\n'
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
    var_17 = "\n__all__ = ['public_func']\ndef public_func(): pass\ndef _private_func(): pass\n"
    var_18 = var_16.parse(var_0, var_17)
    assert var_16.level == {'test_module': 0, 'test_module.public_func': 0, 'test_module._private_func': 0}
    assert var_16.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_16.imp == {'test_module': {'test_module.public_func'}}
    assert var_16.root == {'test_module': 'test_module', 'test_module.public_func': 'test_module', 'test_module._private_func': 'test_module'}
    assert var_16.alias == {'test_module.__all__': "['public_func']"}
    var_19 = var_16.compile()
    assert var_19 == '## Module `test_module`\n<a id="test_module"></a>\n\n### public_func()\n\n*Full name:* `test_module.public_func`\n<a id="test_module-public_func"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_20 = var_16.compile()
    assert var_20 == '## Module `test_module`\n<a id="test_module"></a>\n\n### public_func()\n\n*Full name:* `test_module.public_func`\n<a id="test_module-public_func"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_21 = True
    var_22 = module_0.Parser(toc=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'apimd.parser.Parser'
    assert var_22.link is True
    assert var_22.b_level == 1
    assert var_22.toc is True
    assert var_22.level == {}
    assert var_22.doc == {}
    assert var_22.docstring == {}
    assert var_22.imp == {}
    assert var_22.root == {}
    assert var_22.alias == {}
    assert var_22.const == {}
    var_23 = var_22.parse(var_0, var_0)
    assert var_22.level == {'test_module': 0}
    assert var_22.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_22.imp == {'test_module': {*()}}
    assert var_22.root == {'test_module': 'test_module'}
    var_24 = var_22.compile()
    assert var_24 == '**Table of contents:**\n\n\n'
    var_25 = var_8.parse(var_12, var_2)
    assert var_8.level == {'test_module': 0, '\n': 0}
    assert var_8.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', '\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_8.imp == {'test_module': {*()}, '\n': {*()}}
    assert var_8.root == {'test_module': 'test_module', '\n': '\n'}
    assert var_8.alias == {'test_module.kONSTANT': '42', '\n.List': 'typing.List', '\n.x': '[]'}

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = 'p+*6\x0bjD|Nv\t*a['
    var_1 = '#J!9P'
    var_2 = {var_0: var_0, var_0: var_0, var_1: var_0}
    var_3 = module_0.Parser(docstring=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {'p+*6\x0bjD|Nv\t*a[': 'p+*6\x0bjD|Nv\t*a[', '#J!9P': 'p+*6\x0bjD|Nv\t*a['}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
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
    var_4 = None
    var_5 = [var_4, var_4]
    var_6 = module_2.Constant(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Constant'
    assert var_6.value is None
    assert var_6.kind is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_7 = [var_6, var_6, var_6, var_6]
    var_3.class_api(var_4, var_4, var_7, var_4)

def test_case_27():
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
    var_1 = 'from typing import List\nx: List[int] = []'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'from typing import List\nx: List[int] = []': 0}
    assert var_0.doc == {'from typing import List\nx: List[int] = []': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'from typing import List\nx: List[int] = []': {*()}}
    assert var_0.root == {'from typing import List\nx: List[int] = []': 'from typing import List\nx: List[int] = []'}
    assert var_0.alias == {'from typing import List\nx: List[int] = [].List': 'typing.List', 'from typing import List\nx: List[int] = [].x': '[]'}
    var_3 = var_0.load_docstring(var_1, var_1)
    var_4 = var_0.compile()
    assert var_4 == '\n'

@pytest.mark.xfail(strict=True)
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
    var_1 = 'test_modue'
    var_2 = "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n"
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_modue': 0, 'test_modue.Bar': 0, 'test_modue.Bar.__init__': 0}
    assert var_0.doc == {'test_modue': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_modue.Bar': '### class Bar\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_modue.Bar.__init__': '#### Bar.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | value | return |\n|:----:|:-----:|:------:|\n| `Self` | `int` | `Any` |\n\n'}
    assert var_0.docstring == {'test_modue.Bar': 'This is a class.'}
    assert var_0.imp == {'test_modue': {*()}}
    assert var_0.root == {'test_modue': 'test_modue', 'test_modue.Bar': 'test_modue', 'test_modue.Bar.__init__': 'test_modue'}
    var_4 = (1255.506951+660.583j)
    var_5 = (var_4, var_3)
    var_6 = var_0.load_docstring(var_2, var_5)
    var_7 = var_0.__repr__()
    assert var_7 == 'Parser(link=True, b_level=1, toc=False, level={\'test_modue\': 0, \'test_modue.Bar\': 0, \'test_modue.Bar.__init__\': 0}, doc={\'test_modue\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'test_modue.Bar\': \'### class Bar\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'test_modue.Bar.__init__\': \'#### Bar.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | value | return |\\n|:----:|:-----:|:------:|\\n| `Self` | `int` | `Any` |\\n\\n\'}, docstring={\'test_modue.Bar\': \'This is a class.\'}, imp={\'test_modue\': set()}, root={\'test_modue\': \'test_modue\', \'test_modue.Bar\': \'test_modue\', \'test_modue.Bar.__init__\': \'test_modue\'}, alias={}, const={})'
    var_5.imports(var_3, var_7)

def test_case_29():
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
    var_3 = module_2.Load()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Load'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_4 = [var_2]
    var_5 = module_2.Constant()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Constant'
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_6 = module_2.Assign()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Assign'
    assert module_2.Assign.type_comment is None
    var_7 = module_2.Constant()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    var_8 = module_2.Expr()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Expr'
    var_9 = [var_6, var_8]
    var_10 = module_0.walk_body(var_9)
    var_11 = list(var_10)
    var_12 = True
    var_13 = module_2.Constant()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Constant'
    var_14 = 'y'
    var_15 = module_2.Name()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Name'
    var_16 = [var_15]
    var_17 = module_2.Assign()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Assign'
    var_18 = 4
    var_19 = module_2.Constant()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.Constant'
    var_20 = var_2.__eq__(var_2)
    assert var_20 is True
    var_21 = [var_20]
    var_22 = module_2.If()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.If'
    var_23 = [var_22]
    var_24 = module_0.walk_body(var_23)
    with pytest.raises(AttributeError):
        var_25 = list(var_24)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_2.Subscript(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Subscript'
    assert var_2.value is None
    assert var_2.slice is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_3 = 'O X'
    var_4 = '<x4zgZ4{2'
    var_5 = {var_3: var_4}
    var_6 = module_0.Resolver(var_3, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Resolver'
    assert var_6.root == 'O X'
    assert var_6.alias == {'O X': '<x4zgZ4{2'}
    assert var_6.self_ty == ''
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
    var_7 = var_6.visit_Subscript(var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Subscript'
    assert var_7.value is None
    assert var_7.slice is None
    var_8 = None
    var_9 = module_1.dataclass(repr=var_8, unsafe_hash=var_8, slots=var_8)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_9.load_docstring(var_9, var_9)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = '>1f>UP2L6%'
    var_2 = module_2.ImportFrom()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.ImportFrom'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_3 = '<QW=.*m/'
    var_4 = '\x0ca!'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.Parser(toc=var_0, imp=var_0, root=var_5, alias=var_5, const=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is True
    assert var_7.b_level == 1
    assert var_7.toc is None
    assert var_7.level == {}
    assert var_7.doc == {}
    assert var_7.docstring == {}
    assert var_7.imp is None
    assert var_7.root == {'<QW=.*m/': '\x0ca!'}
    assert var_7.alias == {'<QW=.*m/': '\x0ca!'}
    assert var_7.const == {}
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
    var_8 = var_7.imports(var_0, var_2)
    var_9 = module_3.getdoc(var_1)
    assert var_9 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert f'{type(module_3.mod_dict).__module__}.{type(module_3.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_3.mod_dict) == 168
    assert module_3.k == 512
    assert module_3.v == 'ASYNC_GENERATOR'
    assert module_3.CO_OPTIMIZED == 1
    assert module_3.CO_NEWLOCALS == 2
    assert module_3.CO_VARARGS == 4
    assert module_3.CO_VARKEYWORDS == 8
    assert module_3.CO_NESTED == 16
    assert module_3.CO_GENERATOR == 32
    assert module_3.CO_NOFREE == 64
    assert module_3.CO_COROUTINE == 128
    assert module_3.CO_ITERABLE_COROUTINE == 256
    assert module_3.CO_ASYNC_GENERATOR == 512
    assert module_3.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_3.modulesbyfile == {}
    assert module_3.GEN_CREATED == 'GEN_CREATED'
    assert module_3.GEN_RUNNING == 'GEN_RUNNING'
    assert module_3.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_3.GEN_CLOSED == 'GEN_CLOSED'
    assert module_3.CORO_CREATED == 'CORO_CREATED'
    assert module_3.CORO_RUNNING == 'CORO_RUNNING'
    assert module_3.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_3.CORO_CLOSED == 'CORO_CLOSED'
    var_10 = ' C~tzy\r_w'
    var_11 = {var_1: var_10, var_4: var_4}
    var_12 = module_0.Parser(imp=var_9, alias=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'apimd.parser.Parser'
    assert var_12.link is True
    assert var_12.b_level == 1
    assert var_12.toc is False
    assert var_12.level == {}
    assert var_12.doc == {}
    assert var_12.docstring == {}
    assert var_12.imp == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert var_12.root == {}
    assert var_12.alias == {'>1f>UP2L6%': ' C~tzy\r_w', '\x0ca!': '\x0ca!'}
    assert var_12.const == {}
    var_9.visit_Name(var_9)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = 'N0 e7xT\x0c`8#_d(r#+'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
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
    var_3 = None
    var_4 = 'CONSTANT = 42'
    var_5 = var_2.parse(var_4, var_4)
    assert var_2.level == {'CONSTANT = 42': 0}
    assert var_2.doc == {'CONSTANT = 42': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_2.imp == {'CONSTANT = 42': {*()}}
    assert var_2.root == {'CONSTANT = 42': 'CONSTANT = 42', 'CONSTANT = 42.CONSTANT': 'CONSTANT = 42'}
    assert var_2.alias == {'CONSTANT = 42.CONSTANT': '42'}
    assert var_2.const == {'CONSTANT = 42.CONSTANT': 'int'}
    var_6 = var_2.compile()
    assert var_6 == '## Module `CONSTANT = 42`\n<a id="constant = 42"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n'
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
    var_8 = module_0.Parser(var_3, doc=var_1, root=var_3, alias=var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is None
    assert var_8.b_level == 1
    assert var_8.toc is False
    assert var_8.level == {}
    assert var_8.doc == {'N0 e7xT\x0c`8#_d(r#+': 'N0 e7xT\x0c`8#_d(r#+'}
    assert var_8.docstring == {}
    assert var_8.imp == {}
    assert var_8.root is None
    assert var_8.alias == {'N0 e7xT\x0c`8#_d(r#+': 'N0 e7xT\x0c`8#_d(r#+'}
    assert var_8.const == {}
    var_8.compile()

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = '>1f>UP2L6%'
    var_2 = module_3.getdoc(var_1)
    assert var_2 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert f'{type(module_3.mod_dict).__module__}.{type(module_3.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_3.mod_dict) == 168
    assert module_3.k == 512
    assert module_3.v == 'ASYNC_GENERATOR'
    assert module_3.CO_OPTIMIZED == 1
    assert module_3.CO_NEWLOCALS == 2
    assert module_3.CO_VARARGS == 4
    assert module_3.CO_VARKEYWORDS == 8
    assert module_3.CO_NESTED == 16
    assert module_3.CO_GENERATOR == 32
    assert module_3.CO_NOFREE == 64
    assert module_3.CO_COROUTINE == 128
    assert module_3.CO_ITERABLE_COROUTINE == 256
    assert module_3.CO_ASYNC_GENERATOR == 512
    assert module_3.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_3.modulesbyfile == {}
    assert module_3.GEN_CREATED == 'GEN_CREATED'
    assert module_3.GEN_RUNNING == 'GEN_RUNNING'
    assert module_3.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_3.GEN_CLOSED == 'GEN_CLOSED'
    assert module_3.CORO_CREATED == 'CORO_CREATED'
    assert module_3.CORO_RUNNING == 'CORO_RUNNING'
    assert module_3.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_3.CORO_CLOSED == 'CORO_CLOSED'
    var_3 = var_2.__repr__()
    assert var_3 == '"str(object=\'\') -> str\\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\\n\\nCreate a new string object from the given object. If encoding or\\nerrors is specified, then the object must expose a data buffer\\nthat will be decoded using the given encoding and error handler.\\nOtherwise, returns the result of object.__str__() (if defined)\\nor repr(object).\\nencoding defaults to sys.getdefaultencoding().\\nerrors defaults to \'strict\'."'
    var_4 = '+6Y'
    var_5 = {var_4: var_4, var_1: var_1, var_1: var_4}
    var_6 = module_0.Parser(imp=var_2, alias=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert var_6.root == {}
    assert var_6.alias == {'+6Y': '+6Y', '>1f>UP2L6%': '+6Y'}
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
    var_7 = var_6.compile()
    assert var_7 == '\n'
    var_8 = None
    var_9 = module_0.Parser(toc=var_8, alias=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc is None
    assert var_9.level == {}
    assert var_9.doc == {}
    assert var_9.docstring == {}
    assert var_9.imp == {}
    assert var_9.root == {}
    assert var_9.alias is None
    assert var_9.const == {}
    var_10 = var_9.load_docstring(var_0, var_0)
    var_11 = var_9.parse(var_2, var_3)
    assert var_9.level == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": 7}
    assert var_9.doc == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_9.docstring == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."}
    assert var_9.imp == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": {*()}}
    assert var_9.root == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."}
    var_12 = module_0.const_type(var_0)
    assert var_12 == 'Any'
    var_9.is_public(var_4)

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
    var_1 = 'from typing import List\nx: List[int] = []'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'from typing import List\nx: List[int] = []': 0}
    assert var_0.doc == {'from typing import List\nx: List[int] = []': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'from typing import List\nx: List[int] = []': {*()}}
    assert var_0.root == {'from typing import List\nx: List[int] = []': 'from typing import List\nx: List[int] = []'}
    assert var_0.alias == {'from typing import List\nx: List[int] = [].List': 'typing.List', 'from typing import List\nx: List[int] = [].x': '[]'}
    var_3 = 'CONSTANT= 42'
    var_4 = var_0.parse(var_1, var_3)
    assert var_0.root == {'from typing import List\nx: List[int] = []': 'from typing import List\nx: List[int] = []', 'from typing import List\nx: List[int] = [].CONSTANT': 'from typing import List\nx: List[int] = []'}
    assert var_0.alias == {'from typing import List\nx: List[int] = [].List': 'typing.List', 'from typing import List\nx: List[int] = [].x': '[]', 'from typing import List\nx: List[int] = [].CONSTANT': '42'}
    assert var_0.const == {'from typing import List\nx: List[int] = [].CONSTANT': 'int'}
    var_5 = var_0.compile()
    assert var_5 == '## Module `from typing import List\nx: List[int] = []`\n<a id="from typing import list\nx: list[int] = []"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n'

def test_case_35():
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
    var_2 = 'from typing import List\nx: List[int] = []'
    var_3 = var_1.parse(var_0, var_2)
    assert var_1.level == {'test_module': 0}
    assert var_1.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.imp == {'test_module': {*()}}
    assert var_1.root == {'test_module': 'test_module'}
    assert var_1.alias == {'test_module.List': 'typing.List', 'test_module.x': '[]'}
    var_4 = '\nclass MyClass:\n    def method(self): pass\n'
    var_5 = var_1.parse(var_0, var_4)
    assert var_1.level == {'test_module': 0, 'test_module.MyClass': 0, 'test_module.MyClass.method': 0}
    assert var_1.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_1.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module', 'test_module.MyClass.method': 'test_module'}
    var_6 = var_1.compile()
    assert var_6 == '## Module `test_module`\n<a id="test_module"></a>\n\n### class MyClass\n\n*Full name:* `test_module.MyClass`\n<a id="test_module-myclass"></a>\n\n#### MyClass.method()\n\n*Full name:* `test_module.MyClass.method`\n<a id="test_module-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n'
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
    var_8 = "\n__all__ = ['public_func']\ndef public_func(): pass\ndef _private_func(): pass\n"
    var_9 = var_7.parse(var_0, var_8)
    assert var_7.level == {'test_module': 0, 'test_module.public_func': 0, 'test_module._private_func': 0}
    assert var_7.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_7.imp == {'test_module': {'test_module.public_func'}}
    assert var_7.root == {'test_module': 'test_module', 'test_module.public_func': 'test_module', 'test_module._private_func': 'test_module'}
    assert var_7.alias == {'test_module.__all__': "['public_func']"}
    var_10 = var_7.__eq__(var_3)
    var_11 = var_7.compile()
    assert var_11 == '## Module `test_module`\n<a id="test_module"></a>\n\n### public_func()\n\n*Full name:* `test_module.public_func`\n<a id="test_module-public_func"></a>\n\n| return |\n|:------:|\n| `Any` |\n'

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
    var_1 = 'CONSTANT= 42'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'CONSTANT= 42': 0}
    assert var_0.doc == {'CONSTANT= 42': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'CONSTANT= 42': {*()}}
    assert var_0.root == {'CONSTANT= 42': 'CONSTANT= 42', 'CONSTANT= 42.CONSTANT': 'CONSTANT= 42'}
    assert var_0.alias == {'CONSTANT= 42.CONSTANT': '42'}
    assert var_0.const == {'CONSTANT= 42.CONSTANT': 'int'}

def test_case_37():
    var_0 = 'from typeng import List\nx: List[int] = []'
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
    assert var_1.level == {'from typeng import List\nx: List[int] = []': 0}
    assert var_1.doc == {'from typeng import List\nx: List[int] = []': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.imp == {'from typeng import List\nx: List[int] = []': {*()}}
    assert var_1.root == {'from typeng import List\nx: List[int] = []': 'from typeng import List\nx: List[int] = []'}
    assert var_1.alias == {'from typeng import List\nx: List[int] = [].List': 'typeng.List', 'from typeng import List\nx: List[int] = [].x': '[]'}
    var_3 = module_0.const_type(var_2)
    assert var_3 == 'Any'

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
    var_1 = "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": 1, "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": 1}
    assert var_0.doc == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `bool` |\n\n'}
    assert var_0.docstring == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": 'This is a function.'}
    assert var_0.imp == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": {*()}}
    assert var_0.root == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n", "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n"}
    var_3 = "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n"
    var_4 = var_0.parse(var_3, var_3)
    assert var_0.level == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": 1, "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": 1, "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": 2, "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar": 2, "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar.__init__": 2}
    assert var_0.doc == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `bool` |\n\n', "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar": '### class Bar\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar.__init__": '#### Bar.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | value | return |\n|:----:|:-----:|:------:|\n| `Self` | `int` | `Any` |\n\n'}
    assert var_0.docstring == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": 'This is a function.', "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar": 'This is a class.'}
    assert var_0.imp == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": {*()}, "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": {*()}}
    assert var_0.root == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n", "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n", "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n", "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar": "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n", "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar.__init__": "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n"}
    var_5 = module_3.getdoc(var_0)
    assert var_5 == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
    assert f'{type(module_3.mod_dict).__module__}.{type(module_3.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_3.mod_dict) == 168
    assert module_3.k == 512
    assert module_3.v == 'ASYNC_GENERATOR'
    assert module_3.CO_OPTIMIZED == 1
    assert module_3.CO_NEWLOCALS == 2
    assert module_3.CO_VARARGS == 4
    assert module_3.CO_VARKEYWORDS == 8
    assert module_3.CO_NESTED == 16
    assert module_3.CO_GENERATOR == 32
    assert module_3.CO_NOFREE == 64
    assert module_3.CO_COROUTINE == 128
    assert module_3.CO_ITERABLE_COROUTINE == 256
    assert module_3.CO_ASYNC_GENERATOR == 512
    assert module_3.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_3.modulesbyfile == {}
    assert module_3.GEN_CREATED == 'GEN_CREATED'
    assert module_3.GEN_RUNNING == 'GEN_RUNNING'
    assert module_3.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_3.GEN_CLOSED == 'GEN_CLOSED'
    assert module_3.CORO_CREATED == 'CORO_CREATED'
    assert module_3.CORO_RUNNING == 'CORO_RUNNING'
    assert module_3.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_3.CORO_CLOSED == 'CORO_CLOSED'
    var_6 = var_0.compile()
    assert var_6 == '## Module `\ndef foo(x: int, y: str) -> bool:\n    \'\'\'This is a function.\'\'\'\n    return True\n`\n<a id="\ndef foo(x: int, y: str) -> bool:\n    \'\'\'this is a function-\'\'\'\n    return true\n"></a>\n\n### foo()\n\n*Full name:* `\ndef foo(x: int, y: str) -> bool:\n    \'\'\'This is a function.\'\'\'\n    return True\n.foo`\n<a id="\ndef foo(x: int, y: str) -> bool:\n    \'\'\'this is a function-\'\'\'\n    return true\n-foo"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `bool` |\n\nThis is a function.\n\n## Module `\nclass Bar:\n    \'\'\'This is a class.\'\'\'\n    def __init__(self, value: int):\n        self.value = value\n`\n<a id="\nclass bar:\n    \'\'\'this is a class-\'\'\'\n    def __init__(self, value: int):\n        self-value = value\n"></a>\n\n### class Bar\n\n*Full name:* `\nclass Bar:\n    \'\'\'This is a class.\'\'\'\n    def __init__(self, value: int):\n        self.value = value\n.Bar`\n<a id="\nclass bar:\n    \'\'\'this is a class-\'\'\'\n    def __init__(self, value: int):\n        self-value = value\n-bar"></a>\n\nThis is a class.\n'
    var_5.compile()

def test_case_39():
    var_0 = "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n"
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
    assert var_1.level == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": 1, "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": 1}
    assert var_1.doc == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `bool` |\n\n'}
    assert var_1.docstring == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": 'This is a function.'}
    assert var_1.imp == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": {*()}}
    assert var_1.root == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n", "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n"}
    var_3 = (1255.506951+660.583j)
    var_4 = (var_3, var_2)
    var_5 = var_1.load_docstring(var_0, var_4)
    var_6 = 'CONSTANT = 42'
    var_7 = 'I?F$'
    var_8 = var_1.parse(var_7, var_6)
    assert var_1.level == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": 1, "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": 1, 'I?F$': 0}
    assert var_1.doc == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `bool` |\n\n', 'I?F$': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.imp == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": {*()}, 'I?F$': {*()}}
    assert var_1.root == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n", "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n", 'I?F$': 'I?F$', 'I?F$.CONSTANT': 'I?F$'}
    assert var_1.alias == {'I?F$.CONSTANT': '42'}
    assert var_1.const == {'I?F$.CONSTANT': 'int'}
    var_9 = var_1.compile()
    assert var_9 == '## Module `I?F$`\n<a id="i?f$"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n\n## Module `\ndef foo(x: int, y: str) -> bool:\n    \'\'\'This is a function.\'\'\'\n    return True\n`\n<a id="\ndef foo(x: int, y: str) -> bool:\n    \'\'\'this is a function-\'\'\'\n    return true\n"></a>\n\n### foo()\n\n*Full name:* `\ndef foo(x: int, y: str) -> bool:\n    \'\'\'This is a function.\'\'\'\n    return True\n.foo`\n<a id="\ndef foo(x: int, y: str) -> bool:\n    \'\'\'this is a function-\'\'\'\n    return true\n-foo"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `bool` |\n\nThis is a function.\n'
    var_10 = module_0.Parser(root=var_5, const=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 1
    assert var_10.toc is False
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root is None
    assert var_10.alias == {}
    assert var_10.const == 'CONSTANT = 42'
    var_11 = var_1.parse(var_6, var_0)
    assert var_1.level == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": 1, "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": 1, 'I?F$': 0, 'CONSTANT = 42': 0, 'CONSTANT = 42.foo': 0}
    assert var_1.doc == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `bool` |\n\n', 'I?F$': '## Module `{}`\n<a id="{}"></a>\n\n', 'CONSTANT = 42': '## Module `{}`\n<a id="{}"></a>\n\n', 'CONSTANT = 42.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `bool` |\n\n'}
    assert var_1.docstring == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": 'This is a function.', 'CONSTANT = 42.foo': 'This is a function.'}
    assert var_1.imp == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": {*()}, 'I?F$': {*()}, 'CONSTANT = 42': {*()}}
    assert var_1.root == {"\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n": "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n", "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n.foo": "\ndef foo(x: int, y: str) -> bool:\n    '''This is a function.'''\n    return True\n", 'I?F$': 'I?F$', 'I?F$.CONSTANT': 'I?F$', 'CONSTANT = 42': 'CONSTANT = 42', 'CONSTANT = 42.foo': 'CONSTANT = 42'}

def test_case_40():
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
    var_2 = 'from typing import List\nx: List[int] = []'
    var_3 = var_1.parse(var_0, var_2)
    assert var_1.level == {'test_module': 0}
    assert var_1.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.imp == {'test_module': {*()}}
    assert var_1.root == {'test_module': 'test_module'}
    assert var_1.alias == {'test_module.List': 'typing.List', 'test_module.x': '[]'}
    var_4 = var_1.compile()
    assert var_4 == '\n'
    var_5 = var_1.parse(var_0, var_4)
    var_6 = var_1.compile()
    assert var_6 == '\n'
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
    var_8 = 'CONSTANT = 42'
    var_9 = var_7.parse(var_0, var_8)
    assert var_7.level == {'test_module': 0}
    assert var_7.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_7.imp == {'test_module': {*()}}
    assert var_7.root == {'test_module': 'test_module', 'test_module.CONSTANT': 'test_module'}
    assert var_7.alias == {'test_module.CONSTANT': '42'}
    assert var_7.const == {'test_module.CONSTANT': 'int'}
    var_10 = var_7.compile()
    assert var_10 == '## Module `test_module`\n<a id="test_module"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n'
    var_11 = module_0.code(var_6)
    assert var_11 == '`\n`'
    var_12 = "\n__all__ = ['public_func']\ndef public_func(): pass\ndef _private_func(): pass\n"
    var_13 = var_7.parse(var_0, var_12)
    assert var_7.level == {'test_module': 0, 'test_module.public_func': 0, 'test_module._private_func': 0}
    assert var_7.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_7.imp == {'test_module': {'test_module.public_func'}}
    assert var_7.root == {'test_module': 'test_module', 'test_module.CONSTANT': 'test_module', 'test_module.public_func': 'test_module', 'test_module._private_func': 'test_module'}
    assert var_7.alias == {'test_module.CONSTANT': '42', 'test_module.__all__': "['public_func']"}
    var_14 = var_7.compile()
    assert var_14 == '## Module `test_module`\n<a id="test_module"></a>\n\n### public_func()\n\n*Full name:* `test_module.public_func`\n<a id="test_module-public_func"></a>\n\n| return |\n|:------:|\n| `Any` |\n'

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
    var_1 = None
    var_2 = [var_1, var_1]
    var_3 = module_2.List(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.List'
    assert var_3.elts is None
    assert var_3.ctx is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_4 = var_0.__post_init__()
    var_5 = '7'
    var_6 = 'from typing import List\nx: L|st[int] = []'
    var_7 = var_0.parse(var_5, var_6)
    assert var_0.level == {'7': 0}
    assert var_0.doc == {'7': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'7': {*()}}
    assert var_0.root == {'7': '7'}
    assert var_0.alias == {'7.List': 'typing.List', '7.x': '[]'}
    var_8 = var_0.compile()
    assert var_8 == '\n'
    var_9 = '\nclass MyClass:\n    def method(self): pass\n'
    var_10 = var_0.globals(var_7, var_7)
    var_11 = var_0.parse(var_5, var_9)
    assert var_0.level == {'7': 0, '7.MyClass': 0, '7.MyClass.method': 0}
    assert var_0.doc == {'7': '## Module `{}`\n<a id="{}"></a>\n\n', '7.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '7.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.root == {'7': '7', '7.MyClass': '7', '7.MyClass.method': '7'}
    var_12 = module_0.const_type(var_3)
    assert var_12 == 'list'

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 'p+*6\x0bjD|Nv\t*a['
    var_1 = ''
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
    var_3 = 'from typeng import List\nx: List[int] = []'
    var_4 = var_2.parse(var_1, var_3)
    assert var_2.level == {'': 0}
    assert var_2.doc == {'': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_2.imp == {'': {*()}}
    assert var_2.root == {'': ''}
    assert var_2.alias == {'List': 'typeng.List', 'x': '[]'}
    var_5 = var_2.compile()
    assert var_5 == '\n'
    var_6 = var_2.compile()
    assert var_6 == '\n'
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
    var_8 = None
    var_9 = var_2.load_docstring(var_1, var_8)
    var_7.parse(var_0, var_8)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = '#J!9P'
    var_1 = '"'
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
    var_3 = 'from typeng import List\nx: List[int] = []'
    var_4 = var_2.parse(var_1, var_3)
    assert var_2.level == {'"': 0}
    assert var_2.doc == {'"': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_2.imp == {'"': {*()}}
    assert var_2.root == {'"': '"'}
    assert var_2.alias == {'".List': 'typeng.List', '".x': '[]'}
    var_5 = module_0.code(var_0)
    assert var_5 == '`#J!9P`'
    var_6 = var_2.compile()
    assert var_6 == '\n'
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
    var_8 = None
    var_9 = var_2.load_docstring(var_1, var_8)
    var_10 = '\nclass MyClass:\n    def method(self): pass\n'
    var_11 = 'CONSTANT = 42'
    var_12 = var_2.parse(var_1, var_11)
    assert var_2.root == {'"': '"', '".CONSTANT': '"'}
    assert var_2.alias == {'".List': 'typeng.List', '".x': '[]', '".CONSTANT': '42'}
    assert var_2.const == {'".CONSTANT': 'int'}
    var_13 = var_2.compile()
    assert var_13 == '## Module `"`\n<a id="""></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n'
    var_14 = var_2.parse(var_1, var_11)
    var_15 = var_7.compile()
    assert var_15 == '\n'
    var_16 = var_7.parse(var_1, var_10)
    assert var_7.level == {'"': 0, '".MyClass': 0, '".MyClass.method': 0}
    assert var_7.doc == {'"': '## Module `{}`\n<a id="{}"></a>\n\n', '".MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '".MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_7.imp == {'"': {*()}}
    assert var_7.root == {'"': '"', '".MyClass': '"', '".MyClass.method': '"'}
    var_17 = var_7.__post_init__()
    var_17.visit_Constant(var_14)

def test_case_44():
    var_0 = 'p+*6\x0bjD|Nv\t*a['
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.Parser(docstring=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc == {}
    assert var_2.docstring == {'p+*6\x0bjD|Nv\t*a[': 'p+*6\x0bjD|Nv\t*a['}
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
    assert var_3 == '\n'
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
    var_5 = 'from typing import List\nx: List[int] = []'
    var_6 = var_4.parse(var_3, var_5)
    assert var_4.level == {'\n': 0}
    assert var_4.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_4.imp == {'\n': {*()}}
    assert var_4.root == {'\n': '\n'}
    assert var_4.alias == {'\n.List': 'typing.List', '\n.x': '[]'}
    var_7 = var_4.compile()
    assert var_7 == '\n'
    var_8 = var_4.compile()
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
    var_10 = None
    var_11 = var_4.load_docstring(var_7, var_10)
    var_12 = '\nclass MyClass:\n    def method(self): pass\n'
    var_13 = var_2.compile()
    assert var_13 == '\n'
    var_14 = module_0.const_type(var_10)
    assert var_14 == 'Any'
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
    var_16 = '\ndef _private(): pass\nclass MyClass:\n    def __init__(self): pass\n'
    var_17 = var_15.parse(var_13, var_16)
    assert var_15.level == {'\n': 0, '\n._private': 0, '\n.MyClass': 0, '\n.MyClass.__init__': 0}
    assert var_15.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n._private': '### _private()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\n.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_15.imp == {'\n': {*()}}
    assert var_15.root == {'\n': '\n', '\n._private': '\n', '\n.MyClass': '\n', '\n.MyClass.__init__': '\n'}
    var_18 = False
    var_19 = module_0.Parser(var_18, toc=var_18, root=var_1, alias=var_6)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'apimd.parser.Parser'
    assert var_19.link is False
    assert var_19.b_level == 1
    assert var_19.toc is False
    assert var_19.level == {}
    assert var_19.doc == {}
    assert var_19.docstring == {}
    assert var_19.imp == {}
    assert var_19.root == {'p+*6\x0bjD|Nv\t*a[': 'p+*6\x0bjD|Nv\t*a['}
    assert var_19.alias is None
    assert var_19.const == {}
    var_20 = "\n__all__ = ['public_func']\ndef public_func(): pass\ndef _private_func(): pass\n"
    var_21 = 'FU'
    var_22 = var_19.parse(var_21, var_12)
    assert var_2.docstring == {'p+*6\x0bjD|Nv\t*a[': 'p+*6\x0bjD|Nv\t*a[', 'FU': 'FU', 'FU.MyClass': 'FU', 'FU.MyClass.method': 'FU'}
    assert var_19.level == {'FU': 0, 'FU.MyClass': 0, 'FU.MyClass.method': 0}
    assert var_19.doc == {'FU': '## Module `{}`\n\n', 'FU.MyClass': '### class MyClass\n\n*Full name:* `{}`\n\n', 'FU.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_19.imp == {'FU': {*()}}
    assert var_19.root == {'p+*6\x0bjD|Nv\t*a[': 'p+*6\x0bjD|Nv\t*a[', 'FU': 'FU', 'FU.MyClass': 'FU', 'FU.MyClass.method': 'FU'}
    var_23 = var_15.__repr__()
    assert var_23 == 'Parser(link=True, b_level=1, toc=False, level={\'\\n\': 0, \'\\n._private\': 0, \'\\n.MyClass\': 0, \'\\n.MyClass.__init__\': 0}, doc={\'\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n._private\': \'### _private()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', \'\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n.MyClass.__init__\': \'#### MyClass.\\\\_\\\\_init\\\\_\\\\_()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `Any` |\\n\\n\'}, docstring={}, imp={\'\\n\': set()}, root={\'\\n\': \'\\n\', \'\\n._private\': \'\\n\', \'\\n.MyClass\': \'\\n\', \'\\n.MyClass.__init__\': \'\\n\'}, alias={}, const={})'
    var_24 = 'Tn!W2'
    var_25 = module_0.Resolver(var_24, var_20, var_11)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'apimd.parser.Resolver'
    assert var_25.root == 'Tn!W2'
    assert var_25.alias == "\n__all__ = ['public_func']\ndef public_func(): pass\ndef _private_func(): pass\n"
    assert var_25.self_ty is None

def test_case_45():
    var_0 = "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n"
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
    var_2 = '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n'
    var_3 = var_1.parse(var_2, var_2)
    assert var_1.level == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': 1, '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': 1}
    assert var_1.doc == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': '### async baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `float` | `str` |\n\n'}
    assert var_1.docstring == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': 'This is an async function.'}
    assert var_1.imp == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': {*()}}
    assert var_1.root == {'\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n': '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n', '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n.baz': '\nasync def baz(x: float) -> str:\n    \'\'\'This is an async function.\'\'\'\n    return "hello"\n'}
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
    var_5 = var_4.parse(var_0, var_0)
    assert var_4.level == {"\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": 2, "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar": 2, "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar.__init__": 2}
    assert var_4.doc == {"\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar": '### class Bar\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar.__init__": '#### Bar.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | value | return |\n|:----:|:-----:|:------:|\n| `Self` | `int` | `Any` |\n\n'}
    assert var_4.docstring == {"\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar": 'This is a class.'}
    assert var_4.imp == {"\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": {*()}}
    assert var_4.root == {"\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n": "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n", "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar": "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n", "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n.Bar.__init__": "\nclass Bar:\n    '''This is a class.'''\n    def __init__(self, value: int):\n        self.value = value\n"}

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = 'p+*6\x0bjD|Nv\t*a['
    var_1 = '3K<\\Yl}'
    var_2 = {var_0: var_0, var_1: var_0, var_0: var_0}
    var_3 = module_0.Parser(docstring=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {'p+*6\x0bjD|Nv\t*a[': 'p+*6\x0bjD|Nv\t*a[', '3K<\\Yl}': 'p+*6\x0bjD|Nv\t*a['}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
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
    var_4 = 'W7'
    var_5 = 'def foo(): pass'
    var_6 = var_3.parse(var_4, var_5)
    assert var_3.level == {'W7': 0, 'W7.foo': 0}
    assert var_3.doc == {'W7': '## Module `{}`\n<a id="{}"></a>\n\n', 'W7.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_3.imp == {'W7': {*()}}
    assert var_3.root == {'W7': 'W7', 'W7.foo': 'W7'}
    var_7 = var_3.compile()
    assert var_7 == '## Module `W7`\n<a id="w7"></a>\n\n### foo()\n\n*Full name:* `W7.foo`\n<a id="w7-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
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
    var_9 = 'from typi5g import List\nx: List[int]#= []'
    var_10 = var_8.parse(var_4, var_9)
    assert var_8.level == {'W7': 0}
    assert var_8.doc == {'W7': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_8.imp == {'W7': {*()}}
    assert var_8.root == {'W7': 'W7'}
    assert var_8.alias == {'W7.List': 'typi5g.List'}
    var_11 = var_8.compile()
    assert var_11 == '\n'
    var_12 = var_8.compile()
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
    var_14 = None
    var_15 = var_8.load_docstring(var_4, var_14)
    var_16 = var_3.compile()
    assert var_16 == '## Module `W7`\n<a id="w7"></a>\n\n### foo()\n\n*Full name:* `W7.foo`\n<a id="w7-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_17 = module_0.const_type(var_14)
    assert var_17 == 'Any'
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
    var_8.parse(var_14, var_1)

def test_case_47():
    var_0 = 'p+*6\x0bjD|Nv\t*a['
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.Parser(docstring=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc == {}
    assert var_2.docstring == {'p+*6\x0bjD|Nv\t*a[': 'p+*6\x0bjD|Nv\t*a['}
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
    var_3 = 'W7'
    var_4 = var_2.compile()
    assert var_4 == '\n'
    var_5 = 'from typng import List\nx: List[int] = []'
    var_6 = var_2.parse(var_3, var_5)
    assert var_2.level == {'W7': 0}
    assert var_2.doc == {'W7': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_2.imp == {'W7': {*()}}
    assert var_2.root == {'W7': 'W7'}
    assert var_2.alias == {'W7.List': 'typng.List', 'W7.x': '[]'}
    var_7 = var_2.compile()
    assert var_7 == '\n'
    var_8 = var_2.load_docstring(var_3, var_6)
    var_9 = 'CONSTAN.T = 42'
    var_10 = var_2.parse(var_3, var_9)
    var_11 = var_2.compile()
    assert var_11 == '\n'
    var_12 = module_0.const_type(var_6)
    assert var_12 == 'Any'
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
    var_14 = '\ndef _private(): pass\nclass MyClass:\n    def __init__(self): pass\n'
    var_15 = var_13.parse(var_3, var_14)
    assert var_13.level == {'W7': 0, 'W7._private': 0, 'W7.MyClass': 0, 'W7.MyClass.__init__': 0}
    assert var_13.doc == {'W7': '## Module `{}`\n<a id="{}"></a>\n\n', 'W7._private': '### _private()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'W7.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'W7.MyClass.__init__': '#### MyClass.\\_\\_init\\_\\_()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_13.imp == {'W7': {*()}}
    assert var_13.root == {'W7': 'W7', 'W7._private': 'W7', 'W7.MyClass': 'W7', 'W7.MyClass.__init__': 'W7'}
    var_16 = var_13.compile()
    assert var_16 == '## Module `W7`\n<a id="w7"></a>\n\n### class MyClass\n\n*Full name:* `W7.MyClass`\n<a id="w7-myclass"></a>\n'
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
    var_18 = "\n__all__ = ['public_func']\ndef public_func(): pass\ndef _private_func(): pass\n"
    var_19 = var_17.parse(var_3, var_18)
    assert var_17.level == {'W7': 0, 'W7.public_func': 0, 'W7._private_func': 0}
    assert var_17.doc == {'W7': '## Module `{}`\n<a id="{}"></a>\n\n', 'W7.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'W7._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_17.imp == {'W7': {'W7.public_func'}}
    assert var_17.root == {'W7': 'W7', 'W7.public_func': 'W7', 'W7._private_func': 'W7'}
    assert var_17.alias == {'W7.__all__': "['public_func']"}
    var_20 = var_17.compile()
    assert var_20 == '## Module `W7`\n<a id="w7"></a>\n\n### public_func()\n\n*Full name:* `W7.public_func`\n<a id="w7-public_func"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_21 = var_2.__eq__(var_19)
    var_22 = module_0.Resolver(var_15, var_1)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'apimd.parser.Resolver'
    assert var_22.root is None
    assert var_22.alias == {'p+*6\x0bjD|Nv\t*a[': 'p+*6\x0bjD|Nv\t*a['}
    assert var_22.self_ty == ''

@pytest.mark.xfail(strict=True)
def test_case_48():
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
    var_2 = 't0st_mod1le'
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
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4 = 'from typing import List\nx: List[int] = []'
    var_5 = var_3.parse(var_2, var_4)
    assert var_3.level == {'t0st_mod1le': 0}
    assert var_3.doc == {'t0st_mod1le': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_3.imp == {'t0st_mod1le': {*()}}
    assert var_3.root == {'t0st_mod1le': 't0st_mod1le'}
    assert var_3.alias == {'t0st_mod1le.List': 'typing.List', 't0st_mod1le.x': '[]'}
    var_6 = var_3.globals(var_2, var_5)
    var_7 = '\nclass MyClass:\n    def method(self): pass\n'
    var_8 = var_3.parse(var_2, var_7)
    assert var_3.level == {'t0st_mod1le': 0, 't0st_mod1le.MyClass': 0, 't0st_mod1le.MyClass.method': 0}
    assert var_3.doc == {'t0st_mod1le': '## Module `{}`\n<a id="{}"></a>\n\n', 't0st_mod1le.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 't0st_mod1le.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_3.root == {'t0st_mod1le': 't0st_mod1le', 't0st_mod1le.MyClass': 't0st_mod1le', 't0st_mod1le.MyClass.method': 't0st_mod1le'}
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
    var_10 = [var_8, var_0]
    var_11 = module_2.AnnAssign(*var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.AnnAssign'
    assert var_11.target is None
    assert var_11.annotation is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.AnnAssign.value is None
    var_12 = var_3.globals(var_0, var_11)
    var_9.parse(var_9, var_0)

def test_case_49():
    var_0 = '3K9L<\\Yl}'
    var_1 = '4'
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
    var_3 = 'from typeng import List\nx: List[int] = []'
    var_4 = var_2.parse(var_1, var_3)
    assert var_2.level == {'4': 0}
    assert var_2.doc == {'4': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_2.imp == {'4': {*()}}
    assert var_2.root == {'4': '4'}
    assert var_2.alias == {'4.List': 'typeng.List', '4.x': '[]'}
    var_5 = var_2.__eq__(var_0)
    var_6 = var_2.compile()
    assert var_6 == '\n'
    var_7 = 'CONSANT = 4'
    var_8 = var_2.parse(var_1, var_7)
    assert var_2.root == {'4': '4', '4.CONSANT': '4'}
    assert var_2.alias == {'4.List': 'typeng.List', '4.x': '[]', '4.CONSANT': '4'}
    assert var_2.const == {'4.CONSANT': 'int'}
    var_9 = var_2.compile()
    assert var_9 == '## Module `4`\n<a id="4"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSANT` | `int` |\n'
    var_10 = var_2.parse(var_1, var_1)
    var_11 = module_2.ImportFrom()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.ImportFrom'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_12 = module_0.Resolver(var_8, var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'apimd.parser.Resolver'
    assert var_12.root is None
    assert f'{type(var_12.alias).__module__}.{type(var_12.alias).__qualname__}' == 'builtins.NotImplementedType'
    assert var_12.self_ty == ''

@pytest.mark.xfail(strict=True)
def test_case_50():
    var_0 = True
    var_1 = -1943
    var_2 = 'T/m4.(\t\\\trgi{0!3]&>'
    var_3 = {var_2: var_0, var_2: var_0, var_2: var_0, var_2: var_1}
    var_4 = {}
    var_5 = {var_2: var_2}
    var_6 = module_0.Parser(var_0, var_0, var_0, var_3, imp=var_4, root=var_5, alias=var_5, const=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level is True
    assert var_6.toc is True
    assert var_6.level == {'T/m4.(\t\\\trgi{0!3]&>': -1943}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp == {}
    assert var_6.root == {'T/m4.(\t\\\trgi{0!3]&>': 'T/m4.(\t\\\trgi{0!3]&>'}
    assert var_6.alias == {'T/m4.(\t\\\trgi{0!3]&>': 'T/m4.(\t\\\trgi{0!3]&>'}
    assert var_6.const is True
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
    var_7 = 'from typing import List\nx: List[int] = []'
    var_8 = var_6.parse(var_7, var_7)
    assert var_6.level == {'T/m4.(\t\\\trgi{0!3]&>': -1943, 'from typing import List\nx: List[int] = []': 0}
    assert var_6.doc == {'from typing import List\nx: List[int] = []': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_6.imp == {'from typing import List\nx: List[int] = []': {*()}}
    assert var_6.root == {'T/m4.(\t\\\trgi{0!3]&>': 'T/m4.(\t\\\trgi{0!3]&>', 'from typing import List\nx: List[int] = []': 'from typing import List\nx: List[int] = []', 'from typing import List\nx: List[int] = [].List': 'typing.List', 'from typing import List\nx: List[int] = [].x': '[]'}
    assert var_6.alias == {'T/m4.(\t\\\trgi{0!3]&>': 'T/m4.(\t\\\trgi{0!3]&>', 'from typing import List\nx: List[int] = []': 'from typing import List\nx: List[int] = []', 'from typing import List\nx: List[int] = [].List': 'typing.List', 'from typing import List\nx: List[int] = [].x': '[]'}
    var_9 = '\nclass MyClass:\n    def method(self): pass\n'
    var_10 = var_6.parse(var_2, var_9)
    assert var_6.level == {'T/m4.(\t\\\trgi{0!3]&>': 1, 'from typing import List\nx: List[int] = []': 0, 'T/m4.(\t\\\trgi{0!3]&>.MyClass': 1, 'T/m4.(\t\\\trgi{0!3]&>.MyClass.method': 1}
    assert var_6.doc == {'from typing import List\nx: List[int] = []': '## Module `{}`\n<a id="{}"></a>\n\n', 'T/m4.(\t\\\trgi{0!3]&>': '## Module `{}`\n<a id="{}"></a>\n\n', 'T/m4.(\t\\\trgi{0!3]&>.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'T/m4.(\t\\\trgi{0!3]&>.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_6.imp == {'from typing import List\nx: List[int] = []': {*()}, 'T/m4.(\t\\\trgi{0!3]&>': {*()}}
    assert var_6.root == {'T/m4.(\t\\\trgi{0!3]&>': 'T/m4.(\t\\\trgi{0!3]&>', 'from typing import List\nx: List[int] = []': 'from typing import List\nx: List[int] = []', 'from typing import List\nx: List[int] = [].List': 'typing.List', 'from typing import List\nx: List[int] = [].x': '[]', 'T/m4.(\t\\\trgi{0!3]&>.MyClass': 'T/m4.(\t\\\trgi{0!3]&>', 'T/m4.(\t\\\trgi{0!3]&>.MyClass.method': 'T/m4.(\t\\\trgi{0!3]&>'}
    assert var_6.alias == {'T/m4.(\t\\\trgi{0!3]&>': 'T/m4.(\t\\\trgi{0!3]&>', 'from typing import List\nx: List[int] = []': 'from typing import List\nx: List[int] = []', 'from typing import List\nx: List[int] = [].List': 'typing.List', 'from typing import List\nx: List[int] = [].x': '[]', 'T/m4.(\t\\\trgi{0!3]&>.MyClass': 'T/m4.(\t\\\trgi{0!3]&>', 'T/m4.(\t\\\trgi{0!3]&>.MyClass.method': 'T/m4.(\t\\\trgi{0!3]&>'}
    var_6.compile()

@pytest.mark.xfail(strict=True)
def test_case_51():
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
    var_2 = 'from typing import List\nx: List[int] = []'
    var_3 = var_1.parse(var_0, var_2)
    assert var_1.level == {'test_module': 0}
    assert var_1.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.imp == {'test_module': {*()}}
    assert var_1.root == {'test_module': 'test_module'}
    assert var_1.alias == {'test_module.List': 'typing.List', 'test_module.x': '[]'}
    var_4 = []
    var_5 = module_2.Import(*var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Import'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_1.imports(var_3, var_5)

def test_case_52():
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
    var_4 = module_2.Load()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Load'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_5 = module_2.Name()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    var_6 = [var_5]
    var_7 = 1
    var_8 = module_2.Constant()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Constant'
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_9 = module_2.Assign()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Assign'
    assert module_2.Assign.type_comment is None
    var_10 = 'print'
    var_11 = module_2.Load()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Load'
    var_12 = module_2.Name()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Name'
    var_13 = []
    var_14 = []
    var_15 = module_2.Call(*var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Call'
    var_16 = module_2.Expr()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Expr'
    var_17 = [var_9, var_16]
    var_18 = module_0.walk_body(var_17)
    var_19 = list(var_18)
    var_20 = True
    var_21 = module_2.Constant()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'ast.Constant'
    var_22 = [var_2]
    var_23 = module_0.walk_body(var_22)
    var_24 = list(var_23)
    var_25 = [var_9]
    var_26 = []
    var_27 = [var_16]
    var_28 = []
    var_29 = module_2.Try()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'ast.Try'
    var_30 = [var_29]
    var_31 = module_0.walk_body(var_30)
    with pytest.raises(AttributeError):
        var_32 = list(var_31)