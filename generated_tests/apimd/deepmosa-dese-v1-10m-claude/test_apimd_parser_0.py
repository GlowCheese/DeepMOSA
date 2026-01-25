# Check out: https://github.com/GlowCheese/deepmosa
import ast as module_3
import dataclasses as module_2
import inspect as module_1

import apimd.parser as module_0
import pytest


def test_case_0():
    var_0 = '__init__._private'
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
    var_0 = 'module.__dict__.method'
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

def test_case_2():
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
    var_1 = 'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas': 0, 'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas.A': 0, 'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas.A.method': 0}
    assert var_0.doc == {'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas.A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas.A.method': '#### A.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@clasmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `None` |\n\n'}
    assert var_0.imp == {'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas': {*()}}
    assert var_0.root == {'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas': 'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas', 'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas.A': 'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas', 'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas.A.method': 'class A:\n    @clasmethod\n    def method(cls, x: int) -> None: pas'}

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ''
    var_1 = module_1.getdoc(var_0)
    assert var_1 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert f'{type(module_1.mod_dict).__module__}.{type(module_1.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_1.mod_dict) == 168
    assert module_1.k == 512
    assert module_1.v == 'ASYNC_GENERATOR'
    assert module_1.CO_OPTIMIZED == 1
    assert module_1.CO_NEWLOCALS == 2
    assert module_1.CO_VARARGS == 4
    assert module_1.CO_VARKEYWORDS == 8
    assert module_1.CO_NESTED == 16
    assert module_1.CO_GENERATOR == 32
    assert module_1.CO_NOFREE == 64
    assert module_1.CO_COROUTINE == 128
    assert module_1.CO_ITERABLE_COROUTINE == 256
    assert module_1.CO_ASYNC_GENERATOR == 512
    assert module_1.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_1.modulesbyfile == {}
    assert module_1.GEN_CREATED == 'GEN_CREATED'
    assert module_1.GEN_RUNNING == 'GEN_RUNNING'
    assert module_1.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_1.GEN_CLOSED == 'GEN_CLOSED'
    assert module_1.CORO_CREATED == 'CORO_CREATED'
    assert module_1.CORO_RUNNING == 'CORO_RUNNING'
    assert module_1.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_1.CORO_CLOSED == 'CORO_CLOSED'
    var_2 = module_0.Parser(var_0, toc=var_1, alias=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert var_2.level == {}
    assert var_2.doc == {}
    assert var_2.docstring == {}
    assert var_2.imp == {}
    assert var_2.root == {}
    assert var_2.alias == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
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
    var_3 = 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne:pass'
    var_2.parse(var_0, var_3)

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
    var_1 = None
    var_2 = var_0.load_docstring(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.parent(var_0)

def test_case_7():
    var_0 = 'O\nk{9\\:4K,z'
    var_1 = module_0.Resolver(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Resolver'
    assert var_1.root == 'O\nk{9\\:4K,z'
    assert var_1.alias == 'O\nk{9\\:4K,z'
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
    var_1 = "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n"
    var_2 = '|W"'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'|W"': 0, '|W".public_func': 0, '|W"._private_func': 0, '|W".PublicClass': 0}
    assert var_0.doc == {'|W"': '## Module `{}`\n<a id="{}"></a>\n\n', '|W".public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '|W"._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '|W".PublicClass': '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'|W"': {'|W".public_func', '|W".PublicClass'}}
    assert var_0.root == {'|W"': '|W"', '|W".public_func': '|W"', '|W"._private_func': '|W"', '|W".PublicClass': '|W"'}
    assert var_0.alias == {'|W".__all__': "['public_func', 'PublicClass']"}
    var_4 = var_0.compile()
    assert var_4 == '## Module `|W"`\n<a id="|w""></a>\n\n### public_func()\n\n*Full name:* `|W".public_func`\n<a id="|w"-public_func"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n### class PublicClass\n\n*Full name:* `|W".PublicClass`\n<a id="|w"-publicclass"></a>\n'

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
    var_1 = 'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass': 0, 'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass.A': 0, 'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass.A.metTod': 0}
    assert var_0.doc == {'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass.A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass.A.metTod': '#### A.metTod()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@cl ^ ssmethol` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `N / ne` |\n\n'}
    assert var_0.imp == {'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass': {*()}}
    assert var_0.root == {'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass': 'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass', 'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass.A': 'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass', 'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass.A.metTod': 'class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass`\n<a id="class a:\n    @cl^ssmethol\n    def mettod(cls, x: int) -> n/ne: eass"></a>\n\n### class A\n\n*Full name:* `class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass.A`\n<a id="class a:\n    @cl^ssmethol\n    def mettod(cls, x: int) -> n/ne: eass-a"></a>\n\n#### A.metTod()\n\n*Full name:* `class A:\n    @cl^ssmethol\n    def metTod(cls, x: int) -> N/ne: eass.A.metTod`\n<a id="class a:\n    @cl^ssmethol\n    def mettod(cls, x: int) -> n/ne: eass-a-mettod"></a>\n\n| Decorators |\n|:----------:|\n| `@cl ^ ssmethol` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `N / ne` |\n'

def test_case_13():
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
    var_6 = module_2.field(compare=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'dataclasses.Field'
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_2.Field.compare).__module__}.{type(module_2.Field.compare).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.default).__module__}.{type(module_2.Field.default).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.default_factory).__module__}.{type(module_2.Field.default_factory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.hash).__module__}.{type(module_2.Field.hash).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.init).__module__}.{type(module_2.Field.init).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.kw_only).__module__}.{type(module_2.Field.kw_only).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.metadata).__module__}.{type(module_2.Field.metadata).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.name).__module__}.{type(module_2.Field.name).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.repr).__module__}.{type(module_2.Field.repr).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.type).__module__}.{type(module_2.Field.type).__qualname__}' == 'builtins.member_descriptor'
    var_7 = '^}U3QOqn7\r8P'
    var_8 = module_0.is_public_family(var_7)
    assert var_8 is True
    var_9 = module_0.doctest(var_7)
    assert var_9 == '^}U3QOqn7\n8P'

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
    var_1 = "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n": 0, "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.public_func": 0, "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n._private_func": 0, "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.PublicClass": 0}
    assert var_0.doc == {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.PublicClass": '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n": {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.PublicClass", "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.public_func"}}
    assert var_0.root == {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n": "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.public_func": "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n._private_func": "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.PublicClass": "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n"}
    assert var_0.alias == {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.__all__": "['public_func', 'PublicClass']"}

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
    var_0.imports(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = 'Nc|AO:JOewl]'
    var_2 = 'E'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = module_3.Dict(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Dict'
    assert var_4.Nc|AO:JOewl] is None
    assert var_4.E is None
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    module_0.const_type(var_4)

def test_case_17():
    var_0 = False
    var_1 = 1
    var_2 = module_0.Parser(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is False
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
    var_3 = 'def func(): pass'
    var_4 = module_3.parse(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Module'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'builtins.list'
    assert len(var_4.body) == 1
    assert var_4.type_ignores == []
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_5 = var_4.body[var_0]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    assert var_2.level == {'test_module': 0, 'test_module.func': 0}
    assert var_2.doc == {'test_module': '## Module `{}`\n\n', 'test_module.func': '### func()\n\n*Full name:* `{}`\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_2.imp == {'test_module': {*()}}
    assert var_2.root == {'test_module': 'test_module', 'test_module.func': 'test_module'}
    var_8 = var_2.api(var_6, var_5)

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
    var_1 = var_0.globals(var_0, var_0)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'

def test_case_20():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = module_3.Call(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Call'
    assert var_2.func is None
    assert var_2.args is None
    assert var_2.keywords is None
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
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
    var_4 = module_2.field(repr=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'dataclasses.Field'
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_2.Field.compare).__module__}.{type(module_2.Field.compare).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.default).__module__}.{type(module_2.Field.default).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.default_factory).__module__}.{type(module_2.Field.default_factory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.hash).__module__}.{type(module_2.Field.hash).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.init).__module__}.{type(module_2.Field.init).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.kw_only).__module__}.{type(module_2.Field.kw_only).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.metadata).__module__}.{type(module_2.Field.metadata).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.name).__module__}.{type(module_2.Field.name).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.repr).__module__}.{type(module_2.Field.repr).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.type).__module__}.{type(module_2.Field.type).__qualname__}' == 'builtins.member_descriptor'
    var_5 = '|'
    var_6 = module_0.esc_underscore(var_5)
    assert var_6 == '|'

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
    var_5 = module_2.field(default_factory=var_1, metadata=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'dataclasses.Field'
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_2.Field.compare).__module__}.{type(module_2.Field.compare).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.default).__module__}.{type(module_2.Field.default).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.default_factory).__module__}.{type(module_2.Field.default_factory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.hash).__module__}.{type(module_2.Field.hash).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.init).__module__}.{type(module_2.Field.init).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.kw_only).__module__}.{type(module_2.Field.kw_only).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.metadata).__module__}.{type(module_2.Field.metadata).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.name).__module__}.{type(module_2.Field.name).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.repr).__module__}.{type(module_2.Field.repr).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.type).__module__}.{type(module_2.Field.type).__qualname__}' == 'builtins.member_descriptor'
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
    var_10 = module_2.field(hash=var_0, compare=var_4, metadata=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'dataclasses.Field'
    var_11 = module_0.Resolver(var_7, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'apimd.parser.Resolver'
    assert var_11.root == '<6D(o\t$'
    assert f'{type(var_11.alias).__module__}.{type(var_11.alias).__qualname__}' == 'dataclasses.Field'
    assert var_11.self_ty == ''
    var_12 = ')Rna-|g'
    var_13 = module_0.esc_underscore(var_12)
    assert var_13 == ')Rna-|g'
    var_14 = module_1.getdoc(var_8)
    assert var_14 == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
    assert f'{type(module_1.mod_dict).__module__}.{type(module_1.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_1.mod_dict) == 168
    assert module_1.k == 512
    assert module_1.v == 'ASYNC_GENERATOR'
    assert module_1.CO_OPTIMIZED == 1
    assert module_1.CO_NEWLOCALS == 2
    assert module_1.CO_VARARGS == 4
    assert module_1.CO_VARKEYWORDS == 8
    assert module_1.CO_NESTED == 16
    assert module_1.CO_GENERATOR == 32
    assert module_1.CO_NOFREE == 64
    assert module_1.CO_COROUTINE == 128
    assert module_1.CO_ITERABLE_COROUTINE == 256
    assert module_1.CO_ASYNC_GENERATOR == 512
    assert module_1.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_1.modulesbyfile == {}
    assert module_1.GEN_CREATED == 'GEN_CREATED'
    assert module_1.GEN_RUNNING == 'GEN_RUNNING'
    assert module_1.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_1.GEN_CLOSED == 'GEN_CLOSED'
    assert module_1.CORO_CREATED == 'CORO_CREATED'
    assert module_1.CORO_RUNNING == 'CORO_RUNNING'
    assert module_1.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_1.CORO_CLOSED == 'CORO_CLOSED'
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
    var_1 = '\nclass TestClass:\n   publ\nc_attr: it\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass TestClass:\n   publ\nc_attr: it\n': 0, '\nclass TestClass:\n   publ\nc_attr: it\n.TestClass': 0}
    assert var_0.doc == {'\nclass TestClass:\n   publ\nc_attr: it\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass TestClass:\n   publ\nc_attr: it\n.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nclass TestClass:\n   publ\nc_attr: it\n': {*()}}
    assert var_0.root == {'\nclass TestClass:\n   publ\nc_attr: it\n': '\nclass TestClass:\n   publ\nc_attr: it\n', '\nclass TestClass:\n   publ\nc_attr: it\n.TestClass': '\nclass TestClass:\n   publ\nc_attr: it\n'}

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    var_6 = module_2.field(default_factory=var_2, metadata=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'dataclasses.Field'
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    assert f'{type(module_2.Field.compare).__module__}.{type(module_2.Field.compare).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.default).__module__}.{type(module_2.Field.default).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.default_factory).__module__}.{type(module_2.Field.default_factory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.hash).__module__}.{type(module_2.Field.hash).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.init).__module__}.{type(module_2.Field.init).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.kw_only).__module__}.{type(module_2.Field.kw_only).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.metadata).__module__}.{type(module_2.Field.metadata).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.name).__module__}.{type(module_2.Field.name).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.repr).__module__}.{type(module_2.Field.repr).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Field.type).__module__}.{type(module_2.Field.type).__qualname__}' == 'builtins.member_descriptor'
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
    var_1 = "\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n": 0, "\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n.Child": 0}
    assert var_0.doc == {"\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n.Child": '### class Child\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `Parent1` |\n| `Parent2` |\n\n'}
    assert var_0.docstring == {"\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n.Child": 'Child class'}
    assert var_0.imp == {"\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n": {*()}}
    assert var_0.root == {"\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n": "\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n", "\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n.Child": "\nclass Child(Parent1, Parent2):\n    '''Child class'''\n    pass\n"}

def test_case_25():
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
    var_1 = "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n": 0, "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.public_func": 0, "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n._private_func": 0, "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.PublicClass": 0}
    assert var_0.doc == {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.PublicClass": '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n": {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.PublicClass", "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.public_func"}}
    assert var_0.root == {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n": "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.public_func": "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n._private_func": "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.PublicClass": "\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n"}
    assert var_0.alias == {"\n__all__ = ['public_func', 'PublicClass']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.__all__": "['public_func', 'PublicClass']"}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\n__all__ = [\'public_func\', \'PublicClass\']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n`\n<a id="\n__all__ = [\'public_func\', \'publicclass\']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass publicclass:\n    pass\n"></a>\n\n### public_func()\n\n*Full name:* `\n__all__ = [\'public_func\', \'PublicClass\']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.public_func`\n<a id="\n__all__ = [\'public_func\', \'publicclass\']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass publicclass:\n    pass\n-public_func"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n### class PublicClass\n\n*Full name:* `\n__all__ = [\'public_func\', \'PublicClass\']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    pass\n.PublicClass`\n<a id="\n__all__ = [\'public_func\', \'publicclass\']\n\ndef public_func():\n    pass\n\ndef _private_func():\n    pass\n\nclass publicclass:\n    pass\n-publicclass"></a>\n'

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
    var_1 = 'func1'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'func1': 0}
    assert var_0.doc == {'func1': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'func1': {*()}}
    assert var_0.root == {'func1': 'func1'}

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
    var_1 = 'unc2'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'unc2': 0}
    assert var_0.doc == {'unc2': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'unc2': {*()}}
    assert var_0.root == {'unc2': 'unc2'}
    var_3 = var_0.load_docstring(var_1, var_1)

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
    var_1 = 'unc2'
    var_2 = module_3.Set()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Set'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_3 = var_0.parse(var_1, var_1)
    assert var_0.level == {'unc2': 0}
    assert var_0.doc == {'unc2': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'unc2': {*()}}
    assert var_0.root == {'unc2': 'unc2'}
    var_4 = var_0.compile()
    assert var_4 == '\n'

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
    var_1 = '\nclass TestClass:\n    public_attr: t\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass TestClass:\n    public_attr: t\n': 0, '\nclass TestClass:\n    public_attr: t\n.TestClass': 0}
    assert var_0.doc == {'\nclass TestClass:\n    public_attr: t\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass TestClass:\n    public_attr: t\n.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `public_attr` | `t` |\n\n'}
    assert var_0.imp == {'\nclass TestClass:\n    public_attr: t\n': {*()}}
    assert var_0.root == {'\nclass TestClass:\n    public_attr: t\n': '\nclass TestClass:\n    public_attr: t\n', '\nclass TestClass:\n    public_attr: t\n.TestClass': '\nclass TestClass:\n    public_attr: t\n'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nclass TestClass:\n    public_attr: t\n`\n<a id="\nclass testclass:\n    public_attr: t\n"></a>\n\n### class TestClass\n\n*Full name:* `\nclass TestClass:\n    public_attr: t\n.TestClass`\n<a id="\nclass testclass:\n    public_attr: t\n-testclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `public_attr` | `t` |\n'
    var_4 = var_0.load_docstring(var_3, var_0)

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
    var_1 = 'package.subpackage.module'
    var_2 = module_3.Load()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Load'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_3 = 'y[ee{HB?M7_ ehg'
    var_4 = module_3.parse(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Module'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'builtins.list'
    assert len(var_4.body) == 1
    assert var_4.type_ignores == []
    var_5 = var_0.resolve(var_3, var_4, var_1)
    assert var_5 == 'package.subpackage.module'

def test_case_31():
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
    var_1 = '\nDEBUG = True\nMAX_SIZE: int = 100\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nDEBUG = True\nMAX_SIZE: int = 100\n': 0}
    assert var_0.doc == {'\nDEBUG = True\nMAX_SIZE: int = 100\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nDEBUG = True\nMAX_SIZE: int = 100\n': {*()}}
    assert var_0.root == {'\nDEBUG = True\nMAX_SIZE: int = 100\n': '\nDEBUG = True\nMAX_SIZE: int = 100\n', '\nDEBUG = True\nMAX_SIZE: int = 100\n.DEBUG': '\nDEBUG = True\nMAX_SIZE: int = 100\n', '\nDEBUG = True\nMAX_SIZE: int = 100\n.MAX_SIZE': '\nDEBUG = True\nMAX_SIZE: int = 100\n'}
    assert var_0.alias == {'\nDEBUG = True\nMAX_SIZE: int = 100\n.DEBUG': 'True', '\nDEBUG = True\nMAX_SIZE: int = 100\n.MAX_SIZE': '100'}
    assert var_0.const == {'\nDEBUG = True\nMAX_SIZE: int = 100\n.DEBUG': 'bool', '\nDEBUG = True\nMAX_SIZE: int = 100\n.MAX_SIZE': 'int'}

def test_case_32():
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
    var_1 = "def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass"
    var_2 = module_3.parse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Module'
    assert f'{type(var_2.body).__module__}.{type(var_2.body).__qualname__}' == 'builtins.list'
    assert len(var_2.body) == 1
    assert var_2.type_ignores == []
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_3 = var_0.compile()
    assert var_3 == '\n'
    var_4 = var_0.parse(var_1, var_1)
    assert var_0.level == {"def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass": 0, "def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass.foo": 0}
    assert var_0.doc == {"def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass": '## Module `{}`\n<a id="{}"></a>\n\n', "def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | *args | c | **kwargs | return |\n|:---:|:---:|:-----:|:---:|:--------:|:------:|\n| `int` | `str` | `float` | `bool` | `dict` | `list` |\n|   | `\'x\'` |   | `True` |   |   |\n\n'}
    assert var_0.imp == {"def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass": {*()}}
    assert var_0.root == {"def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass": "def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass", "def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass.foo": "def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass"}
    var_5 = var_0.resolve(var_4, var_2, var_4)
    assert var_5 == 'def foo(a: int, b: str=x, *args: float, c: bool=True, **kwargs: dict) -> list:\n    pass'

def test_case_33():
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
    var_1 = 'def foo(a: int, /, b: str) -> None: pass'
    var_2 = module_3.parse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Module'
    assert f'{type(var_2.body).__module__}.{type(var_2.body).__qualname__}' == 'builtins.list'
    assert len(var_2.body) == 1
    assert var_2.type_ignores == []
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test'
    var_6 = var_0.parse(var_5, var_1)
    assert var_0.level == {'test': 0, 'test.foo': 0}
    assert var_0.doc == {'test': '## Module `{}`\n<a id="{}"></a>\n\n', 'test.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | / | b | return |\n|:---:|:---:|:---:|:------:|\n| `int` | `Any` | `str` | `None` |\n\n'}
    assert var_0.imp == {'test': {*()}}
    assert var_0.root == {'test': 'test', 'test.foo': 'test'}
    var_7 = var_4.args
    var_8 = var_4.returns

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
    var_1 = 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass': 0, 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass.A': 0, 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass.A.method': 0}
    assert var_0.doc == {'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass.A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass.A.method': '#### A.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `None` |\n\n'}
    assert var_0.imp == {'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass': {*()}}
    assert var_0.root == {'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass': 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass', 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass.A': 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass', 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass.A.method': 'class A:\n    @classmethod\n    def method(cls, x: int) -> None: pass'}

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
    var_1 = '\nclass TestClass:\n    public_attr: it\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass TestClass:\n    public_attr: it\n': 0, '\nclass TestClass:\n    public_attr: it\n.TestClass': 0}
    assert var_0.doc == {'\nclass TestClass:\n    public_attr: it\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass TestClass:\n    public_attr: it\n.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `public_attr` | `it` |\n\n'}
    assert var_0.imp == {'\nclass TestClass:\n    public_attr: it\n': {*()}}
    assert var_0.root == {'\nclass TestClass:\n    public_attr: it\n': '\nclass TestClass:\n    public_attr: it\n', '\nclass TestClass:\n    public_attr: it\n.TestClass': '\nclass TestClass:\n    public_attr: it\n'}

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
    var_1 = "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n": 0, "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n.MyClass": 0, "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n.MyClass.method": 0}
    assert var_0.doc == {"\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n.MyClass": '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n.MyClass.method": '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.docstring == {"\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n.MyClass": 'Class docstring', "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n.MyClass.method": 'Method docstring'}
    assert var_0.imp == {"\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n": {*()}}
    assert var_0.root == {"\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n": "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n", "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n.MyClass": "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n", "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n.MyClass.method": "\nclass MyClass:\n    '''Class docstring'''\n    def method(self):\n        '''Method docstring'''\n        pass\n"}

def test_case_37():
    var_0 = True
    var_1 = False
    var_2 = module_0.Parser(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level is True
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
    var_3 = 'async def async_func(): pass'
    var_4 = module_3.parse(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Module'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'builtins.list'
    assert len(var_4.body) == 1
    assert var_4.type_ignores == []
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_5 = var_4.body[var_1]
    var_6 = 'test_module'
    var_7 = var_2.parse(var_6, var_3)
    assert var_2.level == {'test_module': 0, 'test_module.async_func': 0}
    assert var_2.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.async_func': '### async async_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_2.imp == {'test_module': {*()}}
    assert var_2.root == {'test_module': 'test_module', 'test_module.async_func': 'test_module'}
    var_8 = module_0.is_public_family(var_3)
    assert var_8 is True

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
    var_1 = 'ntfo3>'
    var_2 = module_3.ImportFrom()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.ImportFrom'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    assert module_3.ImportFrom.module is None
    assert module_3.ImportFrom.level is None
    var_3 = var_0.imports(var_1, var_2)
    var_0.imports(var_3, var_3)

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
    var_1 = "def foo(a: int, b: str = 'x', *args: float, c: bool = True, **kwargs: dict) -> list: pass"
    var_2 = module_3.parse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Module'
    assert f'{type(var_2.body).__module__}.{type(var_2.body).__qualname__}' == 'builtins.list'
    assert len(var_2.body) == 1
    assert var_2.type_ignores == []
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_3 = var_0.resolve(var_1, var_2)
    assert var_3 == 'def foo(a: int, b: str=x, *args: float, c: bool=True, **kwargs: dict) -> list:\n    pass'

def test_case_40():
    var_0 = 'x = 1\nif True:\n    y = 2\n    try:\n        z = 3\n    except:\n        w = 4\nelse:\n    a = 5'
    var_1 = module_3.parse(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'ast.Module'
    assert f'{type(var_1.body).__module__}.{type(var_1.body).__qualname__}' == 'builtins.list'
    assert len(var_1.body) == 2
    assert var_1.type_ignores == []
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_2 = var_1.body
    var_3 = module_0.walk_body(var_2)
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
    var_4 = list(var_3)
    var_5 = len(var_4)

def test_case_41():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_3.List(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.List'
    assert var_2.elts is None
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_3 = module_0.const_type(var_2)
    assert var_3 == 'list'
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
def test_case_42():
    var_0 = False
    var_1 = module_0.Parser(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is False
    assert var_1.b_level is False
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
    var_2 = 'enum.Enum'
    var_3 = 'eval'
    var_4 = module_3.parse(var_2, mode=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Expression'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'ast.Attribute'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_5 = var_4.body
    var_6 = [var_5]
    var_7 = 'MEMBER: int = 1'
    var_8 = module_3.parse(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Module'
    assert f'{type(var_8.body).__module__}.{type(var_8.body).__qualname__}' == 'builtins.list'
    assert len(var_8.body) == 1
    assert var_8.type_ignores == []
    var_1.class_api(var_7, var_2, var_6, var_3)

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
    var_1 = '\nclass TestClass:\n    public_attr: it\n'
    var_2 = [var_0, var_0]
    var_3 = module_3.Subscript(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Subscript'
    assert f'{type(var_3.value).__module__}.{type(var_3.value).__qualname__}' == 'apimd.parser.Parser'
    assert f'{type(var_3.slice).__module__}.{type(var_3.slice).__qualname__}' == 'apimd.parser.Parser'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_0.resolve(var_1, var_3)

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
    var_1 = '\nfrom collections import defaultdict\nfrom typing import List\nimport sys\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nfrom collections import defaultdict\nfrom typing import List\nimport sys\n': 0}
    assert var_0.doc == {'\nfrom collections import defaultdict\nfrom typing import List\nimport sys\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nfrom collections import defaultdict\nfrom typing import List\nimport sys\n': {*()}}
    assert var_0.root == {'\nfrom collections import defaultdict\nfrom typing import List\nimport sys\n': '\nfrom collections import defaultdict\nfrom typing import List\nimport sys\n'}
    assert var_0.alias == {'\nfrom collections import defaultdict\nfrom typing import List\nimport sys\n.defaultdict': 'collections.defaultdict', '\nfrom collections import defaultdict\nfrom typing import List\nimport sys\n.List': 'typing.List', '\nfrom collections import defaultdict\nfrom typing import List\nimport sys\n.sys': 'sys'}

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
    var_1 = 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass': 0, 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A': 0, 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A.method': 0}
    assert var_0.doc == {'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A.method': '#### A.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `N * ne` |\n\n'}
    assert var_0.imp == {'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass': {*()}}
    assert var_0.root == {'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass': 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass', 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A': 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass', 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A.method': 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass'}
    var_3 = module_3.Assign()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Assign'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    assert module_3.Assign.type_comment is None
    var_4 = []
    var_5 = [var_3, var_1]
    var_0.class_api(var_5, var_5, var_4, var_5)

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
    var_1 = '\nclass TestClass:\n    public_attr: it\n'
    var_2 = var_0.__repr__()
    assert var_2 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_3 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass TestClass:\n    public_attr: it\n': 0, '\nclass TestClass:\n    public_attr: it\n.TestClass': 0}
    assert var_0.doc == {'\nclass TestClass:\n    public_attr: it\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass TestClass:\n    public_attr: it\n.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `public_attr` | `it` |\n\n'}
    assert var_0.imp == {'\nclass TestClass:\n    public_attr: it\n': {*()}}
    assert var_0.root == {'\nclass TestClass:\n    public_attr: it\n': '\nclass TestClass:\n    public_attr: it\n', '\nclass TestClass:\n    public_attr: it\n.TestClass': '\nclass TestClass:\n    public_attr: it\n'}
    var_4 = '\nclass TestClass:\n   publ\nc_attr: it\n'
    var_5 = '.M"\\o!$qKG#'
    var_6 = var_0.parse(var_5, var_4)
    assert var_0.level == {'\nclass TestClass:\n    public_attr: it\n': 0, '\nclass TestClass:\n    public_attr: it\n.TestClass': 0, '.M"\\o!$qKG#': 1, '.M"\\o!$qKG#.TestClass': 1}
    assert var_0.doc == {'\nclass TestClass:\n    public_attr: it\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass TestClass:\n    public_attr: it\n.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `public_attr` | `it` |\n\n', '.M"\\o!$qKG#': '## Module `{}`\n<a id="{}"></a>\n\n', '.M"\\o!$qKG#.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nclass TestClass:\n    public_attr: it\n': {*()}, '.M"\\o!$qKG#': {*()}}
    assert var_0.root == {'\nclass TestClass:\n    public_attr: it\n': '\nclass TestClass:\n    public_attr: it\n', '\nclass TestClass:\n    public_attr: it\n.TestClass': '\nclass TestClass:\n    public_attr: it\n', '.M"\\o!$qKG#': '.M"\\o!$qKG#', '.M"\\o!$qKG#.TestClass': '.M"\\o!$qKG#'}
    var_7 = var_0.compile()
    assert var_7 == '## Module `\nclass TestClass:\n    public_attr: it\n`\n<a id="\nclass testclass:\n    public_attr: it\n"></a>\n\n### class TestClass\n\n*Full name:* `\nclass TestClass:\n    public_attr: it\n.TestClass`\n<a id="\nclass testclass:\n    public_attr: it\n-testclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `public_attr` | `it` |\n\n## Module `.M"\\o!$qKG#`\n<a id="-m"\\o!$qkg#"></a>\n\n### class TestClass\n\n*Full name:* `.M"\\o!$qKG#.TestClass`\n<a id="-m"\\o!$qkg#-testclass"></a>\n'
    var_8 = [var_2]
    var_9 = module_3.Tuple(*var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Tuple'
    assert var_9.elts == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    assert f'{type(module_3.Tuple.dims).__module__}.{type(module_3.Tuple.dims).__qualname__}' == 'builtins.property'
    var_10 = module_0.const_type(var_9)
    assert var_10 == 'tuple'
    var_0.imports(var_3, var_9)

def test_case_47():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'keys'
    var_4 = module_0._attr(var_2, var_3)
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
    var_5 = callable(var_4)

def test_case_48():
    var_0 = module_0._e_type()
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

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = {}
    var_1 = None
    var_2 = ' ]xl*Jj '
    var_3 = module_3.Delete()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Delete'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_4 = [var_3, var_3]
    var_5 = None
    var_6 = module_0.Parser(imp=var_5, root=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp is None
    assert var_6.root is None
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
    var_6.class_api(var_1, var_2, var_0, var_4)

def test_case_50():
    var_0 = None
    var_1 = ''
    var_2 = {var_1: var_1, var_1: var_1, var_1: var_1, var_1: var_1}
    var_3 = module_0.Parser(var_2, toc=var_0, alias=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link == {'': ''}
    assert var_3.b_level == 1
    assert var_3.toc is None
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {'': ''}
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
    var_4 = var_3.parse(var_1, var_1)
    assert var_3.level == {'': 0}
    assert var_3.doc == {'': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_3.imp == {'': {*()}}
    assert var_3.root == {'': ''}
    var_5 = var_3.compile()
    assert var_5 == '\n'
    assert var_3.docstring == {'': ''}

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
    var_1 = "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.publicfunc": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n._pratefunc": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.PublcClass": 0}
    assert var_0.doc == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.publicfunc": '### publicfunc()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n._pratefunc": '### _pratefunc()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.PublcClass": '### class PublcClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.PublicClass", "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.public_func"}}
    assert var_0.root == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.publicfunc": "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n._pratefunc": "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.PublcClass": "\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n"}
    assert var_0.alias == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.__all__": "['public_func', 'PublicClass']"}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\n__all__ = [\'public_func\', \'PublicClass\']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n`\n<a id="\n__all__ = [\'public_func\', \'publicclass\']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass publcclass:\n    pass\n"></a>\n'
    var_4 = 'a2I8\t:Nh#uQ-'
    var_5 = module_0.const_type(var_2)
    assert var_5 == 'Any'
    var_6 = {var_1: var_4}
    var_7 = module_0.Parser(var_2, docstring=var_6, root=var_6, alias=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is None
    assert var_7.b_level == 1
    assert var_7.toc is False
    assert var_7.level == {}
    assert var_7.doc == {}
    assert var_7.docstring == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": 'a2I8\t:Nh#uQ-'}
    assert var_7.imp == {}
    assert var_7.root == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": 'a2I8\t:Nh#uQ-'}
    assert var_7.alias == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": 'a2I8\t:Nh#uQ-'}
    assert var_7.const == {}
    var_8 = var_7.parse(var_4, var_4)
    assert var_7.level == {'a2I8\t:Nh#uQ-': 0}
    assert var_7.doc == {'a2I8\t:Nh#uQ-': '## Module `{}`\n\n'}
    assert var_7.docstring == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": 'a2I8\t:Nh#uQ-', 'a2I8\t:Nh#uQ-': 'a2I8\t:Nh#uQ-'}
    assert var_7.imp == {'a2I8\t:Nh#uQ-': {*()}}
    assert var_7.root == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": 'a2I8\t:Nh#uQ-', 'a2I8\t:Nh#uQ-': 'a2I8\t:Nh#uQ-'}
    assert var_7.alias == {"\n__all__ = ['public_func', 'PublicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": 'a2I8\t:Nh#uQ-', 'a2I8\t:Nh#uQ-': 'a2I8\t:Nh#uQ-'}
    var_7.compile()

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
    var_1 = "\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n": 0, "\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n.foo": 0}
    assert var_0.doc == {"\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {"\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n": 'Module docstring', "\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n.foo": 'Function Socstring'}
    assert var_0.imp == {"\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n": {*()}}
    assert var_0.root == {"\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n": "\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n", "\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n.foo": "\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n"}
    assert var_0.alias == {"\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n.os": 'os', "\n'''Module docstring'''\nimport os\nx = 5\ndef foo():\n    '''Function Socstring'''\n    pass\n.x": '5'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\n\'\'\'Module docstring\'\'\'\nimport os\nx = 5\ndef foo():\n    \'\'\'Function Socstring\'\'\'\n    pass\n`\n<a id="\n\'\'\'module docstring\'\'\'\nimport os\nx = 5\ndef foo():\n    \'\'\'function socstring\'\'\'\n    pass\n"></a>\n\nModule docstring\n\n### foo()\n\n*Full name:* `\n\'\'\'Module docstring\'\'\'\nimport os\nx = 5\ndef foo():\n    \'\'\'Function Socstring\'\'\'\n    pass\n.foo`\n<a id="\n\'\'\'module docstring\'\'\'\nimport os\nx = 5\ndef foo():\n    \'\'\'function socstring\'\'\'\n    pass\n-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n\nFunction Socstring\n'

def test_case_53():
    var_0 = None
    var_1 = '*&'
    var_2 = {var_1: var_1, var_1: var_1}
    var_3 = module_0.Parser(var_2, toc=var_0, alias=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link == {'*&': '*&'}
    assert var_3.b_level == 1
    assert var_3.toc is None
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {'*&': '*&'}
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
    var_4 = 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass'
    var_5 = var_3.parse(var_1, var_4)
    assert var_3.level == {'*&': 0, '*&.A': 0, '*&.A.method': 0}
    assert var_3.doc == {'*&': '## Module `{}`\n<a id="{}"></a>\n\n', '*&.A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '*&.A.method': '#### A.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `N * ne` |\n\n'}
    assert var_3.imp == {'*&': {*()}}
    assert var_3.root == {'*&': '*&', '*&.A': '*&', '*&.A.method': '*&'}
    var_6 = var_3.parse(var_4, var_4)
    assert var_3.level == {'*&': 0, '*&.A': 0, '*&.A.method': 0, 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass': 0, 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A': 0, 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A.method': 0}
    assert var_3.doc == {'*&': '## Module `{}`\n<a id="{}"></a>\n\n', '*&.A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '*&.A.method': '#### A.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `N * ne` |\n\n', 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A.method': '#### A.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `N * ne` |\n\n'}
    assert var_3.imp == {'*&': {*()}, 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass': {*()}}
    assert var_3.root == {'*&': '*&', '*&.A': '*&', '*&.A.method': '*&', 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass': 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass', 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A': 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass', 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A.method': 'class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass'}
    var_7 = module_0.code(var_4)
    assert var_7 == '`class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass`'
    var_8 = var_3.compile()
    assert var_8 == '## Module `*&`\n<a id="*&"></a>\n\n### class A\n\n*Full name:* `*&.A`\n<a id="*&-a"></a>\n\n#### A.method()\n\n*Full name:* `*&.A.method`\n<a id="*&-a-method"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `N * ne` |\n\n## Module `class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass`\n<a id="class a:\n    @classmethod\n    def method(cls, x: int) -> n*ne: pass"></a>\n\n### class A\n\n*Full name:* `class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A`\n<a id="class a:\n    @classmethod\n    def method(cls, x: int) -> n*ne: pass-a"></a>\n\n#### A.method()\n\n*Full name:* `class A:\n    @classmethod\n    def method(cls, x: int) -> N*ne: pass.A.method`\n<a id="class a:\n    @classmethod\n    def method(cls, x: int) -> n*ne: pass-a-method"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `N * ne` |\n'
    assert var_3.docstring == {'*&': '', '*&.A': '', '*&.A.method': ''}
    var_9 = var_3.__repr__()
    assert var_9 == 'Parser(link={\'*&\': \'*&\'}, b_level=1, toc=None, level={\'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass\': 0, \'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass.A\': 0, \'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass.A.method\': 0, \'*&\': 0, \'*&.A\': 0, \'*&.A.method\': 0}, doc={\'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass.A\': \'### class A\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass.A.method\': \'#### A.method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@classmethod` |\\n\\n| cls | x | return |\\n|:---:|:---:|:------:|\\n| `type[Self]` | `int` | `N * ne` |\\n\\n\', \'*&\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'*&.A\': \'### class A\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'*&.A.method\': \'#### A.method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@classmethod` |\\n\\n| cls | x | return |\\n|:---:|:---:|:------:|\\n| `type[Self]` | `int` | `N * ne` |\\n\\n\'}, docstring={\'*&\': \'\', \'*&.A\': \'\', \'*&.A.method\': \'\'}, imp={\'*&\': set(), \'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass\': set()}, root={\'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass\': \'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass\', \'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass.A\': \'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass\', \'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass.A.method\': \'class A:\\n    @classmethod\\n    def method(cls, x: int) -> N*ne: pass\', \'*&\': \'*&\', \'*&.A\': \'*&\', \'*&.A.method\': \'*&\'}, alias={\'*&\': \'*&\'}, const={})'
    var_10 = module_3.Assign()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Assign'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    assert module_3.Assign.type_comment is None

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
    var_1 = '\nclass Testvlass:\n    public_attr: it\n'
    var_2 = None
    var_3 = var_0.globals(var_2, var_2)
    var_4 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass Testvlass:\n    public_attr: it\n': 0, '\nclass Testvlass:\n    public_attr: it\n.Testvlass': 0}
    assert var_0.doc == {'\nclass Testvlass:\n    public_attr: it\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass Testvlass:\n    public_attr: it\n.Testvlass': '### class Testvlass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `public_attr` | `it` |\n\n'}
    assert var_0.imp == {'\nclass Testvlass:\n    public_attr: it\n': {*()}}
    assert var_0.root == {'\nclass Testvlass:\n    public_attr: it\n': '\nclass Testvlass:\n    public_attr: it\n', '\nclass Testvlass:\n    public_attr: it\n.Testvlass': '\nclass Testvlass:\n    public_attr: it\n'}
    var_5 = '\nclass TestClass:\n   publ\nc_attr: i\n'
    var_6 = '._LM"\\!$qKG#'
    var_7 = var_0.parse(var_6, var_5)
    assert var_0.level == {'\nclass Testvlass:\n    public_attr: it\n': 0, '\nclass Testvlass:\n    public_attr: it\n.Testvlass': 0, '._LM"\\!$qKG#': 1, '._LM"\\!$qKG#.TestClass': 1}
    assert var_0.doc == {'\nclass Testvlass:\n    public_attr: it\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass Testvlass:\n    public_attr: it\n.Testvlass': '### class Testvlass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `public_attr` | `it` |\n\n', '._LM"\\!$qKG#': '## Module `{}`\n<a id="{}"></a>\n\n', '._LM"\\!$qKG#.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nclass Testvlass:\n    public_attr: it\n': {*()}, '._LM"\\!$qKG#': {*()}}
    assert var_0.root == {'\nclass Testvlass:\n    public_attr: it\n': '\nclass Testvlass:\n    public_attr: it\n', '\nclass Testvlass:\n    public_attr: it\n.Testvlass': '\nclass Testvlass:\n    public_attr: it\n', '._LM"\\!$qKG#': '._LM"\\!$qKG#', '._LM"\\!$qKG#.TestClass': '._LM"\\!$qKG#'}
    var_8 = var_0.compile()
    assert var_8 == '## Module `\nclass Testvlass:\n    public_attr: it\n`\n<a id="\nclass testvlass:\n    public_attr: it\n"></a>\n\n### class Testvlass\n\n*Full name:* `\nclass Testvlass:\n    public_attr: it\n.Testvlass`\n<a id="\nclass testvlass:\n    public_attr: it\n-testvlass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `public_attr` | `it` |\n'
    var_9 = module_0.esc_underscore(var_6)
    assert var_9 == '._LM"\\!$qKG#'
    var_10 = [var_9]
    var_11 = var_0.load_docstring(var_6, var_6)
    var_12 = var_0.compile()
    assert var_12 == '## Module `\nclass Testvlass:\n    public_attr: it\n`\n<a id="\nclass testvlass:\n    public_attr: it\n"></a>\n\n### class Testvlass\n\n*Full name:* `\nclass Testvlass:\n    public_attr: it\n.Testvlass`\n<a id="\nclass testvlass:\n    public_attr: it\n-testvlass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `public_attr` | `it` |\n'
    var_13 = module_3.ImportFrom(*var_10)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.ImportFrom'
    assert var_13.module == '._LM"\\!$qKG#'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    assert module_3.ImportFrom.module is None
    assert module_3.ImportFrom.level is None
    var_0.imports(var_2, var_13)

def test_case_55():
    var_0 = True
    var_1 = module_0.Parser(var_0)
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
    var_2 = "'''Module doc'''"
    var_3 = var_1.parse(var_2, var_2)
    assert var_1.level == {"'''Module doc'''": 0}
    assert var_1.doc == {"'''Module doc'''": '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.docstring == {"'''Module doc'''": 'Module doc'}
    assert var_1.imp == {"'''Module doc'''": {*()}}
    assert var_1.root == {"'''Module doc'''": "'''Module doc'''"}

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
    var_1 = '\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n': 0, '\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n.foo': 0}
    assert var_0.doc == {'\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `Any` | `Any` | `Any` |\n\n'}
    assert var_0.imp == {'\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n': {*()}}
    assert var_0.root == {'\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n': '\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n', '\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n.foo': '\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n'}
    assert var_0.alias == {'\nx = 5  # type: int\ndef foo(a, b):  # type: (int, str) -> bool\n    return True\n.x': '5'}

def test_case_57():
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
    var_2 = 'x = y = 5'
    var_3 = module_3.parse(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Module'
    assert f'{type(var_3.body).__module__}.{type(var_3.body).__qualname__}' == 'builtins.list'
    assert len(var_3.body) == 1
    assert var_3.type_ignores == []
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)

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
    var_2 = '(x, y) = (1, 2)'
    var_3 = module_3.parse(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Module'
    assert f'{type(var_3.body).__module__}.{type(var_3.body).__qualname__}' == 'builtins.list'
    assert len(var_3.body) == 1
    assert var_3.type_ignores == []
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)

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
    var_1 = 'test_module'
    var_2 = 'CONST = 5'
    var_3 = module_3.parse(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Module'
    assert f'{type(var_3.body).__module__}.{type(var_3.body).__qualname__}' == 'builtins.list'
    assert len(var_3.body) == 1
    assert var_3.type_ignores == []
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    assert var_0.root == {'test_module.CONST': 'test_module'}
    assert var_0.alias == {'test_module.CONST': '5'}
    assert var_0.const == {'test_module.CONST': 'int'}
    var_7 = 'CONST = 10'
    var_8 = module_3.parse(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Module'
    assert f'{type(var_8.body).__module__}.{type(var_8.body).__qualname__}' == 'builtins.list'
    assert len(var_8.body) == 1
    assert var_8.type_ignores == []
    var_9 = var_8.body[var_4]
    var_10 = var_0.globals(var_1, var_9)
    assert var_0.alias == {'test_module.CONST': '10'}

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
    var_1 = 'common'
    var_2 = module_3.Load()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Load'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    var_3 = module_0.parent(var_1)
    assert var_3 == 'common'
    var_4 = 'y[ee{HB?M7_ ehg'
    var_5 = module_3.parse(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Module'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'builtins.list'
    assert len(var_5.body) == 1
    assert var_5.type_ignores == []
    var_6 = var_0.resolve(var_4, var_5, var_1)
    assert var_6 == 'Self'

@pytest.mark.xfail(strict=True)
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
    var_2 = '+.m\rScRq$<['
    var_3 = [var_2, var_2, var_2]
    var_4 = None
    var_5 = {var_1: var_4}
    var_6 = module_3.ImportFrom(*var_3, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.ImportFrom'
    assert var_6.module == '+.m\rScRq$<['
    assert var_6.names == '+.m\rScRq$<['
    assert var_6.level == '+.m\rScRq$<['
    assert var_6.
 is None
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    assert module_3.ImportFrom.module is None
    assert module_3.ImportFrom.level is None
    var_0.imports(var_4, var_6)

@pytest.mark.xfail(strict=True)
def test_case_62():
    var_0 = 'q"ot;yMkrS'
    var_1 = module_0.code(var_0)
    assert var_1 == '`q"ot;yMkrS`'
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
    var_2 = -1726
    var_3 = None
    var_4 = 'p3s\r8J7\\DG:sBb\n\t\t'
    var_5 = 'dW8lOoFP~7aa5y.w)p'
    var_6 = 'Y7!c'
    var_7 = '/nS*#\t\x0bl-\r'
    var_8 = {var_4: var_5, var_4: var_5, var_4: var_6, var_7: var_5}
    var_9 = module_0.Resolver(var_0, var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Resolver'
    assert var_9.root == 'q"ot;yMkrS'
    assert var_9.alias is None
    assert var_9.self_ty == ''
    var_10 = module_0.Parser(b_level=var_2, toc=var_3, doc=var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == -1726
    assert var_10.toc is None
    assert var_10.level == {}
    assert var_10.doc == {'p3s\r8J7\\DG:sBb\n\t\t': 'Y7!c', '/nS*#\t\x0bl-\r': 'dW8lOoFP~7aa5y.w)p'}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
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
    var_12 = "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n"
    var_13 = var_11.parse(var_12, var_12)
    assert var_11.level == {"\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": 0, "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.publicfunc": 0, "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n._pratefunc": 0, "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.PublcClass": 0}
    assert var_11.doc == {"\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.publicfunc": '### publicfunc()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n._pratefunc": '### _pratefunc()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.PublcClass": '### class PublcClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_11.imp == {"\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": {*()}}
    assert var_11.root == {"\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n": "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n", "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.publicfunc": "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n", "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n._pratefunc": "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n", "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.PublcClass": "\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n"}
    assert var_11.alias == {"\n__all__ =B['public_func', 'PuAlicClass']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.__all__": "B['public_func', 'PuAlicClass']"}
    var_14 = var_11.compile()
    assert var_14 == '## Module `\n__all__ =B[\'public_func\', \'PuAlicClass\']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n`\n<a id="\n__all__ =b[\'public_func\', \'pualicclass\']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass publcclass:\n    pass\n"></a>\n\n### class PublcClass\n\n*Full name:* `\n__all__ =B[\'public_func\', \'PuAlicClass\']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.PublcClass`\n<a id="\n__all__ =b[\'public_func\', \'pualicclass\']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass publcclass:\n    pass\n-publcclass"></a>\n\n### publicfunc()\n\n*Full name:* `\n__all__ =B[\'public_func\', \'PuAlicClass\']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass PublcClass:\n    pass\n.publicfunc`\n<a id="\n__all__ =b[\'public_func\', \'pualicclass\']\ndef publicfunc():\n    pass\n\ndef _pratefunc():\n    pass\n\nclass publcclass:\n    pass\n-publicfunc"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_15 = 'Y="i=6B-L}!at['
    var_16 = module_2.dataclass(init=var_3, repr=var_13, slots=var_13)
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_17 = var_11.globals(var_15, var_16)
    var_18 = 'YV4*/`!m?P7gk d'
    var_16.api(var_7, var_16, prefix=var_18)