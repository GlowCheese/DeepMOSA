# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.parser as module_0
import dataclasses as module_1
import ast as module_2
import inspect as module_3

def test_case_0():
    var_0 = 'c\tksat'
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

def test_case_2():
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
def test_case_3():
    var_0 = None
    module_0.table(items=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = {var_0, var_0}
    var_2 = None
    var_3 = module_0.Parser(toc=var_2, alias=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is None
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {None}
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
    var_4 = var_3.load_docstring(var_0, var_1)
    var_5 = var_3.__repr__()
    assert var_5 == 'Parser(link=True, b_level=1, toc=None, level={}, doc={}, docstring={}, imp={}, root={}, alias={None}, const={})'
    var_6 = module_0.const_type(var_5)
    assert var_6 == 'Any'
    var_7 = 959
    var_8 = module_0.Parser(var_5, var_7, var_5, var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is True
    assert var_8.b_level == 959
    assert var_8.toc == 'Parser(link=True, b_level=1, toc=None, level={}, doc={}, docstring={}, imp={}, root={}, alias={None}, const={})'
    assert var_8.level == 'Parser(link=True, b_level=1, toc=None, level={}, doc={}, docstring={}, imp={}, root={}, alias={None}, const={})'
    assert var_8.doc == {}
    assert var_8.docstring == {}
    assert var_8.imp == {}
    assert var_8.root == {}
    assert var_8.alias == {}
    assert var_8.const == {}
    var_9 = '&Bz'
    var_10 = var_3.globals(var_9, var_4)
    var_11 = module_0.is_public_family(var_6)
    assert var_11 is True
    var_3.is_public(var_2)

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
    var_1 = 'class MyClass:\n    def method(self): pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class MyClass:\n    def method(self): pass': 0, 'class MyClass:\n    def method(self): pass.MyClass': 0, 'class MyClass:\n    def method(self): pass.MyClass.method': 0}
    assert var_0.doc == {'class MyClass:\n    def method(self): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class MyClass:\n    def method(self): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class MyClass:\n    def method(self): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'class MyClass:\n    def method(self): pass': {*()}}
    assert var_0.root == {'class MyClass:\n    def method(self): pass': 'class MyClass:\n    def method(self): pass', 'class MyClass:\n    def method(self): pass.MyClass': 'class MyClass:\n    def method(self): pass', 'class MyClass:\n    def method(self): pass.MyClass.method': 'class MyClass:\n    def method(self): pass'}
    var_3 = var_0.load_docstring(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.parent(var_0)

def test_case_8():
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

def test_case_9():
    var_0 = 'mGx\x0b\nO^&PiV*%C5VR'
    var_1 = module_0.code(var_0)
    assert var_1 == '<code>mGx\x0b\nO^&PiV*%C5VR</code>'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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

def test_case_11():
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

def test_case_12():
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

def test_case_13():
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
    var_1 = 'class MyClass:\n    def method(self): pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class MyClass:\n    def method(self): pass': 0, 'class MyClass:\n    def method(self): pass.MyClass': 0, 'class MyClass:\n    def method(self): pass.MyClass.method': 0}
    assert var_0.doc == {'class MyClass:\n    def method(self): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class MyClass:\n    def method(self): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class MyClass:\n    def method(self): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'class MyClass:\n    def method(self): pass': {*()}}
    assert var_0.root == {'class MyClass:\n    def method(self): pass': 'class MyClass:\n    def method(self): pass', 'class MyClass:\n    def method(self): pass.MyClass': 'class MyClass:\n    def method(self): pass', 'class MyClass:\n    def method(self): pass.MyClass.method': 'class MyClass:\n    def method(self): pass'}

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
    var_1 = 'class MyClass:\n    def method(self): pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class MyClass:\n    def method(self): pass': 0, 'class MyClass:\n    def method(self): pass.MyClass': 0, 'class MyClass:\n    def method(self): pass.MyClass.method': 0}
    assert var_0.doc == {'class MyClass:\n    def method(self): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class MyClass:\n    def method(self): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class MyClass:\n    def method(self): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'class MyClass:\n    def method(self): pass': {*()}}
    assert var_0.root == {'class MyClass:\n    def method(self): pass': 'class MyClass:\n    def method(self): pass', 'class MyClass:\n    def method(self): pass.MyClass': 'class MyClass:\n    def method(self): pass', 'class MyClass:\n    def method(self): pass.MyClass.method': 'class MyClass:\n    def method(self): pass'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `class MyClass:\n    def method(self): pass`\n<a id="class myclass:\n    def method(self): pass"></a>\n\n### class MyClass\n\n*Full name:* `class MyClass:\n    def method(self): pass.MyClass`\n<a id="class myclass:\n    def method(self): pass-myclass"></a>\n\n#### MyClass.method()\n\n*Full name:* `class MyClass:\n    def method(self): pass.MyClass.method`\n<a id="class myclass:\n    def method(self): pass-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = '>1B$fUP2L6'
    var_1 = {var_0, var_0}
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_4 = module_0.Parser(imp=var_2, alias=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level == 1
    assert var_4.toc is False
    assert var_4.level == {}
    assert var_4.doc == {}
    assert var_4.docstring == {}
    assert var_4.imp == {'>1B$fUP2L6': {'>1B$fUP2L6'}}
    assert var_4.root == {}
    assert var_4.alias == {'>1B$fUP2L6': '>1B$fUP2L6'}
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
    var_5 = var_4.__repr__()
    var_6 = var_4.func_ann(var_5, var_3, has_self=var_2, cls_method=var_3)
    module_0.table(*var_5, items=var_6)

def test_case_16():
    var_0 = '.'
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

def test_case_17():
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
    var_1 = "CONSTANT = 42\n__all__ = ['foo']"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"CONSTANT = 42\n__all__ = ['foo']": 0}
    assert var_0.doc == {"CONSTANT = 42\n__all__ = ['foo']": '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"CONSTANT = 42\n__all__ = ['foo']": {"CONSTANT = 42\n__all__ = ['foo'].foo"}}
    assert var_0.root == {"CONSTANT = 42\n__all__ = ['foo']": "CONSTANT = 42\n__all__ = ['foo']", "CONSTANT = 42\n__all__ = ['foo'].CONSTANT": "CONSTANT = 42\n__all__ = ['foo']"}
    assert var_0.alias == {"CONSTANT = 42\n__all__ = ['foo'].CONSTANT": '42', "CONSTANT = 42\n__all__ = ['foo'].__all__": "['foo']"}
    assert var_0.const == {"CONSTANT = 42\n__all__ = ['foo'].CONSTANT": 'int'}

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
    var_1 = var_0.__repr__()
    assert var_1 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_2 = 'from typing import it\nVector = List[float]'
    var_3 = var_0.parse(var_2, var_2)
    assert var_0.level == {'from typing import it\nVector = List[float]': 0}
    assert var_0.doc == {'from typing import it\nVector = List[float]': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'from typing import it\nVector = List[float]': {*()}}
    assert var_0.root == {'from typing import it\nVector = List[float]': 'from typing import it\nVector = List[float]'}
    assert var_0.alias == {'from typing import it\nVector = List[float].it': 'typing.it', 'from typing import it\nVector = List[float].Vector': 'List[float]'}

def test_case_20():
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
    var_3 = module_1.field(compare=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'dataclasses.Field'
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
    var_4 = var_3.__repr__()
    assert var_4 == 'Field(name=None,type=None,default=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,default_factory=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,init=True,repr=True,hash=None,compare=None,metadata=mappingproxy({}),kw_only=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,_field_type=None)'
    var_5 = 'L;Ih79D:9f&>'
    var_6 = module_0.Resolver(var_0, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Resolver'
    assert var_6.root is None
    assert var_6.alias == 'Field(name=None,type=None,default=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,default_factory=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,init=True,repr=True,hash=None,compare=None,metadata=mappingproxy({}),kw_only=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,_field_type=None)'
    assert var_6.self_ty == ''
    var_7 = '6:pu.b=]p"rhP?Fs5'
    var_8 = module_0.is_public_family(var_5)
    assert var_8 is True
    var_9 = module_0.esc_underscore(var_4)
    assert var_9 == 'Field(name=None,type=None,default=<dataclasses.\\_MISSING\\_TYPE object at 0x7c95dc9fb010>,default\\_factory=<dataclasses.\\_MISSING\\_TYPE object at 0x7c95dc9fb010>,init=True,repr=True,hash=None,compare=None,metadata=mappingproxy({}),kw\\_only=<dataclasses.\\_MISSING\\_TYPE object at 0x7c95dc9fb010>,\\_field\\_type=None)'
    var_10 = module_0.doctest(var_7)
    assert var_10 == '6:pu.b=]p"rhP?Fs5'

@pytest.mark.xfail(strict=True)
def test_case_21():
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
def test_case_22():
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

def test_case_23():
    var_0 = 'UPo'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_3 = module_0.Parser(imp=var_1, alias=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {'UPo': 'UPo'}
    assert var_3.root == {}
    assert var_3.alias == {'UPo': 'UPo'}
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
    var_4 = var_3.globals(var_0, var_0)
    var_5 = var_3.parse(var_0, var_0)
    assert var_3.level == {'UPo': 0}
    assert var_3.doc == {'UPo': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_3.imp == {'UPo': {*()}}
    assert var_3.root == {'UPo': 'UPo'}

@pytest.mark.xfail(strict=True)
def test_case_24():
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
    var_3 = module_1.field(compare=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'dataclasses.Field'
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
    var_4 = var_3.__repr__()
    assert var_4 == 'Field(name=None,type=None,default=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,default_factory=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,init=True,repr=True,hash=None,compare=None,metadata=mappingproxy({}),kw_only=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,_field_type=None)'
    var_5 = module_0.Resolver(var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Resolver'
    assert var_5.root is None
    assert var_5.alias == 'Field(name=None,type=None,default=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,default_factory=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,init=True,repr=True,hash=None,compare=None,metadata=mappingproxy({}),kw_only=<dataclasses._MISSING_TYPE object at 0x7c95dc9fb010>,_field_type=None)'
    assert var_5.self_ty == ''
    var_6 = module_0.is_public_family(var_4)
    assert var_6 is False
    module_0.esc_underscore(var_0)

def test_case_25():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
    var_2 = "gTXPdyC='8{];"
    var_3 = {var_2: var_1, var_2: var_1}
    var_4 = module_2.Call(*var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Call'
    assert var_4.func is None
    assert var_4.args is None
    assert var_4.keywords is None
    assert var_4.gTXPdyC='8{]; == [None, None, None]
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_5 = module_0.const_type(var_4)
    assert var_5 == 'Any'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_26():
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
    var_1 = '"""Module docstring."""\ndef func(): pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'"""Module docstring."""\ndef func(): pass': 1, '"""Module docstring."""\ndef func(): pass.func': 1}
    assert var_0.doc == {'"""Module docstring."""\ndef func(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', '"""Module docstring."""\ndef func(): pass.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'"""Module docstring."""\ndef func(): pass': 'Module docstring.'}
    assert var_0.imp == {'"""Module docstring."""\ndef func(): pass': {*()}}
    assert var_0.root == {'"""Module docstring."""\ndef func(): pass': '"""Module docstring."""\ndef func(): pass', '"""Module docstring."""\ndef func(): pass.func': '"""Module docstring."""\ndef func(): pass'}
    var_3 = 'test_module8'
    var_4 = '@staticmethod\ndef static_method(): pass'
    var_5 = var_0.parse(var_3, var_4)
    assert var_0.level == {'"""Module docstring."""\ndef func(): pass': 1, '"""Module docstring."""\ndef func(): pass.func': 1, 'test_module8': 0, 'test_module8.static_method': 0}
    assert var_0.doc == {'"""Module docstring."""\ndef func(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', '"""Module docstring."""\ndef func(): pass.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module8': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module8.static_method': '### static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'"""Module docstring."""\ndef func(): pass': {*()}, 'test_module8': {*()}}
    assert var_0.root == {'"""Module docstring."""\ndef func(): pass': '"""Module docstring."""\ndef func(): pass', '"""Module docstring."""\ndef func(): pass.func': '"""Module docstring."""\ndef func(): pass', 'test_module8': 'test_module8', 'test_module8.static_method': 'test_module8'}

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
    var_1 = 'class MyClass:\n    def method(self): pass'
    var_2 = var_0.__repr__()
    assert var_2 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_3 = var_0.__post_init__()
    var_4 = var_0.parse(var_1, var_2)
    assert var_0.level == {'class MyClass:\n    def method(self): pass': 0}
    assert var_0.doc == {'class MyClass:\n    def method(self): pass': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'class MyClass:\n    def method(self): pass': {*()}}
    assert var_0.root == {'class MyClass:\n    def method(self): pass': 'class MyClass:\n    def method(self): pass'}
    var_5 = 'YF3z:37'
    var_6 = var_0.parse(var_5, var_5)
    assert var_0.level == {'class MyClass:\n    def method(self): pass': 0, 'YF3z:37': 0}
    assert var_0.doc == {'class MyClass:\n    def method(self): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'YF3z:37': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'class MyClass:\n    def method(self): pass': {*()}, 'YF3z:37': {*()}}
    assert var_0.root == {'class MyClass:\n    def method(self): pass': 'class MyClass:\n    def method(self): pass', 'YF3z:37': 'YF3z:37'}

@pytest.mark.xfail(strict=True)
def test_case_29():
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
    var_6 = module_1.field(default_factory=var_2, metadata=var_0)
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
def test_case_30():
    var_0 = '>1fUP2L6'
    var_1 = {var_0, var_0}
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = 'ZD[i7R]L!d:01^q?&TY'
    var_4 = {var_3: var_3, var_3: var_3, var_3: var_0}
    var_5 = module_0.Parser(imp=var_2, alias=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is True
    assert var_5.b_level == 1
    assert var_5.toc is False
    assert var_5.level == {}
    assert var_5.doc == {}
    assert var_5.docstring == {}
    assert var_5.imp == {'>1fUP2L6': {'>1fUP2L6'}}
    assert var_5.root == {}
    assert var_5.alias == {'ZD[i7R]L!d:01^q?&TY': '>1fUP2L6'}
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
    var_6 = var_5.compile()
    assert var_6 == '\n'
    var_7 = [var_6]
    var_8 = module_2.Set(*var_7, **var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Set'
    assert var_8.elts == '\n'
    assert var_8.>1fUP2L6 == {'>1fUP2L6'}
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_9 = module_0.const_type(var_8)
    assert var_9 == 'set'
    var_10 = 'eZ8<d'
    var_11 = module_0.is_public_family(var_10)
    assert var_11 is True
    var_5.class_api(var_0, var_3, var_7, var_7)

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
    var_1 = 'import os\ndef bar(): pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.bar': 0}
    assert var_0.doc == {'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'import os\ndef bar(): pass': {*()}}
    assert var_0.root == {'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.bar': 'import os\ndef bar(): pass'}
    assert var_0.alias == {'import os\ndef bar(): pass.os': 'os'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `import os\ndef bar(): pass`\n<a id="import os\ndef bar(): pass"></a>\n\n### bar()\n\n*Full name:* `import os\ndef bar(): pass.bar`\n<a id="import os\ndef bar(): pass-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n'

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = 'UPo'
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_3 = module_0.Parser(imp=var_1, alias=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {'UPo': 'UPo'}
    assert var_3.root == {}
    assert var_3.alias == {'UPo': 'UPo'}
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
    var_5 = module_0.const_type(var_4)
    assert var_5 == 'Any'
    var_6 = True
    var_7 = module_0.Parser(var_6, toc=var_4, doc=var_2, alias=var_4, const=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is True
    assert var_7.b_level == 1
    assert var_7.toc is None
    assert var_7.level == {}
    assert var_7.doc == {'UPo': 'UPo'}
    assert var_7.docstring == {}
    assert var_7.imp == {}
    assert var_7.root == {}
    assert var_7.alias is None
    assert var_7.const == 'UPo'
    var_8 = var_7.__repr__()
    assert var_8 == "Parser(link=True, b_level=1, toc=None, level={}, doc={'UPo': 'UPo'}, docstring={}, imp={}, root={}, alias=None, const='UPo')"
    var_9 = var_7.__post_init__()
    var_10 = var_7.load_docstring(var_5, var_2)
    var_11 = var_3.parse(var_0, var_0)
    assert var_3.level == {'UPo': 0}
    assert var_3.doc == {'UPo': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_3.imp == {'UPo': {*()}}
    assert var_3.root == {'UPo': 'UPo'}
    var_12 = var_3.compile()
    assert var_12 == '\n'
    assert var_3.docstring == {'UPo': ''}
    var_13 = var_8.__repr__()
    assert var_13 == '"Parser(link=True, b_level=1, toc=None, level={}, doc={\'UPo\': \'UPo\'}, docstring={}, imp={}, root={}, alias=None, const=\'UPo\')"'
    var_3.class_api(var_11, var_10, var_8, var_8)

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
    var_1 = 'import os\ndef bar(): pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.bar': 0}
    assert var_0.doc == {'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'import os\ndef bar(): pass': {*()}}
    assert var_0.root == {'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.bar': 'import os\ndef bar(): pass'}
    assert var_0.alias == {'import os\ndef bar(): pass.os': 'os'}

def test_case_34():
    var_0 = 'UPo'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_3 = module_0.Parser(imp=var_1, alias=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {'UPo': 'UPo'}
    assert var_3.root == {}
    assert var_3.alias == {'UPo': 'UPo'}
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
    var_4 = var_3.parse(var_0, var_0)
    assert var_3.level == {'UPo': 0}
    assert var_3.doc == {'UPo': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_3.imp == {'UPo': {*()}}
    assert var_3.root == {'UPo': 'UPo'}
    var_5 = var_3.compile()
    assert var_5 == '\n'
    assert var_3.docstring == {'UPo': ''}

def test_case_35():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_2.Tuple(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Tuple'
    assert var_2.elts is None
    assert var_2.ctx is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert f'{type(module_2.Tuple.dims).__module__}.{type(module_2.Tuple.dims).__qualname__}' == 'builtins.property'
    var_3 = module_0.const_type(var_2)
    assert var_3 == 'tuple'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
def test_case_36():
    var_0 = '1?UCP2o'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_3 = module_0.Parser(imp=var_1, alias=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {'1?UCP2o': '1?UCP2o'}
    assert var_3.root == {}
    assert var_3.alias == {'1?UCP2o': '1?UCP2o'}
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
    var_5 = []
    var_6 = module_2.AnnAssign()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.AnnAssign'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.AnnAssign.value is None
    var_7 = [var_0, var_6, var_0]
    var_3.class_api(var_4, var_1, var_5, var_7)

def test_case_37():
    var_0 = None
    var_1 = '-O'
    var_2 = {var_1, var_1}
    var_3 = {var_1: var_2, var_1: var_2, var_1: var_2}
    var_4 = '+'
    var_5 = {var_4: var_1, var_1: var_4, var_4: var_4}
    var_6 = module_0.Parser(imp=var_3, alias=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp == {'-O': {'-O'}}
    assert var_6.root == {}
    assert var_6.alias == {'+': '+', '-O': '+'}
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
    var_7 = [var_0]
    var_8 = {}
    var_9 = module_2.AnnAssign(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.AnnAssign'
    assert var_9.target is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.AnnAssign.value is None
    var_10 = var_6.globals(var_0, var_9)
    var_11 = module_3.getdoc(var_10)
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

def test_case_38():
    var_0 = '-P9'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = '|P\x0b_PRW%I'
    var_3 = {var_2: var_2, var_2: var_2, var_2: var_0}
    var_4 = module_0.Parser(imp=var_1, alias=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level == 1
    assert var_4.toc is False
    assert var_4.level == {}
    assert var_4.doc == {}
    assert var_4.docstring == {}
    assert var_4.imp == {'-P9': '-P9'}
    assert var_4.root == {}
    assert var_4.alias == {'|P\x0b_PRW%I': '-P9'}
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
    var_5 = module_0.Parser(imp=var_1, root=var_0, alias=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is True
    assert var_5.b_level == 1
    assert var_5.toc is False
    assert var_5.level == {}
    assert var_5.doc == {}
    assert var_5.docstring == {}
    assert var_5.imp == {'-P9': '-P9'}
    assert var_5.root == '-P9'
    assert var_5.alias == '-P9'
    assert var_5.const == {}
    var_6 = None
    var_7 = var_5.globals(var_6, var_0)
    var_8 = var_4.parse(var_0, var_0)
    assert var_4.level == {'-P9': 0}
    assert var_4.doc == {'-P9': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_4.imp == {'-P9': {*()}}
    assert var_4.root == {'-P9': '-P9'}
    assert var_5.imp == {'-P9': {*()}}
    var_9 = 'q=P@'
    var_10 = var_4.compile()
    assert var_10 == '\n'
    var_11 = module_2.ImportFrom()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.ImportFrom'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_12 = var_5.imports(var_9, var_11)

def test_case_39():
    var_0 = '-P9'
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.Parser(imp=var_1, root=var_0, alias=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc == {}
    assert var_2.docstring == {}
    assert var_2.imp == {'-P9': '-P9'}
    assert var_2.root == '-P9'
    assert var_2.alias == '-P9'
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
    var_4 = var_2.globals(var_3, var_0)
    var_5 = module_2.ImportFrom()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.ImportFrom'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_6 = var_2.imports(var_1, var_5)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = '[\x0bxbt]\t%'
    var_1 = module_0.esc_underscore(var_0)
    assert var_1 == '[\x0bxbt]\t%'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_2 = 'p4'
    var_3 = {var_2: var_2, var_2: var_2, var_2: var_2}
    var_4 = {var_2: var_2, var_2: var_2, var_2: var_2}
    var_5 = module_0.Parser(imp=var_3, alias=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is True
    assert var_5.b_level == 1
    assert var_5.toc is False
    assert var_5.level == {}
    assert var_5.doc == {}
    assert var_5.docstring == {}
    assert var_5.imp == {'p4': 'p4'}
    assert var_5.root == {}
    assert var_5.alias == {'p4': 'p4'}
    assert var_5.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_6 = var_5.__repr__()
    assert var_6 == "Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})"
    var_7 = var_5.parse(var_6, var_6)
    assert var_5.level == {"Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})": 0}
    assert var_5.doc == {"Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})": '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_5.imp == {'p4': 'p4', "Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})": {*()}}
    assert var_5.root == {"Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})": "Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})"}
    var_8 = var_5.globals(var_6, var_2)
    var_9 = var_5.parse(var_2, var_2)
    assert var_5.level == {"Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})": 0, 'p4': 0}
    assert var_5.doc == {"Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})": '## Module `{}`\n<a id="{}"></a>\n\n', 'p4': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_5.imp == {'p4': {*()}, "Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})": {*()}}
    assert var_5.root == {"Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})": "Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={'p4': 'p4'}, root={}, alias={'p4': 'p4'}, const={})", 'p4': 'p4'}
    var_10 = var_5.compile()
    assert var_10 == '\n'
    assert var_5.docstring == {'p4': ''}
    var_11 = True
    var_12 = var_5.func_ann(var_6, var_6, has_self=var_11, cls_method=var_8)
    module_0.table(items=var_12)

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
    var_1 = 'test_module2'
    var_2 = 'import os\ndef bar(): pass'
    var_3 = module_0.code(var_1)
    assert var_3 == '`test_module2`'
    var_4 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2'}
    assert var_0.alias == {'test_module2.os': 'os'}
    var_5 = 'class MyClass:\n    def method(self): pass'
    var_6 = var_0.parse(var_2, var_5)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'import os\ndef bar(): pass': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass'}
    var_7 = var_0.__repr__()
    assert var_7 == 'Parser(link=True, b_level=1, toc=False, level={\'test_module2\': 0, \'test_module2.bar\': 0, \'import os\\ndef bar(): pass\': 0, \'import os\\ndef bar(): pass.MyClass\': 0, \'import os\\ndef bar(): pass.MyClass.method\': 0}, doc={\'test_module2\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'test_module2.bar\': \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', \'import os\\ndef bar(): pass\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'import os\\ndef bar(): pass.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'import os\\ndef bar(): pass.MyClass.method\': \'#### MyClass.method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `Any` |\\n\\n\'}, docstring={}, imp={\'test_module2\': set(), \'import os\\ndef bar(): pass\': set()}, root={\'test_module2\': \'test_module2\', \'test_module2.bar\': \'test_module2\', \'import os\\ndef bar(): pass\': \'import os\\ndef bar(): pass\', \'import os\\ndef bar(): pass.MyClass\': \'import os\\ndef bar(): pass\', \'import os\\ndef bar(): pass.MyClass.method\': \'import os\\ndef bar(): pass\'}, alias={\'test_module2.os\': \'os\'}, const={})'
    var_8 = var_0.parse(var_2, var_7)
    var_9 = "CONSTANT = 42\n__all__ = ['foo']"
    var_10 = var_0.parse(var_3, var_9)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0, '`test_module2`': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '`test_module2`': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'import os\ndef bar(): pass': {*()}, '`test_module2`': {'`test_module2`.foo'}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass', '`test_module2`': '`test_module2`', '`test_module2`.CONSTANT': '`test_module2`'}
    assert var_0.alias == {'test_module2.os': 'os', '`test_module2`.CONSTANT': '42', '`test_module2`.__all__': "['foo']"}
    assert var_0.const == {'`test_module2`.CONSTANT': 'int'}
    var_11 = module_1.field(repr=var_6, hash=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'dataclasses.Field'
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
    var_12 = var_0.compile()
    assert var_12 == '## Module ``test_module2``\n<a id="`test_module2`"></a>\n\n## Module `import os\ndef bar(): pass`\n<a id="import os\ndef bar(): pass"></a>\n\n### class MyClass\n\n*Full name:* `import os\ndef bar(): pass.MyClass`\n<a id="import os\ndef bar(): pass-myclass"></a>\n\n#### MyClass.method()\n\n*Full name:* `import os\ndef bar(): pass.MyClass.method`\n<a id="import os\ndef bar(): pass-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n## Module `test_module2`\n<a id="test_module2"></a>\n\n### bar()\n\n*Full name:* `test_module2.bar`\n<a id="test_module2-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_13 = var_0.__repr__()
    assert var_13 == 'Parser(link=True, b_level=1, toc=False, level={\'test_module2\': 0, \'test_module2.bar\': 0, \'import os\\ndef bar(): pass\': 0, \'import os\\ndef bar(): pass.MyClass\': 0, \'import os\\ndef bar(): pass.MyClass.method\': 0, \'`test_module2`\': 0}, doc={\'test_module2\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'test_module2.bar\': \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', \'import os\\ndef bar(): pass\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'import os\\ndef bar(): pass.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'import os\\ndef bar(): pass.MyClass.method\': \'#### MyClass.method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `Any` |\\n\\n\', \'`test_module2`\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={}, imp={\'test_module2\': set(), \'import os\\ndef bar(): pass\': set(), \'`test_module2`\': {\'`test_module2`.foo\'}}, root={\'test_module2\': \'test_module2\', \'test_module2.bar\': \'test_module2\', \'import os\\ndef bar(): pass\': \'import os\\ndef bar(): pass\', \'import os\\ndef bar(): pass.MyClass\': \'import os\\ndef bar(): pass\', \'import os\\ndef bar(): pass.MyClass.method\': \'import os\\ndef bar(): pass\', \'`test_module2`\': \'`test_module2`\', \'`test_module2`.CONSTANT\': \'`test_module2`\'}, alias={\'test_module2.os\': \'os\', \'`test_module2`.CONSTANT\': \'42\', \'`test_module2`.__all__\': "[\'foo\']"}, const={\'`test_module2`.CONSTANT\': \'int\'})'
    var_14 = '@sacmeh\ndef static_method(): pass'
    var_15 = var_0.parse(var_1, var_14)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0, '`test_module2`': 0, 'test_module2.static_method': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '`test_module2`': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.static_method': '### static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@sacmeh` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass', '`test_module2`': '`test_module2`', '`test_module2`.CONSTANT': '`test_module2`', 'test_module2.static_method': 'test_module2'}
    var_16 = var_0.__post_init__()
    var_0.parse(var_7, var_8)

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
    var_1 = 'class MyClas:\n    def -ethod(self):pass'
    var_2 = 'async def async_func(): pass'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'class MyClas:\n    def -ethod(self):pass': 0, 'class MyClas:\n    def -ethod(self):pass.async_func': 0}
    assert var_0.doc == {'class MyClas:\n    def -ethod(self):pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class MyClas:\n    def -ethod(self):pass.async_func': '### async async_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'class MyClas:\n    def -ethod(self):pass': {*()}}
    assert var_0.root == {'class MyClas:\n    def -ethod(self):pass': 'class MyClas:\n    def -ethod(self):pass', 'class MyClas:\n    def -ethod(self):pass.async_func': 'class MyClas:\n    def -ethod(self):pass'}
    var_4 = var_0.compile()
    assert var_4 == '## Module `class MyClas:\n    def -ethod(self):pass`\n<a id="class myclas:\n    def -ethod(self):pass"></a>\n\n### async async_func()\n\n*Full name:* `class MyClas:\n    def -ethod(self):pass.async_func`\n<a id="class myclas:\n    def -ethod(self):pass-async_func"></a>\n\n| return |\n|:------:|\n| `Any` |\n'

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
    var_1 = 'test_module2'
    var_2 = 'F2'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module2': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module2': {*()}}
    assert var_0.root == {'test_module2': 'test_module2'}
    var_4 = module_3.getdoc(var_2)
    assert var_4 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
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
    var_5 = var_0.__repr__()
    assert var_5 == 'Parser(link=True, b_level=1, toc=False, level={\'test_module2\': 0}, doc={\'test_module2\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={}, imp={\'test_module2\': set()}, root={\'test_module2\': \'test_module2\'}, alias={}, const={})'
    var_6 = '"""Module docstring."""\ndef func(): pass'
    var_7 = var_0.parse(var_2, var_6)
    assert var_0.level == {'test_module2': 0, 'F2': 0, 'F2.func': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'F2': '## Module `{}`\n<a id="{}"></a>\n\n', 'F2.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'F2': 'Module docstring.'}
    assert var_0.imp == {'test_module2': {*()}, 'F2': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'F2': 'F2', 'F2.func': 'F2'}
    var_8 = var_0.func_ann(var_5, var_5, has_self=var_3, cls_method=var_7)
    var_9 = 't5s6_module7'
    var_10 = module_1.field(repr=var_4, hash=var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'dataclasses.Field'
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
    var_11 = var_0.compile()
    assert var_11 == '## Module `F2`\n<a id="f2"></a>\n\nModule docstring.\n\n### func()\n\n*Full name:* `F2.func`\n<a id="f2-func"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_12 = '@saicmeh\ndef static_method(): pass'
    var_13 = var_0.parse(var_9, var_12)
    assert var_0.level == {'test_module2': 0, 'F2': 0, 'F2.func': 0, 't5s6_module7': 0, 't5s6_module7.static_method': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'F2': '## Module `{}`\n<a id="{}"></a>\n\n', 'F2.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 't5s6_module7': '## Module `{}`\n<a id="{}"></a>\n\n', 't5s6_module7.static_method': '### static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@saicmeh` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'F2': {*()}, 't5s6_module7': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'F2': 'F2', 'F2.func': 'F2', 't5s6_module7': 't5s6_module7', 't5s6_module7.static_method': 't5s6_module7'}
    var_14 = var_0.load_docstring(var_1, var_6)
    var_3.func_ann(var_3, var_3, has_self=var_5, cls_method=var_13)

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
    var_1 = 'def foo(): pass'
    var_2 = var_0.__repr__()
    assert var_2 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_3 = 'test_module2'
    var_4 = 'import os\ndef bar(): pass'
    var_5 = var_0.parse(var_3, var_4)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2'}
    assert var_0.alias == {'test_module2.os': 'os'}
    var_6 = 'test_module3'
    var_7 = var_0.parse(var_6, var_6)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'test_module3': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module3': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'test_module3': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'test_module3': 'test_module3'}
    var_8 = '9Y-QoN1EMNmeJw2*k'
    var_9 = var_0.parse(var_8, var_1)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'test_module3': 0, '9Y-QoN1EMNmeJw2*k': 0, '9Y-QoN1EMNmeJw2*k.foo': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module3': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'test_module3': {*()}, '9Y-QoN1EMNmeJw2*k': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'test_module3': 'test_module3', '9Y-QoN1EMNmeJw2*k': '9Y-QoN1EMNmeJw2*k', '9Y-QoN1EMNmeJw2*k.foo': '9Y-QoN1EMNmeJw2*k'}
    var_10 = 'test_module5'
    var_11 = '"""Module docstring."""\ndef func(): pass'
    var_12 = var_0.parse(var_10, var_11)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'test_module3': 0, '9Y-QoN1EMNmeJw2*k': 0, '9Y-QoN1EMNmeJw2*k.foo': 0, 'test_module5': 0, 'test_module5.func': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module3': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module5': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module5.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'test_module5': 'Module docstring.'}
    assert var_0.imp == {'test_module2': {*()}, 'test_module3': {*()}, '9Y-QoN1EMNmeJw2*k': {*()}, 'test_module5': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'test_module3': 'test_module3', '9Y-QoN1EMNmeJw2*k': '9Y-QoN1EMNmeJw2*k', '9Y-QoN1EMNmeJw2*k.foo': '9Y-QoN1EMNmeJw2*k', 'test_module5': 'test_module5', 'test_module5.func': 'test_module5'}
    var_13 = 'm)\x0c)Q'
    var_14 = "CONSTANT = 42\n__all__ = ['foo']"
    var_15 = var_0.parse(var_13, var_14)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'test_module3': 0, '9Y-QoN1EMNmeJw2*k': 0, '9Y-QoN1EMNmeJw2*k.foo': 0, 'test_module5': 0, 'test_module5.func': 0, 'm)\x0c)Q': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module3': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module5': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module5.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'm)\x0c)Q': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'test_module3': {*()}, '9Y-QoN1EMNmeJw2*k': {*()}, 'test_module5': {*()}, 'm)\x0c)Q': {'m)\x0c)Q.foo'}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'test_module3': 'test_module3', '9Y-QoN1EMNmeJw2*k': '9Y-QoN1EMNmeJw2*k', '9Y-QoN1EMNmeJw2*k.foo': '9Y-QoN1EMNmeJw2*k', 'test_module5': 'test_module5', 'test_module5.func': 'test_module5', 'm)\x0c)Q': 'm)\x0c)Q', 'm)\x0c)Q.CONSTANT': 'm)\x0c)Q'}
    assert var_0.alias == {'test_module2.os': 'os', 'm)\x0c)Q.CONSTANT': '42', 'm)\x0c)Q.__all__': "['foo']"}
    assert var_0.const == {'m)\x0c)Q.CONSTANT': 'int'}
    var_16 = 'from typing import List\nVector = List[float]'
    var_17 = var_0.parse(var_13, var_16)
    assert var_0.imp == {'test_module2': {*()}, 'test_module3': {*()}, '9Y-QoN1EMNmeJw2*k': {*()}, 'test_module5': {*()}, 'm)\x0c)Q': {*()}}
    assert var_0.alias == {'test_module2.os': 'os', 'm)\x0c)Q.CONSTANT': '42', 'm)\x0c)Q.__all__': "['foo']", 'm)\x0c)Q.List': 'typing.List', 'm)\x0c)Q.Vector': 'List[float]'}
    var_18 = var_0.compile()
    assert var_18 == '## Module `9Y-QoN1EMNmeJw2*k`\n<a id="9y-qon1emnmejw2*k"></a>\n\n### foo()\n\n*Full name:* `9Y-QoN1EMNmeJw2*k.foo`\n<a id="9y-qon1emnmejw2*k-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n## Module `m)\x0c)Q`\n<a id="m)\x0c)q"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n\n## Module `test_module2`\n<a id="test_module2"></a>\n\n### bar()\n\n*Full name:* `test_module2.bar`\n<a id="test_module2-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n## Module `test_module5`\n<a id="test_module5"></a>\n\nModule docstring.\n\n### func()\n\n*Full name:* `test_module5.func`\n<a id="test_module5-func"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_19 = 'def nested(): pass'
    var_20 = var_0.parse(var_1, var_19)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'test_module3': 0, '9Y-QoN1EMNmeJw2*k': 0, '9Y-QoN1EMNmeJw2*k.foo': 0, 'test_module5': 0, 'test_module5.func': 0, 'm)\x0c)Q': 0, 'def foo(): pass': 0, 'def foo(): pass.nested': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module3': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module5': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module5.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'm)\x0c)Q': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass.nested': '### nested()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'test_module3': {*()}, '9Y-QoN1EMNmeJw2*k': {*()}, 'test_module5': {*()}, 'm)\x0c)Q': {*()}, 'def foo(): pass': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'test_module3': 'test_module3', '9Y-QoN1EMNmeJw2*k': '9Y-QoN1EMNmeJw2*k', '9Y-QoN1EMNmeJw2*k.foo': '9Y-QoN1EMNmeJw2*k', 'test_module5': 'test_module5', 'test_module5.func': 'test_module5', 'm)\x0c)Q': 'm)\x0c)Q', 'm)\x0c)Q.CONSTANT': 'm)\x0c)Q', 'def foo(): pass': 'def foo(): pass', 'def foo(): pass.nested': 'def foo(): pass'}
    var_21 = 'test_module8'
    var_22 = '@staticmethod\ndef static_method(): pass'
    var_23 = var_0.parse(var_21, var_22)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'test_module3': 0, '9Y-QoN1EMNmeJw2*k': 0, '9Y-QoN1EMNmeJw2*k.foo': 0, 'test_module5': 0, 'test_module5.func': 0, 'm)\x0c)Q': 0, 'def foo(): pass': 0, 'def foo(): pass.nested': 0, 'test_module8': 0, 'test_module8.static_method': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module3': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module5': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module5.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'm)\x0c)Q': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass.nested': '### nested()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module8': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module8.static_method': '### static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'test_module3': {*()}, '9Y-QoN1EMNmeJw2*k': {*()}, 'test_module5': {*()}, 'm)\x0c)Q': {*()}, 'def foo(): pass': {*()}, 'test_module8': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'test_module3': 'test_module3', '9Y-QoN1EMNmeJw2*k': '9Y-QoN1EMNmeJw2*k', '9Y-QoN1EMNmeJw2*k.foo': '9Y-QoN1EMNmeJw2*k', 'test_module5': 'test_module5', 'test_module5.func': 'test_module5', 'm)\x0c)Q': 'm)\x0c)Q', 'm)\x0c)Q.CONSTANT': 'm)\x0c)Q', 'def foo(): pass': 'def foo(): pass', 'def foo(): pass.nested': 'def foo(): pass', 'test_module8': 'test_module8', 'test_module8.static_method': 'test_module8'}
    var_24 = 'empty_module'
    var_25 = ''
    var_26 = var_0.parse(var_24, var_25)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'test_module3': 0, '9Y-QoN1EMNmeJw2*k': 0, '9Y-QoN1EMNmeJw2*k.foo': 0, 'test_module5': 0, 'test_module5.func': 0, 'm)\x0c)Q': 0, 'def foo(): pass': 0, 'def foo(): pass.nested': 0, 'test_module8': 0, 'test_module8.static_method': 0, 'empty_module': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module3': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '9Y-QoN1EMNmeJw2*k.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module5': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module5.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'm)\x0c)Q': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass.nested': '### nested()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module8': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module8.static_method': '### static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n', 'empty_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'test_module3': {*()}, '9Y-QoN1EMNmeJw2*k': {*()}, 'test_module5': {*()}, 'm)\x0c)Q': {*()}, 'def foo(): pass': {*()}, 'test_module8': {*()}, 'empty_module': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'test_module3': 'test_module3', '9Y-QoN1EMNmeJw2*k': '9Y-QoN1EMNmeJw2*k', '9Y-QoN1EMNmeJw2*k.foo': '9Y-QoN1EMNmeJw2*k', 'test_module5': 'test_module5', 'test_module5.func': 'test_module5', 'm)\x0c)Q': 'm)\x0c)Q', 'm)\x0c)Q.CONSTANT': 'm)\x0c)Q', 'def foo(): pass': 'def foo(): pass', 'def foo(): pass.nested': 'def foo(): pass', 'test_module8': 'test_module8', 'test_module8.static_method': 'test_module8', 'empty_module': 'empty_module'}

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = 'test_module2'
    var_3 = 'import os\ndef bar(): pass'
    var_4 = var_0.parse(var_2, var_3)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2'}
    assert var_0.alias == {'test_module2.os': 'os'}
    var_5 = 'class MyClass:\n    def method(self): pass'
    var_6 = var_0.parse(var_3, var_5)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'import os\ndef bar(): pass': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass'}
    var_7 = '!Y-QN1EMNmeJw2*k'
    var_8 = var_0.parse(var_7, var_5)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0, '!Y-QN1EMNmeJw2*k': 0, '!Y-QN1EMNmeJw2*k.MyClass': 0, '!Y-QN1EMNmeJw2*k.MyClass.method': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '!Y-QN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'import os\ndef bar(): pass': {*()}, '!Y-QN1EMNmeJw2*k': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass', '!Y-QN1EMNmeJw2*k': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass.method': '!Y-QN1EMNmeJw2*k'}
    var_9 = '"""Module docstring."""\ndef func(): pass'
    var_10 = var_0.parse(var_1, var_9)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0, '!Y-QN1EMNmeJw2*k': 0, '!Y-QN1EMNmeJw2*k.MyClass': 0, '!Y-QN1EMNmeJw2*k.MyClass.method': 0, '\n': 0, '\n.func': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '!Y-QN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'\n': 'Module docstring.'}
    assert var_0.imp == {'test_module2': {*()}, 'import os\ndef bar(): pass': {*()}, '!Y-QN1EMNmeJw2*k': {*()}, '\n': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass', '!Y-QN1EMNmeJw2*k': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass.method': '!Y-QN1EMNmeJw2*k', '\n': '\n', '\n.func': '\n'}
    var_11 = 'm)\x0c)Q'
    var_12 = "CONSTA.T = 42\n__all__ = ['foo']"
    var_13 = var_0.parse(var_11, var_12)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0, '!Y-QN1EMNmeJw2*k': 0, '!Y-QN1EMNmeJw2*k.MyClass': 0, '!Y-QN1EMNmeJw2*k.MyClass.method': 0, '\n': 0, '\n.func': 0, 'm)\x0c)Q': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '!Y-QN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'm)\x0c)Q': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'import os\ndef bar(): pass': {*()}, '!Y-QN1EMNmeJw2*k': {*()}, '\n': {*()}, 'm)\x0c)Q': {'m)\x0c)Q.foo'}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass', '!Y-QN1EMNmeJw2*k': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass.method': '!Y-QN1EMNmeJw2*k', '\n': '\n', '\n.func': '\n', 'm)\x0c)Q': 'm)\x0c)Q'}
    assert var_0.alias == {'test_module2.os': 'os', 'm)\x0c)Q.__all__': "['foo']"}
    var_14 = 'from typing import List\nVector = List[float]'
    var_15 = var_0.__post_init__()
    var_16 = var_0.compile()
    assert var_16 == '## Module `\n`\n<a id="\n"></a>\n\nModule docstring.\n\n### func()\n\n*Full name:* `\n.func`\n<a id="\n-func"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n## Module `!Y-QN1EMNmeJw2*k`\n<a id="!y-qn1emnmejw2*k"></a>\n\n### class MyClass\n\n*Full name:* `!Y-QN1EMNmeJw2*k.MyClass`\n<a id="!y-qn1emnmejw2*k-myclass"></a>\n\n#### MyClass.method()\n\n*Full name:* `!Y-QN1EMNmeJw2*k.MyClass.method`\n<a id="!y-qn1emnmejw2*k-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n## Module `import os\ndef bar(): pass`\n<a id="import os\ndef bar(): pass"></a>\n\n### class MyClass\n\n*Full name:* `import os\ndef bar(): pass.MyClass`\n<a id="import os\ndef bar(): pass-myclass"></a>\n\n#### MyClass.method()\n\n*Full name:* `import os\ndef bar(): pass.MyClass.method`\n<a id="import os\ndef bar(): pass-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n## Module `test_module2`\n<a id="test_module2"></a>\n\n### bar()\n\n*Full name:* `test_module2.bar`\n<a id="test_module2-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_17 = 'def nested(): pass'
    var_18 = var_0.parse(var_14, var_17)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0, '!Y-QN1EMNmeJw2*k': 0, '!Y-QN1EMNmeJw2*k.MyClass': 0, '!Y-QN1EMNmeJw2*k.MyClass.method': 0, '\n': 0, '\n.func': 0, 'm)\x0c)Q': 0, 'from typing import List\nVector = List[float]': 0, 'from typing import List\nVector = List[float].nested': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '!Y-QN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'm)\x0c)Q': '## Module `{}`\n<a id="{}"></a>\n\n', 'from typing import List\nVector = List[float]': '## Module `{}`\n<a id="{}"></a>\n\n', 'from typing import List\nVector = List[float].nested': '### nested()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'import os\ndef bar(): pass': {*()}, '!Y-QN1EMNmeJw2*k': {*()}, '\n': {*()}, 'm)\x0c)Q': {'m)\x0c)Q.foo'}, 'from typing import List\nVector = List[float]': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass', '!Y-QN1EMNmeJw2*k': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass.method': '!Y-QN1EMNmeJw2*k', '\n': '\n', '\n.func': '\n', 'm)\x0c)Q': 'm)\x0c)Q', 'from typing import List\nVector = List[float]': 'from typing import List\nVector = List[float]', 'from typing import List\nVector = List[float].nested': 'from typing import List\nVector = List[float]'}
    var_19 = 'test_module8'
    var_20 = '@staticmethod\ndef static_method(): pass'
    var_21 = var_0.parse(var_19, var_20)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0, '!Y-QN1EMNmeJw2*k': 0, '!Y-QN1EMNmeJw2*k.MyClass': 0, '!Y-QN1EMNmeJw2*k.MyClass.method': 0, '\n': 0, '\n.func': 0, 'm)\x0c)Q': 0, 'from typing import List\nVector = List[float]': 0, 'from typing import List\nVector = List[float].nested': 0, 'test_module8': 0, 'test_module8.static_method': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '!Y-QN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'm)\x0c)Q': '## Module `{}`\n<a id="{}"></a>\n\n', 'from typing import List\nVector = List[float]': '## Module `{}`\n<a id="{}"></a>\n\n', 'from typing import List\nVector = List[float].nested': '### nested()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module8': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module8.static_method': '### static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}, 'import os\ndef bar(): pass': {*()}, '!Y-QN1EMNmeJw2*k': {*()}, '\n': {*()}, 'm)\x0c)Q': {'m)\x0c)Q.foo'}, 'from typing import List\nVector = List[float]': {*()}, 'test_module8': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass', '!Y-QN1EMNmeJw2*k': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass.method': '!Y-QN1EMNmeJw2*k', '\n': '\n', '\n.func': '\n', 'm)\x0c)Q': 'm)\x0c)Q', 'from typing import List\nVector = List[float]': 'from typing import List\nVector = List[float]', 'from typing import List\nVector = List[float].nested': 'from typing import List\nVector = List[float]', 'test_module8': 'test_module8', 'test_module8.static_method': 'test_module8'}
    var_22 = var_0.load_docstring(var_7, var_1)
    var_0.is_public(var_21)

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
    var_1 = 'test_module2'
    var_2 = 'import os\ndef bar(): pass'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2'}
    assert var_0.alias == {'test_module2.os': 'os'}
    var_4 = 'class MyClass:\n    def method(self): pass'
    var_5 = '!Y-QN1EMNmeJw2*k'
    var_6 = var_0.parse(var_5, var_4)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, '!Y-QN1EMNmeJw2*k': 0, '!Y-QN1EMNmeJw2*k.MyClass': 0, '!Y-QN1EMNmeJw2*k.MyClass.method': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '!Y-QN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'test_module2': {*()}, '!Y-QN1EMNmeJw2*k': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', '!Y-QN1EMNmeJw2*k': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass.method': '!Y-QN1EMNmeJw2*k'}
    var_7 = '"""Module docstring."""\ndef func(): pass'
    var_8 = var_0.parse(var_5, var_1)
    var_9 = 'm)\x0c)Q'
    var_10 = "CONSTANT = 42\n__all__ = ['foo']"
    var_11 = '_heh6~r|C)a`bM"&'
    var_12 = var_0.parse(var_11, var_7)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, '!Y-QN1EMNmeJw2*k': 0, '!Y-QN1EMNmeJw2*k.MyClass': 0, '!Y-QN1EMNmeJw2*k.MyClass.method': 0, '_heh6~r|C)a`bM"&': 0, '_heh6~r|C)a`bM"&.func': 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '!Y-QN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '_heh6~r|C)a`bM"&': '## Module `{}`\n<a id="{}"></a>\n\n', '_heh6~r|C)a`bM"&.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'_heh6~r|C)a`bM"&': 'Module docstring.'}
    assert var_0.imp == {'test_module2': {*()}, '!Y-QN1EMNmeJw2*k': {*()}, '_heh6~r|C)a`bM"&': {*()}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', '!Y-QN1EMNmeJw2*k': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass.method': '!Y-QN1EMNmeJw2*k', '_heh6~r|C)a`bM"&': '_heh6~r|C)a`bM"&', '_heh6~r|C)a`bM"&.func': '_heh6~r|C)a`bM"&'}
    var_13 = var_0.parse(var_10, var_10)
    assert var_0.level == {'test_module2': 0, 'test_module2.bar': 0, '!Y-QN1EMNmeJw2*k': 0, '!Y-QN1EMNmeJw2*k.MyClass': 0, '!Y-QN1EMNmeJw2*k.MyClass.method': 0, '_heh6~r|C)a`bM"&': 0, '_heh6~r|C)a`bM"&.func': 0, "CONSTANT = 42\n__all__ = ['foo']": 0}
    assert var_0.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '!Y-QN1EMNmeJw2*k': '## Module `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '!Y-QN1EMNmeJw2*k.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '_heh6~r|C)a`bM"&': '## Module `{}`\n<a id="{}"></a>\n\n', '_heh6~r|C)a`bM"&.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "CONSTANT = 42\n__all__ = ['foo']": '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module2': {*()}, '!Y-QN1EMNmeJw2*k': {*()}, '_heh6~r|C)a`bM"&': {*()}, "CONSTANT = 42\n__all__ = ['foo']": {"CONSTANT = 42\n__all__ = ['foo'].foo"}}
    assert var_0.root == {'test_module2': 'test_module2', 'test_module2.bar': 'test_module2', '!Y-QN1EMNmeJw2*k': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass': '!Y-QN1EMNmeJw2*k', '!Y-QN1EMNmeJw2*k.MyClass.method': '!Y-QN1EMNmeJw2*k', '_heh6~r|C)a`bM"&': '_heh6~r|C)a`bM"&', '_heh6~r|C)a`bM"&.func': '_heh6~r|C)a`bM"&', "CONSTANT = 42\n__all__ = ['foo']": "CONSTANT = 42\n__all__ = ['foo']", "CONSTANT = 42\n__all__ = ['foo'].CONSTANT": "CONSTANT = 42\n__all__ = ['foo']"}
    assert var_0.alias == {'test_module2.os': 'os', "CONSTANT = 42\n__all__ = ['foo'].CONSTANT": '42', "CONSTANT = 42\n__all__ = ['foo'].__all__": "['foo']"}
    assert var_0.const == {"CONSTANT = 42\n__all__ = ['foo'].CONSTANT": 'int'}
    var_14 = var_0.compile()
    assert var_14 == '## Module `!Y-QN1EMNmeJw2*k`\n<a id="!y-qn1emnmejw2*k"></a>\n\n### class MyClass\n\n*Full name:* `!Y-QN1EMNmeJw2*k.MyClass`\n<a id="!y-qn1emnmejw2*k-myclass"></a>\n\n#### MyClass.method()\n\n*Full name:* `!Y-QN1EMNmeJw2*k.MyClass.method`\n<a id="!y-qn1emnmejw2*k-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n## Module `CONSTANT = 42\n__all__ = [\'foo\']`\n<a id="constant = 42\n__all__ = [\'foo\']"></a>\n\n## Module `test_module2`\n<a id="test_module2"></a>\n\n### bar()\n\n*Full name:* `test_module2.bar`\n<a id="test_module2-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_14.visit_Constant(var_9)

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
    var_1 = 'tt_mdule2'
    var_2 = 'import os\ndef bar(): pass'
    var_3 = module_0.walk_body(var_0)
    var_4 = 'F2'
    var_5 = var_0.compile()
    assert var_5 == '\n'
    var_6 = var_0.parse(var_1, var_2)
    assert var_0.level == {'tt_mdule2': 0, 'tt_mdule2.bar': 0}
    assert var_0.doc == {'tt_mdule2': '## Module `{}`\n<a id="{}"></a>\n\n', 'tt_mdule2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'tt_mdule2': {*()}}
    assert var_0.root == {'tt_mdule2': 'tt_mdule2', 'tt_mdule2.bar': 'tt_mdule2'}
    assert var_0.alias == {'tt_mdule2.os': 'os'}
    var_7 = 'class MyClass:\n    def method(self): ass'
    var_8 = var_0.parse(var_2, var_7)
    assert var_0.level == {'tt_mdule2': 0, 'tt_mdule2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0}
    assert var_0.doc == {'tt_mdule2': '## Module `{}`\n<a id="{}"></a>\n\n', 'tt_mdule2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.imp == {'tt_mdule2': {*()}, 'import os\ndef bar(): pass': {*()}}
    assert var_0.root == {'tt_mdule2': 'tt_mdule2', 'tt_mdule2.bar': 'tt_mdule2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass'}
    var_9 = var_0.__repr__()
    assert var_9 == 'Parser(link=True, b_level=1, toc=False, level={\'tt_mdule2\': 0, \'tt_mdule2.bar\': 0, \'import os\\ndef bar(): pass\': 0, \'import os\\ndef bar(): pass.MyClass\': 0, \'import os\\ndef bar(): pass.MyClass.method\': 0}, doc={\'tt_mdule2\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'tt_mdule2.bar\': \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', \'import os\\ndef bar(): pass\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'import os\\ndef bar(): pass.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'import os\\ndef bar(): pass.MyClass.method\': \'#### MyClass.method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | return |\\n|:----:|:------:|\\n| `Self` | `Any` |\\n\\n\'}, docstring={}, imp={\'tt_mdule2\': set(), \'import os\\ndef bar(): pass\': set()}, root={\'tt_mdule2\': \'tt_mdule2\', \'tt_mdule2.bar\': \'tt_mdule2\', \'import os\\ndef bar(): pass\': \'import os\\ndef bar(): pass\', \'import os\\ndef bar(): pass.MyClass\': \'import os\\ndef bar(): pass\', \'import os\\ndef bar(): pass.MyClass.method\': \'import os\\ndef bar(): pass\'}, alias={\'tt_mdule2.os\': \'os\'}, const={})'
    var_10 = '"""Module docstring."""\ndef func(): pass'
    var_11 = module_3.getdoc(var_8)
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
    var_12 = var_0.func_ann(var_9, var_9, has_self=var_6, cls_method=var_11)
    var_13 = module_0.const_type(var_9)
    assert var_13 == 'Any'
    var_14 = "CONSTANT = 42\n__all__ =O['foo']"
    var_15 = var_0.parse(var_4, var_14)
    assert var_0.level == {'tt_mdule2': 0, 'tt_mdule2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0, 'F2': 0}
    assert var_0.doc == {'tt_mdule2': '## Module `{}`\n<a id="{}"></a>\n\n', 'tt_mdule2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'F2': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'tt_mdule2': {*()}, 'import os\ndef bar(): pass': {*()}, 'F2': {*()}}
    assert var_0.root == {'tt_mdule2': 'tt_mdule2', 'tt_mdule2.bar': 'tt_mdule2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass', 'F2': 'F2', 'F2.CONSTANT': 'F2'}
    assert var_0.alias == {'tt_mdule2.os': 'os', 'F2.CONSTANT': '42', 'F2.__all__': "O['foo']"}
    assert var_0.const == {'F2.CONSTANT': 'int'}
    var_16 = module_1.field(repr=var_8, hash=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'dataclasses.Field'
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
    var_17 = var_0.compile()
    assert var_17 == '## Module `F2`\n<a id="f2"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n\n## Module `import os\ndef bar(): pass`\n<a id="import os\ndef bar(): pass"></a>\n\n### class MyClass\n\n*Full name:* `import os\ndef bar(): pass.MyClass`\n<a id="import os\ndef bar(): pass-myclass"></a>\n\n#### MyClass.method()\n\n*Full name:* `import os\ndef bar(): pass.MyClass.method`\n<a id="import os\ndef bar(): pass-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n## Module `tt_mdule2`\n<a id="tt_mdule2"></a>\n\n### bar()\n\n*Full name:* `tt_mdule2.bar`\n<a id="tt_mdule2-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_18 = module_2.NodeVisitor()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.NodeVisitor'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_19 = '@acmeh\ndef static_method(): pass'
    var_20 = var_0.parse(var_1, var_19)
    assert var_0.level == {'tt_mdule2': 0, 'tt_mdule2.bar': 0, 'import os\ndef bar(): pass': 0, 'import os\ndef bar(): pass.MyClass': 0, 'import os\ndef bar(): pass.MyClass.method': 0, 'F2': 0, 'tt_mdule2.static_method': 0}
    assert var_0.doc == {'tt_mdule2': '## Module `{}`\n<a id="{}"></a>\n\n', 'tt_mdule2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'import os\ndef bar(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\ndef bar(): pass.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'F2': '## Module `{}`\n<a id="{}"></a>\n\n', 'tt_mdule2.static_method': '### static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@acmeh` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.root == {'tt_mdule2': 'tt_mdule2', 'tt_mdule2.bar': 'tt_mdule2', 'import os\ndef bar(): pass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass': 'import os\ndef bar(): pass', 'import os\ndef bar(): pass.MyClass.method': 'import os\ndef bar(): pass', 'F2': 'F2', 'F2.CONSTANT': 'F2', 'tt_mdule2.static_method': 'tt_mdule2'}
    var_21 = var_0.load_docstring(var_1, var_10)
    var_0.imports(var_18, var_6)

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
    var_1 = None
    var_2 = []
    var_3 = [var_1, var_1]
    var_4 = module_2.Assign(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Assign'
    assert var_4.targets is None
    assert var_4.value is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Assign.type_comment is None
    var_5 = [var_4]
    var_0.class_api(var_1, var_1, var_2, var_5)

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
    var_1 = 'tt_module2'
    var_2 = 'import os\ndef bar(): pass'
    var_3 = module_0.walk_body(var_0)
    var_4 = var_0.compile()
    assert var_4 == '\n'
    var_5 = var_0.parse(var_1, var_2)
    assert var_0.level == {'tt_module2': 0, 'tt_module2.bar': 0}
    assert var_0.doc == {'tt_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'tt_module2.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'tt_module2': {*()}}
    assert var_0.root == {'tt_module2': 'tt_module2', 'tt_module2.bar': 'tt_module2'}
    assert var_0.alias == {'tt_module2.os': 'os'}
    var_6 = module_2.expr()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.expr'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.expr.end_lineno is None
    assert module_2.expr.end_col_offset is None
    var_7 = [var_6, var_6, var_6]
    var_0.class_api(var_5, var_5, var_7, var_5)