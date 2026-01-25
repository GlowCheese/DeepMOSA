# Check out: https://github.com/GlowCheese/deepmosa
import ast as module_1
import dataclasses as module_2
import inspect as module_3

import apimd.parser as module_0
import pytest


def test_case_0():
    var_0 = 'value'
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
    var_1 = 'class MVClass(BaseClass): pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class MVClass(BaseClass): pass': 0, 'class MVClass(BaseClass): pass.MVClass': 0}
    assert var_0.doc == {'class MVClass(BaseClass): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class MVClass(BaseClass): pass.MVClass': '### class MVClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `BaseClass` |\n\n'}
    assert var_0.imp == {'class MVClass(BaseClass): pass': {*()}}
    assert var_0.root == {'class MVClass(BaseClass): pass': 'class MVClass(BaseClass): pass', 'class MVClass(BaseClass): pass.MVClass': 'class MVClass(BaseClass): pass'}

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = False
    var_1 = True
    var_2 = None
    var_3 = module_0.Parser(b_level=var_0, toc=var_1, const=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level is False
    assert var_3.toc is True
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {}
    assert var_3.const is None
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_3.is_public(var_2)

def test_case_3():
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
    var_1 = '\nCONSANT = 42\nANOTHER_CONS: int = 100\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': 0}
    assert var_0.doc == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': {*()}}
    assert var_0.root == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n'}
    assert var_0.alias == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': '42', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': '100'}
    assert var_0.const == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': 'int', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': 'int'}
    var_3 = var_0.load_docstring(var_1, var_2)

def test_case_5():
    pass

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.parent(var_0)

def test_case_7():
    var_0 = "'a|b'"
    var_1 = 'eval'
    var_2 = module_1.parse(var_0, mode=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Expression'
    assert f'{type(var_2.body).__module__}.{type(var_2.body).__qualname__}' == 'ast.Constant'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = var_2.body
    var_4 = [var_3]
    var_5 = module_0._defaults(var_4)
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_6 = list(var_5)
    var_7 = bool('&#124;' in var_6[0])
    assert var_7 is True
    var_8 = bool('<code>' in var_6[0])
    assert var_8 is True

def test_case_8():
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

def test_case_9():
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

def test_case_10():
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
    var_1 = 'h33'
    var_0.is_public(var_1)
    assert var_2 is False

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
    var_1 = "\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n": 1, "\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n.func": 1}
    assert var_0.doc == {"\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n.func": '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n'}
    assert var_0.docstring == {"\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n.func": 'Function with annotations.'}
    assert var_0.imp == {"\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n": {*()}}
    assert var_0.root == {"\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n": "\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n", "\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n.func": "\ndef func(x: int) -> str:\n    '''Function with annotations.'''\n    return str(x)\n"}

def test_case_13():
    var_0 = ').vc>,iu\rAv{9a'
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

def test_case_14():
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
    var_1 = '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n': 0}
    assert var_0.doc == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n': {*()}}
    assert var_0.root == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n': '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n', '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.CONSTANT': '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n', '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.ANOTHER_CONST': '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n'}
    assert var_0.alias == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.CONSTANT': '42', '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.ANOTHER_CONST': '100'}
    assert var_0.const == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.CONSTANT': 'int', '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.ANOTHER_CONST': 'int'}

def test_case_16():
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
    var_1 = "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": 0}
    assert var_0.doc == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar"}}
    assert var_0.root == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n"}
    assert var_0.alias == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.__all__": "['foo', 'bar']"}

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
    var_1 = '\nimport os\nfrom typing import List\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nimport os\nfrom typing import List\n': 0}
    assert var_0.doc == {'\nimport os\nfrom typing import List\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nimport os\nfrom typing import List\n': {*()}}
    assert var_0.root == {'\nimport os\nfrom typing import List\n': '\nimport os\nfrom typing import List\n'}
    assert var_0.alias == {'\nimport os\nfrom typing import List\n.os': 'os', '\nimport os\nfrom typing import List\n.List': 'typing.List'}
    with pytest.raises(TypeError):
        var_3 = bool(var_1 >= 2)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
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
    var_1 = "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.foo": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.bar": 0}
    assert var_0.doc == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.bar": '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n": {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.foo", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.bar"}}
    assert var_0.root == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.foo": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.bar": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n"}
    assert var_0.alias == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.__all__": "['foo', 'bar']"}
    var_3 = var_0.__repr__()
    assert var_3 == 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})'
    var_4 = var_0.globals(var_3, var_3)
    var_5 = 'DxW<'
    var_6 = var_0.parse(var_5, var_1)
    assert var_0.level == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.foo": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.bar": 0, 'DxW<': 0, 'DxW<.foo': 0, 'DxW<.bar': 0}
    assert var_0.doc == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.bar": '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'DxW<': '## Module `{}`\n<a id="{}"></a>\n\n', 'DxW<.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'DxW<.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n": {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.foo", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.bar"}, 'DxW<': {'DxW<.foo', 'DxW<.bar'}}
    assert var_0.root == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.foo": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.bar": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n", 'DxW<': 'DxW<', 'DxW<.foo': 'DxW<', 'DxW<.bar': 'DxW<'}
    assert var_0.alias == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pJss\ndef bar():\n    pss\n.__all__": "['foo', 'bar']", 'DxW<.__all__': "['foo', 'bar']"}
    var_7 = module_0.esc_underscore(var_3)
    assert var_7 == 'Parser(link=True, b\\_level=1, toc=False, level={"\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n": 0, "\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.foo": 0, "\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n": {"\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.foo", "\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n": "\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n", "\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.foo": "\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n", "\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.bar": "\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n"}, alias={"\\n\\_\\_all\\_\\_ = [\'foo\', \'bar\']\\ndef foo():\\n    pJss\\ndef bar():\\n    pss\\n.\\_\\_all\\_\\_": "[\'foo\', \'bar\']"}, const={})'
    var_8 = module_2.field()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'dataclasses.Field'
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
    var_9 = var_0.compile()
    assert var_9 == '## Module `\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pJss\ndef bar():\n    pss\n`\n<a id="\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pjss\ndef bar():\n    pss\n"></a>\n\n### bar()\n\n*Full name:* `\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pJss\ndef bar():\n    pss\n.bar`\n<a id="\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pjss\ndef bar():\n    pss\n-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n### foo()\n\n*Full name:* `\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pJss\ndef bar():\n    pss\n.foo`\n<a id="\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pjss\ndef bar():\n    pss\n-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n## Module `DxW<`\n<a id="dxw<"></a>\n\n### bar()\n\n*Full name:* `DxW<.bar`\n<a id="dxw<-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n### foo()\n\n*Full name:* `DxW<.foo`\n<a id="dxw<-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_10 = var_0.load_docstring(var_5, var_6)
    var_11 = module_0.const_type(var_3)
    assert var_11 == 'Any'
    var_3.visit_Constant(var_6)

@pytest.mark.xfail(strict=True)
def test_case_19():
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

def test_case_20():
    var_0 = False
    var_1 = module_0.Parser(var_0, var_0)
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
    var_2 = '\nclass OuterClass:\n    def inner_method(self):\n        pass\n'
    var_3 = var_1.parse(var_2, var_2)
    assert var_1.level == {'\nclass OuterClass:\n    def inner_method(self):\n        pass\n': 0, '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass': 0, '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass.inner_method': 0}
    assert var_1.doc == {'\nclass OuterClass:\n    def inner_method(self):\n        pass\n': '# Module `{}`\n\n', '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass': '## class OuterClass\n\n*Full name:* `{}`\n\n', '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass.inner_method': '### OuterClass.inner_method()\n\n*Full name:* `{}`\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_1.imp == {'\nclass OuterClass:\n    def inner_method(self):\n        pass\n': {*()}}
    assert var_1.root == {'\nclass OuterClass:\n    def inner_method(self):\n        pass\n': '\nclass OuterClass:\n    def inner_method(self):\n        pass\n', '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass': '\nclass OuterClass:\n    def inner_method(self):\n        pass\n', '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass.inner_method': '\nclass OuterClass:\n    def inner_method(self):\n        pass\n'}

def test_case_21():
    var_0 = None
    var_1 = None
    var_2 = module_0.Parser(docstring=var_1, imp=var_1, alias=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc == {}
    assert var_2.docstring is None
    assert var_2.imp is None
    assert var_2.root == {}
    assert var_2.alias is None
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

@pytest.mark.xfail(strict=True)
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
    var_1 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n'
    var_2 = module_1.parse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Module'
    assert f'{type(var_2.body).__module__}.{type(var_2.body).__qualname__}' == 'builtins.list'
    assert len(var_2.body) == 1
    assert var_2.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = 'test_module.TestClass'
    var_6 = var_4.bases
    var_7 = var_4.body
    var_0.class_api(var_5, var_5, var_6, var_7)

def test_case_23():
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

def test_case_24():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_1.Call(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Call'
    assert var_2.func is None
    assert var_2.args is None
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
    var_1 = 'class MVClass(BaseClass): pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class MVClass(BaseClass): pass': 0, 'class MVClass(BaseClass): pass.MVClass': 0}
    assert var_0.doc == {'class MVClass(BaseClass): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class MVClass(BaseClass): pass.MVClass': '### class MVClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `BaseClass` |\n\n'}
    assert var_0.imp == {'class MVClass(BaseClass): pass': {*()}}
    assert var_0.root == {'class MVClass(BaseClass): pass': 'class MVClass(BaseClass): pass', 'class MVClass(BaseClass): pass.MVClass': 'class MVClass(BaseClass): pass'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `class MVClass(BaseClass): pass`\n<a id="class mvclass(baseclass): pass"></a>\n\n### class MVClass\n\n*Full name:* `class MVClass(BaseClass): pass.MVClass`\n<a id="class mvclass(baseclass): pass-mvclass"></a>\n\n| Bases |\n|:-----:|\n| `BaseClass` |\n'

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
    var_0 = True
    var_1 = module_0.Parser(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is True
    assert var_1.b_level is True
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
    var_2 = '\nclass OuterClass:\n    def inner_method(self):\n        pass\n'
    var_3 = var_1.parse(var_2, var_2)
    assert var_1.level == {'\nclass OuterClass:\n    def inner_method(self):\n        pass\n': 0, '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass': 0, '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass.inner_method': 0}
    assert var_1.doc == {'\nclass OuterClass:\n    def inner_method(self):\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass': '### class OuterClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass.inner_method': '#### OuterClass.inner_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_1.imp == {'\nclass OuterClass:\n    def inner_method(self):\n        pass\n': {*()}}
    assert var_1.root == {'\nclass OuterClass:\n    def inner_method(self):\n        pass\n': '\nclass OuterClass:\n    def inner_method(self):\n        pass\n', '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass': '\nclass OuterClass:\n    def inner_method(self):\n        pass\n', '\nclass OuterClass:\n    def inner_method(self):\n        pass\n.OuterClass.inner_method': '\nclass OuterClass:\n    def inner_method(self):\n        pass\n'}

@pytest.mark.xfail(strict=True)
def test_case_28():
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
    var_1 = '\nCONSANT = 42\nANOTHER_CONS: int = 100\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': 0}
    assert var_0.doc == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': {*()}}
    assert var_0.root == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n'}
    assert var_0.alias == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': '42', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': '100'}
    assert var_0.const == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': 'int', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': 'int'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nCONSANT = 42\nANOTHER_CONS: int = 100\n`\n<a id="\nconsant = 42\nanother_cons: int = 100\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSANT` | `int` |\n| `ANOTHER_CONS` | `int` |\n'

def test_case_30():
    var_0 = None
    var_1 = 'attr'
    var_2 = module_0._attr(var_0, var_1)
    assert var_2 is None
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_1 = "\nclass TestClass:\n    vatue | 42\n    name = 'test'\n"
    var_2 = module_1.parse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Module'
    assert f'{type(var_2.body).__module__}.{type(var_2.body).__qualname__}' == 'builtins.list'
    assert len(var_2.body) == 1
    assert var_2.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = -1
    var_4 = var_2.body[var_3]
    var_5 = var_4.bases
    var_6 = var_4.body
    var_0.class_api(var_4, var_6, var_5, var_6)

@pytest.mark.xfail(strict=True)
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
    var_1 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n'
    var_2 = module_1.parse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Module'
    assert f'{type(var_2.body).__module__}.{type(var_2.body).__qualname__}' == 'builtins.list'
    assert len(var_2.body) == 1
    assert var_2.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_4.bases
    var_6 = var_4.body
    var_0.class_api(var_4, var_4, var_5, var_6)

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
    var_1 = "\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n": 1, "\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n.in_try": 1}
    assert var_0.doc == {"\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n.in_try": '### in_try()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {"\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n.in_try": 'In try block.'}
    assert var_0.imp == {"\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n": {*()}}
    assert var_0.root == {"\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n": "\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n", "\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n.in_try": "\ntry:\n    def in_try():\n        '''In try block.'''\n        pass\nexcept:\n    pass\n"}

def test_case_34():
    var_0 = 'if True:\n    if True:\n        if True:\n            x = 1'
    var_1 = module_1.parse(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'ast.Module'
    assert f'{type(var_1.body).__module__}.{type(var_1.body).__qualname__}' == 'builtins.list'
    assert len(var_1.body) == 1
    assert var_1.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
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
    with pytest.raises(TypeError):
        var_5 = len(var_1)
    assert var_5 == 1

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
    var_1 = '\nCONSANT = 42\nANOTHER_CONS: int = 100\n'
    var_2 = ''
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'': 0}
    assert var_0.doc == {'': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'': {*()}}
    assert var_0.root == {'': '', 'CONSANT': '', 'ANOTHER_CONS': ''}
    assert var_0.alias == {'CONSANT': '42', 'ANOTHER_CONS': '100'}
    assert var_0.const == {'CONSANT': 'int', 'ANOTHER_CONS': 'int'}
    var_4 = var_0.compile()
    assert var_4 == '\n'

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
    var_1 = '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n': 0}
    assert var_0.doc == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n': {*()}}
    assert var_0.root == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n': '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n', '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.CONSTANT': '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n', '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.ANOTHER_CONST': '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n'}
    assert var_0.alias == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.CONSTANT': '42', '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.ANOTHER_CONST': '100'}
    assert var_0.const == {'\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.CONSTANT': 'int', '\nCONSTANT = 42\nANOTHER_CONST: int = 100\n.ANOTHER_CONST': 'int'}
    var_3 = 'YObI'
    var_4 = var_0.load_docstring(var_3, var_1)

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
    var_1 = "\nclass TestClass:\n    vatue | 42\n    name = 'test'\n"
    var_2 = module_1.parse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Module'
    assert f'{type(var_2.body).__module__}.{type(var_2.body).__qualname__}' == 'builtins.list'
    assert len(var_2.body) == 1
    assert var_2.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = ':0juL&9A*^BaXXi!oM'
    var_4 = var_0.resolve(var_3, var_2, var_2)
    assert var_4 == 'class TestClass:\n    vatue | 42\n    name = test'
    with pytest.raises(AttributeError):
        var_5 = var_2.bases

def test_case_38():
    var_0 = None
    var_1 = [var_0, var_0, var_0]
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
    var_4 = bool(var_3 == [' ', ' ', ' '])
    assert var_4 is True

def test_case_39():
    var_0 = []
    var_1 = module_0._defaults(var_0)
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_3 = bool(var_2 == [])
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
    var_2 = 0
    var_3 = 'def func(*args: int, **kwargs: str) -> None: pass'
    var_4 = module_1.parse(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Module'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'builtins.list'
    assert len(var_4.body) == 1
    assert var_4.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = var_4.body[var_2]
    var_6 = 'test_module.func'
    var_7 = var_5.args
    var_8 = var_5.returns
    var_9 = False
    var_10 = False
    var_0.func_api(var_1, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)

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
    var_1 = 0
    var_2 = "def func(x: int, *, y: str = 'test') -> bool: pass"
    var_3 = module_1.parse(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Module'
    assert f'{type(var_3.body).__module__}.{type(var_3.body).__qualname__}' == 'builtins.list'
    assert len(var_3.body) == 1
    assert var_3.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_4 = var_3.body[var_1]
    var_5 = var_4.args
    var_6 = False
    var_7 = False
    var_0.func_api(var_2, var_3, var_5, var_1, has_self=var_6, cls_method=var_7)

def test_case_42():
    var_0 = 2
    var_1 = module_0.Parser(b_level=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is True
    assert var_1.b_level == 2
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
    var_2 = "\n'''Module.'''\n"
    var_3 = var_1.parse(var_2, var_2)
    assert var_1.level == {"\n'''Module.'''\n": 1}
    assert var_1.doc == {"\n'''Module.'''\n": '### Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.docstring == {"\n'''Module.'''\n": 'Module.'}
    assert var_1.imp == {"\n'''Module.'''\n": {*()}}
    assert var_1.root == {"\n'''Module.'''\n": "\n'''Module.'''\n"}
    var_4 = var_1.doc[var_2]

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
    var_1 = '\nimport sys\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nimport sys\n': 0}
    assert var_0.doc == {'\nimport sys\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nimport sys\n': {*()}}
    assert var_0.root == {'\nimport sys\n': '\nimport sys\n'}
    assert var_0.alias == {'\nimport sys\n.sys': 'sys'}

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
    var_1 = "\n@property\ndef prop():\n    '''Property.'''\n    return 42\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'test_module': 0, 'test_module.prop': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.prop': '### prop()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@property` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'test_module.prop': 'Property.'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.prop': 'test_module'}
    var_4 = bool('test_module.prop' in var_0.doc)
    assert var_4 is True

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
    var_1 = "\nasync def async_func():\n    '''Async function.'''\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'test_module': 0, 'test_module.async_func': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.async_func': '### async async_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'test_module.async_func': 'Async function.'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.async_func': 'test_module'}

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
    var_1 = "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n"
    var_2 = 'WV'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'WV': 0, 'WV.foo': 0, 'WV.bar': 0}
    assert var_0.doc == {'WV': '## Module `{}`\n<a id="{}"></a>\n\n', 'WV.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'WV.bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'WV': {'WV.foo', 'WV.bar'}}
    assert var_0.root == {'WV': 'WV', 'WV.foo': 'WV', 'WV.bar': 'WV'}
    assert var_0.alias == {'WV.__all__': "['foo', 'bar']"}
    var_4 = var_0.compile()
    assert var_4 == '## Module `WV`\n<a id="wv"></a>\n\n### bar()\n\n*Full name:* `WV.bar`\n<a id="wv-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n### foo()\n\n*Full name:* `WV.foo`\n<a id="wv-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'

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
    var_1 = 'class MyClass(BaseCl-ss): pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class MyClass(BaseCl-ss): pass': 0, 'class MyClass(BaseCl-ss): pass.MyClass': 0}
    assert var_0.doc == {'class MyClass(BaseCl-ss): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class MyClass(BaseCl-ss): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `BaseCl - ss` |\n\n'}
    assert var_0.imp == {'class MyClass(BaseCl-ss): pass': {*()}}
    assert var_0.root == {'class MyClass(BaseCl-ss): pass': 'class MyClass(BaseCl-ss): pass', 'class MyClass(BaseCl-ss): pass.MyClass': 'class MyClass(BaseCl-ss): pass'}
    var_3 = ' Z\'[L"\'iIB^LT9}"'
    var_4 = []
    var_5 = var_0.compile()
    assert var_5 == '## Module `class MyClass(BaseCl-ss): pass`\n<a id="class myclass(basecl-ss): pass"></a>\n\n### class MyClass\n\n*Full name:* `class MyClass(BaseCl-ss): pass.MyClass`\n<a id="class myclass(basecl-ss): pass-myclass"></a>\n\n| Bases |\n|:-----:|\n| `BaseCl - ss` |\n'
    var_6 = var_0.compile()
    assert var_6 == '## Module `class MyClass(BaseCl-ss): pass`\n<a id="class myclass(basecl-ss): pass"></a>\n\n### class MyClass\n\n*Full name:* `class MyClass(BaseCl-ss): pass.MyClass`\n<a id="class myclass(basecl-ss): pass-myclass"></a>\n\n| Bases |\n|:-----:|\n| `BaseCl - ss` |\n'
    var_7 = {var_1: var_2, var_3: var_2}
    var_8 = module_1.ImportFrom(*var_4, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.ImportFrom'
    assert var_8.class MyClass(BaseCl-ss): pass is None
    assert var_8. Z'[L"'iIB^LT9}" is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_9 = var_0.imports(var_3, var_8)
    var_10 = var_0.globals(var_1, var_2)
    var_11 = var_0.__post_init__()
    var_0.load_docstring(var_2, var_9)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = None
    var_3 = var_0.__eq__(var_2)
    var_4 = "\nclass TestClass:\n    value | 42\n    name = 'test'\n"
    var_5 = module_1.parse(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Module'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'builtins.list'
    assert len(var_5.body) == 1
    assert var_5.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.__post_init__()
    var_9 = 'te:t_wodule'
    var_10 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n'}
    var_11 = var_7.bases
    var_12 = var_7.body
    var_13 = var_0.load_docstring(var_9, var_12)
    var_14 = var_0.class_api(var_9, var_1, var_11, var_12)
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `name` | `str` |\n\n'}

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
    var_2 = var_0.__post_init__()
    var_3 = 'class MyClass(BaseCl-ss): pass'
    var_4 = 1142
    var_5 = module_0.parent(var_3, level=var_4)
    assert var_5 == 'class MyClass(BaseCl-ss): pass'
    var_6 = var_0.parse(var_3, var_3)
    assert var_0.level == {'class MyClass(BaseCl-ss): pass': 0, 'class MyClass(BaseCl-ss): pass.MyClass': 0}
    assert var_0.doc == {'class MyClass(BaseCl-ss): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class MyClass(BaseCl-ss): pass.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `BaseCl - ss` |\n\n'}
    assert var_0.imp == {'class MyClass(BaseCl-ss): pass': {*()}}
    assert var_0.root == {'class MyClass(BaseCl-ss): pass': 'class MyClass(BaseCl-ss): pass', 'class MyClass(BaseCl-ss): pass.MyClass': 'class MyClass(BaseCl-ss): pass'}
    var_7 = module_0.const_type(var_6)
    assert var_7 == 'Any'
    var_8 = ' Z\'[L"\'iIB^LT9}"'
    var_9 = [var_8, var_6, var_1]
    var_10 = module_0.esc_underscore(var_3)
    assert var_10 == 'class MyClass(BaseCl-ss): pass'
    var_11 = {var_3: var_6, var_8: var_6}
    var_12 = module_1.ImportFrom(*var_9, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.ImportFrom'
    assert var_12.module == ' Z\'[L"\'iIB^LT9}"'
    assert var_12.names is None
    assert var_12.level == '\n'
    assert var_12.class MyClass(BaseCl-ss): pass is None
    assert var_12. Z'[L"'iIB^LT9}" is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_13 = module_0.doctest(var_5)
    assert var_13 == 'class MyClass(BaseCl-ss): pass'
    var_0.imports(var_8, var_12)

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
    var_1 = "\nclass Outer:\n    '''Outer class.'''\n    class Inner:\n        '''Inner class.'''\n        pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'test_module': 0, 'test_module.Outer': 0, 'test_module.Outer.Inner': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.Outer': '### class Outer\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.Outer.Inner': '#### class Outer.Inner\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.docstring == {'test_module.Outer': 'Outer class.', 'test_module.Outer.Inner': 'Inner class.'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.Outer': 'test_module', 'test_module.Outer.Inner': 'test_module'}
    var_4 = var_0.compile()
    assert var_4 == '## Module `test_module`\n<a id="test_module"></a>\n\n### class Outer\n\n*Full name:* `test_module.Outer`\n<a id="test_module-outer"></a>\n\nOuter class.\n\n#### class Outer.Inner\n\n*Full name:* `test_module.Outer.Inner`\n<a id="test_module-outer-inner"></a>\n\nInner class.\n'
    var_5 = bool('test_module.Outer' in var_0.doc)
    assert var_5 is True

def test_case_51():
    var_0 = '__name__.public.module'
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
    var_1 = "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": 0}
    assert var_0.doc == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar"}}
    assert var_0.root == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n"}
    assert var_0.alias == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.__all__": "['foo', 'bar']"}
    var_3 = 'WV'
    var_4 = var_0.__repr__()
    assert var_4 == 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})'
    var_5 = var_0.parse(var_4, var_1)
    assert var_0.level == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": 0, 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})': 9, 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).foo': 9, 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).bar': 9}
    assert var_0.doc == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})': '## Module `{}`\n<a id="{}"></a>\n\n', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar"}, 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})': {'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).foo', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).bar'}}
    assert var_0.root == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n", 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})': 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).foo': 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).bar': 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})'}
    assert var_0.alias == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.__all__": "['foo', 'bar']", 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).__all__': "['foo', 'bar']"}
    var_6 = var_0.parse(var_3, var_3)
    assert var_0.level == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": 0, 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})': 9, 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).foo': 9, 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).bar': 9, 'WV': 0}
    assert var_0.doc == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})': '## Module `{}`\n<a id="{}"></a>\n\n', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).bar': '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'WV': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar"}, 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})': {'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).foo', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).bar'}, 'WV': {*()}}
    assert var_0.root == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pss\n", 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})': 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).foo': 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})', 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).bar': 'Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={})', 'WV': 'WV'}
    var_7 = var_0.compile()
    assert var_7 == '## Module `\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pass\ndef bar():\n    pss\n`\n<a id="\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pass\ndef bar():\n    pss\n"></a>\n\n### bar()\n\n*Full name:* `\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pass\ndef bar():\n    pss\n.bar`\n<a id="\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pass\ndef bar():\n    pss\n-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n### foo()\n\n*Full name:* `\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pass\ndef bar():\n    pss\n.foo`\n<a id="\n__all__ = [\'foo\', \'bar\']\ndef foo():\n    pass\ndef bar():\n    pss\n-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n### bar()\n\n*Full name:* `Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).bar`\n<a id="parser(link=true, b_level=1, toc=false, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-foo": \'### foo()\\n\\n*full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-bar": \'### bar()\\n\\n*full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-__all__": "[\'foo\', \'bar\']"}, const={})-bar"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n### foo()\n\n*Full name:* `Parser(link=True, b_level=1, toc=False, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": \'### bar()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n.__all__": "[\'foo\', \'bar\']"}, const={}).foo`\n<a id="parser(link=true, b_level=1, toc=false, level={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-foo": 0, "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-bar": 0}, doc={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": \'## module `{}`\\n<a id="{}"></a>\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-foo": \'### foo()\\n\\n*full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `any` |\\n\\n\', "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-bar": \'### bar()\\n\\n*full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `any` |\\n\\n\'}, docstring={}, imp={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": {"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-foo", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-bar"}}, root={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-foo": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n", "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-bar": "\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n"}, alias={"\\n__all__ = [\'foo\', \'bar\']\\ndef foo():\\n    pass\\ndef bar():\\n    pss\\n-__all__": "[\'foo\', \'bar\']"}, const={})-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_0.imports(var_1, var_2)

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
    var_1 = '\nCONSANT = 42\nANOTHER_CONS: int = 100\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': 0}
    assert var_0.doc == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': {*()}}
    assert var_0.root == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n'}
    assert var_0.alias == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': '42', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': '100'}
    assert var_0.const == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': 'int', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': 'int'}
    var_3 = ''
    var_4 = var_0.parse(var_3, var_1)
    assert var_0.level == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': 0, '': 0}
    assert var_0.doc == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': '## Module `{}`\n<a id="{}"></a>\n\n', '': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': {*()}, '': {*()}}
    assert var_0.root == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '': '', 'CONSANT': '', 'ANOTHER_CONS': ''}
    assert var_0.alias == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': '42', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': '100', 'CONSANT': '42', 'ANOTHER_CONS': '100'}
    assert var_0.const == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': 'int', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': 'int', 'CONSANT': 'int', 'ANOTHER_CONS': 'int'}
    var_5 = var_0.compile()
    assert var_5 == '## Module `\nCONSANT = 42\nANOTHER_CONS: int = 100\n`\n<a id="\nconsant = 42\nanother_cons: int = 100\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSANT` | `int` |\n| `ANOTHER_CONS` | `int` |\n'

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
    var_1 = '\nclass TestClass:\n    attr1: int\n    attr2: str\n    del attr2\n    '
    var_2 = module_1.parse(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Module'
    assert f'{type(var_2.body).__module__}.{type(var_2.body).__qualname__}' == 'builtins.list'
    assert len(var_2.body) == 1
    assert var_2.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = 0
    var_4 = var_2.body[var_3]
    var_5 = var_4.bases
    var_6 = var_4.body
    var_0.class_api(var_1, var_5, var_5, var_6)

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
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x = y = 5'
    var_4 = module_1.parse(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Module'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'builtins.list'
    assert len(var_4.body) == 1
    assert var_4.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = var_0.alias
    var_9 = len(var_8)
    assert var_9 == 0

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
    var_1 = 'test_module'
    var_2 = set()
    var_3 = 'x: int'
    var_4 = module_1.parse(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Module'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'builtins.list'
    assert len(var_4.body) == 1
    assert var_4.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = var_0.globals(var_1, var_6)
    var_8 = var_0.alias
    var_9 = len(var_8)
    assert var_9 == 0

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
    var_2 = 'a, b = 1, 2'
    var_3 = module_1.parse(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Module'
    assert f'{type(var_3.body).__module__}.{type(var_3.body).__qualname__}' == 'builtins.list'
    assert len(var_3.body) == 1
    assert var_3.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_4 = 0
    var_5 = var_3.body[var_4]
    var_6 = var_0.globals(var_1, var_5)
    var_7 = var_0.alias
    var_8 = len(var_7)
    assert var_8 == 0

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
    var_2 = set()
    var_3 = "x = 'hello'  # type: str"
    var_4 = True
    var_5 = module_1.parse(var_3, type_comments=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Module'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'builtins.list'
    assert len(var_5.body) == 1
    assert var_5.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = 0
    var_7 = var_5.body[var_6]
    var_8 = var_0.globals(var_1, var_7)
    assert var_0.alias == {'test_module.x': "'hello'"}
    var_9 = bool('test_module.x' in var_0.alias)
    assert var_9 is True
    with pytest.raises(KeyError):
        var_10 = var_0.const['test_module.x']
    assert var_10 == 'str'

@pytest.mark.xfail(strict=True)
def test_case_59():
    var_0 = False
    var_1 = None
    var_2 = module_0.Parser(var_0, var_1, var_1, const=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is False
    assert var_2.b_level is None
    assert var_2.toc is None
    assert var_2.level == {}
    assert var_2.doc == {}
    assert var_2.docstring == {}
    assert var_2.imp == {}
    assert var_2.root == {}
    assert var_2.alias == {}
    assert var_2.const is None
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_3 = [var_1, var_1]
    var_4 = module_1.Tuple(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Tuple'
    assert var_4.elts is None
    assert var_4.ctx is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_5 = module_0.const_type(var_4)
    assert var_5 == 'tuple'
    var_6 = None
    var_7 = module_3.getdoc(var_1)
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
    var_8 = var_2.globals(var_1, var_7)
    var_9 = module_3.getdoc(var_6)
    var_10 = var_2.__post_init__()
    var_10.compile()

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
    var_1 = '\nCONSANT = 42\nANOTHER_CONS: int = 100\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': 0}
    assert var_0.doc == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': {*()}}
    assert var_0.root == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': '\nCONSANT = 42\nANOTHER_CONS: int = 100\n'}
    assert var_0.alias == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': '42', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': '100'}
    assert var_0.const == {'\nCONSANT = 42\nANOTHER_CONS: int = 100\n.CONSANT': 'int', '\nCONSANT = 42\nANOTHER_CONS: int = 100\n.ANOTHER_CONS': 'int'}
    var_3 = var_0.parse(var_1, var_1)
    var_4 = var_0.compile()
    assert var_4 == '## Module `\nCONSANT = 42\nANOTHER_CONS: int = 100\n`\n<a id="\nconsant = 42\nanother_cons: int = 100\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSANT` | `int` |\n| `ANOTHER_CONS` | `int` |\n'

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
    var_1 = '|p\\e8+#U[L6}bz7:pw'
    var_2 = None
    var_3 = var_0.globals(var_1, var_2)
    var_4 = var_0.compile()
    assert var_4 == '\n'
    var_5 = [var_2, var_2]
    var_6 = module_1.Subscript(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Subscript'
    assert var_6.value is None
    assert var_6.slice is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_0.resolve(var_1, var_6)