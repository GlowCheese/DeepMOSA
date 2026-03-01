# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.parser as module_0
import dataclasses as module_1
import ast as module_2
import inspect as module_3
import builtins as module_4

def test_case_0():
    var_0 = 'YVP&*&\t))'
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
    var_1 = "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.public_func": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._private_func": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.PublicClass": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._PrivateClass": 0}
    assert var_0.doc == {"\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.PublicClass": '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._PrivateClass": '### class _PrivateClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n": {"\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.public_func", "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.PublicClass"}}
    assert var_0.root == {"\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n": "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.public_func": "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._private_func": "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.PublicClass": "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._PrivateClass": "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n"}
    assert var_0.alias == {"\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.__all__": "['public_func', 'PublicClass']"}

def test_case_2():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is True
    assert var_1.b_level == 1
    assert var_1.toc is True
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
    var_2 = var_1.compile()
    assert var_2 == '**Table of contents:**\n\n\n'

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
    var_0 = True
    var_1 = None
    var_2 = module_0.Parser(toc=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is True
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
    var_3 = var_2.load_docstring(var_0, var_1)

def test_case_5():
    pass

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.parent(var_0)

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
    var_1 = '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': 1, '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': 1}
    assert var_0.doc == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | <code>str &#124; None</code> | `bool` |\n|   | `None` |   |   |\n\n'}
    assert var_0.docstring == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': 'Function with annotations.'}
    assert var_0.imp == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': {*()}}
    assert var_0.root == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n', '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n'}
    assert var_0.alias == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.Optional': 'typing.Optional'}

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
    var_1 = "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.pblic_func": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._private_func": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.PublicClass": 0, "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._PrivateClass": 0}
    assert var_0.doc == {"\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.pblic_func": '### pblic_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.PublicClass": '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._PrivateClass": '### class _PrivateClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n": {"\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.PublicClass", "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.public_func"}}
    assert var_0.root == {"\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n": "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.pblic_func": "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._private_func": "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.PublicClass": "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n", "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n._PrivateClass": "\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n"}
    assert var_0.alias == {"\n__all__ = ['public_func', 'PublicClass']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.__all__": "['public_func', 'PublicClass']"}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\n__all__ = [\'public_func\', \'PublicClass\']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n`\n<a id="\n__all__ = [\'public_func\', \'publicclass\']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass publicclass:\n    pass\nclass _privateclass:\n    pass\n"></a>\n\n### class PublicClass\n\n*Full name:* `\n__all__ = [\'public_func\', \'PublicClass\']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n.PublicClass`\n<a id="\n__all__ = [\'public_func\', \'publicclass\']\ndef pblic_func():\n    pass\ndef _private_func():\n    pass\nclass publicclass:\n    pass\nclass _privateclass:\n    pass\n-publicclass"></a>\n'

def test_case_12():
    var_0 = 'T\t'
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

@pytest.mark.xfail(strict=True)
def test_case_14():
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
def test_case_15():
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

@pytest.mark.xfail(strict=True)
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
    var_1 = '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': 1, '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': 1}
    assert var_0.doc == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | <code>str &#124; None</code> | `bool` |\n|   | `None` |   |   |\n\n'}
    assert var_0.docstring == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': 'Function with annotations.'}
    assert var_0.imp == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': {*()}}
    assert var_0.root == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n', '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n'}
    assert var_0.alias == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.Optional': 'typing.Optional'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n`\n<a id="\nfrom typing import optional\ndef foo(x: int, y: optional[str] = none) -> bool:\n    """function with annotations-"""\n    return true\n"></a>\n\n### foo()\n\n*Full name:* `\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo`\n<a id="\nfrom typing import optional\ndef foo(x: int, y: optional[str] = none) -> bool:\n    """function with annotations-"""\n    return true\n-foo"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | <code>str &#124; None</code> | `bool` |\n|   | `None` |   |   |\n\nFunction with annotations.\n'
    var_4 = var_0.globals(var_3, var_1)
    var_0.load_docstring(var_2, var_4)

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
    var_1 = 'test_module.TestClass'
    var_2 = []
    var_3 = 0
    var_4 = "_x: int = 1\n__y: str = 'test'"
    var_5 = module_2.parse(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Module'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'builtins.list'
    assert len(var_5.body) == 2
    assert var_5.type_ignores == []
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_6 = var_5.body[var_3]
    var_7 = [var_6]
    var_8 = var_0.class_api(var_6, var_1, var_2, var_7)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'

@pytest.mark.xfail(strict=True)
def test_case_19():
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
    var_11 = var_10.__repr__()
    var_12 = module_0.is_public_family(var_11)
    assert var_12 is True
    var_13 = module_1.field(repr=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'dataclasses.Field'
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
    var_14 = var_13.__repr__()
    assert var_14 == 'Field(name=None,type=None,default=<dataclasses._MISSING_TYPE object at 0x7aa007843010>,default_factory=<dataclasses._MISSING_TYPE object at 0x7aa007843010>,init=True,repr=None,hash=None,compare=True,metadata=mappingproxy({}),kw_only=<dataclasses._MISSING_TYPE object at 0x7aa007843010>,_field_type=None)'
    var_15 = var_10.compile()
    assert var_15 == '\n'
    var_16 = '~1G2\r*lr7?y,Y\\V]3YU'
    var_10.class_api(var_11, var_16, var_14, var_0)

def test_case_20():
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

def test_case_21():
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
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass.method': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass.static_method': 2}
    assert var_0.doc == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass': 'Class docstring.', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass.method': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n.MyClass.static_method': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n    @staticmethod\n    def static_method():\n        pass\n'}
    var_3 = bool('test_module.MyClass' in var_0.doc)
    assert var_3 is True
    with pytest.raises(KeyError):
        var_4 = var_0.docstring['test_module.MyClass']
    assert var_4 == '```python\n"""Class docstring."""\n```'

def test_case_22():
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
    var_2 = module_2.ImportFrom()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.ImportFrom'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_3 = module_0.parent(var_1)
    assert var_3 == 'Any'
    var_4 = module_0.table(items=var_3)
    assert var_4 == '||\n||\n| A |\n| n |\n| y |\n\n'

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
    var_1 = 'test_module.TestClass'
    var_2 = 0
    var_3 = 'x: int = 1\ndel x'
    var_4 = var_0.parse(var_1, var_3)
    assert var_0.level == {'test_module.TestClass': 1}
    assert var_0.doc == {'test_module.TestClass': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module.TestClass': {*()}}
    assert var_0.root == {'test_module.TestClass': 'test_module.TestClass'}
    assert var_0.alias == {'test_module.TestClass.x': '1'}
    with pytest.raises(AttributeError):
        var_5 = var_4.body[var_2]

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = True
    var_1 = None
    var_2 = module_0.Parser(toc=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is True
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
    var_3 = 'Y-GM'
    var_4 = module_2.List()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.List'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_5 = [var_4]
    var_6 = var_2.__post_init__()
    var_7 = [var_6, var_6, var_6]
    var_2.class_api(var_3, var_1, var_5, var_7)

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
    var_1 = 'pkg'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'pkg': 0}
    assert var_0.doc == {'pkg': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'pkg': {*()}}
    assert var_0.root == {'pkg': 'pkg'}

def test_case_26():
    var_0 = ''
    var_1 = [var_0, var_0, var_0]
    var_2 = module_0._m(*var_1)
    assert var_2 == ''
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_27():
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

def test_case_28():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Resolver'
    assert var_2.root == 'root'
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
    var_3 = 'invalid syntax'
    var_4 = [var_3]
    var_5 = module_2.Constant(*var_4, **var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Constant'
    assert var_5.value == 'invalid syntax'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_6 = var_2.visit_Constant(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Constant'
    assert var_6.value == 'invalid syntax'

def test_case_29():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = {var_1: var_0}
    var_3 = module_0.Resolver(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'root'
    assert var_3.alias == {'root.name': 'root'}
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
    var_4 = 'name'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_2.Constant(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 'name'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_8 = var_3.visit_Constant(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Name'
    assert var_8.id == 'root'
    assert f'{type(var_8.ctx).__module__}.{type(var_8.ctx).__qualname__}' == 'ast.Load'
    assert var_8.lineno == 1
    assert var_8.col_offset == 0
    assert var_8.end_lineno == 1
    assert var_8.end_col_offset == 4

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = {var_1: var_0}
    var_3 = module_0.Resolver(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'root'
    assert var_3.alias == {'root.name': 'root'}
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
    var_4 = [var_1]
    var_5 = {}
    var_6 = module_2.Constant(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Constant'
    assert var_6.value == 'root.name'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_7 = None
    var_8 = var_3.visit_Constant(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Attribute'
    assert f'{type(var_8.value).__module__}.{type(var_8.value).__qualname__}' == 'ast.Name'
    assert var_8.attr == 'name'
    assert f'{type(var_8.ctx).__module__}.{type(var_8.ctx).__qualname__}' == 'ast.Load'
    assert var_8.lineno == 1
    assert var_8.col_offset == 0
    assert var_8.end_lineno == 1
    assert var_8.end_col_offset == 9
    var_9 = module_3.getdoc(var_7)
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
    var_9.compile()

def test_case_31():
    var_0 = True
    var_1 = module_0.Parser(toc=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is True
    assert var_1.b_level == 1
    assert var_1.toc is True
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
    var_2 = module_2.ImportFrom()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.ImportFrom'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_3 = var_1.imports(var_0, var_2)

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
    var_1 = '\nfrom typing import List\nimport os\nx = 1\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'test_module': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module'}
    assert var_0.alias == {'test_module.List': 'typing.List', 'test_module.os': 'os', 'test_module.x': '1'}
    var_4 = bool('os' in var_0.alias['test_module.os'])
    assert var_4 is True
    var_5 = bool('test_module.x' in var_0.alias)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'ali&s'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Resolver'
    assert var_4.root == 'root'
    assert var_4.alias == {'root.name': 'ali&s'}
    assert var_4.self_ty == ''
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_5 = 'name'
    var_6 = [var_5]
    var_7 = None
    var_8 = module_0.walk_body(var_7)
    var_9 = {}
    var_10 = module_2.Constant(*var_6, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == 'name'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = module_0.walk_body(var_3)
    var_12 = [var_6, var_6, var_6]
    var_13 = module_2.Attribute(*var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Attribute'
    assert var_13.value == ['name']
    assert var_13.attr == ['name']
    assert var_13.ctx == ['name']
    var_14 = module_0.const_type(var_13)
    assert var_14 == 'Any'
    var_15 = var_4.visit_Constant(var_10)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.BinOp'
    assert f'{type(var_15.left).__module__}.{type(var_15.left).__qualname__}' == 'ast.Name'
    assert f'{type(var_15.op).__module__}.{type(var_15.op).__qualname__}' == 'ast.BitAnd'
    assert f'{type(var_15.right).__module__}.{type(var_15.right).__qualname__}' == 'ast.Name'
    assert var_15.lineno == 1
    assert var_15.col_offset == 0
    assert var_15.end_lineno == 1
    assert var_15.end_col_offset == 5
    var_16 = module_0.table(items=var_2)
    assert var_16 == '||\n||\n| a |\n| l |\n| i |\n| & |\n| s |\n\n'
    var_17 = var_4.visit(var_13)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Attribute'
    assert var_17.value == ['name']
    assert var_17.attr == ['name']
    assert var_17.ctx == ['name']
    var_18 = module_2.BitOr()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.BitOr'
    var_19 = var_4.generic_visit(var_15)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.BinOp'
    assert f'{type(var_19.left).__module__}.{type(var_19.left).__qualname__}' == 'ast.Name'
    assert f'{type(var_19.op).__module__}.{type(var_19.op).__qualname__}' == 'ast.BitAnd'
    assert f'{type(var_19.right).__module__}.{type(var_19.right).__qualname__}' == 'ast.Name'
    assert var_19.lineno == 1
    assert var_19.col_offset == 0
    assert var_19.end_lineno == 1
    assert var_19.end_col_offset == 5
    var_19.func_api(var_18, var_2, var_2, var_7, has_self=var_19, cls_method=var_15)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 'z( a9hH{!nb&*,u`y<'
    var_1 = 'alias'
    var_2 = {var_0: var_1}
    var_3 = module_0.Resolver(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'z( a9hH{!nb&*,u`y<'
    assert var_3.alias == {'z( a9hH{!nb&*,u`y<': 'alias'}
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
    var_4 = "!Z!Y]<'n\t"
    var_5 = '\x0c'
    var_6 = [var_4, var_5, var_4]
    var_7 = module_0.table(*var_6, items=var_1)
    assert var_7 == "| !Z!Y]<'n\t | \x0c | !Z!Y]<'n\t |\n|:---------:|:---:|:---------:|\n| a |\n| l |\n| i |\n| a |\n| s |\n\n"
    var_8 = [var_5]
    var_9 = {}
    var_10 = module_2.Constant(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == '\x0c'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = module_2.Attribute(*var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Attribute'
    var_12 = module_0.const_type(var_11)
    assert var_12 == 'Any'
    var_3.visit_Constant(var_10)

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
    var_1 = var_0.__post_init__()
    var_2 = [var_1]
    var_3 = 42
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_2.Constant(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 42
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_8 = []
    var_9 = 'targets'
    var_10 = 'value'
    var_11 = {var_9: var_2, var_10: var_7}
    var_12 = module_2.Assign(*var_8, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Assign'
    assert var_12.targets == [None]
    assert f'{type(var_12.value).__module__}.{type(var_12.value).__qualname__}' == 'ast.Constant'
    assert module_2.Assign.type_comment is None
    var_13 = var_0.globals(var_10, var_12)
    with pytest.raises(KeyError):
        var_14 = var_0.alias['root.test_var']
    assert var_14 == '42'

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
    var_2 = 'test_var'
    var_3 = []
    var_4 = 'id'
    var_5 = {var_4: var_2}
    var_6 = module_2.Name(*var_3, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Name'
    assert var_6.id == 'test_var'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_7 = var_0.__repr__()
    assert var_7 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_8 = 42
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_2.Constant(*var_9, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Constant'
    assert var_12.value == 42
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_13 = None
    var_14 = []
    var_15 = 'targets'
    var_16 = 'value'
    var_17 = 'type_comment'
    var_18 = {var_15: var_7, var_16: var_12, var_17: var_13}
    var_19 = module_2.Assign(*var_14, **var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.Assign'
    assert var_19.targets == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert f'{type(var_19.value).__module__}.{type(var_19.value).__qualname__}' == 'ast.Constant'
    assert var_19.type_comment is None
    assert module_2.Assign.type_comment is None
    var_20 = var_0.globals(var_1, var_19)
    var_21 = set()
    with pytest.raises(KeyError):
        var_22 = var_0.imp[var_1]

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
    var_1 = 'test_var'
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
    var_3 = []
    var_4 = {}
    var_5 = module_2.Name(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_6 = var_0.__repr__()
    assert var_6 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_7 = []
    var_8 = 'target'
    var_9 = 'annotation'
    var_10 = 'value'
    var_11 = {var_8: var_2, var_9: var_5, var_10: var_6}
    var_12 = module_2.AnnAssign(*var_7, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.AnnAssign'
    assert var_12.target == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
    assert f'{type(var_12.annotation).__module__}.{type(var_12.annotation).__qualname__}' == 'ast.Name'
    assert var_12.value == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert module_2.AnnAssign.value is None
    var_13 = 'root'
    var_14 = var_0.globals(var_13, var_12)
    with pytest.raises(KeyError):
        var_15 = var_0.alias['root.test_var']
    assert var_15 == '42'

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
    var_1 = 'int'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_2.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == 'int'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_6 = 42
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_2.Constant(*var_7, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == 42
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = []
    var_12 = 'target'
    var_13 = 'annotation'
    var_14 = {var_12: var_5, var_13: var_5, var_3: var_10}
    var_15 = module_2.AnnAssign(*var_11, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.AnnAssign'
    assert f'{type(var_15.target).__module__}.{type(var_15.target).__qualname__}' == 'ast.Name'
    assert f'{type(var_15.annotation).__module__}.{type(var_15.annotation).__qualname__}' == 'ast.Name'
    assert f'{type(var_15.id).__module__}.{type(var_15.id).__qualname__}' == 'ast.Constant'
    assert module_2.AnnAssign.value is None
    var_16 = 'root'
    var_17 = var_0.globals(var_16, var_15)
    with pytest.raises(KeyError):
        var_18 = var_0.const['root.TEST_VAR']
    assert var_18 == 'int'

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
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_2.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == '__all__'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = 'foo'
    var_8 = 'valueG'
    var_9 = {var_8: var_7}
    var_10 = module_2.Constant(*var_2, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.valueG == 'foo'
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = 'elts'
    var_12 = {var_11: var_6}
    var_13 = module_2.Tuple(*var_2, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_13.elts).__module__}.{type(var_13.elts).__qualname__}' == 'builtins.list'
    assert len(var_13.elts) == 1
    assert f'{type(module_2.Tuple.dims).__module__}.{type(module_2.Tuple.dims).__qualname__}' == 'builtins.property'
    var_14 = []
    var_15 = 'targets'
    var_16 = 'value'
    var_17 = {var_15: var_6, var_16: var_13}
    var_18 = module_2.Assign(*var_14, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Assign'
    assert f'{type(var_18.targets).__module__}.{type(var_18.targets).__qualname__}' == 'builtins.list'
    assert len(var_18.targets) == 1
    assert f'{type(var_18.value).__module__}.{type(var_18.value).__qualname__}' == 'ast.Tuple'
    assert module_2.Assign.type_comment is None
    var_19 = 'root'
    var_20 = var_0.globals(var_19, var_18)
    assert var_0.alias == {'root.__all__': '(__all__,)'}
    with pytest.raises(KeyError):
        var_21 = bool('__all__' in var_0.imp['root'])
    assert var_21 is True

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = []
    var_1 = []
    var_2 = 'elts'
    var_3 = {var_2: var_0}
    var_4 = module_2.List(*var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.List'
    assert var_4.elts == []
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_5 = module_0.const_type(var_4)
    assert var_5 == 'list'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_6 = None
    var_7 = None
    var_8 = module_1.dataclass(eq=var_7, order=var_7)
    assert f'{type(module_1.MISSING).__module__}.{type(module_1.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_1.KW_ONLY).__module__}.{type(module_1.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_8.visit_Subscript(var_6)

def test_case_41():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = {var_1: var_1}
    var_3 = module_0.Resolver(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'root'
    assert var_3.alias == {'root.name': 'root.name'}
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
    var_4 = module_0.doctest(var_1)
    assert var_4 == 'root.name'
    var_5 = 'name'
    var_6 = [var_5]
    var_7 = None
    var_8 = module_0.walk_body(var_7)
    var_9 = {}
    var_10 = module_2.Constant(*var_6, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == 'name'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = module_0.walk_body(var_2)
    var_12 = None
    var_13 = [var_12, var_12, var_12]
    var_14 = module_2.Attribute(*var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Attribute'
    assert var_14.value is None
    assert var_14.attr is None
    assert var_14.ctx is None
    var_15 = module_0.const_type(var_14)
    assert var_15 == 'Any'
    var_16 = var_3.visit_Constant(var_10)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Name'
    assert var_16.id == 'name'
    assert f'{type(var_16.ctx).__module__}.{type(var_16.ctx).__qualname__}' == 'ast.Load'
    assert var_16.lineno == 1
    assert var_16.col_offset == 0
    assert var_16.end_lineno == 1
    assert var_16.end_col_offset == 4
    var_17 = module_0.table(items=var_1)
    assert var_17 == '||\n||\n| r |\n| o |\n| o |\n| t |\n| . |\n| n |\n| a |\n| m |\n| e |\n\n'
    var_18 = module_0.is_public_family(var_0)
    assert var_18 is True

def test_case_42():
    var_0 = []
    var_1 = module_0._e_type(*var_0)
    assert var_1 == ''
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_43():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_2.Constant(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Constant'
    assert var_3.value == 1
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_2.Constant(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 'a'
    var_8 = [var_3, var_7]
    var_9 = [var_8]
    var_10 = module_0._e_type(*var_9)
    assert var_10 == '[Any]'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_44():
    var_0 = '__init__'
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
    var_2 = '__main__'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = '__name__'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True

def test_case_45():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Resolver'
    assert var_2.root == 'root'
    assert var_2.alias == {}
    assert var_2.self_ty == 'root'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_4 = {}
    var_5 = module_2.Load(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Load'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_0, var_5]
    var_7 = {}
    var_8 = module_2.Name(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Name'
    assert var_8.id == 'root'
    assert f'{type(var_8.ctx).__module__}.{type(var_8.ctx).__qualname__}' == 'ast.Load'
    var_9 = var_2.visit_Name(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Name'
    assert var_9.id == 'Self'
    assert f'{type(var_9.ctx).__module__}.{type(var_9.ctx).__qualname__}' == 'ast.Load'
    var_10 = var_9.id
    assert var_10 == 'Self'

def test_case_46():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = 'some_func()'
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Resolver'
    assert var_5.root == 'root'
    assert var_5.alias == {'root.name': 'some_func()'}
    assert var_5.self_ty == 'Self'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_6 = 'name'
    var_7 = []
    var_8 = {}
    var_9 = module_2.Load(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Load'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_10 = [var_6, var_9]
    var_11 = {}
    var_12 = module_2.Name(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Name'
    assert var_12.id == 'name'
    assert f'{type(var_12.ctx).__module__}.{type(var_12.ctx).__qualname__}' == 'ast.Load'
    var_13 = var_5.visit_Name(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Call'
    assert f'{type(var_13.func).__module__}.{type(var_13.func).__qualname__}' == 'ast.Name'
    assert var_13.args == []
    assert var_13.keywords == []
    assert var_13.lineno == 1
    assert var_13.col_offset == 0
    assert var_13.end_lineno == 1
    assert var_13.end_col_offset == 11

def test_case_47():
    var_0 = 'root'
    var_1 = 'root.name'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = 'Self'
    var_5 = module_0.Resolver(var_0, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Resolver'
    assert var_5.root == 'root'
    assert var_5.alias == {'root.name': "typing.TypeVar('T')"}
    assert var_5.self_ty == 'Self'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_6 = 'name'
    var_7 = []
    var_8 = {}
    var_9 = module_2.Load(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Load'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_10 = [var_6, var_9]
    var_11 = {}
    var_12 = module_2.Name(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Name'
    assert var_12.id == 'name'
    assert f'{type(var_12.ctx).__module__}.{type(var_12.ctx).__qualname__}' == 'ast.Load'
    var_13 = var_5.visit_Name(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Call'
    assert f'{type(var_13.func).__module__}.{type(var_13.func).__qualname__}' == 'ast.Name'
    assert f'{type(var_13.args).__module__}.{type(var_13.args).__qualname__}' == 'builtins.list'
    assert len(var_13.args) == 1
    assert var_13.keywords == []
    assert var_13.lineno == 1
    assert var_13.col_offset == 0
    assert var_13.end_lineno == 1
    assert var_13.end_col_offset == 19

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
    var_1 = 'test_module'
    var_2 = 'test_module.TestEnum'
    var_3 = 0
    var_4 = 'from enum import Enum\nclass Base(Enum): pass'
    var_5 = module_2.parse(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Module'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'builtins.list'
    assert len(var_5.body) == 2
    assert var_5.type_ignores == []
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = 'A = 1\nB = 2'
    var_8 = module_2.parse(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Module'
    assert f'{type(var_8.body).__module__}.{type(var_8.body).__qualname__}' == 'builtins.list'
    assert len(var_8.body) == 2
    assert var_8.type_ignores == []
    var_9 = var_8.body[var_3]
    var_10 = [var_9]
    var_0.class_api(var_1, var_2, var_6, var_10)

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
    var_1 = 'TestClass'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_2.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == 'TestClass'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_6 = 'int'
    var_7 = 'id'
    var_8 = module_2.Name(*var_2, **var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Name'
    assert var_8.id == 'TestClass'
    var_9 = None
    var_10 = []
    var_11 = 'target'
    var_12 = 'annotation'
    var_13 = {var_11: var_5, var_12: var_8, var_6: var_9}
    var_14 = module_2.AnnAssign(*var_10, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.AnnAssign'
    assert f'{type(var_14.target).__module__}.{type(var_14.target).__qualname__}' == 'ast.Name'
    assert f'{type(var_14.annotation).__module__}.{type(var_14.annotation).__qualname__}' == 'ast.Name'
    assert var_14.int is None
    assert module_2.AnnAssign.value is None
    var_15 = []
    var_16 = 'id'
    var_17 = {var_16: var_16}
    var_18 = module_2.Name(*var_15, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Name'
    assert var_18.id == 'id'
    var_19 = [var_18]
    var_20 = 'targets'
    var_21 = {var_20: var_19}
    var_22 = module_2.Delete(*var_10, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Delete'
    assert f'{type(var_22.targets).__module__}.{type(var_22.targets).__qualname__}' == 'builtins.list'
    assert len(var_22.targets) == 1
    var_23 = [var_14, var_22]
    var_24 = []
    var_25 = 'bases'
    var_26 = 'body'
    var_27 = {var_7: var_1, var_25: var_24, var_26: var_23, var_11: var_24}
    var_28 = module_2.ClassDef(*var_24, **var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'ast.ClassDef'
    assert var_28.id == 'TestClass'
    assert var_28.bases == []
    assert f'{type(var_28.body).__module__}.{type(var_28.body).__qualname__}' == 'builtins.list'
    assert len(var_28.body) == 2
    assert var_28.target == []
    var_29 = 'root'
    var_30 = var_28.bases
    var_31 = var_28.body
    var_0.class_api(var_29, var_1, var_30, var_31)

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
    var_1 = 'TestClass'
    var_2 = []
    var_3 = 'attr1'
    var_4 = 'id'
    var_5 = {var_4: var_3}
    var_6 = []
    var_7 = 'id'
    var_8 = module_2.Name(*var_6, **var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Name'
    assert var_8.id == 'attr1'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_9 = None
    var_10 = []
    var_11 = 'target'
    var_12 = 'annotation'
    var_13 = {var_11: var_8, var_12: var_8, var_7: var_9}
    var_14 = module_2.AnnAssign(*var_10, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.AnnAssign'
    assert f'{type(var_14.target).__module__}.{type(var_14.target).__qualname__}' == 'ast.Name'
    assert f'{type(var_14.annotation).__module__}.{type(var_14.annotation).__qualname__}' == 'ast.Name'
    assert var_14.id is None
    assert module_2.AnnAssign.value is None
    var_15 = var_0.__repr__()
    assert var_15 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_16 = [var_15]
    var_17 = []
    var_18 = 'targets'
    var_19 = {var_18: var_16}
    var_20 = module_2.Delete(*var_17, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.Delete'
    assert var_20.targets == ['Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})']
    var_21 = [var_14, var_20]
    var_22 = []
    var_23 = []
    var_24 = 'bases'
    var_25 = 'body'
    var_26 = 'decorator_list'
    var_27 = {var_25: var_1, var_24: var_2, var_25: var_21, var_26: var_22}
    var_28 = module_2.ClassDef(*var_23, **var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'ast.ClassDef'
    assert f'{type(var_28.body).__module__}.{type(var_28.body).__qualname__}' == 'builtins.list'
    assert len(var_28.body) == 2
    assert var_28.bases == []
    assert var_28.decorator_list == []
    var_29 = 'root'
    var_30 = 'root.TestClass'
    var_31 = var_28.bases
    var_32 = var_28.body
    var_0.class_api(var_29, var_30, var_31, var_32)

def test_case_51():
    var_0 = []
    var_1 = {}
    var_2 = module_2.stmt(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.stmt'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.stmt.end_lineno is None
    assert module_2.stmt.end_col_offset is None
    var_3 = [var_2]
    var_4 = []
    var_5 = {}
    var_6 = module_2.stmt(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.stmt'
    var_7 = [var_6]
    var_8 = [var_3, var_7]
    var_9 = {}
    var_10 = module_2.If(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.If'
    assert f'{type(var_10.test).__module__}.{type(var_10.test).__qualname__}' == 'builtins.list'
    assert len(var_10.test) == 1
    assert f'{type(var_10.body).__module__}.{type(var_10.body).__qualname__}' == 'builtins.list'
    assert len(var_10.body) == 1
    var_11 = [var_10]
    var_12 = module_0.walk_body(var_11)
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
        var_13 = list(var_12)

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
    var_1 = 'TestClass'
    var_2 = []
    var_3 = 'attr1'
    var_4 = 'id'
    var_5 = []
    var_6 = 'id'
    var_7 = {var_6: var_4}
    var_8 = module_2.Name(*var_5, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Name'
    assert var_8.id == 'id'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_9 = None
    var_10 = 'target'
    var_11 = 'annotation'
    var_12 = 'value'
    var_13 = {var_10: var_6, var_11: var_8, var_12: var_9}
    var_14 = module_2.AnnAssign(*var_2, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.AnnAssign'
    assert var_14.target == 'id'
    assert f'{type(var_14.annotation).__module__}.{type(var_14.annotation).__qualname__}' == 'ast.Name'
    assert var_14.value is None
    assert module_2.AnnAssign.value is None
    var_15 = module_0.const_type(var_9)
    assert var_15 == 'Any'
    var_16 = []
    var_17 = 'id'
    var_18 = {var_17: var_3}
    var_19 = module_2.Name(*var_16, **var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.Name'
    assert var_19.id == 'attr1'
    var_20 = module_2.Delete(*var_5, **var_13)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.Delete'
    assert var_20.target == 'id'
    assert f'{type(var_20.annotation).__module__}.{type(var_20.annotation).__qualname__}' == 'ast.Name'
    assert var_20.value is None
    var_21 = [var_14, var_20]
    var_22 = []
    var_23 = 'name'
    var_24 = 'bases'
    var_25 = 'body'
    var_26 = 'decorator_list'
    var_27 = {var_23: var_1, var_24: var_2, var_25: var_21, var_26: var_2}
    var_28 = module_2.ClassDef(*var_22, **var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'ast.ClassDef'
    assert var_28.name == 'TestClass'
    assert var_28.bases == []
    assert f'{type(var_28.body).__module__}.{type(var_28.body).__qualname__}' == 'builtins.list'
    assert len(var_28.body) == 2
    assert var_28.decorator_list == []
    var_29 = 'root'
    var_30 = 'root.TestClass'
    var_31 = var_28.bases
    var_32 = var_28.body
    var_0.class_api(var_29, var_30, var_31, var_32)

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
    var_1 = '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'test_module': 0, 'test_module.foo': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'test_module': 'Module docstring.', 'test_module.foo': 'Function docstring.'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.foo': 'test_module'}
    assert var_0.alias == {'test_module.x': '1'}
    var_4 = bool('test_module' in var_0.doc)
    assert var_4 is True
    var_5 = bool('test_module.foo' in var_0.doc)
    assert var_5 is True
    var_6 = bool('test_module.x' in var_0.alias)
    assert var_6 is True
    var_7 = var_0.docstring['test_module']

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
    var_1 = '\ndef decorator(func):\n    return func\n@decorator\ndef foo():\n    """Decorated function."""\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'test_module': 0, 'test_module.decorator': 0, 'test_module.foo': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.decorator': '### decorator()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| func | return |\n|:----:|:------:|\n| `Any` | `Any` |\n\n', 'test_module.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@decorator` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'test_module.foo': 'Decorated function.'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.decorator': 'test_module', 'test_module.foo': 'test_module'}
    var_4 = bool('test_module.foo' in var_0.doc)
    assert var_4 is True
    var_5 = bool('@decorator' in var_0.doc['test_module.foo'])
    assert var_5 is True
    var_6 = var_0.docstring['test_module.foo']

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
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'test_module': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module'}
    var_3 = var_0.load_docstring(var_1, var_2)
    with pytest.raises(AttributeError):
        var_4 = bool('os' in var_1.alias['test_module.os'])
    assert var_4 is True

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
    var_1 = "\n__all__ = ['public_func', 'PublicClass']\ndef public_func():\n    pass\ndef _private_func():\n    pass\nclass PublicClass:\n    pass\nclass _PrivateClass:\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.__post_init__()
    var_4 = var_0.parse(var_2, var_1)
    assert var_0.level == {'test_module': 0, 'test_module.public_func': 0, 'test_module._private_func': 0, 'test_module.PublicClass': 0, 'test_module._PrivateClass': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', 'test_module.PublicClass': '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module._PrivateClass': '### class _PrivateClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module': {'test_module.PublicClass', 'test_module.public_func'}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.public_func': 'test_module', 'test_module._private_func': 'test_module', 'test_module.PublicClass': 'test_module', 'test_module._PrivateClass': 'test_module'}
    assert var_0.alias == {'test_module.__all__': "['public_func', 'PublicClass']"}
    var_5 = bool('test_module.public_func' in var_0.imp['test_module'])
    assert var_5 is True
    var_6 = 'XV2%\r>Q'
    var_7 = var_0.load_docstring(var_6, var_3)
    var_0.visit_Constant(var_2)

@pytest.mark.xfail(strict=True)
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
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'test_module': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module'}
    var_3 = var_0.compile()
    assert var_3 == '\n'
    var_4 = bool('test_module.public_func' in var_0.imp['test_module'])
    var_5 = bool('test_module.PublicClass' in var_0.imp['test_module'])
    module_0.doctest(var_5)

@pytest.mark.xfail(strict=True)
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
    var_1 = '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': 1, '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': 1}
    assert var_0.doc == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | <code>str &#124; None</code> | `bool` |\n|   | `None` |   |   |\n\n'}
    assert var_0.docstring == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': 'Function with annotations.'}
    assert var_0.imp == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': {*()}}
    assert var_0.root == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n', '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': '\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n'}
    assert var_0.alias == {'\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.Optional': 'typing.Optional'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n`\n<a id="\nfrom typing import optional\ndef foo(x: int, y: optional[str] = none) -> bool:\n    """function with annotations-"""\n    return true\n"></a>\n\n### foo()\n\n*Full name:* `\nfrom typing import Optional\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo`\n<a id="\nfrom typing import optional\ndef foo(x: int, y: optional[str] = none) -> bool:\n    """function with annotations-"""\n    return true\n-foo"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | <code>str &#124; None</code> | `bool` |\n|   | `None` |   |   |\n\nFunction with annotations.\n'
    module_2.unparse(var_3)

def test_case_59():
    var_0 = []
    var_1 = {}
    var_2 = module_4.object(*var_0, **var_1)
    var_3 = 'nonexistent'
    var_4 = module_0._attr(var_2, var_3)
    assert var_4 is None
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_60():
    var_0 = 'root'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Resolver'
    assert var_2.root == 'root'
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
    var_3 = 1
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_2.Constant(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Constant'
    assert var_6.value == 1
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_7 = 0
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_2.Constant(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == 0
    var_11 = []
    var_12 = {}
    var_13 = module_2.Load(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Load'
    var_14 = [var_6, var_10, var_13]
    var_15 = {}
    var_16 = module_2.Subscript(*var_14, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Subscript'
    assert f'{type(var_16.value).__module__}.{type(var_16.value).__qualname__}' == 'ast.Constant'
    assert f'{type(var_16.slice).__module__}.{type(var_16.slice).__qualname__}' == 'ast.Constant'
    assert f'{type(var_16.ctx).__module__}.{type(var_16.ctx).__qualname__}' == 'ast.Load'
    var_17 = var_2.visit_Subscript(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Subscript'
    assert f'{type(var_17.value).__module__}.{type(var_17.value).__qualname__}' == 'ast.Constant'
    assert f'{type(var_17.slice).__module__}.{type(var_17.slice).__qualname__}' == 'ast.Constant'
    assert f'{type(var_17.ctx).__module__}.{type(var_17.ctx).__qualname__}' == 'ast.Load'
    var_18 = bool(var_17 == var_16)
    assert var_18 is True

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
    var_1 = None
    var_2 = var_0.load_docstring(var_1, var_1)
    var_3 = '\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n'
    var_4 = var_0.parse(var_3, var_3)
    assert var_0.level == {'\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': 1, '\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': 1}
    assert var_0.doc == {'\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `Optional[str]` | `bool` |\n|   | `None` |   |   |\n\n'}
    assert var_0.docstring == {'\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': 'Function with annotations.'}
    assert var_0.imp == {'\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': {*()}}
    assert var_0.root == {'\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n': '\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n', '\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo': '\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n'}
    assert var_0.alias == {'\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.Optioal': 'typing.Optioal'}
    var_5 = var_0.compile()
    assert var_5 == '## Module `\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n`\n<a id="\nfrom typing import optioal\ndef foo(x: int, y: optional[str] = none) -> bool:\n    """function with annotations-"""\n    return true\n"></a>\n\n### foo()\n\n*Full name:* `\nfrom typing import Optioal\ndef foo(x: int, y: Optional[str] = None) -> bool:\n    """Function with annotations."""\n    return True\n.foo`\n<a id="\nfrom typing import optioal\ndef foo(x: int, y: optional[str] = none) -> bool:\n    """function with annotations-"""\n    return true\n-foo"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `Optional[str]` | `bool` |\n|   | `None` |   |   |\n\nFunction with annotations.\n'
    var_6 = var_0.__repr__()
    assert var_6 == 'Parser(link=True, b_level=1, toc=False, level={\'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n\': 1, \'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n.foo\': 1}, doc={\'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n.foo\': \'### foo()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| x | y | return |\\n|:---:|:---:|:------:|\\n| `int` | `Optional[str]` | `bool` |\\n|   | `None` |   |   |\\n\\n\'}, docstring={\'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n.foo\': \'Function with annotations.\'}, imp={\'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n\': set()}, root={\'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n\': \'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n\', \'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n.foo\': \'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n\'}, alias={\'\\nfrom typing import Optioal\\ndef foo(x: int, y: Optional[str] = None) -> bool:\\n    """Function with annotations."""\\n    return True\\n.Optioal\': \'typing.Optioal\'}, const={})'
    var_7 = var_0.globals(var_5, var_3)
    var_8 = module_0.Parser(var_4, docstring=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is None
    assert var_8.b_level == 1
    assert var_8.toc is False
    assert var_8.level == {}
    assert var_8.doc == {}
    assert var_8.docstring is None
    assert var_8.imp == {}
    assert var_8.root == {}
    assert var_8.alias == {}
    assert var_8.const == {}
    var_0.load_docstring(var_4, var_7)

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
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_2.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == 'x'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = 1
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_2.Constant(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Constant'
    assert var_11.value == 1
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    var_12 = 'int'
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_6, var_15: var_11, var_16: var_12}
    var_18 = module_2.Assign(*var_13, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Assign'
    assert f'{type(var_18.targets).__module__}.{type(var_18.targets).__qualname__}' == 'builtins.list'
    assert len(var_18.targets) == 1
    assert f'{type(var_18.value).__module__}.{type(var_18.value).__qualname__}' == 'ast.Constant'
    assert var_18.type_comment == 'int'
    assert module_2.Assign.type_comment is None
    var_19 = 'root'
    var_20 = var_0.globals(var_19, var_18)
    assert var_0.alias == {'root.x': '1'}
    var_21 = var_0.alias['root.x']
    assert var_21 == '1'
    with pytest.raises(KeyError):
        var_22 = var_0.const['root.x']
    assert var_22 == 'int'

def test_case_63():
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
    var_1 = 'x'
    var_2 = 'int'
    var_3 = []
    var_4 = {}
    var_5 = module_2.Load(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Load'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_2.Name(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Name'
    assert var_8.id == 'int'
    assert f'{type(var_8.ctx).__module__}.{type(var_8.ctx).__qualname__}' == 'ast.Load'
    var_9 = [var_1, var_8]
    var_10 = {}
    var_11 = module_2.arg(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.arg'
    assert var_11.arg == 'x'
    assert f'{type(var_11.annotation).__module__}.{type(var_11.annotation).__qualname__}' == 'ast.Name'
    assert module_2.arg.annotation is None
    assert module_2.arg.type_comment is None
    assert module_2.arg.end_lineno is None
    assert module_2.arg.end_col_offset is None
    var_12 = '*'
    var_13 = None
    var_14 = [var_12, var_13]
    var_15 = {}
    var_16 = module_2.arg(*var_14, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.arg'
    assert var_16.arg == '*'
    assert var_16.annotation is None
    var_17 = 'y'
    var_18 = 'str'
    var_19 = []
    var_20 = {}
    var_21 = module_2.Load(*var_19, **var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'ast.Load'
    var_22 = [var_18, var_21]
    var_23 = {}
    var_24 = module_2.Name(*var_22, **var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'ast.Name'
    assert var_24.id == 'str'
    assert f'{type(var_24.ctx).__module__}.{type(var_24.ctx).__qualname__}' == 'ast.Load'
    var_25 = [var_17, var_24]
    var_26 = {}
    var_27 = module_2.arg(*var_25, **var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'ast.arg'
    assert var_27.arg == 'y'
    assert f'{type(var_27.annotation).__module__}.{type(var_27.annotation).__qualname__}' == 'ast.Name'
    var_28 = [var_11, var_16, var_27]
    var_29 = 'root'
    var_30 = False
    var_31 = var_0.func_ann(var_29, var_28, has_self=var_30, cls_method=var_30)
    var_32 = list(var_31)
    var_33 = bool(var_32 == ['int', '', 'str'])
    assert var_33 is True

def test_case_64():
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
    var_1 = 'type[Self]'
    var_2 = []
    var_3 = {}
    var_4 = module_2.Load(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Load'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_5 = [var_1, var_4]
    var_6 = {}
    var_7 = module_2.Name(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Name'
    assert var_7.id == 'type[Self]'
    assert f'{type(var_7.ctx).__module__}.{type(var_7.ctx).__qualname__}' == 'ast.Load'
    var_8 = {}
    var_9 = module_2.arg(*var_5, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.arg'
    assert var_9.arg == 'type[Self]'
    assert f'{type(var_9.annotation).__module__}.{type(var_9.annotation).__qualname__}' == 'ast.Load'
    assert module_2.arg.annotation is None
    assert module_2.arg.type_comment is None
    assert module_2.arg.end_lineno is None
    assert module_2.arg.end_col_offset is None
    var_10 = 'x'
    var_11 = 'int'
    var_12 = []
    var_13 = {}
    var_14 = module_2.Load(*var_12, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Load'
    var_15 = [var_11, var_14]
    var_16 = {}
    var_17 = module_2.Name(*var_15, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Name'
    assert var_17.id == 'int'
    assert f'{type(var_17.ctx).__module__}.{type(var_17.ctx).__qualname__}' == 'ast.Load'
    var_18 = [var_10, var_17]
    var_19 = {}
    var_20 = module_2.arg(*var_18, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.arg'
    assert var_20.arg == 'x'
    assert f'{type(var_20.annotation).__module__}.{type(var_20.annotation).__qualname__}' == 'ast.Name'
    var_21 = [var_9, var_20]
    var_22 = 'root'
    var_23 = True
    var_24 = var_0.func_ann(var_22, var_21, has_self=var_23, cls_method=var_23)
    var_25 = list(var_24)
    var_26 = bool(var_25 == ['type[Self]', 'int'])
    assert var_26 is True

def test_case_65():
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
    var_1 = 'cls'
    var_2 = []
    var_3 = {}
    var_4 = module_2.Load(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Load'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_5 = {}
    var_6 = module_2.Name(*var_2, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Name'
    var_7 = [var_1, var_6]
    var_8 = {}
    var_9 = module_2.arg(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.arg'
    assert var_9.arg == 'cls'
    assert f'{type(var_9.annotation).__module__}.{type(var_9.annotation).__qualname__}' == 'ast.Name'
    assert module_2.arg.annotation is None
    assert module_2.arg.type_comment is None
    assert module_2.arg.end_lineno is None
    assert module_2.arg.end_col_offset is None
    var_10 = 'x'
    var_11 = 'int'
    var_12 = []
    var_13 = {}
    var_14 = module_2.Load(*var_12, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Load'
    var_15 = [var_11, var_14]
    var_16 = {}
    var_17 = module_2.Name(*var_15, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Name'
    assert var_17.id == 'int'
    assert f'{type(var_17.ctx).__module__}.{type(var_17.ctx).__qualname__}' == 'ast.Load'
    var_18 = [var_10, var_17]
    var_19 = {}
    var_20 = module_2.arg(*var_18, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.arg'
    assert var_20.arg == 'x'
    assert f'{type(var_20.annotation).__module__}.{type(var_20.annotation).__qualname__}' == 'ast.Name'
    var_21 = [var_9, var_20]
    var_22 = 'root'
    var_23 = True
    var_24 = var_0.func_ann(var_22, var_21, has_self=var_23, cls_method=var_23)
    with pytest.raises(AttributeError):
        var_25 = list(var_24)

def test_case_66():
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
    var_1 = 'self'
    var_2 = 'Self'
    var_3 = []
    var_4 = {}
    var_5 = module_2.Load(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Load'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_2.Name(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Name'
    assert var_8.id == 'Self'
    assert f'{type(var_8.ctx).__module__}.{type(var_8.ctx).__qualname__}' == 'ast.Load'
    var_9 = [var_1, var_8]
    var_10 = {}
    var_11 = module_2.arg(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.arg'
    assert var_11.arg == 'self'
    assert f'{type(var_11.annotation).__module__}.{type(var_11.annotation).__qualname__}' == 'ast.Name'
    assert module_2.arg.annotation is None
    assert module_2.arg.type_comment is None
    assert module_2.arg.end_lineno is None
    assert module_2.arg.end_col_offset is None
    var_12 = 'x'
    var_13 = 'int'
    var_14 = []
    var_15 = {}
    var_16 = module_2.Load(*var_14, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Load'
    var_17 = var_0.__eq__(var_1)
    var_18 = [var_13, var_16]
    var_19 = {}
    var_20 = module_2.Name(*var_18, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.Name'
    assert var_20.id == 'int'
    assert f'{type(var_20.ctx).__module__}.{type(var_20.ctx).__qualname__}' == 'ast.Load'
    var_21 = [var_12, var_20]
    var_22 = {}
    var_23 = module_2.arg(*var_21, **var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'ast.arg'
    assert var_23.arg == 'x'
    assert f'{type(var_23.annotation).__module__}.{type(var_23.annotation).__qualname__}' == 'ast.Name'
    var_24 = [var_11, var_23]
    var_25 = 'root'
    var_26 = True
    var_27 = False
    var_28 = var_0.func_ann(var_25, var_24, has_self=var_26, cls_method=var_27)
    var_29 = list(var_28)
    var_30 = bool(var_29 == ['Self', 'int'])
    assert var_30 is True