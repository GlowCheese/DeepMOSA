# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.parser as module_0
import inspect as module_1
import ast as module_2
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
    var_1 = '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n': 0, '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n.public_func': 0, '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n._private_func': 0}
    assert var_0.doc == {'\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n': {'\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n.public_func'}}
    assert var_0.root == {'\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n': '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n', '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n.public_func': '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n', '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n._private_func': '\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n'}
    assert var_0.alias == {'\n__all__ = ["public_func"]\ndef public_func():\n    pass\ndef _private_func():\n    pass\n.__all__': "['public_func']"}

def test_case_2():
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

@pytest.mark.xfail(strict=True)
def test_case_4():
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
    var_1 = '\nimport os\nfrom sys import path\nx = 1\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nimport os\nfrom sys import path\nx = 1\n': 0}
    assert var_0.doc == {'\nimport os\nfrom sys import path\nx = 1\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nimport os\nfrom sys import path\nx = 1\n': {*()}}
    assert var_0.root == {'\nimport os\nfrom sys import path\nx = 1\n': '\nimport os\nfrom sys import path\nx = 1\n'}
    assert var_0.alias == {'\nimport os\nfrom sys import path\nx = 1\n.os': 'os', '\nimport os\nfrom sys import path\nx = 1\n.path': 'sys.path', '\nimport os\nfrom sys import path\nx = 1\n.x': '1'}

def test_case_6():
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
    var_0 = "}0\x0cQ{\x0bQCLV*\tCN'Q};"
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
def test_case_9():
    var_0 = 'Y])_WTuPzhH#2\x0bSh|'
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
    var_2 = None
    var_3 = '7Ris'
    var_4 = 'YgG8aYG(aYfB>7h;'
    var_5 = "Cu}ypw?'J`[w6dw"
    var_6 = {var_0: var_0, var_0: var_3, var_4: var_5}
    var_7 = module_0.Resolver(var_2, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Resolver'
    assert var_7.root is None
    assert var_7.alias == {'Y])_WTuPzhH#2\x0bSh|': '7Ris', 'YgG8aYG(aYfB>7h;': "Cu}ypw?'J`[w6dw"}
    assert var_7.self_ty == ''
    var_8 = ''
    var_9 = module_0.code(var_8)
    assert var_9 == ' '
    var_10 = '9\r'
    var_11 = module_0.esc_underscore(var_10)
    assert var_11 == '9\r'
    var_12 = module_1.getdoc(var_2)
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
    var_13 = 'iy>EGX\r"6nm*\x0cX'
    var_12.class_api(var_13, var_2, var_2, var_13)

@pytest.mark.xfail(strict=True)
def test_case_10():
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
    var_0.is_public(var_1)

def test_case_11():
    var_0 = 'D)9mO/Z&f'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'D)9mO/Z&f'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_1 = '\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n': 0, '\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n.foo': 0}
    assert var_0.doc == {'\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | return |\n|:---:|:------:|\n| `str` | `None` |\n\n'}
    assert var_0.imp == {'\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n': {*()}}
    assert var_0.root == {'\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n': '\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n', '\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n.foo': '\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n'}
    assert var_0.alias == {'\nx: int = 1\ndef foo(a: str) -> None:\n    pass\n.x': '1'}

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
    var_1 = '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': 0, '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': 0}
    assert var_0.doc == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | return |\n|:---:|:------:|\n| `r` | `None` |\n\n'}
    assert var_0.imp == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': {*()}}
    assert var_0.root == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n', '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'}
    assert var_0.alias == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.x': '1'}
    var_3 = var_0.load_docstring(var_1, var_2)

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
    var_1 = '\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n': 0, '\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n.TestClass': 0}
    assert var_0.doc == {'\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| A |\n| B |\n\n'}
    assert var_0.imp == {'\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n': {*()}}
    assert var_0.root == {'\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n': '\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n', '\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n.TestClass': '\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n'}
    assert var_0.alias == {'\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n.Enum': 'enum.Enum'}

def test_case_15():
    var_0 = 'Syvz*zcWW8rw.i'
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
    var_1 = ''
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'': 0}
    assert var_0.doc == {'': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'': {*()}}
    assert var_0.root == {'': ''}

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
    var_1 = None
    var_2 = var_0.globals(var_1, var_1)

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
    var_1 = "j\rK'b|j^"
    var_2 = module_0.code(var_1)
    assert var_2 == "<code>j\rK'b&#124;j^</code>"
    var_3 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_4 = var_0.parse(var_3, var_3)
    assert var_0.level == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 2}
    assert var_0.doc == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.docstring == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'}
    var_5 = var_0.compile()
    assert var_5 == '## Module `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n"></a>\n\n### class MyClass\n\n*Full name:* `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n-myclass"></a>\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\nMethod docstring.\n'
    var_6 = module_0.esc_underscore(var_3)
    assert var_6 == '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_7 = module_0.const_type(var_4)
    assert var_7 == 'Any'
    var_8 = module_2.ImportFrom()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.ImportFrom'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_9 = module_0.Parser(doc=var_4, alias=var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc is False
    assert var_9.level == {}
    assert var_9.doc is None
    assert var_9.docstring == {}
    assert var_9.imp == {}
    assert var_9.root == {}
    assert var_9.alias == '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    assert var_9.const == {}
    var_10 = var_9.imports(var_4, var_8)

def test_case_19():
    var_0 = 'sKqp'
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
    assert var_1.level == {'sKqp': 0}
    assert var_1.doc == {'sKqp': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.imp == {'sKqp': {*()}}
    assert var_1.root == {'sKqp': 'sKqp'}

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
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 2}
    assert var_0.doc == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.docstring == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'}

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_2.Constant()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'ast.Constant'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.Constant.kind is None
    assert f'{type(module_2.Constant.n).__module__}.{type(module_2.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Constant.s).__module__}.{type(module_2.Constant.s).__qualname__}' == 'builtins.property'
    module_0.const_type(var_0)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'

def test_case_23():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = True
    var_3 = 'lK-[82^0,('
    var_4 = 'F\nc0|2\tyjK3&,|.I'
    var_5 = '#1dn\t`$\x0canM'
    var_6 = {var_4, var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.Parser(toc=var_2, imp=var_7, root=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is True
    assert var_8.b_level == 1
    assert var_8.toc is True
    assert var_8.level == {}
    assert var_8.doc == {}
    assert var_8.docstring == {}
    assert var_8.imp == {'lK-[82^0,(': {'F\nc0|2\tyjK3&,|.I', '#1dn\t`$\x0canM'}}
    assert var_8.root is True
    assert var_8.alias == {}
    assert var_8.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_9 = module_0._defaults(var_1)
    var_10 = list(var_9)

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
    var_1 = '\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n': 0, '\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n.public_func': 0, '\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n._private_fun': 0}
    assert var_0.doc == {'\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n._private_fun': '### \\_private\\_fun()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n': {'\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n.public_unc'}}
    assert var_0.root == {'\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n': '\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n', '\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n.public_func': '\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n', '\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n._private_fun': '\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n'}
    assert var_0.alias == {'\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n.__all__': "['public_unc']"}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n`\n<a id="\n__all__ = ["public_unc"]\ndef public_func():\n   pass\ndef _private_fun():\n   pass\n"></a>\n'

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
    var_1 = 'Syvz*zcW/W8rw.i'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Syvz*zcW/W8rw.i': 1}
    assert var_0.doc == {'Syvz*zcW/W8rw.i': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Syvz*zcW/W8rw.i': {*()}}
    assert var_0.root == {'Syvz*zcW/W8rw.i': 'Syvz*zcW/W8rw.i'}
    var_3 = var_0.compile()
    assert var_3 == '\n'

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
    var_1 = 'Syvz*zcW/W8rw.i'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Syvz*zcW/W8rw.i': 1}
    assert var_0.doc == {'Syvz*zcW/W8rw.i': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Syvz*zcW/W8rw.i': {*()}}
    assert var_0.root == {'Syvz*zcW/W8rw.i': 'Syvz*zcW/W8rw.i'}
    var_3 = var_0.compile()
    assert var_3 == '\n'
    var_4 = '@6\t{f\r'
    var_5 = var_0.load_docstring(var_4, var_2)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = '{nq'
    var_1 = None
    var_2 = module_0.Parser(doc=var_1, docstring=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
    assert var_2.level == {}
    assert var_2.doc is None
    assert var_2.docstring == '{nq'
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
    var_4 = ';Shbcet d 03Q-q3[3'
    var_5 = module_0.esc_underscore(var_4)
    assert var_5 == ';Shbcet d 03Q-q3[3'
    var_6 = [var_1]
    var_7 = module_2.Attribute(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Attribute'
    assert var_7.value is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_2.resolve(var_5, var_7)

def test_case_28():
    var_0 = None
    var_1 = ';Shbcet d 03Q-q3[3'
    var_2 = module_0.esc_underscore(var_1)
    assert var_2 == ';Shbcet d 03Q-q3[3'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_3 = [var_0]
    var_4 = module_2.List(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.List'
    assert var_4.elts is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_5 = module_0.const_type(var_4)
    assert var_5 == 'list'

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_2.Dict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'ast.Dict'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    module_0.const_type(var_0)

def test_case_30():
    var_0 = 'public.__magic__._private'
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

def test_case_31():
    var_0 = 'M\t>aF\r'
    var_1 = module_0.esc_underscore(var_0)
    assert var_1 == 'M\t>aF\r'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_2 = [var_0]
    var_3 = module_2.List(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.List'
    assert var_3.elts == 'M\t>aF\r'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_4 = module_0.const_type(var_3)
    assert var_4 == 'list'

def test_case_32():
    var_0 = 'au]%overQ\x0b.=2'
    var_1 = [var_0]
    var_2 = module_2.Call(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Call'
    assert var_2.func == 'au]%overQ\x0b.=2'
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
    var_1 = 'Bjrz\rpb\r!w'
    var_2 = var_0.globals(var_1, var_1)
    var_3 = module_1.getdoc(var_0)
    assert var_3 == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
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
    var_4 = module_0.const_type(var_3)
    assert var_4 == 'Any'
    var_5 = module_0.doctest(var_3)
    assert var_5 == 'AST parser.\n\nUsage:\n```python\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n```\n\nOr create with parameters:\n```python\n>>> p = Parser.new(link=True, level=1)\n```'
    var_6 = var_0.compile()
    assert var_6 == '\n'

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
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 2}
    assert var_0.doc == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.docstring == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n"></a>\n\n### class MyClass\n\n*Full name:* `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n-myclass"></a>\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\nMethod docstring.\n'
    var_4 = module_0.esc_underscore(var_1)
    assert var_4 == '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_5 = module_0.const_type(var_2)
    assert var_5 == 'Any'
    var_6 = module_2.ImportFrom()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.ImportFrom'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_7 = module_0.Parser(doc=var_2, alias=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is True
    assert var_7.b_level == 1
    assert var_7.toc is False
    assert var_7.level == {}
    assert var_7.doc is None
    assert var_7.docstring == {}
    assert var_7.imp == {}
    assert var_7.root == {}
    assert var_7.alias == '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    assert var_7.const == {}
    var_8 = var_7.imports(var_2, var_6)

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
    var_1 = '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n': 2, '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n.foo': 2}
    assert var_0.doc == {'\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n': 'Module docstring.', '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n.foo': 'Function docstring.'}
    assert var_0.imp == {'\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n': {*()}}
    assert var_0.root == {'\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n': '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n', '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n.foo': '\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n'}
    assert var_0.alias == {'\n"""Module docstring."""\nx = 1\ndef foo():\n    """Function docstring."""\n    pass\n.x': '1'}

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
    var_1 = '\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n': 1, '\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n.foo': 1}
    assert var_0.doc == {'\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@decorator` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n.foo': 'Function docstring.'}
    assert var_0.imp == {'\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n': {*()}}
    assert var_0.root == {'\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n': '\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n', '\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n.foo': '\n@decorator\ndef foo():\n    """Function docstring."""\n    pass\n'}

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
    var_1 = '\nif True:\n    x = 1\nelse:\n    y = 2\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nif True:\n    x = 1\nelse:\n    y = 2\n': 0}
    assert var_0.doc == {'\nif True:\n    x = 1\nelse:\n    y = 2\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nif True:\n    x = 1\nelse:\n    y = 2\n': {*()}}
    assert var_0.root == {'\nif True:\n    x = 1\nelse:\n    y = 2\n': '\nif True:\n    x = 1\nelse:\n    y = 2\n'}
    assert var_0.alias == {'\nif True:\n    x = 1\nelse:\n    y = 2\n.x': '1', '\nif True:\n    x = 1\nelse:\n    y = 2\n.y': '2'}

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
    var_1 = '\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n': 0}
    assert var_0.doc == {'\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n': {*()}}
    assert var_0.root == {'\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n': '\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n'}
    assert var_0.alias == {'\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n.x': '1', '\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n.y': '2', '\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n.z': '3', '\ntry:\n    x = 1\nexcept Exception:\n    y = 2\nelse:\n    z = 3\nfinally:\n    w = 4\n.w': '4'}

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
    var_1 = '\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n': 0, '\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n.foo': 0}
    assert var_0.doc == {'\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | return |\n|:---:|:------:|\n| `xtr` | `None` |\n\n'}
    assert var_0.imp == {'\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n': {*()}}
    assert var_0.root == {'\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n': '\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n', '\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n.foo': '\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n'}
    assert var_0.alias == {'\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n.x': '1'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n`\n<a id="\nx: int = 1\ndef foo(a: xtr) -> none:\n    pass\n"></a>\n\n### foo()\n\n*Full name:* `\nx: int = 1\ndef foo(a: xtr) -> None:\n    pass\n.foo`\n<a id="\nx: int = 1\ndef foo(a: xtr) -> none:\n    pass\n-foo"></a>\n\n| a | return |\n|:---:|:------:|\n| `xtr` | `None` |\n'

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
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 2}
    assert var_0.doc == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.docstring == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n"></a>\n\n### class MyClass\n\n*Full name:* `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n-myclass"></a>\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\nMethod docstring.\n'

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
    var_1 = '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n'
    var_2 = 'yTPt'
    var_3 = []
    var_4 = 'W*q^zC!W;%>O8\x0co]@,P'
    var_5 = None
    var_6 = {var_4: var_5, var_4: var_5}
    var_7 = module_2.stmt(**var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.stmt'
    assert var_7.W*q^zC!W;%>O8o]@,P is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.stmt.end_lineno is None
    assert module_2.stmt.end_col_offset is None
    var_8 = [var_7, var_7, var_7]
    var_9 = var_0.class_api(var_0, var_2, var_3, var_8)
    var_10 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n': 0, '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n.pulic_func': 0, '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n._privte_fun': 0}
    assert var_0.doc == {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n.pulic_func': '### pulic_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n._privte_fun': '### \\_privte\\_fun()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n': {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n.public_unc'}}
    assert var_0.root == {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n': '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n', '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n.pulic_func': '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n', '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n._privte_fun': '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n'}
    assert var_0.alias == {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n.__all__': "['public_unc']"}
    var_11 = module_0.is_public_family(var_2)
    assert var_11 is True
    var_12 = None
    var_13 = var_0.load_docstring(var_1, var_11)
    var_14 = module_0.const_type(var_5)
    assert var_14 == 'Any'
    var_15 = module_2.Call()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Call'
    var_16 = module_2.ImportFrom()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.ImportFrom'
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_17 = True
    var_18 = module_0.Parser(toc=var_17, root=var_12)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'apimd.parser.Parser'
    assert var_18.link is True
    assert var_18.b_level == 1
    assert var_18.toc is True
    assert var_18.level == {}
    assert var_18.doc == {}
    assert var_18.docstring == {}
    assert var_18.imp == {}
    assert var_18.root is None
    assert var_18.alias == {}
    assert var_18.const == {}
    var_19 = var_18.compile()
    assert var_19 == '**Table of contents:**\n\n\n'
    var_0.load_docstring(var_10, var_5)

def test_case_42():
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
    var_1 = '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n'
    var_2 = 'yTPt'
    var_3 = None
    var_4 = {var_2: var_3, var_2: var_3}
    var_5 = module_2.stmt(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.stmt'
    assert var_5.yTPt is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.stmt.end_lineno is None
    assert module_2.stmt.end_col_offset is None
    var_6 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n': 0, '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n.pulic_func': 0, '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n._privte_fun': 0}
    assert var_0.doc == {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n.pulic_func': '### pulic_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n._privte_fun': '### \\_privte\\_fun()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n': {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n.public_unc'}}
    assert var_0.root == {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n': '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n', '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n.pulic_func': '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n', '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n._privte_fun': '\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n'}
    assert var_0.alias == {'\n__all__ = ["public_unc"]\ndef pulic_func():\n   pass\ndef _privte_fun():\n   pass\n.__all__': "['public_unc']"}
    var_7 = var_0.__repr__()
    assert var_7 == 'Parser(link=True, b_level=1, toc=False, level={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': 0, \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.pulic_func\': 0, \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n._privte_fun\': 0}, doc={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.pulic_func\': \'### pulic_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n._privte_fun\': \'### \\\\_privte\\\\_fun()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': {\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.public_unc\'}}, root={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.pulic_func\': \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n._privte_fun\': \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\'}, alias={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.__all__\': "[\'public_unc\']"}, const={})'
    var_8 = var_0.load_docstring(var_1, var_7)
    var_9 = module_0.const_type(var_3)
    assert var_9 == 'Any'
    var_10 = module_2.Call()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Call'
    var_11 = [var_7, var_3, var_7]
    var_12 = module_2.ImportFrom(*var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.ImportFrom'
    assert var_12.module == 'Parser(link=True, b_level=1, toc=False, level={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': 0, \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.pulic_func\': 0, \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n._privte_fun\': 0}, doc={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.pulic_func\': \'### pulic_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n._privte_fun\': \'### \\\\_privte\\\\_fun()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': {\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.public_unc\'}}, root={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.pulic_func\': \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n._privte_fun\': \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\'}, alias={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.__all__\': "[\'public_unc\']"}, const={})'
    assert var_12.names is None
    assert var_12.level == 'Parser(link=True, b_level=1, toc=False, level={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': 0, \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.pulic_func\': 0, \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n._privte_fun\': 0}, doc={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.pulic_func\': \'### pulic_func()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n._privte_fun\': \'### \\\\_privte\\\\_fun()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': {\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.public_unc\'}}, root={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\': \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.pulic_func\': \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\', \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n._privte_fun\': \'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n\'}, alias={\'\\n__all__ = ["public_unc"]\\ndef pulic_func():\\n   pass\\ndef _privte_fun():\\n   pass\\n.__all__\': "[\'public_unc\']"}, const={})'
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_0.imports(var_1, var_12)

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
    var_1 = '\nCONSTANT = 42\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nCONSTANT = 42\n': 0}
    assert var_0.doc == {'\nCONSTANT = 42\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nCONSTANT = 42\n': {*()}}
    assert var_0.root == {'\nCONSTANT = 42\n': '\nCONSTANT = 42\n', '\nCONSTANT = 42\n.CONSTANT': '\nCONSTANT = 42\n'}
    assert var_0.alias == {'\nCONSTANT = 42\n.CONSTANT': '42'}
    assert var_0.const == {'\nCONSTANT = 42\n.CONSTANT': 'int'}

def test_case_45():
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

def test_case_46():
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
    var_1 = '\nCONSTANNT = 42\nV'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nCONSTANNT = 42\nV': 0}
    assert var_0.doc == {'\nCONSTANNT = 42\nV': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nCONSTANNT = 42\nV': {*()}}
    assert var_0.root == {'\nCONSTANNT = 42\nV': '\nCONSTANNT = 42\nV', '\nCONSTANNT = 42\nV.CONSTANNT': '\nCONSTANNT = 42\nV'}
    assert var_0.alias == {'\nCONSTANNT = 42\nV.CONSTANNT': '42'}
    assert var_0.const == {'\nCONSTANNT = 42\nV.CONSTANNT': 'int'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nCONSTANNT = 42\nV`\n<a id="\nconstannt = 42\nv"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANNT` | `int` |\n'

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
    var_1 = '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'
    var_2 = None
    var_3 = var_0.globals(var_1, var_2)
    var_4 = module_1.getdoc(var_1)
    assert var_4 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
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
    var_5 = var_0.parse(var_4, var_1)
    assert var_0.level == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": 7, "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..foo": 7}
    assert var_0.doc == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": '## Module `{}`\n<a id="{}"></a>\n\n', "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | return |\n|:---:|:------:|\n| `r` | `None` |\n\n'}
    assert var_0.imp == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": {*()}}
    assert var_0.root == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..foo": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."}
    assert var_0.alias == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..x": '1'}
    var_6 = var_0.__eq__(var_1)
    var_7 = var_0.compile()
    assert var_7 == '\n'
    var_8 = module_0.const_type(var_5)
    assert var_8 == 'Any'
    var_9 = module_3.dataclass(kw_only=var_6)
    assert f'{type(module_3.MISSING).__module__}.{type(module_3.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_3.KW_ONLY).__module__}.{type(module_3.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_10 = module_2.ImportFrom()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.ImportFrom'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.ImportFrom.module is None
    assert module_2.ImportFrom.level is None
    var_11 = None
    var_12 = var_0.compile()
    assert var_12 == '\n'
    var_13 = var_0.imports(var_11, var_10)

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
    var_1 = '\nCONSTANT = 42\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nCONSTANT = 42\n': 0}
    assert var_0.doc == {'\nCONSTANT = 42\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nCONSTANT = 42\n': {*()}}
    assert var_0.root == {'\nCONSTANT = 42\n': '\nCONSTANT = 42\n', '\nCONSTANT = 42\n.CONSTANT': '\nCONSTANT = 42\n'}
    assert var_0.alias == {'\nCONSTANT = 42\n.CONSTANT': '42'}
    assert var_0.const == {'\nCONSTANT = 42\n.CONSTANT': 'int'}
    var_3 = var_0.parse(var_1, var_1)
    var_4 = 'x)V[RV_$~1M1YCZiv9'
    var_5 = var_0.__repr__()
    var_6 = var_0.compile()
    assert var_6 == '## Module `\nCONSTANT = 42\n`\n<a id="\nconstant = 42\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n'
    var_7 = var_0.__post_init__()
    var_8 = module_0.const_type(var_7)
    assert var_8 == 'Any'
    var_9 = False
    module_3.dataclass(var_4, init=var_3, frozen=var_9, match_args=var_7)

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
    var_1 = 'test_module'
    var_2 = None
    var_3 = [var_2, var_2]
    var_4 = module_2.Subscript(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Subscript'
    assert var_4.value is None
    assert var_4.slice is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_0.resolve(var_1, var_4)

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
    var_1 = None
    var_2 = '?t,'
    var_3 = [var_1, var_1]
    var_4 = module_2.AnnAssign(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.AnnAssign'
    assert var_4.target is None
    assert var_4.annotation is None
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.AnnAssign.value is None
    var_5 = var_0.globals(var_2, var_4)
    var_6 = var_0.compile()
    assert var_6 == '\n'
    var_7 = var_0.__repr__()
    var_8 = 'HH\x0c5,DjnH>wMPKA22'
    var_0.parse(var_7, var_8)

@pytest.mark.xfail(strict=True)
def test_case_52():
    var_0 = 'unknown'
    var_1 = module_2.Name()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'ast.Name'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    var_2 = [var_1]
    var_3 = module_2.Call(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Call'
    assert f'{type(var_3.func).__module__}.{type(var_3.func).__qualname__}' == 'ast.Name'
    module_0.const_type(var_3)
    assert var_4 == 'Any'

@pytest.mark.xfail(strict=True)
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
    var_1 = '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n'
    var_2 = None
    var_3 = var_0.globals(var_1, var_2)
    var_4 = var_0.compile()
    assert var_4 == '\n'
    var_5 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n': 0, '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n.foo': 0}
    assert var_0.doc == {'\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | return |\n|:---:|:------:|\n| `r` | `None` |\n\n'}
    assert var_0.imp == {'\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n': {*()}}
    assert var_0.root == {'\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n': '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n', '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n'}
    var_6 = module_0.const_type(var_2)
    assert var_6 == 'Any'
    var_7 = '\n__all__ = ["pubic_unc"]\ndef public_func():\n   pass\ndef _priate_fun():\n  ass\n'
    var_8 = var_0.parse(var_4, var_7)
    assert var_0.level == {'\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n': 0, '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n.foo': 0, '\n': 0, '\n.public_func': 0, '\n._priate_fun': 0}
    assert var_0.doc == {'\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | return |\n|:---:|:------:|\n| `r` | `None` |\n\n', '\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '\n._priate_fun': '### \\_priate\\_fun()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n': {*()}, '\n': {'\n.pubic_unc'}}
    assert var_0.root == {'\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n': '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n', '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '\nx: n# = 1\ndef foo(a: r) -> None:\n    pass\n', '\n': '\n', '\n.public_func': '\n', '\n._priate_fun': '\n'}
    assert var_0.alias == {'\n.__all__': "['pubic_unc']"}
    var_9 = var_0.load_docstring(var_1, var_3)
    var_10 = var_0.__post_init__()
    var_0.parse(var_5, var_2)

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
    var_1 = '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'
    var_2 = None
    var_3 = var_0.globals(var_1, var_2)
    var_4 = {}
    var_5 = module_0.Resolver(var_1, var_4, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Resolver'
    assert var_5.root == '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'
    assert var_5.alias == {}
    assert var_5.self_ty == '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'
    var_6 = var_0.compile()
    assert var_6 == '\n'
    var_7 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': 0, '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': 0}
    assert var_0.doc == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | return |\n|:---:|:------:|\n| `r` | `None` |\n\n'}
    assert var_0.imp == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': {*()}}
    assert var_0.root == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n', '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'}
    assert var_0.alias == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.x': '1'}
    var_8 = module_0.const_type(var_2)
    assert var_8 == 'Any'
    var_9 = [var_6, var_8, var_5]
    var_10 = module_2.arguments(*var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.arguments'
    assert var_10.posonlyargs == '\n'
    assert var_10.args == 'Any'
    assert f'{type(var_10.vararg).__module__}.{type(var_10.vararg).__qualname__}' == 'apimd.parser.Resolver'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.arguments.vararg is None
    assert module_2.arguments.kwarg is None
    var_0.func_api(var_3, var_2, var_10, var_10, has_self=var_7, cls_method=var_7)

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
    var_1 = '\nimport os\nfrom syB import path\nP = 1\n'
    var_2 = None
    var_3 = module_0.const_type(var_2)
    assert var_3 == 'Any'
    var_4 = module_0.is_public_family(var_1)
    assert var_4 is True
    var_5 = var_0.parse(var_3, var_1)
    assert var_0.level == {'Any': 0}
    assert var_0.doc == {'Any': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Any': {*()}}
    assert var_0.root == {'Any': 'Any', 'Any.P': 'Any'}
    assert var_0.alias == {'Any.os': 'os', 'Any.path': 'syB.path', 'Any.P': '1'}
    assert var_0.const == {'Any.P': 'int'}
    var_6 = 'hjS\th=hfj~xs'
    var_7 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Any': 0, '\nimport os\nfrom syB import path\nP = 1\n': 0}
    assert var_0.doc == {'Any': '## Module `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom syB import path\nP = 1\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Any': {*()}, '\nimport os\nfrom syB import path\nP = 1\n': {*()}}
    assert var_0.root == {'Any': 'Any', 'Any.P': 'Any', '\nimport os\nfrom syB import path\nP = 1\n': '\nimport os\nfrom syB import path\nP = 1\n', '\nimport os\nfrom syB import path\nP = 1\n.P': '\nimport os\nfrom syB import path\nP = 1\n'}
    assert var_0.alias == {'Any.os': 'os', 'Any.path': 'syB.path', 'Any.P': '1', '\nimport os\nfrom syB import path\nP = 1\n.os': 'os', '\nimport os\nfrom syB import path\nP = 1\n.path': 'syB.path', '\nimport os\nfrom syB import path\nP = 1\n.P': '1'}
    assert var_0.const == {'Any.P': 'int', '\nimport os\nfrom syB import path\nP = 1\n.P': 'int'}
    var_8 = var_0.load_docstring(var_6, var_2)
    var_9 = var_0.load_docstring(var_3, var_2)
    var_10 = var_0.compile()
    assert var_10 == '## Module `\nimport os\nfrom syB import path\nP = 1\n`\n<a id="\nimport os\nfrom syb import path\np = 1\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `P` | `int` |\n\n## Module `Any`\n<a id="any"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `P` | `int` |\n'
    var_0.imports(var_5, var_5)

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
    var_1 = '\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n': 0, '\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n.TeslClass': 0}
    assert var_0.doc == {'\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n.TeslClass': '### class TeslClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `num.Enum` |\n\n| Members | Type |\n|:-------:|:----:|\n| `A` | `int` |\n| `B` | `int` |\n\n'}
    assert var_0.imp == {'\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n': {*()}}
    assert var_0.root == {'\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n': '\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n', '\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n.TeslClass': '\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n'}
    assert var_0.alias == {'\nfrom \x0cnum import Enum\nclass TeslClass(Enum):\n    A = 1\n    B = 2\n.Enum': 'num.Enum'}

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
    var_1 = ''
    var_2 = '\nfrom enum import Enum\nclass TestClass(Enum):\n    A = 1\n    B = 2\n'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'': 0, 'TestClass': 0}
    assert var_0.doc == {'': '## Module `{}`\n<a id="{}"></a>\n\n', 'TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `Enum` |\n\n| Members | Type |\n|:-------:|:----:|\n| `A` | `int` |\n| `B` | `int` |\n\n'}
    assert var_0.imp == {'': {*()}}
    assert var_0.root == {'': '', 'TestClass': ''}
    assert var_0.alias == {'Enum': 'enum.Enum'}

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
    var_1 = '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'
    var_2 = None
    var_3 = var_0.globals(var_1, var_2)
    var_4 = var_0.compile()
    assert var_4 == '\n'
    var_5 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': 0, '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': 0}
    assert var_0.doc == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | return |\n|:---:|:------:|\n| `r` | `None` |\n\n'}
    assert var_0.imp == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': {*()}}
    assert var_0.root == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n', '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'}
    assert var_0.alias == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.x': '1'}
    var_6 = module_0.const_type(var_2)
    assert var_6 == 'Any'
    var_7 = '\n__all__ =_["pubic_unc"]\ndef public_func():\n   pass\ndef _priate_fun():\n  ass\n'
    var_8 = module_0.const_type(var_3)
    assert var_8 == 'Any'
    var_9 = var_0.parse(var_4, var_7)
    assert var_0.level == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': 0, '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': 0, '\n': 0, '\n.public_func': 0, '\n._priate_fun': 0}
    assert var_0.doc == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | return |\n|:---:|:------:|\n| `r` | `None` |\n\n', '\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '\n._priate_fun': '### \\_priate\\_fun()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': {*()}, '\n': {*()}}
    assert var_0.root == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n': '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n', '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo': '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n', '\n': '\n', '\n.public_func': '\n', '\n._priate_fun': '\n'}
    assert var_0.alias == {'\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.x': '1', '\n.__all__': "_['pubic_unc']"}
    var_10 = var_0.load_docstring(var_1, var_3)
    var_11 = var_0.compile()
    assert var_11 == '## Module `\n`\n<a id="\n"></a>\n\n### public_func()\n\n*Full name:* `\n.public_func`\n<a id="\n-public_func"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n## Module `\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n`\n<a id="\nx: nt = 1\ndef foo(a: r) -> none:\n    pass\n"></a>\n\n### foo()\n\n*Full name:* `\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n.foo`\n<a id="\nx: nt = 1\ndef foo(a: r) -> none:\n    pass\n-foo"></a>\n\n| a | return |\n|:---:|:------:|\n| `r` | `None` |\n'
    var_12 = '1lq\x0c@Wwb}b,<h\\5*'
    var_13 = "&i{LKJ`!'oQn8zyUp"
    var_0.parse(var_13, var_12)

def test_case_59():
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

@pytest.mark.xfail(strict=True)
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
    var_1 = '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'
    var_2 = None
    var_3 = var_0.globals(var_1, var_2)
    var_4 = {}
    var_5 = module_0.Resolver(var_1, var_4, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Resolver'
    assert var_5.root == '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'
    assert var_5.alias == {}
    assert var_5.self_ty == '\nx: nt = 1\ndef foo(a: r) -> None:\n    pass\n'
    var_6 = module_2.AnnAssign()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.AnnAssign'
    assert module_2.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_2.PyCF_ONLY_AST == 1024
    assert module_2.PyCF_TYPE_COMMENTS == 4096
    assert module_2.AnnAssign.value is None
    var_7 = 'm='
    var_8 = "'#R_O~"
    var_9 = {var_1: var_3, var_1: var_2, var_7: var_3, var_8: var_2}
    var_10 = module_2.Assign(**var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Assign'
    assert var_10.
x: nt = 1
def foo(a: r) -> None:
    pass
 is None
    assert var_10.m= is None
    assert var_10.'#R_O~ is None
    assert module_2.Assign.type_comment is None
    var_11 = [var_6, var_10, var_10]
    var_0.class_api(var_2, var_2, var_4, var_11)