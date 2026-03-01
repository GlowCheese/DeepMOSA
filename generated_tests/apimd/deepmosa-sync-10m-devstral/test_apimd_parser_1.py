# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import apimd.parser as module_0
import ast as module_1
import dataclasses as module_2
import inspect as module_3

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
    var_1 = 'class A:\n    x = 1\n    del x[0]'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class A:\n    x = 1\n    del x[0]': 0, 'class A:\n    x = 1\n    del x[0].A': 0}
    assert var_0.doc == {'class A:\n    x = 1\n    del x[0]': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    x = 1\n    del x[0].A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n'}
    assert var_0.imp == {'class A:\n    x = 1\n    del x[0]': {*()}}
    assert var_0.root == {'class A:\n    x = 1\n    del x[0]': 'class A:\n    x = 1\n    del x[0]', 'class A:\n    x = 1\n    del x[0].A': 'class A:\n    x = 1\n    del x[0]'}

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
    var_1 = "\nfrom os import path\nimport sys\nx = path.join('a', 'b')\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\nfrom os import path\nimport sys\nx = path.join('a', 'b')\n": 1}
    assert var_0.doc == {"\nfrom os import path\nimport sys\nx = path.join('a', 'b')\n": '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {"\nfrom os import path\nimport sys\nx = path.join('a', 'b')\n": {*()}}
    assert var_0.root == {"\nfrom os import path\nimport sys\nx = path.join('a', 'b')\n": "\nfrom os import path\nimport sys\nx = path.join('a', 'b')\n"}
    assert var_0.alias == {"\nfrom os import path\nimport sys\nx = path.join('a', 'b')\n.path": 'os.path', "\nfrom os import path\nimport sys\nx = path.join('a', 'b')\n.sys": 'sys', "\nfrom os import path\nimport sys\nx = path.join('a', 'b')\n.x": "path.join('a', 'b')"}
    var_3 = var_0.compile()
    assert var_3 == '\n'

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

def test_case_9():
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
    var_1 = 'class A:\n    x = 1\n    del x[0]'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class A:\n    x = 1\n    del x[0]': 0, 'class A:\n    x = 1\n    del x[0].A': 0}
    assert var_0.doc == {'class A:\n    x = 1\n    del x[0]': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    x = 1\n    del x[0].A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n'}
    assert var_0.imp == {'class A:\n    x = 1\n    del x[0]': {*()}}
    assert var_0.root == {'class A:\n    x = 1\n    del x[0]': 'class A:\n    x = 1\n    del x[0]', 'class A:\n    x = 1\n    del x[0].A': 'class A:\n    x = 1\n    del x[0]'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `class A:\n    x = 1\n    del x[0]`\n<a id="class a:\n    x = 1\n    del x[0]"></a>\n\n### class A\n\n*Full name:* `class A:\n    x = 1\n    del x[0].A`\n<a id="class a:\n    x = 1\n    del x[0]-a"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n'

def test_case_11():
    var_0 = '*-'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '*-'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_1 = 'x: int = 1'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'x: int = 1': 0}
    assert var_0.doc == {'x: int = 1': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'x: int = 1': {*()}}
    assert var_0.root == {'x: int = 1': 'x: int = 1'}
    assert var_0.alias == {'x: int = 1.x': '1'}

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
    var_1 = "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.foo": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.bar": 0, "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.baz": 0}
    assert var_0.doc == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.foo": '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.bar": '### bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.baz": '### baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n": {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.foo", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.bar"}}
    assert var_0.root == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.foo": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.bar": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n", "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.baz": "\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n"}
    assert var_0.alias == {"\n__all__ = ['foo', 'bar']\ndef foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n.__all__": "['foo', 'bar']"}

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
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 2}
    assert var_0.doc == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.docstring == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'}
    var_3 = var_0.load_docstring(var_1, var_2)

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
    var_1 = '\n@decorator\ndef foo():\n    pass\n'
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'test_module': 0, 'test_module.foo': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@decorator` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.foo': 'test_module'}
    var_4 = bool('@decorator' in var_0.doc['test_module.foo'])
    assert var_4 is True

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
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'BaseClass'
    var_4 = 'eval'
    var_5 = module_1.parse(var_3, mode=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Expression'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'ast.Name'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = var_5.body
    var_7 = [var_6]
    var_8 = []
    var_0.class_api(var_1, var_2, var_7, var_8)

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
    var_1 = 'x = 1'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'x = 1': 0}
    assert var_0.doc == {'x = 1': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'x = 1': {*()}}
    assert var_0.root == {'x = 1': 'x = 1'}
    assert var_0.alias == {'x = 1.x': '1'}

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
    var_1 = 'class A:\n    x = 1\n    del x[0]'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class A:\n    x = 1\n    del x[0]': 0, 'class A:\n    x = 1\n    del x[0].A': 0}
    assert var_0.doc == {'class A:\n    x = 1\n    del x[0]': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    x = 1\n    del x[0].A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n'}
    assert var_0.imp == {'class A:\n    x = 1\n    del x[0]': {*()}}
    assert var_0.root == {'class A:\n    x = 1\n    del x[0]': 'class A:\n    x = 1\n    del x[0]', 'class A:\n    x = 1\n    del x[0].A': 'class A:\n    x = 1\n    del x[0]'}
    var_3 = var_0.globals(var_1, var_2)

def test_case_19():
    var_0 = 'cTaR|U;:w:FlDp9}42'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'cTaR|U;:w:FlDp9}42'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    assert var_2 == '<code>cTaR&#124;U;:w:FlDp9}42</code>'
    var_3 = module_0.is_magic(var_0)
    assert var_3 is False

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
    var_1 = []
    var_2 = var_0.class_api(var_1, var_1, var_1, var_1)

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
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 2}
    assert var_0.doc == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.docstring == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'}

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_1.Constant()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'ast.Constant'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    module_0.const_type(var_0)

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
    var_1 = 'class A:\n    x = 1\n    del x[0]'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class A:\n    x = 1\n    del x[0]': 0, 'class A:\n    x = 1\n    del x[0].A': 0}
    assert var_0.doc == {'class A:\n    x = 1\n    del x[0]': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    x = 1\n    del x[0].A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n'}
    assert var_0.imp == {'class A:\n    x = 1\n    del x[0]': {*()}}
    assert var_0.root == {'class A:\n    x = 1\n    del x[0]': 'class A:\n    x = 1\n    del x[0]', 'class A:\n    x = 1\n    del x[0].A': 'class A:\n    x = 1\n    del x[0]'}
    var_3 = True
    var_4 = '%1+UrT/}DShOT'
    var_5 = "\r\r'z?i5ft\n4\n"
    var_6 = 'aheiR3DE~\r]}HdOG'
    var_7 = {var_1: var_4, var_5: var_6}
    var_8 = module_0.Parser(toc=var_3, docstring=var_7, imp=var_2, alias=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is True
    assert var_8.b_level == 1
    assert var_8.toc is True
    assert var_8.level == {}
    assert var_8.doc == {}
    assert var_8.docstring == {'class A:\n    x = 1\n    del x[0]': '%1+UrT/}DShOT', "\r\r'z?i5ft\n4\n": 'aheiR3DE~\r]}HdOG'}
    assert var_8.imp is None
    assert var_8.root == {}
    assert var_8.alias is True
    assert var_8.const == {}

def test_case_25():
    var_0 = 'jNauqMhzHwR<2rq"s>5'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'jNauqMhzHwR<2rq"s>5'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_3 = '?A\r!:+~4~@6zY_{'
    var_4 = {}
    var_5 = module_0.Parser(var_2, level=var_4, docstring=var_1, alias=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is None
    assert var_5.b_level == 1
    assert var_5.toc is False
    assert var_5.level == {}
    assert var_5.doc == {}
    assert var_5.docstring == 'jNauqMhzHwR<2rq"s>5'
    assert var_5.imp == {}
    assert var_5.root == {}
    assert var_5.alias is None
    assert var_5.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_6 = module_0.esc_underscore(var_3)
    assert var_6 == '?A\r!:+~4~@6zY_{'
    var_7 = module_0.Parser(doc=var_2, docstring=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is True
    assert var_7.b_level == 1
    assert var_7.toc is False
    assert var_7.level == {}
    assert var_7.doc is None
    assert var_7.docstring == '?A\r!:+~4~@6zY_{'
    assert var_7.imp == {}
    assert var_7.root == {}
    assert var_7.alias == {}
    assert var_7.const == {}
    var_8 = 'U\r2uqQ'
    var_9 = []
    var_10 = var_7.class_api(var_8, var_8, var_9, var_8)
    var_11 = None
    var_12 = module_0.const_type(var_11)
    assert var_12 == 'Any'
    var_13 = var_7.__repr__()
    assert var_13 == "Parser(link=True, b_level=1, toc=False, level={}, doc=None, docstring='?A\\r!:+~4~@6zY_{', imp={}, root={}, alias={}, const={})"
    var_14 = var_7.__post_init__()
    var_15 = module_0.table(*var_13, items=var_13)
    assert var_15 == "| P | a | r | s | e | r | ( | l | i | n | k | = | T | r | u | e | , |   | b | _ | l | e | v | e | l | = | 1 | , |   | t | o | c | = | F | a | l | s | e | , |   | l | e | v | e | l | = | { | } | , |   | d | o | c | = | N | o | n | e | , |   | d | o | c | s | t | r | i | n | g | = | ' | ? | A | \\ | r | ! | : | + | ~ | 4 | ~ | @ | 6 | z | Y | _ | { | ' | , |   | i | m | p | = | { | } | , |   | r | o | o | t | = | { | } | , |   | a | l | i | a | s | = | { | } | , |   | c | o | n | s | t | = | { | } | ) |\n|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n| P |\n| a |\n| r |\n| s |\n| e |\n| r |\n| ( |\n| l |\n| i |\n| n |\n| k |\n| = |\n| T |\n| r |\n| u |\n| e |\n| , |\n|   |\n| b |\n| _ |\n| l |\n| e |\n| v |\n| e |\n| l |\n| = |\n| 1 |\n| , |\n|   |\n| t |\n| o |\n| c |\n| = |\n| F |\n| a |\n| l |\n| s |\n| e |\n| , |\n|   |\n| l |\n| e |\n| v |\n| e |\n| l |\n| = |\n| { |\n| } |\n| , |\n|   |\n| d |\n| o |\n| c |\n| = |\n| N |\n| o |\n| n |\n| e |\n| , |\n|   |\n| d |\n| o |\n| c |\n| s |\n| t |\n| r |\n| i |\n| n |\n| g |\n| = |\n| ' |\n| ? |\n| A |\n| \\ |\n| r |\n| ! |\n| : |\n| + |\n| ~ |\n| 4 |\n| ~ |\n| @ |\n| 6 |\n| z |\n| Y |\n| _ |\n| { |\n| ' |\n| , |\n|   |\n| i |\n| m |\n| p |\n| = |\n| { |\n| } |\n| , |\n|   |\n| r |\n| o |\n| o |\n| t |\n| = |\n| { |\n| } |\n| , |\n|   |\n| a |\n| l |\n| i |\n| a |\n| s |\n| = |\n| { |\n| } |\n| , |\n|   |\n| c |\n| o |\n| n |\n| s |\n| t |\n| = |\n| { |\n| } |\n| ) |\n\n"
    var_16 = var_7.__repr__()
    assert var_16 == "Parser(link=True, b_level=1, toc=False, level={}, doc=None, docstring='?A\\r!:+~4~@6zY_{', imp={}, root={}, alias={}, const={})"
    var_17 = var_7.__repr__()
    assert var_17 == "Parser(link=True, b_level=1, toc=False, level={}, doc=None, docstring='?A\\r!:+~4~@6zY_{', imp={}, root={}, alias={}, const={})"
    var_18 = module_0.esc_underscore(var_17)
    assert var_18 == "Parser(link=True, b\\_level=1, toc=False, level={}, doc=None, docstring='?A\\r!:+~4~@6zY\\_{', imp={}, root={}, alias={}, const={})"
    with pytest.raises(ValueError):
        module_2.field(default=var_16, default_factory=var_6, init=var_13, repr=var_16, compare=var_13, metadata=var_6)

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
    var_1 = '\ndef foo(x: int, y: str) -> bool:\n    pass\n'
    var_2 = 'r(E,.q0+$'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'r(E,.q0+$': 1, 'r(E,.q0+$.foo': 1}
    assert var_0.doc == {'r(E,.q0+$': '## Module `{}`\n<a id="{}"></a>\n\n', 'r(E,.q0+$.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `bool` |\n\n'}
    assert var_0.imp == {'r(E,.q0+$': {*()}}
    assert var_0.root == {'r(E,.q0+$': 'r(E,.q0+$', 'r(E,.q0+$.foo': 'r(E,.q0+$'}
    var_4 = 'bool'
    with pytest.raises(AttributeError):
        var_5 = bool('bool' in var_4.doc['test_module.foo'])
    assert var_5 is True

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
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 2}
    assert var_0.doc == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.docstring == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n"></a>\n\n### class MyClass\n\n*Full name:* `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n-myclass"></a>\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\nMethod docstring.\n'
    var_4 = ';v=K*'
    var_5 = var_0.load_docstring(var_4, var_2)

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
    var_1 = 'self'
    var_2 = None
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_1.arg(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.arg'
    assert var_5.arg == 'self'
    assert var_5.annotation is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.arg.annotation is None
    assert module_1.arg.type_comment is None
    assert module_1.arg.end_lineno is None
    assert module_1.arg.end_col_offset is None
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = [var_6, var_2, var_7, var_8, var_2, var_9, var_2]
    var_11 = {}
    var_12 = module_1.arguments(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.arguments'
    assert f'{type(var_12.posonlyargs).__module__}.{type(var_12.posonlyargs).__qualname__}' == 'builtins.list'
    assert len(var_12.posonlyargs) == 1
    assert var_12.args is None
    assert var_12.vararg == []
    assert var_12.kwonlyargs == []
    assert var_12.kw_defaults is None
    assert var_12.kwarg == []
    assert var_12.defaults is None
    assert module_1.arguments.vararg is None
    assert module_1.arguments.kwarg is None
    var_13 = 'root'
    var_14 = 'func'
    var_15 = True
    var_0.func_api(var_13, var_14, var_12, var_2, has_self=var_15, cls_method=var_15)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 'module'
    var_1 = 'module.T'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Resolver'
    assert var_4.root == 'module'
    assert var_4.alias == {'module.T': "typing.TypeVar('T')"}
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
    var_5 = 'T'
    var_6 = []
    var_7 = {}
    var_8 = module_1.Load(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_9 = [var_5, var_8]
    var_10 = {}
    var_11 = module_1.Name(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Name'
    assert var_11.id == 'T'
    assert f'{type(var_11.ctx).__module__}.{type(var_11.ctx).__qualname__}' == 'ast.Load'
    var_4.visit_Name(var_11)

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
    var_1 = 'test_module'
    var_2 = []
    var_3 = 0
    var_4 = '_private: int = 1'
    var_5 = module_1.parse(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Module'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'builtins.list'
    assert len(var_5.body) == 1
    assert var_5.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = var_5.body[var_3]
    var_7 = [var_6]
    var_8 = var_0.class_api(var_1, var_1, var_2, var_7)
    with pytest.raises(KeyError):
        var_9 = bool('_private' not in var_0.doc[var_6])
    assert var_9 is True

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
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = []
    var_4 = 0
    var_5 = 'x: int = 1\ndel x'
    var_6 = module_1.parse(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Module'
    assert f'{type(var_6.body).__module__}.{type(var_6.body).__qualname__}' == 'builtins.list'
    assert len(var_6.body) == 2
    assert var_6.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = var_6.body[var_4]
    var_8 = [var_7]
    var_0.class_api(var_1, var_2, var_3, var_8)

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
    var_1 = []
    var_2 = 0
    var_3 = 'x: q.int = 1'
    var_4 = 'exec'
    var_5 = module_1.parse(var_3, mode=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Module'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'builtins.list'
    assert len(var_5.body) == 1
    assert var_5.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = var_0.globals(var_5, var_5)
    var_7 = var_5.body[var_2]
    var_8 = 'del x'
    var_9 = module_1.parse(var_8, mode=var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Module'
    assert f'{type(var_9.body).__module__}.{type(var_9.body).__qualname__}' == 'builtins.list'
    assert len(var_9.body) == 1
    assert var_9.type_ignores == []
    var_10 = var_9.body[var_2]
    var_11 = [var_7, var_10]
    var_12 = module_0.doctest(var_4)
    assert var_12 == 'exec'
    var_13 = var_0.class_api(var_3, var_4, var_1, var_11)
    var_14 = module_0.Parser(b_level=var_7, level=var_13, root=var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'apimd.parser.Parser'
    assert var_14.link is True
    assert f'{type(var_14.b_level).__module__}.{type(var_14.b_level).__qualname__}' == 'ast.AnnAssign'
    assert var_14.toc is False
    assert var_14.level is None
    assert var_14.doc == {}
    assert var_14.docstring == {}
    assert var_14.imp == {}
    assert f'{type(var_14.root).__module__}.{type(var_14.root).__qualname__}' == 'ast.Module'
    assert var_14.alias == {}
    assert var_14.const == {}

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
    var_1 = []
    var_2 = 0
    var_3 = 'exec'
    var_4 = 'del x'
    var_5 = module_1.parse(var_4, mode=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Module'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'builtins.list'
    assert len(var_5.body) == 1
    assert var_5.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = var_5.body[var_2]
    var_7 = [var_5, var_6]
    var_8 = var_0.class_api(var_3, var_3, var_1, var_7)
    var_9 = bool(var_6 in var_0.doc)
    assert var_9 is True
    with pytest.raises(AttributeError):
        var_10 = bool('x' not in var_5.doc[var_3])
    assert var_10 is True

def test_case_34():
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
    var_5 = module_1.Constant(*var_4, **var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Constant'
    assert var_5.value == 'invalid syntax'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_6 = var_2.visit_Constant(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Constant'
    assert var_6.value == 'invalid syntax'
    var_7 = bool(var_6 == var_5)
    assert var_7 is True

def test_case_35():
    var_0 = 'module'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'module'
    assert var_3.alias == {}
    assert var_3.self_ty == 'T'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = [var_2, var_6]
    var_8 = {}
    var_9 = module_1.Name(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Name'
    assert var_9.id == 'T'
    assert f'{type(var_9.ctx).__module__}.{type(var_9.ctx).__qualname__}' == 'ast.Load'
    var_10 = var_3.visit_Name(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Name'
    assert var_10.id == 'Self'
    assert f'{type(var_10.ctx).__module__}.{type(var_10.ctx).__qualname__}' == 'ast.Load'
    var_11 = var_10.id
    assert var_11 == 'Self'

def test_case_36():
    var_0 = 'module'
    var_1 = 'module.T'
    var_2 = {var_1: var_0}
    var_3 = module_0.Resolver(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'module'
    assert var_3.alias == {'module.T': 'module'}
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
    var_4 = 'T'
    var_5 = module_3.getdoc(var_2)
    assert var_5 == "dict() -> new empty dictionary\ndict(mapping) -> new dictionary initialized from a mapping object's\n    (key, value) pairs\ndict(iterable) -> new dictionary initialized as if via:\n    d = {}\n    for k, v in iterable:\n        d[k] = v\ndict(**kwargs) -> new dictionary initialized with the name=value pairs\n    in the keyword argument list.  For example:  dict(one=1, two=2)"
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
    var_6 = [var_4, var_5]
    var_7 = {}
    var_8 = module_1.Name(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Name'
    assert var_8.id == 'T'
    assert var_8.ctx == "dict() -> new empty dictionary\ndict(mapping) -> new dictionary initialized from a mapping object's\n    (key, value) pairs\ndict(iterable) -> new dictionary initialized as if via:\n    d = {}\n    for k, v in iterable:\n        d[k] = v\ndict(**kwargs) -> new dictionary initialized with the name=value pairs\n    in the keyword argument list.  For example:  dict(one=1, two=2)"
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_9 = var_3.visit_Name(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Name'
    assert var_9.id == 'module'
    assert f'{type(var_9.ctx).__module__}.{type(var_9.ctx).__qualname__}' == 'ast.Load'
    assert var_9.lineno == 1
    assert var_9.col_offset == 0
    assert var_9.end_lineno == 1
    assert var_9.end_col_offset == 6
    var_10 = var_9.id

def test_case_37():
    var_0 = 'module'
    var_1 = 'module.T'
    var_2 = {var_1: var_1}
    var_3 = module_0.Resolver(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'module'
    assert var_3.alias == {'module.T': 'module.T'}
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
    var_4 = 'T'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_1.Name(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Name'
    assert var_10.id == 'T'
    assert f'{type(var_10.ctx).__module__}.{type(var_10.ctx).__qualname__}' == 'ast.Load'
    var_11 = var_3.visit_Name(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Name'
    assert var_11.id == 'T'
    assert f'{type(var_11.ctx).__module__}.{type(var_11.ctx).__qualname__}' == 'ast.Load'
    var_12 = var_11.id
    assert var_12 == 'T'

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
    var_1 = None
    var_2 = [var_1, var_1]
    var_3 = module_1.Attribute(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Attribute'
    assert var_3.value is None
    assert var_3.attr is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_0.resolve(var_1, var_3)

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
    var_1 = 'K|`'
    var_2 = None
    var_3 = var_0.load_docstring(var_1, var_2)
    var_4 = 0
    var_5 = 'x: int = 1'
    var_6 = 'exec'
    var_7 = module_1.parse(var_5, mode=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Module'
    assert f'{type(var_7.body).__module__}.{type(var_7.body).__qualname__}' == 'builtins.list'
    assert len(var_7.body) == 1
    assert var_7.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = var_7.body[var_4]
    var_9 = 'del x'
    var_10 = module_1.parse(var_9, mode=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Module'
    assert f'{type(var_10.body).__module__}.{type(var_10.body).__qualname__}' == 'builtins.list'
    assert len(var_10.body) == 1
    assert var_10.type_ignores == []
    var_11 = var_10.body[var_4]
    var_12 = [var_8, var_11]
    var_13 = module_0.const_type(var_8)
    assert var_13 == 'Any'
    var_0.class_api(var_5, var_6, var_12, var_12)

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
    var_1 = '.fI~,|\tuY04*('
    var_2 = 'a:d'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'.fI~,|\tuY04*(': 1}
    assert var_0.doc == {'.fI~,|\tuY04*(': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'.fI~,|\tuY04*(': {*()}}
    assert var_0.root == {'.fI~,|\tuY04*(': '.fI~,|\tuY04*('}
    var_0.load_docstring(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 'cp5jNIV'
    var_1 = '8E>/eHb\re?#:v'
    var_2 = {var_0: var_0, var_1: var_1}
    var_3 = module_0.Parser(doc=var_2, alias=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {'cp5jNIV': 'cp5jNIV', '8E>/eHb\re?#:v': '8E>/eHb\re?#:v'}
    assert var_3.docstring == {}
    assert var_3.imp == {}
    assert var_3.root == {}
    assert var_3.alias == {'cp5jNIV': 'cp5jNIV', '8E>/eHb\re?#:v': '8E>/eHb\re?#:v'}
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
def test_case_42():
    var_0 = 'a'
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_1.Constant(*var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Constant'
    assert var_4.value == 'a'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_5 = 'b'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_1.Constant(*var_6, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Constant'
    assert var_9.value == 'b'
    var_10 = [var_4, var_9]
    var_11 = 1
    var_12 = 'value'
    var_13 = {var_12: var_11}
    var_14 = module_1.Constant(*var_1, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Constant'
    assert var_14.value == 1
    var_15 = 2
    var_16 = []
    var_17 = 'value'
    var_18 = {var_17: var_15}
    var_19 = module_1.Constant(*var_16, **var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.Constant'
    assert var_19.value == 2
    var_20 = [var_14, var_19]
    var_21 = []
    var_22 = 'keys'
    var_23 = {var_22: var_10, var_17: var_20}
    var_24 = module_1.Dict(*var_21, **var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'ast.Dict'
    assert f'{type(var_24.keys).__module__}.{type(var_24.keys).__qualname__}' == 'builtins.list'
    assert len(var_24.keys) == 2
    assert f'{type(var_24.value).__module__}.{type(var_24.value).__qualname__}' == 'builtins.list'
    assert len(var_24.value) == 2
    module_0.const_type(var_24)
    assert var_25 == 'dict[str, str, int, int]'

def test_case_43():
    var_0 = []
    var_1 = 'elts'
    var_2 = {var_1: var_0}
    var_3 = module_1.Tuple(*var_0, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Tuple'
    assert var_3.elts == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_4 = module_0.const_type(var_3)
    assert var_4 == 'tuple'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_0 = 1
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_1.Constant(*var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Constant'
    assert var_4.value == 1
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_5 = 'a'
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_1.Constant(*var_6, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Constant'
    assert var_9.value == 'a'
    var_10 = [var_4, var_9]
    var_11 = 2
    var_12 = []
    var_13 = 'value'
    var_14 = {var_13: var_11}
    var_15 = module_1.Constant(*var_12, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Constant'
    assert var_15.value == 2
    var_16 = 'b'
    var_17 = []
    var_18 = 'value'
    var_19 = {var_18: var_16}
    var_20 = module_1.Constant(*var_17, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.Constant'
    assert var_20.value == 'b'
    var_21 = [var_15, var_20]
    var_22 = []
    var_23 = 'keys'
    var_24 = 'values'
    var_25 = {var_23: var_10, var_24: var_21}
    var_26 = module_1.Dict(*var_22, **var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'ast.Dict'
    assert f'{type(var_26.keys).__module__}.{type(var_26.keys).__qualname__}' == 'builtins.list'
    assert len(var_26.keys) == 2
    assert f'{type(var_26.values).__module__}.{type(var_26.values).__qualname__}' == 'builtins.list'
    assert len(var_26.values) == 2
    var_27 = module_0.const_type(var_26)
    assert var_27 == 'dict[Any, Any]'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_1 = []
    var_2 = 0
    var_3 = 'md=h,##(Sf)?}z)'
    var_4 = 'exec'
    var_5 = module_1.parse(var_3, mode=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Module'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'builtins.list'
    assert len(var_5.body) == 1
    assert var_5.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = var_5.body[var_2]
    var_7 = [var_6, var_5]
    var_0.class_api(var_3, var_4, var_1, var_7)

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
    var_1 = 'class A:\n    x = 1\n    del x[0]'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class A:\n    x = 1\n    del x[0]': 0, 'class A:\n    x = 1\n    del x[0].A': 0}
    assert var_0.doc == {'class A:\n    x = 1\n    del x[0]': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    x = 1\n    del x[0].A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n'}
    assert var_0.imp == {'class A:\n    x = 1\n    del x[0]': {*()}}
    assert var_0.root == {'class A:\n    x = 1\n    del x[0]': 'class A:\n    x = 1\n    del x[0]', 'class A:\n    x = 1\n    del x[0].A': 'class A:\n    x = 1\n    del x[0]'}
    var_3 = module_1.ImportFrom()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.ImportFrom'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_4 = var_0.imports(var_2, var_3)
    var_5 = var_0.compile()
    assert var_5 == '## Module `class A:\n    x = 1\n    del x[0]`\n<a id="class a:\n    x = 1\n    del x[0]"></a>\n\n### class A\n\n*Full name:* `class A:\n    x = 1\n    del x[0].A`\n<a id="class a:\n    x = 1\n    del x[0]-a"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n'

def test_case_47():
    var_0 = ''
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
    with pytest.raises(AttributeError):
        var_3 = list(var_2)

def test_case_48():
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
    var_1 = '__all__'
    var_2 = [var_1]
    var_3 = 'foo'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 'foo'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_8 = 'bar'
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Constant'
    assert var_12.value == 'bar'
    var_13 = [var_7, var_12]
    var_14 = []
    var_15 = 'elts'
    var_16 = {var_15: var_13}
    var_17 = module_1.List(*var_14, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.List'
    assert f'{type(var_17.elts).__module__}.{type(var_17.elts).__qualname__}' == 'builtins.list'
    assert len(var_17.elts) == 2
    var_18 = []
    var_19 = 'targets'
    var_20 = 'value'
    var_21 = {var_19: var_2, var_20: var_17}
    var_22 = module_1.Assign(*var_18, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Assign'
    assert var_22.targets == ['__all__']
    assert f'{type(var_22.value).__module__}.{type(var_22.value).__qualname__}' == 'ast.List'
    assert module_1.Assign.type_comment is None
    var_23 = 'test'
    var_24 = var_0.globals(var_23, var_22)
    with pytest.raises(KeyError):
        var_25 = var_0.imp['test']

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
    with pytest.raises(AttributeError):
        var_5 = bool('test_module.foo' in var_4.doc)
    assert var_5 is True

def test_case_51():
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
    var_1 = 'm`KS$a%iv7\t>a57:'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == 'm`KS$a%iv7\t>a57:'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = 'hello'
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Constant'
    assert var_11.value == 'hello'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_12 = []
    var_13 = 'targets'
    var_14 = 'value'
    var_15 = {var_13: var_6, var_14: var_11, var_13: var_2}
    var_16 = module_1.Assign(*var_12, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Assign'
    assert var_16.targets == []
    assert f'{type(var_16.value).__module__}.{type(var_16.value).__qualname__}' == 'ast.Constant'
    assert module_1.Assign.type_comment is None
    var_17 = 'test'
    var_18 = var_0.globals(var_17, var_16)
    with pytest.raises(KeyError):
        var_19 = var_0.alias['test.y']
    assert var_19 == "'hello'"

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
    var_1 = 'x'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == 'x'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = 42
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Constant'
    assert var_11.value == 42
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_12 = 'int'
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_6, var_15: var_11, var_16: var_12}
    var_18 = module_1.Assign(*var_13, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Assign'
    assert f'{type(var_18.targets).__module__}.{type(var_18.targets).__qualname__}' == 'builtins.list'
    assert len(var_18.targets) == 1
    assert f'{type(var_18.value).__module__}.{type(var_18.value).__qualname__}' == 'ast.Constant'
    assert var_18.type_comment == 'int'
    assert module_1.Assign.type_comment is None
    var_19 = 'root'
    var_20 = 'root.Class'
    var_21 = []
    var_22 = [var_18]
    var_0.class_api(var_19, var_20, var_21, var_22)

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
    var_1 = '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 2, '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 2}
    assert var_0.doc == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n'}
    assert var_0.docstring == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': 'Class docstring.', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n', '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method': '\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n"></a>\n\n### class MyClass\n\n*Full name:* `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n-myclass"></a>\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\nclass MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n.MyClass.method`\n<a id="\nclass myclass:\n    """class docstring-"""\n    def method(self):\n        """method docstring-"""\n        pass\n-myclass-method"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\nMethod docstring.\n'

def test_case_55():
    var_0 = True
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_1.Constant(*var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Constant'
    assert var_4.value is True
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_5 = 'x'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_1.Name(*var_6, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Name'
    assert var_9.id == 'x'
    var_10 = [var_9]
    var_11 = []
    var_12 = 'value'
    var_13 = {var_12: var_0}
    var_14 = module_1.Constant(*var_11, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Constant'
    assert var_14.value is True
    var_15 = []
    var_16 = 'targets'
    var_17 = 'value'
    var_18 = {var_16: var_10, var_17: var_14}
    var_19 = module_1.Assign(*var_15, **var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.Assign'
    assert f'{type(var_19.targets).__module__}.{type(var_19.targets).__qualname__}' == 'builtins.list'
    assert len(var_19.targets) == 1
    assert f'{type(var_19.value).__module__}.{type(var_19.value).__qualname__}' == 'ast.Constant'
    assert module_1.Assign.type_comment is None
    var_20 = [var_19]
    var_21 = []
    var_22 = []
    var_23 = 'test'
    var_24 = 'body'
    var_25 = 'orelse'
    var_26 = {var_23: var_4, var_24: var_20, var_25: var_21}
    var_27 = module_1.If(*var_22, **var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'ast.If'
    assert f'{type(var_27.test).__module__}.{type(var_27.test).__qualname__}' == 'ast.Constant'
    assert f'{type(var_27.body).__module__}.{type(var_27.body).__qualname__}' == 'builtins.list'
    assert len(var_27.body) == 1
    assert var_27.orelse == []
    var_28 = [var_27]
    var_29 = module_0.walk_body(var_28)
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_30 = list(var_29)
    var_31 = 'id'
    var_32 = {var_31: var_5}
    var_33 = module_1.Name(*var_22, **var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'ast.Name'
    assert var_33.id == 'x'
    var_34 = [var_33]
    var_35 = []
    var_36 = 'value'
    var_37 = {var_36: var_0}
    var_38 = module_1.Constant(*var_35, **var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'ast.Constant'
    assert var_38.value is True
    var_39 = []
    var_40 = 'targets'
    var_41 = 'value'
    var_42 = {var_40: var_34, var_41: var_38}
    var_43 = module_1.Assign(*var_39, **var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'ast.Assign'
    assert f'{type(var_43.targets).__module__}.{type(var_43.targets).__qualname__}' == 'builtins.list'
    assert len(var_43.targets) == 1
    assert f'{type(var_43.value).__module__}.{type(var_43.value).__qualname__}' == 'ast.Constant'
    var_44 = [var_43]
    var_45 = bool(var_30 == var_44)

def test_case_56():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Name'
    assert var_7.id == 'x'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = 1
    var_9 = []
    var_10 = 'value'
    var_11 = {var_3: var_8, var_10: var_8}
    var_12 = module_1.Constant(*var_9, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Constant'
    assert var_12.x == 1
    assert var_12.value == 1
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_13 = []
    var_14 = module_1.Assign(*var_13, **var_11)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Assign'
    assert var_14.x == 1
    assert var_14.value == 1
    assert module_1.Assign.type_comment is None
    var_15 = [var_14]
    var_16 = []
    var_17 = 'body'
    var_18 = 'handlers'
    var_19 = 'orelse'
    var_20 = 'finalbody'
    var_21 = {var_17: var_0, var_18: var_1, var_19: var_2, var_20: var_15}
    var_22 = module_1.Try(*var_16, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Try'
    assert var_22.body == []
    assert var_22.handlers == []
    assert var_22.orelse == []
    assert f'{type(var_22.finalbody).__module__}.{type(var_22.finalbody).__qualname__}' == 'builtins.list'
    assert len(var_22.finalbody) == 1
    var_23 = [var_22]
    var_24 = module_0.walk_body(var_23)
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_25 = list(var_24)
    var_26 = []
    var_27 = 'id'
    var_28 = {var_27: var_3}
    var_29 = module_1.Name(*var_26, **var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'ast.Name'
    assert var_29.id == 'x'
    var_30 = []
    var_31 = module_1.Assign(*var_30, **var_21)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'ast.Assign'
    assert var_31.body == []
    assert var_31.handlers == []
    assert var_31.orelse == []
    assert f'{type(var_31.finalbody).__module__}.{type(var_31.finalbody).__qualname__}' == 'builtins.list'
    assert len(var_31.finalbody) == 1
    var_32 = [var_31]
    var_33 = bool(var_25 == var_32)

@pytest.mark.xfail(strict=True)
def test_case_57():
    var_0 = True
    var_1 = None
    var_2 = module_0.Resolver(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Resolver'
    assert var_2.root is None
    assert var_2.alias is None
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
    var_3 = module_0.Parser(var_0, toc=var_0)
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
    assert var_3.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_4 = var_3.compile()
    assert var_4 == '**Table of contents:**\n\n\n'
    var_5 = None
    module_1.parse(var_5)

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
    var_1 = 'z'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == 'z'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = 3.14
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Constant'
    assert var_11.value == pytest.approx(3.14, abs=0.01, rel=0.01)
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_12 = 'float'
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = 'type_comment'
    var_17 = {var_14: var_6, var_15: var_11, var_16: var_12}
    var_18 = module_1.Assign(*var_13, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Assign'
    assert f'{type(var_18.targets).__module__}.{type(var_18.targets).__qualname__}' == 'builtins.list'
    assert len(var_18.targets) == 1
    assert f'{type(var_18.value).__module__}.{type(var_18.value).__qualname__}' == 'ast.Constant'
    assert var_18.type_comment == 'float'
    assert module_1.Assign.type_comment is None
    var_19 = 'module'
    var_20 = var_0.globals(var_19, var_18)
    assert var_0.alias == {'module.z': '3.14'}
    var_21 = var_0.alias['module.z']
    assert var_21 == '3.14'
    with pytest.raises(KeyError):
        var_22 = var_0.const['module.z']
    assert var_22 == 'float'

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
    var_1 = 'class A:\n    x = 1\n    del x[0]'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class A:\n    x = 1\n    del x[0]': 0, 'class A:\n    x = 1\n    del x[0].A': 0}
    assert var_0.doc == {'class A:\n    x = 1\n    del x[0]': '## Module `{}`\n<a id="{}"></a>\n\n', 'class A:\n    x = 1\n    del x[0].A': '### class A\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `x` | `int` |\n\n'}
    assert var_0.imp == {'class A:\n    x = 1\n    del x[0]': {*()}}
    assert var_0.root == {'class A:\n    x = 1\n    del x[0]': 'class A:\n    x = 1\n    del x[0]', 'class A:\n    x = 1\n    del x[0].A': 'class A:\n    x = 1\n    del x[0]'}
    var_3 = module_3.getdoc(var_0)
    assert var_3 == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
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
    var_4 = module_0.doctest(var_3)
    assert var_4 == 'AST parser.\n\nUsage:\n```python\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n```\n\nOr create with parameters:\n```python\n>>> p = Parser.new(link=True, level=1)\n```'
    var_5 = None
    module_0.doctest(var_5)

def test_case_60():
    var_0 = []
    var_1 = {}
    var_2 = module_1.Load(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = module_3.getdoc(var_1)
    assert var_3 == "dict() -> new empty dictionary\ndict(mapping) -> new dictionary initialized from a mapping object's\n    (key, value) pairs\ndict(iterable) -> new dictionary initialized as if via:\n    d = {}\n    for k, v in iterable:\n        d[k] = v\ndict(**kwargs) -> new dictionary initialized with the name=value pairs\n    in the keyword argument list.  For example:  dict(one=1, two=2)"
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
    var_4 = []
    var_5 = 'func'
    var_6 = {var_5: var_3}
    var_7 = module_1.Call(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Call'
    assert var_7.func == "dict() -> new empty dictionary\ndict(mapping) -> new dictionary initialized from a mapping object's\n    (key, value) pairs\ndict(iterable) -> new dictionary initialized as if via:\n    d = {}\n    for k, v in iterable:\n        d[k] = v\ndict(**kwargs) -> new dictionary initialized with the name=value pairs\n    in the keyword argument list.  For example:  dict(one=1, two=2)"
    var_8 = module_0.const_type(var_7)
    assert var_8 == 'Any'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

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
    var_1 = 'self'
    var_2 = []
    var_3 = {}
    var_4 = module_1.Load(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = []
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_6, var_7: var_4}
    var_9 = module_1.Name(*var_5, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Name'
    assert var_9.id == 'id'
    assert f'{type(var_9.ctx).__module__}.{type(var_9.ctx).__qualname__}' == 'ast.Load'
    var_10 = []
    var_11 = 'arg'
    var_12 = 'annotation'
    var_13 = {var_11: var_1, var_12: var_9}
    var_14 = module_1.arg(*var_10, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.arg'
    assert var_14.arg == 'self'
    assert f'{type(var_14.annotation).__module__}.{type(var_14.annotation).__qualname__}' == 'ast.Name'
    assert module_1.arg.annotation is None
    assert module_1.arg.type_comment is None
    assert module_1.arg.end_lineno is None
    assert module_1.arg.end_col_offset is None
    var_15 = [var_14]
    var_16 = 'root'
    var_17 = True
    var_18 = var_0.func_ann(var_16, var_15, has_self=var_17, cls_method=var_17)
    var_19 = list(var_18)
    var_20 = var_19[0]
    assert var_20 == 'type[Self]'