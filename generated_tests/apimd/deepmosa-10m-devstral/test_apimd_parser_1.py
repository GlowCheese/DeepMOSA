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
    var_0 = "Regular line 1\n>>> print('hello')\nRegular line 2"
    var_1 = module_0.esc_underscore(var_0)
    assert var_1 == "Regular line 1\n>>> print('hello')\nRegular line 2"
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_2 = module_0.doctest(var_0)
    assert var_2 == "Regular line 1\n```python\n>>> print('hello')\n```\nRegular line 2"

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
    var_1 = 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass': 0, 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass.Foo': 0, 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass.Foo.bar': 0, 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass.Foo.baz': 0}
    assert var_0.doc == {'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass.Foo': '### class Foo\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass.Foo.bar': '#### Foo.bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass.Foo.baz': '#### Foo.baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticethod` |\n\n| return |\n|:------:|\n| `Self` |\n\n'}
    assert var_0.imp == {'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass': {*()}}
    assert var_0.root == {'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass': 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass', 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass.Foo': 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass', 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass.Foo.bar': 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass', 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass.Foo.baz': 'class Foo:\n    def bar(self):\n      pass\n    @staticethod\n    def baz():\n        pass'}

def test_case_3():
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
    var_1 = 'import os\nfrom sys import path\nx = 1'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'import os\nfrom sys import path\nx = 1': 0}
    assert var_0.doc == {'import os\nfrom sys import path\nx = 1': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'import os\nfrom sys import path\nx = 1': {*()}}
    assert var_0.root == {'import os\nfrom sys import path\nx = 1': 'import os\nfrom sys import path\nx = 1'}
    assert var_0.alias == {'import os\nfrom sys import path\nx = 1.os': 'os', 'import os\nfrom sys import path\nx = 1.path': 'sys.path', 'import os\nfrom sys import path\nx = 1.x': '1'}

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
    var_1 = 'test_module'
    var_2 = 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass'
    var_3 = module_0.const_type(var_0)
    assert var_3 == 'Any'
    var_4 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module': 0, 'test_module.Foo': 0, 'test_module.Foo.bar': 0, 'test_module.Foo.baz': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.Foo': '### class Foo\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.Foo.bar': '#### Foo.bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'test_module.Foo.baz': '#### Foo.baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.Foo': 'test_module', 'test_module.Foo.bar': 'test_module', 'test_module.Foo.baz': 'test_module'}
    var_5 = var_0.compile()
    assert var_5 == '## Module `test_module`\n<a id="test_module"></a>\n\n### class Foo\n\n*Full name:* `test_module.Foo`\n<a id="test_module-foo"></a>\n\n#### Foo.bar()\n\n*Full name:* `test_module.Foo.bar`\n<a id="test_module-foo-bar"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n#### Foo.baz()\n\n*Full name:* `test_module.Foo.baz`\n<a id="test_module-foo-baz"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n'
    var_6 = module_1.expr()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.expr'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.expr.end_lineno is None
    assert module_1.expr.end_col_offset is None
    var_7 = [var_6]
    var_8 = [var_4, var_4, var_4]
    var_9 = module_1.AsyncFunctionDef(*var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.AsyncFunctionDef'
    assert var_9.name is None
    assert var_9.args is None
    assert var_9.body is None
    assert module_1.AsyncFunctionDef.returns is None
    assert module_1.AsyncFunctionDef.type_comment is None
    var_10 = [var_9, var_9, var_9, var_9]
    var_11 = var_0.class_api(var_5, var_1, var_7, var_10)
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n|   |\n\n', 'test_module.Foo': '### class Foo\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.Foo.bar': '#### Foo.bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'test_module.Foo.baz': '#### Foo.baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    var_12 = 'test_module.Foo.bar'
    var_13 = {var_12: var_3, var_2: var_1, var_1: var_1}
    var_14 = module_0.Parser(var_4, doc=var_4, docstring=var_4, root=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'apimd.parser.Parser'
    assert var_14.link is None
    assert var_14.b_level == 1
    assert var_14.toc is False
    assert var_14.level == {}
    assert var_14.doc is None
    assert var_14.docstring is None
    assert var_14.imp == {}
    assert var_14.root == {'test_module.Foo.bar': 'Any', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass': 'test_module', 'test_module': 'test_module'}
    assert var_14.alias == {}
    assert var_14.const == {}
    var_15 = var_14.globals(var_13, var_4)
    var_16 = var_0.load_docstring(var_12, var_5)
    var_17 = module_2.field()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'dataclasses.Field'
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
    var_14.imports(var_17, var_17)

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
    var_1 = 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass': 0, 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo': 0, 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.bar': 0, 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.baz': 0}
    assert var_0.doc == {'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo': '### class Foo\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.bar': '#### Foo.bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.baz': '#### Foo.baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass': {*()}}
    assert var_0.root == {'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass': 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo': 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.bar': 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.baz': 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass`\n<a id="class foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass"></a>\n\n### class Foo\n\n*Full name:* `class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo`\n<a id="class foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass-foo"></a>\n\n#### Foo.bar()\n\n*Full name:* `class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.bar`\n<a id="class foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass-foo-bar"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n#### Foo.baz()\n\n*Full name:* `class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.baz`\n<a id="class foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass-foo-baz"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n'

def test_case_12():
    var_0 = "Regular line 1\n>>> print('hello')\nRegular line 2"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "Regular line 1\n```python\n>>> print('hello')\n```\nRegular line 2"
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_1 = [var_0, var_0]
    var_2 = module_1.List(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.List'
    assert var_2.elts is None
    assert var_2.ctx is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
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
    var_1 = module_0.const_type(var_0)
    assert var_1 == 'Any'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Any': 0}
    assert var_0.doc == {'Any': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Any': {*()}}
    assert var_0.root == {'Any': 'Any'}
    var_3 = var_0.load_docstring(var_1, var_1)

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
    var_1 = 'test_module'
    var_2 = 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass'
    var_3 = module_0.const_type(var_0)
    assert var_3 == 'Any'
    var_4 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module': 0, 'test_module.Foo': 0, 'test_module.Foo.bar': 0, 'test_module.Foo.baz': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.Foo': '### class Foo\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.Foo.bar': '#### Foo.bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'test_module.Foo.baz': '#### Foo.baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.Foo': 'test_module', 'test_module.Foo.bar': 'test_module', 'test_module.Foo.baz': 'test_module'}
    var_5 = var_0.compile()
    assert var_5 == '## Module `test_module`\n<a id="test_module"></a>\n\n### class Foo\n\n*Full name:* `test_module.Foo`\n<a id="test_module-foo"></a>\n\n#### Foo.bar()\n\n*Full name:* `test_module.Foo.bar`\n<a id="test_module-foo-bar"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n#### Foo.baz()\n\n*Full name:* `test_module.Foo.baz`\n<a id="test_module-foo-baz"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n'
    var_6 = 'test_module.Foo.bar'
    var_7 = var_0.load_docstring(var_6, var_5)
    var_0.class_api(var_4, var_1, var_5, var_1)

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
    var_1 = module_0.const_type(var_0)
    assert var_1 == 'Any'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Any': 0}
    assert var_0.doc == {'Any': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Any': {*()}}
    assert var_0.root == {'Any': 'Any'}

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
    var_1 = 'mkEz=I'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'mkEz=I': 0}
    assert var_0.doc == {'mkEz=I': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'mkEz=I': {*()}}
    assert var_0.root == {'mkEz=I': 'mkEz=I'}
    assert var_0.alias == {'mkEz=I.mkEz': 'I'}

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
    var_1 = module_0.const_type(var_0)
    assert var_1 == 'Any'
    var_2 = module_0.doctest(var_1)
    assert var_2 == 'Any'
    var_3 = var_0.load_docstring(var_2, var_2)
    var_4 = var_0.globals(var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = module_2.dataclass()
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_2 = 'V9=\\"|I0G@%'
    var_3 = module_0.code(var_2)
    assert var_3 == '<code>V9=\\"&#124;I0G@%</code>'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_1.visit_Subscript(var_0)

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
    var_2 = var_0.class_api(var_0, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_1.Constant()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'ast.Constant'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
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

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = True
    var_2 = {}
    var_3 = module_0.Parser(toc=var_1, imp=var_2)
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
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
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
    assert var_4 == '**Table of contents:**\n\n\n'
    module_0.table(items=var_0)

def test_case_24():
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
    var_1 = 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass': 0, 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo': 0, 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.bar': 0, 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.baz': 0}
    assert var_0.doc == {'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo': '### class Foo\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.bar': '#### Foo.bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| sef | return |\n|:---:|:------:|\n| `Self` | `Any` |\n\n', 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.baz': '#### Foo.baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass': {*()}}
    assert var_0.root == {'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass': 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass', 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo': 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass', 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.bar': 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass', 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.baz': 'class Foo:\n    def bar(sef):\n        pass\n    @staticmethod\n    def baz():\n        pass'}

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
    var_1 = 'import os\nfrom sys import path\nx = 1'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'import os\nfrom sys import path\nx = 1': 0}
    assert var_0.doc == {'import os\nfrom sys import path\nx = 1': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'import os\nfrom sys import path\nx = 1': {*()}}
    assert var_0.root == {'import os\nfrom sys import path\nx = 1': 'import os\nfrom sys import path\nx = 1'}
    assert var_0.alias == {'import os\nfrom sys import path\nx = 1.os': 'os', 'import os\nfrom sys import path\nx = 1.path': 'sys.path', 'import os\nfrom sys import path\nx = 1.x': '1'}
    var_3 = var_0.compile()
    assert var_3 == '\n'

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
    var_1 = 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass': 0, 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo': 0, 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo.bar': 0, 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo.baz': 0}
    assert var_0.doc == {'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo': '### class Fo\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo.bar': '#### Fo.bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo.baz': '#### Fo.baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Self` |\n\n'}
    assert var_0.imp == {'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass': {*()}}
    assert var_0.root == {'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass': 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass', 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo': 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass', 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo.bar': 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass', 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo.baz': 'class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass'}
    var_3 = '~Dk'
    var_4 = var_0.load_docstring(var_3, var_2)
    var_5 = var_0.compile()
    assert var_5 == '## Module `class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass`\n<a id="class fo:\n    def bar(self):\n        ws\n    cotaticmethod\n    def baz():\n       pass"></a>\n\n### class Fo\n\n*Full name:* `class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo`\n<a id="class fo:\n    def bar(self):\n        ws\n    cotaticmethod\n    def baz():\n       pass-fo"></a>\n\n#### Fo.bar()\n\n*Full name:* `class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo.bar`\n<a id="class fo:\n    def bar(self):\n        ws\n    cotaticmethod\n    def baz():\n       pass-fo-bar"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n#### Fo.baz()\n\n*Full name:* `class Fo:\n    def bar(self):\n        Ws\n    cOtaticmethod\n    def baz():\n       pass.Fo.baz`\n<a id="class fo:\n    def bar(self):\n        ws\n    cotaticmethod\n    def baz():\n       pass-fo-baz"></a>\n\n| return |\n|:------:|\n| `Self` |\n'

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
    var_1 = '__:all__'
    var_2 = None
    var_3 = var_0.__repr__()
    assert var_3 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_4 = var_0.parse(var_1, var_1)
    assert var_0.level == {'__:all__': 0}
    assert var_0.doc == {'__:all__': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'__:all__': {*()}}
    assert var_0.root == {'__:all__': '__:all__'}
    var_5 = var_0.compile()
    assert var_5 == '\n'
    var_6 = None
    var_7 = var_0.load_docstring(var_1, var_2)
    var_8 = module_3.getdoc(var_7)
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
    var_9 = []
    var_10 = 'elts'
    var_11 = {var_10: var_6}
    var_12 = module_1.List(*var_9, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.List'
    assert var_12.elts is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_13 = module_0.is_magic(var_5)
    assert var_13 is False
    var_14 = module_0.const_type(var_8)
    assert var_14 == 'Any'
    var_15 = module_0.const_type(var_12)
    assert var_15 == 'list'
    var_0.is_public(var_7)

@pytest.mark.xfail(strict=True)
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
    var_1 = '__all__'
    var_2 = []
    var_3 = {var_1: var_1}
    var_4 = module_1.Name(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Name'
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
    var_10 = [var_9, var_9, var_9]
    var_11 = []
    var_12 = 'elts'
    var_13 = {var_12: var_10}
    var_14 = module_1.List(*var_11, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.List'
    assert f'{type(var_14.elts).__module__}.{type(var_14.elts).__qualname__}' == 'builtins.list'
    assert len(var_14.elts) == 3
    var_15 = []
    var_16 = 'targets'
    var_17 = {var_16: var_5, var_7: var_14}
    var_18 = module_1.Assign(*var_15, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Assign'
    assert f'{type(var_18.targets).__module__}.{type(var_18.targets).__qualname__}' == 'builtins.list'
    assert len(var_18.targets) == 1
    assert f'{type(var_18.value).__module__}.{type(var_18.value).__qualname__}' == 'ast.List'
    assert module_1.Assign.type_comment is None
    var_19 = 'module'
    var_0.globals(var_19, var_18)

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
    var_1 = '__all__'
    var_2 = var_0.__repr__()
    assert var_2 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_3 = var_0.parse(var_1, var_1)
    assert var_0.level == {'__all__': 0}
    assert var_0.doc == {'__all__': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'__all__': {*()}}
    assert var_0.root == {'__all__': '__all__'}
    var_4 = var_0.compile()
    assert var_4 == '\n'
    var_5 = var_0.load_docstring(var_1, var_3)
    var_6 = module_3.getdoc(var_5)
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
    var_7 = [var_6, var_6, var_6]
    var_8 = []
    var_9 = 'elts'
    var_10 = {var_9: var_7}
    var_11 = module_1.List(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.List'
    assert var_11.elts == [None, None, None]
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_12 = module_0.const_type(var_6)
    assert var_12 == 'Any'
    var_13 = module_0.const_type(var_11)
    assert var_13 == 'list'
    var_0.is_public(var_5)

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
    var_1 = 'var'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == 'var'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = 1
    var_7 = module_1.unparse(var_2)
    assert var_7 == ''
    var_8 = 'value'
    var_9 = {var_8: var_6}
    var_10 = module_1.Constant(*var_7, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == 1
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = []
    var_12 = 'target'
    var_13 = 'annotation'
    var_14 = 'value'
    var_15 = {var_12: var_5, var_13: var_5, var_14: var_10}
    var_16 = module_1.AnnAssign(*var_11, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.AnnAssign'
    assert f'{type(var_16.target).__module__}.{type(var_16.target).__qualname__}' == 'ast.Name'
    assert f'{type(var_16.annotation).__module__}.{type(var_16.annotation).__qualname__}' == 'ast.Name'
    assert f'{type(var_16.value).__module__}.{type(var_16.value).__qualname__}' == 'ast.Constant'
    assert module_1.AnnAssign.value is None
    var_17 = '!\nNM\r\\\\W )lr.fC(?{ '
    var_18 = var_0.globals(var_17, var_16)
    assert var_0.alias == {'!\nNM\r\\\\W )lr.fC(?{ .var': '1'}
    with pytest.raises(KeyError):
        var_19 = var_0.const['module.VAR']
    assert var_19 == 'int'

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
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == '__all__'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = 'value'
    var_8 = 'elts'
    var_9 = {var_8: var_2}
    var_10 = module_1.List(*var_2, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.List'
    assert var_10.elts == []
    var_11 = 'targets'
    var_12 = {var_11: var_6, var_1: var_4, var_7: var_10}
    var_13 = module_1.Assign(*var_2, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Assign'
    assert f'{type(var_13.targets).__module__}.{type(var_13.targets).__qualname__}' == 'builtins.list'
    assert len(var_13.targets) == 1
    assert f'{type(var_13.value).__module__}.{type(var_13.value).__qualname__}' == 'ast.List'
    assert module_1.Assign.type_comment is None
    var_14 = 'modul,'
    var_15 = var_0.globals(var_14, var_13)
    assert var_0.alias == {'modul,.__all__': '[]'}

@pytest.mark.xfail(strict=True)
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
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == '__all__'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = 'func'
    var_8 = 'value'
    var_9 = {var_8: var_7}
    var_10 = module_1.Constant(*var_2, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == 'func'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = [var_10]
    var_12 = []
    var_13 = 'elts'
    var_14 = {var_13: var_11}
    var_15 = module_1.List(*var_12, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.List'
    assert f'{type(var_15.elts).__module__}.{type(var_15.elts).__qualname__}' == 'builtins.list'
    assert len(var_15.elts) == 1
    var_16 = []
    var_17 = 'targets'
    var_18 = 'value'
    var_19 = {var_17: var_6, var_18: var_15}
    var_20 = module_1.Assign(*var_16, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.Assign'
    assert f'{type(var_20.targets).__module__}.{type(var_20.targets).__qualname__}' == 'builtins.list'
    assert len(var_20.targets) == 1
    assert f'{type(var_20.value).__module__}.{type(var_20.value).__qualname__}' == 'ast.List'
    assert module_1.Assign.type_comment is None
    var_21 = 'module'
    var_0.globals(var_21, var_20)

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
    var_1 = var_0.__post_init__()
    var_2 = [var_1]
    var_3 = 'func'
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 'func'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_8 = [var_7]
    var_9 = []
    var_10 = 'elts'
    var_11 = {var_10: var_8}
    var_12 = module_1.List(*var_9, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.List'
    assert f'{type(var_12.elts).__module__}.{type(var_12.elts).__qualname__}' == 'builtins.list'
    assert len(var_12.elts) == 1
    var_13 = []
    var_14 = 'targets'
    var_15 = 'value'
    var_16 = {var_14: var_2, var_15: var_12}
    var_17 = module_1.Assign(*var_13, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Assign'
    assert var_17.targets == [None]
    assert f'{type(var_17.value).__module__}.{type(var_17.value).__qualname__}' == 'ast.List'
    assert module_1.Assign.type_comment is None
    var_18 = 'module'
    var_19 = var_0.globals(var_18, var_17)
    with pytest.raises(KeyError):
        var_20 = bool('module.func' in var_0.imp['module'])
    assert var_20 is True

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
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == '__all__'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = var_0.__repr__()
    assert var_6 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_7 = []
    var_8 = 'value'
    var_9 = {var_8: var_1}
    var_10 = module_1.Constant(*var_7, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == '__all__'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = [var_10]
    var_12 = []
    var_13 = 'elts'
    var_14 = {var_13: var_11}
    var_15 = module_1.List(*var_12, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.List'
    assert f'{type(var_15.elts).__module__}.{type(var_15.elts).__qualname__}' == 'builtins.list'
    assert len(var_15.elts) == 1
    var_16 = []
    var_17 = 'targets'
    var_18 = 'value'
    var_19 = {var_17: var_6, var_18: var_15}
    var_20 = module_1.Assign(*var_16, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.Assign'
    assert var_20.targets == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    assert f'{type(var_20.value).__module__}.{type(var_20.value).__qualname__}' == 'ast.List'
    assert module_1.Assign.type_comment is None
    var_21 = 'module'
    var_22 = var_0.globals(var_21, var_20)
    with pytest.raises(KeyError):
        var_23 = bool('module.func' in var_0.imp['module'])
    assert var_23 is True

def test_case_36():
    var_0 = 'test.name'
    var_1 = 'test.alias'
    var_2 = {var_0: var_1}
    var_3 = module_0.Resolver(var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'test.alias'
    assert var_3.alias == {'test.name': 'test.alias'}
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
    assert var_7.value == 'test.name'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_8 = var_3.visit_Constant(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Attribute'
    assert f'{type(var_8.value).__module__}.{type(var_8.value).__qualname__}' == 'ast.Name'
    assert var_8.attr == 'name'
    assert f'{type(var_8.ctx).__module__}.{type(var_8.ctx).__qualname__}' == 'ast.Load'
    assert var_8.lineno == 1
    assert var_8.col_offset == 0
    assert var_8.end_lineno == 1
    assert var_8.end_col_offset == 9

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
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == '__all__'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = var_0.compile()
    assert var_7 == '\n'
    var_8 = "KYIv$<v<'n"
    var_9 = module_0.is_public_family(var_8)
    assert var_9 is True
    var_10 = module_0.esc_underscore(var_8)
    assert var_10 == "KYIv$<v<'n"
    var_11 = module_0.const_type(var_7)
    assert var_11 == 'Any'
    var_12 = []
    var_13 = module_1.Constant(*var_12, **var_4)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Constant'
    assert var_13.id == '__all__'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_14 = 'elts'
    var_15 = var_0.parse(var_7, var_11)
    assert var_0.level == {'\n': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n'}
    var_16 = {var_14: var_6}
    var_17 = module_1.List(*var_12, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.List'
    assert f'{type(var_17.elts).__module__}.{type(var_17.elts).__qualname__}' == 'builtins.list'
    assert len(var_17.elts) == 1
    var_18 = 'targets'
    var_19 = 'value'
    var_20 = {var_18: var_6, var_19: var_17}
    var_21 = None
    var_22 = var_0.resolve(var_21, var_17)
    assert var_22 == '[__all__]'
    var_23 = module_1.Assign(*var_12, **var_20)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'ast.Assign'
    assert f'{type(var_23.targets).__module__}.{type(var_23.targets).__qualname__}' == 'builtins.list'
    assert len(var_23.targets) == 1
    assert f'{type(var_23.value).__module__}.{type(var_23.value).__qualname__}' == 'ast.List'
    assert module_1.Assign.type_comment is None
    var_24 = var_0.globals(var_1, var_23)
    assert var_0.alias == {'__all__.__all__': '[__all__]'}
    with pytest.raises(AttributeError):
        var_25 = bool('module.func' in var_12.imp['module'])
    assert var_25 is True

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
    var_1 = 'var'
    var_2 = 'id'
    var_3 = {var_2: var_1}
    var_4 = var_0.__post_init__()
    var_5 = 'int'
    var_6 = []
    var_7 = module_1.Name(*var_6, **var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Name'
    assert var_7.id == 'var'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = 1
    var_9 = []
    var_10 = {var_5: var_8, var_1: var_8, var_1: var_8, var_1: var_8}
    var_11 = module_1.Constant(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Constant'
    assert var_11.int == 1
    assert var_11.var == 1
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_12 = []
    var_13 = 'target'
    var_14 = 'value'
    var_15 = {var_13: var_4, var_14: var_7, var_14: var_11}
    var_16 = module_1.AnnAssign(*var_12, **var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.AnnAssign'
    assert var_16.target is None
    assert f'{type(var_16.value).__module__}.{type(var_16.value).__qualname__}' == 'ast.Constant'
    assert module_1.AnnAssign.value is None
    var_17 = 'moUule'
    var_18 = var_0.globals(var_17, var_16)
    with pytest.raises(KeyError):
        var_19 = var_0.alias['module.var']
    assert var_19 == '1'

def test_case_39():
    var_0 = '__magic__'
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

def test_case_40():
    var_0 = 'module._private_function'
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

def test_case_41():
    var_0 = 'test'
    var_1 = {}
    var_2 = module_0.Resolver(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Resolver'
    assert var_2.root == 'test'
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
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.Constant(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 'invalid syntax'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_8 = var_2.visit_Constant(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Constant'
    assert var_8.value == 'invalid syntax'
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = '47q4{H\nXt#\\'
    var_1 = 'uT-l68i|K6%VUQsZ~a'
    var_2 = 'x/dnq'
    var_3 = {var_1: var_1, var_1: var_0, var_0: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Resolver'
    assert var_4.root == '47q4{H\nXt#\\'
    assert var_4.alias == {'uT-l68i|K6%VUQsZ~a': '47q4{H\nXt#\\', '47q4{H\nXt#\\': 'x/dnq'}
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
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_6 = '__all__'
    var_7 = []
    var_8 = 'id'
    var_9 = {var_8: var_6}
    var_10 = module_1.Name(*var_7, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Name'
    assert var_10.id == '__all__'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_11 = [var_10]
    var_12 = var_5.compile()
    assert var_12 == '\n'
    var_13 = "KYIv$<v<'n"
    var_14 = module_0.is_public_family(var_13)
    assert var_14 is True
    var_15 = '>b>$fc'
    var_16 = '@3{xo\\.)hm*wl{X`'
    var_17 = var_5.resolve(var_16, var_10)
    assert var_17 == '__all__'
    var_18 = {var_13: var_15}
    var_19 = module_1.Constant(*var_11, **var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.Constant'
    assert f'{type(var_19.value).__module__}.{type(var_19.value).__qualname__}' == 'ast.Name'
    assert var_19.KYIv$<v<'n == '>b>$fc'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_20 = [var_19]
    var_21 = []
    var_22 = 'elts'
    var_23 = {var_22: var_20}
    var_24 = module_1.List(*var_21, **var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'ast.List'
    assert f'{type(var_24.elts).__module__}.{type(var_24.elts).__qualname__}' == 'builtins.list'
    assert len(var_24.elts) == 1
    var_25 = var_5.compile()
    assert var_25 == '\n'
    var_26 = []
    var_27 = 'targets'
    var_28 = 'vy{u~We'
    var_29 = {var_27: var_11, var_28: var_24}
    var_30 = None
    var_31 = var_5.resolve(var_30, var_24)
    var_32 = module_1.Assign(*var_26, **var_29)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'ast.Assign'
    assert f'{type(var_32.targets).__module__}.{type(var_32.targets).__qualname__}' == 'builtins.list'
    assert len(var_32.targets) == 1
    assert f'{type(var_32.vy{u~We).__module__}.{type(var_32.vy{u~We).__qualname__}' == 'ast.List'
    assert module_1.Assign.type_comment is None
    var_33 = 'module'
    var_5.globals(var_33, var_32)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 'GQ'
    var_1 = False
    var_2 = '~Gnv>3#z(?@LU'
    var_3 = 'SAw>Mu|,'
    var_4 = {var_2: var_3, var_0: var_2, var_2: var_3}
    var_5 = module_0.Parser(var_1, doc=var_2, alias=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is False
    assert var_5.b_level == 1
    assert var_5.toc is False
    assert var_5.level == {}
    assert var_5.doc == '~Gnv>3#z(?@LU'
    assert var_5.docstring == {}
    assert var_5.imp == {}
    assert var_5.root == {}
    assert var_5.alias == {'~Gnv>3#z(?@LU': 'SAw>Mu|,', 'GQ': '~Gnv>3#z(?@LU'}
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

def test_case_44():
    var_0 = 'test'
    var_1 = 'test.name'
    var_2 = 'test.alias'
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Resolver'
    assert var_4.root == 'test'
    assert var_4.alias == {'test.name': 'test.alias'}
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
    var_6 = []
    var_7 = 'value'
    var_8 = {var_7: var_5}
    var_9 = module_1.Constant(*var_6, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Constant'
    assert var_9.value == 'name'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_10 = var_4.visit_Constant(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Attribute'
    assert f'{type(var_10.value).__module__}.{type(var_10.value).__qualname__}' == 'ast.Name'
    assert var_10.attr == 'alias'
    assert f'{type(var_10.ctx).__module__}.{type(var_10.ctx).__qualname__}' == 'ast.Load'
    assert var_10.lineno == 1
    assert var_10.col_offset == 0
    assert var_10.end_lineno == 1
    assert var_10.end_col_offset == 10

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
    var_2 = 'id'
    var_3 = {var_2: var_2}
    var_4 = None
    var_5 = module_0.walk_body(var_4)
    var_6 = module_0.table(items=var_3)
    assert var_6 == '||\n||\n| id |\n\n'
    var_7 = module_1.Name(*var_1, **var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Name'
    assert var_7.id == 'id'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = var_0.__repr__()
    assert var_8 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_9 = var_0.compile()
    assert var_9 == '\n'
    var_10 = ')1]t'
    var_11 = module_0.is_public_family(var_10)
    assert var_11 is True
    var_12 = module_0.esc_underscore(var_10)
    assert var_12 == ')1]t'
    var_13 = module_0.doctest(var_2)
    assert var_13 == 'id'
    var_14 = []
    var_15 = 'value'
    var_16 = {var_15: var_9}
    var_17 = module_1.Constant(*var_14, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Constant'
    assert var_17.value == '\n'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_18 = [var_17]
    var_19 = []
    var_20 = 'elts'
    var_21 = {var_20: var_18}
    var_22 = module_1.List(*var_19, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.List'
    assert f'{type(var_22.elts).__module__}.{type(var_22.elts).__qualname__}' == 'builtins.list'
    assert len(var_22.elts) == 1
    var_23 = None
    var_24 = var_0.load_docstring(var_23, var_23)
    var_25 = None
    var_0.resolve(var_25, var_22)

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
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_2}
    var_4 = module_1.Name(*var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Name'
    assert var_4.id == 'id'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = [var_4, var_4, var_4]
    var_6 = 'targets'
    var_7 = {var_6: var_5}
    var_8 = module_1.Delete(*var_1, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Delete'
    assert f'{type(var_8.targets).__module__}.{type(var_8.targets).__qualname__}' == 'builtins.list'
    assert len(var_8.targets) == 3
    var_9 = [var_8, var_8, var_8, var_8, var_8]
    var_10 = var_0.class_api(var_6, var_6, var_1, var_9)

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
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == '__all__'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = 'func'
    var_8 = 'o_qZAH\\J'
    var_9 = None
    var_10 = var_0.load_docstring(var_8, var_9)
    var_11 = 'value'
    var_12 = {var_11: var_7}
    var_13 = var_0.compile()
    assert var_13 == '\n'
    var_14 = None
    var_15 = module_1.arguments(*var_6)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.arguments'
    assert f'{type(var_15.posonlyargs).__module__}.{type(var_15.posonlyargs).__qualname__}' == 'ast.Name'
    assert module_1.arguments.vararg is None
    assert module_1.arguments.kwarg is None
    var_0.func_api(var_14, var_14, var_15, var_9, has_self=var_12, cls_method=var_14)

def test_case_48():
    var_0 = 'root'
    var_1 = {}
    var_2 = 'T'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'root'
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
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_2, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Name'
    assert var_11.id == 'T'
    assert f'{type(var_11.ctx).__module__}.{type(var_11.ctx).__qualname__}' == 'ast.Load'
    var_12 = var_3.visit_Name(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Name'
    assert var_12.id == 'Self'
    assert f'{type(var_12.ctx).__module__}.{type(var_12.ctx).__qualname__}' == 'ast.Load'
    var_13 = var_12.id
    assert var_13 == 'Self'

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = 'root'
    var_1 = 'root.T'
    var_2 = "typing.TypeVar('T')"
    var_3 = {var_1: var_2}
    var_4 = module_0.Resolver(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Resolver'
    assert var_4.root == 'root'
    assert var_4.alias == {'root.T': "typing.TypeVar('T')"}
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
    var_9 = 'id'
    var_10 = {var_9: var_5, var_0: var_8}
    var_11 = module_1.Name(*var_6, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Name'
    assert var_11.id == 'T'
    assert f'{type(var_11.root).__module__}.{type(var_11.root).__qualname__}' == 'ast.Load'
    var_4.visit_Name(var_11)

def test_case_50():
    var_0 = ">>> print('hello')"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "```python\n>>> print('hello')\n```"
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    with pytest.raises(TypeError):
        module_1.get_docstring(var_2)

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
    var_1 = 'test_odul'
    var_2 = []
    var_3 = 'att'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Name'
    assert var_7.id == 'att'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = 'O3o::o@38Q'
    var_9 = var_0.resolve(var_8, var_7)
    assert var_9 == 'att'
    var_10 = var_0.__repr__()
    assert var_10 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_11 = 'targets'
    var_12 = {var_11: var_10}
    var_13 = module_1.Delete(*var_4, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Delete'
    assert var_13.targets == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_14 = 'attrh'
    var_15 = [var_1]
    var_16 = {var_3: var_14}
    var_17 = module_1.Name(*var_15, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Name'
    assert var_17.id == 'test_odul'
    assert var_17.att == 'attrh'
    var_18 = 'targets'
    var_19 = var_0.__repr__()
    assert var_19 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_20 = var_0.load_docstring(var_1, var_19)
    var_21 = [var_13, var_13, var_13, var_13]
    var_22 = var_0.class_api(var_1, var_18, var_2, var_21)
    with pytest.raises(KeyError):
        var_23 = bool('attr1' not in var_0.doc[var_14])
    assert var_23 is True

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
    var_1 = '__all__'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == '__all__'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = None
    var_8 = var_0.parse(var_1, var_3)
    assert var_0.level == {'__all__': 0}
    assert var_0.doc == {'__all__': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'__all__': {*()}}
    assert var_0.root == {'__all__': '__all__'}
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_7}
    var_12 = var_0.compile()
    assert var_12 == '\n'
    var_13 = None
    var_14 = module_0.const_type(var_13)
    assert var_14 == 'Any'
    var_15 = module_1.Constant(*var_9, **var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Constant'
    assert var_15.value is None
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_16 = '.473!h'
    var_17 = var_0.func_ann(var_16, var_7, has_self=var_13, cls_method=var_13)
    var_18 = [var_15, var_15, var_15]
    var_19 = []
    var_20 = 'elts'
    var_21 = {var_20: var_18}
    var_22 = module_1.List(*var_19, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.List'
    assert f'{type(var_22.elts).__module__}.{type(var_22.elts).__qualname__}' == 'builtins.list'
    assert len(var_22.elts) == 3
    var_23 = 'targets'
    var_24 = 'value'
    var_25 = {var_23: var_6, var_24: var_22}
    var_26 = var_0.__post_init__()
    var_27 = module_1.Assign(*var_19, **var_25)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'ast.Assign'
    assert f'{type(var_27.targets).__module__}.{type(var_27.targets).__qualname__}' == 'builtins.list'
    assert len(var_27.targets) == 1
    assert f'{type(var_27.value).__module__}.{type(var_27.value).__qualname__}' == 'ast.List'
    assert module_1.Assign.type_comment is None
    var_28 = module_0.is_magic(var_20)
    assert var_28 is False
    var_29 = '^Q\\;[O'
    var_30 = True
    var_31 = var_0.func_ann(var_29, var_13, has_self=var_30, cls_method=var_13)
    var_32 = 'modu'
    var_33 = var_0.globals(var_32, var_27)
    assert var_0.alias == {'modu.__all__': '[None, None, None]'}

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
    var_1 = module_1.Dict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'ast.Dict'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    module_0.const_type(var_1)

def test_case_54():
    var_0 = 'bl'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.Name(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Name'
    assert var_3.id == 'bl'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_4 = []
    var_5 = [var_3, var_4]
    var_6 = module_1.Call(*var_5, **var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Call'
    assert f'{type(var_6.func).__module__}.{type(var_6.func).__qualname__}' == 'ast.Name'
    assert var_6.args == []
    var_7 = module_0.const_type(var_6)
    assert var_7 == 'Any'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_55():
    var_0 = 2
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.Constant(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Constant'
    assert var_3.value == 2
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_4 = 'a'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_1.Constant(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Constant'
    assert var_7.value == 'a'
    var_8 = [var_7, var_7]
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_1.Tuple(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_11.elts).__module__}.{type(var_11.elts).__qualname__}' == 'builtins.list'
    assert len(var_11.elts) == 2
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_12 = module_0.const_type(var_11)
    assert var_12 == 'tuple[str]'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_13 = [var_0]
    var_14 = {}
    var_15 = module_1.Constant(*var_13, **var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Constant'
    assert var_15.value == 2
    var_16 = [var_4]
    var_17 = {}
    var_18 = module_1.Constant(*var_16, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Constant'
    assert var_18.value == 'a'
    var_19 = [var_15, var_18]
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_1.Tuple(*var_20, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Tuple'
    assert f'{type(var_22.elts).__module__}.{type(var_22.elts).__qualname__}' == 'builtins.list'
    assert len(var_22.elts) == 2
    var_23 = module_0.const_type(var_22)
    assert var_23 == 'tuple[Any]'

def test_case_56():
    var_0 = "Regular line\n>>> print('hello')\n>>> print('world')\nAnother regular line"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "Regular line\n```python\n>>> print('hello')\n>>> print('world')\n```\nAnother regular line"
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'

def test_case_57():
    var_0 = 'root'
    var_1 = 'root.Name'
    var_2 = {var_1: var_1}
    var_3 = module_0.Resolver(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'root'
    assert var_3.alias == {'root.Name': 'root.Name'}
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
    var_4 = 'Name'
    var_5 = []
    var_6 = {}
    var_7 = module_1.Load(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = []
    var_9 = 'id'
    var_10 = 'ctx'
    var_11 = {var_9: var_4, var_10: var_7}
    var_12 = module_1.Name(*var_8, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Name'
    assert var_12.id == 'Name'
    assert f'{type(var_12.ctx).__module__}.{type(var_12.ctx).__qualname__}' == 'ast.Load'
    var_13 = var_3.visit_Name(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Name'
    assert var_13.id == 'Name'
    assert f'{type(var_13.ctx).__module__}.{type(var_13.ctx).__qualname__}' == 'ast.Load'
    var_14 = var_13.id

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
    var_7 = 1
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_7}
    var_11 = module_1.Constant(*var_8, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Constant'
    assert var_11.value == 1
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
    var_19 = var_0.globals(var_9, var_18)
    assert var_0.alias == {'value.x': '1'}

def test_case_59():
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
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = None
    var_3 = ' $rF4!G3q"<Vl'
    var_4 = {var_3: var_2, var_3: var_2, var_3: var_2, var_3: var_2}
    var_5 = module_1.ImportFrom(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.ImportFrom'
    assert var_5. $rF4!G3q"<Vl is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_6 = var_0.imports(var_2, var_5)
    var_7 = 'attrh'
    var_8 = var_0.__repr__()
    assert var_8 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_9 = var_0.load_docstring(var_7, var_8)
    with pytest.raises(KeyError):
        var_10 = bool('attr2' not in var_0.doc[var_9])
    assert var_10 is True

def test_case_61():
    var_0 = []
    var_1 = [var_0, var_0]
    var_2 = {}
    var_3 = module_1.If(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.If'
    assert var_3.test == []
    assert var_3.body == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_4 = [var_3]
    var_5 = module_0.walk_body(var_4)
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
        var_6 = list(var_5)

@pytest.mark.xfail(strict=True)
def test_case_62():
    var_0 = None
    var_1 = [var_0]
    var_2 = module_1.Attribute(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Attribute'
    assert var_2.value is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = 'fD{Q<PYB{-7ZOg/0_I'
    var_4 = None
    var_5 = module_0.Resolver(var_3, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Resolver'
    assert var_5.root == 'fD{Q<PYB{-7ZOg/0_I'
    assert var_5.alias is None
    assert var_5.self_ty is None
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_6 = var_5.visit_Attribute(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Attribute'
    assert var_6.value is None
    var_7 = None
    var_8 = module_1.Call()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Call'
    var_9 = 'Hw",]\n\x0bG\'z'
    var_10 = module_0.walk_body(var_8)
    var_11 = None
    var_12 = None
    var_13 = module_2.field(init=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'dataclasses.Field'
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
    var_14 = module_0.Parser(b_level=var_13, imp=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'apimd.parser.Parser'
    assert var_14.link is True
    assert f'{type(var_14.b_level).__module__}.{type(var_14.b_level).__qualname__}' == 'dataclasses.Field'
    assert var_14.toc is False
    assert var_14.level == {}
    assert var_14.doc == {}
    assert var_14.docstring == {}
    assert f'{type(var_14.imp).__module__}.{type(var_14.imp).__qualname__}' == 'dataclasses.Field'
    assert var_14.root == {}
    assert var_14.alias == {}
    assert var_14.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_15 = var_14.__eq__(var_11)
    var_15.resolve(var_7, var_8, var_9)

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
    var_1 = '"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass': 2, '"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass.foo': 2}
    assert var_0.doc == {'"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass': '## Module `{}`\n<a id="{}"></a>\n\n', '"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass': 'Module docstring.', '"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass.foo': 'Function docstring.'}
    assert var_0.imp == {'"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass': {*()}}
    assert var_0.root == {'"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass': '"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass', '"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass.foo': '"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass'}
    var_3 = bool('test_module' in var_0.docstring)
    assert var_3 is True
    with pytest.raises(KeyError):
        var_4 = bool('Module docstring.' in var_0.docstring['test_module'])
    assert var_4 is True

def test_case_64():
    var_0 = False
    var_1 = module_0.Parser(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is False
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
    var_2 = 'bY`,Gx\r.QFv8lp,.'
    var_3 = 'def foo():\n    pass'
    var_4 = var_1.parse(var_2, var_3)
    assert var_1.level == {'bY`,Gx\r.QFv8lp,.': 2, 'bY`,Gx\r.QFv8lp,..foo': 2}
    assert var_1.doc == {'bY`,Gx\r.QFv8lp,.': '## Module `{}`\n\n', 'bY`,Gx\r.QFv8lp,..foo': '### foo()\n\n*Full name:* `{}`\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_1.imp == {'bY`,Gx\r.QFv8lp,.': {*()}}
    assert var_1.root == {'bY`,Gx\r.QFv8lp,.': 'bY`,Gx\r.QFv8lp,.', 'bY`,Gx\r.QFv8lp,..foo': 'bY`,Gx\r.QFv8lp,.'}
    with pytest.raises(KeyError):
        var_5 = bool('<a id="test_module"></a>' in var_1.doc['test_module'])
    assert var_5 is True

def test_case_65():
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
    var_2 = var_1.__repr__()
    assert var_2 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_3 = 'def foo():\n    pass'
    var_4 = var_1.parse(var_3, var_3)
    assert var_1.level == {'def foo():\n    pass': 0, 'def foo():\n    pass.foo': 0}
    assert var_1.doc == {'def foo():\n    pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo():\n    pass.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_1.imp == {'def foo():\n    pass': {*()}}
    assert var_1.root == {'def foo():\n    pass': 'def foo():\n    pass', 'def foo():\n    pass.foo': 'def foo():\n    pass'}

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
    var_1 = 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass': 0, 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo': 0, 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.bar': 0, 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.baz': 0}
    assert var_0.doc == {'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo': '### class Foo\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.bar': '#### Foo.bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.baz': '#### Foo.baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass': {*()}}
    assert var_0.root == {'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass': 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo': 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.bar': 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass', 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass.Foo.baz': 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass'}

@pytest.mark.xfail(strict=True)
def test_case_67():
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
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
    var_2 = 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module': 0, 'test_module.Foo': 0, 'test_module.Foo.bar': 0, 'test_module.Foo.baz': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.Foo': '### class Foo\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.Foo.bar': '#### Foo.bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', 'test_module.Foo.baz': '#### Foo.baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.Foo': 'test_module', 'test_module.Foo.bar': 'test_module', 'test_module.Foo.baz': 'test_module'}
    var_4 = None
    var_5 = [var_4, var_4]
    var_6 = module_1.AsyncFunctionDef(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.AsyncFunctionDef'
    assert var_6.name is None
    assert var_6.args is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.AsyncFunctionDef.returns is None
    assert module_1.AsyncFunctionDef.type_comment is None
    var_0.api(var_1, var_6)

def test_case_68():
    var_0 = []
    var_1 = {}
    var_2 = module_1.Try(*var_0, **var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.Try'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_3 = [var_2, var_2, var_2]
    var_4 = module_0.walk_body(var_3)
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
        var_5 = list(var_4)

@pytest.mark.xfail(strict=True)
def test_case_69():
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'class Foo:\n    def bar(self):\n        pass\n    @staticmethod\n    def baz():\n        pass'
    var_2 = var_0.__repr__()
    assert var_2 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_3 = module_3.getdoc(var_2)
    assert var_3 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
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
    var_4 = var_0.parse(var_3, var_1)
    assert var_0.level == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": 7, "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..Foo": 7, "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..Foo.bar": 7, "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..Foo.baz": 7}
    assert var_0.doc == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": '## Module `{}`\n<a id="{}"></a>\n\n', "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..Foo": '### class Foo\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..Foo.bar": '#### Foo.bar()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `Any` |\n\n', "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..Foo.baz": '#### Foo.baz()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": {*()}}
    assert var_0.root == {"str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..Foo": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..Foo.bar": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'.", "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'..Foo.baz": "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."}
    var_5 = bool('test_module.Foo' in var_0.doc)
    var_6 = var_0.compile()
    assert var_6 == '\n'
    var_7 = bool('test_module.Foo.baz' in var_0.doc)
    var_8 = bool('test_module.Foo.bar' in var_0.root)
    var_2.class_api(var_6, var_4, var_8, var_5)

def test_case_70():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_1.Name(*var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Name'
    assert var_4.id == 'x'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = []
    var_6 = {}
    var_7 = module_1.BitOr(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.BitOr'
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Name'
    assert var_12.id == 'y'
    var_13 = []
    var_14 = 'left'
    var_15 = 'op'
    var_16 = 'right'
    var_17 = {var_14: var_4, var_15: var_7, var_16: var_12}
    var_18 = module_1.BinOp(*var_13, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.BinOp'
    assert f'{type(var_18.left).__module__}.{type(var_18.left).__qualname__}' == 'ast.Name'
    assert f'{type(var_18.op).__module__}.{type(var_18.op).__qualname__}' == 'ast.BitOr'
    assert f'{type(var_18.right).__module__}.{type(var_18.right).__qualname__}' == 'ast.Name'
    var_19 = [var_18]
    var_20 = module_0._defaults(var_19)
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_21 = list(var_20)
    var_22 = bool(var_21 == ['`x &#124; y`'])

def test_case_71():
    var_0 = 'bl'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_1.Name(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Name'
    assert var_3.id == 'bl'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_4 = {}
    var_5 = module_1.Call(*var_1, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Call'
    assert var_5.func == 'bl'
    var_6 = module_0.const_type(var_5)
    assert var_6 == 'Any'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_7 = '{=+)4d]?[Ojg\x0c Uu'
    var_8 = module_0.is_magic(var_7)
    assert var_8 is False

def test_case_72():
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = 'H2'
    var_2 = []
    var_3 = 'id'
    var_4 = {var_3: var_1}
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == 'H2'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = 'func'
    var_8 = 'value'
    var_9 = {var_8: var_7}
    var_10 = module_1.Constant(*var_2, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    assert var_10.value == 'func'
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_11 = [var_10]
    var_12 = 'elts'
    var_13 = {var_12: var_11}
    var_14 = module_1.List(*var_2, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.List'
    assert f'{type(var_14.elts).__module__}.{type(var_14.elts).__qualname__}' == 'builtins.list'
    assert len(var_14.elts) == 1
    var_15 = []
    var_16 = 'targets'
    var_17 = {var_16: var_6, var_8: var_14}
    var_18 = module_1.Assign(*var_15, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Assign'
    assert f'{type(var_18.targets).__module__}.{type(var_18.targets).__qualname__}' == 'builtins.list'
    assert len(var_18.targets) == 1
    assert f'{type(var_18.value).__module__}.{type(var_18.value).__qualname__}' == 'ast.List'
    assert module_1.Assign.type_comment is None
    var_19 = 'module'
    var_20 = var_0.globals(var_19, var_18)
    assert var_0.root == {'module.H2': 'module'}
    assert var_0.alias == {'module.H2': "['func']"}
    assert var_0.const == {'module.H2': 'list[str]'}
    with pytest.raises(KeyError):
        var_21 = bool('module.func' in var_0.imp['module'])
    assert var_21 is True

def test_case_73():
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
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
    var_2 = '"""Module docstring."""\ndef foo():\n    """Function docstring."""\n    pass'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module': 0, 'test_module.foo': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'test_module': 'Module docstring.', 'test_module.foo': 'Function docstring.'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.foo': 'test_module'}
    var_4 = var_0.compile()
    assert var_4 == '## Module `test_module`\n<a id="test_module"></a>\n\nModule docstring.\n\n### foo()\n\n*Full name:* `test_module.foo`\n<a id="test_module-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n\nFunction docstring.\n'
    var_5 = bool('test_module' in var_0.docstring)
    assert var_5 is True
    var_6 = bool('Module docstring.' in var_0.docstring['test_module'])
    assert var_6 is True
    var_7 = bool('test_module.foo' in var_0.docstring)
    assert var_7 is True
    var_8 = bool('Function docstring.' in var_0.docstring['test_module.foo'])
    assert var_8 is True

def test_case_74():
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
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
    var_5 = module_1.Load(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Name'
    assert var_10.id == 'int'
    assert f'{type(var_10.ctx).__module__}.{type(var_10.ctx).__qualname__}' == 'ast.Load'
    var_11 = [var_1, var_10]
    var_12 = {}
    var_13 = module_1.arg(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.arg'
    assert var_13.arg == 'x'
    assert f'{type(var_13.annotation).__module__}.{type(var_13.annotation).__qualname__}' == 'ast.Name'
    assert module_1.arg.annotation is None
    assert module_1.arg.type_comment is None
    assert module_1.arg.end_lineno is None
    assert module_1.arg.end_col_offset is None
    var_14 = '*'
    var_15 = None
    var_16 = [var_14, var_15]
    var_17 = {}
    var_18 = module_1.arg(*var_16, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.arg'
    assert var_18.arg == '*'
    assert var_18.annotation is None
    var_19 = 'y'
    var_20 = []
    var_21 = {}
    var_22 = module_1.Load(*var_20, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Load'
    var_23 = [var_19, var_10]
    var_24 = {}
    var_25 = module_1.arg(*var_23, **var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'ast.arg'
    assert var_25.arg == 'y'
    assert f'{type(var_25.annotation).__module__}.{type(var_25.annotation).__qualname__}' == 'ast.Name'
    var_26 = [var_13, var_18, var_25]
    var_27 = 'module'
    var_28 = False
    var_29 = var_0.func_ann(var_27, var_26, has_self=var_28, cls_method=var_28)
    var_30 = list(var_29)
    var_31 = bool(var_30 == ['int', '', 'str'])

def test_case_75():
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
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
    var_2 = 'SomeClass'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = 'id'
    var_7 = 'ctx'
    var_8 = {var_6: var_2, var_7: var_5}
    var_9 = module_1.Name(*var_3, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Name'
    assert var_9.id == 'SomeClass'
    assert f'{type(var_9.ctx).__module__}.{type(var_9.ctx).__qualname__}' == 'ast.Load'
    var_10 = [var_1, var_9]
    var_11 = module_1.arg(*var_10, **var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.arg'
    assert var_11.arg == 'cls'
    assert f'{type(var_11.annotation).__module__}.{type(var_11.annotation).__qualname__}' == 'ast.Name'
    assert var_11.id == 'SomeClass'
    assert f'{type(var_11.ctx).__module__}.{type(var_11.ctx).__qualname__}' == 'ast.Load'
    assert module_1.arg.annotation is None
    assert module_1.arg.type_comment is None
    assert module_1.arg.end_lineno is None
    assert module_1.arg.end_col_offset is None
    var_12 = []
    var_13 = {}
    var_14 = module_1.Load(*var_12, **var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Load'
    var_15 = []
    var_16 = module_1.Name(*var_15, **var_13)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Name'
    var_17 = {}
    var_18 = module_1.arg(*var_3, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.arg'
    var_19 = [var_11, var_18]
    var_20 = 'module'
    var_21 = True
    var_22 = var_0.func_ann(var_20, var_19, has_self=var_21, cls_method=var_21)
    with pytest.raises(AttributeError):
        var_23 = list(var_22)

def test_case_76():
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
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
    var_2 = 'SomeClass'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_2, var_2: var_5}
    var_9 = module_1.Name(*var_6, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Name'
    assert var_9.id == 'SomeClass'
    assert f'{type(var_9.SomeClass).__module__}.{type(var_9.SomeClass).__qualname__}' == 'ast.Load'
    var_10 = [var_1, var_9]
    var_11 = {}
    var_12 = module_1.arg(*var_10, **var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.arg'
    assert var_12.arg == 'cls'
    assert f'{type(var_12.annotation).__module__}.{type(var_12.annotation).__qualname__}' == 'ast.Name'
    assert module_1.arg.annotation is None
    assert module_1.arg.type_comment is None
    assert module_1.arg.end_lineno is None
    assert module_1.arg.end_col_offset is None
    var_13 = 'x'
    var_14 = 'int'
    var_15 = []
    var_16 = {}
    var_17 = module_1.Load(*var_15, **var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Load'
    var_18 = []
    var_19 = 'id'
    var_20 = 'ctx'
    var_21 = {var_19: var_14, var_20: var_17}
    var_22 = module_1.Name(*var_18, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Name'
    assert var_22.id == 'int'
    assert f'{type(var_22.ctx).__module__}.{type(var_22.ctx).__qualname__}' == 'ast.Load'
    var_23 = [var_13, var_22]
    var_24 = {}
    var_25 = module_1.arg(*var_23, **var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'ast.arg'
    assert var_25.arg == 'x'
    assert f'{type(var_25.annotation).__module__}.{type(var_25.annotation).__qualname__}' == 'ast.Name'
    var_26 = [var_12, var_25]
    var_27 = True
    var_28 = var_0.func_ann(var_14, var_26, has_self=var_27, cls_method=var_27)
    var_29 = list(var_28)
    var_30 = bool(var_29 == ['type[Self]', 'int'])
    assert var_30 is True

def test_case_77():
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

def test_case_78():
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
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
    var_2 = 'SomeClass'
    var_3 = []
    var_4 = {}
    var_5 = module_1.Load(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Load'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = []
    var_7 = 'id'
    var_8 = 'ctx'
    var_9 = {var_7: var_2, var_8: var_5}
    var_10 = module_1.Name(*var_6, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Name'
    assert var_10.id == 'SomeClass'
    assert f'{type(var_10.ctx).__module__}.{type(var_10.ctx).__qualname__}' == 'ast.Load'
    var_11 = [var_1, var_10]
    var_12 = {}
    var_13 = module_1.arg(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.arg'
    assert var_13.arg == 'self'
    assert f'{type(var_13.annotation).__module__}.{type(var_13.annotation).__qualname__}' == 'ast.Name'
    assert module_1.arg.annotation is None
    assert module_1.arg.type_comment is None
    assert module_1.arg.end_lineno is None
    assert module_1.arg.end_col_offset is None
    var_14 = 'x'
    var_15 = 'int'
    var_16 = []
    var_17 = {}
    var_18 = module_1.Load(*var_16, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Load'
    var_19 = []
    var_20 = 'ctx'
    var_21 = {var_15: var_15, var_20: var_18}
    var_22 = module_1.Name(*var_19, **var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Name'
    assert var_22.int == 'int'
    assert f'{type(var_22.ctx).__module__}.{type(var_22.ctx).__qualname__}' == 'ast.Load'
    var_23 = [var_14, var_22]
    var_24 = {}
    var_25 = module_1.arg(*var_23, **var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'ast.arg'
    assert var_25.arg == 'x'
    assert f'{type(var_25.annotation).__module__}.{type(var_25.annotation).__qualname__}' == 'ast.Name'
    var_26 = [var_13, var_25]
    var_27 = None
    var_28 = var_0.__eq__(var_27)
    var_29 = 'module'
    var_30 = True
    var_31 = False
    var_32 = var_0.func_ann(var_29, var_26, has_self=var_30, cls_method=var_31)
    with pytest.raises(AttributeError):
        var_33 = list(var_32)

def test_case_79():
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
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
    var_5 = module_1.Name(*var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Name'
    assert var_5.id == '__all__'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = [var_5]
    var_7 = None
    var_8 = var_0.parse(var_1, var_3)
    assert var_0.level == {'__all__': 0}
    assert var_0.doc == {'__all__': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'__all__': {*()}}
    assert var_0.root == {'__all__': '__all__'}
    var_9 = []
    var_10 = 'value'
    var_11 = {var_10: var_7}
    var_12 = var_0.compile()
    assert var_12 == '\n'
    var_13 = None
    var_14 = module_0.const_type(var_7)
    assert var_14 == 'Any'
    var_15 = module_0.code(var_3)
    assert var_15 == '`id`'
    var_16 = module_0.const_type(var_13)
    assert var_16 == 'Any'
    var_17 = module_1.Constant(*var_9, **var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Constant'
    assert var_17.value is None
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_18 = var_0.__post_init__()
    var_19 = []
    var_20 = 'elts'
    var_21 = 'vP9%sq]3V'
    var_22 = var_0.load_docstring(var_21, var_7)
    var_23 = module_0.doctest(var_10)
    assert var_23 == 'value'
    var_24 = 'targets'
    var_25 = 'value'
    var_26 = {var_24: var_6, var_25: var_2}
    var_27 = var_0.__post_init__()
    var_28 = module_1.Assign(*var_19, **var_26)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'ast.Assign'
    assert f'{type(var_28.targets).__module__}.{type(var_28.targets).__qualname__}' == 'builtins.list'
    assert len(var_28.targets) == 1
    assert var_28.value == []
    assert module_1.Assign.type_comment is None
    var_29 = module_0.is_magic(var_20)
    assert var_29 is False
    var_30 = '^Q\\;['
    var_31 = True
    var_32 = var_0.func_ann(var_30, var_13, has_self=var_31, cls_method=var_13)
    var_33 = 'modu'
    var_34 = var_0.globals(var_33, var_28)
    assert var_0.alias == {'modu.__all__': ''}