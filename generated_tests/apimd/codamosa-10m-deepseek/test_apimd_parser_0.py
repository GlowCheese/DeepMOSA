# Check out: https://github.com/GlowCheese/deepmosa
import ast as module_1
import inspect as module_2

import apimd.parser as module_0
import pytest


def test_case_0():
    var_0 = 'wH*>zbG:H'
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
    var_0 = '%'
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
    var_1 = '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': 2, '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': 2, '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': 2, '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': 2}
    assert var_0.doc == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': '## Module `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\n', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': '### class Example\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': '#### Example.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `list[int]` | `None` |\n\n'}
    assert var_0.docstring == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': 'Example function.', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': 'Example class.'}
    assert var_0.imp == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': {*()}}
    assert var_0.root == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '}
    assert var_0.alias == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .os': 'os', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .List': 'typing.List', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': '42'}
    assert var_0.const == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': 'int'}

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '#\\eW*jb|\\i'
    var_1 = '5\x0cPX[/~mz{x:'
    var_2 = module_0.esc_underscore(var_1)
    assert var_2 == '5\x0cPX[/~mz{x:'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_3 = None
    var_4 = 'L_3\x0cy)$=~(d\x0b;("`K+'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True
    var_6 = module_0.const_type(var_0)
    assert var_6 == 'Any'
    var_7 = module_1.Import()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Import'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = module_0.Parser(toc=var_5, docstring=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is True
    assert var_8.b_level == 1
    assert var_8.toc is True
    assert var_8.level == {}
    assert var_8.doc == {}
    assert var_8.docstring is None
    assert var_8.imp == {}
    assert var_8.root == {}
    assert var_8.alias == {}
    assert var_8.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_8.imports(var_2, var_7)

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
    var_1 = var_0.__repr__()
    assert var_1 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 0}
    assert var_0.doc == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': {*()}}
    assert var_0.root == {'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})': 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'}
    var_3 = var_0.load_docstring(var_1, var_1)

def test_case_6():
    var_0 = 'u6,CN\n*77A4'
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
    var_2 = '-].V8M\rE9'
    var_3 = module_0.parent(var_2)
    assert var_3 == '-]'

def test_case_7():
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

def test_case_8():
    var_0 = 'U%.|L3cs,e,p'
    var_1 = module_0.doctest(var_0)
    assert var_1 == 'U%.|L3cs,e,p'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_0 = '2CD#f`"@WQQ'
    var_1 = module_0.table(items=var_0)
    assert var_1 == '||\n||\n| 2 |\n| C |\n| D |\n| # |\n| f |\n| ` |\n| " |\n| @ |\n| W |\n| Q |\n| Q |\n\n'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_0 = module_1.List()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'ast.List'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    module_0.const_type(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = '8xI:u@#3A<'
    var_2 = '2M7`T'
    var_3 = 'B`|?D>/5'
    var_4 = {var_2: var_3, var_2: var_1, var_2: var_1, var_1: var_2, var_2: var_3}
    var_5 = module_0.Parser(var_0, var_0, level=var_0, doc=var_4, docstring=var_0, root=var_0, alias=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is None
    assert var_5.b_level is None
    assert var_5.toc is False
    assert var_5.level is None
    assert var_5.doc == {'2M7`T': 'B`|?D>/5', '8xI:u@#3A<': '2M7`T'}
    assert var_5.docstring is None
    assert var_5.imp == {}
    assert var_5.root is None
    assert var_5.alias == {'2M7`T': 'B`|?D>/5', '8xI:u@#3A<': '2M7`T'}
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
    var_6 = module_0.code(var_3)
    assert var_6 == '<code>B`&#124;?D>/5</code>'
    var_5.compile()

def test_case_12():
    var_0 = None
    var_1 = '5\x0cPX[/~mz{x:'
    var_2 = module_0.esc_underscore(var_1)
    assert var_2 == '5\x0cPX[/~mz{x:'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_3 = {var_1: var_1, var_1: var_1}
    var_4 = module_0.Resolver(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Resolver'
    assert var_4.root is None
    assert var_4.alias == {'5\x0cPX[/~mz{x:': '5\x0cPX[/~mz{x:'}
    assert var_4.self_ty == ''
    var_5 = module_0.code(var_2)
    assert var_5 == '`5\x0cPX[/~mz{x:`'
    var_6 = [var_0]
    var_7 = module_1.Call(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Call'
    assert var_7.func is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = module_0.const_type(var_7)
    assert var_8 == 'Any'
    var_9 = module_0.const_type(var_7)
    assert var_9 == 'Any'
    var_10 = module_0.is_magic(var_1)
    assert var_10 is False

def test_case_13():
    var_0 = '5\x0cPX[/~mz{x:'
    var_1 = module_0.esc_underscore(var_0)
    assert var_1 == '5\x0cPX[/~mz{x:'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_3 = module_0.walk_body(var_2)
    var_4 = module_0.is_public_family(var_1)
    assert var_4 is True
    var_5 = [var_1]
    var_6 = module_1.Tuple(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Tuple'
    assert var_6.elts == '5\x0cPX[/~mz{x:'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_7 = module_0.const_type(var_6)
    assert var_7 == 'tuple'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'u6,CN\n*77A4'
    var_1 = None
    var_2 = '`r7Q[1x9K66i'
    var_3 = module_0.Resolver(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Resolver'
    assert var_3.root == 'u6,CN\n*77A4'
    assert var_3.alias is None
    assert var_3.self_ty == '`r7Q[1x9K66i'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_4 = module_1.Dict()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Dict'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    module_0.const_type(var_4)

def test_case_15():
    var_0 = '1Tn0>MXvKp'
    var_1 = False
    var_2 = None
    var_3 = module_0.Parser(toc=var_1, imp=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is False
    assert var_3.level == {}
    assert var_3.doc == {}
    assert var_3.docstring == {}
    assert var_3.imp is None
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
    var_4 = var_3.globals(var_0, var_0)

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
    var_1 = "UG:`j4;BbEvg>'"
    var_2 = var_0.compile()
    assert var_2 == '\n'
    var_3 = [var_2, var_2, var_1, var_2]
    var_4 = module_1.AnnAssign(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.AnnAssign'
    assert var_4.target == '\n'
    assert var_4.annotation == '\n'
    assert var_4.value == "UG:`j4;BbEvg>'"
    assert var_4.simple == '\n'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.AnnAssign.value is None
    var_5 = 'y9Pyvy'
    var_6 = var_0.globals(var_5, var_4)

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
    var_1 = 'test_module'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'test_module': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module'}

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
    var_1 = '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': 2, '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': 2, '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': 2, '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': 2}
    assert var_0.doc == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': '## Module `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\n', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': '### class Example\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': '#### Example.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `list[int]` | `None` |\n\n'}
    assert var_0.docstring == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': 'Example function.', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': 'Example class.'}
    assert var_0.imp == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': {*()}}
    assert var_0.root == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '}
    assert var_0.alias == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .os': 'os', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .List': 'typing.List', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': '42'}
    assert var_0.const == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': 'int'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    `\n<a id="\nimport os\nfrom typing import list\n\nconstant = 42\n\ndef func(a: int, b: str) -> str:\n    """example function-"""\n    return b\n\nclass example:\n    """example class-"""\n    def method(self, x: list[int]) -> none:\n        pass\n    "></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n\n### class Example\n\n*Full name:* `\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example`\n<a id="\nimport os\nfrom typing import list\n\nconstant = 42\n\ndef func(a: int, b: str) -> str:\n    """example function-"""\n    return b\n\nclass example:\n    """example class-"""\n    def method(self, x: list[int]) -> none:\n        pass\n    -example"></a>\n\nExample class.\n\n#### Example.method()\n\n*Full name:* `\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method`\n<a id="\nimport os\nfrom typing import list\n\nconstant = 42\n\ndef func(a: int, b: str) -> str:\n    """example function-"""\n    return b\n\nclass example:\n    """example class-"""\n    def method(self, x: list[int]) -> none:\n        pass\n    -example-method"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `list[int]` | `None` |\n\n### func()\n\n*Full name:* `\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func`\n<a id="\nimport os\nfrom typing import list\n\nconstant = 42\n\ndef func(a: int, b: str) -> str:\n    """example function-"""\n    return b\n\nclass example:\n    """example class-"""\n    def method(self, x: list[int]) -> none:\n        pass\n    -func"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\nExample function.\n'

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = '_1fd\n%OB|D^Zq6V'
    var_1 = module_0.code(var_0)
    assert var_1 == '<code>_1fd\n%OB&#124;D^Zq6V</code>'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_2 = {var_1: var_1, var_0: var_0, var_0: var_0, var_1: var_1}
    var_3 = module_0.esc_underscore(var_0)
    assert var_3 == '_1fd\n%OB|D^Zq6V'
    var_4 = None
    var_5 = None
    var_6 = module_0.code(var_3)
    assert var_6 == '<code>_1fd\n%OB&#124;D^Zq6V</code>'
    var_7 = 'L_3\x0cy)$=~(d\x0b;("`K+'
    var_8 = module_0.is_public_family(var_7)
    assert var_8 is True
    var_9 = module_0.doctest(var_0)
    assert var_9 == '_1fd\n%OB|D^Zq6V'
    var_10 = module_0.is_public_family(var_3)
    assert var_10 is False
    var_11 = module_0.const_type(var_5)
    assert var_11 == 'Any'
    var_12 = {}
    var_13 = module_0.Parser(b_level=var_4, level=var_5, doc=var_2, imp=var_12, const=var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'apimd.parser.Parser'
    assert var_13.link is True
    assert var_13.b_level is None
    assert var_13.toc is False
    assert var_13.level is None
    assert var_13.doc == {'<code>_1fd\n%OB&#124;D^Zq6V</code>': '<code>_1fd\n%OB&#124;D^Zq6V</code>', '_1fd\n%OB|D^Zq6V': '_1fd\n%OB|D^Zq6V'}
    assert var_13.docstring == {}
    assert var_13.imp == {}
    assert var_13.root == {}
    assert var_13.alias == {}
    assert var_13.const is None
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_14 = var_13.globals(var_4, var_5)
    var_15 = module_2.getdoc(var_4)
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
    var_16 = '['
    var_13.imports(var_16, var_4)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = '_1fd\n%OB|D^Zq6V'
    var_2 = module_0.code(var_1)
    assert var_2 == '<code>_1fd\n%OB&#124;D^Zq6V</code>'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_3 = {var_2: var_2, var_1: var_1, var_1: var_1}
    var_4 = None
    var_5 = None
    var_6 = ''
    var_7 = 'Q<j\x0bs5,?<CQ'
    var_8 = 'r2Eru>o6_|lJI3Kw'
    var_9 = {var_1: var_6, var_7: var_8}
    var_10 = module_0.Parser(var_5, docstring=var_5, alias=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is None
    assert var_10.b_level == 1
    assert var_10.toc is False
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring is None
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias is None
    assert var_10.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_11 = module_0.Parser(toc=var_4, docstring=var_0, alias=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'apimd.parser.Parser'
    assert var_11.link is True
    assert var_11.b_level == 1
    assert var_11.toc is None
    assert var_11.level == {}
    assert var_11.doc == {}
    assert var_11.docstring is None
    assert var_11.imp == {}
    assert var_11.root == {}
    assert var_11.alias == {'_1fd\n%OB|D^Zq6V': '', 'Q<j\x0bs5,?<CQ': 'r2Eru>o6_|lJI3Kw'}
    assert var_11.const == {}
    var_12 = var_11.__eq__(var_6)
    var_13 = module_0.code(var_6)
    assert var_13 == ' '
    var_14 = 'L_3\x0cy)$=~(d\x0b;("`K+'
    var_15 = module_0.is_public_family(var_14)
    assert var_15 is True
    var_16 = module_0.doctest(var_1)
    assert var_16 == '_1fd\n%OB|D^Zq6V'
    var_17 = "+'%`|\t[O`n_vnG}?n$"
    var_18 = module_0.is_public_family(var_17)
    assert var_18 is True
    var_19 = module_0.const_type(var_5)
    assert var_19 == 'Any'
    var_20 = module_2.getdoc(var_0)
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
    var_21 = module_0.Parser(b_level=var_4, level=var_5, doc=var_3, imp=var_20, const=var_5)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'apimd.parser.Parser'
    assert var_21.link is True
    assert var_21.b_level is None
    assert var_21.toc is False
    assert var_21.level is None
    assert var_21.doc == {'<code>_1fd\n%OB&#124;D^Zq6V</code>': '<code>_1fd\n%OB&#124;D^Zq6V</code>', '_1fd\n%OB|D^Zq6V': '_1fd\n%OB|D^Zq6V'}
    assert var_21.docstring == {}
    assert var_21.imp is None
    assert var_21.root == {}
    assert var_21.alias == {}
    assert var_21.const is None
    var_22 = var_21.globals(var_4, var_0)
    var_23 = module_2.getdoc(var_4)
    var_21.imports(var_4, var_23)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = '_1fd\n%OB|D^Zq6V'
    var_2 = module_0.code(var_1)
    assert var_2 == '<code>_1fd\n%OB&#124;D^Zq6V</code>'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_3 = module_0.esc_underscore(var_1)
    assert var_3 == '_1fd\n%OB|D^Zq6V'
    var_4 = ''
    var_5 = {var_1: var_4, var_0: var_2}
    var_6 = module_0.Parser(var_0, docstring=var_0, alias=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is None
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring is None
    assert var_6.imp == {}
    assert var_6.root == {}
    assert var_6.alias is None
    assert var_6.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_7 = module_0.Parser(toc=var_0, docstring=var_0, alias=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is True
    assert var_7.b_level == 1
    assert var_7.toc is None
    assert var_7.level == {}
    assert var_7.doc == {}
    assert var_7.docstring is None
    assert var_7.imp == {}
    assert var_7.root == {}
    assert var_7.alias == {'_1fd\n%OB|D^Zq6V': '', None: '<code>_1fd\n%OB&#124;D^Zq6V</code>'}
    assert var_7.const == {}
    var_8 = var_7.compile()
    assert var_8 == '\n'
    var_9 = module_0.code(var_3)
    assert var_9 == '<code>_1fd\n%OB&#124;D^Zq6V</code>'
    var_10 = module_0.doctest(var_1)
    assert var_10 == '_1fd\n%OB|D^Zq6V'
    var_11 = var_6.__eq__(var_0)
    var_12 = "+'%`|\t[O`n_vnG}?n$"
    var_13 = module_0.is_public_family(var_12)
    assert var_13 is True
    var_14 = module_0.const_type(var_0)
    assert var_14 == 'Any'
    var_15 = 'l"2+,$^m&'
    var_6.parse(var_10, var_15)

def test_case_22():
    var_0 = ">>> print('Hello, World!')"
    var_1 = module_0.doctest(var_0)
    assert var_1 == "```python\n>>> print('Hello, World!')\n```"
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_2 = '>>> x = 5\n>>> y = 10\n>>> print(x + y)'
    var_3 = module_0.doctest(var_2)
    assert var_3 == '```python\n>>> x = 5\n>>> y = 10\n>>> print(x + y)\n```'
    var_4 = "This is not a doctest.\n>>> print('This is a doctest.')"
    var_5 = module_0.doctest(var_4)
    assert var_5 == "This is not a doctest.\n```python\n>>> print('This is a doctest.')\n```"
    var_6 = ">>> print('End of doctest')"
    var_7 = module_0.doctest(var_6)
    assert var_7 == "```python\n>>> print('End of doctest')\n```"
    var_8 = ''
    var_9 = module_0.doctest(var_8)
    assert var_9 == ''
    var_10 = 'This is just text.\nMore text.'
    var_11 = module_0.doctest(var_10)
    assert var_11 == 'This is just text.\nMore text.'

def test_case_23():
    var_0 = ">>> print('Hello, World!')"
    var_1 = "```python\n>>> print('Hello, World!')\n```"
    var_2 = module_0.doctest(var_0)
    assert var_2 == "```python\n>>> print('Hello, World!')\n```"
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_3 = module_0.doctest(var_1)
    assert var_3 == "```python\n```python\n>>> print('Hello, World!')\n```\n```"
    var_4 = "This is not a doctest.\n>>> print('This is a doctest.')"
    var_5 = module_0.doctest(var_4)
    assert var_5 == "This is not a doctest.\n```python\n>>> print('This is a doctest.')\n```"
    var_6 = ">>> print('End of doctest')"
    var_7 = module_0.doctest(var_6)
    assert var_7 == "```python\n>>> print('End of doctest')\n```"
    var_8 = ''
    var_9 = module_0.doctest(var_8)
    assert var_9 == ''
    var_10 = 'This is just text.\nMore text.'
    var_11 = module_0.doctest(var_10)
    assert var_11 == 'This is just text.\nMore text.'
    var_12 = "Text before doctest.\n>>> print('Doctest line 1')\nText between.\n>>> print('Doctest line 2')"
    var_13 = module_0.doctest(var_12)
    assert var_13 == "Text before doctest.\n```python\n>>> print('Doctest line 1')\n```\nText between.\n```python\n>>> print('Doctest line 2')\n```"

@pytest.mark.xfail(strict=True)
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
    var_1 = 'P 1w\rLr'
    var_2 = module_0.esc_underscore(var_1)
    assert var_2 == 'P 1w\rLr'
    var_3 = 'j[<q\rpwy5>*N)H]*#'
    var_4 = None
    var_5 = module_1.expr()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.expr'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.expr.end_lineno is None
    assert module_1.expr.end_col_offset is None
    var_6 = [var_5]
    var_0.class_api(var_3, var_4, var_6, var_4)

def test_case_25():
    var_0 = None
    var_1 = None
    var_2 = ''
    var_3 = 'Q<j\x0bs5,?<CQ'
    var_4 = 'r2Eru>o6_|lJI3Kw'
    var_5 = {var_4: var_2, var_3: var_4}
    var_6 = module_0.Parser(var_1, docstring=var_1, alias=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is None
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring is None
    assert var_6.imp == {}
    assert var_6.root == {}
    assert var_6.alias is None
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
    var_7 = {}
    var_8 = module_0.Parser(imp=var_7)
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
    var_9 = module_0.Parser(toc=var_3, docstring=var_0, alias=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc == 'Q<j\x0bs5,?<CQ'
    assert var_9.level == {}
    assert var_9.doc == {}
    assert var_9.docstring is None
    assert var_9.imp == {}
    assert var_9.root == {}
    assert var_9.alias == {'r2Eru>o6_|lJI3Kw': '', 'Q<j\x0bs5,?<CQ': 'r2Eru>o6_|lJI3Kw'}
    assert var_9.const == {}
    var_10 = var_8.compile()
    assert var_10 == '\n'
    var_11 = 'L_3\x0cy)$=~(d\x0b;("`K+'
    var_12 = module_0.is_public_family(var_11)
    assert var_12 is True
    var_13 = module_0.doctest(var_2)
    assert var_13 == ''
    var_14 = var_6.__eq__(var_0)
    var_15 = "+'%`|\t[O`n_vnG}?n$"
    var_16 = module_0.is_public_family(var_15)
    assert var_16 is True
    var_17 = {}
    var_18 = var_9.load_docstring(var_17, var_15)
    var_19 = module_0.const_type(var_1)
    assert var_19 == 'Any'
    var_20 = var_9.compile()
    assert var_20 == '**Table of contents:**\n\n\n'
    var_21 = module_2.getdoc(var_18)
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

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = '{xD=:u@#3q<'
    var_2 = 'fs3=? sch|NKk\x0c}p1'
    var_3 = '`|?D>/5'
    var_4 = {var_2: var_3, var_2: var_1, var_2: var_1, var_1: var_2, var_2: var_3}
    var_5 = module_0.Parser(var_0, var_0, level=var_0, doc=var_4, docstring=var_0, root=var_0, alias=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is None
    assert var_5.b_level is None
    assert var_5.toc is False
    assert var_5.level is None
    assert var_5.doc == {'fs3=? sch|NKk\x0c}p1': '`|?D>/5', '{xD=:u@#3q<': 'fs3=? sch|NKk\x0c}p1'}
    assert var_5.docstring is None
    assert var_5.imp == {}
    assert var_5.root is None
    assert var_5.alias == {'fs3=? sch|NKk\x0c}p1': '`|?D>/5', '{xD=:u@#3q<': 'fs3=? sch|NKk\x0c}p1'}
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
    var_1 = var_0.__repr__()
    assert var_1 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_2 = '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '
    var_3 = var_0.parse(var_2, var_2)
    assert var_0.level == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': 2, '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': 2, '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': 2, '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': 2}
    assert var_0.doc == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': '## Module `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\n', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': '### class Example\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': '#### Example.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `list[int]` | `None` |\n\n'}
    assert var_0.docstring == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': 'Example function.', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': 'Example class.'}
    assert var_0.imp == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': {*()}}
    assert var_0.root == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '}
    assert var_0.alias == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .os': 'os', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .List': 'typing.List', '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': '42'}
    assert var_0.const == {'\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': 'int'}
    var_4 = var_0.load_docstring(var_1, var_1)

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

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = '#\\\x0b*jb|yi'
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
    var_2 = module_0.esc_underscore(var_0)
    assert var_2 == '#\\\x0b*jb|yi'
    var_3 = {var_2: var_0, var_2: var_2}
    var_4 = module_0.Parser(docstring=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level == 1
    assert var_4.toc is False
    assert var_4.level == {}
    assert var_4.doc == {}
    assert var_4.docstring == {'#\\\x0b*jb|yi': '#\\\x0b*jb|yi'}
    assert var_4.imp == {}
    assert var_4.root == {}
    assert var_4.alias == {}
    assert var_4.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_5 = None
    var_6 = var_4.__post_init__()
    var_7 = var_4.compile()
    assert var_7 == '\n'
    var_8 = module_0.doctest(var_7)
    assert var_8 == ''
    var_9 = [var_8]
    var_10 = module_1.Tuple(*var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Tuple'
    assert var_10.elts == ''
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_11 = module_0.const_type(var_10)
    assert var_11 == 'tuple'
    var_12 = module_0.Parser(var_1, docstring=var_3, alias=var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'apimd.parser.Parser'
    assert var_12.link is True
    assert var_12.b_level == 1
    assert var_12.toc is False
    assert var_12.level == {}
    assert var_12.doc == {}
    assert var_12.docstring == {'#\\\x0b*jb|yi': '#\\\x0b*jb|yi'}
    assert var_12.imp == {}
    assert var_12.root == {}
    assert var_12.alias is None
    assert var_12.const == {}
    var_6.compile()

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = '\nimport os\nfrom typing import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'\n': 0, '\n.func': 0, '\n.Example': 0, '\n.Example.method': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\n', '\n.Example': '### class Example\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\n.Example.method': '#### Example.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `list[int]` | `None` |\n\n'}
    assert var_0.docstring == {'\n.func': 'Example function.', '\n.Example': 'Example class.'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n', '\n.CONSTANT': '\n', '\n.func': '\n', '\n.Example': '\n', '\n.Example.method': '\n'}
    assert var_0.alias == {'\n.os': 'os', '\n.List': 'typing.List', '\n.CONSTANT': '42'}
    assert var_0.const == {'\n.CONSTANT': 'int'}
    var_4 = var_0.parse(var_1, var_2)

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
    var_1 = 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    '
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    ': 2, 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .func': 2, 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .Exampl': 2, 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .Exampl.method': 2}
    assert var_0.doc == {'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    ': '## Module `{}`\n<a id="{}"></a>\n\n', 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\n', 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .Exampl': '### class Exampl\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .Exampl.method': '#### Exampl.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `typin.List[int]` | `Non` |\n\n'}
    assert var_0.docstring == {'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .func': 'Example function.', 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .Exampl': 'Example class.'}
    assert var_0.imp == {'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    ': {*()}}
    assert var_0.root == {'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    ': 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    ', 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .CONSTANT': 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    ', 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .func': 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    ', 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .Exampl': 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    ', 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .Exampl.method': 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    '}
    assert var_0.alias == {'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .os': 'os', 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .List': 'typin.List', 'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .CONSTANT': '42'}
    assert var_0.const == {'import os\nfrom typin import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    .CONSTANT': 'int'}

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
    var_1 = module_2.getdoc(var_0)
    assert var_1 == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
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
    var_2 = '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)': 5, 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).func': 5, 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).Example': 5, 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).Example.method': 5}
    assert var_0.doc == {'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)': '## Module `{}`\n<a id="{}"></a>\n\n', 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\n', 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).Example': '### class Example\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).Example.method': '#### Example.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `t7ping.List[int]` | `None` |\n\n'}
    assert var_0.docstring == {'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).func': 'Example function.', 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).Example': 'Example class.'}
    assert var_0.imp == {'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)': {*()}}
    assert var_0.root == {'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)': 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)', 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).CONSTANT': 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)', 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).func': 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)', 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).Example': 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)', 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).Example.method': 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'}
    assert var_0.alias == {'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).os': 'os', 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).List': 't7ping.List', 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).CONSTANT': '42'}
    assert var_0.const == {'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).CONSTANT': 'int'}
    var_4 = var_0.load_docstring(var_1, var_1)
    var_5 = {}
    var_6 = '|/xG/]M{<}iAF!B-m'
    var_7 = 'hA:\r77I\rG~uDAqN\r\t?$'
    var_8 = ':~$l|S\tTz[*'
    var_9 = 'ER9hx5d'
    var_10 = '=VdYtdnJRy{u'
    var_11 = '](B(5\nr$_(3zi\nDYq'
    var_12 = {var_6: var_2, var_7: var_6, var_8: var_9, var_10: var_11}
    var_13 = module_0.Parser(b_level=var_4, doc=var_5, root=var_12, const=var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'apimd.parser.Parser'
    assert var_13.link is True
    assert var_13.b_level is None
    assert var_13.toc is False
    assert var_13.level == {}
    assert var_13.doc == {}
    assert var_13.docstring == {}
    assert var_13.imp == {}
    assert var_13.root == {'|/xG/]M{<}iAF!B-m': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', 'hA:\r77I\rG~uDAqN\r\t?$': '|/xG/]M{<}iAF!B-m', ':~$l|S\tTz[*': 'ER9hx5d', '=VdYtdnJRy{u': '](B(5\nr$_(3zi\nDYq'}
    assert var_13.alias == {}
    assert var_13.const == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
    var_14 = module_0.esc_underscore(var_1)
    assert var_14 == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg\\_path", \'r\') as f:\n>>>     p.parse(\'pkg\\_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
    var_15 = var_13.load_docstring(var_13, var_1)
    var_16 = module_0.Resolver(var_3, var_5)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'apimd.parser.Resolver'
    assert var_16.root is None
    assert var_16.alias == {}
    assert var_16.self_ty == ''
    var_17 = var_0.compile()
    assert var_17 == '## Module `AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)`\n<a id="ast parser-\n\nusage:\n>>> p = parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p-parse(\'pkg_name\', f-read())\n>>> s = p-compile()\n\nor create with parameters:\n>>> p = parser-new(link=true, level=1)"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n\n### class Example\n\n*Full name:* `AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).Example`\n<a id="ast parser-\n\nusage:\n>>> p = parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p-parse(\'pkg_name\', f-read())\n>>> s = p-compile()\n\nor create with parameters:\n>>> p = parser-new(link=true, level=1)-example"></a>\n\nExample class.\n\n#### Example.method()\n\n*Full name:* `AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).Example.method`\n<a id="ast parser-\n\nusage:\n>>> p = parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p-parse(\'pkg_name\', f-read())\n>>> s = p-compile()\n\nor create with parameters:\n>>> p = parser-new(link=true, level=1)-example-method"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `t7ping.List[int]` | `None` |\n\n### func()\n\n*Full name:* `AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1).func`\n<a id="ast parser-\n\nusage:\n>>> p = parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p-parse(\'pkg_name\', f-read())\n>>> s = p-compile()\n\nor create with parameters:\n>>> p = parser-new(link=true, level=1)-func"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\nExample function.\n'

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'\n': 0, '\n.func': 0, '\n.Example': 0, '\n.Example.method': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\n', '\n.Example': '### class Example\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\n.Example.method': '#### Example.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `t7ping.List[int]` | `None` |\n\n'}
    assert var_0.docstring == {'\n.func': 'Example function.', '\n.Example': 'Example class.'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n', '\n.CONSTANT': '\n', '\n.func': '\n', '\n.Example': '\n', '\n.Example.method': '\n'}
    assert var_0.alias == {'\n.os': 'os', '\n.List': 't7ping.List', '\n.CONSTANT': '42'}
    assert var_0.const == {'\n.CONSTANT': 'int'}
    var_4 = var_0.load_docstring(var_1, var_1)
    var_5 = {}
    var_6 = '|/xG/]M{<}iAF!B-m'
    var_7 = ':~$l|S\tTz[*'
    var_8 = module_1.ImportFrom()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.ImportFrom'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_9 = var_0.imports(var_3, var_8)
    var_10 = module_0.const_type(var_6)
    assert var_10 == 'Any'
    var_11 = module_0.Parser(b_level=var_4, doc=var_5, root=var_5, const=var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'apimd.parser.Parser'
    assert var_11.link is True
    assert var_11.b_level is None
    assert var_11.toc is False
    assert var_11.level == {}
    assert var_11.doc == {}
    assert var_11.docstring == {}
    assert var_11.imp == {}
    assert var_11.root == {}
    assert var_11.alias == {}
    assert var_11.const == '\n'
    var_12 = module_0.esc_underscore(var_1)
    assert var_12 == '\n'
    var_13 = module_0.Resolver(var_3, var_4, var_2)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'apimd.parser.Resolver'
    assert var_13.root is None
    assert var_13.alias is None
    assert var_13.self_ty == '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '
    var_14 = module_2.getdoc(var_7)
    assert var_14 == "str(object='') -> str\nstr(bytes_or_buffer[, encoding[, errors]]) -> str\n\nCreate a new string object from the given object. If encoding or\nerrors is specified, then the object must expose a data buffer\nthat will be decoded using the given encoding and error handler.\nOtherwise, returns the result of object.__str__() (if defined)\nor repr(object).\nencoding defaults to sys.getdefaultencoding().\nerrors defaults to 'strict'."
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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = 'import os\nfrom typin import List\nqCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    '
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'\n': 0, '\n.func': 0, '\n.Exampl': 0, '\n.Exampl.method': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\n', '\n.Exampl': '### class Exampl\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\n.Exampl.method': '#### Exampl.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `typin.List[int]` | `Non` |\n\n'}
    assert var_0.docstring == {'\n.func': 'Example function.', '\n.Exampl': 'Example class.'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n', '\n.func': '\n', '\n.Exampl': '\n', '\n.Exampl.method': '\n'}
    assert var_0.alias == {'\n.os': 'os', '\n.List': 'typin.List', '\n.qCONSTANT': '42'}

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = 'import os\nfrom typin import List\nqCONSTANT = 42\n\ndef func(a: int, b: str) -> str:\n    """Example function."""\n    return b\n\nclass Exampl:\n    """Example class."""\n    def method(self, x: List[int]) -> Non :\n        pass\n    '
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'\n': 0, '\n.func': 0, '\n.Exampl': 0, '\n.Exampl.method': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\n', '\n.Exampl': '### class Exampl\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\n.Exampl.method': '#### Exampl.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `typin.List[int]` | `Non` |\n\n'}
    assert var_0.docstring == {'\n.func': 'Example function.', '\n.Exampl': 'Example class.'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n', '\n.func': '\n', '\n.Exampl': '\n', '\n.Exampl.method': '\n'}
    assert var_0.alias == {'\n.os': 'os', '\n.List': 'typin.List', '\n.qCONSTANT': '42'}
    var_4 = var_0.compile()
    assert var_4 == '## Module `\n`\n<a id="\n"></a>\n\n### class Exampl\n\n*Full name:* `\n.Exampl`\n<a id="\n-exampl"></a>\n\nExample class.\n\n#### Exampl.method()\n\n*Full name:* `\n.Exampl.method`\n<a id="\n-exampl-method"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `typin.List[int]` | `Non` |\n\n### func()\n\n*Full name:* `\n.func`\n<a id="\n-func"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `str` |\n\nExample function.\n'

def test_case_36():
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
    var_3 = 'test'
    var_4 = module_1.Constant()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Constant'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.Constant.kind is None
    assert f'{type(module_1.Constant.n).__module__}.{type(module_1.Constant.n).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Constant.s).__module__}.{type(module_1.Constant.s).__qualname__}' == 'builtins.property'
    var_5 = module_1.Expr()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Expr'
    var_6 = [var_5]
    var_7 = module_0.walk_body(var_6)
    var_8 = list(var_7)
    var_9 = True
    var_10 = module_1.Constant()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Constant'
    var_11 = 'if_true'
    var_12 = module_1.Constant()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Constant'
    var_13 = module_1.Expr()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.Expr'
    var_14 = [var_13]
    var_15 = 'if_false'
    var_16 = module_1.Constant()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.Constant'
    var_17 = module_1.Expr()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Expr'
    var_18 = [var_17]
    var_19 = module_1.If()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.If'
    var_20 = module_1.Constant()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'ast.Constant'
    var_21 = module_1.Expr()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'ast.Expr'
    var_22 = module_1.Constant()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Constant'
    var_23 = module_1.Expr()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'ast.Expr'
    var_24 = [var_19]
    var_25 = module_0.walk_body(var_24)
    with pytest.raises(AttributeError):
        var_26 = list(var_25)

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
    var_2 = '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '
    var_3 = var_0.parse(var_2, var_2)
    assert var_0.level == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': 2, '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': 2, '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': 2, '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': 2}
    assert var_0.doc == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': '## Module `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `st` | `str` |\n\n', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': '### class Example\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': '#### Example.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `t7ping.List[int]` | `None` |\n\n'}
    assert var_0.docstring == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': 'Example function.', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': 'Example class.'}
    assert var_0.imp == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': {*()}}
    assert var_0.root == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    '}
    assert var_0.alias == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .os': 'os', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .List': 't7ping.List', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': '42'}
    assert var_0.const == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': 'int'}
    var_4 = var_0.parse(var_1, var_2)
    assert var_0.level == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': 2, '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': 2, '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': 2, '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': 2, '\n': 0, '\n.func': 0, '\n.Example': 0, '\n.Example.method': 0}
    assert var_0.doc == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': '## Module `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `st` | `str` |\n\n', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': '### class Example\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': '#### Example.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `t7ping.List[int]` | `None` |\n\n', '\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.func': '### func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `st` | `str` |\n\n', '\n.Example': '### class Example\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\n.Example.method': '#### Example.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `t7ping.List[int]` | `None` |\n\n'}
    assert var_0.docstring == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': 'Example function.', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': 'Example class.', '\n.func': 'Example function.', '\n.Example': 'Example class.'}
    assert var_0.imp == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': {*()}, '\n': {*()}}
    assert var_0.root == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method': '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    ', '\n': '\n', '\n.CONSTANT': '\n', '\n.func': '\n', '\n.Example': '\n', '\n.Example.method': '\n'}
    assert var_0.alias == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .os': 'os', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .List': 't7ping.List', '\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': '42', '\n.os': 'os', '\n.List': 't7ping.List', '\n.CONSTANT': '42'}
    assert var_0.const == {'\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .CONSTANT': 'int', '\n.CONSTANT': 'int'}
    var_5 = var_0.load_docstring(var_1, var_1)
    var_6 = module_0.const_type(var_5)
    assert var_6 == 'Any'
    var_7 = var_0.__eq__(var_5)
    var_8 = module_0.esc_underscore(var_6)
    assert var_8 == 'Any'
    var_9 = var_0.compile()
    assert var_9 == '## Module `\n`\n<a id="\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n\n### class Example\n\n*Full name:* `\n.Example`\n<a id="\n-example"></a>\n\nExample class.\n\n#### Example.method()\n\n*Full name:* `\n.Example.method`\n<a id="\n-example-method"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `t7ping.List[int]` | `None` |\n\n### func()\n\n*Full name:* `\n.func`\n<a id="\n-func"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `st` | `str` |\n\nExample function.\n\n## Module `\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    `\n<a id="\nimport os\nfrom t7ping import list\n\nconstant = 42\n\ndef func(a:int, b: st) -> str:\n    """example function-"""\n    return b\n\nclass example:\n    """example class-"""\n    def method(self, x: list[int]) -> none:\n        pass\n    "></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONSTANT` | `int` |\n\n### class Example\n\n*Full name:* `\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example`\n<a id="\nimport os\nfrom t7ping import list\n\nconstant = 42\n\ndef func(a:int, b: st) -> str:\n    """example function-"""\n    return b\n\nclass example:\n    """example class-"""\n    def method(self, x: list[int]) -> none:\n        pass\n    -example"></a>\n\nExample class.\n\n#### Example.method()\n\n*Full name:* `\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .Example.method`\n<a id="\nimport os\nfrom t7ping import list\n\nconstant = 42\n\ndef func(a:int, b: st) -> str:\n    """example function-"""\n    return b\n\nclass example:\n    """example class-"""\n    def method(self, x: list[int]) -> none:\n        pass\n    -example-method"></a>\n\n| self | x | return |\n|:----:|:---:|:------:|\n| `Self` | `t7ping.List[int]` | `None` |\n\n### func()\n\n*Full name:* `\nimport os\nfrom t7ping import List\n\nCONSTANT = 42\n\ndef func(a:int, b: st) -> str:\n    """Example function."""\n    return b\n\nclass Example:\n    """Example class."""\n    def method(self, x: List[int]) -> None:\n        pass\n    .func`\n<a id="\nimport os\nfrom t7ping import list\n\nconstant = 42\n\ndef func(a:int, b: st) -> str:\n    """example function-"""\n    return b\n\nclass example:\n    """example class-"""\n    def method(self, x: list[int]) -> none:\n        pass\n    -func"></a>\n\n| a | b | return |\n|:---:|:---:|:------:|\n| `int` | `st` | `str` |\n\nExample function.\n'

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
    var_1 = 'example_module'
    var_2 = 'example_module.example_function'
    var_3 = 'arg1'
    var_4 = None
    var_5 = module_1.arg()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.arg'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.arg.annotation is None
    assert module_1.arg.type_comment is None
    assert module_1.arg.end_lineno is None
    assert module_1.arg.end_col_offset is None
    var_6 = [var_5]
    var_7 = 'arg2'
    var_8 = module_1.arg()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.arg'
    var_9 = [var_8]
    var_10 = '*args'
    var_11 = module_1.arg()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.arg'
    var_12 = 'kwarg1'
    var_13 = module_1.arg()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'ast.arg'
    var_14 = [var_13]
    var_15 = '**kwargs'
    var_16 = module_1.arg()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'ast.arg'
    var_17 = []
    var_18 = []
    var_19 = module_1.arguments(*var_9)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'ast.arguments'
    assert f'{type(var_19.posonlyargs).__module__}.{type(var_19.posonlyargs).__qualname__}' == 'ast.arg'
    assert module_1.arguments.vararg is None
    assert module_1.arguments.kwarg is None
    var_20 = None
    var_21 = False
    var_22 = False
    var_0.func_api(var_1, var_2, var_19, var_20, has_self=var_21, cls_method=var_22)