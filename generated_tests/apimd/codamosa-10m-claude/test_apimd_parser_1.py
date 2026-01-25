# Check out: https://github.com/GlowCheese/deepmosa
import ast as module_1
import dataclasses as module_2
import inspect as module_3

import apimd.parser as module_0
import pytest


def test_case_0():
    var_0 = 'p*9$'
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
    var_0 = 'def foo(): pass'
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
    assert var_1.level == {'def foo(): pass': 0, 'def foo(): pass.foo': 0}
    assert var_1.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_1.imp == {'def foo(): pass': {*()}}
    assert var_1.root == {'def foo(): pass': 'def foo(): pass', 'def foo(): pass.foo': 'def foo(): pass'}

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.table(items=var_0)

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
    var_1 = "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 3, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 3, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": 3, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 3, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 3}
    assert var_0.doc == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr` | `int` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": '#### PublicClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_0.docstring == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 'Module docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 'Function docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 'Class docstringr', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 'Method docstring.'}
    assert var_0.imp == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func"}}
    assert var_0.root == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"}
    assert var_0.alias == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.os": 'os', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.List": 'typing.List', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.__all__": "['public_func', 'PUBLIC_CONST']", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": '42'}
    assert var_0.const == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstringr'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": 'int'}

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
    var_1 = "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 4}
    assert var_0.doc == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr` | `int` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": '#### PublicClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_0.docstring == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 'Module docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 'Function docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 'Class docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 'Method docstring.'}
    assert var_0.imp == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST"}}
    assert var_0.root == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"}
    assert var_0.alias == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.os": 'os', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.List": 'typing.List', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.__all__": "['public_func', 'PUBLIC_CONST']", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": '42'}
    assert var_0.const == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": 'int'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\n\'\'\'Module docstring.\'\'\'\nimport os\nfrom typing import List\n\n__all__ = [\'public_func\', \'PUBLIC_CONST\']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'Function docstring.\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    \'\'\'Class docstring.\'\'\'\n    attr: int = 10\n    \n    def method(self) -> None:\n        \'\'\'Method docstring.\'\'\'\n        pass\n`\n<a id="\n\'\'\'module docstring-\'\'\'\nimport os\nfrom typing import list\n\n__all__ = [\'public_func\', \'public_const\']\n\npublic_const: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'function docstring-\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass publicclass:\n    \'\'\'class docstring-\'\'\'\n    attr: int = 10\n    \n    def method(self) -> none:\n        \'\'\'method docstring-\'\'\'\n        pass\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `PUBLIC_CONST` | `int` |\n\nModule docstring.\n\n### public_func()\n\n*Full name:* `\n\'\'\'Module docstring.\'\'\'\nimport os\nfrom typing import List\n\n__all__ = [\'public_func\', \'PUBLIC_CONST\']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'Function docstring.\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    \'\'\'Class docstring.\'\'\'\n    attr: int = 10\n    \n    def method(self) -> None:\n        \'\'\'Method docstring.\'\'\'\n        pass\n.public_func`\n<a id="\n\'\'\'module docstring-\'\'\'\nimport os\nfrom typing import list\n\n__all__ = [\'public_func\', \'public_const\']\n\npublic_const: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'function docstring-\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass publicclass:\n    \'\'\'class docstring-\'\'\'\n    attr: int = 10\n    \n    def method(self) -> none:\n        \'\'\'method docstring-\'\'\'\n        pass\n-public_func"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\nFunction docstring.\n'

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

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = -56
    var_1 = True
    var_2 = 'MTJ%n;oFR(uN[\x0bu"'
    var_3 = ':Wp'
    var_4 = {var_2: var_3}
    var_5 = module_0.Parser(b_level=var_0, toc=var_1, const=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'apimd.parser.Parser'
    assert var_5.link is True
    assert var_5.b_level == -56
    assert var_5.toc is True
    assert var_5.level == {}
    assert var_5.doc == {}
    assert var_5.docstring == {}
    assert var_5.imp == {}
    assert var_5.root == {}
    assert var_5.alias == {}
    assert var_5.const == {'MTJ%n;oFR(uN[\x0bu"': ':Wp'}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
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
    assert var_6 == '**Table of contents:**\n\n\n'
    var_7 = 'C]TeP3*EPG?CG(\\N_=m'
    var_8 = module_0.is_public_family(var_7)
    assert var_8 is True
    var_9 = None
    module_0.esc_underscore(var_9)

def test_case_10():
    var_0 = 'def foo(): pass'
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
    var_2 = var_1.compile()
    assert var_2 == '\n'
    var_3 = var_1.parse(var_2, var_0)
    assert var_1.level == {'\n': 0, '\n.foo': 0}
    assert var_1.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_1.imp == {'\n': {*()}}
    assert var_1.root == {'\n': '\n', '\n.foo': '\n'}

def test_case_11():
    var_0 = 'p'
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

def test_case_12():
    var_0 = '"U\x0bDWH'
    var_1 = module_0.doctest(var_0)
    assert var_1 == '"U\nDWH'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_0 = '"U\x0bDWH'
    var_1 = module_0.code(var_0)
    assert var_1 == '`"U\x0bDWH`'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
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
    var_0 = None
    var_1 = module_1.Delete()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'ast.Delete'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_2 = module_0.Parser(level=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'apimd.parser.Parser'
    assert var_2.link is True
    assert var_2.b_level == 1
    assert var_2.toc is False
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
    var_3 = var_2.compile()
    assert var_3 == '\n'
    var_4 = None
    var_5 = 'ZTTq(qOl'
    var_6 = module_0.doctest(var_5)
    assert var_6 == 'ZTTq(qOl'
    var_7 = 'v'
    var_8 = var_2.load_docstring(var_0, var_4)
    var_9 = {var_5: var_6}
    var_10 = module_0.Resolver(var_7, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Resolver'
    assert var_10.root == 'v'
    assert var_10.alias == {'ZTTq(qOl': 'ZTTq(qOl'}
    assert var_10.self_ty == ''
    var_11 = module_1.Dict()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Dict'
    var_12 = var_2.__eq__(var_0)
    module_0.const_type(var_11)

def test_case_15():
    var_0 = 'Ykh?BZF+66q"QS~O\\|'
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
    var_2 = module_0.code(var_0)
    assert var_2 == '<code>Ykh?BZF+66q"QS~O\\&#124;</code>'
    var_3 = "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"
    var_4 = var_1.parse(var_3, var_3)
    assert var_1.level == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 4}
    assert var_1.doc == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr` | `int` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": '#### PublicClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_1.docstring == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 'Module docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 'Function docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 'Class docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 'Method docstring.'}
    assert var_1.imp == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST"}}
    assert var_1.root == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"}
    assert var_1.alias == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.os": 'os', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.List": 'typing.List', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.__all__": "['public_func', 'PUBLIC_CONST']", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": '42'}
    assert var_1.const == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": 'int'}
    var_5 = var_1.__eq__(var_2)
    var_6 = module_0.const_type(var_5)
    assert var_6 == 'Any'

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
    var_1 = "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 4}
    assert var_0.doc == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr` | `int` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": '#### PublicClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_0.docstring == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 'Module docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 'Function docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 'Class docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 'Method docstring.'}
    assert var_0.imp == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST"}}
    assert var_0.root == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"}
    assert var_0.alias == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.os": 'os', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.List": 'typing.List', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.__all__": "['public_func', 'PUBLIC_CONST']", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": '42'}
    assert var_0.const == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": 'int'}
    var_3 = var_0.load_docstring(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = module_0.Parser(level=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is True
    assert var_1.b_level == 1
    assert var_1.toc is False
    assert var_1.level is None
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
    var_2 = '*3I'
    var_3 = var_1.globals(var_2, var_0)
    var_1.parse(var_3, var_0)

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
    var_1 = '\nclass TestClass:\n    pass\n'
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
    var_7 = var_0.class_api(var_1, var_6, var_5, var_6)
    var_8 = '\nclass TestClass(BaseClass):\n    pass\n'
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
    var_10 = module_1.parse(var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'ast.Module'
    assert f'{type(var_10.body).__module__}.{type(var_10.body).__qualname__}' == 'builtins.list'
    assert len(var_10.body) == 1
    assert var_10.type_ignores == []
    var_11 = var_10.body[var_3]
    var_12 = var_11.bases
    var_13 = var_11.body
    var_9.class_api(var_6, var_8, var_12, var_13)

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
    var_1 = 'decorated_module'
    var_2 = "\nfrom functools import wraps\n\n@wraps\ndef decorated_func():\n    '''Decorated function.'''\n    pass\n"
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'decorated_module': 0, 'decorated_module.decorated_func': 0}
    assert var_0.doc == {'decorated_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'decorated_module.decorated_func': '### decorated_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.wraps` |\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {'decorated_module.decorated_func': 'Decorated function.'}
    assert var_0.imp == {'decorated_module': {*()}}
    assert var_0.root == {'decorated_module': 'decorated_module', 'decorated_module.decorated_func': 'decorated_module'}
    assert var_0.alias == {'decorated_module.wraps': 'functools.wraps'}

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.Parser(level=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'apimd.parser.Parser'
    assert var_1.link is True
    assert var_1.b_level == 1
    assert var_1.toc is False
    assert var_1.level is None
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
    var_2 = '}*\x0b&a?l">S}GPx*m'
    var_3 = module_1.ImportFrom()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.ImportFrom'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_4 = module_0.Parser(level=var_1, imp=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level == 1
    assert var_4.toc is False
    assert f'{type(var_4.level).__module__}.{type(var_4.level).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.doc == {}
    assert var_4.docstring == {}
    assert f'{type(var_4.imp).__module__}.{type(var_4.imp).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.root == {}
    assert var_4.alias == {}
    assert var_4.const == {}
    var_5 = var_4.imports(var_0, var_3)
    var_2.visit(var_0)

@pytest.mark.xfail(strict=True)
def test_case_21():
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
    var_3 = module_2.field(default_factory=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'dataclasses.Field'
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
    var_4 = '\x0cveg/tRre|'
    var_3.is_public(var_4)

def test_case_22():
    var_0 = 'pkg.mod}le2'
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
    var_2 = 'CONST = 42'
    var_3 = var_1.parse(var_0, var_2)
    assert var_1.level == {'pkg.mod}le2': 1}
    assert var_1.doc == {'pkg.mod}le2': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.imp == {'pkg.mod}le2': {*()}}
    assert var_1.root == {'pkg.mod}le2': 'pkg.mod}le2', 'pkg.mod}le2.CONST': 'pkg.mod}le2'}
    assert var_1.alias == {'pkg.mod}le2.CONST': '42'}
    assert var_1.const == {'pkg.mod}le2.CONST': 'int'}

def test_case_23():
    var_0 = 'module'
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
    var_2 = 'module.submodule'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = 'module.submodule.Class'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True
    var_6 = 'module.Class.method'
    var_7 = module_0.is_public_family(var_6)
    assert var_7 is True
    var_8 = 'module.__init__'
    var_9 = module_0.is_public_family(var_8)
    assert var_9 is True
    var_10 = 'module.__init__.submodule'
    var_11 = module_0.is_public_family(var_10)
    assert var_11 is True
    var_12 = '__main__'
    var_13 = module_0.is_public_family(var_12)
    assert var_13 is True
    var_14 = '__main__.module'
    var_15 = module_0.is_public_family(var_14)
    assert var_15 is True
    var_16 = 'module.__dict__.submodule'
    var_17 = module_0.is_public_family(var_16)
    assert var_17 is True
    var_18 = '_private'
    var_19 = module_0.is_public_family(var_18)
    assert var_19 is False
    var_20 = 'module._private'
    var_21 = module_0.is_public_family(var_20)
    assert var_21 is False
    var_22 = 'module._private.Class'
    var_23 = module_0.is_public_family(var_22)
    assert var_23 is False
    var_24 = 'module.Class._private'
    var_25 = module_0.is_public_family(var_24)
    assert var_25 is False
    var_26 = '_module.public'
    var_27 = module_0.is_public_family(var_26)
    assert var_27 is False
    var_28 = 'module.normal_name'
    var_29 = module_0.is_public_family(var_28)
    assert var_29 is True
    var_30 = 'module.name_with_underscores'
    var_31 = module_0.is_public_family(var_30)
    assert var_31 is True
    var_32 = 'A'
    var_33 = module_0.is_public_family(var_32)
    assert var_33 is True
    var_34 = '_A'
    var_35 = module_0.is_public_family(var_34)
    assert var_35 is False
    var_36 = '__A__'
    var_37 = module_0.is_public_family(var_36)
    assert var_37 is True
    var_38 = 'module.__A'
    var_39 = module_0.is_public_family(var_38)
    assert var_39 is False
    var_40 = 'module.A_'
    var_41 = module_0.is_public_family(var_40)
    assert var_41 is True

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n'}
    var_3 = 'def func1(): pass'
    var_4 = var_0.parse(var_3, var_3)
    assert var_0.level == {'\n': 0, 'def func1(): pass': 0, 'def func1(): pass.func1': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', 'def func1(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def func1(): pass.func1': '### func1()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.imp == {'\n': {*()}, 'def func1(): pass': {*()}}
    assert var_0.root == {'\n': '\n', 'def func1(): pass': 'def func1(): pass', 'def func1(): pass.func1': 'def func1(): pass'}
    var_5 = var_0.compile()
    assert var_5 == '## Module `def func1(): pass`\n<a id="def func1(): pass"></a>\n\n### func1()\n\n*Full name:* `def func1(): pass.func1`\n<a id="def func1(): pass-func1"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_6 = var_0.__repr__()
    assert var_6 == 'Parser(link=True, b_level=1, toc=False, level={\'\\n\': 0, \'def func1(): pass\': 0, \'def func1(): pass.func1\': 0}, doc={\'\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'def func1(): pass\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'def func1(): pass.func1\': \'### func1()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| return |\\n|:------:|\\n| `Any` |\\n\\n\'}, docstring={}, imp={\'\\n\': set(), \'def func1(): pass\': set()}, root={\'\\n\': \'\\n\', \'def func1(): pass\': \'def func1(): pass\', \'def func1(): pass.func1\': \'def func1(): pass\'}, alias={}, const={})'
    var_7 = var_0.__post_init__()

def test_case_25():
    var_0 = 'def foo(): pass'
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
    assert var_1.level == {'def foo(): pass': 0, 'def foo(): pass.foo': 0}
    assert var_1.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_1.imp == {'def foo(): pass': {*()}}
    assert var_1.root == {'def foo(): pass': 'def foo(): pass', 'def foo(): pass.foo': 'def foo(): pass'}
    var_3 = var_1.compile()
    assert var_3 == '## Module `def foo(): pass`\n<a id="def foo(): pass"></a>\n\n### foo()\n\n*Full name:* `def foo(): pass.foo`\n<a id="def foo(): pass-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'

@pytest.mark.xfail(strict=True)
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
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n'}
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
    var_4 = var_3.__post_init__()
    var_5 = module_0.const_type(var_2)
    assert var_5 == 'Any'
    var_6 = module_0.Parser()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level == 1
    assert var_6.toc is False
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp == {}
    assert var_6.root == {}
    assert var_6.alias == {}
    assert var_6.const == {}
    var_7 = [var_4, var_2]
    var_8 = module_1.Subscript(*var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.Subscript'
    assert var_8.value is None
    assert var_8.slice is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_9 = ''
    var_3.resolve(var_4, var_8, var_9)

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
    var_1 = "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"
    var_2 = None
    var_3 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas.method": 4}
    assert var_0.doc == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas": '### class PublicClas\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr` | `int` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas.method": '#### PublicClas.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_0.docstring == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 'Module docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 'Function docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas": 'Class docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas.method": 'Method docstring.'}
    assert var_0.imp == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST"}}
    assert var_0.root == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas.method": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"}
    assert var_0.alias == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.os": 'os', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.List": 'typing.List', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.__all__": "['public_func', 'PUBLIC_CONST']", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": '42'}
    assert var_0.const == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": 'int'}
    var_4 = None
    var_5 = var_0.load_docstring(var_1, var_4)
    var_6 = '\r?:\x0b. i=T'
    var_7 = var_0.parse(var_6, var_1)
    assert var_0.level == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas.method": 4, '\r?:\x0b. i=T': 1, '\r?:\x0b. i=T.public_func': 1, '\r?:\x0b. i=T._private_func': 1, '\r?:\x0b. i=T.PublicClas': 1, '\r?:\x0b. i=T.PublicClas.method': 1}
    assert var_0.doc == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas": '### class PublicClas\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr` | `int` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas.method": '#### PublicClas.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n', '\r?:\x0b. i=T': '## Module `{}`\n<a id="{}"></a>\n\n', '\r?:\x0b. i=T.public_func': '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n', '\r?:\x0b. i=T._private_func': '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', '\r?:\x0b. i=T.PublicClas': '### class PublicClas\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr` | `int` |\n\n', '\r?:\x0b. i=T.PublicClas.method': '#### PublicClas.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_0.docstring == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 'Module docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 'Function docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas": 'Class docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas.method": 'Method docstring.', '\r?:\x0b. i=T': 'Module docstring.', '\r?:\x0b. i=T.public_func': 'Function docstring.', '\r?:\x0b. i=T.PublicClas': 'Class docstring.', '\r?:\x0b. i=T.PublicClas.method': 'Method docstring.'}
    assert var_0.imp == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST"}, '\r?:\x0b. i=T': {'\r?:\x0b. i=T.public_func', '\r?:\x0b. i=T.PUBLIC_CONST'}}
    assert var_0.root == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClas.method": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", '\r?:\x0b. i=T': '\r?:\x0b. i=T', '\r?:\x0b. i=T.PUBLIC_CONST': '\r?:\x0b. i=T', '\r?:\x0b. i=T.public_func': '\r?:\x0b. i=T', '\r?:\x0b. i=T._private_func': '\r?:\x0b. i=T', '\r?:\x0b. i=T.PublicClas': '\r?:\x0b. i=T', '\r?:\x0b. i=T.PublicClas.method': '\r?:\x0b. i=T'}
    assert var_0.alias == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.os": 'os', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.List": 'typing.List', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.__all__": "['public_func', 'PUBLIC_CONST']", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": '42', '\r?:\x0b. i=T.os': 'os', '\r?:\x0b. i=T.List': 'typing.List', '\r?:\x0b. i=T.__all__': "['public_func', 'PUBLIC_CONST']", '\r?:\x0b. i=T.PUBLIC_CONST': '42'}
    assert var_0.const == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": 'int', '\r?:\x0b. i=T.PUBLIC_CONST': 'int'}
    var_8 = var_0.__post_init__()
    var_9 = var_0.__eq__(var_2)
    var_10 = module_0.const_type(var_9)
    assert var_10 == 'Any'
    var_11 = var_0.compile()
    assert var_11 == '## Module `\r?:\x0b. i=T`\n<a id="\r?:\x0b- i=t"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `PUBLIC_CONST` | `int` |\n\nModule docstring.\n\n### public_func()\n\n*Full name:* `\r?:\x0b. i=T.public_func`\n<a id="\r?:\x0b- i=t-public_func"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\nFunction docstring.\n\n## Module `\n\'\'\'Module docstring.\'\'\'\nimport os\nfrom typing import List\n\n__all__ = [\'public_func\', \'PUBLIC_CONST\']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'Function docstring.\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    \'\'\'Class docstring.\'\'\'\n    attr: int = 10\n    \n    def method(self) -> None:\n        \'\'\'Method docstring.\'\'\'\n        pass\n`\n<a id="\n\'\'\'module docstring-\'\'\'\nimport os\nfrom typing import list\n\n__all__ = [\'public_func\', \'public_const\']\n\npublic_const: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'function docstring-\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass publicclas:\n    \'\'\'class docstring-\'\'\'\n    attr: int = 10\n    \n    def method(self) -> none:\n        \'\'\'method docstring-\'\'\'\n        pass\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `PUBLIC_CONST` | `int` |\n\nModule docstring.\n\n### public_func()\n\n*Full name:* `\n\'\'\'Module docstring.\'\'\'\nimport os\nfrom typing import List\n\n__all__ = [\'public_func\', \'PUBLIC_CONST\']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'Function docstring.\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    \'\'\'Class docstring.\'\'\'\n    attr: int = 10\n    \n    def method(self) -> None:\n        \'\'\'Method docstring.\'\'\'\n        pass\n.public_func`\n<a id="\n\'\'\'module docstring-\'\'\'\nimport os\nfrom typing import list\n\n__all__ = [\'public_func\', \'public_const\']\n\npublic_const: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'function docstring-\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass publicclas:\n    \'\'\'class docstring-\'\'\'\n    attr: int = 10\n    \n    def method(self) -> none:\n        \'\'\'method docstring-\'\'\'\n        pass\n-public_func"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\nFunction docstring.\n'

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
    var_1 = "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 4, "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 4}
    assert var_0.doc == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr` | `int` |\n\n', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": '#### PublicClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_0.docstring == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 'Module docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": 'Function docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 'Class docstring.', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 'Method docstring.'}
    assert var_0.imp == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST"}}
    assert var_0.root == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"}
    assert var_0.alias == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.os": 'os', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.List": 'typing.List', "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.__all__": "['public_func', 'PUBLIC_CONST']", "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": '42'}
    assert var_0.const == {"\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST": 'int'}
    var_3 = '32)v&d|l3"88(\rQh:a'
    var_4 = None
    var_5 = var_0.load_docstring(var_3, var_4)
    var_6 = var_0.compile()
    assert var_6 == '## Module `\n\'\'\'Module docstring.\'\'\'\nimport os\nfrom typing import List\n\n__all__ = [\'public_func\', \'PUBLIC_CONST\']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'Function docstring.\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    \'\'\'Class docstring.\'\'\'\n    attr: int = 10\n    \n    def method(self) -> None:\n        \'\'\'Method docstring.\'\'\'\n        pass\n`\n<a id="\n\'\'\'module docstring-\'\'\'\nimport os\nfrom typing import list\n\n__all__ = [\'public_func\', \'public_const\']\n\npublic_const: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'function docstring-\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass publicclass:\n    \'\'\'class docstring-\'\'\'\n    attr: int = 10\n    \n    def method(self) -> none:\n        \'\'\'method docstring-\'\'\'\n        pass\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `PUBLIC_CONST` | `int` |\n\nModule docstring.\n\n### public_func()\n\n*Full name:* `\n\'\'\'Module docstring.\'\'\'\nimport os\nfrom typing import List\n\n__all__ = [\'public_func\', \'PUBLIC_CONST\']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'Function docstring.\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    \'\'\'Class docstring.\'\'\'\n    attr: int = 10\n    \n    def method(self) -> None:\n        \'\'\'Method docstring.\'\'\'\n        pass\n.public_func`\n<a id="\n\'\'\'module docstring-\'\'\'\nimport os\nfrom typing import list\n\n__all__ = [\'public_func\', \'public_const\']\n\npublic_const: int = 42\n\ndef public_func(x: int) -> str:\n    \'\'\'function docstring-\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass publicclass:\n    \'\'\'class docstring-\'\'\'\n    attr: int = 10\n    \n    def method(self) -> none:\n        \'\'\'method docstring-\'\'\'\n        pass\n-public_func"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\nFunction docstring.\n'

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
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
    var_3 = module_0.code(var_1)
    assert var_3 == '`\n`'
    var_4 = 'def foo(): pass'
    var_5 = var_2.parse(var_1, var_4)
    assert var_2.level == {'\n': 0, '\n.foo': 0}
    assert var_2.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_2.imp == {'\n': {*()}}
    assert var_2.root == {'\n': '\n', '\n.foo': '\n'}
    var_6 = True
    var_7 = None
    var_8 = var_2.__post_init__()
    var_9 = var_0.globals(var_7, var_8)
    var_10 = module_0.Parser(toc=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 1
    assert var_10.toc is True
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    var_11 = var_10.__repr__()
    assert var_11 == 'Parser(link=True, b_level=1, toc=True, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_12 = var_10.parse(var_4, var_4)
    assert var_10.level == {'def foo(): pass': 0, 'def foo(): pass.foo': 0}
    assert var_10.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_10.imp == {'def foo(): pass': {*()}}
    assert var_10.root == {'def foo(): pass': 'def foo(): pass', 'def foo(): pass.foo': 'def foo(): pass'}
    var_13 = var_10.compile()
    assert var_13 == '**Table of contents:**\n+ [`def foo(): pass`](#def foo(): pass)\n    + [`def foo(): pass.foo`](#def foo(): pass-foo)\n\n## Module `def foo(): pass`\n<a id="def foo(): pass"></a>\n\n### foo()\n\n*Full name:* `def foo(): pass.foo`\n<a id="def foo(): pass-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_14 = module_0.Parser()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'apimd.parser.Parser'
    assert var_14.link is True
    assert var_14.b_level == 1
    assert var_14.toc is False
    assert var_14.level == {}
    assert var_14.doc == {}
    assert var_14.docstring == {}
    assert var_14.imp == {}
    assert var_14.root == {}
    assert var_14.alias == {}
    assert var_14.const == {}
    var_15 = 'pkg.module2'
    var_16 = var_14.parse(var_15, var_1)
    assert var_14.level == {'pkg.module2': 1}
    assert var_14.doc == {'pkg.module2': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_14.imp == {'pkg.module2': {*()}}
    assert var_14.root == {'pkg.module2': 'pkg.module2'}
    var_17 = 'Z24.|2'
    var_18 = module_0.doctest(var_17)
    assert var_18 == 'Z24.|2'
    var_19 = var_14.__repr__()
    assert var_19 == 'Parser(link=True, b_level=1, toc=False, level={\'pkg.module2\': 1}, doc={\'pkg.module2\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={}, imp={\'pkg.module2\': set()}, root={\'pkg.module2\': \'pkg.module2\'}, alias={}, const={})'
    var_20 = module_0.const_type(var_19)
    assert var_20 == 'Any'
    var_21 = var_18.__eq__(var_5)
    var_22 = var_0.load_docstring(var_16, var_16)
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
    var_24 = 'COE.\x0cNST = 42'
    var_25 = var_23.parse(var_1, var_24)
    assert var_23.level == {'\n': 0}
    assert var_23.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_23.imp == {'\n': {*()}}
    assert var_23.root == {'\n': '\n'}
    var_26 = module_2.dataclass(eq=var_21, frozen=var_19)
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_27 = var_10.compile()
    assert var_27 == '**Table of contents:**\n+ [`def foo(): pass`](#def foo(): pass)\n    + [`def foo(): pass.foo`](#def foo(): pass-foo)\n\n## Module `def foo(): pass`\n<a id="def foo(): pass"></a>\n\n### foo()\n\n*Full name:* `def foo(): pass.foo`\n<a id="def foo(): pass-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_28 = var_23.compile()
    assert var_28 == '\n'
    var_29 = module_0.Parser()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'apimd.parser.Parser'
    assert var_29.link is True
    assert var_29.b_level == 1
    assert var_29.toc is False
    assert var_29.level == {}
    assert var_29.doc == {}
    assert var_29.docstring == {}
    assert var_29.imp == {}
    assert var_29.root == {}
    assert var_29.alias == {}
    assert var_29.const == {}
    var_30 = var_29.compile()
    assert var_30 == '\n'
    var_31 = module_0.Parser(var_6)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'apimd.parser.Parser'
    assert var_31.link is True
    assert var_31.b_level == 1
    assert var_31.toc is False
    assert var_31.level == {}
    assert var_31.doc == {}
    assert var_31.docstring == {}
    assert var_31.imp == {}
    assert var_31.root == {}
    assert var_31.alias == {}
    assert var_31.const == {}
    var_32 = var_31.compile()
    assert var_32 == '\n'
    var_33 = module_0.Parser()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'apimd.parser.Parser'
    assert var_33.link is True
    assert var_33.b_level == 1
    assert var_33.toc is False
    assert var_33.level == {}
    assert var_33.doc == {}
    assert var_33.docstring == {}
    assert var_33.imp == {}
    assert var_33.root == {}
    assert var_33.alias == {}
    assert var_33.const == {}
    var_34 = var_33.compile()
    assert var_34 == '\n'
    var_35 = module_0.Parser()
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'apimd.parser.Parser'
    assert var_35.link is True
    assert var_35.b_level == 1
    assert var_35.toc is False
    assert var_35.level == {}
    assert var_35.doc == {}
    assert var_35.docstring == {}
    assert var_35.imp == {}
    assert var_35.root == {}
    assert var_35.alias == {}
    assert var_35.const == {}
    var_36 = '__all__ = ["func1"]'
    var_37 = var_35.parse(var_4, var_36)
    assert var_35.level == {'def foo(): pass': 0}
    assert var_35.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_35.imp == {'def foo(): pass': {'def foo(): pass.func1'}}
    assert var_35.root == {'def foo(): pass': 'def foo(): pass'}
    assert var_35.alias == {'def foo(): pass.__all__': "['func1']"}
    var_38 = var_26.__repr__()
    var_29.imports(var_12, var_15)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
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
    var_3 = module_0.code(var_1)
    assert var_3 == '`\n`'
    var_4 = 'def foo(): pass'
    var_5 = var_2.__post_init__()
    var_6 = True
    var_7 = module_0.Parser(toc=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'apimd.parser.Parser'
    assert var_7.link is True
    assert var_7.b_level == 1
    assert var_7.toc is True
    assert var_7.level == {}
    assert var_7.doc == {}
    assert var_7.docstring == {}
    assert var_7.imp == {}
    assert var_7.root == {}
    assert var_7.alias == {}
    assert var_7.const == {}
    var_8 = var_7.__repr__()
    assert var_8 == 'Parser(link=True, b_level=1, toc=True, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_9 = var_7.parse(var_4, var_4)
    assert var_7.level == {'def foo(): pass': 0, 'def foo(): pass.foo': 0}
    assert var_7.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_7.imp == {'def foo(): pass': {*()}}
    assert var_7.root == {'def foo(): pass': 'def foo(): pass', 'def foo(): pass.foo': 'def foo(): pass'}
    var_10 = module_0.const_type(var_8)
    assert var_10 == 'Any'
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
    var_12 = 'pkg.module2'
    var_13 = var_11.parse(var_12, var_1)
    assert var_11.level == {'pkg.module2': 1}
    assert var_11.doc == {'pkg.module2': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_11.imp == {'pkg.module2': {*()}}
    assert var_11.root == {'pkg.module2': 'pkg.module2'}
    var_14 = 'Z24.|2'
    var_15 = module_0.doctest(var_14)
    assert var_15 == 'Z24.|2'
    var_16 = var_11.__repr__()
    assert var_16 == 'Parser(link=True, b_level=1, toc=False, level={\'pkg.module2\': 1}, doc={\'pkg.module2\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={}, imp={\'pkg.module2\': set()}, root={\'pkg.module2\': \'pkg.module2\'}, alias={}, const={})'
    var_17 = module_0.const_type(var_16)
    assert var_17 == 'Any'
    var_18 = var_15.__eq__(var_5)
    var_19 = var_0.load_docstring(var_13, var_13)
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
    var_21 = 'CONST = 42'
    var_22 = var_20.parse(var_1, var_21)
    assert var_20.level == {'\n': 0}
    assert var_20.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_20.imp == {'\n': {*()}}
    assert var_20.root == {'\n': '\n', '\n.CONST': '\n'}
    assert var_20.alias == {'\n.CONST': '42'}
    assert var_20.const == {'\n.CONST': 'int'}
    var_23 = module_2.dataclass(eq=var_18, frozen=var_16)
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_24 = var_7.compile()
    assert var_24 == '**Table of contents:**\n+ [`def foo(): pass`](#def foo(): pass)\n    + [`def foo(): pass.foo`](#def foo(): pass-foo)\n\n## Module `def foo(): pass`\n<a id="def foo(): pass"></a>\n\n### foo()\n\n*Full name:* `def foo(): pass.foo`\n<a id="def foo(): pass-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_25 = var_20.compile()
    assert var_25 == '## Module `\n`\n<a id="\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONST` | `int` |\n'
    var_26 = module_0.Parser()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'apimd.parser.Parser'
    assert var_26.link is True
    assert var_26.b_level == 1
    assert var_26.toc is False
    assert var_26.level == {}
    assert var_26.doc == {}
    assert var_26.docstring == {}
    assert var_26.imp == {}
    assert var_26.root == {}
    assert var_26.alias == {}
    assert var_26.const == {}
    var_27 = var_26.compile()
    assert var_27 == '\n'
    var_28 = module_0.Parser(var_6)
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
    var_29 = var_28.compile()
    assert var_29 == '\n'
    var_30 = module_0.Parser()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'apimd.parser.Parser'
    assert var_30.link is True
    assert var_30.b_level == 1
    assert var_30.toc is False
    assert var_30.level == {}
    assert var_30.doc == {}
    assert var_30.docstring == {}
    assert var_30.imp == {}
    assert var_30.root == {}
    assert var_30.alias == {}
    assert var_30.const == {}
    var_31 = var_30.compile()
    assert var_31 == '\n'
    var_32 = module_0.Parser()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'apimd.parser.Parser'
    assert var_32.link is True
    assert var_32.b_level == 1
    assert var_32.toc is False
    assert var_32.level == {}
    assert var_32.doc == {}
    assert var_32.docstring == {}
    assert var_32.imp == {}
    assert var_32.root == {}
    assert var_32.alias == {}
    assert var_32.const == {}
    var_33 = '__all__ =A ["func1"]'
    var_34 = var_32.parse(var_4, var_33)
    assert var_32.level == {'def foo(): pass': 0}
    assert var_32.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_32.imp == {'def foo(): pass': {*()}}
    assert var_32.root == {'def foo(): pass': 'def foo(): pass'}
    assert var_32.alias == {'def foo(): pass.__all__': "A['func1']"}
    var_35 = var_23.__repr__()
    var_36 = var_32.load_docstring(var_4, var_29)
    var_26.imports(var_9, var_12)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = None
    var_3 = ']ev'
    var_4 = True
    var_5 = 'w,eR0'
    var_6 = 'sbs0/tAw$}LM'
    var_7 = {var_3: var_4, var_5: var_4, var_6: var_4, var_1: var_4}
    var_8 = {var_3: var_3}
    var_9 = module_0.Parser(var_2, level=var_7, docstring=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is None
    assert var_9.b_level == 1
    assert var_9.toc is False
    assert var_9.level == {']ev': True, 'w,eR0': True, 'sbs0/tAw$}LM': True, '\n': True}
    assert var_9.doc == {}
    assert var_9.docstring == {']ev': ']ev'}
    assert var_9.imp == {}
    assert var_9.root == {}
    assert var_9.alias == {}
    assert var_9.const == {}
    var_10 = module_0.code(var_6)
    assert var_10 == '`sbs0/tAw$}LM`'
    var_11 = 'def foo(): pass'
    var_12 = 'pkg.module2'
    var_13 = var_9.parse(var_12, var_11)
    assert var_9.level == {']ev': True, 'w,eR0': True, 'sbs0/tAw$}LM': True, '\n': True, 'pkg.module2': 1, 'pkg.module2.foo': 1}
    assert var_9.doc == {'pkg.module2': '## Module `{}`\n\n', 'pkg.module2.foo': '### foo()\n\n*Full name:* `{}`\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_9.imp == {'pkg.module2': {*()}}
    assert var_9.root == {'pkg.module2': 'pkg.module2', 'pkg.module2.foo': 'pkg.module2'}
    module_0.doctest(var_2)

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
    var_1 = 'test_module'
    var_2 = '\nclass TestClass:\n    pass\n'
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
    var_6 = var_5.bases
    var_7 = var_5.body
    var_8 = var_0.class_api(var_1, var_7, var_6, var_7)
    var_9 = '\nclass TestClass(BaseClass):\n    pass\n'
    var_10 = module_0.Parser()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 1
    assert var_10.toc is False
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    var_11 = module_1.parse(var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Module'
    assert f'{type(var_11.body).__module__}.{type(var_11.body).__qualname__}' == 'builtins.list'
    assert len(var_11.body) == 1
    assert var_11.type_ignores == []
    var_12 = var_11.body[var_4]
    var_13 = var_12.bases
    var_14 = var_12.body
    var_10.class_api(var_1, var_9, var_13, var_14)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = 'x = 1\ny = 2'
    var_1 = module_1.parse(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'ast.Module'
    assert f'{type(var_1.body).__module__}.{type(var_1.body).__qualname__}' == 'builtins.list'
    assert len(var_1.body) == 2
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
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8]
    var_10 = 'if True:\n    x = 1\nelse:\n    y = 2'
    var_11 = module_1.parse(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Module'
    assert f'{type(var_11.body).__module__}.{type(var_11.body).__qualname__}' == 'builtins.list'
    assert len(var_11.body) == 1
    assert var_11.type_ignores == []
    var_12 = var_11.body
    var_13 = module_0.walk_body(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'if True:\n    if False:\n        x = 1\n    y = 2'
    var_17 = module_1.parse(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'ast.Module'
    assert f'{type(var_17.body).__module__}.{type(var_17.body).__qualname__}' == 'builtins.list'
    assert len(var_17.body) == 1
    assert var_17.type_ignores == []
    var_18 = var_17.body
    var_19 = list(var_2)
    var_20 = len(var_19)
    assert var_20 == 2
    module_1.parse(var_20)

def test_case_34():
    var_0 = 'x = 1\ny = 2'
    var_1 = module_1.parse(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'ast.Module'
    assert f'{type(var_1.body).__module__}.{type(var_1.body).__qualname__}' == 'builtins.list'
    assert len(var_1.body) == 2
    assert var_1.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_2 = var_1.body
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = 1
    var_8 = 'if True:\n    x = 1\nelse:\n    y = 2'
    var_9 = module_1.parse(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'ast.Module'
    assert f'{type(var_9.body).__module__}.{type(var_9.body).__qualname__}' == 'builtins.list'
    assert len(var_9.body) == 1
    assert var_9.type_ignores == []
    var_10 = var_9.body
    var_11 = module_0.walk_body(var_10)
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'if True:\n    if False:\n        x = 1\n    y = 2'
    var_15 = module_1.parse(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Module'
    assert f'{type(var_15.body).__module__}.{type(var_15.body).__qualname__}' == 'builtins.list'
    assert len(var_15.body) == 1
    assert var_15.type_ignores == []
    var_16 = var_15.body
    var_17 = module_0.walk_body(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 'try:\n    x = 1\nexcept:\n    y = 2\nfinally:\n    z = 3'
    var_21 = module_1.parse(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'ast.Module'
    assert f'{type(var_21.body).__module__}.{type(var_21.body).__qualname__}' == 'builtins.list'
    assert len(var_21.body) == 1
    assert var_21.type_ignores == []
    var_22 = var_21.body
    var_23 = module_0.walk_body(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 3
    var_26 = 'try:\n    x = 1\nexcept ValueError:\n    y = 2\nexcept KeyError:\n    z = 3'
    var_27 = module_1.parse(var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'ast.Module'
    assert f'{type(var_27.body).__module__}.{type(var_27.body).__qualname__}' == 'builtins.list'
    assert len(var_27.body) == 1
    assert var_27.type_ignores == []
    var_28 = var_27.body
    var_29 = module_0.walk_body(var_28)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 3
    var_32 = 'try:\n    x = 1\nexcept:\n    y = 2\nelse:\n    z = 3'
    var_33 = module_1.parse(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'ast.Module'
    assert f'{type(var_33.body).__module__}.{type(var_33.body).__qualname__}' == 'builtins.list'
    assert len(var_33.body) == 1
    assert var_33.type_ignores == []
    var_34 = var_33.body
    var_35 = module_0.walk_body(var_34)
    var_36 = list(var_35)
    var_37 = len(var_36)
    assert var_37 == 3
    var_38 = []
    var_39 = module_0.walk_body(var_38)
    var_40 = list(var_39)
    var_41 = len(var_40)
    assert var_41 == 0
    var_42 = 'x = 1\nif True:\n    y = 2\nz = 3'
    var_43 = module_1.parse(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'ast.Module'
    assert f'{type(var_43.body).__module__}.{type(var_43.body).__qualname__}' == 'builtins.list'
    assert len(var_43.body) == 3
    assert var_43.type_ignores == []
    var_44 = var_43.body
    var_45 = module_0.walk_body(var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 3
    var_48 = var_46[var_5]
    var_49 = var_46[var_7]
    var_50 = 2
    var_51 = var_46[var_50]
    var_52 = 'if True:\n    try:\n        x = 1\n    except:\n        y = 2\n    z = 3'
    var_53 = module_1.parse(var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'ast.Module'
    assert f'{type(var_53.body).__module__}.{type(var_53.body).__qualname__}' == 'builtins.list'
    assert len(var_53.body) == 1
    assert var_53.type_ignores == []
    var_54 = var_53.body
    var_55 = module_0.walk_body(var_54)
    var_56 = list(var_55)
    var_57 = len(var_56)
    assert var_57 == 3

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
    var_1 = '\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n': 0}
    assert var_0.doc == {'\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n': {*()}}
    assert var_0.root == {'\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n': '\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n', '\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n.VAR': '\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n', '\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n.CONST': '\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n'}
    assert var_0.alias == {'\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n.Optional': 'typing.Optional', '\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n.VAR': 'None', '\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n.CONST': "'value'"}
    assert var_0.const == {'\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n.VAR': 'int | None', '\nfrom typing import Optional\n\nVAR: Optional[int] = None\nCONST: str = "value"\n.CONST': 'str'}

def test_case_36():
    var_0 = 'Test Resolver.visit_Subscript method.'
    var_1 = 'test_module'
    var_2 = 'typing.Union'
    var_3 = {var_0: var_2}
    var_4 = module_0.Resolver(var_1, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Resolver'
    assert var_4.root == 'test_module'
    assert var_4.alias == {'Test Resolver.visit_Subscript method.': 'typing.Union'}
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
    var_5 = 0
    var_6 = 'Union[int, str]'
    var_7 = module_1.parse(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Module'
    assert f'{type(var_7.body).__module__}.{type(var_7.body).__qualname__}' == 'builtins.list'
    assert len(var_7.body) == 1
    assert var_7.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = var_7.body[var_5]
    var_9 = var_8.value
    var_10 = 'test_module.Optional'
    var_11 = 'typing.Optional'
    var_12 = {var_10: var_11}
    var_13 = module_0.Resolver(var_1, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'apimd.parser.Resolver'
    assert var_13.root == 'test_module'
    assert var_13.alias == {'test_module.Optional': 'typing.Optional'}
    assert var_13.self_ty == ''
    var_14 = 'Optional[int]'
    var_15 = module_1.parse(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Module'
    assert f'{type(var_15.body).__module__}.{type(var_15.body).__qualname__}' == 'builtins.list'
    assert len(var_15.body) == 1
    assert var_15.type_ignores == []
    var_16 = var_15.body[var_5]
    var_17 = var_16.value
    var_18 = 'test_module.Dict'
    var_19 = 'typing.Dict'
    var_20 = {var_18: var_19}
    var_21 = module_0.Resolver(var_1, var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'apimd.parser.Resolver'
    assert var_21.root == 'test_module'
    assert var_21.alias == {'test_module.Dict': 'typing.Dict'}
    assert var_21.self_ty == ''
    var_22 = 'Dict[str, int]'
    var_23 = module_1.parse(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'ast.Module'
    assert f'{type(var_23.body).__module__}.{type(var_23.body).__qualname__}' == 'builtins.list'
    assert len(var_23.body) == 1
    assert var_23.type_ignores == []
    var_24 = var_23.body[var_5]
    var_25 = var_24.value
    var_26 = {}
    var_27 = module_0.Resolver(var_1, var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'apimd.parser.Resolver'
    assert var_27.root == 'test_module'
    assert var_27.alias == {}
    assert var_27.self_ty == ''
    var_28 = 'some_func()[int]'
    var_29 = module_1.parse(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'ast.Module'
    assert f'{type(var_29.body).__module__}.{type(var_29.body).__qualname__}' == 'builtins.list'
    assert len(var_29.body) == 1
    assert var_29.type_ignores == []
    var_30 = var_29.body[var_5]
    var_31 = var_30.value
    var_32 = {}
    var_33 = module_0.Resolver(var_1, var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'apimd.parser.Resolver'
    assert var_33.root == 'test_module'
    assert var_33.alias == {}
    assert var_33.self_ty == ''
    var_34 = var_13.generic_visit(var_24)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'ast.Expr'
    assert f'{type(var_34.value).__module__}.{type(var_34.value).__qualname__}' == 'ast.Subscript'
    assert var_34.lineno == 1
    assert var_34.col_offset == 0
    assert var_34.end_lineno == 1
    assert var_34.end_col_offset == 14
    with pytest.raises(AttributeError):
        var_35 = var_34.body[var_5]

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
    var_1 = 'nested_module'
    var_2 = "\nclass OuterClass:\n    '''Outer class.'''\n    \n    class InnerClass:\n        '''Inner class.'''\n        pass\n"
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'nested_module': 0, 'nested_module.OuterClass': 0, 'nested_module.OuterClass.InnerClass': 0}
    assert var_0.doc == {'nested_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'nested_module.OuterClass': '### class OuterClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'nested_module.OuterClass.InnerClass': '#### class OuterClass.InnerClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.docstring == {'nested_module.OuterClass': 'Outer class.', 'nested_module.OuterClass.InnerClass': 'Inner class.'}
    assert var_0.imp == {'nested_module': {*()}}
    assert var_0.root == {'nested_module': 'nested_module', 'nested_module.OuterClass': 'nested_module', 'nested_module.OuterClass.InnerClass': 'nested_module'}

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
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
    var_3 = module_0.code(var_1)
    assert var_3 == '`\n`'
    var_4 = None
    var_5 = [var_4, var_4]
    var_6 = module_1.AnnAssign(*var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.AnnAssign'
    assert var_6.target is None
    assert var_6.annotation is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.AnnAssign.value is None
    var_7 = var_0.globals(var_4, var_6)
    var_8 = '>Fvb\t>qaJg'
    var_0.parse(var_4, var_8)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
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
    var_3 = module_0.code(var_1)
    assert var_3 == '`\n`'
    var_4 = 'def foo(): pass'
    var_5 = None
    var_6 = [var_5, var_5]
    var_7 = module_1.AnnAssign(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.AnnAssign'
    assert var_7.target is None
    assert var_7.annotation is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.AnnAssign.value is None
    var_8 = var_0.globals(var_5, var_7)
    var_9 = True
    var_10 = module_0.Parser(toc=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 1
    assert var_10.toc is True
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    var_11 = var_10.__repr__()
    assert var_11 == 'Parser(link=True, b_level=1, toc=True, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_12 = var_10.parse(var_4, var_4)
    assert var_10.level == {'def foo(): pass': 0, 'def foo(): pass.foo': 0}
    assert var_10.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_10.imp == {'def foo(): pass': {*()}}
    assert var_10.root == {'def foo(): pass': 'def foo(): pass', 'def foo(): pass.foo': 'def foo(): pass'}
    var_13 = var_10.compile()
    assert var_13 == '**Table of contents:**\n+ [`def foo(): pass`](#def foo(): pass)\n    + [`def foo(): pass.foo`](#def foo(): pass-foo)\n\n## Module `def foo(): pass`\n<a id="def foo(): pass"></a>\n\n### foo()\n\n*Full name:* `def foo(): pass.foo`\n<a id="def foo(): pass-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_14 = module_0.Parser()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'apimd.parser.Parser'
    assert var_14.link is True
    assert var_14.b_level == 1
    assert var_14.toc is False
    assert var_14.level == {}
    assert var_14.doc == {}
    assert var_14.docstring == {}
    assert var_14.imp == {}
    assert var_14.root == {}
    assert var_14.alias == {}
    assert var_14.const == {}
    var_15 = 'pkg.module2'
    var_16 = var_14.parse(var_15, var_1)
    assert var_14.level == {'pkg.module2': 1}
    assert var_14.doc == {'pkg.module2': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_14.imp == {'pkg.module2': {*()}}
    assert var_14.root == {'pkg.module2': 'pkg.module2'}
    var_17 = 'Z24.|2'
    var_18 = module_0.const_type(var_12)
    assert var_18 == 'Any'
    var_19 = module_0.doctest(var_17)
    assert var_19 == 'Z24.|2'
    var_20 = var_14.__repr__()
    assert var_20 == 'Parser(link=True, b_level=1, toc=False, level={\'pkg.module2\': 1}, doc={\'pkg.module2\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={}, imp={\'pkg.module2\': set()}, root={\'pkg.module2\': \'pkg.module2\'}, alias={}, const={})'
    var_21 = module_0.const_type(var_20)
    assert var_21 == 'Any'
    var_22 = var_19.__eq__(var_20)
    assert var_22 is False
    var_23 = var_0.load_docstring(var_16, var_16)
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
    var_25 = 'CONST =4'
    var_26 = var_24.parse(var_1, var_25)
    assert var_24.level == {'\n': 0}
    assert var_24.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_24.imp == {'\n': {*()}}
    assert var_24.root == {'\n': '\n', '\n.CONST': '\n'}
    assert var_24.alias == {'\n.CONST': '4'}
    assert var_24.const == {'\n.CONST': 'int'}
    var_27 = module_2.dataclass(eq=var_22, frozen=var_20)
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_28 = var_10.compile()
    assert var_28 == '**Table of contents:**\n+ [`def foo(): pass`](#def foo(): pass)\n    + [`def foo(): pass.foo`](#def foo(): pass-foo)\n\n## Module `def foo(): pass`\n<a id="def foo(): pass"></a>\n\n### foo()\n\n*Full name:* `def foo(): pass.foo`\n<a id="def foo(): pass-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_29 = var_24.compile()
    assert var_29 == '## Module `\n`\n<a id="\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONST` | `int` |\n'
    var_30 = module_0.Parser()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'apimd.parser.Parser'
    assert var_30.link is True
    assert var_30.b_level == 1
    assert var_30.toc is False
    assert var_30.level == {}
    assert var_30.doc == {}
    assert var_30.docstring == {}
    assert var_30.imp == {}
    assert var_30.root == {}
    assert var_30.alias == {}
    assert var_30.const == {}
    var_31 = var_2.compile()
    assert var_31 == '\n'
    var_32 = module_0.Parser(var_9)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'apimd.parser.Parser'
    assert var_32.link is True
    assert var_32.b_level == 1
    assert var_32.toc is False
    assert var_32.level == {}
    assert var_32.doc == {}
    assert var_32.docstring == {}
    assert var_32.imp == {}
    assert var_32.root == {}
    assert var_32.alias == {}
    assert var_32.const == {}
    var_33 = var_32.compile()
    assert var_33 == '\n'
    var_34 = module_0.Parser()
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'apimd.parser.Parser'
    assert var_34.link is True
    assert var_34.b_level == 1
    assert var_34.toc is False
    assert var_34.level == {}
    assert var_34.doc == {}
    assert var_34.docstring == {}
    assert var_34.imp == {}
    assert var_34.root == {}
    assert var_34.alias == {}
    assert var_34.const == {}
    var_35 = var_34.compile()
    assert var_35 == '\n'
    var_36 = module_0.Parser()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'apimd.parser.Parser'
    assert var_36.link is True
    assert var_36.b_level == 1
    assert var_36.toc is False
    assert var_36.level == {}
    assert var_36.doc == {}
    assert var_36.docstring == {}
    assert var_36.imp == {}
    assert var_36.root == {}
    assert var_36.alias == {}
    assert var_36.const == {}
    var_37 = '__all__ = ["func1"]'
    var_38 = var_36.parse(var_4, var_37)
    assert var_36.level == {'def foo(): pass': 0}
    assert var_36.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_36.imp == {'def foo(): pass': {'def foo(): pass.func1'}}
    assert var_36.root == {'def foo(): pass': 'def foo(): pass'}
    assert var_36.alias == {'def foo(): pass.__all__': "['func1']"}
    var_39 = var_27.__repr__()
    var_40 = var_24.parse(var_31, var_25)
    var_30.imports(var_12, var_15)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
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
    var_3 = module_0.code(var_1)
    assert var_3 == '`\n`'
    var_4 = 'def foo(): pass'
    var_5 = None
    var_6 = [var_5, var_5]
    var_7 = module_1.AnnAssign(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.AnnAssign'
    assert var_7.target is None
    assert var_7.annotation is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.AnnAssign.value is None
    var_8 = var_0.globals(var_5, var_7)
    var_9 = True
    var_10 = module_0.Parser(toc=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 1
    assert var_10.toc is True
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    var_11 = var_10.__repr__()
    assert var_11 == 'Parser(link=True, b_level=1, toc=True, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_12 = var_10.parse(var_4, var_4)
    assert var_10.level == {'def foo(): pass': 0, 'def foo(): pass.foo': 0}
    assert var_10.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_10.imp == {'def foo(): pass': {*()}}
    assert var_10.root == {'def foo(): pass': 'def foo(): pass', 'def foo(): pass.foo': 'def foo(): pass'}
    var_13 = var_10.compile()
    assert var_13 == '**Table of contents:**\n+ [`def foo(): pass`](#def foo(): pass)\n    + [`def foo(): pass.foo`](#def foo(): pass-foo)\n\n## Module `def foo(): pass`\n<a id="def foo(): pass"></a>\n\n### foo()\n\n*Full name:* `def foo(): pass.foo`\n<a id="def foo(): pass-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_14 = module_0.Parser()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'apimd.parser.Parser'
    assert var_14.link is True
    assert var_14.b_level == 1
    assert var_14.toc is False
    assert var_14.level == {}
    assert var_14.doc == {}
    assert var_14.docstring == {}
    assert var_14.imp == {}
    assert var_14.root == {}
    assert var_14.alias == {}
    assert var_14.const == {}
    var_15 = 'pkg.module2'
    var_16 = var_14.parse(var_15, var_1)
    assert var_14.level == {'pkg.module2': 1}
    assert var_14.doc == {'pkg.module2': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_14.imp == {'pkg.module2': {*()}}
    assert var_14.root == {'pkg.module2': 'pkg.module2'}
    var_17 = 'Z24.|2'
    var_18 = module_0.const_type(var_12)
    assert var_18 == 'Any'
    var_19 = module_0.doctest(var_17)
    assert var_19 == 'Z24.|2'
    var_20 = var_14.__repr__()
    assert var_20 == 'Parser(link=True, b_level=1, toc=False, level={\'pkg.module2\': 1}, doc={\'pkg.module2\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={}, imp={\'pkg.module2\': set()}, root={\'pkg.module2\': \'pkg.module2\'}, alias={}, const={})'
    var_21 = module_0.const_type(var_20)
    assert var_21 == 'Any'
    var_22 = var_19.__eq__(var_20)
    assert var_22 is False
    var_23 = var_0.load_docstring(var_16, var_16)
    var_24 = 'CONST =4'
    var_25 = var_0.parse(var_1, var_24)
    assert var_0.level == {'\n': 0}
    assert var_0.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'\n': {*()}}
    assert var_0.root == {'\n': '\n', '\n.CONST': '\n'}
    assert var_0.alias == {'\n.CONST': '4'}
    assert var_0.const == {'\n.CONST': 'int'}
    var_26 = module_2.dataclass(eq=var_22, frozen=var_20)
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_27 = var_10.compile()
    assert var_27 == '**Table of contents:**\n+ [`def foo(): pass`](#def foo(): pass)\n    + [`def foo(): pass.foo`](#def foo(): pass-foo)\n\n## Module `def foo(): pass`\n<a id="def foo(): pass"></a>\n\n### foo()\n\n*Full name:* `def foo(): pass.foo`\n<a id="def foo(): pass-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_28 = var_0.compile()
    assert var_28 == '## Module `\n`\n<a id="\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONST` | `int` |\n'
    var_29 = module_0.Parser()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'apimd.parser.Parser'
    assert var_29.link is True
    assert var_29.b_level == 1
    assert var_29.toc is False
    assert var_29.level == {}
    assert var_29.doc == {}
    assert var_29.docstring == {}
    assert var_29.imp == {}
    assert var_29.root == {}
    assert var_29.alias == {}
    assert var_29.const == {}
    var_30 = var_2.compile()
    assert var_30 == '\n'
    var_31 = module_0.Parser(var_9)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'apimd.parser.Parser'
    assert var_31.link is True
    assert var_31.b_level == 1
    assert var_31.toc is False
    assert var_31.level == {}
    assert var_31.doc == {}
    assert var_31.docstring == {}
    assert var_31.imp == {}
    assert var_31.root == {}
    assert var_31.alias == {}
    assert var_31.const == {}
    var_32 = var_31.compile()
    assert var_32 == '\n'
    var_33 = module_0.Parser()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'apimd.parser.Parser'
    assert var_33.link is True
    assert var_33.b_level == 1
    assert var_33.toc is False
    assert var_33.level == {}
    assert var_33.doc == {}
    assert var_33.docstring == {}
    assert var_33.imp == {}
    assert var_33.root == {}
    assert var_33.alias == {}
    assert var_33.const == {}
    var_34 = var_33.compile()
    assert var_34 == '\n'
    var_35 = module_0.Parser()
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'apimd.parser.Parser'
    assert var_35.link is True
    assert var_35.b_level == 1
    assert var_35.toc is False
    assert var_35.level == {}
    assert var_35.doc == {}
    assert var_35.docstring == {}
    assert var_35.imp == {}
    assert var_35.root == {}
    assert var_35.alias == {}
    assert var_35.const == {}
    var_36 = '__all__ = ["fun1"@t]'
    var_37 = var_35.parse(var_4, var_36)
    assert var_35.level == {'def foo(): pass': 0}
    assert var_35.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_35.imp == {'def foo(): pass': {*()}}
    assert var_35.root == {'def foo(): pass': 'def foo(): pass'}
    assert var_35.alias == {'def foo(): pass.__all__': "['fun1' @ t]"}
    var_38 = var_26.__repr__()
    var_39 = var_35.load_docstring(var_4, var_32)
    var_29.imports(var_12, var_15)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
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
    var_3 = module_0.code(var_1)
    assert var_3 == '`\n`'
    var_4 = 'def foo(): pass'
    var_5 = None
    var_6 = [var_5, var_5]
    var_7 = module_1.AnnAssign(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.AnnAssign'
    assert var_7.target is None
    assert var_7.annotation is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.AnnAssign.value is None
    var_8 = module_3.getdoc(var_2)
    assert var_8 == 'AST parser.\n\nUsage:\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n\nOr create with parameters:\n>>> p = Parser.new(link=True, level=1)'
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
    var_9 = True
    var_10 = module_0.Parser(toc=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 1
    assert var_10.toc is True
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    var_11 = var_10.parse(var_4, var_4)
    assert var_10.level == {'def foo(): pass': 0, 'def foo(): pass.foo': 0}
    assert var_10.doc == {'def foo(): pass': '## Module `{}`\n<a id="{}"></a>\n\n', 'def foo(): pass.foo': '### foo()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_10.imp == {'def foo(): pass': {*()}}
    assert var_10.root == {'def foo(): pass': 'def foo(): pass', 'def foo(): pass.foo': 'def foo(): pass'}
    var_12 = var_10.compile()
    assert var_12 == '**Table of contents:**\n+ [`def foo(): pass`](#def foo(): pass)\n    + [`def foo(): pass.foo`](#def foo(): pass-foo)\n\n## Module `def foo(): pass`\n<a id="def foo(): pass"></a>\n\n### foo()\n\n*Full name:* `def foo(): pass.foo`\n<a id="def foo(): pass-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
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
    var_14 = 'pkg.module2'
    var_15 = var_13.parse(var_14, var_1)
    assert var_13.level == {'pkg.module2': 1}
    assert var_13.doc == {'pkg.module2': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_13.imp == {'pkg.module2': {*()}}
    assert var_13.root == {'pkg.module2': 'pkg.module2'}
    var_16 = 'Z24.|2'
    var_17 = module_0.const_type(var_11)
    assert var_17 == 'Any'
    var_18 = module_0.doctest(var_16)
    assert var_18 == 'Z24.|2'
    var_19 = var_13.__repr__()
    assert var_19 == 'Parser(link=True, b_level=1, toc=False, level={\'pkg.module2\': 1}, doc={\'pkg.module2\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\'}, docstring={}, imp={\'pkg.module2\': set()}, root={\'pkg.module2\': \'pkg.module2\'}, alias={}, const={})'
    var_20 = module_0.const_type(var_19)
    assert var_20 == 'Any'
    var_21 = module_0.doctest(var_8)
    assert var_21 == 'AST parser.\n\nUsage:\n```python\n>>> p = Parser()\n>>> with open("pkg_path", \'r\') as f:\n>>>     p.parse(\'pkg_name\', f.read())\n>>> s = p.compile()\n```\n\nOr create with parameters:\n```python\n>>> p = Parser.new(link=True, level=1)\n```'
    var_22 = var_0.load_docstring(var_15, var_15)
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
    var_24 = 'CONST =4'
    var_25 = var_23.parse(var_1, var_24)
    assert var_23.level == {'\n': 0}
    assert var_23.doc == {'\n': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_23.imp == {'\n': {*()}}
    assert var_23.root == {'\n': '\n', '\n.CONST': '\n'}
    assert var_23.alias == {'\n.CONST': '4'}
    assert var_23.const == {'\n.CONST': 'int'}
    var_26 = module_2.dataclass(eq=var_21, frozen=var_19)
    assert f'{type(module_2.MISSING).__module__}.{type(module_2.MISSING).__qualname__}' == 'dataclasses._MISSING_TYPE'
    assert f'{type(module_2.KW_ONLY).__module__}.{type(module_2.KW_ONLY).__qualname__}' == 'dataclasses._KW_ONLY_TYPE'
    var_27 = var_10.compile()
    assert var_27 == '**Table of contents:**\n+ [`def foo(): pass`](#def foo(): pass)\n    + [`def foo(): pass.foo`](#def foo(): pass-foo)\n\n## Module `def foo(): pass`\n<a id="def foo(): pass"></a>\n\n### foo()\n\n*Full name:* `def foo(): pass.foo`\n<a id="def foo(): pass-foo"></a>\n\n| return |\n|:------:|\n| `Any` |\n'
    var_28 = var_23.compile()
    assert var_28 == '## Module `\n`\n<a id="\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `CONST` | `int` |\n'
    var_29 = module_0.Parser()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'apimd.parser.Parser'
    assert var_29.link is True
    assert var_29.b_level == 1
    assert var_29.toc is False
    assert var_29.level == {}
    assert var_29.doc == {}
    assert var_29.docstring == {}
    assert var_29.imp == {}
    assert var_29.root == {}
    assert var_29.alias == {}
    assert var_29.const == {}
    var_30 = var_2.compile()
    assert var_30 == '\n'
    var_31 = module_0.Parser(var_9)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'apimd.parser.Parser'
    assert var_31.link is True
    assert var_31.b_level == 1
    assert var_31.toc is False
    assert var_31.level == {}
    assert var_31.doc == {}
    assert var_31.docstring == {}
    assert var_31.imp == {}
    assert var_31.root == {}
    assert var_31.alias == {}
    assert var_31.const == {}
    var_32 = var_31.compile()
    assert var_32 == '\n'
    var_33 = module_0.Parser()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'apimd.parser.Parser'
    assert var_33.link is True
    assert var_33.b_level == 1
    assert var_33.toc is False
    assert var_33.level == {}
    assert var_33.doc == {}
    assert var_33.docstring == {}
    assert var_33.imp == {}
    assert var_33.root == {}
    assert var_33.alias == {}
    assert var_33.const == {}
    var_34 = var_33.compile()
    assert var_34 == '\n'
    var_35 = var_29.__repr__()
    assert var_35 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_35.load_docstring(var_17, var_5)

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
    var_1 = 'test_module'
    var_2 = '\nx = 42  # type: int\ny = "hello"  # type: str\n'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module'}
    assert var_0.alias == {'test_module.x': '42', 'test_module.y': "'hello'"}

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
    var_1 = "\nasync def async_func():\n    '''Async function.'''\n    pass\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\nasync def async_func():\n    '''Async function.'''\n    pass\n": 1, "\nasync def async_func():\n    '''Async function.'''\n    pass\n.async_func": 1}
    assert var_0.doc == {"\nasync def async_func():\n    '''Async function.'''\n    pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\nasync def async_func():\n    '''Async function.'''\n    pass\n.async_func": '### async async_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {"\nasync def async_func():\n    '''Async function.'''\n    pass\n.async_func": 'Async function.'}
    assert var_0.imp == {"\nasync def async_func():\n    '''Async function.'''\n    pass\n": {*()}}
    assert var_0.root == {"\nasync def async_func():\n    '''Async function.'''\n    pass\n": "\nasync def async_func():\n    '''Async function.'''\n    pass\n", "\nasync def async_func():\n    '''Async function.'''\n    pass\n.async_func": "\nasync def async_func():\n    '''Async function.'''\n    pass\n"}

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
    var_1 = None
    var_2 = [var_1, var_1]
    var_3 = {}
    var_4 = module_1.Call(*var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Call'
    assert var_4.func is None
    assert var_4.args is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = module_0.const_type(var_4)
    assert var_5 == 'Any'

def test_case_45():
    var_0 = 'Unit test for Parser.parse method.'
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
    var_2 = None
    var_3 = var_1.load_docstring(var_0, var_2)
    var_4 = "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"
    var_5 = var_1.parse(var_4, var_4)
    assert var_1.level == {"\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 4, "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.eublic_func": 4, "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": 4, "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 4, "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 4}
    assert var_1.doc == {"\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.eublic_func": '### eublic_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n', "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": '### class PublicClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr` | `int` |\n\n', "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": '#### PublicClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | return |\n|:----:|:------:|\n| `Self` | `None` |\n\n'}
    assert var_1.docstring == {"\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": 'Module docstrin.', "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.eublic_func": 'Function docstring.', "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": 'Class docstring.', "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": 'Method docstring.'}
    assert var_1.imp == {"\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": {"\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.public_func", "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CONST"}}
    assert var_1.root == {"\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n": "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CNST": "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.eublic_func": "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n._private_func": "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass": "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n", "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PublicClass.method": "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"}
    assert var_1.alias == {"\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.os": 'os', "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.List": 'typing.List', "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.__all__": "['public_func', 'PUBLIC_CONST']", "\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CNST": '42'}
    assert var_1.const == {"\n'''Module docstrin.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n.PUBLIC_CNST": 'int'}
    var_6 = module_0.const_type(var_5)
    assert var_6 == 'Any'
    var_7 = var_1.compile()
    assert var_7 == '## Module `\n\'\'\'Module docstrin.\'\'\'\nimport os\nfrom typing import List\n\n__all__ = [\'public_func\', \'PUBLIC_CONST\']\n\nPUBLIC_CNST: int = 42\n\ndef eublic_func(x: int) -> str:\n    \'\'\'Function docstring.\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClass:\n    \'\'\'Class docstring.\'\'\'\n    attr: int = 10\n    \n    def method(self) -> None:\n        \'\'\'Method docstring.\'\'\'\n        pass\n`\n<a id="\n\'\'\'module docstrin-\'\'\'\nimport os\nfrom typing import list\n\n__all__ = [\'public_func\', \'public_const\']\n\npublic_cnst: int = 42\n\ndef eublic_func(x: int) -> str:\n    \'\'\'function docstring-\'\'\'\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass publicclass:\n    \'\'\'class docstring-\'\'\'\n    attr: int = 10\n    \n    def method(self) -> none:\n        \'\'\'method docstring-\'\'\'\n        pass\n"></a>\n\nModule docstrin.\n'

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
    var_1 = None
    var_2 = var_0.load_docstring(var_1, var_1)
    var_3 = [var_1, var_1]
    var_4 = 'X\\27:{`!v\x0bUC8M'
    var_5 = '>dk9VUb+H'
    var_6 = {var_4: var_2, var_5: var_1}
    var_7 = module_1.List(*var_3, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.List'
    assert var_7.elts is None
    assert var_7.ctx is None
    assert var_7.X\27:{`!vUC8M is None
    assert var_7.>dk9VUb+H is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_8 = module_0.const_type(var_7)
    assert var_8 == 'list'
    var_9 = var_0.compile()
    assert var_9 == '\n'
    module_0.is_magic(var_1)

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
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass'
    var_3 = 'class Base: pass'
    var_4 = 0
    var_5 = module_1.parse(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Module'
    assert f'{type(var_5.body).__module__}.{type(var_5.body).__qualname__}' == 'builtins.list'
    assert len(var_5.body) == 1
    assert var_5.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_6 = var_5.body[var_4]
    var_7 = var_6.bases
    var_8 = []
    var_9 = var_0.class_api(var_1, var_2, var_7, var_8)
    var_10 = module_0.Parser()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 1
    assert var_10.toc is False
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    var_11 = 'test_module'
    var_12 = 'test_module.TestClass'
    var_13 = []
    var_14 = '\nclass TestClass:\n    public_attr: int\n    _private_attr: str\n    '
    var_15 = module_1.parse(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Module'
    assert f'{type(var_15.body).__module__}.{type(var_15.body).__qualname__}' == 'builtins.list'
    assert len(var_15.body) == 1
    assert var_15.type_ignores == []
    var_16 = var_15.body[var_4]
    var_17 = var_16.body
    var_10.class_api(var_11, var_12, var_13, var_17)

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
    var_1 = "\n'''Module docstring.'''\nimport os\nfrom typing import List\n\n__all__ = ['public_func', 'PUBLIC_CONST']\n\nPUBLIC_CONST: int = 42\n\ndef public_func(x: int) -> str:\n    '''Function docstring.'''\n    return str(x)\n\ndef _private_func():\n    pass\n\nclass PublicClas:\n    '''Class docstring.'''\n    attr: int = 10\n    \n    def method(self) -> None:\n        '''Method docstring.'''\n        pass\n"
    var_2 = 'qifVm"K'
    var_3 = 'B:X'
    var_4 = var_0.parse(var_2, var_3)
    assert var_0.level == {'qifVm"K': 0}
    assert var_0.doc == {'qifVm"K': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'qifVm"K': {*()}}
    assert var_0.root == {'qifVm"K': 'qifVm"K'}
    var_5 = var_0.__eq__(var_4)
    var_6 = var_0.load_docstring(var_1, var_5)
    var_7 = 'FH,mMjS^g/|,.OE'
    var_5.parse(var_7, var_5)

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
    var_1 = 'test_module'
    var_2 = 'test_module.test_func'
    var_3 = '\ndef test_func(x: int, y: str = "default", *args, z: float = 1.0, **kwargs) -> bool:\n    pass\n'
    var_4 = var_0.parse(var_1, var_3)
    assert var_0.level == {'test_module': 0, 'test_module.test_func': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.test_func': '### test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | *args | z | **kwargs | return |\n|:---:|:---:|:-----:|:---:|:--------:|:------:|\n| `int` | `str` | `Any` | `float` | `Any` | `bool` |\n|   | `\'default\'` |   | `1.0` |   |   |\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.test_func': 'test_module'}
    var_5 = 0
    var_6 = module_1.parse(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Module'
    assert f'{type(var_6.body).__module__}.{type(var_6.body).__qualname__}' == 'builtins.list'
    assert len(var_6.body) == 1
    assert var_6.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = var_6.body[var_5]
    var_8 = var_7.args
    var_9 = var_7.returns
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_1, var_2, var_8, var_9, has_self=var_10, cls_method=var_11)
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.test_func': '### test_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | *args | z | **kwargs | return |\n|:---:|:---:|:-----:|:---:|:--------:|:------:|\n| `int` | `str` | `Any` | `float` | `Any` | `bool` |\n|   | `\'default\'` |   | `1.0` |   |   |\n\n| x | y | *args | z | **kwargs | return |\n|:---:|:---:|:-----:|:---:|:--------:|:------:|\n| `int` | `str` | `Any` | `float` | `Any` | `bool` |\n|   | `\'default\'` |   | `1.0` |   |   |\n\n'}

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
    var_2 = 'test_module.func_kwonly'
    var_3 = '\ndef func_kwonly(x: int, *, y: str, z: float = 1.0) -> None:\n    pass\n'
    var_4 = var_0.parse(var_1, var_3)
    assert var_0.level == {'test_module': 0, 'test_module.func_kwonly': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.func_kwonly': '### func_kwonly()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | * | y | z | return |\n|:---:|:---:|:---:|:---:|:------:|\n| `int` |   | `str` | `float` | `None` |\n|   |   |   | `1.0` |   |\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.func_kwonly': 'test_module'}
    var_5 = 0
    var_6 = module_1.parse(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Module'
    assert f'{type(var_6.body).__module__}.{type(var_6.body).__qualname__}' == 'builtins.list'
    assert len(var_6.body) == 1
    assert var_6.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = var_6.body[var_5]
    var_8 = var_7.args
    var_9 = var_7.returns
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_1, var_2, var_8, var_9, has_self=var_10, cls_method=var_11)
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.func_kwonly': '### func_kwonly()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | * | y | z | return |\n|:---:|:---:|:---:|:---:|:------:|\n| `int` |   | `str` | `float` | `None` |\n|   |   |   | `1.0` |   |\n\n| x | * | y | z | return |\n|:---:|:---:|:---:|:---:|:------:|\n| `int` |   | `str` | `float` | `None` |\n|   |   |   | `1.0` |   |\n\n'}

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
    var_1 = 'test_module'
    var_2 = 'test_module.func_posonly'
    var_3 = '\ndef func_posonly(x: int, /, y: str) -> None:\n    pass\n'
    var_4 = var_0.parse(var_1, var_3)
    assert var_0.level == {'test_module': 0, 'test_module.func_posonly': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.func_posonly': '### func_posonly()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | / | y | return |\n|:---:|:---:|:---:|:------:|\n| `int` | `Any` | `str` | `None` |\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.func_posonly': 'test_module'}
    var_5 = 0
    var_6 = module_1.parse(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Module'
    assert f'{type(var_6.body).__module__}.{type(var_6.body).__qualname__}' == 'builtins.list'
    assert len(var_6.body) == 1
    assert var_6.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = var_6.body[var_5]
    var_8 = var_7.args
    var_9 = var_7.returns
    var_10 = False
    var_11 = False
    var_12 = var_0.func_api(var_1, var_2, var_8, var_9, has_self=var_10, cls_method=var_11)
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.func_posonly': '### func_posonly()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | / | y | return |\n|:---:|:---:|:---:|:------:|\n| `int` | `Any` | `str` | `None` |\n\n| x | / | y | return |\n|:---:|:---:|:---:|:------:|\n| `int` | `Any` | `str` | `None` |\n\n'}

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
    var_1 = 'test_module'
    var_2 = 'test_module.TestClass.test_classmethod'
    var_3 = '\nclass TestClass:\n    @classmethod\n    def test_classmethod(cls, x: int) -> str:\n        pass\n'
    var_4 = var_0.parse(var_1, var_3)
    assert var_0.level == {'test_module': 0, 'test_module.TestClass': 0, 'test_module.TestClass.test_classmethod': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.TestClass.test_classmethod': '#### TestClass.test_classmethod()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `str` |\n\n'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.TestClass': 'test_module', 'test_module.TestClass.test_classmethod': 'test_module'}
    var_5 = 0
    var_6 = module_1.parse(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.Module'
    assert f'{type(var_6.body).__module__}.{type(var_6.body).__qualname__}' == 'builtins.list'
    assert len(var_6.body) == 1
    assert var_6.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_7 = var_6.body[var_5]
    var_8 = var_7.body[var_5]
    var_9 = var_8.args
    var_10 = var_8.returns
    var_11 = True
    var_12 = var_0.func_api(var_1, var_2, var_9, var_10, has_self=var_11, cls_method=var_11)
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.TestClass': '### class TestClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'test_module.TestClass.test_classmethod': '#### TestClass.test_classmethod()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `str` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `str` |\n\n'}