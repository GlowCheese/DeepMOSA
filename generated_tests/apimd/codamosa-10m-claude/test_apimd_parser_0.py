# Check out: https://github.com/GlowCheese/deepmosa
import ast as module_1
import inspect as module_2

import apimd.parser as module_0
import pytest


def test_case_0():
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
    var_1 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 4}
    assert var_0.doc == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_0.docstring == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 'Module docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 'Function docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 'Class docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': {*()}}
    assert var_0.root == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'}
    assert var_0.alias == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.os': 'os', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.List': 'typing.List', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.Optional': 'typing.Optional', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '42'}
    assert var_0.const == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': 'int'}

def test_case_2():
    var_0 = '1Tn0>MXvKp'
    var_1 = True
    var_2 = None
    var_3 = module_0.Parser(toc=var_1, imp=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'apimd.parser.Parser'
    assert var_3.link is True
    assert var_3.b_level == 1
    assert var_3.toc is True
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
    var_1 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 4}
    assert var_0.doc == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_0.docstring == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 'Module docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 'Function docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 'Class docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': {*()}}
    assert var_0.root == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'}
    assert var_0.alias == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.os': 'os', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.List': 'typing.List', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.Optional': 'typing.Optional', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '42'}
    assert var_0.const == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': 'int'}
    var_3 = var_0.load_docstring(var_1, var_0)

def test_case_5():
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

def test_case_6():
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
    var_2 = module_1.NodeTransformer()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'ast.NodeTransformer'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096

def test_case_7():
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
    var_1 = '\nclass MyClass:\n    x: int\n    y: str\n    del x\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\nclass MyClass:\n    x: int\n    y: str\n    del x\n': 0, '\nclass MyClass:\n    x: int\n    y: str\n    del x\n.MyClass': 0}
    assert var_0.doc == {'\nclass MyClass:\n    x: int\n    y: str\n    del x\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\nclass MyClass:\n    x: int\n    y: str\n    del x\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `y` | `str` |\n\n'}
    assert var_0.imp == {'\nclass MyClass:\n    x: int\n    y: str\n    del x\n': {*()}}
    assert var_0.root == {'\nclass MyClass:\n    x: int\n    y: str\n    del x\n': '\nclass MyClass:\n    x: int\n    y: str\n    del x\n', '\nclass MyClass:\n    x: int\n    y: str\n    del x\n.MyClass': '\nclass MyClass:\n    x: int\n    y: str\n    del x\n'}

def test_case_9():
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_1 = "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n": 3, "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass": 3, "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.static_method": 3, "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.class_method": 3}
    assert var_0.doc == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass": '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.static_method": '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.class_method": '#### MyClass.class_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\n'}
    assert var_0.docstring == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass": 'Test class.', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.static_method": 'Static method.', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.class_method": 'Class method.'}
    assert var_0.imp == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n": {*()}}
    assert var_0.root == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n": "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n", "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass": "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n", "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.static_method": "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n", "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.class_method": "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"}
    assert var_0.alias == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.staticmethod": 'functools.staticmethod', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.classmethod": 'functools.classmethod'}

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
    var_1 = "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n": 2, "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.public_func": 2, "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n._private_func": 2}
    assert var_0.doc == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.public_func": 'Public function.', "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n._private_func": 'Private function.'}
    assert var_0.imp == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n": {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.public_func"}}
    assert var_0.root == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n": "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n", "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.public_func": "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n", "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n._private_func": "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n"}
    assert var_0.alias == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.__all__": "['public_func']"}

def test_case_11():
    var_0 = 'U%e|L3cs,e,,'
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_3 = module_1.Call(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Call'
    assert var_3.func is None
    assert var_3.args is None
    assert var_3.keywords is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_4 = module_0.code(var_0)
    assert var_4 == '<code>U%e&#124;L3cs,e,,</code>'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_5 = module_0.const_type(var_3)
    assert var_5 == 'Any'

def test_case_12():
    var_0 = 'U%.|L3cs,e,p'
    var_1 = None
    var_2 = [var_1, var_1, var_1]
    var_3 = module_1.Call(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'ast.Call'
    assert var_3.func is None
    assert var_3.args is None
    assert var_3.keywords is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_4 = module_0.const_type(var_3)
    assert var_4 == 'Any'
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
    assert f'{type(module_0.logger.handlers).__module__}.{type(module_0.logger.handlers).__qualname__}' == 'builtins.list'
    assert len(module_0.logger.handlers) == 2
    assert module_0.logger.disabled is False
    assert module_0.PEP585 == {'typing.Tuple': 'tuple', 'typing.List': 'list', 'typing.Dict': 'dict', 'typing.Set': 'set', 'typing.FrozenSet': 'frozenset', 'typing.Type': 'type', 'typing.Deque': 'collections.deque', 'typing.DefaultDict': 'collections.defaultdict', 'typing.OrderedDict': 'collections.OrderedDict', 'typing.Counter': 'collections.Counter', 'typing.ChainMap': 'collections.ChainMap', 'typing.Awaitable': 'collections.abc.Awaitable', 'typing.Coroutine': 'collections.abc.Coroutine', 'typing.AsyncIterable': 'collections.abc.AsyncIterable', 'typing.AsyncIterator': 'collections.abc.AsyncIterator', 'typing.Iterable': 'collections.abc.Iterable', 'typing.Iterator': 'collections.abc.Iterator', 'typing.Generator': 'collections.abc.Generator', 'typing.Reversible': 'collections.abc.Reversible', 'typing.Container': 'collections.abc.Container', 'typing.Collection': 'collections.abc.Collection', 'typing.AbstractSet': 'collections.abc.Set', 'typing.MutableSet': 'collections.abc.MutableSet', 'typing.Mapping': 'collections.abc.Mapping', 'typing.MutableMapping': 'collections.abc.MutableMapping', 'typing.Sequence': 'collections.abc.Sequence', 'typing.MutableSequence': 'collections.abc.MutableSequence', 'typing.ByteString': 'collections.abc.ByteString', 'typing.MappingView': 'collections.abc.MappingView', 'typing.KeysView': 'collections.abc.KeysView', 'typing.ItemsView': 'collections.abc.ItemsView', 'typing.ValuesView': 'collections.abc.ValuesView', 'typing.ContextManager': 'contextlib.AbstractContextManager', 'typing.AsyncContextManager': 'contextlib.AsyncContextManager', 'typing.Pattern': 're.Pattern', 'typing.re.Pattern': 're.Pattern', 'typing.Match': 're.Match', 'typing.re.Match': 're.Match'}
    assert module_0.ANY == 'Any'
    var_5 = module_0.doctest(var_0)
    assert var_5 == 'U%.|L3cs,e,p'

@pytest.mark.xfail(strict=True)
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
    var_1 = None
    var_0.imports(var_1, var_1)

@pytest.mark.xfail(strict=True)
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
    var_1 = '\nclass MyClass:\n    value = 42  # type: int\n    name = "test"  # type: str\n'
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
    var_5 = 'test_module.MyClass'
    var_6 = var_4.bases
    var_7 = var_4.body
    var_0.class_api(var_5, var_5, var_6, var_7)

def test_case_15():
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
    var_8 = module_1.NodeTransformer()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'ast.NodeTransformer'

@pytest.mark.xfail(strict=True)
def test_case_16():
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
    var_1 = '1Q=<PjiH/[qM,'
    var_2 = var_0.globals(var_1, var_0)
    var_3 = var_0.__post_init__()
    var_4 = var_0.compile()
    assert var_4 == '\n'

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
    var_0 = module_0.Parser()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'apimd.parser.Parser'
    assert var_0.link is True
    assert var_0.b_level == 1
    assert var_0.toc is False
    assert var_0.level == {}
    assert var_0.doc == {}
    assert var_0.docstring == {}
    assert var_0.imp == {}
    assert var_0.root == {}
    assert var_0.alias == {}
    assert var_0.const == {}
    assert f'{type(module_0.logger).__module__}.{type(module_0.logger).__qualname__}' == 'logging.RootLogger'
    assert module_0.logger.filters == []
    assert module_0.logger.name == 'root'
    assert module_0.logger.level == 10
    assert module_0.logger.parent is None
    assert module_0.logger.propagate is True
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
    var_2 = '\nclass MyClass:\n    count: int\n    name: str\n    _internal: bool\n    value: float = 3.14\n'
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
    var_6 = 'test_module.MyClass'
    var_7 = var_5.bases
    var_8 = var_5.body
    var_0.class_api(var_1, var_6, var_7, var_8)

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
    var_10 = module_0.Parser(toc=var_4, docstring=var_0, alias=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level == 1
    assert var_10.toc is None
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring is None
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == {'_1fd\n%OB|D^Zq6V': '', 'Q<j\x0bs5,?<CQ': 'r2Eru>o6_|lJI3Kw'}
    assert var_10.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_11 = var_10.__eq__(var_6)
    var_12 = module_0.code(var_6)
    assert var_12 == ' '
    var_13 = 'L_3\x0cy)$=~(d\x0b;("`K+'
    var_14 = module_0.is_public_family(var_13)
    assert var_14 is True
    var_15 = module_0.doctest(var_1)
    assert var_15 == '_1fd\n%OB|D^Zq6V'
    var_16 = "+'%`|\t[O`n_vnG}?n$"
    var_17 = module_0.is_public_family(var_16)
    assert var_17 is True
    var_18 = module_0.const_type(var_5)
    assert var_18 == 'Any'
    var_19 = module_2.getdoc(var_0)
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
    var_20 = module_0.Parser(b_level=var_4, level=var_5, doc=var_3, imp=var_19, const=var_5)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'apimd.parser.Parser'
    assert var_20.link is True
    assert var_20.b_level is None
    assert var_20.toc is False
    assert var_20.level is None
    assert var_20.doc == {'<code>_1fd\n%OB&#124;D^Zq6V</code>': '<code>_1fd\n%OB&#124;D^Zq6V</code>', '_1fd\n%OB|D^Zq6V': '_1fd\n%OB|D^Zq6V'}
    assert var_20.docstring == {}
    assert var_20.imp is None
    assert var_20.root == {}
    assert var_20.alias == {}
    assert var_20.const is None
    var_21 = var_20.globals(var_4, var_0)
    var_22 = module_2.getdoc(var_4)
    var_20.imports(var_4, var_22)

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
    var_1 = "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n": 3, "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass": 3, "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.static_method": 3, "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.class_method": 3}
    assert var_0.doc == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass": '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.static_method": '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.class_method": '#### MyClass.class_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\n'}
    assert var_0.docstring == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass": 'Test class.', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.static_method": 'Static method.', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.class_method": 'Class method.'}
    assert var_0.imp == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n": {*()}}
    assert var_0.root == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n": "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n", "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass": "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n", "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.static_method": "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n", "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.MyClass.class_method": "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"}
    assert var_0.alias == {"\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.staticmethod": 'functools.staticmethod', "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n.classmethod": 'functools.classmethod'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    \'\'\'Test class.\'\'\'\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        \'\'\'Static method.\'\'\'\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        \'\'\'Class method.\'\'\'\n        return x\n`\n<a id="\nfrom functools import staticmethod, classmethod\n\nclass myclass:\n    \'\'\'test class-\'\'\'\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        \'\'\'static method-\'\'\'\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        \'\'\'class method-\'\'\'\n        return x\n"></a>\n\n### class MyClass\n\n*Full name:* `\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    \'\'\'Test class.\'\'\'\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        \'\'\'Static method.\'\'\'\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        \'\'\'Class method.\'\'\'\n        return x\n.MyClass`\n<a id="\nfrom functools import staticmethod, classmethod\n\nclass myclass:\n    \'\'\'test class-\'\'\'\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        \'\'\'static method-\'\'\'\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        \'\'\'class method-\'\'\'\n        return x\n-myclass"></a>\n\nTest class.\n\n#### MyClass.class_method()\n\n*Full name:* `\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    \'\'\'Test class.\'\'\'\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        \'\'\'Static method.\'\'\'\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        \'\'\'Class method.\'\'\'\n        return x\n.MyClass.class_method`\n<a id="\nfrom functools import staticmethod, classmethod\n\nclass myclass:\n    \'\'\'test class-\'\'\'\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        \'\'\'static method-\'\'\'\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        \'\'\'class method-\'\'\'\n        return x\n-myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    \'\'\'Test class.\'\'\'\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        \'\'\'Static method.\'\'\'\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        \'\'\'Class method.\'\'\'\n        return x\n.MyClass.static_method`\n<a id="\nfrom functools import staticmethod, classmethod\n\nclass myclass:\n    \'\'\'test class-\'\'\'\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        \'\'\'static method-\'\'\'\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        \'\'\'class method-\'\'\'\n        return x\n-myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\nStatic method.\n'

@pytest.mark.xfail(strict=True)
def test_case_22():
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
    var_4 = None
    var_5 = ''
    var_6 = 'r2Eru>o6_|lJI3Kw'
    var_7 = {var_1: var_5, var_4: var_6}
    var_8 = module_0.Parser(var_0, docstring=var_0, alias=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is None
    assert var_8.b_level == 1
    assert var_8.toc is False
    assert var_8.level == {}
    assert var_8.doc == {}
    assert var_8.docstring is None
    assert var_8.imp == {}
    assert var_8.root == {}
    assert var_8.alias is None
    assert var_8.const == {}
    assert module_0.Parser.link is True
    assert module_0.Parser.b_level == 1
    assert module_0.Parser.toc is False
    assert f'{type(module_0.Parser.new).__module__}.{type(module_0.Parser.new).__qualname__}' == 'builtins.method'
    var_9 = module_0.Parser(toc=var_4, docstring=var_0, alias=var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'apimd.parser.Parser'
    assert var_9.link is True
    assert var_9.b_level == 1
    assert var_9.toc is None
    assert var_9.level == {}
    assert var_9.doc == {}
    assert var_9.docstring is None
    assert var_9.imp == {}
    assert var_9.root == {}
    assert var_9.alias == {'_1fd\n%OB|D^Zq6V': '', None: 'r2Eru>o6_|lJI3Kw'}
    assert var_9.const == {}
    var_10 = var_9.compile()
    assert var_10 == '\n'
    var_11 = module_0.code(var_3)
    assert var_11 == '<code>_1fd\n%OB&#124;D^Zq6V</code>'
    var_12 = 'L_3\x0cy)$=~(d\x0b;("`K+'
    var_13 = module_0.is_public_family(var_12)
    assert var_13 is True
    var_14 = module_0.doctest(var_1)
    assert var_14 == '_1fd\n%OB|D^Zq6V'
    var_15 = var_8.__eq__(var_0)
    var_16 = "+'%`|\t[O`n_vnG}?n$"
    var_17 = module_0.is_public_family(var_16)
    assert var_17 is True
    var_18 = module_0.const_type(var_0)
    assert var_18 == 'Any'
    var_19 = 'l"2+,$^m&'
    var_8.parse(var_14, var_19)

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    var_3 = 'x = 1'
    var_4 = module_1.parse(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Module'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'builtins.list'
    assert len(var_4.body) == 1
    assert var_4.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = var_4.body
    var_6 = module_0.walk_body(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = 'if True:\n    x = 1\nelse:\n    y = 2'
    var_12 = module_1.parse(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Module'
    assert f'{type(var_12.body).__module__}.{type(var_12.body).__qualname__}' == 'builtins.list'
    assert len(var_12.body) == 1
    assert var_12.type_ignores == []
    var_13 = var_12.body
    var_14 = module_0.walk_body(var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 2
    module_1.parse(var_15)

def test_case_24():
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
    var_3 = 'x = 1'
    var_4 = module_1.parse(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Module'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'builtins.list'
    assert len(var_4.body) == 1
    assert var_4.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = var_4.body
    var_6 = module_0.walk_body(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = 'if True:\n    x = 1\nelse:\n    y = 2'
    var_12 = module_1.parse(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'ast.Module'
    assert f'{type(var_12.body).__module__}.{type(var_12.body).__qualname__}' == 'builtins.list'
    assert len(var_12.body) == 1
    assert var_12.type_ignores == []
    var_13 = var_12.body
    var_14 = module_0.walk_body(var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = 'if True:\n    if False:\n        x = 1\n    y = 2'
    var_18 = module_1.parse(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'ast.Module'
    assert f'{type(var_18.body).__module__}.{type(var_18.body).__qualname__}' == 'builtins.list'
    assert len(var_18.body) == 1
    assert var_18.type_ignores == []
    var_19 = var_18.body
    var_20 = module_0.walk_body(var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = 'try:\n    x = 1\nexcept:\n    y = 2\nfinally:\n    z = 3'
    var_24 = module_1.parse(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'ast.Module'
    assert f'{type(var_24.body).__module__}.{type(var_24.body).__qualname__}' == 'builtins.list'
    assert len(var_24.body) == 1
    assert var_24.type_ignores == []
    var_25 = var_24.body
    var_26 = module_0.walk_body(var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 3
    var_29 = 'try:\n    x = 1\nexcept:\n    y = 2\nelse:\n    z = 3'
    var_30 = module_1.parse(var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'ast.Module'
    assert f'{type(var_30.body).__module__}.{type(var_30.body).__qualname__}' == 'builtins.list'
    assert len(var_30.body) == 1
    assert var_30.type_ignores == []
    var_31 = var_30.body
    var_32 = module_0.walk_body(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 3
    var_35 = 'if True:\n    try:\n        x = 1\n    except:\n        y = 2\nelse:\n    z = 3'
    var_36 = module_1.parse(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'ast.Module'
    assert f'{type(var_36.body).__module__}.{type(var_36.body).__qualname__}' == 'builtins.list'
    assert len(var_36.body) == 1
    assert var_36.type_ignores == []
    var_37 = var_36.body
    var_38 = module_0.walk_body(var_37)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 3
    var_41 = 'x = 1\ny = 2\nz = 3'
    var_42 = module_1.parse(var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'ast.Module'
    assert f'{type(var_42.body).__module__}.{type(var_42.body).__qualname__}' == 'builtins.list'
    assert len(var_42.body) == 3
    assert var_42.type_ignores == []
    var_43 = var_42.body
    var_44 = module_0.walk_body(var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 3
    var_47 = 'if True:\n    x = 1\n    y = 2\nelse:\n    z = 3\n    w = 4'
    var_48 = module_1.parse(var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'ast.Module'
    assert f'{type(var_48.body).__module__}.{type(var_48.body).__qualname__}' == 'builtins.list'
    assert len(var_48.body) == 1
    assert var_48.type_ignores == []
    var_49 = var_48.body
    var_50 = module_0.walk_body(var_49)
    var_51 = list(var_50)
    var_52 = len(var_51)
    assert var_52 == 4
    var_53 = 'try:\n    x = 1\nexcept ValueError:\n    y = 2\nexcept TypeError:\n    z = 3'
    var_54 = module_1.parse(var_53)
    assert f'{type(var_54).__module__}.{type(var_54).__qualname__}' == 'ast.Module'
    assert f'{type(var_54.body).__module__}.{type(var_54.body).__qualname__}' == 'builtins.list'
    assert len(var_54.body) == 1
    assert var_54.type_ignores == []
    var_55 = var_54.body
    var_56 = module_0.walk_body(var_55)
    var_57 = list(var_56)
    var_58 = len(var_57)
    assert var_58 == 3
    var_59 = 'print(1)'
    var_60 = module_1.parse(var_59)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'ast.Module'
    assert f'{type(var_60.body).__module__}.{type(var_60.body).__qualname__}' == 'builtins.list'
    assert len(var_60.body) == 1
    assert var_60.type_ignores == []
    var_61 = var_60.body
    var_62 = module_0.walk_body(var_61)
    var_63 = list(var_62)
    var_64 = len(var_63)
    assert var_64 == 1
    var_65 = var_63[var_9]
    var_66 = 'if True:\n    if True:\n        if True:\n            x = 1'
    var_67 = module_1.parse(var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'ast.Module'
    assert f'{type(var_67.body).__module__}.{type(var_67.body).__qualname__}' == 'builtins.list'
    assert len(var_67.body) == 1
    assert var_67.type_ignores == []
    var_68 = var_67.body
    var_69 = module_0.walk_body(var_68)
    var_70 = list(var_69)
    var_71 = len(var_70)
    assert var_71 == 1
    var_72 = var_70[var_9]

def test_case_25():
    var_0 = ''
    var_1 = module_0.doctest(var_0)
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
    var_2 = 'This is a comment'
    var_3 = module_0.doctest(var_2)
    assert var_3 == 'This is a comment'
    var_4 = ">>> print('hello')"
    var_5 = module_0.doctest(var_4)
    assert var_5 == "```python\n>>> print('hello')\n```"
    var_6 = '>>> x = 1\n>>> print(x)'
    var_7 = module_0.doctest(var_6)
    assert var_7 == '```python\n>>> x = 1\n>>> print(x)\n```'
    var_8 = ">>> print('hello')\nhello"
    var_9 = module_0.doctest(var_8)
    assert var_9 == "```python\n>>> print('hello')\n```\nhello"
    var_10 = '>>> x = 1\nsome comment\n>>> y = 2'
    var_11 = module_0.doctest(var_10)
    assert var_11 == '```python\n>>> x = 1\n```\nsome comment\n```python\n>>> y = 2\n```'
    var_12 = "Some comment\n>>> print('end')"
    var_13 = module_0.doctest(var_12)
    assert var_13 == "Some comment\n```python\n>>> print('end')\n```"
    var_14 = '>>> a = 1\n>>> b = 2\n>>> c = a + b'
    var_15 = module_0.doctest(var_14)
    assert var_15 == '```python\n>>> a = 1\n>>> b = 2\n>>> c = a + b\n```'
    var_16 = '>>> x = 5\n5\n>>> y = 10'
    var_17 = module_0.doctest(var_16)
    assert var_17 == '```python\n>>> x = 5\n```\n5\n```python\n>>> y = 10\n```'
    var_18 = '>>> x = 1\ncomment line\n>>> y = 2'
    var_19 = module_0.doctest(var_18)
    assert var_19 == '```python\n>>> x = 1\n```\ncomment line\n```python\n>>> y = 2\n```'
    var_20 = 'output line 1\noutput line 2'
    var_21 = module_0.doctest(var_20)
    assert var_21 == 'output line 1\noutput line 2'

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
    var_1 = '\nclass MyClass:\n    value = 42  # type: int\n    name = "test"  # type: str\n'
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
    var_5 = var_0.__repr__()
    assert var_5 == 'Parser(link=True, b_level=1, toc=False, level={}, doc={}, docstring={}, imp={}, root={}, alias={}, const={})'
    var_6 = 'test_module.MyClass'
    var_7 = var_4.body
    var_0.class_api(var_6, var_6, var_7, var_7)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = 'Test Parser.class_api with type comments.'
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
    var_2 = 'test_module'
    var_3 = '\nclass MyClass:\n    value = 42  # type: int\n    name = "test"  # type: str\n'
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
    var_7 = 'test_module.MyClass'
    var_8 = var_6.body
    var_1.class_api(var_2, var_7, var_0, var_8)

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
    var_1 = 'test_module_comment'
    var_2 = '\nx = 42  # type: int\ny = "hello"  # type: str\n'
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_module_comment': 0}
    assert var_0.doc == {'test_module_comment': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module_comment': {*()}}
    assert var_0.root == {'test_module_comment': 'test_module_comment'}
    assert var_0.alias == {'test_module_comment.x': '42', 'test_module_comment.y': "'hello'"}

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
    var_1 = 'test_async'
    var_2 = "\nasync def async_func(x: int) -> str:\n    '''Async function.'''\n    return str(x)\n"
    var_3 = var_0.parse(var_1, var_2)
    assert var_0.level == {'test_async': 0, 'test_async.async_func': 0}
    assert var_0.doc == {'test_async': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_async.async_func': '### async async_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | return |\n|:---:|:------:|\n| `int` | `str` |\n\n'}
    assert var_0.docstring == {'test_async.async_func': 'Async function.'}
    assert var_0.imp == {'test_async': {*()}}
    assert var_0.root == {'test_async': 'test_async', 'test_async.async_func': 'test_async'}

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
    var_1 = '\ndef kw_only(a: int, *, b: str) -> None:\n    pass\n'
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
    var_5 = 'test_module'
    var_6 = 'test_module.kw_only'
    var_7 = var_4.args
    var_8 = var_4.returns
    var_9 = False
    var_10 = False
    var_0.func_api(var_5, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)

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
    var_1 = '\ndef pos_only(a: int, /, b: str) -> None:\n    pass\n'
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
    var_5 = 'test_module'
    var_6 = 'test_module.pos_only'
    var_7 = var_4.args
    var_8 = var_4.returns
    var_9 = False
    var_10 = False
    var_0.func_api(var_5, var_6, var_7, var_8, has_self=var_9, cls_method=var_10)

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
    var_1 = var_0.compile()
    assert var_1 == '\n'
    var_2 = True
    var_3 = False
    var_4 = module_0.Parser(var_2, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'apimd.parser.Parser'
    assert var_4.link is True
    assert var_4.b_level is True
    assert var_4.toc is False
    assert var_4.level == {}
    assert var_4.doc == {}
    assert var_4.docstring == {}
    assert var_4.imp == {}
    assert var_4.root == {}
    assert var_4.alias == {}
    assert var_4.const == {}
    var_5 = var_4.compile()
    assert var_5 == '\n'
    var_6 = module_0.Parser(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'apimd.parser.Parser'
    assert var_6.link is True
    assert var_6.b_level is True
    assert var_6.toc is True
    assert var_6.level == {}
    assert var_6.doc == {}
    assert var_6.docstring == {}
    assert var_6.imp == {}
    assert var_6.root == {}
    assert var_6.alias == {}
    assert var_6.const == {}
    var_7 = var_6.compile()
    assert var_7 == '**Table of contents:**\n\n\n'
    var_8 = module_0.Parser(var_3, var_2, var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'apimd.parser.Parser'
    assert var_8.link is False
    assert var_8.b_level is True
    assert var_8.toc is False
    assert var_8.level == {}
    assert var_8.doc == {}
    assert var_8.docstring == {}
    assert var_8.imp == {}
    assert var_8.root == {}
    assert var_8.alias == {}
    assert var_8.const == {}
    var_9 = var_8.compile()
    assert var_9 == '\n'
    var_10 = module_0.Parser(var_2, var_2, var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'apimd.parser.Parser'
    assert var_10.link is True
    assert var_10.b_level is True
    assert var_10.toc is False
    assert var_10.level == {}
    assert var_10.doc == {}
    assert var_10.docstring == {}
    assert var_10.imp == {}
    assert var_10.root == {}
    assert var_10.alias == {}
    assert var_10.const == {}
    var_11 = var_10.compile()
    assert var_11 == '\n'
    var_12 = module_0.Parser(var_2, var_2, var_2)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'apimd.parser.Parser'
    assert var_12.link is True
    assert var_12.b_level is True
    assert var_12.toc is True
    assert var_12.level == {}
    assert var_12.doc == {}
    assert var_12.docstring == {}
    assert var_12.imp == {}
    assert var_12.root == {}
    assert var_12.alias == {}
    assert var_12.const == {}
    var_13 = var_12.compile()
    assert var_13 == '**Table of contents:**\n\n\n'

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
    var_1 = 'test_module_comment'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'test_module_comment': 0}
    assert var_0.doc == {'test_module_comment': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_0.imp == {'test_module_comment': {*()}}
    assert var_0.root == {'test_module_comment': 'test_module_comment'}
    var_3 = module_0.Parser(imp=var_2)
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
    var_4 = var_0.compile()
    assert var_4 == '\n'

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
    var_1 = "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n"
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n": 2, "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.public_func": 2, "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n._private_func": 2}
    assert var_0.doc == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n": '## Module `{}`\n<a id="{}"></a>\n\n', "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.public_func": '### public_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n', "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n._private_func": '### \\_private\\_func()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| return |\n|:------:|\n| `Any` |\n\n'}
    assert var_0.docstring == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.public_func": 'Public function.', "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n._private_func": 'Private function.'}
    assert var_0.imp == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n": {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.public_func"}}
    assert var_0.root == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n": "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n", "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.public_func": "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n", "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n._private_func": "\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n"}
    assert var_0.alias == {"\ndef public_func():\n    '''Public function.'''\n    pass\n\ndef _private_func():\n    '''Private function.'''\n    pass\n\n__all__ = ['public_func']\n.__all__": "['public_func']"}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\ndef public_func():\n    \'\'\'Public function.\'\'\'\n    pass\n\ndef _private_func():\n    \'\'\'Private function.\'\'\'\n    pass\n\n__all__ = [\'public_func\']\n`\n<a id="\ndef public_func():\n    \'\'\'public function-\'\'\'\n    pass\n\ndef _private_func():\n    \'\'\'private function-\'\'\'\n    pass\n\n__all__ = [\'public_func\']\n"></a>\n\n### public_func()\n\n*Full name:* `\ndef public_func():\n    \'\'\'Public function.\'\'\'\n    pass\n\ndef _private_func():\n    \'\'\'Private function.\'\'\'\n    pass\n\n__all__ = [\'public_func\']\n.public_func`\n<a id="\ndef public_func():\n    \'\'\'public function-\'\'\'\n    pass\n\ndef _private_func():\n    \'\'\'private function-\'\'\'\n    pass\n\n__all__ = [\'public_func\']\n-public_func"></a>\n\n| return |\n|:------:|\n| `Any` |\n\nPublic function.\n'

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
    var_1 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 4}
    assert var_0.doc == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_0.docstring == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 'Module docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 'Function docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 'Class docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': {*()}}
    assert var_0.root == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'}
    assert var_0.alias == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.os': 'os', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.List': 'typing.List', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.Optional': 'typing.Optional', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '42'}
    assert var_0.const == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': 'int'}
    var_3 = var_0.compile()
    assert var_3 == '## Module `\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n`\n<a id="\n\'\'\'module docstring-\'\'\'\n\nimport os\nfrom typing import list, optional\n\nmy_constant: int = 42\n\ndef my_function(x: int, y: str = "default") -> list[str]:\n    \'\'\'function docstring-\'\'\'\n    return [y] * x\n\nclass myclass:\n    \'\'\'class docstring-\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'method docstring-\'\'\'\n        return str(arg)\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `MY_CONSTANT` | `int` |\n\nModule docstring.\n\n### my_function()\n\n*Full name:* `\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function`\n<a id="\n\'\'\'module docstring-\'\'\'\n\nimport os\nfrom typing import list, optional\n\nmy_constant: int = 42\n\ndef my_function(x: int, y: str = "default") -> list[str]:\n    \'\'\'function docstring-\'\'\'\n    return [y] * x\n\nclass myclass:\n    \'\'\'class docstring-\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'method docstring-\'\'\'\n        return str(arg)\n-my_function"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\nFunction docstring.\n\n### class MyClass\n\n*Full name:* `\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass`\n<a id="\n\'\'\'module docstring-\'\'\'\n\nimport os\nfrom typing import list, optional\n\nmy_constant: int = 42\n\ndef my_function(x: int, y: str = "default") -> list[str]:\n    \'\'\'function docstring-\'\'\'\n    return [y] * x\n\nclass myclass:\n    \'\'\'class docstring-\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'method docstring-\'\'\'\n        return str(arg)\n-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method`\n<a id="\n\'\'\'module docstring-\'\'\'\n\nimport os\nfrom typing import list, optional\n\nmy_constant: int = 42\n\ndef my_function(x: int, y: str = "default") -> list[str]:\n    \'\'\'function docstring-\'\'\'\n    return [y] * x\n\nclass myclass:\n    \'\'\'class docstring-\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'method docstring-\'\'\'\n        return str(arg)\n-myclass-method"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\nMethod docstring.\n'

def test_case_36():
    var_0 = '~`%H'
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
    var_2 = 'naT-k+~!1OR2*aCAu'
    var_3 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_4 = var_1.parse(var_2, var_3)
    assert var_1.level == {'naT-k+~!1OR2*aCAu': 0, 'naT-k+~!1OR2*aCAu.my_function': 0, 'naT-k+~!1OR2*aCAu.MyClass': 0, 'naT-k+~!1OR2*aCAu.MyClass.method': 0}
    assert var_1.doc == {'naT-k+~!1OR2*aCAu': '## Module `{}`\n<a id="{}"></a>\n\n', 'naT-k+~!1OR2*aCAu.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', 'naT-k+~!1OR2*aCAu.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', 'naT-k+~!1OR2*aCAu.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_1.docstring == {'naT-k+~!1OR2*aCAu': 'Module docstring.', 'naT-k+~!1OR2*aCAu.my_function': 'Function docstring.', 'naT-k+~!1OR2*aCAu.MyClass': 'Class docstring.', 'naT-k+~!1OR2*aCAu.MyClass.method': 'Method docstring.'}
    assert var_1.imp == {'naT-k+~!1OR2*aCAu': {*()}}
    assert var_1.root == {'naT-k+~!1OR2*aCAu': 'naT-k+~!1OR2*aCAu', 'naT-k+~!1OR2*aCAu.MY_CONSTANT': 'naT-k+~!1OR2*aCAu', 'naT-k+~!1OR2*aCAu.my_function': 'naT-k+~!1OR2*aCAu', 'naT-k+~!1OR2*aCAu.MyClass': 'naT-k+~!1OR2*aCAu', 'naT-k+~!1OR2*aCAu.MyClass.method': 'naT-k+~!1OR2*aCAu'}
    assert var_1.alias == {'naT-k+~!1OR2*aCAu.os': 'os', 'naT-k+~!1OR2*aCAu.List': 'typing.List', 'naT-k+~!1OR2*aCAu.Optional': 'typing.Optional', 'naT-k+~!1OR2*aCAu.MY_CONSTANT': '42'}
    assert var_1.const == {'naT-k+~!1OR2*aCAu.MY_CONSTANT': 'int'}
    var_5 = "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_6 = var_1.parse(var_0, var_5)
    assert var_1.level == {'naT-k+~!1OR2*aCAu': 0, 'naT-k+~!1OR2*aCAu.my_function': 0, 'naT-k+~!1OR2*aCAu.MyClass': 0, 'naT-k+~!1OR2*aCAu.MyClass.method': 0, '~`%H': 0, '~`%H.MyClass': 0, '~`%H.MyClass.static_method': 0, '~`%H.MyClass.class_method': 0}
    assert var_1.doc == {'naT-k+~!1OR2*aCAu': '## Module `{}`\n<a id="{}"></a>\n\n', 'naT-k+~!1OR2*aCAu.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', 'naT-k+~!1OR2*aCAu.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', 'naT-k+~!1OR2*aCAu.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n', '~`%H': '## Module `{}`\n<a id="{}"></a>\n\n', '~`%H.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '~`%H.MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', '~`%H.MyClass.class_method': '#### MyClass.class_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\n'}
    assert var_1.docstring == {'naT-k+~!1OR2*aCAu': 'Module docstring.', 'naT-k+~!1OR2*aCAu.my_function': 'Function docstring.', 'naT-k+~!1OR2*aCAu.MyClass': 'Class docstring.', 'naT-k+~!1OR2*aCAu.MyClass.method': 'Method docstring.', '~`%H.MyClass': 'Test class.', '~`%H.MyClass.static_method': 'Static method.', '~`%H.MyClass.class_method': 'Class method.'}
    assert var_1.imp == {'naT-k+~!1OR2*aCAu': {*()}, '~`%H': {*()}}
    assert var_1.root == {'naT-k+~!1OR2*aCAu': 'naT-k+~!1OR2*aCAu', 'naT-k+~!1OR2*aCAu.MY_CONSTANT': 'naT-k+~!1OR2*aCAu', 'naT-k+~!1OR2*aCAu.my_function': 'naT-k+~!1OR2*aCAu', 'naT-k+~!1OR2*aCAu.MyClass': 'naT-k+~!1OR2*aCAu', 'naT-k+~!1OR2*aCAu.MyClass.method': 'naT-k+~!1OR2*aCAu', '~`%H': '~`%H', '~`%H.MyClass': '~`%H', '~`%H.MyClass.static_method': '~`%H', '~`%H.MyClass.class_method': '~`%H'}
    assert var_1.alias == {'naT-k+~!1OR2*aCAu.os': 'os', 'naT-k+~!1OR2*aCAu.List': 'typing.List', 'naT-k+~!1OR2*aCAu.Optional': 'typing.Optional', 'naT-k+~!1OR2*aCAu.MY_CONSTANT': '42', '~`%H.staticmethod': 'functools.staticmethod', '~`%H.classmethod': 'functools.classmethod'}
    var_7 = var_1.compile()
    assert var_7 == '## Module `naT-k+~!1OR2*aCAu`\n<a id="nat-k+~!1or2*acau"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `MY_CONSTANT` | `int` |\n\nModule docstring.\n\n### my_function()\n\n*Full name:* `naT-k+~!1OR2*aCAu.my_function`\n<a id="nat-k+~!1or2*acau-my_function"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\nFunction docstring.\n\n### class MyClass\n\n*Full name:* `naT-k+~!1OR2*aCAu.MyClass`\n<a id="nat-k+~!1or2*acau-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `naT-k+~!1OR2*aCAu.MyClass.method`\n<a id="nat-k+~!1or2*acau-myclass-method"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\nMethod docstring.\n\n## Module `~`%H`\n<a id="~`%h"></a>\n\n### class MyClass\n\n*Full name:* `~`%H.MyClass`\n<a id="~`%h-myclass"></a>\n\nTest class.\n\n#### MyClass.class_method()\n\n*Full name:* `~`%H.MyClass.class_method`\n<a id="~`%h-myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `~`%H.MyClass.static_method`\n<a id="~`%h-myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\nStatic method.\n'

def test_case_37():
    var_0 = 'Test Parser.parse mehod.'
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
    var_2 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_3 = var_1.parse(var_0, var_2)
    assert var_1.level == {'Test Parser.parse mehod.': 2, 'Test Parser.parse mehod..my_function': 2, 'Test Parser.parse mehod..MyClass': 2, 'Test Parser.parse mehod..MyClass.method': 2}
    assert var_1.doc == {'Test Parser.parse mehod.': '## Module `{}`\n<a id="{}"></a>\n\n', 'Test Parser.parse mehod..my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', 'Test Parser.parse mehod..MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', 'Test Parser.parse mehod..MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_1.docstring == {'Test Parser.parse mehod.': 'Module docstring.', 'Test Parser.parse mehod..my_function': 'Function docstring.', 'Test Parser.parse mehod..MyClass': 'Class docstring.', 'Test Parser.parse mehod..MyClass.method': 'Method docstring.'}
    assert var_1.imp == {'Test Parser.parse mehod.': {*()}}
    assert var_1.root == {'Test Parser.parse mehod.': 'Test Parser.parse mehod.', 'Test Parser.parse mehod..MY_CONSTANT': 'Test Parser.parse mehod.', 'Test Parser.parse mehod..my_function': 'Test Parser.parse mehod.', 'Test Parser.parse mehod..MyClass': 'Test Parser.parse mehod.', 'Test Parser.parse mehod..MyClass.method': 'Test Parser.parse mehod.'}
    assert var_1.alias == {'Test Parser.parse mehod..os': 'os', 'Test Parser.parse mehod..List': 'typing.List', 'Test Parser.parse mehod..Optional': 'typing.Optional', 'Test Parser.parse mehod..MY_CONSTANT': '42'}
    assert var_1.const == {'Test Parser.parse mehod..MY_CONSTANT': 'int'}
    var_4 = var_1.parse(var_0, var_2)

def test_case_38():
    var_0 = 'Test Pasrparse metod.'
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
    var_2 = '\\~8'
    var_3 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_4 = var_1.parse(var_2, var_3)
    assert var_1.level == {'\\~8': 0, '\\~8.my_function': 0, '\\~8.MyClass': 0, '\\~8.MyClass.method': 0}
    assert var_1.doc == {'\\~8': '## Module `{}`\n<a id="{}"></a>\n\n', '\\~8.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '\\~8.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '\\~8.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_1.docstring == {'\\~8': 'Module docstring.', '\\~8.my_function': 'Function docstring.', '\\~8.MyClass': 'Class docstring.', '\\~8.MyClass.method': 'Method docstring.'}
    assert var_1.imp == {'\\~8': {*()}}
    assert var_1.root == {'\\~8': '\\~8', '\\~8.MY_CONSTANT': '\\~8', '\\~8.my_function': '\\~8', '\\~8.MyClass': '\\~8', '\\~8.MyClass.method': '\\~8'}
    assert var_1.alias == {'\\~8.os': 'os', '\\~8.List': 'typing.List', '\\~8.Optional': 'typing.Optional', '\\~8.MY_CONSTANT': '42'}
    assert var_1.const == {'\\~8.MY_CONSTANT': 'int'}
    var_5 = "\nfrom functools import staticmethod, classm0thod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_6 = var_1.parse(var_0, var_5)
    assert var_1.level == {'\\~8': 0, '\\~8.my_function': 0, '\\~8.MyClass': 0, '\\~8.MyClass.method': 0, 'Test Pasrparse metod.': 1, 'Test Pasrparse metod..MyClass': 1, 'Test Pasrparse metod..MyClass.static_method': 1, 'Test Pasrparse metod..MyClass.class_method': 1}
    assert var_1.doc == {'\\~8': '## Module `{}`\n<a id="{}"></a>\n\n', '\\~8.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '\\~8.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '\\~8.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n', 'Test Pasrparse metod.': '## Module `{}`\n<a id="{}"></a>\n\n', 'Test Pasrparse metod..MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'Test Pasrparse metod..MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', 'Test Pasrparse metod..MyClass.class_method': '#### MyClass.class_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `int` |\n\n'}
    assert var_1.docstring == {'\\~8': 'Module docstring.', '\\~8.my_function': 'Function docstring.', '\\~8.MyClass': 'Class docstring.', '\\~8.MyClass.method': 'Method docstring.', 'Test Pasrparse metod..MyClass': 'Test class.', 'Test Pasrparse metod..MyClass.static_method': 'Static method.', 'Test Pasrparse metod..MyClass.class_method': 'Class method.'}
    assert var_1.imp == {'\\~8': {*()}, 'Test Pasrparse metod.': {*()}}
    assert var_1.root == {'\\~8': '\\~8', '\\~8.MY_CONSTANT': '\\~8', '\\~8.my_function': '\\~8', '\\~8.MyClass': '\\~8', '\\~8.MyClass.method': '\\~8', 'Test Pasrparse metod.': 'Test Pasrparse metod.', 'Test Pasrparse metod..MyClass': 'Test Pasrparse metod.', 'Test Pasrparse metod..MyClass.static_method': 'Test Pasrparse metod.', 'Test Pasrparse metod..MyClass.class_method': 'Test Pasrparse metod.'}
    assert var_1.alias == {'\\~8.os': 'os', '\\~8.List': 'typing.List', '\\~8.Optional': 'typing.Optional', '\\~8.MY_CONSTANT': '42', 'Test Pasrparse metod..staticmethod': 'functools.staticmethod', 'Test Pasrparse metod..classm0thod': 'functools.classm0thod'}
    var_7 = var_1.compile()
    assert var_7 == '## Module `\\~8`\n<a id="\\~8"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `MY_CONSTANT` | `int` |\n\nModule docstring.\n\n### my_function()\n\n*Full name:* `\\~8.my_function`\n<a id="\\~8-my_function"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\nFunction docstring.\n\n### class MyClass\n\n*Full name:* `\\~8.MyClass`\n<a id="\\~8-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\\~8.MyClass.method`\n<a id="\\~8-myclass-method"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\nMethod docstring.\n\n## Module `Test Pasrparse metod.`\n<a id="test pasrparse metod-"></a>\n\n### class MyClass\n\n*Full name:* `Test Pasrparse metod..MyClass`\n<a id="test pasrparse metod--myclass"></a>\n\nTest class.\n\n#### MyClass.class_method()\n\n*Full name:* `Test Pasrparse metod..MyClass.class_method`\n<a id="test pasrparse metod--myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `Test Pasrparse metod..MyClass.static_method`\n<a id="test pasrparse metod--myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\nStatic method.\n'
    var_8 = var_1.compile()
    assert var_8 == '## Module `\\~8`\n<a id="\\~8"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `MY_CONSTANT` | `int` |\n\nModule docstring.\n\n### my_function()\n\n*Full name:* `\\~8.my_function`\n<a id="\\~8-my_function"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\nFunction docstring.\n\n### class MyClass\n\n*Full name:* `\\~8.MyClass`\n<a id="\\~8-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\\~8.MyClass.method`\n<a id="\\~8-myclass-method"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\nMethod docstring.\n\n## Module `Test Pasrparse metod.`\n<a id="test pasrparse metod-"></a>\n\n### class MyClass\n\n*Full name:* `Test Pasrparse metod..MyClass`\n<a id="test pasrparse metod--myclass"></a>\n\nTest class.\n\n#### MyClass.class_method()\n\n*Full name:* `Test Pasrparse metod..MyClass.class_method`\n<a id="test pasrparse metod--myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `Test Pasrparse metod..MyClass.static_method`\n<a id="test pasrparse metod--myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\nStatic method.\n'

def test_case_39():
    var_0 = '%m"'
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
    var_2 = '\nN"xW;&jL8N6n`yY,9H'
    var_3 = "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_method(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_4 = var_1.parse(var_0, var_3)
    assert var_1.level == {'%m"': 0, '%m".MyClass': 0, '%m".MyClass.static_method': 0, '%m".MyClass.class_method': 0}
    assert var_1.doc == {'%m"': '## Module `{}`\n<a id="{}"></a>\n\n', '%m".MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '%m".MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', '%m".MyClass.class_method': '#### MyClass.class_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\n'}
    assert var_1.docstring == {'%m".MyClass': 'Test class.', '%m".MyClass.static_method': 'Static method.', '%m".MyClass.class_method': 'Class method.'}
    assert var_1.imp == {'%m"': {*()}}
    assert var_1.root == {'%m"': '%m"', '%m".MyClass': '%m"', '%m".MyClass.static_method': '%m"', '%m".MyClass.class_method': '%m"'}
    assert var_1.alias == {'%m".staticmethod': 'functools.staticmethod', '%m".classmethod': 'functools.classmethod'}
    var_5 = var_1.parse(var_2, var_3)
    assert var_1.level == {'%m"': 0, '%m".MyClass': 0, '%m".MyClass.static_method': 0, '%m".MyClass.class_method': 0, '\nN"xW;&jL8N6n`yY,9H': 0, '\nN"xW;&jL8N6n`yY,9H.MyClass': 0, '\nN"xW;&jL8N6n`yY,9H.MyClass.static_method': 0, '\nN"xW;&jL8N6n`yY,9H.MyClass.class_method': 0}
    assert var_1.doc == {'%m"': '## Module `{}`\n<a id="{}"></a>\n\n', '%m".MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '%m".MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', '%m".MyClass.class_method': '#### MyClass.class_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\n', '\nN"xW;&jL8N6n`yY,9H': '## Module `{}`\n<a id="{}"></a>\n\n', '\nN"xW;&jL8N6n`yY,9H.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '\nN"xW;&jL8N6n`yY,9H.MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', '\nN"xW;&jL8N6n`yY,9H.MyClass.class_method': '#### MyClass.class_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\n'}
    assert var_1.docstring == {'%m".MyClass': 'Test class.', '%m".MyClass.static_method': 'Static method.', '%m".MyClass.class_method': 'Class method.', '\nN"xW;&jL8N6n`yY,9H.MyClass': 'Test class.', '\nN"xW;&jL8N6n`yY,9H.MyClass.static_method': 'Static method.', '\nN"xW;&jL8N6n`yY,9H.MyClass.class_method': 'Class method.'}
    assert var_1.imp == {'%m"': {*()}, '\nN"xW;&jL8N6n`yY,9H': {*()}}
    assert var_1.root == {'%m"': '%m"', '%m".MyClass': '%m"', '%m".MyClass.static_method': '%m"', '%m".MyClass.class_method': '%m"', '\nN"xW;&jL8N6n`yY,9H': '\nN"xW;&jL8N6n`yY,9H', '\nN"xW;&jL8N6n`yY,9H.MyClass': '\nN"xW;&jL8N6n`yY,9H', '\nN"xW;&jL8N6n`yY,9H.MyClass.static_method': '\nN"xW;&jL8N6n`yY,9H', '\nN"xW;&jL8N6n`yY,9H.MyClass.class_method': '\nN"xW;&jL8N6n`yY,9H'}
    assert var_1.alias == {'%m".staticmethod': 'functools.staticmethod', '%m".classmethod': 'functools.classmethod', '\nN"xW;&jL8N6n`yY,9H.staticmethod': 'functools.staticmethod', '\nN"xW;&jL8N6n`yY,9H.classmethod': 'functools.classmethod'}
    var_6 = var_1.__repr__()
    assert var_6 == 'Parser(link=True, b_level=1, toc=False, level={\'%m"\': 0, \'%m".MyClass\': 0, \'%m".MyClass.static_method\': 0, \'%m".MyClass.class_method\': 0, \'\\nN"xW;&jL8N6n`yY,9H\': 0, \'\\nN"xW;&jL8N6n`yY,9H.MyClass\': 0, \'\\nN"xW;&jL8N6n`yY,9H.MyClass.static_method\': 0, \'\\nN"xW;&jL8N6n`yY,9H.MyClass.class_method\': 0}, doc={\'%m"\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'%m".MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'%m".MyClass.static_method\': \'#### MyClass.static_method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.staticmethod` |\\n\\n| x | return |\\n|:---:|:------:|\\n| `Self` | `Self` |\\n\\n\', \'%m".MyClass.class_method\': \'#### MyClass.class_method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.classmethod` |\\n\\n| cls | x | return |\\n|:---:|:---:|:------:|\\n| `Self` | `int` | `int` |\\n\\n\', \'\\nN"xW;&jL8N6n`yY,9H\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\nN"xW;&jL8N6n`yY,9H.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'\\nN"xW;&jL8N6n`yY,9H.MyClass.static_method\': \'#### MyClass.static_method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.staticmethod` |\\n\\n| x | return |\\n|:---:|:------:|\\n| `Self` | `Self` |\\n\\n\', \'\\nN"xW;&jL8N6n`yY,9H.MyClass.class_method\': \'#### MyClass.class_method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.classmethod` |\\n\\n| cls | x | return |\\n|:---:|:---:|:------:|\\n| `Self` | `int` | `int` |\\n\\n\'}, docstring={\'%m".MyClass\': \'Test class.\', \'%m".MyClass.static_method\': \'Static method.\', \'%m".MyClass.class_method\': \'Class method.\', \'\\nN"xW;&jL8N6n`yY,9H.MyClass\': \'Test class.\', \'\\nN"xW;&jL8N6n`yY,9H.MyClass.static_method\': \'Static method.\', \'\\nN"xW;&jL8N6n`yY,9H.MyClass.class_method\': \'Class method.\'}, imp={\'%m"\': set(), \'\\nN"xW;&jL8N6n`yY,9H\': set()}, root={\'%m"\': \'%m"\', \'%m".MyClass\': \'%m"\', \'%m".MyClass.static_method\': \'%m"\', \'%m".MyClass.class_method\': \'%m"\', \'\\nN"xW;&jL8N6n`yY,9H\': \'\\nN"xW;&jL8N6n`yY,9H\', \'\\nN"xW;&jL8N6n`yY,9H.MyClass\': \'\\nN"xW;&jL8N6n`yY,9H\', \'\\nN"xW;&jL8N6n`yY,9H.MyClass.static_method\': \'\\nN"xW;&jL8N6n`yY,9H\', \'\\nN"xW;&jL8N6n`yY,9H.MyClass.class_method\': \'\\nN"xW;&jL8N6n`yY,9H\'}, alias={\'%m".staticmethod\': \'functools.staticmethod\', \'%m".classmethod\': \'functools.classmethod\', \'\\nN"xW;&jL8N6n`yY,9H.staticmethod\': \'functools.staticmethod\', \'\\nN"xW;&jL8N6n`yY,9H.classmethod\': \'functools.classmethod\'}, const={})'
    var_7 = var_1.load_docstring(var_0, var_6)
    var_8 = var_1.compile()
    assert var_8 == '## Module `\nN"xW;&jL8N6n`yY,9H`\n<a id="\nn"xw;&jl8n6n`yy,9h"></a>\n\n### class MyClass\n\n*Full name:* `\nN"xW;&jL8N6n`yY,9H.MyClass`\n<a id="\nn"xw;&jl8n6n`yy,9h-myclass"></a>\n\nTest class.\n\n#### MyClass.class_method()\n\n*Full name:* `\nN"xW;&jL8N6n`yY,9H.MyClass.class_method`\n<a id="\nn"xw;&jl8n6n`yy,9h-myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `\nN"xW;&jL8N6n`yY,9H.MyClass.static_method`\n<a id="\nn"xw;&jl8n6n`yy,9h-myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\nStatic method.\n\n## Module `%m"`\n<a id="%m""></a>\n\n### class MyClass\n\n*Full name:* `%m".MyClass`\n<a id="%m"-myclass"></a>\n\nTest class.\n\n#### MyClass.class_method()\n\n*Full name:* `%m".MyClass.class_method`\n<a id="%m"-myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `%m".MyClass.static_method`\n<a id="%m"-myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\nStatic method.\n'

def test_case_40():
    var_0 = ''
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
    var_2 = '9~'
    var_3 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_4 = var_1.parse(var_2, var_3)
    assert var_1.level == {'9~': 0, '9~.my_function': 0, '9~.MyClass': 0, '9~.MyClass.method': 0}
    assert var_1.doc == {'9~': '## Module `{}`\n<a id="{}"></a>\n\n', '9~.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '9~.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '9~.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_1.docstring == {'9~': 'Module docstring.', '9~.my_function': 'Function docstring.', '9~.MyClass': 'Class docstring.', '9~.MyClass.method': 'Method docstring.'}
    assert var_1.imp == {'9~': {*()}}
    assert var_1.root == {'9~': '9~', '9~.MY_CONSTANT': '9~', '9~.my_function': '9~', '9~.MyClass': '9~', '9~.MyClass.method': '9~'}
    assert var_1.alias == {'9~.os': 'os', '9~.List': 'typing.List', '9~.Optional': 'typing.Optional', '9~.MY_CONSTANT': '42'}
    assert var_1.const == {'9~.MY_CONSTANT': 'int'}
    var_5 = "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_mEthod(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_6 = var_1.parse(var_0, var_5)
    assert var_1.level == {'9~': 0, '9~.my_function': 0, '9~.MyClass': 0, '9~.MyClass.method': 0, '': 0, 'MyClass': 0, 'MyClass.static_method': 0, 'MyClass.class_mEthod': 0}
    assert var_1.doc == {'9~': '## Module `{}`\n<a id="{}"></a>\n\n', '9~.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '9~.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '9~.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n', '': '## Module `{}`\n<a id="{}"></a>\n\n', 'MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', 'MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `int` | `int` |\n\n', 'MyClass.class_mEthod': '#### MyClass.class_mEthod()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `int` |\n\n'}
    assert var_1.docstring == {'9~': 'Module docstring.', '9~.my_function': 'Function docstring.', '9~.MyClass': 'Class docstring.', '9~.MyClass.method': 'Method docstring.', 'MyClass': 'Test class.', 'MyClass.static_method': 'Static method.', 'MyClass.class_mEthod': 'Class method.'}
    assert var_1.imp == {'9~': {*()}, '': {*()}}
    assert var_1.root == {'9~': '9~', '9~.MY_CONSTANT': '9~', '9~.my_function': '9~', '9~.MyClass': '9~', '9~.MyClass.method': '9~', '': '', 'MyClass': '', 'MyClass.static_method': '', 'MyClass.class_mEthod': ''}
    assert var_1.alias == {'9~.os': 'os', '9~.List': 'typing.List', '9~.Optional': 'typing.Optional', '9~.MY_CONSTANT': '42', 'staticmethod': 'functools.staticmethod', 'classmethod': 'functools.classmethod'}
    var_7 = var_1.compile()
    assert var_7 == '## Module `9~`\n<a id="9~"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `MY_CONSTANT` | `int` |\n\nModule docstring.\n\n### my_function()\n\n*Full name:* `9~.my_function`\n<a id="9~-my_function"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\nFunction docstring.\n\n### class MyClass\n\n*Full name:* `9~.MyClass`\n<a id="9~-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `9~.MyClass.method`\n<a id="9~-myclass-method"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\nMethod docstring.\n\n### class MyClass\n\n*Full name:* `MyClass`\n<a id="myclass"></a>\n\nTest class.\n\n#### MyClass.class_mEthod()\n\n*Full name:* `MyClass.class_mEthod`\n<a id="myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `MyClass.static_method`\n<a id="myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `int` | `int` |\n\nStatic method.\n'
    var_8 = var_1.load_docstring(var_0, var_6)
    var_9 = None
    var_10 = True
    var_11 = module_1.Tuple()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'ast.Tuple'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert f'{type(module_1.Tuple.dims).__module__}.{type(module_1.Tuple.dims).__qualname__}' == 'builtins.property'
    var_12 = {var_7: var_3}
    var_13 = module_0.Parser(toc=var_10, root=var_12, alias=var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'apimd.parser.Parser'
    assert var_13.link is True
    assert var_13.b_level == 1
    assert var_13.toc is True
    assert var_13.level == {}
    assert var_13.doc == {}
    assert var_13.docstring == {}
    assert var_13.imp == {}
    assert var_13.root == {'## Module `9~`\n<a id="9~"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `MY_CONSTANT` | `int` |\n\nModule docstring.\n\n### my_function()\n\n*Full name:* `9~.my_function`\n<a id="9~-my_function"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\nFunction docstring.\n\n### class MyClass\n\n*Full name:* `9~.MyClass`\n<a id="9~-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `9~.MyClass.method`\n<a id="9~-myclass-method"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\nMethod docstring.\n\n### class MyClass\n\n*Full name:* `MyClass`\n<a id="myclass"></a>\n\nTest class.\n\n#### MyClass.class_mEthod()\n\n*Full name:* `MyClass.class_mEthod`\n<a id="myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `type[Self]` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `MyClass.static_method`\n<a id="myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `int` | `int` |\n\nStatic method.\n': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'}
    assert var_13.alias is None
    assert var_13.const == {}
    var_14 = module_0.const_type(var_9)
    assert var_14 == 'Any'
    var_15 = '%--wWqwkWL]C'
    var_16 = var_1.load_docstring(var_15, var_9)
    var_17 = module_0.const_type(var_2)
    assert var_17 == 'Any'

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = '%H'
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
    var_2 = '9b~'
    var_3 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_4 = var_1.parse(var_2, var_3)
    assert var_1.level == {'9b~': 0, '9b~.my_function': 0, '9b~.MyClass': 0, '9b~.MyClass.method': 0}
    assert var_1.doc == {'9b~': '## Module `{}`\n<a id="{}"></a>\n\n', '9b~.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '9b~.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '9b~.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_1.docstring == {'9b~': 'Module docstring.', '9b~.my_function': 'Function docstring.', '9b~.MyClass': 'Class docstring.', '9b~.MyClass.method': 'Method docstring.'}
    assert var_1.imp == {'9b~': {*()}}
    assert var_1.root == {'9b~': '9b~', '9b~.MY_CONSTANT': '9b~', '9b~.my_function': '9b~', '9b~.MyClass': '9b~', '9b~.MyClass.method': '9b~'}
    assert var_1.alias == {'9b~.os': 'os', '9b~.List': 'typing.List', '9b~.Optional': 'typing.Optional', '9b~.MY_CONSTANT': '42'}
    assert var_1.const == {'9b~.MY_CONSTANT': 'int'}
    var_5 = "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_mEthod(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_6 = module_1.ImportFrom()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'ast.ImportFrom'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    assert module_1.ImportFrom.module is None
    assert module_1.ImportFrom.level is None
    var_7 = var_1.imports(var_4, var_6)
    var_8 = var_1.parse(var_0, var_5)
    assert var_1.level == {'9b~': 0, '9b~.my_function': 0, '9b~.MyClass': 0, '9b~.MyClass.method': 0, '%H': 0, '%H.MyClass': 0, '%H.MyClass.static_method': 0, '%H.MyClass.class_mEthod': 0}
    assert var_1.doc == {'9b~': '## Module `{}`\n<a id="{}"></a>\n\n', '9b~.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '9b~.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '9b~.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n', '%H': '## Module `{}`\n<a id="{}"></a>\n\n', '%H.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '%H.MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', '%H.MyClass.class_mEthod': '#### MyClass.class_mEthod()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\n'}
    assert var_1.docstring == {'9b~': 'Module docstring.', '9b~.my_function': 'Function docstring.', '9b~.MyClass': 'Class docstring.', '9b~.MyClass.method': 'Method docstring.', '%H.MyClass': 'Test class.', '%H.MyClass.static_method': 'Static method.', '%H.MyClass.class_mEthod': 'Class method.'}
    assert var_1.imp == {'9b~': {*()}, '%H': {*()}}
    assert var_1.root == {'9b~': '9b~', '9b~.MY_CONSTANT': '9b~', '9b~.my_function': '9b~', '9b~.MyClass': '9b~', '9b~.MyClass.method': '9b~', '%H': '%H', '%H.MyClass': '%H', '%H.MyClass.static_method': '%H', '%H.MyClass.class_mEthod': '%H'}
    assert var_1.alias == {'9b~.os': 'os', '9b~.List': 'typing.List', '9b~.Optional': 'typing.Optional', '9b~.MY_CONSTANT': '42', '%H.staticmethod': 'functools.staticmethod', '%H.classmethod': 'functools.classmethod'}
    var_9 = var_1.compile()
    assert var_9 == '## Module `%H`\n<a id="%h"></a>\n\n### class MyClass\n\n*Full name:* `%H.MyClass`\n<a id="%h-myclass"></a>\n\nTest class.\n\n#### MyClass.class_mEthod()\n\n*Full name:* `%H.MyClass.class_mEthod`\n<a id="%h-myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `%H.MyClass.static_method`\n<a id="%h-myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\nStatic method.\n\n## Module `9b~`\n<a id="9b~"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `MY_CONSTANT` | `int` |\n\nModule docstring.\n\n### my_function()\n\n*Full name:* `9b~.my_function`\n<a id="9b~-my_function"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\nFunction docstring.\n\n### class MyClass\n\n*Full name:* `9b~.MyClass`\n<a id="9b~-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `9b~.MyClass.method`\n<a id="9b~-myclass-method"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\nMethod docstring.\n'
    var_10 = var_1.load_docstring(var_0, var_8)
    var_11 = "9\n-O6'\x0bvSt#pcd\r"
    var_1.is_public(var_11)

def test_case_42():
    var_0 = '%H'
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
    var_2 = '9b~'
    var_3 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_4 = var_1.parse(var_2, var_3)
    assert var_1.level == {'9b~': 0, '9b~.my_function': 0, '9b~.MyClass': 0, '9b~.MyClass.method': 0}
    assert var_1.doc == {'9b~': '## Module `{}`\n<a id="{}"></a>\n\n', '9b~.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '9b~.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '9b~.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_1.docstring == {'9b~': 'Module docstring.', '9b~.my_function': 'Function docstring.', '9b~.MyClass': 'Class docstring.', '9b~.MyClass.method': 'Method docstring.'}
    assert var_1.imp == {'9b~': {*()}}
    assert var_1.root == {'9b~': '9b~', '9b~.MY_CONSTANT': '9b~', '9b~.my_function': '9b~', '9b~.MyClass': '9b~', '9b~.MyClass.method': '9b~'}
    assert var_1.alias == {'9b~.os': 'os', '9b~.List': 'typing.List', '9b~.Optional': 'typing.Optional', '9b~.MY_CONSTANT': '42'}
    assert var_1.const == {'9b~.MY_CONSTANT': 'int'}
    var_5 = "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_mEthod(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_6 = var_1.parse(var_0, var_5)
    assert var_1.level == {'9b~': 0, '9b~.my_function': 0, '9b~.MyClass': 0, '9b~.MyClass.method': 0, '%H': 0, '%H.MyClass': 0, '%H.MyClass.static_method': 0, '%H.MyClass.class_mEthod': 0}
    assert var_1.doc == {'9b~': '## Module `{}`\n<a id="{}"></a>\n\n', '9b~.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '9b~.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '9b~.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n', '%H': '## Module `{}`\n<a id="{}"></a>\n\n', '%H.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '%H.MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', '%H.MyClass.class_mEthod': '#### MyClass.class_mEthod()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\n'}
    assert var_1.docstring == {'9b~': 'Module docstring.', '9b~.my_function': 'Function docstring.', '9b~.MyClass': 'Class docstring.', '9b~.MyClass.method': 'Method docstring.', '%H.MyClass': 'Test class.', '%H.MyClass.static_method': 'Static method.', '%H.MyClass.class_mEthod': 'Class method.'}
    assert var_1.imp == {'9b~': {*()}, '%H': {*()}}
    assert var_1.root == {'9b~': '9b~', '9b~.MY_CONSTANT': '9b~', '9b~.my_function': '9b~', '9b~.MyClass': '9b~', '9b~.MyClass.method': '9b~', '%H': '%H', '%H.MyClass': '%H', '%H.MyClass.static_method': '%H', '%H.MyClass.class_mEthod': '%H'}
    assert var_1.alias == {'9b~.os': 'os', '9b~.List': 'typing.List', '9b~.Optional': 'typing.Optional', '9b~.MY_CONSTANT': '42', '%H.staticmethod': 'functools.staticmethod', '%H.classmethod': 'functools.classmethod'}
    var_7 = var_1.compile()
    assert var_7 == '## Module `%H`\n<a id="%h"></a>\n\n### class MyClass\n\n*Full name:* `%H.MyClass`\n<a id="%h-myclass"></a>\n\nTest class.\n\n#### MyClass.class_mEthod()\n\n*Full name:* `%H.MyClass.class_mEthod`\n<a id="%h-myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `%H.MyClass.static_method`\n<a id="%h-myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\nStatic method.\n\n## Module `9b~`\n<a id="9b~"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `MY_CONSTANT` | `int` |\n\nModule docstring.\n\n### my_function()\n\n*Full name:* `9b~.my_function`\n<a id="9b~-my_function"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\nFunction docstring.\n\n### class MyClass\n\n*Full name:* `9b~.MyClass`\n<a id="9b~-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `9b~.MyClass.method`\n<a id="9b~-myclass-method"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\nMethod docstring.\n'
    var_8 = var_1.load_docstring(var_0, var_6)
    var_9 = None
    var_10 = True
    var_11 = {var_7: var_3}
    var_12 = module_0.Parser(toc=var_10, root=var_11, alias=var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'apimd.parser.Parser'
    assert var_12.link is True
    assert var_12.b_level == 1
    assert var_12.toc is True
    assert var_12.level == {}
    assert var_12.doc == {}
    assert var_12.docstring == {}
    assert var_12.imp == {}
    assert var_12.root == {'## Module `%H`\n<a id="%h"></a>\n\n### class MyClass\n\n*Full name:* `%H.MyClass`\n<a id="%h-myclass"></a>\n\nTest class.\n\n#### MyClass.class_mEthod()\n\n*Full name:* `%H.MyClass.class_mEthod`\n<a id="%h-myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `%H.MyClass.static_method`\n<a id="%h-myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\nStatic method.\n\n## Module `9b~`\n<a id="9b~"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `MY_CONSTANT` | `int` |\n\nModule docstring.\n\n### my_function()\n\n*Full name:* `9b~.my_function`\n<a id="9b~-my_function"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\nFunction docstring.\n\n### class MyClass\n\n*Full name:* `9b~.MyClass`\n<a id="9b~-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `9b~.MyClass.method`\n<a id="9b~-myclass-method"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\nMethod docstring.\n': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'}
    assert var_12.alias is None
    assert var_12.const == {}
    var_13 = [var_9]
    var_14 = module_1.Set(*var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'ast.Set'
    assert var_14.elts is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_15 = module_0.const_type(var_14)
    assert var_15 == 'set'

def test_case_43():
    var_0 = 'Test is_public_family function.'
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
    var_2 = 'sys'
    var_3 = module_0.is_public_family(var_2)
    assert var_3 is True
    var_4 = 'collections.abc.Sequence'
    var_5 = module_0.is_public_family(var_4)
    assert var_5 is True
    var_6 = 'module.__init__'
    var_7 = module_0.is_public_family(var_6)
    assert var_7 is True
    var_8 = '__main__'
    var_9 = module_0.is_public_family(var_8)
    assert var_9 is True
    var_10 = 'os.__dict__'
    var_11 = module_0.is_public_family(var_10)
    assert var_11 is True
    var_12 = '_private'
    var_13 = module_0.is_public_family(var_12)
    assert var_13 is False
    var_14 = 'os._internal'
    var_15 = module_0.is_public_family(var_14)
    assert var_15 is False
    var_16 = module_0.is_public_family(var_2)
    assert var_16 is True
    var_17 = 'os.path._private'
    var_18 = module_0.is_public_family(var_17)
    assert var_18 is False
    var_19 = 'os.__name__.public'
    var_20 = module_0.is_public_family(var_19)
    assert var_20 is True
    var_21 = 'public.__magic__.func'
    var_22 = module_0.is_public_family(var_21)
    assert var_22 is True
    var_23 = 'public._private.__magic__'
    var_24 = module_0.is_public_family(var_23)
    assert var_24 is False
    var_25 = ''
    var_26 = module_0.is_public_family(var_25)
    assert var_26 is True
    var_27 = 'a'
    var_28 = module_0.is_public_family(var_27)
    assert var_28 is True
    var_29 = '_'
    var_30 = module_0.is_public_family(var_29)
    assert var_30 is False
    var_31 = 'package.module.Class.method'
    var_32 = module_0.is_public_family(var_31)
    assert var_32 is True
    var_33 = 'package._internal.Class'
    var_34 = module_0.is_public_family(var_33)
    assert var_34 is False
    var_35 = 'package.module._private_class.method'
    var_36 = module_0.is_public_family(var_35)
    assert var_36 is False

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
    var_1 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_2 = var_0.parse(var_1, var_1)
    assert var_0.level == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 4}
    assert var_0.doc == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `typig.List[str]` |\n|   | `\'default\'` |   |   |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_0.docstring == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 'Module docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 'Function docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 'Class docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 'Method docstring.'}
    assert var_0.imp == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': {*()}}
    assert var_0.root == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'}
    assert var_0.alias == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.os': 'os', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.List': 'typig.List', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.Optional': 'typig.Optional', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '42'}
    assert var_0.const == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typig import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "vale"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': 'int'}

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
    var_1 = 'test_module'
    var_2 = 'test_module.test_func'
    var_3 = "def func(a: int, b: str = 'default', *args, c: float = 1.0, **kwargs) -> bool: pass"
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
    var_7 = var_6.args
    var_8 = var_6.returns
    var_9 = False
    var_10 = False
    var_0.func_api(var_1, var_2, var_7, var_8, has_self=var_9, cls_method=var_10)

def test_case_46():
    var_0 = '_tU(2 J?WLH0rw"_Rr'
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
    var_2 = False
    var_3 = None
    var_4 = var_1.func_ann(var_0, var_2, has_self=var_2, cls_method=var_3)
    var_5 = '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'
    var_6 = var_1.parse(var_5, var_5)
    assert var_1.level == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 4}
    assert var_1.doc == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n'}
    assert var_1.docstring == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 'Module docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 'Function docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 'Class docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 'Method docstring.'}
    assert var_1.imp == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': {*()}}
    assert var_1.root == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n'}
    assert var_1.alias == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.os': 'os', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.List': 'typing.List', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.Optional': 'typing.Optional', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '42'}
    assert var_1.const == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': 'int'}
    var_7 = "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_mEthod(cls,x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_8 = var_1.parse(var_0, var_7)
    assert var_1.level == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 4, '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 4, '_tU(2 J?WLH0rw"_Rr': 0, '_tU(2 J?WLH0rw"_Rr.MyClass': 0, '_tU(2 J?WLH0rw"_Rr.MyClass.static_method': 0, '_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod': 0}
    assert var_1.doc == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '## Module `{}`\n<a id="{}"></a>\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '### my_function()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '#### MyClass.method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\n', '_tU(2 J?WLH0rw"_Rr': '## Module `{}`\n<a id="{}"></a>\n\n', '_tU(2 J?WLH0rw"_Rr.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '_tU(2 J?WLH0rw"_Rr.MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', '_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod': '#### MyClass.class_mEthod()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\n'}
    assert var_1.docstring == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': 'Module docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': 'Function docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': 'Class docstring.', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': 'Method docstring.', '_tU(2 J?WLH0rw"_Rr.MyClass': 'Test class.', '_tU(2 J?WLH0rw"_Rr.MyClass.static_method': 'Static method.', '_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod': 'Class method.'}
    assert var_1.imp == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': {*()}, '_tU(2 J?WLH0rw"_Rr': {*()}}
    assert var_1.root == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method': '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n', '_tU(2 J?WLH0rw"_Rr': '_tU(2 J?WLH0rw"_Rr', '_tU(2 J?WLH0rw"_Rr.MyClass': '_tU(2 J?WLH0rw"_Rr', '_tU(2 J?WLH0rw"_Rr.MyClass.static_method': '_tU(2 J?WLH0rw"_Rr', '_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod': '_tU(2 J?WLH0rw"_Rr'}
    assert var_1.alias == {'\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.os': 'os', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.List': 'typing.List', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.Optional': 'typing.Optional', '\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MY_CONSTANT': '42', '_tU(2 J?WLH0rw"_Rr.staticmethod': 'functools.staticmethod', '_tU(2 J?WLH0rw"_Rr.classmethod': 'functools.classmethod'}
    var_9 = var_1.globals(var_6, var_5)
    var_10 = var_1.compile()
    assert var_10 == '## Module `\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n`\n<a id="\n\'\'\'module docstring-\'\'\'\n\nimport os\nfrom typing import list, optional\n\nmy_constant: int = 42\n\ndef my_function(x: int, y: str = "default") -> list[str]:\n    \'\'\'function docstring-\'\'\'\n    return [y] * x\n\nclass myclass:\n    \'\'\'class docstring-\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'method docstring-\'\'\'\n        return str(arg)\n"></a>\n\n| Constants | Type |\n|:---------:|:----:|\n| `MY_CONSTANT` | `int` |\n\nModule docstring.\n\n### my_function()\n\n*Full name:* `\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.my_function`\n<a id="\n\'\'\'module docstring-\'\'\'\n\nimport os\nfrom typing import list, optional\n\nmy_constant: int = 42\n\ndef my_function(x: int, y: str = "default") -> list[str]:\n    \'\'\'function docstring-\'\'\'\n    return [y] * x\n\nclass myclass:\n    \'\'\'class docstring-\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'method docstring-\'\'\'\n        return str(arg)\n-my_function"></a>\n\n| x | y | return |\n|:---:|:---:|:------:|\n| `int` | `str` | `list[str]` |\n|   | `\'default\'` |   |   |\n\nFunction docstring.\n\n### class MyClass\n\n*Full name:* `\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass`\n<a id="\n\'\'\'module docstring-\'\'\'\n\nimport os\nfrom typing import list, optional\n\nmy_constant: int = 42\n\ndef my_function(x: int, y: str = "default") -> list[str]:\n    \'\'\'function docstring-\'\'\'\n    return [y] * x\n\nclass myclass:\n    \'\'\'class docstring-\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'method docstring-\'\'\'\n        return str(arg)\n-myclass"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\nClass docstring.\n\n#### MyClass.method()\n\n*Full name:* `\n\'\'\'Module docstring.\'\'\'\n\nimport os\nfrom typing import List, Optional\n\nMY_CONSTANT: int = 42\n\ndef my_function(x: int, y: str = "default") -> List[str]:\n    \'\'\'Function docstring.\'\'\'\n    return [y] * x\n\nclass MyClass:\n    \'\'\'Class docstring.\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'Method docstring.\'\'\'\n        return str(arg)\n.MyClass.method`\n<a id="\n\'\'\'module docstring-\'\'\'\n\nimport os\nfrom typing import list, optional\n\nmy_constant: int = 42\n\ndef my_function(x: int, y: str = "default") -> list[str]:\n    \'\'\'function docstring-\'\'\'\n    return [y] * x\n\nclass myclass:\n    \'\'\'class docstring-\'\'\'\n    attr1: int\n    attr2: str = "value"\n    \n    def method(self, arg: int) -> str:\n        \'\'\'method docstring-\'\'\'\n        return str(arg)\n-myclass-method"></a>\n\n| self | arg | return |\n|:----:|:---:|:------:|\n| `Self` | `int` | `str` |\n\nMethod docstring.\n'
    var_11 = var_1.load_docstring(var_0, var_8)
    var_12 = var_1.__repr__()
    assert var_12 == 'Parser(link=True, b_level=1, toc=False, level={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\': 4, \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.my_function\': 4, \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass\': 4, \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass.method\': 4, \'_tU(2 J?WLH0rw"_Rr\': 0, \'_tU(2 J?WLH0rw"_Rr.MyClass\': 0, \'_tU(2 J?WLH0rw"_Rr.MyClass.static_method\': 0, \'_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod\': 0}, doc={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.my_function\': \'### my_function()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| x | y | return |\\n|:---:|:---:|:------:|\\n| `int` | `str` | `list[str]` |\\n|   | `\\\'default\\\'` |   |   |\\n\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Members | Type |\\n|:-------:|:----:|\\n| `attr1` | `int` |\\n| `attr2` | `str` |\\n\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass.method\': \'#### MyClass.method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | arg | return |\\n|:----:|:---:|:------:|\\n| `Self` | `int` | `str` |\\n\\n\', \'_tU(2 J?WLH0rw"_Rr\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'_tU(2 J?WLH0rw"_Rr.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'_tU(2 J?WLH0rw"_Rr.MyClass.static_method\': \'#### MyClass.static_method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.staticmethod` |\\n\\n| x | return |\\n|:---:|:------:|\\n| `Self` | `Self` |\\n\\n\', \'_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod\': \'#### MyClass.class_mEthod()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.classmethod` |\\n\\n| cls | x | return |\\n|:---:|:---:|:------:|\\n| `Self` | `int` | `int` |\\n\\n\'}, docstring={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\': \'Module docstring.\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.my_function\': \'Function docstring.\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass\': \'Class docstring.\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass.method\': \'Method docstring.\', \'_tU(2 J?WLH0rw"_Rr.MyClass\': \'Test class.\', \'_tU(2 J?WLH0rw"_Rr.MyClass.static_method\': \'Static method.\', \'_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod\': \'Class method.\'}, imp={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\': set(), \'_tU(2 J?WLH0rw"_Rr\': set()}, root={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\': \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MY_CONSTANT\': \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.my_function\': \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass\': \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass.method\': \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\', \'_tU(2 J?WLH0rw"_Rr\': \'_tU(2 J?WLH0rw"_Rr\', \'_tU(2 J?WLH0rw"_Rr.MyClass\': \'_tU(2 J?WLH0rw"_Rr\', \'_tU(2 J?WLH0rw"_Rr.MyClass.static_method\': \'_tU(2 J?WLH0rw"_Rr\', \'_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod\': \'_tU(2 J?WLH0rw"_Rr\'}, alias={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.os\': \'os\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.List\': \'typing.List\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.Optional\': \'typing.Optional\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MY_CONSTANT\': \'42\', \'_tU(2 J?WLH0rw"_Rr.staticmethod\': \'functools.staticmethod\', \'_tU(2 J?WLH0rw"_Rr.classmethod\': \'functools.classmethod\'}, const={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MY_CONSTANT\': \'int\'})'
    var_13 = True
    var_14 = [var_8, var_5]
    var_15 = var_1.func_ann(var_11, var_14, has_self=var_11, cls_method=var_11)
    var_16 = module_0.walk_body(var_5)
    var_17 = module_0.table(items=var_16)
    assert var_17 == '||\n||\n| \n |\n| \' |\n| \' |\n| \' |\n| M |\n| o |\n| d |\n| u |\n| l |\n| e |\n|   |\n| d |\n| o |\n| c |\n| s |\n| t |\n| r |\n| i |\n| n |\n| g |\n| . |\n| \' |\n| \' |\n| \' |\n| \n |\n| \n |\n| i |\n| m |\n| p |\n| o |\n| r |\n| t |\n|   |\n| o |\n| s |\n| \n |\n| f |\n| r |\n| o |\n| m |\n|   |\n| t |\n| y |\n| p |\n| i |\n| n |\n| g |\n|   |\n| i |\n| m |\n| p |\n| o |\n| r |\n| t |\n|   |\n| L |\n| i |\n| s |\n| t |\n| , |\n|   |\n| O |\n| p |\n| t |\n| i |\n| o |\n| n |\n| a |\n| l |\n| \n |\n| \n |\n| M |\n| Y |\n| _ |\n| C |\n| O |\n| N |\n| S |\n| T |\n| A |\n| N |\n| T |\n| : |\n|   |\n| i |\n| n |\n| t |\n|   |\n| = |\n|   |\n| 4 |\n| 2 |\n| \n |\n| \n |\n| d |\n| e |\n| f |\n|   |\n| m |\n| y |\n| _ |\n| f |\n| u |\n| n |\n| c |\n| t |\n| i |\n| o |\n| n |\n| ( |\n| x |\n| : |\n|   |\n| i |\n| n |\n| t |\n| , |\n|   |\n| y |\n| : |\n|   |\n| s |\n| t |\n| r |\n|   |\n| = |\n|   |\n| " |\n| d |\n| e |\n| f |\n| a |\n| u |\n| l |\n| t |\n| " |\n| ) |\n|   |\n| - |\n| > |\n|   |\n| L |\n| i |\n| s |\n| t |\n| [ |\n| s |\n| t |\n| r |\n| ] |\n| : |\n| \n |\n|   |\n|   |\n|   |\n|   |\n| \' |\n| \' |\n| \' |\n| F |\n| u |\n| n |\n| c |\n| t |\n| i |\n| o |\n| n |\n|   |\n| d |\n| o |\n| c |\n| s |\n| t |\n| r |\n| i |\n| n |\n| g |\n| . |\n| \' |\n| \' |\n| \' |\n| \n |\n|   |\n|   |\n|   |\n|   |\n| r |\n| e |\n| t |\n| u |\n| r |\n| n |\n|   |\n| [ |\n| y |\n| ] |\n|   |\n| * |\n|   |\n| x |\n| \n |\n| \n |\n| c |\n| l |\n| a |\n| s |\n| s |\n|   |\n| M |\n| y |\n| C |\n| l |\n| a |\n| s |\n| s |\n| : |\n| \n |\n|   |\n|   |\n|   |\n|   |\n| \' |\n| \' |\n| \' |\n| C |\n| l |\n| a |\n| s |\n| s |\n|   |\n| d |\n| o |\n| c |\n| s |\n| t |\n| r |\n| i |\n| n |\n| g |\n| . |\n| \' |\n| \' |\n| \' |\n| \n |\n|   |\n|   |\n|   |\n|   |\n| a |\n| t |\n| t |\n| r |\n| 1 |\n| : |\n|   |\n| i |\n| n |\n| t |\n| \n |\n|   |\n|   |\n|   |\n|   |\n| a |\n| t |\n| t |\n| r |\n| 2 |\n| : |\n|   |\n| s |\n| t |\n| r |\n|   |\n| = |\n|   |\n| " |\n| v |\n| a |\n| l |\n| u |\n| e |\n| " |\n| \n |\n|   |\n|   |\n|   |\n|   |\n| \n |\n|   |\n|   |\n|   |\n|   |\n| d |\n| e |\n| f |\n|   |\n| m |\n| e |\n| t |\n| h |\n| o |\n| d |\n| ( |\n| s |\n| e |\n| l |\n| f |\n| , |\n|   |\n| a |\n| r |\n| g |\n| : |\n|   |\n| i |\n| n |\n| t |\n| ) |\n|   |\n| - |\n| > |\n|   |\n| s |\n| t |\n| r |\n| : |\n| \n |\n|   |\n|   |\n|   |\n|   |\n|   |\n|   |\n|   |\n|   |\n| \' |\n| \' |\n| \' |\n| M |\n| e |\n| t |\n| h |\n| o |\n| d |\n|   |\n| d |\n| o |\n| c |\n| s |\n| t |\n| r |\n| i |\n| n |\n| g |\n| . |\n| \' |\n| \' |\n| \' |\n| \n |\n|   |\n|   |\n|   |\n|   |\n|   |\n|   |\n|   |\n|   |\n| r |\n| e |\n| t |\n| u |\n| r |\n| n |\n|   |\n| s |\n| t |\n| r |\n| ( |\n| a |\n| r |\n| g |\n| ) |\n| \n |\n\n'
    var_18 = {}
    var_19 = module_0.Parser(toc=var_13, root=var_18, alias=var_8)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'apimd.parser.Parser'
    assert var_19.link is True
    assert var_19.b_level == 1
    assert var_19.toc is True
    assert var_19.level == {}
    assert var_19.doc == {}
    assert var_19.docstring == {}
    assert var_19.imp == {}
    assert var_19.root == {}
    assert var_19.alias is None
    assert var_19.const == {}
    var_20 = module_0.const_type(var_17)
    assert var_20 == 'Any'
    var_21 = [var_12]
    var_22 = module_1.Set(*var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Set'
    assert var_22.elts == 'Parser(link=True, b_level=1, toc=False, level={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\': 4, \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.my_function\': 4, \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass\': 4, \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass.method\': 4, \'_tU(2 J?WLH0rw"_Rr\': 0, \'_tU(2 J?WLH0rw"_Rr.MyClass\': 0, \'_tU(2 J?WLH0rw"_Rr.MyClass.static_method\': 0, \'_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod\': 0}, doc={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.my_function\': \'### my_function()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| x | y | return |\\n|:---:|:---:|:------:|\\n| `int` | `str` | `list[str]` |\\n|   | `\\\'default\\\'` |   |   |\\n\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Members | Type |\\n|:-------:|:----:|\\n| `attr1` | `int` |\\n| `attr2` | `str` |\\n\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass.method\': \'#### MyClass.method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| self | arg | return |\\n|:----:|:---:|:------:|\\n| `Self` | `int` | `str` |\\n\\n\', \'_tU(2 J?WLH0rw"_Rr\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'_tU(2 J?WLH0rw"_Rr.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'_tU(2 J?WLH0rw"_Rr.MyClass.static_method\': \'#### MyClass.static_method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.staticmethod` |\\n\\n| x | return |\\n|:---:|:------:|\\n| `Self` | `Self` |\\n\\n\', \'_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod\': \'#### MyClass.class_mEthod()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.classmethod` |\\n\\n| cls | x | return |\\n|:---:|:---:|:------:|\\n| `Self` | `int` | `int` |\\n\\n\'}, docstring={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\': \'Module docstring.\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.my_function\': \'Function docstring.\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass\': \'Class docstring.\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass.method\': \'Method docstring.\', \'_tU(2 J?WLH0rw"_Rr.MyClass\': \'Test class.\', \'_tU(2 J?WLH0rw"_Rr.MyClass.static_method\': \'Static method.\', \'_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod\': \'Class method.\'}, imp={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\': set(), \'_tU(2 J?WLH0rw"_Rr\': set()}, root={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\': \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MY_CONSTANT\': \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.my_function\': \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass\': \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MyClass.method\': \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n\', \'_tU(2 J?WLH0rw"_Rr\': \'_tU(2 J?WLH0rw"_Rr\', \'_tU(2 J?WLH0rw"_Rr.MyClass\': \'_tU(2 J?WLH0rw"_Rr\', \'_tU(2 J?WLH0rw"_Rr.MyClass.static_method\': \'_tU(2 J?WLH0rw"_Rr\', \'_tU(2 J?WLH0rw"_Rr.MyClass.class_mEthod\': \'_tU(2 J?WLH0rw"_Rr\'}, alias={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.os\': \'os\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.List\': \'typing.List\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.Optional\': \'typing.Optional\', \'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MY_CONSTANT\': \'42\', \'_tU(2 J?WLH0rw"_Rr.staticmethod\': \'functools.staticmethod\', \'_tU(2 J?WLH0rw"_Rr.classmethod\': \'functools.classmethod\'}, const={\'\\n\\\'\\\'\\\'Module docstring.\\\'\\\'\\\'\\n\\nimport os\\nfrom typing import List, Optional\\n\\nMY_CONSTANT: int = 42\\n\\ndef my_function(x: int, y: str = "default") -> List[str]:\\n    \\\'\\\'\\\'Function docstring.\\\'\\\'\\\'\\n    return [y] * x\\n\\nclass MyClass:\\n    \\\'\\\'\\\'Class docstring.\\\'\\\'\\\'\\n    attr1: int\\n    attr2: str = "value"\\n    \\n    def method(self, arg: int) -> str:\\n        \\\'\\\'\\\'Method docstring.\\\'\\\'\\\'\\n        return str(arg)\\n.MY_CONSTANT\': \'int\'})'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_23 = module_0.const_type(var_22)
    assert var_23 == 'set'

def test_case_47():
    var_0 = '/'
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
    var_2 = False
    var_3 = None
    var_4 = var_1.func_ann(var_0, var_2, has_self=var_2, cls_method=var_3)
    var_5 = 'pPIo,IHrA=Byu#p'
    var_6 = var_1.parse(var_5, var_5)
    assert var_1.level == {'pPIo,IHrA=Byu#p': 0}
    assert var_1.doc == {'pPIo,IHrA=Byu#p': '## Module `{}`\n<a id="{}"></a>\n\n'}
    assert var_1.imp == {'pPIo,IHrA=Byu#p': {*()}}
    assert var_1.root == {'pPIo,IHrA=Byu#p': 'pPIo,IHrA=Byu#p'}
    var_7 = "\nfrom functools import staticmethod, classmethod\n\nclass MyClass:\n    '''Test class.'''\n    \n    @staticmethod\n    def static_method(x: int) -> int:\n        '''Static method.'''\n        return x\n    \n    @classmethod\n    def class_mEthod(cls, x: int) -> int:\n        '''Class method.'''\n        return x\n"
    var_8 = var_1.parse(var_0, var_7)
    assert var_1.level == {'pPIo,IHrA=Byu#p': 0, '/': 0, '/.MyClass': 0, '/.MyClass.static_method': 0, '/.MyClass.class_mEthod': 0}
    assert var_1.doc == {'pPIo,IHrA=Byu#p': '## Module `{}`\n<a id="{}"></a>\n\n', '/': '## Module `{}`\n<a id="{}"></a>\n\n', '/.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n', '/.MyClass.static_method': '#### MyClass.static_method()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\n', '/.MyClass.class_mEthod': '#### MyClass.class_mEthod()\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\n'}
    assert var_1.docstring == {'/.MyClass': 'Test class.', '/.MyClass.static_method': 'Static method.', '/.MyClass.class_mEthod': 'Class method.'}
    assert var_1.imp == {'pPIo,IHrA=Byu#p': {*()}, '/': {*()}}
    assert var_1.root == {'pPIo,IHrA=Byu#p': 'pPIo,IHrA=Byu#p', '/': '/', '/.MyClass': '/', '/.MyClass.static_method': '/', '/.MyClass.class_mEthod': '/'}
    assert var_1.alias == {'/.staticmethod': 'functools.staticmethod', '/.classmethod': 'functools.classmethod'}
    var_9 = var_1.__repr__()
    assert var_9 == 'Parser(link=True, b_level=1, toc=False, level={\'pPIo,IHrA=Byu#p\': 0, \'/\': 0, \'/.MyClass\': 0, \'/.MyClass.static_method\': 0, \'/.MyClass.class_mEthod\': 0}, doc={\'pPIo,IHrA=Byu#p\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'/\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'/.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'/.MyClass.static_method\': \'#### MyClass.static_method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.staticmethod` |\\n\\n| x | return |\\n|:---:|:------:|\\n| `Self` | `Self` |\\n\\n\', \'/.MyClass.class_mEthod\': \'#### MyClass.class_mEthod()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.classmethod` |\\n\\n| cls | x | return |\\n|:---:|:---:|:------:|\\n| `Self` | `int` | `int` |\\n\\n\'}, docstring={\'/.MyClass\': \'Test class.\', \'/.MyClass.static_method\': \'Static method.\', \'/.MyClass.class_mEthod\': \'Class method.\'}, imp={\'pPIo,IHrA=Byu#p\': set(), \'/\': set()}, root={\'pPIo,IHrA=Byu#p\': \'pPIo,IHrA=Byu#p\', \'/\': \'/\', \'/.MyClass\': \'/\', \'/.MyClass.static_method\': \'/\', \'/.MyClass.class_mEthod\': \'/\'}, alias={\'/.staticmethod\': \'functools.staticmethod\', \'/.classmethod\': \'functools.classmethod\'}, const={})'
    var_10 = var_1.compile()
    assert var_10 == '## Module `/`\n<a id="/"></a>\n\n### class MyClass\n\n*Full name:* `/.MyClass`\n<a id="/-myclass"></a>\n\nTest class.\n\n#### MyClass.class_mEthod()\n\n*Full name:* `/.MyClass.class_mEthod`\n<a id="/-myclass-class_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.classmethod` |\n\n| cls | x | return |\n|:---:|:---:|:------:|\n| `Self` | `int` | `int` |\n\nClass method.\n\n#### MyClass.static_method()\n\n*Full name:* `/.MyClass.static_method`\n<a id="/-myclass-static_method"></a>\n\n| Decorators |\n|:----------:|\n| `@functools.staticmethod` |\n\n| x | return |\n|:---:|:------:|\n| `Self` | `Self` |\n\nStatic method.\n'
    var_11 = var_1.__repr__()
    assert var_11 == 'Parser(link=True, b_level=1, toc=False, level={\'pPIo,IHrA=Byu#p\': 0, \'/\': 0, \'/.MyClass\': 0, \'/.MyClass.static_method\': 0, \'/.MyClass.class_mEthod\': 0}, doc={\'pPIo,IHrA=Byu#p\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'/\': \'## Module `{}`\\n<a id="{}"></a>\\n\\n\', \'/.MyClass\': \'### class MyClass\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n\', \'/.MyClass.static_method\': \'#### MyClass.static_method()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.staticmethod` |\\n\\n| x | return |\\n|:---:|:------:|\\n| `Self` | `Self` |\\n\\n\', \'/.MyClass.class_mEthod\': \'#### MyClass.class_mEthod()\\n\\n*Full name:* `{}`\\n<a id="{}"></a>\\n\\n| Decorators |\\n|:----------:|\\n| `@functools.classmethod` |\\n\\n| cls | x | return |\\n|:---:|:---:|:------:|\\n| `Self` | `int` | `int` |\\n\\n\'}, docstring={\'/.MyClass\': \'Test class.\', \'/.MyClass.static_method\': \'Static method.\', \'/.MyClass.class_mEthod\': \'Class method.\'}, imp={\'pPIo,IHrA=Byu#p\': set(), \'/\': set()}, root={\'pPIo,IHrA=Byu#p\': \'pPIo,IHrA=Byu#p\', \'/\': \'/\', \'/.MyClass\': \'/\', \'/.MyClass.static_method\': \'/\', \'/.MyClass.class_mEthod\': \'/\'}, alias={\'/.staticmethod\': \'functools.staticmethod\', \'/.classmethod\': \'functools.classmethod\'}, const={})'
    var_12 = var_1.parse(var_0, var_7)
    var_13 = True
    var_14 = [var_8, var_5]
    var_15 = var_1.func_ann(var_11, var_14, has_self=var_11, cls_method=var_11)
    var_16 = module_0.walk_body(var_5)
    var_17 = module_0.table(items=var_16)
    assert var_17 == '||\n||\n| p |\n| P |\n| I |\n| o |\n| , |\n| I |\n| H |\n| r |\n| A |\n| = |\n| B |\n| y |\n| u |\n| # |\n| p |\n\n'
    var_18 = {}
    var_19 = module_0.Parser(toc=var_13, root=var_18, alias=var_8)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'apimd.parser.Parser'
    assert var_19.link is True
    assert var_19.b_level == 1
    assert var_19.toc is True
    assert var_19.level == {}
    assert var_19.doc == {}
    assert var_19.docstring == {}
    assert var_19.imp == {}
    assert var_19.root == {}
    assert var_19.alias is None
    assert var_19.const == {}
    var_20 = module_0.const_type(var_17)
    assert var_20 == 'Any'
    var_21 = [var_12]
    var_22 = module_1.Set(*var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'ast.Set'
    assert var_22.elts is None
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096

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
    var_1 = "\nclass MyClass(BaseClass, OtherBase):\n    '''Class docstring'''\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'test_module': 0, 'test_module.MyClass': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `BaseClass` |\n| `OtherBase` |\n\n'}
    assert var_0.docstring == {'test_module.MyClass': 'Class docstring'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module'}
    var_4 = module_1.parse(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Module'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'builtins.list'
    assert len(var_4.body) == 1
    assert var_4.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = 'test_module.MyClass'
    var_8 = var_6.bases
    var_9 = var_6.body
    var_10 = var_0.class_api(var_2, var_7, var_8, var_9)
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `BaseClass` |\n| `OtherBase` |\n\n| Bases |\n|:-----:|\n| `BaseClass` |\n| `OtherBase` |\n\n'}
    var_11 = "\nclass MyClass2:\n    '''Class docstring'''\n    attr1: int = 5\n    attr2: str\n    _private: float = 3.14\n"
    var_12 = 'test_module2'
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
    var_14 = var_13.parse(var_12, var_11)
    assert var_13.level == {'test_module2': 0, 'test_module2.MyClass2': 0}
    assert var_13.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.MyClass2': '### class MyClass2\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n'}
    assert var_13.docstring == {'test_module2.MyClass2': 'Class docstring'}
    assert var_13.imp == {'test_module2': {*()}}
    assert var_13.root == {'test_module2': 'test_module2', 'test_module2.MyClass2': 'test_module2'}
    var_15 = module_1.parse(var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Module'
    assert f'{type(var_15.body).__module__}.{type(var_15.body).__qualname__}' == 'builtins.list'
    assert len(var_15.body) == 1
    assert var_15.type_ignores == []
    var_16 = var_15.body[var_5]
    var_17 = 'test_module2.MyClass2'
    var_18 = var_16.bases
    var_19 = var_16.body
    var_20 = var_13.class_api(var_12, var_17, var_18, var_19)
    assert var_13.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.MyClass2': '### class MyClass2\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n'}
    var_21 = '\nfrom enum import Enum\n\nclass MyEnum(Enum):\n    \'\'\'Enum docstring\'\'\'\n    VALUE1: int = 1\n    VALUE2: str = "test"\n'
    var_22 = 'test_module3'
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
    var_24 = var_23.parse(var_22, var_21)
    assert var_23.level == {'test_module3': 0, 'test_module3.MyEnum': 0}
    assert var_23.doc == {'test_module3': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module3.MyEnum': '### class MyEnum\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| VALUE1 |\n| VALUE2 |\n\n'}
    assert var_23.docstring == {'test_module3.MyEnum': 'Enum docstring'}
    assert var_23.imp == {'test_module3': {*()}}
    assert var_23.root == {'test_module3': 'test_module3', 'test_module3.MyEnum': 'test_module3'}
    assert var_23.alias == {'test_module3.Enum': 'enum.Enum'}
    var_25 = module_1.parse(var_21)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'ast.Module'
    assert f'{type(var_25.body).__module__}.{type(var_25.body).__qualname__}' == 'builtins.list'
    assert len(var_25.body) == 2
    assert var_25.type_ignores == []
    var_26 = 1
    var_27 = var_25.body[var_26]
    var_28 = 'test_module3.MyEnum'
    var_29 = var_27.bases
    var_30 = var_23.class_api(var_22, var_28, var_29, var_18)
    assert var_23.doc == {'test_module3': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module3.MyEnum': '### class MyEnum\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n| Enums |\n|:-----:|\n| VALUE1 |\n| VALUE2 |\n\n| Bases |\n|:-----:|\n| `enum.Enum` |\n\n'}
    var_31 = '\nclass MyClass4:\n    \'\'\'Class docstring\'\'\'\n    attr1: int = 1\n    attr2: str = "test"\n    del attr1\n'
    var_32 = 'test_module4'
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
    var_34 = var_33.parse(var_32, var_31)
    assert var_33.level == {'test_module4': 0, 'test_module4.MyClass4': 0}
    assert var_33.doc == {'test_module4': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module4.MyClass4': '### class MyClass4\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr2` | `str` |\n\n'}
    assert var_33.docstring == {'test_module4.MyClass4': 'Class docstring'}
    assert var_33.imp == {'test_module4': {*()}}
    assert var_33.root == {'test_module4': 'test_module4', 'test_module4.MyClass4': 'test_module4'}
    var_35 = module_1.parse(var_31)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'ast.Module'
    assert f'{type(var_35.body).__module__}.{type(var_35.body).__qualname__}' == 'builtins.list'
    assert len(var_35.body) == 1
    assert var_35.type_ignores == []
    var_36 = var_35.body[var_5]
    var_37 = 'test_module4.MyClass4'
    var_38 = var_36.bases
    var_39 = var_36.body
    var_40 = var_33.class_api(var_32, var_37, var_38, var_39)
    assert var_33.doc == {'test_module4': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module4.MyClass4': '### class MyClass4\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr2` | `str` |\n\n| Members | Type |\n|:-------:|:----:|\n| `attr2` | `str` |\n\n'}
    var_41 = '\nclass EmptyClass:\n    pass\n'
    var_42 = 'test_module5'
    var_43 = module_0.Parser()
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'apimd.parser.Parser'
    assert var_43.link is True
    assert var_43.b_level == 1
    assert var_43.toc is False
    assert var_43.level == {}
    assert var_43.doc == {}
    assert var_43.docstring == {}
    assert var_43.imp == {}
    assert var_43.root == {}
    assert var_43.alias == {}
    assert var_43.const == {}
    var_44 = var_43.parse(var_42, var_41)
    assert var_43.level == {'test_module5': 0, 'test_module5.EmptyClass': 0}
    assert var_43.doc == {'test_module5': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module5.EmptyClass': '### class EmptyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n'}
    assert var_43.imp == {'test_module5': {*()}}
    assert var_43.root == {'test_module5': 'test_module5', 'test_module5.EmptyClass': 'test_module5'}
    var_45 = module_1.parse(var_41)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'ast.Module'
    assert f'{type(var_45.body).__module__}.{type(var_45.body).__qualname__}' == 'builtins.list'
    assert len(var_45.body) == 1
    assert var_45.type_ignores == []
    var_46 = var_45.body[var_5]
    var_47 = 'test_module5.EmptyClass'
    var_48 = var_46.bases
    var_49 = var_46.body
    var_50 = var_43.class_api(var_42, var_47, var_48, var_49)

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
    var_1 = "\nclass MyClass(BaseClass, OtherBase):\n    '''Class docstring'''\n    pass\n"
    var_2 = 'test_module'
    var_3 = var_0.parse(var_2, var_1)
    assert var_0.level == {'test_module': 0, 'test_module.MyClass': 0}
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `BaseClass` |\n| `OtherBase` |\n\n'}
    assert var_0.docstring == {'test_module.MyClass': 'Class docstring'}
    assert var_0.imp == {'test_module': {*()}}
    assert var_0.root == {'test_module': 'test_module', 'test_module.MyClass': 'test_module'}
    var_4 = module_1.parse(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'ast.Module'
    assert f'{type(var_4.body).__module__}.{type(var_4.body).__qualname__}' == 'builtins.list'
    assert len(var_4.body) == 1
    assert var_4.type_ignores == []
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
    var_5 = 0
    var_6 = var_4.body[var_5]
    var_7 = 'test_module.MyClass'
    var_8 = var_6.bases
    var_9 = var_6.body
    var_10 = var_0.class_api(var_2, var_7, var_8, var_9)
    assert var_0.doc == {'test_module': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module.MyClass': '### class MyClass\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Bases |\n|:-----:|\n| `BaseClass` |\n| `OtherBase` |\n\n| Bases |\n|:-----:|\n| `BaseClass` |\n| `OtherBase` |\n\n'}
    var_11 = "\nclass MyClass2:\n    '''Class docstring'''\n    attr1: int = 5\n    attr2: str\n    _private: float = 3.14\n"
    var_12 = 'test_module2'
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
    var_14 = var_13.parse(var_12, var_11)
    assert var_13.level == {'test_module2': 0, 'test_module2.MyClass2': 0}
    assert var_13.doc == {'test_module2': '## Module `{}`\n<a id="{}"></a>\n\n', 'test_module2.MyClass2': '### class MyClass2\n\n*Full name:* `{}`\n<a id="{}"></a>\n\n| Members | Type |\n|:-------:|:----:|\n| `attr1` | `int` |\n| `attr2` | `str` |\n\n'}
    assert var_13.docstring == {'test_module2.MyClass2': 'Class docstring'}
    assert var_13.imp == {'test_module2': {*()}}
    assert var_13.root == {'test_module2': 'test_module2', 'test_module2.MyClass2': 'test_module2'}
    var_15 = module_1.parse(var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'ast.Module'
    assert f'{type(var_15.body).__module__}.{type(var_15.body).__qualname__}' == 'builtins.list'
    assert len(var_15.body) == 1
    assert var_15.type_ignores == []
    var_16 = var_15.body[var_5]
    var_17 = 'test_module2.MyClass2'
    var_18 = var_16.bases
    var_13.class_api(var_12, var_17, var_18, var_16)