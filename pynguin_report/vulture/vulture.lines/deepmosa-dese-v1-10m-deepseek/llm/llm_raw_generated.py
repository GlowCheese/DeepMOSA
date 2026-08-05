####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_get_first_line_number_with_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = [MockDecorator(10)]
            self.lineno = 15
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_without_decorator_list():
    class MockNode:
        def __init__(self):
            self.lineno = 20
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 20

def test_get_first_line_number_with_empty_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 25
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 25

def test_get_first_line_number_with_multiple_decorators():
    class MockNode:
        def __init__(self):
            self.decorator_list = [MockDecorator(5), MockDecorator(8)]
            self.lineno = 12
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 5
```


# LLM-generated content at query #2
#--------------------------

def test_get_first_line_number_with_decorator():
    import ast
    code = """
@decorator
def func():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    result = get_first_line_number(func_node)
    assert result == 2

def test_get_first_line_number_without_decorator():
    import ast
    code = """
def func():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    result = get_first_line_number(func_node)
    assert result == 2

def test_get_first_line_number_with_multiple_decorators():
    import ast
    code = """
@decorator1
@decorator2
def func():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    result = get_first_line_number(func_node)
    assert result == 2

def test_get_first_line_number_class_with_decorator():
    import ast
    code = """
@decorator
class MyClass:
    pass
"""
    tree = ast.parse(code)
    class_node = tree.body[0]
    result = get_first_line_number(class_node)
    assert result == 2

def test_get_first_line_number_class_without_decorator():
    import ast
    code = """
class MyClass:
    pass
"""
    tree = ast.parse(code)
    class_node = tree.body[0]
    result = get_first_line_number(class_node)
    assert result == 2

def test_get_first_line_number_async_function_with_decorator():
    import ast
    code = """
@decorator
async def func():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    result = get_first_line_number(func_node)
    assert result == 2


# LLM-generated content at query #3
#--------------------------

def test_get_first_line_number_with_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = [MockDecorator(10)]
            self.lineno = 20
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    node = MockNode()
    assert get_first_line_number(node) == 10

def test_get_first_line_number_without_decorator_list():
    class MockNode:
        def __init__(self):
            self.lineno = 30
    node = MockNode()
    assert get_first_line_number(node) == 30

def test_get_first_line_number_with_empty_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 40
    node = MockNode()
    assert get_first_line_number(node) == 40

def test_get_first_line_number_with_node_no_decorator_attribute():
    class MockNode:
        def __init__(self):
            self.lineno = 50
    node = MockNode()
    assert get_first_line_number(node) == 50


# LLM-generated content at query #4
#--------------------------

```
def test_get_first_line_number_with_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = [MockDecorator(5)]
            self.lineno = 10
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_without_decorator_list():
    class MockNode:
        def __init__(self):
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_empty_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_no_decorator_attribute():
    class MockNode:
        def __init__(self):
            self.lineno = 7
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 7
```


# LLM-generated content at query #5
#--------------------------

```
def test_get_first_line_number_with_decorators_first_decorator_lineno():
    class FakeDecorator:
        lineno = 5
    class FakeNode:
        decorator_list = [FakeDecorator()]
        lineno = 10
    node = FakeNode()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_without_decorators_node_lineno():
    class FakeNode:
        decorator_list = []
        lineno = 15
    node = FakeNode()
    result = get_first_line_number(node)
    assert result == 15

def test_get_first_line_number_with_empty_decorator_list_node_lineno():
    class FakeNode:
        decorator_list = []
        lineno = 20
    node = FakeNode()
    result = get_first_line_number(node)
    assert result == 20

def test_get_first_line_number_with_multiple_decorators_first_decorator_lineno():
    class FakeDecorator1:
        lineno = 2
    class FakeDecorator2:
        lineno = 3
    class FakeNode:
        decorator_list = [FakeDecorator1(), FakeDecorator2()]
        lineno = 10
    node = FakeNode()
    result = get_first_line_number(node)
    assert result == 2

def test_get_first_line_number_node_without_decorator_attribute_lineno():
    class FakeNode:
        lineno = 25
    node = FakeNode()
    result = get_first_line_number(node)
    assert result == 25
```


# LLM-generated content at query #6
#--------------------------

```
def test_get_first_line_number_with_decorators():
    import ast
    code = """
@some_decorator
def foo():
    pass
"""
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    decorators = getattr(node, "decorator_list", [])
    assert bool(decorators) == True
```


# LLM-generated content at query #7
#--------------------------

```
def test_get_first_line_number_with_decorator():
    node = type('MockNode', (object,), {'decorator_list': [type('MockDecorator', (object,), {'lineno': 42})()], 'lineno': 10})()
    result = get_first_line_number(node)
    assert result == 42
```


# LLM-generated content at query #8
#--------------------------

```
def test_get_first_line_number_with_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('', (), {'lineno': 5})()]
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_empty_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 15
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 15

def test_get_first_line_number_no_decorator_list():
    class MockNode:
        def __init__(self):
            self.lineno = 20
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 20

def test_get_first_line_number_multiple_decorators():
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('', (), {'lineno': 3})(), type('', (), {'lineno': 4})()]
            self.lineno = 8
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 3
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_first_line_number_with_decorators():
    class MockNode:
        decorator_list = [type('MockDecorator', (), {'lineno': 5})()]
        lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_without_decorators():
    class MockNode:
        decorator_list = []
        lineno = 15
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 15

def test_get_first_line_number_with_none_decorator_list():
    class MockNode:
        lineno = 20
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 20


# LLM-generated content at query #2
#--------------------------

def test_get_first_line_number_with_decorators():
    class MockNode:
        def __init__(self):
            self.decorator_list = [MockDecorator()]
            self.lineno = 10
    class MockDecorator:
        def __init__(self):
            self.lineno = 5
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_without_decorators():
    class MockNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_with_empty_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = None
            self.lineno = 15
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 15


# LLM-generated content at query #3
#--------------------------

```
def test_get_first_line_number_with_decorators():
    class MockNode:
        decorator_list = [type('MockDecorator', (), {'lineno': 42})()]
        lineno = 10
    assert get_first_line_number(MockNode()) == 42
```


# LLM-generated content at query #4
#--------------------------

def test_get_first_line_number_with_decorator():
    import ast
    code = "@decorator\ndef foo():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == 1

def test_get_first_line_number_without_decorator():
    import ast
    code = "def foo():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == 1

def test_get_first_line_number_with_multiple_decorators():
    import ast
    code = "@d1\n@d2\ndef foo():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == 1

def test_get_first_line_number_class_with_decorator():
    import ast
    code = "@decorator\nclass Foo:\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == 1

def test_get_first_line_number_class_without_decorator():
    import ast
    code = "class Foo:\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == 1


# LLM-generated content at query #5
#--------------------------

```
def test_get_first_line_number_with_decorator():
    import ast
    code = """
@some_decorator
def foo():
    pass
"""
    tree = ast.parse(code)
    func_def = tree.body[0]
    result = get_first_line_number(func_def)
    assert result == func_def.decorator_list[0].lineno
```


# LLM-generated content at query #6
#--------------------------

```python
def test_decorators_list_not_empty_returns_first_decorator_lineno():
    import ast
    code = """
@some_decorator
def foo():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    result = get_first_line_number(func_node)
    assert result == 2
```


# LLM-generated content at query #7
#--------------------------

```
def test_get_first_line_number_with_decorators():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 10})()], 'lineno': 5})()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_without_decorators():
    node = type('Node', (), {'decorator_list': [], 'lineno': 5})()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_with_no_decorator_attribute():
    node = type('Node', (), {'lineno': 5})()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_with_multiple_decorators():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 10})(), type('Decorator', (), {'lineno': 12})()], 'lineno': 5})()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_with_none_decorator():
    node = type('Node', (), {'decorator_list': None, 'lineno': 5})()
    result = get_first_line_number(node)
    assert result == 5
```


# LLM-generated content at query #8
#--------------------------

def test_get_first_line_number_with_decorators():
    import ast
    code = "@dec\nclass A: pass"
    node = ast.parse(code).body[0]
    decorators = getattr(node, "decorator_list", [])
    assert bool(decorators) == True
    assert get_first_line_number(node) == decorators[0].lineno


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_first_line_number_with_decorator():
    class FakeDecorator:
        lineno = 5
    class FakeNode:
        decorator_list = [FakeDecorator()]
        lineno = 10
    node = FakeNode()
    assert get_first_line_number(node) == 5

def test_get_first_line_number_without_decorator():
    class FakeNode:
        decorator_list = []
        lineno = 10
    node = FakeNode()
    assert get_first_line_number(node) == 10

def test_get_first_line_number_with_no_decorator_attribute():
    class FakeNode:
        lineno = 10
    node = FakeNode()
    assert get_first_line_number(node) == 10

def test_get_first_line_number_with_multiple_decorators():
    class FakeDecorator1:
        lineno = 2
    class FakeDecorator2:
        lineno = 3
    class FakeNode:
        decorator_list = [FakeDecorator1(), FakeDecorator2()]
        lineno = 10
    node = FakeNode()
    assert get_first_line_number(node) == 2


# LLM-generated content at query #2
#--------------------------

def test_get_first_line_number_with_decorators():
    class MockDecorator:
        lineno = 5
    class MockNode:
        decorator_list = [MockDecorator()]
        lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 5
    assert result != node.lineno


# LLM-generated content at query #3
#--------------------------

def test_get_first_line_number_with_decorator():
    class MockDecorator:
        lineno = 42
    class MockNode:
        decorator_list = [MockDecorator()]
        lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 42
    assert bool(node.decorator_list) is True


# LLM-generated content at query #4
#--------------------------

def test_get_first_line_number_with_decorators():
    class FakeNode:
        decorator_list = [type('FakeDecorator', (), {'lineno': 42})()]
        lineno = 100
    node = FakeNode()
    result = get_first_line_number(node)
    assert result == 42


# LLM-generated content at query #5
#--------------------------

def test_get_first_line_number_with_decorators():
    class FakeDecorator:
        lineno = 42
    class FakeNode:
        decorator_list = [FakeDecorator()]
        lineno = 100
    node = FakeNode()
    result = get_first_line_number(node)
    assert result == 42
    assert bool(node.decorator_list) is True


# LLM-generated content at query #6
#--------------------------

def test_get_first_line_number_with_decorators():
    class FakeDecorator:
        lineno = 42
    class FakeNode:
        decorator_list = [FakeDecorator()]
        lineno = 10
    node = FakeNode()
    assert get_first_line_number(node) == 42
    assert bool(node.decorator_list) == True


# LLM-generated content at query #7
#--------------------------

def test_get_first_line_number_with_decorators():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 5})()], 'lineno': 10})()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_without_decorators():
    node = type('Node', (), {'decorator_list': [], 'lineno': 10})()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_with_no_decorator_attribute():
    node = type('Node', (), {'lineno': 10})()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_with_multiple_decorators():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 5})(), type('Decorator', (), {'lineno': 6})()], 'lineno': 10})()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_with_empty_decorator_list():
    node = type('Node', (), {'decorator_list': [], 'lineno': 0})()
    result = get_first_line_number(node)
    assert result == 0


# LLM-generated content at query #8
#--------------------------

def test_get_first_line_number_with_decorator():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 5})()], 'lineno': 10})()
    assert get_first_line_number(node) == 5

def test_get_first_line_number_without_decorator():
    node = type('Node', (), {'decorator_list': [], 'lineno': 10})()
    assert get_first_line_number(node) == 10

def test_get_first_line_number_with_multiple_decorators():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 5})(), type('Decorator', (), {'lineno': 6})()], 'lineno': 10})()
    assert get_first_line_number(node) == 5

def test_get_first_line_number_no_decorator_attribute():
    node = type('Node', (), {'lineno': 10})()
    assert get_first_line_number(node) == 10

def test_get_first_line_number_none_decorator_list():
    node = type('Node', (), {'decorator_list': None, 'lineno': 10})()
    assert get_first_line_number(node) == 10


