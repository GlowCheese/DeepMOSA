####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_get_first_line_number_with_decorator():
    class FakeNode:
        decorator_list = [type('FakeDecorator', (), {'lineno': 10})()]
        lineno = 15
    node = FakeNode()
    assert get_first_line_number(node) == 10

def test_get_first_line_number_without_decorator():
    class FakeNode:
        decorator_list = []
        lineno = 20
    node = FakeNode()
    assert get_first_line_number(node) == 20

def test_get_first_line_number_no_decorator_attribute():
    class FakeNode:
        lineno = 25
    node = FakeNode()
    assert get_first_line_number(node) == 25

def test_get_first_line_number_multiple_decorators():
    class FakeNode:
        decorator_list = [
            type('FakeDecorator', (), {'lineno': 5})(),
            type('FakeDecorator', (), {'lineno': 8})()
        ]
        lineno = 12
    node = FakeNode()
    assert get_first_line_number(node) == 5
```


# LLM-generated content at query #2
#--------------------------

```
def test_get_first_line_number_with_decorators():
    import ast
    code = """
@some_decorator
def decorated_function():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    result = get_first_line_number(func_node)
    assert decorators = getattr(func_node, "decorator_list", [])
    assert decorators
```


# LLM-generated content at query #3
#--------------------------

```
def test_predicate_true_when_decorators_present():
    import ast
    code = """
@some_decorator
def foo():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    decorators = getattr(func_node, "decorator_list", [])
    assert bool(decorators) == True
```


# LLM-generated content at query #4
#--------------------------

def test_get_first_line_number_with_decorator():
    class FakeDecorator:
        lineno = 42
    class FakeNode:
        decorator_list = [FakeDecorator()]
        lineno = 10
    node = FakeNode()
    result = get_first_line_number(node)
    assert result == 42


# LLM-generated content at query #5
#--------------------------

```
def test_get_first_line_number_returns_decorator_lineno_when_decorators_exist():
    import ast
    code = """
@some_decorator
def foo():
    pass
"""
    tree = ast.parse(code.strip())
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == node.decorator_list[0].lineno
    assert result != node.lineno
```


# LLM-generated content at query #6
#--------------------------

```
def test_has_decorators_returns_first_decorator_lineno():
    node = type('MockNode', (), {'decorator_list': [type('MockDecorator', (), {'lineno': 42})()], 'lineno': 10})()
    result = get_first_line_number(node)
    assert result == 42
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_first_line_number_with_decorator_list():
    import ast
    code = """
@decorator
def foo():
    pass
"""
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == 2

def test_get_first_line_number_without_decorator_list():
    import ast
    code = """
def foo():
    pass
"""
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == 2

def test_get_first_line_number_empty_decorator_list():
    import ast
    class MockNode:
        lineno = 10
        decorator_list = []
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 10


# LLM-generated content at query #2
#--------------------------

```
def test_get_first_line_number_with_decorators():
    class MockNode:
        decorator_list = [type('MockDecorator', (), {'lineno': 10})()]
        lineno = 15
    result = get_first_line_number(MockNode())
    assert result == 10

def test_get_first_line_number_without_decorators():
    class MockNode:
        decorator_list = []
        lineno = 20
    result = get_first_line_number(MockNode())
    assert result == 20

def test_get_first_line_number_empty_decorator_list():
    class MockNode:
        decorator_list = []
        lineno = 25
    result = get_first_line_number(MockNode())
    assert result == 25

def test_get_first_line_number_multiple_decorators():
    class MockNode:
        decorator_list = [type('MockDecorator', (), {'lineno': 5})(), type('MockDecorator', (), {'lineno': 6})()]
        lineno = 30
    result = get_first_line_number(MockNode())
    assert result == 5

def test_get_first_line_number_decorator_list_none():
    class MockNode:
        lineno = 35
    result = get_first_line_number(MockNode())
    assert result == 35
```


# LLM-generated content at query #3
#--------------------------

```
def test_get_first_line_number_with_decorator():
    class MockNode:
        decorator_list = [type('MockDecorator', (), {'lineno': 42})()]
        lineno = 10
    result = get_first_line_number(MockNode())
    assert result == 42
```


# LLM-generated content at query #4
#--------------------------

```
def test_get_first_line_number_with_decorators():
    class MockDecorator:
        def __init__(self):
            self.lineno = 42
    class MockNode:
        def __init__(self):
            self.decorator_list = [MockDecorator()]
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 42
```


# LLM-generated content at query #5
#--------------------------

```python
def test_get_first_line_number_with_decorator():
    import ast
    code = """
@decorator
def func():
    pass
"""
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == 1
```


# LLM-generated content at query #6
#--------------------------

```
def test_get_first_line_number_with_decorator():
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('', (), {'lineno': 5})()]
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_without_decorator():
    class MockNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_with_multiple_decorators():
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('', (), {'lineno': 3})(), type('', (), {'lineno': 7})()]
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 3

def test_get_first_line_number_node_without_decorator_list():
    class MockNode:
        def __init__(self):
            self.lineno = 15
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 15

def test_get_first_line_number_decorator_list_is_none():
    class MockNode:
        def __init__(self):
            self.decorator_list = None
            self.lineno = 20
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 20

def test_get_first_line_number_with_lineno_only():
    class MockNode:
        def __init__(self):
            self.lineno = 25
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 25

def test_get_first_line_number_with_decorator_and_lineno():
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('', (), {'lineno': 30})()]
            self.lineno = 35
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 30

def test_get_first_line_number_empty_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 40
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 40

def test_get_first_line_number_decorator_with_negative_lineno():
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('', (), {'lineno': -5})()]
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == -5

def test_get_first_line_number_decorator_with_zero_lineno():
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('', (), {'lineno': 0})()]
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 0

def test_get_first_line_number_decorator_with_large_lineno():
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('', (), {'lineno': 1000000})()]
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 1000000
```


# LLM-generated content at query #7
#--------------------------

```python
def test_get_first_line_number_returns_first_decorator_lineno():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 5})()], 'lineno': 10})()
    assert get_first_line_number(node) == 5
```


