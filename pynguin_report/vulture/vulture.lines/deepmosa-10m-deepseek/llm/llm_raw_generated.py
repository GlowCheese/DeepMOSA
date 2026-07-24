####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_first_line_number_without_decorators():
    class MockNode:
        lineno = 42
        decorator_list = []
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 42

def test_get_first_line_number_with_decorators():
    class MockDecorator:
        lineno = 10
    class MockNode:
        lineno = 20
        decorator_list = [MockDecorator()]
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_with_decorator_list_none():
    class MockNode:
        lineno = 5
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 5


# LLM-generated content at query #2
#--------------------------

def test_get_first_line_number_with_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('', (), {'lineno': 5})()]
            self.lineno = 10
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_without_decorator_list():
    class MockNode:
        def __init__(self):
            self.lineno = 7
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 7

def test_get_first_line_number_empty_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 3
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 3


# LLM-generated content at query #3
#--------------------------

def test_get_first_line_number_with_decorators():
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

def test_get_first_line_number_without_decorators():
    class MockNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 15
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 15

def test_get_first_line_number_with_empty_decorator_list():
    class MockNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 20

def test_get_first_line_number_with_no_decorator_attribute():
    class MockNode:
        def __init__(self):
            self.lineno = 25
    node = MockNode()
    result = get_first_line_number(node)
    assert result == 25


# LLM-generated content at query #4
#--------------------------

```python
def test_get_first_line_number_with_decorators():
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
```


# LLM-generated content at query #5
#--------------------------

```python

def test_decorator_list_not_empty_returns_first_decorator_lineno():
    import ast
    code = "@decorator1\n@decorator2\ndef func(): pass"
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == 1

```


# LLM-generated content at query #6
#--------------------------

```
def test_get_first_line_number_with_decorator():
    class FakeNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})()]
        lineno = 15
    result = get_first_line_number(FakeNode())
    assert result == 10
```


# LLM-generated content at query #7
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    import ast
    code = "@decorator1\n@decorator2\ndef foo():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == 1
```


# LLM-generated content at query #8
#--------------------------

```
def test_decorator_list_not_empty_returns_first_decorator_lineno():
    import ast
    code = """
@some_decorator
def foo():
    pass
"""
    tree = ast.parse(code)
    node = tree.body[0]
    result = get_first_line_number(node)
    assert result == node.decorator_list[0].lineno
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_get_first_line_number_with_decorator_list():
    node = type('Node', (object,), {'decorator_list': [type('Decorator', (object,), {'lineno': 5})()], 'lineno': 10})()
    result = get_first_line_number(node)
    assert result == 5

def test_get_first_line_number_without_decorator_list():
    node = type('Node', (object,), {'lineno': 10})()
    result = get_first_line_number(node)
    assert result == 10

def test_get_first_line_number_empty_decorator_list():
    node = type('Node', (object,), {'decorator_list': [], 'lineno': 15})()
    result = get_first_line_number(node)
    assert result == 15

def test_get_first_line_number_multiple_decorators():
    node = type('Node', (object,), {'decorator_list': [type('Decorator', (object,), {'lineno': 3})(), type('Decorator', (object,), {'lineno': 7})()], 'lineno': 12})()
    result = get_first_line_number(node)
    assert result == 3
```


# LLM-generated content at query #2
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    node = type("MockNode", (), {"decorator_list": [type("MockDecorator", (), {"lineno": 5})()], "lineno": 10})()
    result = get_first_line_number(node)
    assert result == 5
```


