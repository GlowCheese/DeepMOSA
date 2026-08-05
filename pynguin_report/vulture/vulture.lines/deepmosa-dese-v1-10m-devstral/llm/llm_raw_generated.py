####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 5})()]})
    assert get_first_line_number(node) == 5

def test_get_first_line_number_without_decorators():
    node = type('Node', (), {'lineno': 10})
    assert get_first_line_number(node) == 10

def test_get_first_line_number_with_empty_decorator_list():
    node = type('Node', (), {'decorator_list': [], 'lineno': 15})
    assert get_first_line_number(node) == 15


# LLM-generated content at query #2
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    class MockNode:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10, decorator_list=[MockDecorator(lineno=5)])
    assert get_first_line_number(node) == 5

def test_get_first_line_number_without_decorators():
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    class MockNode:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10, decorators=[MockDecorator(lineno=5), MockDecorator(lineno=6)])
    assert get_first_line_number(node) == 5

def test_get_first_line_number_without_decorators():
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10


# LLM-generated content at query #2
#--------------------------

```python
def test_decorators_predicate():
    class MockNode:
        def __init__(self, decorator_list, lineno):
            self.decorator_list = decorator_list
            self.lineno = lineno

    mock_decorator = type('MockDecorator', (), {'lineno': 10})()
    node = MockNode([mock_decorator], 20)
    assert node.decorator_list


# LLM-generated content at query #3
#--------------------------

```python
def test_decorators_predicate_evaluates_to_true():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 1})()]})
    assert getattr(node, "decorator_list", [])


# LLM-generated content at query #4
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    class MockNode:
        def __init__(self, decorator_list, lineno):
            self.decorator_list = decorator_list
            self.lineno = lineno

    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode([MockDecorator(10)], 20)
    assert get_first_line_number(node) == 10


# LLM-generated content at query #5
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 5})()], 'lineno': 10})
    assert get_first_line_number(node()) == 5

def test_get_first_line_number_without_decorators():
    node = type('Node', (), {'lineno': 10})
    assert get_first_line_number(node()) == 10


# LLM-generated content at query #6
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('MockDecorator', (), {'lineno': 5})()]

    node = MockNode()
    assert get_first_line_number(node) == 5


# LLM-generated content at query #7
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    class MockNode:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10, decorator_list=[MockDecorator(lineno=5)])
    assert get_first_line_number(node) == 5


# LLM-generated content at query #8
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    class MockNode:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    decorators = [MockDecorator(5), MockDecorator(6)]
    node = MockNode(10, decorators)
    assert get_first_line_number(node) == 5


