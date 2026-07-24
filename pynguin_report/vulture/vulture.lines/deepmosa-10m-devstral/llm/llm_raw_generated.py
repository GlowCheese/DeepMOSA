####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    node = type('obj', (), {'decorator_list': [type('obj', (), {'lineno': 10})()]})()
    assert get_first_line_number(node) == 10

def test_get_first_line_number_without_decorators():
    node = type('obj', (), {'lineno': 5})()
    assert get_first_line_number(node) == 5


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

    node = MockNode(lineno=10, decorators=[MockDecorator(lineno=5), MockDecorator(lineno=7)])
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

def test_get_first_line_number_without_decorators():
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []

    node = MockNode(10)
    assert get_first_line_number(node) == 10


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

```python
def test_get_first_line_number_with_decorators():
    node = type('Node', (), {'decorator_list': [type('Decorator', (), {'lineno': 5})()]})
    assert get_first_line_number(node) == 5

def test_get_first_line_number_without_decorators():
    node = type('Node', (), {'lineno': 10})
    assert get_first_line_number(node) == 10


