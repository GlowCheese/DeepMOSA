####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorators=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorators=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #2
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #3
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_multiple_decorators = MockNodeWithMultipleDecorators(20, [3, 7, 11])
    assert get_first_line_number(node_multiple_decorators) == 3


# LLM-generated content at query #4
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #5
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first decorator's line)
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 5, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #6
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #7
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(20, [3, 7, 11])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #8
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #9
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(20, [3, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #10
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(5), MockDecorator(6)]
    node_with_decorators = MockNodeWithDecorators(10, decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(10, [])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #11
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #12
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #13
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 4, 5])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #14
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 4, 5])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #15
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [MockDecorator(5)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 10

    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecoratorList:
        def __init__(self):
            self.lineno = 10

    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 10

class MockDecorator:
    def __init__(self, lineno):
        self.lineno = lineno


# LLM-generated content at query #16
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorators=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #17
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test with a node that has decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        lineno = 20
        decorator_list = [MockDecorator(15), MockDecorator(16)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 15


# LLM-generated content at query #18
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockDecorator1:
        lineno = 3

    class MockDecorator2:
        lineno = 4

    class MockNodeWithMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #19
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        lineno = 20
        decorator_list = [MockDecorator(15), MockDecorator(16)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 15

    # Test case 3: Node with empty decorator list
    class MockNodeEmptyDecorators:
        lineno = 30
        decorator_list = []

    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 30


# LLM-generated content at query #20
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(15, [5, 6])
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(20, [])
    assert get_first_line_number(node_empty_decorators) == 20


# LLM-generated content at query #21
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        lineno = 10
        decorator_list = []

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with no decorator_list attribute
    class MockNodeNoDecoratorList:
        lineno = 15

    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 15


# LLM-generated content at query #22
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [type('MockDecorator', (), {'lineno': 5})()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 10

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 10

    # Test with a node that has an empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #23
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first decorator's lineno)
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #24
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test with a node that has decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #25
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        lineno = 20
        decorator_list = [MockDecorator(15), MockDecorator(16)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 15


# LLM-generated content at query #26
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first decorator's lineno)
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #27
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 4, 5])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #28
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(5), MockDecorator(6)]
    node_with_decorators = MockNodeWithDecorators(10, decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_with_empty_decorators = MockNodeWithDecorators(10, [])
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #29
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #30
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(15, [5, 7])
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #31
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    node_with_multiple_decorators = MockNodeWithDecorators(20, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(7))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #32
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #33
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 5, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #34
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #35
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 5, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #36
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [type('MockDecorator', (), {'lineno': 5})()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 2: Node without decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 3: Node without decorator_list attribute
    class MockNodeNoDecoratorList:
        def __init__(self):
            self.lineno = 10

    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 10


# LLM-generated content at query #37
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    node_with_multiple_decorators = MockNodeWithDecorators(10, 5)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(3))
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #38
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #39
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first decorator's lineno)
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #40
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 5, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #41
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 5
    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 3

    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3

    # Test case 3: Node with multiple decorators (should return first one)
    class MockDecorator1:
        lineno = 2

    class MockDecorator2:
        lineno = 1

    class MockNodeWithMultipleDecorators:
        lineno = 5
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 2


# LLM-generated content at query #42
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 4, 5])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #43
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        lineno = 5

    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(3), MockDecorator(4)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3


# LLM-generated content at query #44
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(15, [5, 7])
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #45
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node_without_decorators = MockNode(lineno=10)
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorators=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #46
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #47
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #48
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #49
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test with a node that has decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #50
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(5), MockDecorator(6)]
    node_with_decorators = MockNodeWithDecorators(10, decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(10, [])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #51
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #52
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('obj', (object,), {'lineno': 5})()]
            self.lineno = 10

    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 2: Node without decorators
    class UndecoratedNode:
        def __init__(self):
            self.lineno = 7

    node_without_decorators = UndecoratedNode()
    assert get_first_line_number(node_without_decorators) == 7

    # Test case 3: Node with empty decorator list
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 8

    node_with_empty_decorators = EmptyDecoratorNode()
    assert get_first_line_number(node_with_empty_decorators) == 8


# LLM-generated content at query #53
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #54
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #55
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #56
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node) == 5


# LLM-generated content at query #57
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #58
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #59
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(15, [5, 8])
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []

    node_empty_decorators = MockNodeWithEmptyDecorators(20)
    assert get_first_line_number(node_empty_decorators) == 20


# LLM-generated content at query #60
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(20, [3, 7, 11])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #61
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(5), MockDecorator(6)]
    node_with_decorators = MockNodeWithDecorators(10, decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #62
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node_without_decorators = MockNode(lineno=10)
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorators=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #63
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #64
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
        end_lineno = 20

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        end_lineno = 20
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        end_lineno = 20
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #65
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #66
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node_without_decorators = MockNode(lineno=10)
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorators=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #67
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #68
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
        end_lineno = 20

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        end_lineno = 20
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeEmptyDecorators:
        lineno = 15
        end_lineno = 25
        decorator_list = []

    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 15


# LLM-generated content at query #69
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNode:
        lineno = 5
    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test node with decorators
    class MockDecorator:
        lineno = 3

    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3


# LLM-generated content at query #70
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #71
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #72
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test node with multiple decorators (should return first decorator's lineno)
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 5, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #73
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        lineno = 5
        decorator_list = []

    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(3), MockDecorator(4)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3


# LLM-generated content at query #74
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockDecorator1:
        lineno = 3

    class MockDecorator2:
        lineno = 4

    class MockNodeWithMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #75
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
        decorator_list = []

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #76
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node_without_decorators = MockNode(lineno=10)
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorators=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_with_empty_decorators = MockNodeWithDecorators(lineno=10, decorators=[])
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #77
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 4, 5])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #78
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #79
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_multiple_decorators = MockNodeWithMultipleDecorators(20, [3, 6, 9])
    assert get_first_line_number(node_multiple_decorators) == 3


# LLM-generated content at query #80
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #81
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #82
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    node_with_multiple_decorators = MockNodeWithDecorators(10, 5)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(3))
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #83
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #84
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #85
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #86
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node = MockNodeWithDecorators(10, [5, 6, 7])
    assert get_first_line_number(node) == 5

    # Test case 3: Node with empty decorator list
    node = MockNodeWithDecorators(10, [])
    assert get_first_line_number(node) == 10


# LLM-generated content at query #87
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(5), MockDecorator(6)]
    node_with_decorators = MockNodeWithDecorators(10, decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #88
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(20, [3, 4, 5])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #89
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 4, 5])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #90
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockDecorator1:
        lineno = 3

    class MockDecorator2:
        lineno = 4

    class MockNodeWithMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #91
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #92
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorators=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorators=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #93
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=7)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #94
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #95
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #96
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [MockDecorator(5), MockDecorator(7)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 2: Node without decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 15

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 15

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 20

class MockDecorator:
    def __init__(self, lineno):
        self.lineno = lineno


# LLM-generated content at query #97
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorators=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorators=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #98
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #2
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first decorator's lineno)
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #3
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self):
            self.lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 15
            self.decorator_list = [MockDecorator(5), MockDecorator(7)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 20


# LLM-generated content at query #4
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_multiple_decorators = MockNodeWithMultipleDecorators(20, [3, 7, 11])
    assert get_first_line_number(node_multiple_decorators) == 3


# LLM-generated content at query #5
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_multiple_decorators = MockNodeWithMultipleDecorators(20, [3, 7, 11])
    assert get_first_line_number(node_multiple_decorators) == 3


# LLM-generated content at query #6
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockDecorator1:
        lineno = 3

    class MockDecorator2:
        lineno = 4

    class MockNodeWithMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #7
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(10, [])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #8
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [type('MockDecorator', (), {'lineno': 5})()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 2: Node without decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 3: Node without decorator_list attribute
    class MockNodeNoDecoratorList:
        def __init__(self):
            self.lineno = 10

    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 10


# LLM-generated content at query #9
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #10
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockDecorator1:
        lineno = 3

    class MockDecorator2:
        lineno = 4

    class MockNodeWithMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #11
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 15
            self.decorator_list = [MockDecorator(5), MockDecorator(7)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 20


# LLM-generated content at query #12
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNodeWithoutDecorators:
        lineno = 5

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 5

    # Test node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(3), MockDecorator(4)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3


# LLM-generated content at query #13
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(5), MockDecorator(6)]
    node_with_decorators = MockNodeWithDecorators(10, decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(10, [])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #14
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNode:
        lineno = 5
    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test node with decorators
    class MockDecorator:
        lineno = 3

    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3


# LLM-generated content at query #15
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        lineno = 20
        decorator_list = [MockDecorator(15), MockDecorator(16)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 15

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 30
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 30


# LLM-generated content at query #16
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [type('MockDecorator', (), {'lineno': 5})()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 2: Node without decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 10

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #17
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #18
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #19
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(15, [5, 7])
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators(20)
    assert get_first_line_number(node_with_empty_decorators) == 20


# LLM-generated content at query #20
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first decorator's lineno)
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #21
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #22
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(15, [5, 7])
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(20, [])
    assert get_first_line_number(node_empty_decorators) == 20


# LLM-generated content at query #23
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 5
    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 3

    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3

    # Test case 3: Node with multiple decorators (should return first one)
    class MockDecorator1:
        lineno = 2

    class MockDecorator2:
        lineno = 1

    class MockNodeWithMultipleDecorators:
        lineno = 5
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #25
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
        decorator_list = []

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #26
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockDecorator1:
        lineno = 3

    class MockDecorator2:
        lineno = 4

    class MockNodeWithMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #27
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    node_with_multiple_decorators = MockNodeWithDecorators(10, 5)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(3))
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #28
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #29
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    class MockDecorator1:
        lineno = 3

    class MockDecorator2:
        lineno = 4

    class MockNodeWithMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #30
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #31
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(5), MockDecorator(6)]
    node_with_decorators = MockNodeWithDecorators(10, decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(10, [])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #32
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #33
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #34
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 4, 5])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #35
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d_lineno) for d_lineno in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(15, [5, 7])
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #36
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node_without_decorators = MockNode(lineno=10)
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorators=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #37
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [type('MockDecorator', (), {'lineno': 5})()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 2: Node without decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 3: Node without decorator_list attribute
    class MockNodeNoDecoratorList:
        def __init__(self):
            self.lineno = 10

    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 10


# LLM-generated content at query #38
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 5
        decorator_list = []

    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 3

    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3

    # Test case 3: Node with empty decorator_list attribute
    class MockNodeEmptyDecorators:
        lineno = 7
        decorator_list = []

    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 7


# LLM-generated content at query #39
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #40
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #41
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 5
        decorator_list = []

    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 3

    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3

    # Test case 3: Node with empty decorator_list attribute
    class MockNodeEmptyDecorators:
        lineno = 7
        decorator_list = []

    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 7


# LLM-generated content at query #42
#--------------------------

```python
def test_get_first_line_number():
    # Test for node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test for node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #43
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #44
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #45
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #46
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []

    node_without_decorators = MockNode(10)
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first decorator's lineno)
    class MockNodeMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_multiple_decorators = MockNodeMultipleDecorators(20, [3, 6, 9])
    assert get_first_line_number(node_multiple_decorators) == 3


# LLM-generated content at query #47
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first decorator's lineno)
    node_multiple_decorators = MockNodeWithDecorators(10, 5)
    node_multiple_decorators.decorator_list.append(MockDecorator(3))
    assert get_first_line_number(node_multiple_decorators) == 5


# LLM-generated content at query #48
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 4, 5])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #49
#--------------------------

```python
def test_get_first_line_number():
    # Test for node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test for node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test for node with multiple decorators (should return first one)
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #50
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 5, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #51
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(5), MockDecorator(7)]
    node_with_decorators = MockNodeWithDecorators(10, decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(10, [])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #52
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator(), MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #53
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    class MockDecorator1:
        lineno = 3

    class MockDecorator2:
        lineno = 4

    class MockNodeWithMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #54
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(15, [5, 7])
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeEmptyDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []

    node_empty_decorators = MockNodeEmptyDecorators(20)
    assert get_first_line_number(node_empty_decorators) == 20


# LLM-generated content at query #55
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(15, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(20, [3, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #56
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(5), MockDecorator(6)]
    node_with_decorators = MockNodeWithDecorators(10, decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_with_empty_decorators = MockNodeWithDecorators(10, [])
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #57
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #58
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class MockNode:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [type('obj', (object,), {'lineno': 5})()]

    node_with_decorators = MockNode()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 2: Node without decorators
    class MockNodeNoDecorators:
        def __init__(self):
            self.lineno = 10

    node_without_decorators = MockNodeNoDecorators()
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 3: Node with empty decorator list
    class MockNodeEmptyDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []

    node_with_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #59
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #60
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 5, 7])
    assert get_first_line_number(node_multiple_decorators) == 3


# LLM-generated content at query #61
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #62
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockDecorator1:
        lineno = 3

    class MockDecorator2:
        lineno = 4

    class MockNodeWithMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #63
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first decorator's lineno)
    node_with_multiple_decorators = MockNodeWithDecorators(10, [MockDecorator(5), MockDecorator(7)])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #64
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #65
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    class MockDecorator1:
        lineno = 3

    class MockDecorator2:
        lineno = 4

    class MockNodeWithMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #66
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #67
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10
    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeEmptyDecorators:
        lineno = 10
        decorator_list = []

    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #68
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorators):
            self.lineno = lineno
            self.decorator_list = decorators

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorators=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #69
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [type('MockDecorator', (), {'lineno': 5})()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 2: Node without decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 10

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #70
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self):
            self.lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = [MockDecorator(15), MockDecorator(16)]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 15

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self):
            self.lineno = 30
            self.decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node_with_empty_decorators) == 30


# LLM-generated content at query #71
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #72
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 5
    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 3

    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3

    # Test case 3: Node with multiple decorators (should return first one)
    class MockDecorator1:
        lineno = 2

    class MockDecorator2:
        lineno = 3

    class MockNodeWithMultipleDecorators:
        lineno = 5
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 2


# LLM-generated content at query #73
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #74
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    node_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_multiple_decorators) == 3


# LLM-generated content at query #75
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #76
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=7)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #77
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node_without_decorators = MockNode(lineno=10)
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #78
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #79
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(10, [])
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #80
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node_without_decorators = MockNode(lineno=10)
    assert get_first_line_number(node_without_decorators) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorator1 = MockDecorator(lineno=5)
    decorator2 = MockDecorator(lineno=6)
    node_with_decorators = MockNodeWithDecorators(
        lineno=10,
        decorator_list=[decorator1, decorator2]
    )
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_with_empty_decorators = MockNodeWithDecorators(
        lineno=10,
        decorator_list=[]
    )
    assert get_first_line_number(node_with_empty_decorators) == 10


# LLM-generated content at query #81
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    node_with_multiple_decorators = MockNodeWithDecorators(10, 5)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(3))
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #82
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    node_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_multiple_decorators) == 3


# LLM-generated content at query #83
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators (should return first one)
    node_with_multiple_decorators = MockNodeWithDecorators(10, 5)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(3))
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #84
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(d) for d in decorator_linenos]

    node_with_decorators = MockNodeWithDecorators(15, [5, 7])
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []

    node_with_empty_decorators = MockNodeWithEmptyDecorators(20)
    assert get_first_line_number(node_with_empty_decorators) == 20


# LLM-generated content at query #85
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        lineno = 10

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5


# LLM-generated content at query #86
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 5, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #87
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    node_with_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_with_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #88
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [3, 5, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 3


# LLM-generated content at query #89
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #90
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(ln) for ln in decorator_linenos]

    node_with_multiple_decorators = MockNodeWithMultipleDecorators(10, [5, 6, 7])
    assert get_first_line_number(node_with_multiple_decorators) == 5


# LLM-generated content at query #91
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_lineno):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(decorator_lineno)]

    node_with_decorators = MockNodeWithDecorators(10, 5)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with multiple decorators
    node_multiple_decorators = MockNodeWithDecorators(10, 3)
    node_multiple_decorators.decorator_list.append(MockDecorator(4))
    assert get_first_line_number(node_multiple_decorators) == 3


# LLM-generated content at query #92
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class MockNode:
        def __init__(self, lineno):
            self.lineno = lineno

    node = MockNode(lineno=10)
    assert get_first_line_number(node) == 10

    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list

    decorators = [MockDecorator(lineno=5), MockDecorator(lineno=6)]
    node_with_decorators = MockNodeWithDecorators(lineno=10, decorator_list=decorators)
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MockNodeWithDecorators(lineno=10, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 10


