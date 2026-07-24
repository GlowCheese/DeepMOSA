####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class MockNodeNoDecorators:
        lineno = 5
    
    result = get_first_line_number(MockNodeNoDecorators())
    assert result == 5, "Should return node.lineno when no decorators"

    # Test with decorators
    class MockDecorator:
        lineno = 3
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    result = get_first_line_number(MockNodeWithDecorators())
    assert result == 3, "Should return first decorator's lineno"

    # Test with multiple decorators
    class MockDecorator1:
        lineno = 2
    
    class MockDecorator2:
        lineno = 4
    
    class MockNodeMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 10
    
    result = get_first_line_number(MockNodeMultipleDecorators())
    assert result == 2, "Should return first decorator's lineno from multiple decorators"

    # Test with empty decorator_list
    class MockNodeEmptyDecorators:
        decorator_list = []
        lineno = 7
    
    result = get_first_line_number(MockNodeEmptyDecorators())
    assert result == 7, "Should return node.lineno when decorator_list is empty"

    # Test with node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 8
    
    result = get_first_line_number(MockNodeNoDecoratorAttr())
    assert result == 8, "Should return node.lineno when decorator_list attribute doesn't exist"
```


# LLM-generated content at query #2
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class MockDecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
        end_lineno = 15
    
    node = MockDecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a decorated class
    class MockDecoratedClass:
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
        lineno = 8
    
    node2 = MockDecoratedClass()
    assert get_first_line_number(node2) == 3
    
    # Test with a non-decorated function
    class MockNode:
        decorator_list = []
        lineno = 20
    
    node3 = MockNode()
    assert get_first_line_number(node3) == 20
    
    # Test with a node that has no decorator_list attribute
    class MockNodeNoDec:
        lineno = 25
    
    node4 = MockNodeNoDec()
    assert get_first_line_number(node4) == 25
    
    # Test with multiple decorators
    class MockMultiDecorated:
        decorator_list = [
            type('Decorator', (), {'lineno': 1})(),
            type('Decorator', (), {'lineno': 2})()
        ]
        lineno = 10
    
    node5 = MockMultiDecorated()
    assert get_first_line_number(node5) == 1
```


# LLM-generated content at query #3
#--------------------------

```python
def test_get_first_line_number():
    # Create a simple node with no decorators
    class SimpleNode:
        lineno = 10
    node = SimpleNode()
    assert get_first_line_number(node) == 10

    # Create a node with decorators
    class DecoratorNode:
        lineno = 15
        decorator_list = []
    
    class Decorator:
        lineno = 12
    
    node2 = DecoratorNode()
    node2.decorator_list = [Decorator()]
    assert get_first_line_number(node2) == 12

    # Test with multiple decorators (should return first decorator's lineno)
    class DecoratorNodeMultiple:
        lineno = 20
        decorator_list = []
    
    node3 = DecoratorNodeMultiple()
    decorator1 = type('Decorator1', (), {'lineno': 16})()
    decorator2 = type('Decorator2', (), {'lineno': 18})()
    node3.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node3) == 16

    # Test without decorator_list attribute (should fallback to lineno)
    class NoDecoratorList:
        lineno = 5
    node4 = NoDecoratorList()
    assert get_first_line_number(node4) == 5
```


# LLM-generated content at query #4
#--------------------------

```python
def test_get_first_line_number():
    # Test with a function that has decorators
    import ast
    code_with_decorators = """
@some_decorator
@another_decorator
def my_function():
    pass
"""
    tree = ast.parse(code_with_decorators)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # lineno of first decorator

    # Test with a function without decorators
    code_without_decorators = """
def my_function():
    pass
"""
    tree = ast.parse(code_without_decorators)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # lineno of function definition

    # Test with a class that has decorators
    code_class_with_decorators = """
@some_decorator
class MyClass:
    pass
"""
    tree = ast.parse(code_class_with_decorators)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2  # lineno of first decorator

    # Test with a class without decorators
    code_class_without_decorators = """
class MyClass:
    pass
"""
    tree = ast.parse(code_class_without_decorators)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2  # lineno of class definition

    # Test with a node that has no decorator_list attribute (e.g., a simple expression)
    code_simple = """
x = 1
"""
    tree = ast.parse(code_simple)
    assign_node = tree.body[0]
    assert get_first_line_number(assign_node) == 2  # lineno of assignment
```


# LLM-generated content at query #5
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = []
    
    decorator1 = type('Decorator', (), {'lineno': 7})()
    decorator2 = type('Decorator', (), {'lineno': 8})()
    node_with_decorators = DecoratorNode()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    
    assert get_first_line_number(node_with_decorators) == 7
    
    # Test with node that doesn't have decorator_list attribute
    class NodeWithoutDecoratorList:
        lineno = 15
    
    node_no_decorator_list = NodeWithoutDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 15
```


# LLM-generated content at query #6
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node_no_decorator = SimpleNode()
    assert get_first_line_number(node_no_decorator) == 5
    
    # Test node with decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    node_with_decorator = DecoratorNode()
    assert get_first_line_number(node_with_decorator) == 3
    
    # Test node with multiple decorators
    class MultiDecoratorNode:
        lineno = 20
        decorator_list = [
            type('Decorator', (), {'lineno': 1})(),
            type('Decorator', (), {'lineno': 2})()
        ]
    
    node_multi_decorator = MultiDecoratorNode()
    assert get_first_line_number(node_multi_decorator) == 1
    
    # Test node without decorator_list attribute
    class NoDecoratorListNode:
        lineno = 15
    
    node_no_decorator_list = NoDecoratorListNode()
    assert get_first_line_number(node_no_decorator_list) == 15
    
    # Test node with empty decorator_list
    class EmptyDecoratorNode:
        lineno = 25
        decorator_list = []
    
    node_empty_decorator = EmptyDecoratorNode()
    assert get_first_line_number(node_empty_decorator) == 25
```


# LLM-generated content at query #7
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test node with decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3

    # Test node with multiple decorators
    class MultiDecoratedNode:
        lineno = 20
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1

    # Test node without decorator_list attribute
    class NodeWithoutDecorators:
        lineno = 15
    node_without_decorators = NodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 15
```


# LLM-generated content at query #8
#--------------------------

```python
def test_get_first_line_number():
    import ast
    
    # Test node without decorators
    code_without_decorators = """
def simple_function():
    pass
"""
    tree = ast.parse(code_without_decorators)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2
    
    # Test node with decorators
    code_with_decorators = """
@property
def decorated_method(self):
    pass
"""
    tree = ast.parse(code_with_decorators)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2
    
    # Test node with multiple decorators
    code_multiple_decorators = """
@classmethod
@property
def multi_decorated_method(self):
    pass
"""
    tree = ast.parse(code_multiple_decorators)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2
    
    # Test class without decorators
    code_class = """
class MyClass:
    def method(self):
        pass
"""
    tree = ast.parse(code_class)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2
    
    # Test class with decorator
    code_class_decorator = """
@dataclass
class MyDataClass:
    def method(self):
        pass
"""
    tree = ast.parse(code_class_decorator)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2
    
    # Test async function without decorators
    code_async = """
async def async_function():
    pass
"""
    tree = ast.parse(code_async)
    async_func_node = tree.body[0]
    assert get_first_line_number(async_func_node) == 2
    
    # Test async function with decorators
    code_async_decorator = """
@some_decorator
async def decorated_async_function():
    pass
"""
    tree = ast.parse(code_async_decorator)
    async_func_node = tree.body[0]
    assert get_first_line_number(async_func_node) == 2
    
    # Test node with decorator_list attribute but empty list
    code_empty_decorator = """
def function_with_empty_decorator_list():
    pass
"""
    tree = ast.parse(code_empty_decorator)
    func_node = tree.body[0]
    # Manually set empty decorator_list to ensure path is tested
    func_node.decorator_list = []
    assert get_first_line_number(func_node) == func_node.lineno
```


# LLM-generated content at query #9
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test node with decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    node = DecoratedNode()
    assert get_first_line_number(node) == 3

    # Test node with multiple decorators
    class MultiDecoratedNode:
        lineno = 15
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 1

    # Test node with empty decorator_list
    class EmptyDecoratorNode:
        lineno = 20
        decorator_list = []
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 20

    # Test node without decorator_list attribute
    class NoDecoratorListNode:
        lineno = 25
    node = NoDecoratorListNode()
    assert get_first_line_number(node) == 25
```


# LLM-generated content at query #10
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorator
    node_no_decorator = type('Node', (), {'lineno': 10, 'decorator_list': []})()
    assert get_first_line_number(node_no_decorator) == 10

    # Test with decorator
    decorator = type('Decorator', (), {'lineno': 5})()
    node_with_decorator = type('Node', (), {'lineno': 10, 'decorator_list': [decorator]})()
    assert get_first_line_number(node_with_decorator) == 5

    # Test with multiple decorators - should return first decorator's lineno
    decorator1 = type('Decorator', (), {'lineno': 3})()
    decorator2 = type('Decorator', (), {'lineno': 4})()
    node_multiple_decorators = type('Node', (), {'lineno': 10, 'decorator_list': [decorator1, decorator2]})()
    assert get_first_line_number(node_multiple_decorators) == 3

    # Test with node that has no decorator_list attribute (older Python versions)
    node_no_decorator_list = type('Node', (), {'lineno': 15})()
    assert get_first_line_number(node_no_decorator_list) == 15

    # Test edge case: node with decorator_list = None
    node_none_decorator = type('Node', (), {'lineno': 20, 'decorator_list': None})()
    assert get_first_line_number(node_none_decorator) == 20
```


# LLM-generated content at query #11
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test node with decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 3

    # Test node without decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 7
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 7

    # Test node with empty decorator_list
    class EmptyDecoratorListNode:
        lineno = 12
        decorator_list = []
    
    node = EmptyDecoratorListNode()
    assert get_first_line_number(node) == 12

    # Test node with multiple decorators (should return first one's lineno)
    class MultipleDecoratorsNode:
        lineno = 20
        decorator_list = [
            type('Decorator', (), {'lineno': 1})(),
            type('Decorator', (), {'lineno': 2})()
        ]
    
    node = MultipleDecoratorsNode()
    assert get_first_line_number(node) == 1
```


# LLM-generated content at query #12
#--------------------------

```python
def test_get_first_line_number():
    # Test when node has no decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test when node has decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = []
    
    decorator1 = type('Decorator', (), {'lineno': 3})()
    decorator2 = type('Decorator', (), {'lineno': 4})()
    decorated_node = DecoratorNode()
    decorated_node.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(decorated_node) == 3

    # Test when node has empty decorator_list
    class EmptyDecoratorNode:
        lineno = 7
        decorator_list = []
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 7

    # Test when node doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 12
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 12
```


# LLM-generated content at query #13
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test node with decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    node = DecoratedNode()
    assert get_first_line_number(node) == 3

    # Test node with multiple decorators
    class MultiDecoratedNode:
        lineno = 15
        decorator_list = [
            type('Decorator1', (), {'lineno': 7})(),
            type('Decorator2', (), {'lineno': 8})()
        ]
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 7

    # Test node with empty decorator list
    class EmptyDecoratorNode:
        lineno = 20
        decorator_list = []
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 20
```


# LLM-generated content at query #14
#--------------------------

```python
def test_get_first_line_number():
    # Test node with decorators
    class MockDecoratedNode:
        decorator_list = [type('Dec', (), {'lineno': 5})()]
        lineno = 10
    
    node = MockDecoratedNode()
    assert get_first_line_number(node) == 5  # Returns first decorator lineno
    
    # Test node without decorators
    class MockNode:
        decorator_list = []
        lineno = 20
    
    node2 = MockNode()
    assert get_first_line_number(node2) == 20  # Returns node.lineno
    
    # Test node without decorator_list attribute
    class MockNodeNoDecorators:
        lineno = 30
    
    node3 = MockNodeNoDecorators()
    assert get_first_line_number(node3) == 30  # Returns node.lineno


# LLM-generated content at query #15
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test node with decorators
    class DecoratorNode:
        lineno = 10
        
        class Decorator:
            lineno = 8
        
        decorator_list = [Decorator()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 8
    
    # Test node without decorator_list attribute
    class NoDecoratorListNode:
        lineno = 15
    
    node = NoDecoratorListNode()
    assert get_first_line_number(node) == 15
    
    # Test node with multiple decorators
    class MultiDecoratorNode:
        lineno = 25
        
        class Decorator1:
            lineno = 20
        
        class Decorator2:
            lineno = 22
        
        decorator_list = [Decorator1(), Decorator2()]
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 20
```


# LLM-generated content at query #16
#--------------------------

```python
def test_get_first_line_number():
    # Test node with no decorator
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test node with decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})(),
                          type('Decorator', (), {'lineno': 4})()]
    
    decorated = DecoratedNode()
    assert get_first_line_number(decorated) == 3
    
    # Test node with no decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 7
    
    no_attr = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr) == 7
    
    # Test node with empty decorator_list
    class EmptyDecoratorNode:
        lineno = 12
        decorator_list = []
    
    empty = EmptyDecoratorNode()
    assert get_first_line_number(empty) == 12
    
    # Test node with single decorator
    class SingleDecoratorNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    
    single = SingleDecoratorNode()
    assert get_first_line_number(single) == 15
```


# LLM-generated content at query #17
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self):
            self.decorator_list = [MockDecorator(5)]
            self.lineno = 10
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 15
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 15
    
    # Test with a node that has multiple decorators
    class MockNodeMultipleDecorators:
        def __init__(self):
            self.decorator_list = [MockDecorator(3), MockDecorator(8)]
            self.lineno = 20
    
    node_multiple = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple) == 3
    
    # Test with a node that has empty decorator_list
    class MockNodeEmptyDecorators:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 25
    
    node_empty = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty) == 25
    
    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecorators:
        def __init__(self):
            self.lineno = 30
    
    node_no_decorators = MockNodeNoDecorators()
    assert get_first_line_number(node_no_decorators) == 30
```


# LLM-generated content at query #18
#--------------------------

```python
def test_get_first_line_number():
    # Test with a function that has no decorators
    class MockNodeWithoutDecorators:
        lineno = 5
    assert get_first_line_number(MockNodeWithoutDecorators()) == 5
    
    # Test with a function that has decorators
    class MockDecorator:
        lineno = 3
    
    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator()]
    assert get_first_line_number(MockNodeWithDecorators()) == 3
    
    # Test with a class that has no decorators
    class MockClassWithoutDecorators:
        lineno = 10
    assert get_first_line_number(MockClassWithoutDecorators()) == 10
    
    # Test with a class that has decorators
    class MockDecorator2:
        lineno = 7
    
    class MockClassWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator2()]
    assert get_first_line_number(MockClassWithDecorators()) == 7
    
    # Test with multiple decorators (should return first decorator's lineno)
    class MockDecoratorFirst:
        lineno = 1
    
    class MockDecoratorSecond:
        lineno = 2
    
    class MockNodeWithMultipleDecorators:
        lineno = 5
        decorator_list = [MockDecoratorFirst(), MockDecoratorSecond()]
    assert get_first_line_number(MockNodeWithMultipleDecorators()) == 1
    
    # Test with empty decorator_list
    class MockNodeWithEmptyDecoratorList:
        lineno = 15
        decorator_list = []
    assert get_first_line_number(MockNodeWithEmptyDecoratorList()) == 15
    
    # Test with node that has no decorator_list attribute
    class MockNodeWithoutDecoratorAttr:
        lineno = 20
    assert get_first_line_number(MockNodeWithoutDecoratorAttr()) == 20


# LLM-generated content at query #19
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test node with decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 3
    
    # Test node with multiple decorators (should return first decorator's lineno)
    class MultiDecoratorNode:
        lineno = 15
        decorator_list = [
            type('Decorator1', (), {'lineno': 7})(),
            type('Decorator2', (), {'lineno': 8})()
        ]
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 7
```


# LLM-generated content at query #20
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class MockNodeNoDecorators:
        lineno = 5
        decorator_list = []
    
    node_no_decorators = MockNodeNoDecorators()
    assert get_first_line_number(node_no_decorators) == 5

    # Test with decorators
    class MockDecorator:
        lineno = 2
    
    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator()]
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 2

    # Test with node that has no decorator_list attribute
    class MockNodeNoDecoratorList:
        lineno = 10
    
    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 10

    # Test with multiple decorators
    class MockDecorator1:
        lineno = 1
    
    class MockDecorator2:
        lineno = 3
    
    class MockNodeMultipleDecorators:
        lineno = 5
        decorator_list = [MockDecorator1(), MockDecorator2()]
    
    node_multiple_decorators = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 1

    # Test with empty decorator list
    class MockNodeEmptyDecorators:
        lineno = 8
        decorator_list = []
    
    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 8
```


# LLM-generated content at query #21
#--------------------------

```python
def test_get_first_line_number():
    # Mock a node with decorators
    class MockDecorator:
        lineno = 1
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 5
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 1
    
    # Mock a node without decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 10
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 10
    
    # Mock a node without decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 15
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 15
```


# LLM-generated content at query #22
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test node with decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = []
    node = DecoratedNode()
    decorator1 = type('Decorator', (), {'lineno': 3})()
    decorator2 = type('Decorator', (), {'lineno': 4})()
    node.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node) == 3

    # Test node with empty decorator list
    class EmptyDecoratorNode:
        lineno = 15
        decorator_list = []
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 15

    # Test node without decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 20
```


# LLM-generated content at query #23
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.decorator_list = [MockDecorator(5), MockDecorator(6)]
            self.lineno = 10
    
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoAttribute:
        def __init__(self):
            self.lineno = 30
    
    node_no_attribute = MockNodeNoAttribute()
    assert get_first_line_number(node_no_attribute) == 30
    
    # Test with edge case - empty decorator list
    class MockNodeEmptyDecorators:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 40
    
    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 40
```


# LLM-generated content at query #24
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [
            type('Decorator', (), {'lineno': 2})(),
            type('Decorator', (), {'lineno': 3})()
        ]
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 2
    
    # Test with a node that has no decorators
    class MockNodeNoDecorators:
        lineno = 10
    
    node_no_decorators = MockNodeNoDecorators()
    assert get_first_line_number(node_no_decorators) == 10
    
    # Test with a node that has empty decorator list
    class MockNodeEmptyDecorators:
        lineno = 15
        decorator_list = []
    
    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 15
    
    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 20
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 20
```


# LLM-generated content at query #25
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node_no_decorator = SimpleNode()
    assert get_first_line_number(node_no_decorator) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = []
    
    decorator1 = type('Decorator', (), {'lineno': 3})()
    decorator2 = type('Decorator', (), {'lineno': 4})()
    node_with_decorators = DecoratorNode()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    
    assert get_first_line_number(node_with_decorators) == 3
    
    # Test with a node that has no decorator_list attribute
    class NodeWithoutDecoratorList:
        lineno = 15
    
    node_no_decorator_list = NodeWithoutDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 15
    
    # Test with an empty decorator list
    class NodeEmptyDecoratorList:
        lineno = 20
        decorator_list = []
    
    node_empty_decorators = NodeEmptyDecoratorList()
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #26
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.decorator_list = [
                type('MockDecorator', (), {'lineno': 5})(),
                type('MockDecorator', (), {'lineno': 6})()
            ]
            self.lineno = 10

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 15

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 15

    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecoratorList:
        def __init__(self):
            self.lineno = 20

    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 20
```


# LLM-generated content at query #27
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        lineno = 5
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    result = get_first_line_number(MockNodeWithDecorators())
    assert result == 5, f"Expected 5, got {result}"
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        lineno = 15
    
    result = get_first_line_number(MockNodeWithoutDecorators())
    assert result == 15, f"Expected 15, got {result}"
    
    # Test with a node that has an empty decorator list
    class MockNodeEmptyDecorators:
        decorator_list = []
        lineno = 20
    
    result = get_first_line_number(MockNodeEmptyDecorators())
    assert result == 20, f"Expected 20, got {result}"
    
    # Test with a node that has multiple decorators (should return first decorator's lineno)
    class MockDecorator1:
        lineno = 1
    
    class MockDecorator2:
        lineno = 2
    
    class MockNodeMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 25
    
    result = get_first_line_number(MockNodeMultipleDecorators())
    assert result == 1, f"Expected 1, got {result}"
    
    # Test with a node that has no decorator_list attribute
    class MockNodeNoAttr:
        lineno = 30
    
    result = get_first_line_number(MockNodeNoAttr())
    assert result == 30, f"Expected 30, got {result}"


# LLM-generated content at query #28
#--------------------------

```python
def test_get_first_line_number():
    # Mock a node without decorators
    class MockNodeNoDecorators:
        lineno = 5
        decorator_list = []

    node_no_decorators = MockNodeNoDecorators()
    assert get_first_line_number(node_no_decorators) == 5

    # Mock a node with decorators
    class MockDecorator:
        lineno = 3

    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3

    # Mock a node without decorator_list attribute (older Python versions)
    class MockNodeNoDecoratorList:
        lineno = 7

    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 7
```


# LLM-generated content at query #29
#--------------------------

```python
def test_get_first_line_number():
    # Mock node with no decorators
    class MockNodeNoDecorators:
        lineno = 5
    
    assert get_first_line_number(MockNodeNoDecorators()) == 5

    # Mock node with decorators
    class MockDecorator:
        lineno = 3
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    assert get_first_line_number(MockNodeWithDecorators()) == 3

    # Mock node with decorator_list attribute set to empty list explicitly
    class MockNodeEmptyDecoratorList:
        decorator_list = []
        lineno = 8
    
    assert get_first_line_number(MockNodeEmptyDecoratorList()) == 8

    # Test node without decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 15
    
    assert get_first_line_number(MockNodeNoDecoratorAttr()) == 15
```


# LLM-generated content at query #30
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class NodeNoDecorator:
        lineno = 5
    node_no_decorator = NodeNoDecorator()
    assert get_first_line_number(node_no_decorator) == 5

    # Test with decorators
    class Decorator:
        lineno = 2
    
    class NodeWithDecorator:
        decorator_list = [Decorator()]
        lineno = 10
    
    node_with_decorator = NodeWithDecorator()
    assert get_first_line_number(node_with_decorator) == 2

    # Test with empty decorator list
    class NodeEmptyDecorator:
        decorator_list = []
        lineno = 15
    node_empty = NodeEmptyDecorator()
    assert get_first_line_number(node_empty) == 15

    # Test with multiple decorators
    class Decorator1:
        lineno = 1
    class Decorator2:
        lineno = 3
    
    class NodeMultipleDecorators:
        decorator_list = [Decorator1(), Decorator2()]
        lineno = 20
    
    node_multiple = NodeMultipleDecorators()
    assert get_first_line_number(node_multiple) == 1
```


# LLM-generated content at query #31
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        lineno = 5
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    node_with_decorators = MockNodeWithDecorators()
    result = get_first_line_number(node_with_decorators)
    assert result == 5, f"Expected 5, got {result}"

    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        lineno = 10
    
    node_without_decorators = MockNodeWithoutDecorators()
    result = get_first_line_number(node_without_decorators)
    assert result == 10, f"Expected 10, got {result}"

    # Test with a node that has an empty decorator list
    class MockNodeWithEmptyDecorators:
        decorator_list = []
        lineno = 10
    
    node_with_empty_decorators = MockNodeWithEmptyDecorators()
    result = get_first_line_number(node_with_empty_decorators)
    assert result == 10, f"Expected 10, got {result}"


# LLM-generated content at query #32
#--------------------------

```python
def test_get_first_line_number():
    # Mock a node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(l) for l in decorator_linenos]
    
    class MockNodeWithoutDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []
    
    class MockNodeWithoutDecoratorList:
        def __init__(self, lineno):
            self.lineno = lineno
    
    # Test with decorators - should return first decorator's lineno
    node_with_decorators = MockNodeWithDecorators(5, [2, 3])
    assert get_first_line_number(node_with_decorators) == 2
    
    # Test with multiple decorators - should return first one
    node_with_multiple_decorators = MockNodeWithDecorators(10, [7, 8, 9])
    assert get_first_line_number(node_with_multiple_decorators) == 7
    
    # Test without decorators - should return node's lineno
    node_without_decorators = MockNodeWithoutDecorators(15)
    assert get_first_line_number(node_without_decorators) == 15
    
    # Test without decorator_list attribute - should return node's lineno
    node_without_decorator_list = MockNodeWithoutDecoratorList(20)
    assert get_first_line_number(node_without_decorator_list) == 20
    
    # Test with empty decorator list - should return node's lineno
    node_empty_decorators = MockNodeWithoutDecorators(25)
    assert get_first_line_number(node_empty_decorators) == 25
```


# LLM-generated content at query #33
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
    
    class Decorator:
        lineno = 3
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [Decorator()]
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that has no decorator_list attribute
    class NodeWithoutDecorators:
        lineno = 15
    
    node_without = NodeWithoutDecorators()
    assert get_first_line_number(node_without) == 15
    
    # Test with multiple decorators (should return first decorator's lineno)
    class MultiDecoratedNode:
        lineno = 20
    
    class Decorator1:
        lineno = 7
    
    class Decorator2:
        lineno = 8
    
    multi_decorated = MultiDecoratedNode()
    multi_decorated.decorator_list = [Decorator1(), Decorator2()]
    assert get_first_line_number(multi_decorated) == 7
    
    # Test with empty decorator_list
    class EmptyDecoratorListNode:
        lineno = 25
        decorator_list = []
    
    empty_node = EmptyDecoratorListNode()
    assert get_first_line_number(empty_node) == 25
```


# LLM-generated content at query #34
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []

    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a node that has decorators
    class NodeWithDecorators:
        lineno = 10
        decorator_list = []

    node_with_deco = NodeWithDecorators()
    decorator1 = type('Decorator', (), {'lineno': 3})()
    decorator2 = type('Decorator', (), {'lineno': 4})()
    node_with_deco.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_deco) == 3

    # Test with a node that has decorator_list attribute but empty
    class NodeWithEmptyDecorators:
        lineno = 7
        decorator_list = []

    node_empty = NodeWithEmptyDecorators()
    assert get_first_line_number(node_empty) == 7

    # Test with a node that doesn't have decorator_list attribute
    class NodeWithoutDecorators:
        lineno = 12

    node_without = NodeWithoutDecorators()
    assert get_first_line_number(node_without) == 12

    # Test edge case: node with lineno 0 and no decorators
    class ZeroLinenoNode:
        lineno = 0
        decorator_list = []

    zero_node = ZeroLinenoNode()
    assert get_first_line_number(zero_node) == 0
```


# LLM-generated content at query #35
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class MockNodeNoDecorators:
        lineno = 5
    
    result = get_first_line_number(MockNodeNoDecorators())
    assert result == 5, f"Expected 5, got {result}"

    # Test node with decorators
    class MockDecorator:
        lineno = 3
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    result = get_first_line_number(MockNodeWithDecorators())
    assert result == 3, f"Expected 3, got {result}"

    # Test node with empty decorator list
    class MockNodeEmptyDecorators:
        decorator_list = []
        lineno = 8
    
    result = get_first_line_number(MockNodeEmptyDecorators())
    assert result == 8, f"Expected 8, got {result}"

    # Test node without decorator_list attribute
    class MockNodeNoAttr:
        lineno = 15
    
    result = get_first_line_number(MockNodeNoAttr())
    assert result == 15, f"Expected 15, got {result}"

    # Test node with multiple decorators - should return first decorator's lineno
    class MockDecorator1:
        lineno = 1
    
    class MockDecorator2:
        lineno = 2
    
    class MockNodeMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 20
    
    result = get_first_line_number(MockNodeMultipleDecorators())
    assert result == 1, f"Expected 1, got {result}"
```


# LLM-generated content at query #36
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.decorator_list = [MockDecorator(5)]
            self.lineno = 10

    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test with a node without decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20

    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecoratorList:
        def __init__(self):
            self.lineno = 30

    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 30
```


# LLM-generated content at query #37
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator', (), {'lineno': 5})(),
                type('Decorator', (), {'lineno': 6})()
            ]
            self.lineno = 10
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 5

    # Test with a node that has no decorators
    class NodeWithoutDecorators:
        def __init__(self):
            self.lineno = 15
    
    node_without_decorators = NodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 15

    # Test with a node that has empty decorator list
    class NodeWithEmptyDecorators:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    node_empty_decorators = NodeWithEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #38
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = []
    
    class Decorator:
        lineno = 3
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [Decorator()]
    assert get_first_line_number(decorated_node) == 3

    # Test with multiple decorators - should return first decorator's lineno
    class MultiDecoratorNode:
        lineno = 15
        decorator_list = []
    
    decorator1 = Decorator()
    decorator1.lineno = 7
    decorator2 = Decorator()
    decorator2.lineno = 8
    
    multi_decorator_node = MultiDecoratorNode()
    multi_decorator_node.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(multi_decorator_node) == 7

    # Test with empty decorator_list
    class EmptyDecoratorNode:
        lineno = 20
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 20

    # Test with node that doesn't have decorator_list attribute (should use lineno)
    class NoDecoratorListNode:
        lineno = 25
    
    no_decorator_node = NoDecoratorListNode()
    assert get_first_line_number(no_decorator_node) == 25
```


# LLM-generated content at query #39
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    node2 = DecoratorNode()
    assert get_first_line_number(node2) == 3

    # Test with multiple decorators
    class MultiDecoratorNode:
        lineno = 15
        decorator_list = [
            type('Decorator', (), {'lineno': 7})(),
            type('Decorator', (), {'lineno': 9})()
        ]
    node3 = MultiDecoratorNode()
    assert get_first_line_number(node3) == 7

    # Test with empty decorator list
    class EmptyDecoratorNode:
        lineno = 20
        decorator_list = []
    node4 = EmptyDecoratorNode()
    assert get_first_line_number(node4) == 20
```


# LLM-generated content at query #40
#--------------------------

```python
def test_get_first_line_number():
    # Mock AST node without decorators
    class MockNodeNoDecorators:
        lineno = 5
    
    node_no_decorators = MockNodeNoDecorators()
    assert get_first_line_number(node_no_decorators) == 5
    
    # Mock AST node with decorators
    class MockDecorator:
        lineno = 3
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 5
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3
    
    # Mock AST node with empty decorator list
    class MockNodeEmptyDecorators:
        decorator_list = []
        lineno = 5
    
    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 5
    
    # Mock AST node without decorator_list attribute
    class MockNodeNoAttr:
        lineno = 7
    
    node_no_attr = MockNodeNoAttr()
    assert get_first_line_number(node_no_attr) == 7


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        def __init__(self):
            self.lineno = 5
            self.decorator_list = []
    
    node_without_decorators = SimpleNode()
    assert get_first_line_number(node_without_decorators) == 5
    
    # Test node with decorators
    class DecoratedNode:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [DecoratorNode()]
    
    class DecoratorNode:
        def __init__(self):
            self.lineno = 3
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 3
    
    # Test node without decorator_list attribute
    class NodeWithoutDecoratorList:
        def __init__(self):
            self.lineno = 20
    
    node_no_decorator_list = NodeWithoutDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 20
    
    # Test node with empty decorator_list
    class EmptyDecoratorListNode:
        def __init__(self):
            self.lineno = 15
            self.decorator_list = []
    
    node_empty_decorators = EmptyDecoratorListNode()
    assert get_first_line_number(node_empty_decorators) == 15
    
    # Test node with multiple decorators (should return first decorator's lineno)
    class MultipleDecoratorsNode:
        def __init__(self):
            self.lineno = 25
            self.decorator_list = [
                DecoratorNode2(1),
                DecoratorNode2(2),
                DecoratorNode2(3)
            ]
    
    class DecoratorNode2:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_multiple_decorators = MultipleDecoratorsNode()
    assert get_first_line_number(node_multiple_decorators) == 1
```


# LLM-generated content at query #2
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 10
    
    class Decorator:
        lineno = 3
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [Decorator()]
    assert get_first_line_number(decorated_node) == 3
    
    # Test with multiple decorators
    class MultiDecoratedNode:
        decorator_list = []
        lineno = 15
    
    decorator1 = Decorator()
    decorator1.lineno = 1
    decorator2 = Decorator()
    decorator2.lineno = 2
    
    multi_decorated_node = MultiDecoratedNode()
    multi_decorated_node.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(multi_decorated_node) == 1
    
    # Test with node that doesn't have decorator_list attribute
    class NodeWithoutDecorators:
        lineno = 20
    
    node_without = NodeWithoutDecorators()
    assert get_first_line_number(node_without) == 20
    
    # Test with empty decorator_list
    class EmptyDecoratorNode:
        lineno = 25
        decorator_list = []
    
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 25
```


# LLM-generated content at query #3
#--------------------------

```python
def test_get_first_line_number():
    # Mock node with decorator_list
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self, dec_lineno, node_lineno):
            self.decorator_list = [MockDecorator(dec_lineno)]
            self.lineno = node_lineno
            self.end_lineno = node_lineno
    
    # Test with decorators
    node_with_decorators = MockNodeWithDecorators(dec_lineno=5, node_lineno=10)
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test without decorators (empty decorator_list)
    class MockNodeEmptyDecorators:
        def __init__(self, lineno):
            self.decorator_list = []
            self.lineno = lineno
            self.end_lineno = lineno
    
    node_empty_decorators = MockNodeEmptyDecorators(lineno=20)
    assert get_first_line_number(node_empty_decorators) == 20
    
    # Test without decorator_list attribute
    class MockNodeNoDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
            self.end_lineno = lineno
    
    node_no_decorators = MockNodeNoDecorators(lineno=30)
    assert get_first_line_number(node_no_decorators) == 30
    
    # Test with multiple decorators
    class MockNodeMultipleDecorators:
        def __init__(self):
            self.decorator_list = [MockDecorator(2), MockDecorator(3)]
            self.lineno = 10
            self.end_lineno = 10
    
    node_multiple = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple) == 2  # Should return first decorator's lineno
```


# LLM-generated content at query #4
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function node (no decorators)
    import ast
    
    code_without_decorators = """
def foo():
    pass
"""
    tree = ast.parse(code_without_decorators)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # lineno starts at 1, function starts at line 2
    
    # Test with a function that has decorators
    code_with_decorators = """
@decorator1
@decorator2
def bar():
    pass
"""
    tree = ast.parse(code_with_decorators)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # Should return lineno of first decorator (line 2)
    
    # Test with a class that has decorators
    code_class_with_decorators = """
@class_decorator
class MyClass:
    pass
"""
    tree = ast.parse(code_class_with_decorators)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2  # Should return lineno of first decorator
    
    # Test with a function that has no decorator_list attribute (edge case)
    code_no_decorator_list = """
def baz():
    pass
"""
    tree = ast.parse(code_no_decorator_list)
    func_node = tree.body[0]
    # Remove decorator_list to test the fallback
    if hasattr(func_node, 'decorator_list'):
        del func_node.decorator_list
    assert get_first_line_number(func_node) == func_node.lineno  # Should fall back to node.lineno
    
    # Test with empty decorator list
    code_empty_decorators = """
def qux():
    pass
"""
    tree = ast.parse(code_empty_decorators)
    func_node = tree.body[0]
    # Explicitly set empty decorator list
    func_node.decorator_list = []
    assert get_first_line_number(func_node) == func_node.lineno  # Should return node.lineno when decorator list is empty
```


# LLM-generated content at query #5
#--------------------------

```python
def test_get_first_line_number():
    # Create a mock AST node with no decorators
    class MockNode:
        lineno = 5
    
    node_no_decorators = MockNode()
    assert get_first_line_number(node_no_decorators) == 5
    
    # Create a mock AST node with decorators
    class MockDecorator:
        lineno = 3
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 7
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3
    
    # Create a mock AST node with empty decorator list
    class MockNodeEmptyDecorators:
        decorator_list = []
        lineno = 10
    
    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 10
```


# LLM-generated content at query #6
#--------------------------

```python
def test_get_first_line_number():
    # Test with node that has decorators
    class MockDecorator:
        lineno = 5
    
    class MockDecoratedNode:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    node_with_decorators = MockDecoratedNode()
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with node without decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 20
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorList:
        lineno = 30
    
    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 30
```


# LLM-generated content at query #7
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test node with decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = []
    decorated_node = DecoratedNode()
    decorator1 = type('Decorator', (), {'lineno': 3})()
    decorator2 = type('Decorator', (), {'lineno': 4})()
    decorated_node.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(decorated_node) == 3

    # Test node with decorator_list attribute but empty
    class EmptyDecoratorNode:
        lineno = 7
        decorator_list = []
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 7

    # Test node without decorator_list attribute (older Python versions)
    class NoDecoratorListNode:
        lineno = 15
    no_decorator_node = NoDecoratorListNode()
    assert get_first_line_number(no_decorator_node) == 15
```


# LLM-generated content at query #8
#--------------------------

```python
def test_get_first_line_number():
    # Mock node with decorators - should return first decorator's lineno
    class MockNodeWithDecorators:
        def __init__(self):
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 5})(),
                type('Decorator2', (), {'lineno': 6})()
            ]
            self.lineno = 10
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5
    
    # Mock node without decorators - should return node's lineno
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 15
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 15
    
    # Mock node with decorator_list attribute missing - should return node's lineno
    class MockNodeNoDecoratorList:
        def __init__(self):
            self.lineno = 20
    
    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 20
    
    # Mock node with decorators at different line numbers
    class MockNodeMultipleDecorators:
        def __init__(self):
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 1})(),
                type('Decorator2', (), {'lineno': 2})(),
                type('Decorator3', (), {'lineno': 3})()
            ]
            self.lineno = 10
    
    node_multiple = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple) == 1


# LLM-generated content at query #9
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [
            type('Decorator1', (), {'lineno': 3})(),
            type('Decorator2', (), {'lineno': 4})()
        ]
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3

    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        lineno = 5
        decorator_list = []
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 5

    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecoratorList:
        lineno = 10
    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 10

    # Test with a node that has single decorator
    class MockNodeSingleDecorator:
        lineno = 7
        decorator_list = [type('Decorator', (), {'lineno': 2})()]
    node_single_decorator = MockNodeSingleDecorator()
    assert get_first_line_number(node_single_decorator) == 2

    # Test with node at line 1
    class MockNodeLine1:
        lineno = 1
        decorator_list = []
    node_line1 = MockNodeLine1()
    assert get_first_line_number(node_line1) == 1
```


# LLM-generated content at query #10
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class NodeWithoutDecorators:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = NodeWithoutDecorators()
    assert get_first_line_number(node) == 5
    
    # Test node with decorators
    class Decorator:
        lineno = 3
    
    class NodeWithDecorators:
        lineno = 5
        end_lineno = 10
        decorator_list = [Decorator()]
    
    node = NodeWithDecorators()
    assert get_first_line_number(node) == 3
    
    # Test node with decorator_list attribute set to None
    class NodeWithNoneDecorators:
        lineno = 7
        end_lineno = 12
        decorator_list = None
    
    node = NodeWithNoneDecorators()
    assert get_first_line_number(node) == 7
    
    # Test node without decorator_list attribute
    class NodeWithoutDecoratorList:
        lineno = 9
        end_lineno = 15
    
    node = NodeWithoutDecoratorList()
    assert get_first_line_number(node) == 9
    
    # Test node with multiple decorators
    class Decorator1:
        lineno = 1
    
    class Decorator2:
        lineno = 2
    
    class NodeWithMultipleDecorators:
        lineno = 5
        end_lineno = 10
        decorator_list = [Decorator1(), Decorator2()]
    
    node = NodeWithMultipleDecorators()
    assert get_first_line_number(node) == 1
```


# LLM-generated content at query #11
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = []
    
    # Create mock decorator objects
    class Decorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node = DecoratedNode()
    node.decorator_list = [Decorator(3), Decorator(4)]
    assert get_first_line_number(node) == 3
    
    # Test with empty decorator list
    node.decorator_list = []
    assert get_first_line_number(node) == 10
    
    # Test with node that doesn't have decorator_list attribute
    class NodeWithoutDecorators:
        lineno = 7
    
    node = NodeWithoutDecorators()
    assert get_first_line_number(node) == 7
    
    # Test with multiple decorators
    node = DecoratedNode()
    node.decorator_list = [Decorator(1), Decorator(2), Decorator(3)]
    assert get_first_line_number(node) == 1
```


# LLM-generated content at query #12
#--------------------------

```python
def test_get_first_line_number():
    # Test node with decorators
    class NodeWithDecorators:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    
    result = get_first_line_number(NodeWithDecorators())
    assert result == 5, f"Expected 5, got {result}"
    
    # Test node without decorators
    class NodeWithoutDecorators:
        decorator_list = []
        lineno = 15
    
    result = get_first_line_number(NodeWithoutDecorators())
    assert result == 15, f"Expected 15, got {result}"
    
    # Test node without decorator_list attribute
    class NodeWithoutDecoratorList:
        lineno = 20
    
    result = get_first_line_number(NodeWithoutDecoratorList())
    assert result == 20, f"Expected 20, got {result}"
    
    # Test node with empty decorator_list
    class NodeWithEmptyDecoratorList:
        decorator_list = []
        lineno = 25
    
    result = get_first_line_number(NodeWithEmptyDecoratorList())
    assert result == 25, f"Expected 25, got {result}"


# LLM-generated content at query #13
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class MockNodeWithoutDecorators:
        lineno = 5
    
    node = MockNodeWithoutDecorators()
    assert get_first_line_number(node) == 5
    
    # Test with decorators
    class MockDecorator:
        lineno = 3
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    node = MockNodeWithDecorators()
    assert get_first_line_number(node) == 3
    
    # Test with multiple decorators, should return first decorator's lineno
    class MockDecorator1:
        lineno = 2
    
    class MockDecorator2:
        lineno = 4
    
    class MockNodeWithMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 10
    
    node = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node) == 2
    
    # Test with empty decorator list
    class MockNodeWithEmptyDecorators:
        decorator_list = []
        lineno = 7
    
    node = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node) == 7
    
    # Test with node that has no decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 9
    
    node = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node) == 9
```


# LLM-generated content at query #14
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        
        class Decorator:
            lineno = 3
        
        decorator_list = [Decorator()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 3
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 20
        
        class Decorator1:
            lineno = 7
        
        class Decorator2:
            lineno = 8
        
        decorator_list = [Decorator1(), Decorator2()]
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 7
    
    # Test with a node where decorator_list is not present
    class NoDecoratorListNode:
        lineno = 15
    
    node = NoDecoratorListNode()
    assert get_first_line_number(node) == 15
    
    # Test with a node where decorator_list is None
    class NoneDecoratorListNode:
        lineno = 25
        decorator_list = None
    
    node = NoneDecoratorListNode()
    assert get_first_line_number(node) == 25
```


# LLM-generated content at query #15
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test node with decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [
            type('Decorator', (), {'lineno': 3})(),
            type('Decorator', (), {'lineno': 4})()
        ]
    
    node2 = DecoratedNode()
    assert get_first_line_number(node2) == 3
    
    # Test node with no decorator_list attribute
    class NoDecoratorListNode:
        lineno = 8
    
    node3 = NoDecoratorListNode()
    assert get_first_line_number(node3) == 8
    
    # Test empty decorator list
    class EmptyDecoratorNode:
        lineno = 15
        decorator_list = []
    
    node4 = EmptyDecoratorNode()
    assert get_first_line_number(node4) == 15
    
    # Test edge case: decorator at line 1
    class FirstLineDecoratorNode:
        lineno = 10
        decorator_list = [
            type('Decorator', (), {'lineno': 1})()
        ]
    
    node5 = FirstLineDecoratorNode()
    assert get_first_line_number(node5) == 1
```


# LLM-generated content at query #16
#--------------------------

```python
def test_get_first_line_number():
    # Mock node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self):
            self.decorator_list = [MockDecorator(5), MockDecorator(6)]
            self.lineno = 10
    
    node_with_deco = MockNodeWithDecorators()
    assert get_first_line_number(node_with_deco) == 5
    
    # Mock node without decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 10
    
    node_without_deco = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_deco) == 10
    
    # Mock node with empty decorator list
    class MockNodeEmptyDecorators:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 15
    
    node_empty_deco = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_deco) == 15
```


# LLM-generated content at query #17
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class SimpleFunction:
        pass
    node = SimpleFunction
    node.lineno = 10
    result = get_first_line_number(node)
    assert result == 10, "Should return node.lineno when there are no decorators"

    # Test with decorators
    class DecoratedFunction:
        pass
    node = DecoratedFunction
    node.lineno = 15
    node.decorator_list = [type('Decorator', (), {'lineno': 12})()]
    result = get_first_line_number(node)
    assert result == 12, "Should return first decorator's lineno when decorators exist"

    # Test with empty decorator list
    class EmptyDecorators:
        pass
    node = EmptyDecorators
    node.lineno = 20
    node.decorator_list = []
    result = get_first_line_number(node)
    assert result == 20, "Should return node.lineno when decorator_list is empty"

    # Test with multiple decorators
    class MultipleDecorators:
        pass
    node = MultipleDecorators
    node.lineno = 25
    node.decorator_list = [
        type('Decorator', (), {'lineno': 22})(),
        type('Decorator', (), {'lineno': 23})()
    ]
    result = get_first_line_number(node)
    assert result == 22, "Should return lineno of first decorator in the list"
```


# LLM-generated content at query #18
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorator
    class MockNodeNoDecorator:
        lineno = 10
    
    result = get_first_line_number(MockNodeNoDecorator())
    assert result == 10, f"Expected 10, got {result}"
    
    # Test with decorators
    class MockNodeWithDecorator:
        decorator_list = [
            type('Decorator', (), {'lineno': 5})(),
            type('Decorator', (), {'lineno': 6})()
        ]
        lineno = 10
    
    result = get_first_line_number(MockNodeWithDecorator())
    assert result == 5, f"Expected 5, got {result}"
    
    # Test with empty decorator list
    class MockNodeEmptyDecorator:
        decorator_list = []
        lineno = 15
    
    result = get_first_line_number(MockNodeEmptyDecorator())
    assert result == 15, f"Expected 15, got {result}"
    
    # Test with decorator but no lineno attribute
    class MockNodeNoLineno:
        decorator_list = [
            type('Decorator', (), {'lineno': 20})()
        ]
    
    result = get_first_line_number(MockNodeNoLineno())
    assert result == 20, f"Expected 20, got {result}"
```


# LLM-generated content at query #19
#--------------------------

```python
def test_get_first_line_number():
    # Mock an AST node without decorators
    class MockNode:
        lineno = 5
    
    node_no_decorators = MockNode()
    assert get_first_line_number(node_no_decorators) == 5
    
    # Mock an AST node with decorators
    class MockDecorator:
        lineno = 2
    
    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator()]
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 2
    
    # Mock an AST node with empty decorator list
    class MockNodeEmptyDecorators:
        lineno = 10
        decorator_list = []
    
    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 10


# LLM-generated content at query #20
#--------------------------

```python
def test_get_first_line_number():
    # Mock a node with decorators
    class MockDecorator:
        lineno = 5
    
    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5, "Should return first decorator line number"
    
    # Mock a node without decorators
    class MockNodeWithoutDecorators:
        lineno = 15
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 15, "Should return node's own line number"
    
    # Mock a node with empty decorator list
    class MockNodeEmptyDecorators:
        lineno = 20
        decorator_list = []
    
    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 20, "Should return node's own line number when decorator list is empty"
    
    # Mock a node without decorator_list attribute
    class MockNodeNoDecoratorsAttr:
        lineno = 25
    
    node_no_decorators_attr = MockNodeNoDecoratorsAttr()
    assert get_first_line_number(node_no_decorators_attr) == 25, "Should return node's own line number when no decorator_list attribute"


# LLM-generated content at query #21
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    assert get_first_line_number(SimpleNode()) == 5
    
    # Test node with decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = []
    
    decorator1 = type('Decorator', (), {'lineno': 3})()
    decorator2 = type('Decorator', (), {'lineno': 4})()
    DecoratedNode.decorator_list = [decorator1, decorator2]
    
    assert get_first_line_number(DecoratedNode()) == 3
    
    # Test node without decorator_list attribute
    class NoDecoratorListNode:
        lineno = 7
    
    assert get_first_line_number(NoDecoratorListNode()) == 7
    
    # Test node with empty decorator_list
    class EmptyDecoratorNode:
        lineno = 15
        decorator_list = []
    
    assert get_first_line_number(EmptyDecoratorNode()) == 15
    
    # Test node with multiple decorators
    class MultiDecoratedNode:
        lineno = 20
        decorator_list = []
    
    d1 = type('Decorator', (), {'lineno': 1})()
    d2 = type('Decorator', (), {'lineno': 2})()
    d3 = type('Decorator', (), {'lineno': 3})()
    MultiDecoratedNode.decorator_list = [d1, d2, d3]
    
    assert get_first_line_number(MultiDecoratedNode()) == 1
```


# LLM-generated content at query #22
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        lineno = 5
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    result = get_first_line_number(MockNodeWithDecorators())
    assert result == 5, "Should return first decorator's line number when decorators exist"
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 15
    
    result = get_first_line_number(MockNodeWithoutDecorators())
    assert result == 15, "Should return node's line number when no decorators exist"
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 20
    
    result = get_first_line_number(MockNodeNoDecoratorAttr())
    assert result == 20, "Should return node's line number when decorator_list attribute doesn't exist"


# LLM-generated content at query #23
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test node with decorators
    class NodeWithDecorator:
        decorator_list = []
        lineno = 10
    
    decorator = type('Decorator', (), {'lineno': 3})()
    NodeWithDecorator.decorator_list = [decorator]
    node2 = NodeWithDecorator()
    assert get_first_line_number(node2) == 3
    
    # Test node with multiple decorators
    class NodeWithMultipleDecorators:
        decorator_list = []
        lineno = 15
    
    decorator1 = type('Decorator1', (), {'lineno': 7})()
    decorator2 = type('Decorator2', (), {'lineno': 8})()
    NodeWithMultipleDecorators.decorator_list = [decorator1, decorator2]
    node3 = NodeWithMultipleDecorators()
    assert get_first_line_number(node3) == 7
    
    # Test node with no decorator_list attribute
    class NodeWithoutDecoratorList:
        lineno = 20
    
    node4 = NodeWithoutDecoratorList()
    assert get_first_line_number(node4) == 20
```


# LLM-generated content at query #24
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class MockNodeWithoutDecorators:
        lineno = 5
    
    node = MockNodeWithoutDecorators()
    assert get_first_line_number(node) == 5
    
    # Test with decorators
    class MockDecorator:
        lineno = 2
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 5
    
    node = MockNodeWithDecorators()
    assert get_first_line_number(node) == 2
    
    # Test with empty decorator_list
    class MockNodeWithEmptyDecorators:
        decorator_list = []
        lineno = 10
    
    node = MockNodeWithEmptyDecorators()
    assert get_first_line_number(node) == 10
    
    # Test with multiple decorators - should return first one
    class MockDecorator1:
        lineno = 1
    
    class MockDecorator2:
        lineno = 3
    
    class MockNodeWithMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 5
    
    node = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node) == 1
    
    # Test with node that has no decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 7
    
    node = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node) == 7
    
    # Edge case: lineno at 0
    class MockNodeLineZero:
        lineno = 0
    
    node = MockNodeLineZero()
    assert get_first_line_number(node) == 0
```


# LLM-generated content at query #25
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = []
    node2 = DecoratedNode()
    
    # Add mock decorators with different line numbers
    class Decorator:
        def __init__(self, line):
            self.lineno = line
    
    node2.decorator_list = [Decorator(3), Decorator(4)]
    assert get_first_line_number(node2) == 3
    
    # Test with empty decorator list
    node3 = DecoratedNode()
    node3.decorator_list = []
    assert get_first_line_number(node3) == 10
    
    # Test with node that has no decorator_list attribute
    class NoDecoratorNode:
        lineno = 20
    node4 = NoDecoratorNode()
    assert get_first_line_number(node4) == 20
```


# LLM-generated content at query #26
#--------------------------

```python
def test_get_first_line_number():
    # Mock a node with decorators
    class MockDecorator:
        lineno = 5
    
    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Mock a node without decorators
    class MockNodeWithoutDecorators:
        lineno = 20
        decorator_list = []
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20

    # Mock a node without decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 30
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 30
```


# LLM-generated content at query #27
#--------------------------

```python
def test_get_first_line_number():
    # Mock a node with decorators
    class MockDecorator:
        lineno = 5
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5
    
    # Mock a node without decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 15
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 15


# LLM-generated content at query #28
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class FakeNodeNoDecorators:
        lineno = 5
    
    node_no_decorators = FakeNodeNoDecorators()
    assert get_first_line_number(node_no_decorators) == 5
    
    # Test with one decorator
    class FakeDecorator1:
        lineno = 3
    
    class FakeNodeOneDecorator:
        lineno = 5
        decorator_list = [FakeDecorator1()]
    
    node_one_decorator = FakeNodeOneDecorator()
    assert get_first_line_number(node_one_decorator) == 3
    
    # Test with multiple decorators
    class FakeDecorator2:
        lineno = 1
    
    class FakeNodeMultipleDecorators:
        lineno = 5
        decorator_list = [FakeDecorator1(), FakeDecorator2()]
    
    node_multiple = FakeNodeMultipleDecorators()
    assert get_first_line_number(node_multiple) == 3
    
    # Test with empty decorator_list
    class FakeNodeEmptyDecorators:
        lineno = 7
        decorator_list = []
    
    node_empty = FakeNodeEmptyDecorators()
    assert get_first_line_number(node_empty) == 7
    
    # Test with node that doesn't have decorator_list attribute
    class FakeNodeNoDecoratorAttr:
        lineno = 10
    
    node_no_attr = FakeNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_attr) == 10
```


# LLM-generated content at query #29
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class MockNodeNoDecorators:
        lineno = 5
        decorator_list = []
    
    assert get_first_line_number(MockNodeNoDecorators()) == 5
    
    # Test with decorators
    class MockDecorator:
        lineno = 2
    
    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator()]
    
    assert get_first_line_number(MockNodeWithDecorators()) == 2
    
    # Test with node that has no decorator_list attribute
    class MockNodeNoAttribute:
        lineno = 10
    
    assert get_first_line_number(MockNodeNoAttribute()) == 10
    
    # Test with empty decorator list
    class MockNodeEmptyDecorators:
        lineno = 15
        decorator_list = []
    
    assert get_first_line_number(MockNodeEmptyDecorators()) == 15
    
    # Test with multiple decorators
    class MockDecorator1:
        lineno = 3
    class MockDecorator2:
        lineno = 4
    
    class MockNodeMultipleDecorators:
        lineno = 6
        decorator_list = [MockDecorator1(), MockDecorator2()]
    
    assert get_first_line_number(MockNodeMultipleDecorators()) == 3
```


# LLM-generated content at query #30
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test node with decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3

    # Test node with multiple decorators (should return first decorator's line)
    class MultiDecoratedNode:
        lineno = 20
        decorator_list = [
            type('Decorator', (), {'lineno': 7})(),
            type('Decorator', (), {'lineno': 8})()
        ]
    
    multi_node = MultiDecoratedNode()
    assert get_first_line_number(multi_node) == 7

    # Test node with no decorator_list attribute
    class NoDecoratorListNode:
        lineno = 15
    
    no_deco_node = NoDecoratorListNode()
    assert get_first_line_number(no_deco_node) == 15

    # Test node with empty decorator_list
    class EmptyDecoratorListNode:
        lineno = 25
        decorator_list = []
    
    empty_node = EmptyDecoratorListNode()
    assert get_first_line_number(empty_node) == 25
```


# LLM-generated content at query #31
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    node_without_decorator = type('Node', (), {'lineno': 5, 'decorator_list': []})()
    assert get_first_line_number(node_without_decorator) == 5
    
    # Test with decorators
    decorator1 = type('Decorator', (), {'lineno': 3})()
    decorator2 = type('Decorator', (), {'lineno': 4})()
    node_with_decorators = type('Node', (), {'lineno': 5, 'decorator_list': [decorator1, decorator2]})()
    assert get_first_line_number(node_with_decorators) == 3
    
    # Test with empty decorator_list
    node_empty_decorators = type('Node', (), {'lineno': 10, 'decorator_list': []})()
    assert get_first_line_number(node_empty_decorators) == 10
    
    # Test with node that has no decorator_list attribute
    node_no_decorator_attr = type('Node', (), {'lineno': 7})()
    assert get_first_line_number(node_no_decorator_attr) == 7
    
    # Test edge case: decorator_list is None
    node_none_decorators = type('Node', (), {'lineno': 8, 'decorator_list': None})()
    assert get_first_line_number(node_none_decorators) == 8
```


# LLM-generated content at query #32
#--------------------------

```python
def test_get_first_line_number():
    # Mock node with decorator_list but no decorators
    node_no_decorators = type('Node', (), {'lineno': 5, 'decorator_list': []})()
    assert get_first_line_number(node_no_decorators) == 5

    # Mock node with decorators
    decorator_mock = type('Decorator', (), {'lineno': 3})()
    node_with_decorators = type('Node', (), {'lineno': 5, 'decorator_list': [decorator_mock]})()
    assert get_first_line_number(node_with_decorators) == 3

    # Mock node without decorator_list attribute
    node_no_attribute = type('Node', (), {'lineno': 10})()
    assert get_first_line_number(node_no_attribute) == 10

    # Mock node with multiple decorators
    decorator1 = type('Decorator', (), {'lineno': 1})()
    decorator2 = type('Decorator', (), {'lineno': 2})()
    node_multiple = type('Node', (), {'lineno': 5, 'decorator_list': [decorator1, decorator2]})()
    assert get_first_line_number(node_multiple) == 1
```


# LLM-generated content at query #33
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class MockNode:
        lineno = 5
        decorator_list = []
    assert get_first_line_number(MockNode()) == 5

    # Test with decorators
    class MockDecorator:
        lineno = 3
    
    class MockNodeWithDecorators:
        lineno = 5
        decorator_list = [MockDecorator()]
    assert get_first_line_number(MockNodeWithDecorators()) == 3

    # Test with multiple decorators - should return first decorator's lineno
    class MockDecorator1:
        lineno = 2
    class MockDecorator2:
        lineno = 4
    
    class MockNodeMultipleDecorators:
        lineno = 6
        decorator_list = [MockDecorator1(), MockDecorator2()]
    assert get_first_line_number(MockNodeMultipleDecorators()) == 2

    # Test when node has no decorator_list attribute
    class MockNodeNoDecorators:
        lineno = 10
    assert get_first_line_number(MockNodeNoDecorators()) == 10

    # Test with empty decorator_list
    class MockNodeEmptyDecorators:
        lineno = 8
        decorator_list = []
    assert get_first_line_number(MockNodeEmptyDecorators()) == 8
```


# LLM-generated content at query #34
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class FakeNodeWithoutDecorators:
        lineno = 5
        decorator_list = []
    
    node_without_decorators = FakeNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 5
    
    # Test with decorators
    class FakeDecorator:
        lineno = 3
    
    class FakeNodeWithDecorators:
        lineno = 10
        decorator_list = [FakeDecorator()]
    
    node_with_decorators = FakeNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 3
    
    # Test with multiple decorators (should return first decorator's lineno)
    class FakeDecorator1:
        lineno = 1
    
    class FakeDecorator2:
        lineno = 2
    
    class FakeNodeMultipleDecorators:
        lineno = 10
        decorator_list = [FakeDecorator1(), FakeDecorator2()]
    
    node_multiple_decorators = FakeNodeMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 1
```


# LLM-generated content at query #35
#--------------------------

```python
def test_get_first_line_number():
    # Test with node that has decorators
    class MockDecorator:
        lineno = 5
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    assert get_first_line_number(MockNodeWithDecorators()) == 5

    # Test with node that has no decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 15
    
    assert get_first_line_number(MockNodeWithoutDecorators()) == 15

    # Test with node that doesn't have decorator_list attribute
    class MockNodeNoDecAttr:
        lineno = 20
    
    assert get_first_line_number(MockNodeNoDecAttr()) == 20

    # Test with multiple decorators - should return first decorator's lineno
    class MockDecorator1:
        lineno = 1
    class MockDecorator2:
        lineno = 2
    
    class MockNodeMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 10
    
    assert get_first_line_number(MockNodeMultipleDecorators()) == 1

    # Test edge case: empty decorator list
    class MockNodeEmptyDecoratorList:
        decorator_list = []
        lineno = 25
    
    assert get_first_line_number(MockNodeEmptyDecoratorList()) == 25
```


# LLM-generated content at query #36
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5, "Should return node.lineno when no decorators"
    
    # Test node with decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 10
    
    class FakeDecorator:
        lineno = 3
    
    node = DecoratorNode()
    node.decorator_list = [FakeDecorator()]
    assert get_first_line_number(node) == 3, "Should return first decorator's lineno"
    
    # Test node with multiple decorators
    class FakeDecorator2:
        lineno = 7
    
    node.decorator_list = [FakeDecorator(), FakeDecorator2()]
    assert get_first_line_number(node) == 3, "Should return first decorator's lineno even with multiple decorators"
```


# LLM-generated content at query #37
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class MockNodeNoDecorators:
        lineno = 10
        decorator_list = []

    node_no_decorators = MockNodeNoDecorators()
    assert get_first_line_number(node_no_decorators) == 10

    # Test with a node that has decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorators:
        lineno = 10
        decorator_list = [MockDecorator()]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test with a node that has an empty decorator_list
    class MockNodeEmptyDecorators:
        lineno = 20
        decorator_list = []

    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 20

    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecoratorList:
        lineno = 30

    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 30

    # Test with a node that has multiple decorators
    class MockDecorator1:
        lineno = 1

    class MockDecorator2:
        lineno = 2

    class MockNodeMultipleDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]

    node_multiple_decorators = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 1
```


# LLM-generated content at query #38
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockNodeWithDecorators:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    assert get_first_line_number(MockNodeWithDecorators()) == 5
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 15
    assert get_first_line_number(MockNodeWithoutDecorators()) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorList:
        lineno = 20
    assert get_first_line_number(MockNodeNoDecoratorList()) == 20
    
    # Test with multiple decorators
    class MockNodeMultipleDecorators:
        decorator_list = [
            type('Decorator1', (), {'lineno': 2})(),
            type('Decorator2', (), {'lineno': 3})()
        ]
        lineno = 8
    assert get_first_line_number(MockNodeMultipleDecorators()) == 2
```


# LLM-generated content at query #39
#--------------------------

```python
def test_get_first_line_number():
    # Mock a node without decorators
    class MockNode:
        lineno = 5
    
    node = MockNode()
    assert get_first_line_number(node) == 5

    # Mock a node with decorators
    class MockDecorator:
        lineno = 3
    
    class MockDecoratedNode:
        decorator_list = [MockDecorator()]
        lineno = 5
    
    decorated_node = MockDecoratedNode()
    assert get_first_line_number(decorated_node) == 3

    # Mock a node with multiple decorators (should return first decorator's lineno)
    class MockDecorator1:
        lineno = 1
    
    class MockDecorator2:
        lineno = 2
    
    class MockMultiDecoratedNode:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 5
    
    multi_decorated_node = MockMultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1

    # Mock a node without decorator_list attribute (old Python versions)
    class MockNodeNoDecorators:
        lineno = 10
    
    node_no_decorators = MockNodeNoDecorators()
    assert get_first_line_number(node_no_decorators) == 10

    # Mock a node with empty decorator list
    class MockNodeEmptyDecorators:
        decorator_list = []
        lineno = 7
    
    node_empty_decorators = MockNodeEmptyDecorators()
    assert get_first_line_number(node_empty_decorators) == 7
```


# LLM-generated content at query #40
#--------------------------

```python
def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test node with decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = []
    
    decorator_node = type('Decorator', (), {'lineno': 3})()
    node = DecoratedNode()
    node.decorator_list = [decorator_node]
    assert get_first_line_number(node) == 3
    
    # Test node with multiple decorators
    decorator1 = type('Decorator', (), {'lineno': 1})()
    decorator2 = type('Decorator', (), {'lineno': 2})()
    node = DecoratedNode()
    node.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node) == 1
    
    # Test edge case where decorator_list is None
    class NodeWithNoneDecorator:
        lineno = 7
        decorator_list = None
    
    node = NodeWithNoneDecorator()
    assert get_first_line_number(node) == 7
```


