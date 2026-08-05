####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with decorators
    class DecoratedNode:
        lineno = 20
        end_lineno = 25
        decorator_list = [type('Decorator', (), {'lineno': 18})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 18
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultiDecoratedNode:
        lineno = 30
        end_lineno = 35
        decorator_list = [
            type('Decorator1', (), {'lineno': 25})(),
            type('Decorator2', (), {'lineno': 26})()
        ]
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 25
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 40
    
    no_attr_node = NoDecoratorAttr()
    assert get_first_line_number(no_attr_node) == 40
    
    # Test edge case with empty decorator_list
    class EmptyDecoratorList:
        lineno = 50
        decorator_list = []
    
    empty_list_node = EmptyDecoratorList()
    assert get_first_line_number(empty_list_node) == 50
```


# LLM-generated content at query #2
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorsNode:
        def __init__(self):
            self.lineno = 30
    
    no_decorators_node = NoDecoratorsNode()
    assert get_first_line_number(no_decorators_node) == 30
    
    # Test edge case with empty decorator list
    empty_decorators_node = PlainNode()
    empty_decorators_node.decorator_list = []
    assert get_first_line_number(empty_decorators_node) == 20


# LLM-generated content at query #3
#--------------------------

def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
        end_lineno = 25
    
    # Create a mock decorator with a lineno attribute
    class MockDecorator:
        def __init__(self, line):
            self.lineno = line
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [MockDecorator(5)]
    assert get_first_line_number(decorated_node) == 5
    
    # Test with multiple decorators - should return the first decorator's line
    decorated_node.decorator_list = [MockDecorator(3), MockDecorator(7)]
    assert get_first_line_number(decorated_node) == 3
    
    # Test with an empty decorator_list
    empty_decorated_node = DecoratedNode()
    empty_decorated_node.decorator_list = []
    assert get_first_line_number(empty_decorated_node) == 20


# LLM-generated content at query #4
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})]
        lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node without decorators
    class PlainNode:
        decorator_list = []
        lineno = 10
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 10
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 15
    
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 15
```


# LLM-generated content at query #5
#--------------------------

def test_get_first_line_number():
    # Test with a decorated function
    class DecoratedFunc:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    
    node = DecoratedFunc()
    assert get_first_line_number(node) == 5
    
    # Test with a function without decorators
    class UndecoratedFunc:
        decorator_list = []
        lineno = 20
    
    node = UndecoratedFunc()
    assert get_first_line_number(node) == 20
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 30
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 30


# LLM-generated content at query #6
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        lineno = 10
    
    simple_func = SimpleFunction()
    assert get_first_line_number(simple_func) == 10
    
    # Test with a function that has decorators
    class DecoratedFunction:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    
    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 5
    
    # Test with empty decorator list
    class EmptyDecorators:
        decorator_list = []
        lineno = 20
    
    empty_decorators = EmptyDecorators()
    assert get_first_line_number(empty_decorators) == 20
    
    # Test with no decorator attribute at all
    class NoDecoratorAttribute:
        lineno = 30
    
    no_decorator = NoDecoratorAttribute()
    assert get_first_line_number(no_decorator) == 30
```


# LLM-generated content at query #7
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a function that has decorators
    class DecoratedNode:
        decorator_list = []
        
        def __init__(self):
            self.lineno = 10
    
    decorated = DecoratedNode()
    
    # Create a mock decorator with lineno attribute
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorated.decorator_list = [MockDecorator(3), MockDecorator(4)]
    assert get_first_line_number(decorated) == 3
    
    # Test with node that has decorator_list as None or missing
    class NoDecoratorList:
        pass
    
    node_no_decorators = NoDecoratorList()
    node_no_decorators.lineno = 7
    assert get_first_line_number(node_no_decorators) == 7


# LLM-generated content at query #8
#--------------------------

```python
def test_get_first_line_number():
    # Test for a decorated function
    class DecoratedFunction:
        decorator_list = [type('Decorator', (), {'lineno': 10})()]
        lineno = 15
    
    node = DecoratedFunction()
    assert get_first_line_number(node) == 10
    
    # Test for a function without decorators
    class PlainFunction:
        decorator_list = []
        lineno = 20
    
    node = PlainFunction()
    assert get_first_line_number(node) == 20
    
    # Test for a function where decorator_list attribute doesn't exist
    class NoDecoratorList:
        lineno = 25
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 25
```


# LLM-generated content at query #9
#--------------------------

def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})()]
        lineno = 20
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has no decorators
    class PlainNode:
        decorator_list = []
        lineno = 30
    
    node_without_decorators = PlainNode()
    assert get_first_line_number(node_without_decorators) == 30
    
    # Test with a node that has an empty decorator_list attribute
    class EmptyDecoratorNode:
        decorator_list = None
        lineno = 40
    
    node_with_empty_decorators = EmptyDecoratorNode()
    assert get_first_line_number(node_with_empty_decorators) == 40


# LLM-generated content at query #10
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15

    node = SimpleNode()
    assert get_first_line_number(node) == 10

    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = [SimpleNode()]  # decorator with lineno = 10
        lineno = 20
        end_lineno = 25

    decorated_node = DecoratorNode()
    assert get_first_line_number(decorated_node) == 10

    # Test with a node that has multiple decorators (should return first decorator's lineno)
    class MultiDecoratorNode:
        decorator_list = [SimpleNode(), SimpleNode()]  # both have lineno = 10
        lineno = 30
        end_lineno = 35

    multi_decorated_node = MultiDecoratorNode()
    assert get_first_line_number(multi_decorated_node) == 10

    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 40
        end_lineno = 45

    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 40
```


# LLM-generated content at query #11
#--------------------------

```python
def test_get_first_line_number():
    # Test function without decorators
    class SimpleFunction:
        lineno = 42
        decorator_list = []
    simple_func = SimpleFunction()
    assert get_first_line_number(simple_func) == 42

    # Test function with a single decorator
    class DecoratedFunction:
        lineno = 100
        decorator_list = [type('Decorator', (), {'lineno': 50})()]
    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 50

    # Test function with multiple decorators
    class MultiDecoratedFunction:
        lineno = 200
        decorator_list = [
            type('Decorator1', (), {'lineno': 10}),
            type('Decorator2', (), {'lineno': 20})
        ]
    multi_decorated = MultiDecoratedFunction()
    assert get_first_line_number(multi_decorated) == 10

    # Test function with empty decorator_list attribute
    class EmptyDecoratorList:
        lineno = 300
        decorator_list = []
    empty_decorator = EmptyDecoratorList()
    assert get_first_line_number(empty_decorator) == 300

    # Test function without decorator_list attribute
    class NoDecoratorList:
        lineno = 400
    no_decorator = NoDecoratorList()
    assert get_first_line_number(no_decorator) == 400
```


# LLM-generated content at query #12
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 10
    
    # Create mock decorator objects
    class Decorator:
        def __init__(self, line):
            self.lineno = line
    
    node = DecoratorNode()
    node.decorator_list = [Decorator(3), Decorator(4)]
    assert get_first_line_number(node) == 3
    
    # Test with empty decorator_list
    node2 = DecoratorNode()
    node2.decorator_list = []
    assert get_first_line_number(node2) == 10
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorNode:
        lineno = 7
    
    node3 = NoDecoratorNode()
    assert get_first_line_number(node3) == 7


# LLM-generated content at query #13
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with a node that doesn't have decorators
    class PlainNode:
        def __init__(self):
            self.lineno = 15
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 15
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorsNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    empty_decorators_node = EmptyDecoratorsNode()
    assert get_first_line_number(empty_decorators_node) == 20
    
    # Test with a node that has no decorator_list attribute at all
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 25
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 25
```


# LLM-generated content at query #14
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})]
        lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 2}),
            type('Decorator2', (), {'lineno': 3})
        ]
        lineno = 10
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 2


# LLM-generated content at query #15
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
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 20
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 30
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 30
```


# LLM-generated content at query #16
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 5
        decorator_list = [type('Decorator', (), {'lineno': 1})()]
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 1

    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 8
        decorator_list = [
            type('Decorator1', (), {'lineno': 2})(),
            type('Decorator2', (), {'lineno': 3})()
        ]
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 2

    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        lineno = 15
        decorator_list = []
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttribute:
        lineno = 20
    no_decorator_attribute = NoDecoratorAttribute()
    assert get_first_line_number(no_decorator_attribute) == 20
```


# LLM-generated content at query #17
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 8
        end_lineno = 15
    
    class Decorator:
        lineno = 3
    
    decorator_node = DecoratorNode()
    decorator_node.decorator_list = [Decorator()]
    assert get_first_line_number(decorator_node) == 3
    
    # Test with multiple decorators - should return the first decorator's lineno
    class SecondDecorator:
        lineno = 4
    
    decorator_node.decorator_list = [Decorator(), SecondDecorator()]
    assert get_first_line_number(decorator_node) == 3
    
    # Test with node that has no decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 12
        end_lineno = 20
    
    no_decorator_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_node) == 12
    
    # Test edge case: decorator_list is None
    class NoneDecoratorNode:
        lineno = 7
        end_lineno = 14
        decorator_list = None
    
    none_decorator_node = NoneDecoratorNode()
    assert get_first_line_number(none_decorator_node) == 7
    
    # Test edge case: empty decorator_list
    class EmptyDecoratorNode:
        lineno = 9
        end_lineno = 16
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 9
```


# LLM-generated content at query #18
#--------------------------

```python
def test_get_first_line_number():
    # Test function without decorators
    class SimpleFunction:
        lineno = 5
        decorator_list = []
    
    simple_func = SimpleFunction()
    assert get_first_line_number(simple_func) == 5
    
    # Test function with decorators
    class DecoratedFunction:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 3
    
    # Test class without decorators
    class SimpleClass:
        lineno = 20
        decorator_list = []
    
    simple_class = SimpleClass()
    assert get_first_line_number(simple_class) == 20
    
    # Test class with multiple decorators
    class MultiDecoratedClass:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    
    multi_decorated = MultiDecoratedClass()
    assert get_first_line_number(multi_decorated) == 1
    
    # Test node without decorator_list attribute
    class NoDecoratorList:
        lineno = 40
    
    no_decorator = NoDecoratorList()
    assert get_first_line_number(no_decorator) == 40
    
    # Test with actual AST node (function definition)
    import ast
    code = """
@decorator
def my_function():
    pass
"""
    tree = ast.parse(code)
    func_def = tree.body[0]
    assert get_first_line_number(func_def) == 1
    
    # Test with actual AST node (plain function)
    code2 = """
def simple_function():
    pass
"""
    tree2 = ast.parse(code2)
    func_def2 = tree2.body[0]
    assert get_first_line_number(func_def2) == 2
    
    # Test with actual AST node (class with decorator)
    code3 = """
@class_decorator
class MyClass:
    pass
"""
    tree3 = ast.parse(code3)
    class_def = tree3.body[0]
    assert get_first_line_number(class_def) == 1
    
    # Test with actual AST node (plain class)
    code4 = """
class SimpleClass:
    pass
"""
    tree4 = ast.parse(code4)
    class_def4 = tree4.body[0]
    assert get_first_line_number(class_def4) == 2
```


# LLM-generated content at query #19
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    node = type('Node', (), {'lineno': 10, 'decorator_list': []})()
    assert get_first_line_number(node) == 10
    
    # Test with a function that has decorators
    decorator1 = type('Decorator', (), {'lineno': 5})()
    decorator2 = type('Decorator', (), {'lineno': 6})()
    node = type('Node', (), {'lineno': 10, 'decorator_list': [decorator1, decorator2]})()
    assert get_first_line_number(node) == 5
    
    # Test with a class that has decorators
    decorator = type('Decorator', (), {'lineno': 3})()
    node = type('Class', (), {'lineno': 8, 'decorator_list': [decorator]})()
    assert get_first_line_number(node) == 3
    
    # Test with a node that has no decorator_list attribute
    node = type('Node', (), {'lineno': 15})()
    assert get_first_line_number(node) == 15


# LLM-generated content at query #20
#--------------------------

def test_get_first_line_number():
    # Test 1: Function without decorators
    class SimpleFunction:
        def func(self):
            pass
    
    simple_func = SimpleFunction.func
    assert get_first_line_number(simple_func) == simple_func.lineno
    
    # Test 2: Function with decorators
    def decorator(func):
        return func
    
    @decorator
    def decorated_func():
        pass
    
    assert get_first_line_number(decorated_func) == decorated_func.decorator_list[0].lineno
    
    # Test 3: Class with decorators
    def class_decorator(cls):
        return cls
    
    @class_decorator
    class DecoratedClass:
        pass
    
    assert get_first_line_number(DecoratedClass) == DecoratedClass.decorator_list[0].lineno
    
    # Test 4: Class without decorators
    class PlainClass:
        pass
    
    assert get_first_line_number(PlainClass) == PlainClass.lineno
    
    # Test 5: Function from a class with decorators
    class ClassWithDecoratedMethod:
        @decorator
        def method(self):
            pass
    
    method = ClassWithDecoratedMethod.method
    assert get_first_line_number(method) == method.decorator_list[0].lineno
    
    # Test 6: Ensure decorator_list is empty returns node.lineno
    class NoDecoratorClass:
        def method(self):
            pass
    
    no_decorator_method = NoDecoratorClass.method
    assert get_first_line_number(no_decorator_method) == no_decorator_method.lineno
    
    # Test 7: Multiple decorators - should return first decorator's line
    def decorator1(func):
        return func
    
    def decorator2(func):
        return func
    
    @decorator2
    @decorator1
    def multi_decorated():
        pass
    
    assert get_first_line_number(multi_decorated) == multi_decorated.decorator_list[0].lineno


# LLM-generated content at query #21
#--------------------------

```python
def test_get_first_line_number():
    class MockNode:
        def __init__(self, lineno, decorators=None):
            self.lineno = lineno
            self.decorator_list = decorators or []

    # Test node without decorators
    node = MockNode(lineno=5)
    assert get_first_line_number(node) == 5

    # Test node with decorators
    decorator1 = type('Decorator', (), {'lineno': 10})()
    decorator2 = type('Decorator', (), {'lineno': 12})()
    node = MockNode(lineno=15, decorators=[decorator1, decorator2])
    assert get_first_line_number(node) == 10

    # Test node with empty decorator list
    node = MockNode(lineno=20, decorators=[])
    assert get_first_line_number(node) == 20
```


# LLM-generated content at query #22
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class DecoratedFunction:
        pass
    
    decorated_node = DecoratedFunction()
    decorated_node.decorator_list = [type('Decorator', (), {'lineno': 5})()]
    decorated_node.lineno = 10
    
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a non-decorated function
    plain_node = type('PlainNode', (), {'lineno': 20})()
    assert get_first_line_number(plain_node) == 20
    
    # Test with a node that has an empty decorator list
    empty_decorator_node = type('EmptyDecoratorNode', (), {'decorator_list': [], 'lineno': 30})()
    assert get_first_line_number(empty_decorator_node) == 30
    
    # Test with a node that doesn't have decorator_list attribute
    no_decorator_attr_node = type('NoDecoratorAttrNode', (), {'lineno': 40})()
    assert get_first_line_number(no_decorator_attr_node) == 40
    
    # Test with a class definition
    class_node = type('ClassNode', (), {'lineno': 50})()
    assert get_first_line_number(class_node) == 50

```


# LLM-generated content at query #23
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 10
    node = DecoratedNode()
    node.decorator_list = [type('Decorator', (), {'lineno': 3})()]
    assert get_first_line_number(node) == 3

    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 15

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 20

    # Test with multiple decorators, should return first one's lineno
    class MultipleDecoratorsNode:
        decorator_list = []
        lineno = 25
    node = MultipleDecoratorsNode()
    node.decorator_list = [
        type('Decorator1', (), {'lineno': 2})(),
        type('Decorator2', (), {'lineno': 4})()
    ]
    assert get_first_line_number(node) == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_get_first_line_number():
    # Test when node has no decorators
    class SimpleFunction:
        pass
    node = SimpleFunction()
    node.lineno = 10
    node.decorator_list = []
    assert get_first_line_number(node) == 10

    # Test when node has decorators
    class DecoratedFunction:
        pass
    node2 = DecoratedFunction()
    node2.lineno = 20
    decorator1 = type('decorator', (), {'lineno': 15})()
    decorator2 = type('decorator', (), {'lineno': 12})()
    node2.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node2) == 15

    # Test when decorator_list attribute doesn't exist (older Python versions)
    class NoDecoratorAttr:
        pass
    node3 = NoDecoratorAttr()
    node3.lineno = 30
    delattr(node3, 'decorator_list')
    assert get_first_line_number(node3) == 30

    # Test with empty decorator list
    class EmptyDecoratorList:
        pass
    node4 = EmptyDecoratorList()
    node4.lineno = 40
    node4.decorator_list = []
    assert get_first_line_number(node4) == 40

    # Test with multiple decorators - should return first decorator's line
    class MultipleDecorators:
        pass
    node5 = MultipleDecorators()
    node5.lineno = 50
    dec1 = type('dec', (), {'lineno': 45})()
    dec2 = type('dec', (), {'lineno': 48})()
    node5.decorator_list = [dec1, dec2]
    assert get_first_line_number(node5) == 45


# LLM-generated content at query #25
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class Node:
        lineno = 10
    node = Node()
    assert get_first_line_number(node) == 10

    # Test with decorators
    class Decorator:
        lineno = 5
    class NodeWithDecorator:
        decorator_list = [Decorator()]
        lineno = 10
    node_with_decorator = NodeWithDecorator()
    assert get_first_line_number(node_with_decorator) == 5

    # Test with multiple decorators (should return first decorator's lineno)
    class Decorator2:
        lineno = 7
    class NodeWithMultipleDecorators:
        decorator_list = [Decorator(), Decorator2()]
        lineno = 10
    node_with_multiple = NodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple) == 5

    # Test with empty decorator list
    class NodeEmptyDecorators:
        decorator_list = []
        lineno = 15
    node_empty = NodeEmptyDecorators()
    assert get_first_line_number(node_empty) == 15
```


# LLM-generated content at query #26
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 20
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 15
        end_lineno = 25
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    node2 = DecoratedNode()
    assert get_first_line_number(node2) == 5
    
    # Test with a node where decorator_list is not present
    class NoDecoratorAttrNode:
        lineno = 30
        end_lineno = 40
    
    node3 = NoDecoratorAttrNode()
    assert get_first_line_number(node3) == 30
```


# LLM-generated content at query #27
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        lineno = 5
        decorator_list = []
    
    simple_node = SimpleFunction()
    assert get_first_line_number(simple_node) == 5

    # Test with a function that has decorators
    class DecoratedFunction:
        decorator_list = [type('Decorator', (), {'lineno': 10})]
        lineno = 15
    
    decorated_node = DecoratedFunction()
    assert get_first_line_number(decorated_node) == 10

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttribute:
        lineno = 20
    
    no_decorator_node = NoDecoratorAttribute()
    assert get_first_line_number(no_decorator_node) == 20

    # Test with multiple decorators
    class MultiDecoratedFunction:
        decorator_list = [
            type('Decorator1', (), {'lineno': 25}),
            type('Decorator2', (), {'lineno': 30})
        ]
        lineno = 35
    
    multi_decorated_node = MultiDecoratedFunction()
    assert get_first_line_number(multi_decorated_node) == 25

    # Test edge case with empty decorator list
    class EmptyDecoratorList:
        decorator_list = []
        lineno = 40
    
    empty_decorator_node = EmptyDecoratorList()
    assert get_first_line_number(empty_decorator_node) == 40
```


# LLM-generated content at query #28
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has a lineno attribute
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [SimpleNode()]  # decorator has lineno=5
        lineno = 10
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        decorator_list = [SimpleNode(), SimpleNode()]  # both have lineno=5
        lineno = 10
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 15
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorNode:
        lineno = 20
    
    node = NoDecoratorNode()
    assert get_first_line_number(node) == 20
    
    # Test edge case with node having decorator_list as None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 25
    
    node = NoneDecoratorNode()
    assert get_first_line_number(node) == 25  # Should return node.lineno since None is falsy
    
    # Test with a node where decorators have different line numbers
    class DiffDecoratorNode:
        decorator_list = [SimpleNode()]  # lineno=5
        lineno = 30
    
    node = DiffDecoratorNode()
    assert get_first_line_number(node) == 5
```


# LLM-generated content at query #29
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
    node = SimpleNode()
    assert get_first_line_number(node) == 10

    # Test with decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
        end_lineno = 25
    
    decorator1 = SimpleNode()
    decorator1.lineno = 5
    decorator2 = SimpleNode()
    decorator2.lineno = 7
    
    node_with_decorators = DecoratedNode()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 20
    assert get_first_line_number(node_with_decorators) == 5

    # Test with empty decorator list
    node_empty_decorators = DecoratedNode()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 30
    assert get_first_line_number(node_empty_decorators) == 30

    # Test with node that has decorator_list attribute but it's None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 40
    node_none = NoneDecoratorNode()
    assert get_first_line_number(node_none) == 40

    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 50
    node_no_attr = NoDecoratorAttrNode()
    assert get_first_line_number(node_no_attr) == 50
```


# LLM-generated content at query #30
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 20
    
    decorated_node = DecoratedNode()
    
    # Create a mock decorator with a lineno attribute
    class MockDecorator:
        lineno = 5
    
    decorated_node.decorator_list = [MockDecorator()]
    
    assert get_first_line_number(decorated_node) == 5
    
    # Test with multiple decorators - should return the first one's lineno
    class MockDecorator2:
        lineno = 7
    
    decorated_node.decorator_list = [MockDecorator(), MockDecorator2()]
    assert get_first_line_number(decorated_node) == 5
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 30
    
    no_decorator_list_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_list_node) == 30
    
    # Test edge case with empty decorator list
    class EmptyDecoratorNode:
        lineno = 40
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 40
```


# LLM-generated content at query #31
#--------------------------

def test_get_first_line_number():
    class MockNode:
        def __init__(self, lineno=10, decorator_list=None):
            self.lineno = lineno
            self.decorator_list = decorator_list or []

    # Test without decorators
    node = MockNode(lineno=15)
    assert get_first_line_number(node) == 15

    # Test with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    decorators = [MockDecorator(5), MockDecorator(6)]
    node = MockNode(lineno=15, decorator_list=decorators)
    assert get_first_line_number(node) == 5

    # Test with empty decorator list
    node = MockNode(lineno=20, decorator_list=[])
    assert get_first_line_number(node) == 20


# LLM-generated content at query #32
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 1})()]
            self.lineno = 5
            self.end_lineno = 10
    
    decorator_node = DecoratorNode()
    assert get_first_line_number(decorator_node) == 1
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 7
            self.end_lineno = 12
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 7
    
    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator', (), {'lineno': 1})(),
                type('Decorator', (), {'lineno': 2})()
            ]
            self.lineno = 5
            self.end_lineno = 10
    
    multi_decorator_node = MultiDecoratorNode()
    assert get_first_line_number(multi_decorator_node) == 1
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 9
            self.end_lineno = 15
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 9
```


# LLM-generated content at query #33
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with decorators
    class DecoratedNode:
        decorator_list = [
            type('DecoratorNode', (), {'lineno': 2})(),
            type('DecoratorNode', (), {'lineno': 3})()
        ]
        lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 2
    
    # Test with decorator_list attribute missing
    class NoDecoratorList:
        lineno = 7
    
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 7
    
    # Test edge case with empty decorator_list
    class EmptyDecoratorList:
        decorator_list = []
        lineno = 8
    
    empty_decorator_node = EmptyDecoratorList()
    assert get_first_line_number(empty_decorator_node) == 8
```


# LLM-generated content at query #34
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        lineno = 10
        decorator_list = []
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 10
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has multiple decorators
    class MultipleDecoratorsNode:
        lineno = 10
        decorator_list = [
            type('Decorator1', (), {'lineno': 3}),
            type('Decorator2', (), {'lineno': 4}),
            type('Decorator3', (), {'lineno': 5})
        ]
    
    node = MultipleDecoratorsNode()
    assert get_first_line_number(node) == 3
```


# LLM-generated content at query #35
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        lineno = 5
    
    simple_func = SimpleFunction()
    assert get_first_line_number(simple_func) == 5
    
    # Test with a decorated function
    class DecoratedFunction:
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
        lineno = 10
    
    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 3
    
    # Test with empty decorator_list
    class NoDecorators:
        decorator_list = []
        lineno = 7
    
    no_decorators = NoDecorators()
    assert get_first_line_number(no_decorators) == 7
    
    # Test with multiple decorators (should return first decorator's lineno)
    class MultipleDecorators:
        decorator_list = [
            type('Decorator1', (), {'lineno': 2})(),
            type('Decorator2', (), {'lineno': 4})()
        ]
        lineno = 8
    
    multiple_decorators = MultipleDecorators()
    assert get_first_line_number(multiple_decorators) == 2
```


# LLM-generated content at query #36
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with a node that has no decorators
    class SimpleNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 8
    
    node_without_decorators = SimpleNode()
    assert get_first_line_number(node_without_decorators) == 8
    
    # Test with a node that doesn't have decorator_list attribute
    class MinimalNode:
        def __init__(self):
            self.lineno = 3
    
    minimal_node = MinimalNode()
    assert get_first_line_number(minimal_node) == 3
    
    # Test with a node that has decorator_list but it's None
    class NoneDecoratorNode:
        def __init__(self):
            self.decorator_list = None
            self.lineno = 15
    
    none_decorator_node = NoneDecoratorNode()
    # Since decorator_list is None, getattr returns the default []
    # and we should get the node's lineno
    assert get_first_line_number(none_decorator_node) == 15
```


# LLM-generated content at query #37
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    node = type('Node', (), {'lineno': 5, 'decorator_list': []})()
    assert get_first_line_number(node) == 5

    # Test with a function that has decorators
    decorator = type('Decorator', (), {'lineno': 2})()
    node = type('Node', (), {'lineno': 5, 'decorator_list': [decorator]})()
    assert get_first_line_number(node) == 2

    # Test with a function that has multiple decorators
    decorator1 = type('Decorator', (), {'lineno': 2})()
    decorator2 = type('Decorator', (), {'lineno': 3})()
    node = type('Node', (), {'lineno': 5, 'decorator_list': [decorator1, decorator2]})()
    assert get_first_line_number(node) == 2

    # Test with a node that doesn't have decorator_list attribute
    node = type('Node', (), {'lineno': 10})()
    assert get_first_line_number(node) == 10

    # Test with a node that has empty decorator_list
    node = type('Node', (), {'lineno': 7, 'decorator_list': []})()
    assert get_first_line_number(node) == 7


# LLM-generated content at query #38
#--------------------------

```python
def test_get_first_line_number():
    # Test case for a node without decorators
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test case for a node with decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [type('Decorator', (), {'lineno': 5})()]
    assert get_first_line_number(decorated_node) == 5
    
    # Test case for a node with multiple decorators
    class MultiDecoratedNode:
        decorator_list = []
        lineno = 30
    
    multi_decorated_node = MultiDecoratedNode()
    multi_decorated_node.decorator_list = [
        type('Decorator', (), {'lineno': 8})(),
        type('Decorator', (), {'lineno': 12})()
    ]
    assert get_first_line_number(multi_decorated_node) == 8
    
    # Test case for node without decorator_list attribute
    class NoDecoratorListAttr:
        lineno = 40
    
    no_decorator_list_node = NoDecoratorListAttr()
    assert get_first_line_number(no_decorator_list_node) == 40
    
    # Test edge case with empty decorator list
    class EmptyDecoratorList:
        decorator_list = []
        lineno = 50
    
    empty_decorator_list_node = EmptyDecoratorList()
    assert get_first_line_number(empty_decorator_list_node) == 50
```


# LLM-generated content at query #39
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function node (no decorators)
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    decorator_node = DecoratorNode()
    assert get_first_line_number(decorator_node) == 5
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        lineno = 20
        decorator_list = []
    
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 25
    
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 25
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecoratorsNode:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    
    multi_node = MultipleDecoratorsNode()
    assert get_first_line_number(multi_node) == 1
```


# LLM-generated content at query #40
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    import ast
    code = '''
@decorator
def foo():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == func_node.decorator_list[0].lineno

    # Test with a non-decorated function
    code2 = '''
def bar():
    pass
'''
    tree2 = ast.parse(code2)
    func_node2 = tree2.body[0]
    assert get_first_line_number(func_node2) == func_node2.lineno

    # Test with a decorated class
    code3 = '''
@decorator
class MyClass:
    pass
'''
    tree3 = ast.parse(code3)
    class_node = tree3.body[0]
    assert get_first_line_number(class_node) == class_node.decorator_list[0].lineno

    # Test with a non-decorated class
    code4 = '''
class MyClass2:
    pass
'''
    tree4 = ast.parse(code4)
    class_node2 = tree4.body[0]
    assert get_first_line_number(class_node2) == class_node2.lineno
```


# LLM-generated content at query #41
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [SimpleNode()]  # reuse SimpleNode as decorator
        lineno = 10
    decorated_node = DecoratedNode()
    decorated_node.decorator_list[0].lineno = 7
    assert get_first_line_number(decorated_node) == 7

    # Test with multiple decorators (should return first decorator's lineno)
    class MultiDecoratedNode:
        decorator_list = [SimpleNode(), SimpleNode()]
        lineno = 20
    multi_decorated_node = MultiDecoratedNode()
    multi_decorated_node.decorator_list[0].lineno = 15
    multi_decorated_node.decorator_list[1].lineno = 18
    assert get_first_line_number(multi_decorated_node) == 15

    # Test with empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 30
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 30

    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 40
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 40
```


# LLM-generated content at query #42
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 10
            
        class Decorator:
            lineno = 3
    
    node = DecoratorNode()
    node.decorator_list = [node.Decorator()]
    assert get_first_line_number(node) == 3
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultiDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
            
        class Decorator1:
            lineno = 1
            
        class Decorator2:
            lineno = 2
    
    node = MultiDecoratorNode()
    node.decorator_list = [node.Decorator1(), node.Decorator2()]
    assert get_first_line_number(node) == 1
    
    # Test edge case with empty decorator list
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 30
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 30
```


# LLM-generated content at query #43
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    node = type('Node', (), {'lineno': 10, 'decorator_list': []})()
    assert get_first_line_number(node) == 10
    
    # Test with a function that has decorators
    decorator1 = type('Decorator', (), {'lineno': 5})()
    decorator2 = type('Decorator', (), {'lineno': 6})()
    node = type('Node', (), {'lineno': 10, 'decorator_list': [decorator1, decorator2]})()
    assert get_first_line_number(node) == 5
    
    # Test with a node that doesn't have decorator_list attribute
    node = type('Node', (), {'lineno': 20})()
    assert get_first_line_number(node) == 20
    
    # Test with empty decorator list
    node = type('Node', (), {'lineno': 30, 'decorator_list': []})()
    assert get_first_line_number(node) == 30


# LLM-generated content at query #44
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Simple node without decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test case 2: Node with decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 2})]
        lineno = 10
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 2

    # Test case 3: Empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 7
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 7

    # Test case 4: Node without decorator_list attribute
    class NoDecoratorAttr:
        lineno = 3
    no_decorator_attr = NoDecoratorAttr()
    assert get_first_line_number(no_decorator_attr) == 3

    # Test case 5: Multiple decorators - first one is returned
    class MultipleDecorators:
        decorator_list = [
            type('Decorator1', (), {'lineno': 1}),
            type('Decorator2', (), {'lineno': 4})
        ]
        lineno = 8
    multiple_decorators = MultipleDecorators()
    assert get_first_line_number(multiple_decorators) == 1


# LLM-generated content at query #45
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Function with decorator
    class DecoratedFunc:
        @staticmethod
        def func():
            pass
    
    decorated_node = DecoratedFunc.func.__func__
    assert get_first_line_number(decorated_node) == decorated_node.decorator_list[0].lineno
    
    # Test case 2: Function without decorator
    def simple_func():
        pass
    
    assert get_first_line_number(simple_func) == simple_func.lineno
    
    # Test case 3: Class with decorator
    @staticmethod
    class DecoratedClass:
        pass
    
    assert get_first_line_number(DecoratedClass) == DecoratedClass.decorator_list[0].lineno
    
    # Test case 4: Class without decorator
    class SimpleClass:
        pass
    
    assert get_first_line_number(SimpleClass) == SimpleClass.lineno
    
    # Test case 5: Node with empty decorator_list
    node = SimpleClass
    node.decorator_list = []
    assert get_first_line_number(node) == node.lineno
    
    # Test case 6: Edge case - node with decorator_list attribute set to None
    node = SimpleClass
    node.decorator_list = None
    assert get_first_line_number(node) == node.lineno
```


# LLM-generated content at query #46
#--------------------------

def test_get_first_line_number():
    import ast
    
    # Test with no decorators
    code = "x = 1\n"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1
    
    # Test with decorators
    code = "@decorator\n@another_decorator\ndef func():\n    pass\n"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1
    
    # Test with multiple decorators
    code = "@dec1\n@dec2\n@dec3\ndef func():\n    pass\n"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1
    
    # Test with decorator and other statements
    code = "x = 1\n@decorator\ndef func():\n    pass\n"
    tree = ast.parse(code)
    node = tree.body[1]
    assert get_first_line_number(node) == 2
    
    # Test with class decorators
    code = "@decorator\nclass MyClass:\n    pass\n"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1
    
    # Test with no decorator on class
    code = "class MyClass:\n    pass\n"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1
    
    # Test with decorator on class with preceding code
    code = "x = 1\n@decorator\nclass MyClass:\n    pass\n"
    tree = ast.parse(code)
    node = tree.body[1]
    assert get_first_line_number(node) == 2


# LLM-generated content at query #47
#--------------------------

```python
def test_get_first_line_number():
    # Test with a plain function (no decorators)
    class PlainFunction:
        lineno = 5
        decorator_list = []
    plain_func = PlainFunction()
    assert get_first_line_number(plain_func) == 5

    # Test with a decorated function
    class DecoratorNode:
        lineno = 2
    class DecoratedFunction:
        lineno = 10
        decorator_list = [DecoratorNode()]
    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 2

    # Test with a class that has decorators
    class DecoratedClass:
        lineno = 20
        decorator_list = [DecoratorNode()]
    decorated_class = DecoratedClass()
    assert get_first_line_number(decorated_class) == 2

    # Test with an object that doesn't have decorator_list attribute
    class NoDecorators:
        lineno = 30
    no_decorators = NoDecorators()
    assert get_first_line_number(no_decorators) == 30

    # Test with a class that has no decorators but has decorator_list as empty list
    class EmptyDecorators:
        lineno = 40
        decorator_list = []
    empty_decorators = EmptyDecorators()
    assert get_first_line_number(empty_decorators) == 40
```


# LLM-generated content at query #48
#--------------------------

def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 2})()]
            self.lineno = 10
    
    decorated_node = DecoratorNode()
    assert get_first_line_number(decorated_node) == 2
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 7
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 7


# LLM-generated content at query #49
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    import ast
    code = '''
@decorator
def foo():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with a plain function (no decorators)
    code = '''
def bar():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with a decorated class
    code = '''
@class_decorator
class MyClass:
    pass
'''
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 1

    # Test with a plain class (no decorators)
    code = '''
class MyClass2:
    pass
'''
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 1

    # Test with multiple decorators - should return first decorator's line
    code = '''
@first_decorator
@second_decorator
def baz():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    # The first decorator is on line 1, but the function definition starts on line 3
    assert get_first_line_number(func_node) == 1

    # Test with node that has no decorator_list attribute (e.g., ast.Expr)
    code = '''
x = 5
'''
    tree = ast.parse(code)
    expr_node = tree.body[0]
    assert get_first_line_number(expr_node) == 1
```


# LLM-generated content at query #50
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 3})()]
            self.lineno = 10
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 3
    
    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 5
    
    node_without_decorators = PlainNode()
    assert get_first_line_number(node_without_decorators) == 5
    
    # Test with a node that doesn't have decorator_list attribute
    class MinimalNode:
        def __init__(self):
            self.lineno = 7
    
    minimal_node = MinimalNode()
    assert get_first_line_number(minimal_node) == 7
```


# LLM-generated content at query #51
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function (Python 3.8+ behavior)
    class FakeDecorator:
        lineno = 5
    
    class FakeFunction:
        lineno = 10
        decorator_list = [FakeDecorator()]
    
    node = FakeFunction()
    assert get_first_line_number(node) == 5
    
    # Test without decorators
    class FakeFunctionNoDecorator:
        lineno = 15
        decorator_list = []
    
    node = FakeFunctionNoDecorator()
    assert get_first_line_number(node) == 15
    
    # Test with node that doesn't have decorator_list attribute
    class FakeNode:
        lineno = 20
    
    node = FakeNode()
    assert get_first_line_number(node) == 20
    
    # Test with multiple decorators (should return first)
    class FakeDecorator1:
        lineno = 3
    
    class FakeDecorator2:
        lineno = 4
    
    class FakeFunctionMultiDecorator:
        lineno = 10
        decorator_list = [FakeDecorator1(), FakeDecorator2()]
    
    node = FakeFunctionMultiDecorator()
    assert get_first_line_number(node) == 3


# LLM-generated content at query #52
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has lineno but no decorators
    class SimpleNode:
        lineno = 10
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    decorated_node = DecoratorNode()
    assert get_first_line_number(decorated_node) == 5

    # Test with a node that has empty decorator list
    class EmptyDecoratorNode:
        lineno = 30
        decorator_list = []
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 30

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 40
    no_decorator_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_node) == 40

    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecoratorNode:
        lineno = 50
        decorator_list = [
            type('Decorator1', (), {'lineno': 15}),
            type('Decorator2', (), {'lineno': 25})
        ]
    multiple_decorator_node = MultipleDecoratorNode()
    assert get_first_line_number(multiple_decorator_node) == 15
```


# LLM-generated content at query #53
#--------------------------

def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class SimpleNode:
        def __init__(self):
            self.lineno = 5
            self.decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test case 2: Node with decorators
    class DecoratedNode:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [DecoratorStub()]
    
    class DecoratorStub:
        def __init__(self):
            self.lineno = 3
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 3
    
    # Test case 3: Node with multiple decorators (should return first decorator's line)
    class MultiDecoratedNode:
        def __init__(self):
            self.lineno = 15
            self.decorator_list = [DecoratorStub2(), DecoratorStub3()]
    
    class DecoratorStub2:
        def __init__(self):
            self.lineno = 7
    
    class DecoratorStub3:
        def __init__(self):
            self.lineno = 8
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 7
    
    # Test case 4: Node without decorator_list attribute (Python < 3.8)
    class LegacyNode:
        def __init__(self):
            self.lineno = 20
    
    node = LegacyNode()
    assert get_first_line_number(node) == 20


# LLM-generated content at query #54
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class FakeDecorator:
        lineno = 10
    
    class FakeFunctionDef:
        lineno = 20
        decorator_list = [FakeDecorator()]
    
    node = FakeFunctionDef()
    assert get_first_line_number(node) == 10
    
    # Test without decorators
    class FakeFunctionDefNoDecorators:
        lineno = 30
        decorator_list = []
    
    node = FakeFunctionDefNoDecorators()
    assert get_first_line_number(node) == 30
    
    # Test with a node that doesn't have decorator_list attribute
    class FakeNodeWithoutDecorators:
        lineno = 40
    
    node = FakeNodeWithoutDecorators()
    assert get_first_line_number(node) == 40
    
    # Test with multiple decorators
    class FakeDecorator1:
        lineno = 50
    
    class FakeDecorator2:
        lineno = 60
    
    class FakeFunctionDefMultipleDecorators:
        lineno = 70
        decorator_list = [FakeDecorator1(), FakeDecorator2()]
    
    node = FakeFunctionDefMultipleDecorators()
    assert get_first_line_number(node) == 50
```


# LLM-generated content at query #55
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 15
    
    no_decorator_node = NoDecoratorAttr()
    assert get_first_line_number(no_decorator_node) == 15
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultiDecoratorNode:
        lineno = 20
        decorator_list = [
            type('Decorator1', (), {'lineno': 7})(),
            type('Decorator2', (), {'lineno': 8})()
        ]
    
    multi_decorator_node = MultiDecoratorNode()
    assert get_first_line_number(multi_decorator_node) == 7
    
    # Test with empty decorator_list
    class EmptyDecoratorNode:
        lineno = 25
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 25
```


# LLM-generated content at query #56
#--------------------------

```python
def test_get_first_line_number():
    # Test with a function that has no decorators
    class FakeNodeNoDecorator:
        lineno = 10
    
    node = FakeNodeNoDecorator()
    assert get_first_line_number(node) == 10
    
    # Test with a function that has decorators
    class FakeDecorator:
        lineno = 5
    
    class FakeNodeWithDecorator:
        decorator_list = [FakeDecorator()]
        lineno = 10
    
    node = FakeNodeWithDecorator()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has no decorator_list attribute
    class FakeNodeNoAttribute:
        lineno = 15
    
    node = FakeNodeNoAttribute()
    assert get_first_line_number(node) == 15
    
    # Test with empty decorator_list
    class FakeNodeEmptyDecorator:
        decorator_list = []
        lineno = 20
    
    node = FakeNodeEmptyDecorator()
    assert get_first_line_number(node) == 20
```


# LLM-generated content at query #57
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a function that has decorators
    class DecoratedNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 30
    
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 30
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecoratorsNode:
        lineno = 40
        decorator_list = [
            type('Decorator1', (), {'lineno': 35})(),
            type('Decorator2', (), {'lineno': 38})()
        ]
    
    multi_decorator_node = MultipleDecoratorsNode()
    assert get_first_line_number(multi_decorator_node) == 35
    
    # Test with empty decorator list
    class EmptyDecoratorNode:
        lineno = 50
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 50
```


# LLM-generated content at query #58
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    node = PlainNode()
    assert get_first_line_number(node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 30
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 30
```


# LLM-generated content at query #59
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
        decorator_list = [type('Decorator', (), {'lineno': 10})()]
        lineno = 15
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has multiple decorators (should return first decorator's lineno)
    class MultiDecoratedNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 20})(),
            type('Decorator2', (), {'lineno': 25})()
        ]
        lineno = 30
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 20
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 35
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 35
```


# LLM-generated content at query #60
#--------------------------

def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 15
        end_lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 12})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 12
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 25
        end_lineno = 30
    
    no_decorator_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_node) == 25
    
    # Test with a node that has empty decorator_list (explicitly set)
    class EmptyDecoratorNode:
        lineno = 35
        end_lineno = 40
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 35


# LLM-generated content at query #61
#--------------------------

```python
def test_get_first_line_number():
    class MockNode:
        def __init__(self, lineno, decorator_list=None):
            self.lineno = lineno
            self.decorator_list = decorator_list or []

    node_with_decorator = MockNode(10, [MockNode(5)])
    assert get_first_line_number(node_with_decorator) == 5

    node_without_decorator = MockNode(10)
    assert get_first_line_number(node_without_decorator) == 10

    node_with_empty_decorators = MockNode(10, [])
    assert get_first_line_number(node_with_empty_decorators) == 10

    node_with_multiple_decorators = MockNode(20, [MockNode(1), MockNode(2)])
    assert get_first_line_number(node_with_multiple_decorators) == 1

    node_with_no_lineno = MockNode(None)
    assert get_first_line_number(node_with_no_lineno) is None
```


# LLM-generated content at query #62
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})()]
        lineno = 15
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has no decorators
    class PlainNode:
        decorator_list = []
        lineno = 20
    
    node_without_decorators = PlainNode()
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 25
    
    node_no_decorator_attr = NoDecoratorAttrNode()
    assert get_first_line_number(node_no_decorator_attr) == 25
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        decorator_list = [type('Decorator1', (), {'lineno': 30}), 
                         type('Decorator2', (), {'lineno': 35})]
        lineno = 40
    
    node_multi_decorated = MultiDecoratedNode()
    assert get_first_line_number(node_multi_decorated) == 30
```


# LLM-generated content at query #63
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a decorated function
    class DecoratorNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 15
    
    # Test with a class without decorators
    class ClassNode:
        lineno = 30
        decorator_list = []
    
    node = ClassNode()
    assert get_first_line_number(node) == 30
    
    # Test with a class with multiple decorators
    class MultiDecoratorNode:
        lineno = 40
        decorator_list = [
            type('Decorator1', (), {'lineno': 35})(),
            type('Decorator2', (), {'lineno': 36})()
        ]
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 35
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttribute:
        lineno = 50
    
    node = NoDecoratorAttribute()
    assert get_first_line_number(node) == 50
    
    # Test with a node where decorator_list is None
    class NoneDecorator:
        lineno = 60
        decorator_list = None
    
    node = NoneDecorator()
    assert get_first_line_number(node) == 60
```


# LLM-generated content at query #64
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a function that has decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    node = DecoratedNode()
    assert get_first_line_number(node) == 3

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 7
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 7

    # Test with multiple decorators, should return the first decorator's lineno
    class MultipleDecoratorsNode:
        lineno = 20
        decorator_list = [
            type('Decorator1', (), {'lineno': 15}),
            type('Decorator2', (), {'lineno': 16})
        ]
    node = MultipleDecoratorsNode()
    assert get_first_line_number(node) == 15

    # Test with empty decorator list explicitly
    class EmptyDecoratorListNode:
        lineno = 30
        decorator_list = []
    node = EmptyDecoratorListNode()
    assert get_first_line_number(node) == 30
```


# LLM-generated content at query #65
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunc:
        lineno = 10
        decorator_list = []
    
    simple_func = SimpleFunc()
    assert get_first_line_number(simple_func) == 10
    
    # Test with a function that has decorators
    class DecoratedFunc:
        lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 12})()]
    
    decorated_func = DecoratedFunc()
    assert get_first_line_number(decorated_func) == 12
    
    # Test with an object that doesn't have decorator_list attribute
    class NoDecorators:
        lineno = 20
    
    no_decorators = NoDecorators()
    assert get_first_line_number(no_decorators) == 20
    
    # Test with an object that has empty decorator_list
    class EmptyDecorators:
        lineno = 25
        decorator_list = []
    
    empty_decorators = EmptyDecorators()
    assert get_first_line_number(empty_decorators) == 25
    
    # Test with multiple decorators - should return first decorator's line
    class MultipleDecorators:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 27})(),
            type('Decorator2', (), {'lineno': 28})()
        ]
    
    multiple_decorators = MultipleDecorators()
    assert get_first_line_number(multiple_decorators) == 27
    
    # Test with decorator_list as tuple (in case AST returns tuple)
    class TupleDecorators:
        lineno = 35
        decorator_list = (type('Decorator', (), {'lineno': 33})(),)
    
    tuple_decorators = TupleDecorators()
    assert get_first_line_number(tuple_decorators) == 33
```


# LLM-generated content at query #66
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        pass
    SimpleFunction.lineno = 10
    node = SimpleFunction()
    assert get_first_line_number(node) == 10

    # Test with a function that has decorators
    class DecoratedFunction:
        pass
    DecoratedFunction.lineno = 20
    DecoratedFunction.decorator_list = [type("Decorator", (), {"lineno": 15})()]
    node = DecoratedFunction()
    assert get_first_line_number(node) == 15

    # Test with a class that has decorators
    class DecoratedClass:
        pass
    DecoratedClass.lineno = 30
    DecoratedClass.decorator_list = [type("Decorator", (), {"lineno": 25})()]
    node = DecoratedClass()
    assert get_first_line_number(node) == 25

    # Test with a class without decorators
    class SimpleClass:
        pass
    SimpleClass.lineno = 40
    node = SimpleClass()
    assert get_first_line_number(node) == 40
```


# LLM-generated content at query #67
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function (simulating Python 3.8+ behavior)
    class DecoratedNode:
        def __init__(self):
            self.lineno = 5
            self.decorator_list = [type('Decorator', (), {'lineno': 2})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 2

    # Test with a function without decorators
    class PlainNode:
        def __init__(self):
            self.lineno = 7
            self.decorator_list = []
    
    node = PlainNode()
    assert get_first_line_number(node) == 7

    # Test with a node that has no decorator_list attribute
    class NoDecoratorList:
        def __init__(self):
            self.lineno = 10
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 10

    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecorators:
        def __init__(self):
            self.lineno = 15
            self.decorator_list = [
                type('Decorator', (), {'lineno': 3})(),
                type('Decorator', (), {'lineno': 4})()
            ]
    
    node = MultipleDecorators()
    assert get_first_line_number(node) == 3
```


# LLM-generated content at query #68
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 5
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 10
    
    decorator_node = DecoratorNode()
    
    # Create mock decorator objects with lineno attribute
    class MockDecorator:
        def __init__(self, line):
            self.lineno = line
    
    decorator_node.decorator_list = [MockDecorator(3), MockDecorator(4)]
    assert get_first_line_number(decorator_node) == 3
    
    # Test with empty decorator_list
    decorator_node.decorator_list = []
    assert get_first_line_number(decorator_node) == 10
    
    # Test with node that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 20
    
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 20
```


# LLM-generated content at query #69
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Function without decorators
    class SimpleFunction:
        pass
    SimpleFunction.lineno = 10
    SimpleFunction.decorator_list = []
    assert get_first_line_number(SimpleFunction) == 10

    # Test case 2: Function with decorators
    class DecoratedFunction:
        pass
    DecoratedFunction.lineno = 20
    decorator = type('Decorator', (), {'lineno': 15})()
    DecoratedFunction.decorator_list = [decorator]
    assert get_first_line_number(DecoratedFunction) == 15

    # Test case 3: Class with decorators
    class DecoratedClass:
        pass
    DecoratedClass.lineno = 30
    decorator = type('Decorator', (), {'lineno': 25})()
    DecoratedClass.decorator_list = [decorator]
    assert get_first_line_number(DecoratedClass) == 25

    # Test case 4: Class without decorators
    class SimpleClass:
        pass
    SimpleClass.lineno = 40
    SimpleClass.decorator_list = []
    assert get_first_line_number(SimpleClass) == 40

    # Test case 5: Decorated module-level function
    import ast
    code = "def decorated_func():\n    pass\n"
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    code_with_decorator = "@decorator\ndef decorated_func():\n    pass\n"
    tree = ast.parse(code_with_decorator)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test case 6: Edge case - no decorator_list attribute
    class NoDecoratorList:
        pass
    NoDecoratorList.lineno = 50
    assert get_first_line_number(NoDecoratorList) == 50


# LLM-generated content at query #70
#--------------------------

def test_get_first_line_number():
    # Test with a simple node (no decorators)
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 2})()]
            self.lineno = 8
            self.end_lineno = 15
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 2
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 3
        end_lineno = 7
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 3
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttributeNode:
        lineno = 12
        end_lineno = 20
    
    no_decorator_attr_node = NoDecoratorAttributeNode()
    assert get_first_line_number(no_decorator_attr_node) == 12
    
    # Test with multiple decorators
    class MultipleDecoratorsNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 1})(),
                type('Decorator2', (), {'lineno': 2})(),
                type('Decorator3', (), {'lineno': 3})()
            ]
            self.lineno = 10
            self.end_lineno = 25
    
    multiple_decorators_node = MultipleDecoratorsNode()
    assert get_first_line_number(multiple_decorators_node) == 1


# LLM-generated content at query #71
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 20
        decorator_list = []
    
    node = DecoratorNode()
    node.decorator_list = [type('Decorator', (), {'lineno': 5})()]
    assert get_first_line_number(node) == 5
    
    # Test with an empty decorator list
    node = DecoratorNode()
    node.decorator_list = []
    assert get_first_line_number(node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    node = SimpleNode()
    assert get_first_line_number(node) == 10
```


# LLM-generated content at query #72
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a function that has decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    node = DecoratedNode()
    assert get_first_line_number(node) == 3

    # Test with a class that has multiple decorators
    class MultiDecoratedNode:
        lineno = 20
        decorator_list = [
            type('Decorator1', (), {'lineno': 7}),
            type('Decorator2', (), {'lineno': 8})
        ]
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 7

    # Test with a node that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 15
    node = NoDecoratorList()
    assert get_first_line_number(node) == 15

    # Test with an empty decorator_list
    class EmptyDecoratorList:
        lineno = 25
        decorator_list = []
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 25

    # Test with a class definition
    class ClassNode:
        lineno = 30
        decorator_list = [type('Decorator', (), {'lineno': 28})()]
    node = ClassNode()
    assert get_first_line_number(node) == 28
```


# LLM-generated content at query #73
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 10
    
    # Create a mock decorator with a lineno attribute
    class MockDecorator:
        lineno = 3
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [MockDecorator()]
    assert get_first_line_number(decorated_node) == 3

    # Test with multiple decorators (should return first decorator's lineno)
    decorated_node.decorator_list = [MockDecorator(), MockDecorator()]
    decorated_node.decorator_list[1].lineno = 7
    assert get_first_line_number(decorated_node) == 3

    # Test with empty decorator_list
    empty_decorated_node = DecoratedNode()
    empty_decorated_node.decorator_list = []
    assert get_first_line_number(empty_decorated_node) == 10

    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 15
    no_decorator_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_node) == 15
```


# LLM-generated content at query #74
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    simple_func = SimpleFunction()
    assert get_first_line_number(simple_func) == 5
    
    # Test with a function that has decorators
    class DecoratedFunction:
        lineno = 10
        end_lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 7})()]
    
    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 7
    
    # Test that decorator_list is empty returns the node's lineno
    class NoDecorators:
        lineno = 20
        end_lineno = 25
        decorator_list = []
    
    no_decorators = NoDecorators()
    assert get_first_line_number(no_decorators) == 20
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecorators:
        lineno = 30
        end_lineno = 35
        decorator_list = [
            type('Decorator1', (), {'lineno': 25}),
            type('Decorator2', (), {'lineno': 26}),
            type('Decorator3', (), {'lineno': 27})
        ]
    
    multi_decorators = MultipleDecorators()
    assert get_first_line_number(multi_decorators) == 25
    
    # Test that the function handles missing decorator_list attribute
    class NoDecoratorList:
        lineno = 40
        end_lineno = 45
    
    no_decorator_list = NoDecoratorList()
    assert get_first_line_number(no_decorator_list) == 40
    
    # Test with a class that has decorator_list but it's not a list
    class NonListDecorators:
        lineno = 50
        end_lineno = 55
        decorator_list = None
    
    non_list_decorators = NonListDecorators()
    assert get_first_line_number(non_list_decorators) == 50
```


# LLM-generated content at query #75
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a function that has decorators
    class DecoratorNode:
        def __init__(self):
            self.decorator_list = [DecoratorLine(10), DecoratorLine(12)]
            self.lineno = 15
    
    class DecoratorLine:
        def __init__(self, line):
            self.lineno = line
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 10
    
    # Test with an object that doesn't have decorator_list attribute
    class NoDecoratorsNode:
        lineno = 20
    
    node = NoDecoratorsNode()
    assert get_first_line_number(node) == 20
    
    # Test with empty decorator_list
    class EmptyDecoratorsNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 25
    
    node = EmptyDecoratorsNode()
    assert get_first_line_number(node) == 25
    
    # Test with multiple decorators - should return first decorator's line number
    class MultiDecoratorNode:
        def __init__(self):
            self.decorator_list = [DecoratorLine(30), DecoratorLine(32), DecoratorLine(35)]
            self.lineno = 40
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 30


# LLM-generated content at query #76
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 8
        end_lineno = 12
    node_with_decorators = DecoratorNode()
    
    # Create mock decorator objects with lineno attribute
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_with_decorators.decorator_list = [MockDecorator(3), MockDecorator(4)]
    assert get_first_line_number(node_with_decorators) == 3

    # Test with node that has empty decorator_list
    node_with_empty_decorators = SimpleNode()
    node_with_empty_decorators.decorator_list = []
    assert get_first_line_number(node_with_empty_decorators) == 5

    # Test with node that doesn't have decorator_list attribute
    node_without_decorator_attr = SimpleNode()
    assert get_first_line_number(node_without_decorator_attr) == 5

    # Test with a node that has decorator_list but no decorators
    node_with_none_decorator = SimpleNode()
    node_with_none_decorator.decorator_list = None
    assert get_first_line_number(node_with_none_decorator) == 5

    # Test with a node where decorator_list exists but is empty
    node_with_empty_list = SimpleNode()
    node_with_empty_list.decorator_list = []
    assert get_first_line_number(node_with_empty_list) == 5

    # Test with a real AST node
    import ast
    code = """
@decorator
def function():
    pass
"""
    tree = ast.parse(code)
    function_node = tree.body[0]
    assert get_first_line_number(function_node) == 2  # line of decorator

    # Test with a real AST node without decorators
    code2 = """
def plain_function():
    pass
"""
    tree2 = ast.parse(code2)
    plain_function_node = tree2.body[0]
    assert get_first_line_number(plain_function_node) == 2  # line of def
```


# LLM-generated content at query #77
#--------------------------

def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 10
        end_lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 3
    
    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        lineno = 10
        end_lineno = 15
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 1
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorListAttr:
        lineno = 20
        end_lineno = 25
    
    node = NoDecoratorListAttr()
    assert get_first_line_number(node) == 20


# LLM-generated content at query #78
#--------------------------

def test_get_first_line_number():
    # Test with a function that has decorators
    class FakeDecorator:
        lineno = 10
    
    class FakeNodeWithDecorators:
        decorator_list = [FakeDecorator()]
        lineno = 20
    
    node_with_decorators = FakeNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has no decorators
    class FakeNodeWithoutDecorators:
        decorator_list = []
        lineno = 30
    
    node_without_decorators = FakeNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 30
    
    # Test with a node that doesn't have decorator_list attribute
    class FakeNodeNoDecoratorAttr:
        lineno = 40
    
    node_no_decorator_attr = FakeNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 40


# LLM-generated content at query #79
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Node with decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 3

    # Test case 3: Node with multiple decorators
    class MultiDecoratorNode:
        lineno = 20
        decorator_list = [
            type('Decorator1', (), {'lineno': 7}),
            type('Decorator2', (), {'lineno': 8})
        ]
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 7

    # Test case 4: Node without decorator_list attribute
    class NoDecoratorList:
        lineno = 15
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 15

    # Test case 5: Node with empty decorator_list
    class EmptyDecorator:
        lineno = 25
        decorator_list = []
    
    node = EmptyDecorator()
    assert get_first_line_number(node) == 25
```


# LLM-generated content at query #80
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.lineno = 20
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 20
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 30
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 30
```


# LLM-generated content at query #81
#--------------------------

def test_get_first_line_number():
    # Test case 1: Node with decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    node = DecoratedNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Node without decorators
    class PlainNode:
        decorator_list = []
        lineno = 20
    node = PlainNode()
    assert get_first_line_number(node) == 20

    # Test case 3: Node with no decorator_list attribute
    class NoDecoratorList:
        lineno = 30
    node = NoDecoratorList()
    assert get_first_line_number(node) == 30

    # Test case 4: Node with empty decorator_list
    class EmptyDecoratorList:
        decorator_list = []
        lineno = 40
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 40

    # Test case 5: Multiple decorators, should return first decorator's lineno
    class MultipleDecorators:
        decorator_list = [
            type('Decorator1', (), {'lineno': 50})(),
            type('Decorator2', (), {'lineno': 60})()
        ]
        lineno = 70
    node = MultipleDecorators()
    assert get_first_line_number(node) == 50


# LLM-generated content at query #82
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 5
        end_lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 2})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 2
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 5
        end_lineno = 10
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 1
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 7
        end_lineno = 12
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 7
```


# LLM-generated content at query #83
#--------------------------

```python
def test_get_first_line_number():
    # Test with a function that has no decorators
    class SimpleFunction:
        pass
    simple_func = SimpleFunction()
    simple_func.lineno = 10
    simple_func.decorator_list = []
    assert get_first_line_number(simple_func) == 10

    # Test with a function that has decorators
    class DecoratedFunction:
        pass
    decorated_func = DecoratedFunction()
    decorated_func.lineno = 20
    decorator1 = type('Decorator', (), {'lineno': 15})()
    decorator2 = type('Decorator', (), {'lineno': 16})()
    decorated_func.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(decorated_func) == 15

    # Test with a function that has no lineno attribute
    class NoLineNumber:
        pass
    no_line_func = NoLineNumber()
    no_line_func.decorator_list = []
    try:
        get_first_line_number(no_line_func)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with a class that has no decorator_list attribute
    class NoDecoratorList:
        pass
    no_decorator_func = NoDecoratorList()
    no_decorator_func.lineno = 30
    assert get_first_line_number(no_decorator_func) == 30

    # Test with a function that has decorators but no lineno on decorator
    class DecoratorNoLine:
        pass
    decorator_no_line = DecoratorNoLine()
    try:
        get_first_line_number(decorator_no_line)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass
```


# LLM-generated content at query #84
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a function that has decorators
    class DecoratorNode:
        lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 20
    
    # Test with empty decorator_list
    class EmptyDecoratorListNode:
        lineno = 25
        decorator_list = []
    
    node = EmptyDecoratorListNode()
    assert get_first_line_number(node) == 25
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecoratorsNode:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 7})(),
            type('Decorator2', (), {'lineno': 8})()
        ]
    
    node = MultipleDecoratorsNode()
    assert get_first_line_number(node) == 7
```


# LLM-generated content at query #85
#--------------------------

```python
def test_get_first_line_number():
    # Test a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test a function with decorators
    class DecoratedNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultiDecoratedNode:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1
    
    # Test with empty decorator_list (explicitly set)
    class EmptyDecoratorNode:
        lineno = 40
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 40
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 50
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 50
    
    # Test with decorator_list set to None
    class NoneDecoratorNode:
        lineno = 60
        decorator_list = None
    
    none_decorator_node = NoneDecoratorNode()
    assert get_first_line_number(none_decorator_node) == 60
```


# LLM-generated content at query #86
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function node (no decorators)
    class MockNode:
        lineno = 10
        decorator_list = []
    
    node = MockNode()
    assert get_first_line_number(node) == 10
    
    # Test with a function node that has decorators
    class MockDecorator:
        lineno = 5
    
    class MockDecoratedNode:
        lineno = 10
        decorator_list = [MockDecorator()]
    
    decorated_node = MockDecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that doesn't have decorator_list attribute
    class SimpleNode:
        lineno = 20
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 20
    
    # Test with empty decorator_list
    class EmptyDecoratorNode:
        lineno = 30
        decorator_list = []
    
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 30
    
    # Test with multiple decorators - should return first decorator's line
    class MultiDecoratorNode:
        lineno = 50
        decorator_list = [MockDecorator(), MockDecorator()]
    
    multi_node = MultiDecoratorNode()
    assert get_first_line_number(multi_node) == 5
```


# LLM-generated content at query #87
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple AST node (no decorators)
    import ast
    code = "x = 1"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1
    
    # Test with a decorated function
    code = """
@decorator
def foo():
    pass
"""
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1  # decorator line
    
    # Test with multiple decorators
    code = """
@decorator1
@decorator2
def bar():
    pass
"""
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1  # first decorator line
    
    # Test with decorator on class
    code = """
@class_decorator
class MyClass:
    pass
"""
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1  # decorator line
    
    # Test with function defined after some lines
    code = """
# comment
x = 1

@decorator
def baz():
    pass
"""
    tree = ast.parse(code)
    node = tree.body[1]
    assert get_first_line_number(node) == 5  # decorator line number
    
    # Test with node that has no decorator_list attribute
    code = "print('hello')"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1  # just the node's lineno
```


# LLM-generated content at query #88
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list.append(type('Decorator', (), {'lineno': 15})())
    assert get_first_line_number(decorated_node) == 15
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 30
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 30
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorListNode:
        lineno = 40
    
    no_decorator_list_node = NoDecoratorListNode()
    assert get_first_line_number(no_decorator_list_node) == 40
```


# LLM-generated content at query #89
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node without decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = []
            
    node = DecoratedNode()
    decorator1 = SimpleNode()
    decorator1.lineno = 5
    decorator2 = SimpleNode()
    decorator2.lineno = 7
    node.decorator_list = [decorator1, decorator2]
    node.lineno = 10
    
    assert get_first_line_number(node) == 5
    
    # Test with empty decorator list (treated as no decorators)
    node2 = DecoratedNode()
    node2.decorator_list = []
    node2.lineno = 20
    assert get_first_line_number(node2) == 20
```


# LLM-generated content at query #90
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.lineno = 5
            self.decorator_list = [type('Decorator', (), {'lineno': 2})()]
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 2
    
    # Test with a node without decorators
    class SimpleNode:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []
    
    node_without_decorators = SimpleNode()
    assert get_first_line_number(node_without_decorators) == 10
    
    # Test with a node missing decorator_list attribute entirely
    class MinimalNode:
        def __init__(self):
            self.lineno = 3
    
    minimal_node = MinimalNode()
    assert get_first_line_number(minimal_node) == 3
    
    # Test with empty decorator_list
    class EmptyDecoratorNode:
        def __init__(self):
            self.lineno = 7
            self.decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 7


# LLM-generated content at query #91
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function node without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 15
    
    # Test with multiple decorators (should return first decorator's line)
    class MultiDecoratorNode:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 25})(),
            type('Decorator2', (), {'lineno': 26})()
        ]
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 25
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 40
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 40
```


# LLM-generated content at query #92
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Simple function without decorators
    class SimpleFunction:
        lineno = 5
        decorator_list = []
    
    simple_func = SimpleFunction()
    assert get_first_line_number(simple_func) == 5
    
    # Test case 2: Function with decorators
    class DecoratedFunction:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 3
    
    # Test case 3: Function with multiple decorators (should return first decorator's line)
    class MultiDecoratedFunction:
        lineno = 20
        decorator_list = [
            type('Decorator1', (), {'lineno': 7})(),
            type('Decorator2', (), {'lineno': 8})()
        ]
    
    multi_decorated = MultiDecoratedFunction()
    assert get_first_line_number(multi_decorated) == 7
    
    # Test case 4: Node without decorator_list attribute
    class NoDecoratorList:
        lineno = 15
    
    no_decorator = NoDecoratorList()
    assert get_first_line_number(no_decorator) == 15
    
    # Test case 5: Edge case - empty decorator list
    class EmptyDecoratorList:
        lineno = 25
        decorator_list = []
    
    empty_decorator = EmptyDecoratorList()
    assert get_first_line_number(empty_decorator) == 25
```


# LLM-generated content at query #93
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 2})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 2
    
    # Test with a node that has an empty decorator_list (should use lineno)
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 15
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttributeNode:
        lineno = 20
    
    no_decorator_node = NoDecoratorAttributeNode()
    assert get_first_line_number(no_decorator_node) == 20


# LLM-generated content at query #94
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
        end_lineno = 15
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorator_list but it's empty
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 20
        end_lineno = 25
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 30
        end_lineno = 35
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 30
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecoratorsNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 2})(),
            type('Decorator2', (), {'lineno': 3})()
        ]
        lineno = 10
        end_lineno = 15
    
    node = MultipleDecoratorsNode()
    assert get_first_line_number(node) == 2
```


# LLM-generated content at query #95
#--------------------------

def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 7
    
    node = DecoratedNode()
    node.decorator_list.append(type('Decorator', (), {'lineno': 3})())
    assert get_first_line_number(node) == 3
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 12
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 12
    
    # Test with a node that has decorator_list attribute but it's None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 15
    
    node = NoneDecoratorNode()
    assert get_first_line_number(node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 20


# LLM-generated content at query #96
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
    
    decorator1 = type('Decorator', (), {'lineno': 15})()
    decorator2 = type('Decorator', (), {'lineno': 18})()
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(decorated_node) == 15

    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 30
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 30

    # Test with a node that has decorator_list attribute set to None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 40
    none_decorator_node = NoneDecoratorNode()
    assert get_first_line_number(none_decorator_node) == 40

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 50
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 50
```


# LLM-generated content at query #97
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a function that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 3})]
        lineno = 7
    node = DecoratedNode()
    assert get_first_line_number(node) == 3

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 10
    node = NoDecoratorList()
    assert get_first_line_number(node) == 10

    # Test with empty decorator_list
    class EmptyDecoratorList:
        decorator_list = []
        lineno = 12
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 12

    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecorators:
        decorator_list = [
            type('Decorator1', (), {'lineno': 2}),
            type('Decorator2', (), {'lineno': 4})
        ]
        lineno = 8
    node = MultipleDecorators()
    assert get_first_line_number(node) == 2


# LLM-generated content at query #98
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    import ast
    code = "def foo():\n    pass"
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with a decorated function
    code = "@decorator\ndef foo():\n    pass"
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with multiple decorators
    code = "@decorator1\n@decorator2\ndef foo():\n    pass"
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with a class without decorators
    code = "class Foo:\n    pass"
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 1

    # Test with a decorated class
    code = "@decorator\nclass Foo:\n    pass"
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 1

    # Test with code on different lines
    code = "x = 1\n\ndef foo():\n    pass"
    tree = ast.parse(code)
    func_node = tree.body[1]
    assert get_first_line_number(func_node) == 3

    # Test with decorated function on different lines
    code = "x = 1\n\n@decorator\ndef foo():\n    pass"
    tree = ast.parse(code)
    func_node = tree.body[1]
    assert get_first_line_number(func_node) == 4

    # Test with decorator on multiple lines
    code = "x = 1\n\n@decorator1\n@decorator2\ndef foo():\n    pass"
    tree = ast.parse(code)
    func_node = tree.body[1]
    assert get_first_line_number(func_node) == 4


# LLM-generated content at query #99
#--------------------------

```python
def test_get_first_line_number():
    # Test with a plain function node (no decorators)
    class MockNode:
        lineno = 10
        end_lineno = 20
        decorator_list = []
    
    node = MockNode()
    assert get_first_line_number(node) == 10
    
    # Test with a decorated function node
    class MockDecorator:
        lineno = 5
    
    class MockDecoratedNode:
        lineno = 10
        end_lineno = 20
        decorator_list = [MockDecorator()]
    
    decorated_node = MockDecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecorators:
        lineno = 15
        end_lineno = 25
    
    node_no_decorators = MockNodeNoDecorators()
    assert get_first_line_number(node_no_decorators) == 15
    
    # Test with multiple decorators (should return first decorator's lineno)
    class MockDecorator2:
        lineno = 8
    
    class MockMultiDecoratedNode:
        lineno = 10
        end_lineno = 20
        decorator_list = [MockDecorator(), MockDecorator2()]
    
    multi_decorated_node = MockMultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 5
```


# LLM-generated content at query #100
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
        lineno = 7
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 3
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 10
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorator_list attribute set to None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 15
    
    node = NoneDecoratorNode()
    assert get_first_line_number(node) == 15
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 8
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        end_lineno = 15
        decorator_list = [SimpleNode()]
    
    node_with_decorator = DecoratedNode()
    node_with_decorator.decorator_list[0].lineno = 3  # Set decorator's line number
    assert get_first_line_number(node_with_decorator) == 3
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 20
        end_lineno = 25
        decorator_list = [SimpleNode(), SimpleNode()]
    
    node_multi = MultiDecoratedNode()
    node_multi.decorator_list[0].lineno = 2
    node_multi.decorator_list[1].lineno = 4
    assert get_first_line_number(node_multi) == 2
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttr:
        lineno = 30
        end_lineno = 35
    
    node_no_attr = NoDecoratorAttr()
    assert get_first_line_number(node_no_attr) == 30


# LLM-generated content at query #2
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a function that has decorators
    class DecoratedNode:
        lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 20
    
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 20
    
    # Test with empty decorator list
    class EmptyDecoratorNode:
        lineno = 25
        decorator_list = []
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 25


# LLM-generated content at query #3
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_node = DecoratorNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        lineno = 15
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 20
```


# LLM-generated content at query #4
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has lineno but no decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 8
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        end_lineno = 15
        decorator_list = [
            type('Decorator', (), {'lineno': 7})(),
            type('Decorator', (), {'lineno': 8})()
        ]
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 7

    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        lineno = 20
        end_lineno = 25
        decorator_list = []
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 20

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 30
        end_lineno = 35
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 30

    # Test with a node that has 'decorator_list' set to None
    class NoneDecoratorNode:
        lineno = 40
        end_lineno = 45
        decorator_list = None
    none_decorator_node = NoneDecoratorNode()
    assert get_first_line_number(none_decorator_node) == 40
```


# LLM-generated content at query #5
#--------------------------

def test_get_first_line_number():
    # Test node without decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test node with decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 10
    
    decorator1 = type('Decorator', (), {'lineno': 3})()
    decorator2 = type('Decorator', (), {'lineno': 2})()
    node = DecoratedNode()
    node.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node) == 3  # Returns first decorator's line

    # Test node with empty decorator_list
    node = DecoratedNode()
    node.decorator_list = []
    assert get_first_line_number(node) == 10  # Returns node's own lineno

    # Test node with decorator_list attribute set to None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 15
    node = NoneDecoratorNode()
    assert get_first_line_number(node) == 15  # Falls back to node.lineno

    # Test node without decorator_list attribute
    class NoDecoratorAttr:
        lineno = 20
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 20  # Falls back to node.lineno


# LLM-generated content at query #6
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator1', (), {'lineno': 5}), type('Decorator2', (), {'lineno': 6})]
        lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15


# LLM-generated content at query #7
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 7
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 2})(),
            type('Decorator', (), {'lineno': 3})()
        ]
        lineno = 10
        end_lineno = 12
    
    decorated_node = DecoratorNode()
    assert get_first_line_number(decorated_node) == 2

    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
        end_lineno = 18
    
    empty_decorated_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorated_node) == 15

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
        end_lineno = 22
    
    no_decorator_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_node) == 20

    # Test with a real AST node for a decorated function
    import ast
    code = """
@decorator
def foo():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # decorator line

    # Test with a real AST node for an undecorated function
    code2 = """
def bar():
    pass
"""
    tree2 = ast.parse(code2)
    func_node2 = tree2.body[0]
    assert get_first_line_number(func_node2) == 2  # function definition line
```


# LLM-generated content at query #8
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 20
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorListNode:
        def __init__(self):
            self.lineno = 30
    
    no_decorator_node = NoDecoratorListNode()
    assert get_first_line_number(no_decorator_node) == 30
    
    # Test with multiple decorators - should return the first decorator's lineno
    class MultiDecoratedNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 1})(),
                type('Decorator2', (), {'lineno': 2})()
            ]
            self.lineno = 3
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1
```


# LLM-generated content at query #9
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 20
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has empty decorator_list attribute
    class EmptyDecoratorNode:
        lineno = 15
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 25
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 25
```


# LLM-generated content at query #10
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        lineno = 10
        end_lineno = 15
        decorator_list = []
    
    node = SimpleFunction()
    assert get_first_line_number(node) == 10
    
    # Test with a function that has decorators
    class DecoratedFunction:
        lineno = 20
        end_lineno = 25
        decorator_list = [type('Decorator', (), {'lineno': 18})()]
    
    node = DecoratedFunction()
    assert get_first_line_number(node) == 18
    
    # Test with a class that has decorators
    class DecoratedClass:
        lineno = 30
        end_lineno = 35
        decorator_list = [type('Decorator', (), {'lineno': 28})()]
    
    node = DecoratedClass()
    assert get_first_line_number(node) == 28
    
    # Test with an object that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 40
        end_lineno = 45
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 40
    
    # Test with multiple decorators - should return the first decorator's line
    class MultipleDecorators:
        lineno = 50
        end_lineno = 55
        decorator_list = [
            type('Decorator1', (), {'lineno': 44}),
            type('Decorator2', (), {'lineno': 46})
        ]
    
    node = MultipleDecorators()
    assert get_first_line_number(node) == 44


# LLM-generated content at query #11
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 3})]
        lineno = 7
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 3
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 10
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorator_list attribute but it's None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 12
    
    node = NoneDecoratorNode()
    assert get_first_line_number(node) == 12


# LLM-generated content at query #12
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})]
        lineno = 15
    
    decorator_node = DecoratorNode()
    assert get_first_line_number(decorator_node) == 10
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 20
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 20
    
    # Test with an actual AST node
    import ast
    code = """
@decorator
def foo():
    pass
"""
    tree = ast.parse(code)
    function_node = tree.body[0]
    assert get_first_line_number(function_node) == 2  # line of decorator
    
    # Test with a node without decorator_list attribute
    class NoDecoratorAttr:
        lineno = 25
    
    no_decorator_attr_node = NoDecoratorAttr()
    assert get_first_line_number(no_decorator_attr_node) == 25
```


# LLM-generated content at query #13
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Simple node without decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test case 2: Node with decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5

    # Test case 3: Node with empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 20
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 20

    # Test case 4: Node without decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 30
    no_decorator_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_node) == 30
```


# LLM-generated content at query #14
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 2})()]
            self.lineno = 5
            self.end_lineno = 10
    
    decorated_node = DecoratorNode()
    assert get_first_line_number(decorated_node) == 2
    
    # Test with empty decorator_list
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 7
            self.end_lineno = 12
    
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 7
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 3
        end_lineno = 8
    
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 3
```


# LLM-generated content at query #15
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class DecoratedFunction:
        pass
    
    # Create a mock node with decorator_list
    class MockNode:
        def __init__(self, lineno, decorator_list=None):
            self.lineno = lineno
            self.decorator_list = decorator_list or []
    
    # Test with decorators - should return first decorator's line number
    decorator1 = MockNode(lineno=10)
    decorator2 = MockNode(lineno=15)
    node_with_decorators = MockNode(lineno=20, decorator_list=[decorator1, decorator2])
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test without decorators - should return node's own line number
    node_without_decorators = MockNode(lineno=30)
    assert get_first_line_number(node_without_decorators) == 30
    
    # Test with empty decorator list
    node_empty_decorators = MockNode(lineno=40, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 40
    
    # Test with AST-like node (using actual ast module structure)
    import ast
    code = """
@decorator
def my_function():
    pass
"""
    tree = ast.parse(code)
    function_node = tree.body[0]
    assert get_first_line_number(function_node) == 1
    
    # Test with plain function without decorators
    code2 = """
def plain_function():
    pass
"""
    tree2 = ast.parse(code2)
    function_node2 = tree2.body[0]
    assert get_first_line_number(function_node2) == 1
```


# LLM-generated content at query #16
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})]
        lineno = 5
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node without decorators
    class PlainNode:
        decorator_list = []
        lineno = 15
    
    node_without_decorators = PlainNode()
    assert get_first_line_number(node_without_decorators) == 15
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 20
    
    node_no_decorator_list = NoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 20
    
    # Test with an empty decorator_list
    class EmptyDecoratorList:
        decorator_list = []
        lineno = 25
    
    node_empty_decorators = EmptyDecoratorList()
    assert get_first_line_number(node_empty_decorators) == 25
    
    # Test with multiple decorators (should return first decorator's line)
    class MultipleDecorators:
        decorator_list = [
            type('Decorator1', (), {'lineno': 30}),
            type('Decorator2', (), {'lineno': 31})
        ]
        lineno = 28
    
    node_multiple_decorators = MultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 30
```


# LLM-generated content at query #17
#--------------------------

```python
def test_get_first_line_number():
    # Test with a function that has decorators
    class MockDecorator:
        lineno = 5
    
    class MockFunctionWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 10
    
    node_with_decorators = MockFunctionWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with a node that has no decorators
    class MockFunctionWithoutDecorators:
        decorator_list = []
        lineno = 15
    
    node_without_decorators = MockFunctionWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class MockSimpleNode:
        lineno = 20
    
    simple_node = MockSimpleNode()
    assert get_first_line_number(simple_node) == 20
    
    # Test with multiple decorators - should return first decorator's lineno
    class MockDecorator1:
        lineno = 25
    
    class MockDecorator2:
        lineno = 30
    
    class MockFunctionMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 35
    
    node_multiple_decorators = MockFunctionMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 25


# LLM-generated content at query #18
#--------------------------

def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 10
    
    # Create a mock decorator with a lineno attribute
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [MockDecorator(3), MockDecorator(4)]
    assert get_first_line_number(decorated_node) == 3

    # Test with a node that has an empty decorator_list (should return node.lineno)
    class EmptyDecoratorsNode:
        decorator_list = []
        lineno = 15
    empty_decorators_node = EmptyDecoratorsNode()
    assert get_first_line_number(empty_decorators_node) == 15

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorsAttr:
        lineno = 20
    no_decorators_attr = NoDecoratorsAttr()
    assert get_first_line_number(no_decorators_attr) == 20


# LLM-generated content at query #19
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        lineno = 10
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 15
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 10

    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 20
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20

    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 25
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 25
```


# LLM-generated content at query #20
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 20
    
    decorator1 = type('Decorator', (), {'lineno': 5})()
    decorator2 = type('Decorator', (), {'lineno': 7})()
    
    decorated_node = DecoratorNode()
    decorated_node.decorator_list = [decorator1, decorator2]
    
    assert get_first_line_number(decorated_node) == 5
    
    # Test with empty decorator_list
    empty_decorator_node = DecoratorNode()
    empty_decorator_node.decorator_list = []
    
    assert get_first_line_number(empty_decorator_node) == 20
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 30
    
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 30
```


# LLM-generated content at query #21
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    node_with_decorator = DecoratedNode()
    assert get_first_line_number(node_with_decorator) == 5
    
    # Test case 2: Node without decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    node_without_decorator = PlainNode()
    assert get_first_line_number(node_without_decorator) == 20
    
    # Test case 3: Node without decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 30
    
    node_no_decorator_attr = NoDecoratorAttrNode()
    assert get_first_line_number(node_no_decorator_attr) == 30
    
    # Test case 4: Empty decorator list
    class EmptyDecoratorListNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 40
    
    node_empty_decorator = EmptyDecoratorListNode()
    assert get_first_line_number(node_empty_decorator) == 40
```


# LLM-generated content at query #22
#--------------------------

```python
def test_get_first_line_number():
    # Test case for a function with decorators
    node = type('Node', (), {
        'decorator_list': [type('Decorator', (), {'lineno': 10})()],
        'lineno': 5
    })()
    assert get_first_line_number(node) == 10

    # Test case for a function without decorators
    node = type('Node', (), {
        'decorator_list': [],
        'lineno': 7
    })()
    assert get_first_line_number(node) == 7

    # Test case for a node without decorator_list attribute
    node = type('Node', (), {
        'lineno': 3
    })()
    assert get_first_line_number(node) == 3

    # Test case with multiple decorators
    node = type('Node', (), {
        'decorator_list': [
            type('Decorator', (), {'lineno': 15})(),
            type('Decorator', (), {'lineno': 16})()
        ],
        'lineno': 20
    })()
    assert get_first_line_number(node) == 15
```


# LLM-generated content at query #23
#--------------------------

def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 10
    
    # Create a mock decorator with a lineno attribute
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [MockDecorator(3), MockDecorator(4)]
    decorated_node.lineno = 10
    assert get_first_line_number(decorated_node) == 3

    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 20


# LLM-generated content at query #24
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})]
        lineno = 20
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test case 2: Node without decorators
    class PlainNode:
        decorator_list = []
        lineno = 30
    
    node_without_decorators = PlainNode()
    assert get_first_line_number(node_without_decorators) == 30
    
    # Test case 3: Node without decorator_list attribute
    class SimpleNode:
        lineno = 40
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 40
    
    # Test case 4: Multiple decorators - should return first decorator's lineno
    class MultiDecoratedNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 50}),
            type('Decorator2', (), {'lineno': 60})
        ]
        lineno = 70
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 50
```


# LLM-generated content at query #25
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
    assert get_first_line_number(node_with_decorators) == 5

    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 20
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20

    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 30
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 30

    # Test with multiple decorators - should return first decorator's lineno
    class MockDecorator1:
        lineno = 1
    
    class MockDecorator2:
        lineno = 2
    
    class MockNodeMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 15
    
    node_multiple_decorators = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 1
```


# LLM-generated content at query #26
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    node_with_decorator = DecoratedNode()
    assert get_first_line_number(node_with_decorator) == 5
    
    # Test with a node without decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    node_without_decorator = PlainNode()
    assert get_first_line_number(node_without_decorator) == 20
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 30
    
    node_no_attr = NoDecoratorAttrNode()
    assert get_first_line_number(node_no_attr) == 30
    
    # Test with a node that has decorator_list as None
    class NoneDecoratorNode:
        def __init__(self):
            self.decorator_list = None
            self.lineno = 40
    
    node_none_decorator = NoneDecoratorNode()
    assert get_first_line_number(node_none_decorator) == 40
```


# LLM-generated content at query #27
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class DecoratedFunc:
        @staticmethod
        def method():
            pass
    
    decorated_node = DecoratedFunc.method.__func__
    # Simulate the AST node with decorator_list
    import ast
    tree = ast.parse("@staticmethod\ndef method():\n    pass")
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1
    
    # Test with a simple function without decorators
    tree2 = ast.parse("def simple():\n    pass")
    func_node2 = tree2.body[0]
    assert get_first_line_number(func_node2) == 1
    
    # Test with a class without decorators
    tree3 = ast.parse("class MyClass:\n    pass")
    class_node = tree3.body[0]
    assert get_first_line_number(class_node) == 1
    
    # Test with a decorated class
    tree4 = ast.parse("@decorator\nclass MyDecoratedClass:\n    pass")
    class_node4 = tree4.body[0]
    assert get_first_line_number(class_node4) == 1
```


# LLM-generated content at query #28
#--------------------------

```python
def test_get_first_line_number():
    # Test with a function that has decorators
    source = '''
@decorator1
@decorator2
def my_function():
    pass
'''
    tree = ast.parse(source)
    func_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    assert get_first_line_number(func_node) == 2  # line of first decorator

    # Test with a function without decorators
    source = '''
def my_function():
    pass
'''
    tree = ast.parse(source)
    func_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    assert get_first_line_number(func_node) == 2  # line of function definition

    # Test with a class that has decorators
    source = '''
@decorator
class MyClass:
    pass
'''
    tree = ast.parse(source)
    class_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    assert get_first_line_number(class_node) == 2  # line of first decorator

    # Test with a class without decorators
    source = '''
class MyClass:
    pass
'''
    tree = ast.parse(source)
    class_node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    assert get_first_line_number(class_node) == 2  # line of class definition

    # Test with a node that has no decorator_list attribute (e.g., assignment)
    source = '''
x = 5
'''
    tree = ast.parse(source)
    assign_node = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign))
    assert get_first_line_number(assign_node) == 2  # line of assignment
```


# LLM-generated content at query #29
#--------------------------

def test_get_first_line_number():
    # Test case 1: Node without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test case 2: Node with decorators
    class DecoratedNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 15
    
    # Test case 3: Node with multiple decorators
    class MultiDecoratedNode:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 21}),
            type('Decorator2', (), {'lineno': 22})
        ]
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 21
    
    # Test case 4: Node with decorator but no lineno attribute
    class NodeWithoutLineno:
        decorator_list = [type('Decorator', (), {'lineno': 25})()]
    
    node_without_lineno = NodeWithoutLineno()
    assert get_first_line_number(node_without_lineno) == 25


# LLM-generated content at query #30
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 2})()]
            self.lineno = 7
            self.end_lineno = 12
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 2

    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 1})(),
                type('Decorator2', (), {'lineno': 3})()
            ]
            self.lineno = 10
            self.end_lineno = 15
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 1

    # Test with a node that has an empty decorator_list attribute
    class EmptyDecoratorListNode:
        decorator_list = []
        lineno = 20
    
    node = EmptyDecoratorListNode()
    assert get_first_line_number(node) == 20

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorListAttr:
        lineno = 30
    
    node = NoDecoratorListAttr()
    assert get_first_line_number(node) == 30
```


# LLM-generated content at query #31
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
        decorator_list.append(type('Decorator', (), {'lineno': 5})())
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 30
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 30
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 40
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 40
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultiDecoratorNode:
        decorator_list = []
        lineno = 50
        decorator_list.append(type('Decorator1', (), {'lineno': 1})())
        decorator_list.append(type('Decorator2', (), {'lineno': 2})())
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 1
```


# LLM-generated content at query #32
#--------------------------

def test_get_first_line_number():
    # Test function without decorators
    class FuncNode:
        lineno = 5
        decorator_list = []
    node = FuncNode()
    assert get_first_line_number(node) == 5

    # Test function with decorators
    class DecoratedFuncNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    node = DecoratedFuncNode()
    assert get_first_line_number(node) == 3

    # Test function with multiple decorators, should return first decorator's line
    class MultiDecoratedFuncNode:
        lineno = 15
        decorator_list = [
            type('Decorator1', (), {'lineno': 7})(),
            type('Decorator2', (), {'lineno': 8})()
        ]
    node = MultiDecoratedFuncNode()
    assert get_first_line_number(node) == 7

    # Test class with decorators
    class DecoratedClassNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 12})()]
    node = DecoratedClassNode()
    assert get_first_line_number(node) == 12

    # Test edge case: node without decorator_list attribute
    class NoDecoratorList:
        lineno = 25
    node = NoDecoratorList()
    assert get_first_line_number(node) == 25


# LLM-generated content at query #33
#--------------------------

def test_get_first_line_number():
    # Test with a simple node (no decorators)
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})()]
        lineno = 15
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 20
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 20
    
    # Test with a node that has multiple decorators (should return first decorator's lineno)
    class MultiDecoratorNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 25}),
            type('Decorator2', (), {'lineno': 30})
        ]
        lineno = 35
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 25
    
    # Test with a node that has a decorator but no lineno attribute on node itself
    class DecoratorNoLinenoNode:
        decorator_list = [type('Decorator', (), {'lineno': 40})()]
    
    node = DecoratorNoLinenoNode()
    assert get_first_line_number(node) == 40


# LLM-generated content at query #34
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
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
        lineno = 10
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3

    # Test with a node that has multiple decorators (should return first decorator's lineno)
    class MultiDecoratedNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 2})(),
            type('Decorator2', (), {'lineno': 4})()
        ]
        lineno = 8
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 2

    # Test with an empty decorator_list
    class EmptyDecoratorListNode:
        decorator_list = []
        lineno = 7
    empty_decorator_node = EmptyDecoratorListNode()
    assert get_first_line_number(empty_decorator_node) == 7
```


# LLM-generated content at query #35
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})]
        lineno = 10
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorsNode:
        decorator_list = []
        lineno = 15
    
    node = EmptyDecoratorsNode()
    assert get_first_line_number(node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttributeNode:
        lineno = 20
    
    node = NoDecoratorAttributeNode()
    assert get_first_line_number(node) == 20
    
    # Test with multiple decorators - should return the first one
    class MultipleDecoratorsNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 3}),
            type('Decorator2', (), {'lineno': 4}),
            type('Decorator3', (), {'lineno': 5})
        ]
        lineno = 6
    
    node = MultipleDecoratorsNode()
    assert get_first_line_number(node) == 3
```


# LLM-generated content at query #36
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5

    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 20
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 20

    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 30
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 30
```


# LLM-generated content at query #37
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
    
    class FakeDecorator:
        lineno = 5
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [FakeDecorator()]
    assert get_first_line_number(decorated_node) == 5
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultiDecoratedNode:
        decorator_list = []
        lineno = 30
    
    class FakeDecorator2:
        lineno = 7
    
    multi_decorated_node = MultiDecoratedNode()
    multi_decorated_node.decorator_list = [FakeDecorator2(), FakeDecorator()]
    assert get_first_line_number(multi_decorated_node) == 7
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 40
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 40


# LLM-generated content at query #38
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.lineno = 20
            self.end_lineno = 25
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    decorated_node = DecoratorNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has multiple decorators - should return first decorator's lineno
    class MultiDecoratorNode:
        def __init__(self):
            self.lineno = 30
            self.end_lineno = 35
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 1})(),
                type('Decorator2', (), {'lineno': 2})()
            ]
    
    multi_decorated_node = MultiDecoratorNode()
    assert get_first_line_number(multi_decorated_node) == 1
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 40
        end_lineno = 45
    
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 40
    
    # Test edge case where decorator_list is None
    class NoneDecoratorNode:
        lineno = 50
        end_lineno = 55
        decorator_list = None
    
    none_decorator_node = NoneDecoratorNode()
    assert get_first_line_number(none_decorator_node) == 50
```


# LLM-generated content at query #39
#--------------------------

def test_get_first_line_number():
    # Test with a decorated function
    import ast
    code = '''
@decorator
def foo():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == func_node.decorator_list[0].lineno

    # Test with an undecorated function
    code = '''
def bar():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == func_node.lineno

    # Test with a decorated class
    code = '''
@class_decorator
class MyClass:
    pass
'''
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == class_node.decorator_list[0].lineno

    # Test with an undecorated class
    code = '''
class MyClass2:
    pass
'''
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == class_node.lineno


# LLM-generated content at query #40
#--------------------------

def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 10
    
    node = DecoratorNode()
    
    # Create mock decorator objects with lineno attribute
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node.decorator_list = [MockDecorator(3), MockDecorator(4)]
    assert get_first_line_number(node) == 3
    
    # Test with empty decorator list
    node.decorator_list = []
    assert get_first_line_number(node) == 10
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 15
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 15


# LLM-generated content at query #41
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a function that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 10
    decorator_node = DecoratorNode()
    decorator_node.decorator_list = [type('Decorator', (), {'lineno': 3})()]
    assert get_first_line_number(decorator_node) == 3

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorNode:
        lineno = 7
    no_decorator_node = NoDecoratorNode()
    assert get_first_line_number(no_decorator_node) == 7

    # Test with empty decorator list explicitly set
    class EmptyDecoratorNode:
        lineno = 12
        decorator_list = []
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 12

    # Test with multiple decorators, should return first one's line
    class MultiDecoratorNode:
        decorator_list = []
        lineno = 20
    multi_decorator_node = MultiDecoratorNode()
    multi_decorator_node.decorator_list = [
        type('Decorator', (), {'lineno': 15})(),
        type('Decorator', (), {'lineno': 16})()
    ]
    assert get_first_line_number(multi_decorator_node) == 15


# LLM-generated content at query #42
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    import ast
    code = """
def simple_function():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2

    # Test with a function that has a decorator
    code = """
@decorator
def decorated_function():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2

    # Test with a function that has multiple decorators
    code = """
@decorator1
@decorator2
def multi_decorated_function():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2

    # Test with a class definition
    code = """
class MyClass:
    pass
"""
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2

    # Test with a class that has a decorator
    code = """
@class_decorator
class DecoratedClass:
    pass
"""
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2

    # Test with a function in the middle of the file
    code = """
def first_func():
    pass


def second_func():
    pass
"""
    tree = ast.parse(code)
    first_func = tree.body[0]
    second_func = tree.body[1]
    assert get_first_line_number(first_func) == 2
    assert get_first_line_number(second_func) == 6

    # Test with a decorated function in the middle of the file
    code = """
def first_func():
    pass


@decorator
def decorated_middle_func():
    pass


def last_func():
    pass
"""
    tree = ast.parse(code)
    first_func = tree.body[0]
    decorated_func = tree.body[1]
    last_func = tree.body[2]
    assert get_first_line_number(first_func) == 2
    assert get_first_line_number(decorated_func) == 6
    assert get_first_line_number(last_func) == 10

    # Test with a function that starts on line 1 (no leading newline)
    code = "def one_liner():\n    pass\n"
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with a decorated one-liner function
    code = "@decorator\ndef decorated_one_liner():\n    pass\n"
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with async function
    code = """
async def async_function():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2

    # Test with async function that has decorators
    code = """
@decorator
async def decorated_async_function():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2


# LLM-generated content at query #43
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 10
    decorated_node = DecoratedNode()
    
    # Create a mock decorator with a lineno attribute
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    decorated_node.decorator_list = [MockDecorator(2), MockDecorator(3)]
    
    assert get_first_line_number(decorated_node) == 2

    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15

    # Test with a node that has decorator_list attribute set to None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 20
    none_decorator_node = NoneDecoratorNode()
    assert get_first_line_number(none_decorator_node) == 20

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 25
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 25


# LLM-generated content at query #44
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 5
        end_lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 2})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 2
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 5
        end_lineno = 10
        decorator_list = [
            type('Decorator1', (), {'lineno': 2})(),
            type('Decorator2', (), {'lineno': 3})()
        ]
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 2
    
    # Test with a node that has no lineno attribute (should raise AttributeError)
    class NoLinenoNode:
        end_lineno = 10
        decorator_list = []
    
    node = NoLinenoNode()
    try:
        get_first_line_number(node)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #45
#--------------------------

```python
def test_get_first_line_number():
    # Test with a function that has decorators
    class MockDecorator:
        lineno = 10
    
    class MockFunctionWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 15
    
    node_with_decorators = MockFunctionWithDecorators()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a function that has no decorators
    class MockFunctionWithoutDecorators:
        decorator_list = []
        lineno = 20
    
    node_without_decorators = MockFunctionWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class MockSimpleNode:
        lineno = 25
    
    simple_node = MockSimpleNode()
    assert get_first_line_number(simple_node) == 25
    
    # Test with multiple decorators - should return first decorator's line
    class MockDecorator1:
        lineno = 30
    
    class MockDecorator2:
        lineno = 35
    
    class MockFunctionWithMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 40
    
    node_with_multiple_decorators = MockFunctionWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple_decorators) == 30


# LLM-generated content at query #46
#--------------------------

```python
def test_get_first_line_number():
    # Create a mock node with decorator_list
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self, decorator_linenos, node_lineno):
            self.decorator_list = [MockDecorator(lineno) for lineno in decorator_linenos]
            self.lineno = node_lineno
    
    class MockNodeWithoutDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []
    
    # Test with decorators - should return first decorator's lineno
    node_with_decorators = MockNodeWithDecorators([10, 20, 30], 40)
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with multiple decorators - should return the first one
    node_with_multiple_decorators = MockNodeWithDecorators([15, 25], 35)
    assert get_first_line_number(node_with_multiple_decorators) == 15
    
    # Test without decorators - should return node's lineno
    node_without_decorators = MockNodeWithoutDecorators(50)
    assert get_first_line_number(node_without_decorators) == 50
    
    # Test with empty decorator_list - should return node's lineno
    node_empty_decorators = MockNodeWithoutDecorators(60)
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 60
    
    # Test with a node that doesn't have decorator_list attribute
    class SimpleNode:
        def __init__(self, lineno):
            self.lineno = lineno
    
    simple_node = SimpleNode(70)
    assert get_first_line_number(simple_node) == 70
```


# LLM-generated content at query #47
#--------------------------

def test_get_first_line_number():
    # Test case 1: Function with no decorators
    class Node1:
        lineno = 5
        decorator_list = []
    node1 = Node1()
    assert get_first_line_number(node1) == 5

    # Test case 2: Function with decorators
    class Node2:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    node2 = Node2()
    assert get_first_line_number(node2) == 3

    # Test case 3: Function with multiple decorators
    class Node3:
        lineno = 15
        decorator_list = [
            type('Decorator1', (), {'lineno': 7})(),
            type('Decorator2', (), {'lineno': 8})()
        ]
    node3 = Node3()
    assert get_first_line_number(node3) == 7

    # Test case 4: Node without decorator_list attribute
    class Node4:
        lineno = 20
    node4 = Node4()
    assert get_first_line_number(node4) == 20

    # Test case 5: Empty decorator_list
    class Node5:
        lineno = 25
        decorator_list = []
    node5 = Node5()
    assert get_first_line_number(node5) == 25


# LLM-generated content at query #48
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 30
    
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 30
    
    # Test with empty decorator_list
    class EmptyDecoratorList:
        lineno = 40
        decorator_list = []
    
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 40
```


# LLM-generated content at query #49
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        lineno = 10
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 5
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node without decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 7
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 7
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 3
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 3
```


# LLM-generated content at query #50
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple AST node without decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 20
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 10
        
        class Decorator:
            lineno = 5
            
        decorator_list = [Decorator()]
    
    decorator_node = DecoratorNode()
    assert get_first_line_number(decorator_node) == 5
    
    # Test with a node where decorator_list is not present
    class NoDecoratorAttrNode:
        lineno = 15
        end_lineno = 25
    
    no_decorator_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_node) == 15
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        lineno = 30
        end_lineno = 40
        decorator_list = []
    
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 30
    
    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        lineno = 50
        
        class Decorator1:
            lineno = 35
            
        class Decorator2:
            lineno = 40
            
        decorator_list = [Decorator1(), Decorator2()]
    
    multi_node = MultiDecoratorNode()
    assert get_first_line_number(multi_node) == 35
```


# LLM-generated content at query #51
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
    
    decorated_node = DecoratedNode()
    
    # Create a mock decorator with a lineno attribute
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorated_node.decorator_list = [MockDecorator(5)]
    assert get_first_line_number(decorated_node) == 5
    
    # Test with multiple decorators - should return the first one
    decorated_node.decorator_list = [MockDecorator(5), MockDecorator(8)]
    assert get_first_line_number(decorated_node) == 5
    
    # Test with empty decorator_list
    decorated_node.decorator_list = []
    assert get_first_line_number(decorated_node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorNode:
        lineno = 30
    
    no_decorator_node = NoDecoratorNode()
    assert get_first_line_number(no_decorator_node) == 30


# LLM-generated content at query #52
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 3})()]
            self.lineno = 10
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 3
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 7
    
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 7


# LLM-generated content at query #53
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 7
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        end_lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
        end_lineno = 25
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 20
```


# LLM-generated content at query #54
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        lineno = 10
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 20
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node without decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 30
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 30
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 40
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 40
    
    # Test with a node that has multiple decorators (should return first one)
    class MockDecorator1:
        lineno = 5
    
    class MockDecorator2:
        lineno = 7
    
    class MockNodeMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 15
    
    node_multiple_decorators = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 5
```


# LLM-generated content at query #55
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function (should return decorator line)
    class FakeDecorator:
        lineno = 5
    class FakeFunction:
        lineno = 10
        decorator_list = [FakeDecorator()]
    
    node = FakeFunction()
    assert get_first_line_number(node) == 5
    
    # Test with a function without decorators (should return function line)
    class FakeFunctionNoDecorator:
        lineno = 20
        decorator_list = []
    
    node = FakeFunctionNoDecorator()
    assert get_first_line_number(node) == 20
    
    # Test with a node that has no decorator_list attribute
    class FakeNode:
        lineno = 30
    
    node = FakeNode()
    assert get_first_line_number(node) == 30
    
    # Test with multiple decorators (should return first decorator's line)
    class FakeDecorator1:
        lineno = 40
    class FakeDecorator2:
        lineno = 45
    class FakeFunctionMultiDecorator:
        lineno = 50
        decorator_list = [FakeDecorator1(), FakeDecorator2()]
    
    node = FakeFunctionMultiDecorator()
    assert get_first_line_number(node) == 40
```


# LLM-generated content at query #56
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 10})()]
            self.lineno = 15
            self.end_lineno = 20
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 10
    
    # Test case 2: Node without decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 25
            self.end_lineno = 30
    
    node2 = PlainNode()
    assert get_first_line_number(node2) == 25
    
    # Test case 3: Node with multiple decorators
    class MultiDecoratedNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator', (), {'lineno': 5})(),
                type('Decorator', (), {'lineno': 6})()
            ]
            self.lineno = 15
            self.end_lineno = 20
    
    node3 = MultiDecoratedNode()
    assert get_first_line_number(node3) == 5
    
    # Test case 4: Node with decorator_list attribute but empty
    class EmptyDecoratedNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 100
            self.end_lineno = 110
    
    node4 = EmptyDecoratedNode()
    assert get_first_line_number(node4) == 100


# LLM-generated content at query #57
#--------------------------

def test_get_first_line_number():
    # Test with a simple function node
    class MockNode:
        lineno = 5
        decorator_list = []
    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test with a function that has decorators
    class MockDecorator:
        lineno = 10
    class MockDecoratedNode:
        lineno = 15
        decorator_list = [MockDecorator()]
    decorated_node = MockDecoratedNode()
    assert get_first_line_number(decorated_node) == 10

    # Test with a node that has multiple decorators
    class MockMultiDecoratorNode:
        lineno = 20
        decorator_list = [MockDecorator(), MockDecorator()]
    multi_decorator_node = MockMultiDecoratorNode()
    assert get_first_line_number(multi_decorator_node) == 10


# LLM-generated content at query #58
#--------------------------

```python
def test_get_first_line_number():
    # Test simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test function with decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 15
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5

    # Test class with decorators
    class ClassNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 20})()]
            self.lineno = 30
    class_node = ClassNode()
    assert get_first_line_number(class_node) == 20

    # Test method with decorators
    class MethodNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 25})()]
            self.lineno = 35
    method_node = MethodNode()
    assert get_first_line_number(method_node) == 25

    # Test node without decorator_list attribute
    class NoDecoratorAttr:
        lineno = 40
    no_decorator_node = NoDecoratorAttr()
    assert get_first_line_number(no_decorator_node) == 40

    # Test with multiple decorators - should return first decorator's lineno
    class MultiDecoratedNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 1})(),
                type('Decorator2', (), {'lineno': 2})()
            ]
            self.lineno = 50
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1

    # Test with empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 60
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 60
```


# LLM-generated content at query #59
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node with decorators
    class NodeWithDecorators:
        decorator_list = [type('Decorator', (), {'lineno': 5})]
        lineno = 10
    
    node1 = NodeWithDecorators()
    assert get_first_line_number(node1) == 5
    
    # Test case 2: Node without decorators
    class NodeWithoutDecorators:
        decorator_list = []
        lineno = 20
    
    node2 = NodeWithoutDecorators()
    assert get_first_line_number(node2) == 20
    
    # Test case 3: Node without decorator_list attribute
    class NodeNoDecoratorList:
        lineno = 30
    
    node3 = NodeNoDecoratorList()
    assert get_first_line_number(node3) == 30
    
    # Test case 4: Multiple decorators - should use first decorator's lineno
    class NodeMultipleDecorators:
        decorator_list = [
            type('Decorator1', (), {'lineno': 1}),
            type('Decorator2', (), {'lineno': 2})
        ]
        lineno = 3
    
    node4 = NodeMultipleDecorators()
    assert get_first_line_number(node4) == 1
```


# LLM-generated content at query #60
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 10

    # Test with a function with decorators
    class DecoratedNode:
        lineno = 20
        decorator_list = []
    
    class Decorator:
        lineno = 15
    
    node = DecoratedNode()
    node.decorator_list = [Decorator()]
    assert get_first_line_number(node) == 15

    # Test with multiple decorators (should return first decorator's line)
    class MultiDecoratorNode:
        lineno = 30
        decorator_list = []
    
    class Decorator1:
        lineno = 25
    
    class Decorator2:
        lineno = 26
    
    node = MultiDecoratorNode()
    node.decorator_list = [Decorator1(), Decorator2()]
    assert get_first_line_number(node) == 25

    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 40
    
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 40

    # Test with empty decorator_list
    class EmptyDecoratorList:
        lineno = 50
        decorator_list = []
    
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 50
```


# LLM-generated content at query #61
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 20
        end_lineno = 25
        decorator_list = [type('Decorator', (), {'lineno': 18})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 18
    
    # Test edge case: node without decorator_list attribute
    class NoDecoratorListNode:
        lineno = 30
        end_lineno = 35
    
    node = NoDecoratorListNode()
    assert get_first_line_number(node) == 30
    
    # Test empty decorator list explicitly
    class EmptyDecoratorListNode:
        lineno = 40
        end_lineno = 45
        decorator_list = []
    
    node = EmptyDecoratorListNode()
    assert get_first_line_number(node) == 40
```


# LLM-generated content at query #62
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
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 7
    
    no_decorator_attr = NoDecoratorAttr()
    assert get_first_line_number(no_decorator_attr) == 7
```


# LLM-generated content at query #63
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function node
    class MockNode:
        lineno = 10
        end_lineno = 20
        decorator_list = []

    node = MockNode()
    assert get_first_line_number(node) == 10

    # Test with decorators
    class MockDecorator:
        lineno = 5

    class MockNodeWithDecorator:
        lineno = 10
        end_lineno = 20
        decorator_list = [MockDecorator()]

    node_with_decorator = MockNodeWithDecorator()
    assert get_first_line_number(node_with_decorator) == 5

    # Test with class node
    class MockClassNode:
        lineno = 15
        end_lineno = 30
        decorator_list = []

    class_node = MockClassNode()
    assert get_first_line_number(class_node) == 15

    # Test with decorated class
    class MockClassWithDecorator:
        lineno = 15
        end_lineno = 30
        decorator_list = [MockDecorator()]

    class_with_decorator = MockClassWithDecorator()
    assert get_first_line_number(class_with_decorator) == 5

    # Test edge case: node without decorator_list attribute
    class MockNodeNoDecorator:
        lineno = 25
        end_lineno = 35

    node_no_decorator = MockNodeNoDecorator()
    assert get_first_line_number(node_no_decorator) == 25
```


# LLM-generated content at query #64
#--------------------------

```python
def test_get_first_line_number():
    # Test with a function that has decorators
    import ast
    code = """
@decorator1
@decorator2
def my_function():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # line of first decorator

    # Test with a function without decorators
    code = """
def my_function():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # line of function definition

    # Test with a class that has decorators
    code = """
@decorator1
class MyClass:
    pass
"""
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2  # line of first decorator

    # Test with a class without decorators
    code = """
class MyClass:
    pass
"""
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2  # line of class definition

    # Test with edge case: no decorator_list attribute (e.g., Module node)
    code = """
pass
"""
    tree = ast.parse(code)
    module_node = tree
    assert get_first_line_number(module_node) == 2  # line of pass statement
```


# LLM-generated content at query #65
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 20
    
    # Create mock decorator objects
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_with_decorator = DecoratorNode()
    node_with_decorator.decorator_list = [MockDecorator(5), MockDecorator(6)]
    assert get_first_line_number(node_with_decorator) == 5
    
    # Test with empty decorator_list
    node_empty_decorators = DecoratorNode()
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorNode:
        lineno = 30
    
    node_no_decorators = NoDecoratorNode()
    assert get_first_line_number(node_no_decorators) == 30


# LLM-generated content at query #66
#--------------------------

```python
def test_get_first_line_number():
    # Test simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test function with decorators
    class DecoratedNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5

    # Test function with multiple decorators
    class MultiDecoratedNode:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1

    # Test node without decorator_list attribute
    class NoDecoratorList:
        lineno = 40
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 40

    # Test node with empty decorator_list
    class EmptyDecoratorNode:
        lineno = 50
        decorator_list = []
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 50
```


# LLM-generated content at query #67
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        lineno = 5
        decorator_list = []
    
    node = SimpleFunction()
    assert get_first_line_number(node) == 5
    
    # Test with a function that has decorators
    class DecoratedFunction:
        pass
    
    node = DecoratedFunction()
    node.decorator_list = [type('Decorator', (), {'lineno': 3})()]
    node.lineno = 5
    assert get_first_line_number(node) == 3
    
    # Test with a function that has multiple decorators
    node = DecoratedFunction()
    node.decorator_list = [
        type('Decorator', (), {'lineno': 1})(),
        type('Decorator', (), {'lineno': 2})()
    ]
    node.lineno = 7
    assert get_first_line_number(node) == 1
    
    # Test with a class without decorators
    class SimpleClass:
        lineno = 10
        decorator_list = []
    
    node = SimpleClass()
    assert get_first_line_number(node) == 10
    
    # Test with a class that has decorators
    node = DecoratedFunction()
    node.decorator_list = [type('Decorator', (), {'lineno': 8})()]
    node.lineno = 12
    assert get_first_line_number(node) == 8


# LLM-generated content at query #68
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [SimpleNode(5)]
            self.lineno = 10
    
    class SimpleNode:
        def __init__(self, line):
            self.lineno = line
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node without decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 30
    
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 30
```


# LLM-generated content at query #69
#--------------------------

```python
def test_get_first_line_number():
    # Test a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 10

    # Test a function with decorators
    class DecoratorNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    node = DecoratorNode()
    assert get_first_line_number(node) == 15

    # Test a node with multiple decorators (should return first decorator's lineno)
    class MultiDecoratorNode:
        lineno = 30
        decorator_list = [
            type('Decorator', (), {'lineno': 21})(),
            type('Decorator', (), {'lineno': 22})()
        ]
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 21

    # Test edge case where decorator_list is None (should use node.lineno)
    class NoneDecoratorNode:
        lineno = 40
        decorator_list = None
    node = NoneDecoratorNode()
    assert get_first_line_number(node) == 40

    # Test edge case where decorator_list is empty tuple
    class EmptyTupleNode:
        lineno = 50
        decorator_list = ()
    node = EmptyTupleNode()
    assert get_first_line_number(node) == 50
```


# LLM-generated content at query #70
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a function that has decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    node = DecoratedNode()
    assert get_first_line_number(node) == 3

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 7
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 7

    # Test with empty decorator_list
    class EmptyDecorators:
        lineno = 12
        decorator_list = []
    node = EmptyDecorators()
    assert get_first_line_number(node) == 12

    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecorators:
        lineno = 20
        decorator_list = [
            type('Decorator1', (), {'lineno': 15})(),
            type('Decorator2', (), {'lineno': 16})()
        ]
    node = MultipleDecorators()
    assert get_first_line_number(node) == 15


# LLM-generated content at query #71
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test case 2: Node with decorators
    class DecoratorNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})]
        lineno = 10
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 5
    
    # Test case 3: Node with multiple decorators
    class MultiDecoratorNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 3}),
            type('Decorator2', (), {'lineno': 4})
        ]
        lineno = 8
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 3
    
    # Test case 4: Node without decorator_list attribute
    class NoDecoratorList:
        lineno = 20
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 20
    
    # Test case 5: Empty decorator_list
    class EmptyDecoratorList:
        decorator_list = []
        lineno = 30
    
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 30
```


# LLM-generated content at query #72
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class MockDecorator:
        lineno = 5
    
    class MockFunctionDef:
        lineno = 10
        decorator_list = [MockDecorator()]
    
    node = MockFunctionDef()
    assert get_first_line_number(node) == 5
    
    # Test without decorators
    class MockFunctionDefNoDecorators:
        lineno = 10
        decorator_list = []
    
    node = MockFunctionDefNoDecorators()
    assert get_first_line_number(node) == 10
    
    # Test with node that doesn't have decorator_list attribute
    class MockNodeWithoutDecorators:
        lineno = 15
    
    node = MockNodeWithoutDecorators()
    assert get_first_line_number(node) == 15
    
    # Test with multiple decorators (should return first one's line)
    class MockDecorator1:
        lineno = 3
    
    class MockDecorator2:
        lineno = 4
    
    class MockFunctionDefMultipleDecors:
        lineno = 8
        decorator_list = [MockDecorator1(), MockDecorator2()]
    
    node = MockFunctionDefMultipleDecors()
    assert get_first_line_number(node) == 3


# LLM-generated content at query #73
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorators
    class MockNode:
        lineno = 5
        decorator_list = []
    
    node = MockNode()
    assert get_first_line_number(node) == 5
    
    # Test with decorators
    class MockDecorator:
        lineno = 10
    
    class MockNodeWithDecorators:
        lineno = 15
        decorator_list = [MockDecorator()]
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with multiple decorators
    class MockDecorator2:
        lineno = 20
    
    class MockNodeWithMultipleDecorators:
        lineno = 25
        decorator_list = [MockDecorator(), MockDecorator2()]
    
    node_with_multiple = MockNodeWithMultipleDecorators()
    assert get_first_line_number(node_with_multiple) == 10
    
    # Test with no decorator_list attribute
    class MockNodeNoDecoratorList:
        lineno = 30
    
    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 30
```


# LLM-generated content at query #74
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})]
        lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 5}),
            type('Decorator2', (), {'lineno': 7})
        ]
        lineno = 10
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 5
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 20
```


# LLM-generated content at query #75
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test case 2: Function with decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test case 3: Node without decorator_list attribute
    class NoDecoratorList:
        lineno = 20
    
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 20
    
    # Test case 4: Multiple decorators - should return first decorator's lineno
    class MultipleDecorators:
        decorator_list = [
            type('Decorator1', (), {'lineno': 3})(),
            type('Decorator2', (), {'lineno': 4})()
        ]
        lineno = 10
    
    multi_decorator_node = MultipleDecorators()
    assert get_first_line_number(multi_decorator_node) == 3
    
    # Test case 5: Empty decorator_list
    class EmptyDecorators:
        decorator_list = []
        lineno = 15
    
    empty_decorator_node = EmptyDecorators()
    assert get_first_line_number(empty_decorator_node) == 15
```


# LLM-generated content at query #76
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        pass
    
    def sample_function():
        pass
    
    node = sample_function.__code__
    # Create a mock AST node for testing
    import ast
    tree = ast.parse("def sample_function():\n    pass")
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1
    
    # Test with a decorated function
    tree = ast.parse("@decorator\ndef decorated_function():\n    pass")
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1
    
    # Test with multiple decorators
    tree = ast.parse("@decorator1\n@decorator2\ndef multi_decorated():\n    pass")
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1
    
    # Test with a class
    tree = ast.parse("class MyClass:\n    def method(self):\n        pass")
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 1
    
    # Test with a decorated class
    tree = ast.parse("@decorator\nclass DecoratedClass:\n    pass")
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 1
    
    # Test with a lambda assignment
    tree = ast.parse("x = lambda: 1")
    assign_node = tree.body[0]
    assert get_first_line_number(assign_node) == 1
    
    # Test with a function that has decorators on multiple lines
    tree = ast.parse("@decorator1\n@decorator2\ndef multiline_decorated():\n    pass")
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1
    
    # Test with a function at a specific line number
    code = "x = 1\n\ndef later_function():\n    pass"
    tree = ast.parse(code)
    func_node = tree.body[1]
    assert get_first_line_number(func_node) == 3
    
    # Test with a decorated function at a specific line number
    code = "x = 1\n\n@decorator\ndef decorated_later():\n    pass"
    tree = ast.parse(code)
    func_node = tree.body[1]
    assert get_first_line_number(func_node) == 4
    
    # Test with a function that has no decorators but starts at line 1
    tree = ast.parse("def no_decorator():\n    pass")
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1
    
    # Test with a function that has an empty decorator list
    tree = ast.parse("def no_decorators():\n    pass")
    func_node = tree.body[0]
    func_node.decorator_list = []
    assert get_first_line_number(func_node) == 1


# LLM-generated content at query #77
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class MockDecorator:
        lineno = 5
    
    class MockFunction:
        lineno = 10
        decorator_list = [MockDecorator()]
    
    node = MockFunction()
    assert get_first_line_number(node) == 5
    
    # Test with a function without decorators
    class MockFunctionNoDecorators:
        lineno = 20
        decorator_list = []
    
    node = MockFunctionNoDecorators()
    assert get_first_line_number(node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNode:
        lineno = 30
    
    node = MockNode()
    assert get_first_line_number(node) == 30
    
    # Test with a decorated class
    class MockClass:
        lineno = 40
        decorator_list = [MockDecorator()]  # reusing MockDecorator with lineno=5
    
    node = MockClass()
    assert get_first_line_number(node) == 5
    
    # Test with multiple decorators
    class MockMultiDecorator:
        lineno = 100
    
    class MockFunctionMultiDecorators:
        lineno = 50
        decorator_list = [MockMultiDecorator(), MockDecorator()]
    
    node = MockFunctionMultiDecorators()
    assert get_first_line_number(node) == 100
```


# LLM-generated content at query #78
#--------------------------

def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 20
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self, decorators):
            self.lineno = 30
            self.end_lineno = 40
            self.decorator_list = decorators
    
    class Decorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorator1 = Decorator(5)
    decorator2 = Decorator(6)
    node_with_decorators = DecoratorNode([decorator1, decorator2])
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with a node that has an empty decorator list
    node_empty_decorators = DecoratorNode([])
    assert get_first_line_number(node_empty_decorators) == 30


# LLM-generated content at query #79
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 10

    # Test case 2: Function with decorators
    class DecoratedNode:
        lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    node = DecoratedNode()
    assert get_first_line_number(node) == 5

    # Test case 3: Class without decorators
    class SimpleClassNode:
        lineno = 20
        decorator_list = []
    node = SimpleClassNode()
    assert get_first_line_number(node) == 20

    # Test case 4: Class with multiple decorators
    class MultiDecoratedNode:
        lineno = 25
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 1

    # Test case 5: Node without decorator_list attribute (older Python versions)
    class OldStyleNode:
        lineno = 30
    node = OldStyleNode()
    assert get_first_line_number(node) == 30
```


# LLM-generated content at query #80
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a function that has decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 7})()]
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 7

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 15
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 15

    # Test with empty decorator list
    class EmptyDecoratorList:
        lineno = 20
        decorator_list = []
    empty_decorator_node = EmptyDecoratorList()
    assert get_first_line_number(empty_decorator_node) == 20

    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecorators:
        lineno = 25
        decorator_list = [
            type('Decorator1', (), {'lineno': 21}),
            type('Decorator2', (), {'lineno': 22})
        ]
    multi_decorator_node = MultipleDecorators()
    assert get_first_line_number(multi_decorator_node) == 21

    # Test with decorator_list set to None (edge case)
    class NoneDecoratorList:
        lineno = 30
        decorator_list = None
    none_decorator_node = NoneDecoratorList()
    assert get_first_line_number(none_decorator_node) == 30
```


# LLM-generated content at query #81
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.decorator_list = [type('Dec', (), {'lineno': 2})()]
            self.lineno = 5
    decorator_node = DecoratorNode()
    assert get_first_line_number(decorator_node) == 2

    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 7
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 7

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 9
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 9

    # Test with a node that has multiple decorators
    class MultipleDecoratorsNode:
        def __init__(self):
            self.decorator_list = [
                type('Dec1', (), {'lineno': 3})(),
                type('Dec2', (), {'lineno': 4})()
            ]
            self.lineno = 10
    multiple_decorators_node = MultipleDecoratorsNode()
    assert get_first_line_number(multiple_decorators_node) == 3
```


# LLM-generated content at query #82
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class Decorator:
        lineno = 2
    
    class DecoratedNode:
        lineno = 10
        end_lineno = 15
        decorator_list = [Decorator()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 2
    
    # Test with a node that has multiple decorators (should return first decorator's lineno)
    class Decorator2:
        lineno = 3
    
    class MultiDecoratedNode:
        lineno = 20
        end_lineno = 25
        decorator_list = [Decorator(), Decorator2()]
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 2
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 7
        end_lineno = 12
    
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 7


# LLM-generated content at query #83
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = [type('Decorator', (), {'lineno': 3})]
        lineno = 10
    node = DecoratorNode()
    assert get_first_line_number(node) == 3

    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 7
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 7

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 12
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 12

    # Test with multiple decorators to ensure first one is used
    class MultipleDecoratorsNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 2}),
            type('Decorator2', (), {'lineno': 4})
        ]
        lineno = 15
    node = MultipleDecoratorsNode()
    assert get_first_line_number(node) == 2
```


# LLM-generated content at query #84
#--------------------------

```python
def test_get_first_line_number():
    # Test with a plain function (no decorators)
    class PlainFunction:
        lineno = 10
    
    plain_node = PlainFunction()
    assert get_first_line_number(plain_node) == 10
    
    # Test with a decorated function
    class DecoratedFunction:
        decorator_list = []
        lineno = 20
    
    decorated_node = DecoratedFunction()
    # No decorators, should return node.lineno
    assert get_first_line_number(decorated_node) == 20
    
    # Test with decorators present
    class Decorator:
        lineno = 5
    
    class DecoratedWithDecorators:
        decorator_list = [Decorator()]
        lineno = 20
    
    decorated_with_decorators = DecoratedWithDecorators()
    # Should return first decorator's lineno (5), not node.lineno (20)
    assert get_first_line_number(decorated_with_decorators) == 5
    
    # Test with multiple decorators
    class Decorator1:
        lineno = 3
    
    class Decorator2:
        lineno = 7
    
    class MultipleDecorators:
        decorator_list = [Decorator1(), Decorator2()]
        lineno = 30
    
    multiple_decorators = MultipleDecorators()
    # Should return first decorator's lineno (3)
    assert get_first_line_number(multiple_decorators) == 3
```


# LLM-generated content at query #85
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function node
    class SimpleFunction:
        lineno = 10
        end_lineno = 15
        decorator_list = []

    simple_func = SimpleFunction()
    assert get_first_line_number(simple_func) == 10

    # Test with a decorated function
    class DecoratedFunction:
        lineno = 20
        end_lineno = 25
        decorator_list = [type('Decorator', (), {'lineno': 18})(), type('Decorator', (), {'lineno': 19})()]

    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 18

    # Test with a class node
    class SimpleClass:
        lineno = 30
        end_lineno = 35
        decorator_list = []

    simple_class = SimpleClass()
    assert get_first_line_number(simple_class) == 30

    # Test with a decorated class
    class DecoratedClass:
        lineno = 40
        end_lineno = 45
        decorator_list = [type('Decorator', (), {'lineno': 38})()]

    decorated_class = DecoratedClass()
    assert get_first_line_number(decorated_class) == 38

    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttr:
        lineno = 50
        end_lineno = 55

    no_decorator = NoDecoratorAttr()
    assert get_first_line_number(no_decorator) == 50
```


# LLM-generated content at query #86
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class MockDecorator:
        lineno = 10
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 20
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node without decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 30
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 30
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 40
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 40
    
    # Test with multiple decorators (should return first decorator's lineno)
    class MockDecorator2:
        lineno = 15
    
    class MockNodeMultipleDecorators:
        decorator_list = [MockDecorator(), MockDecorator2()]
        lineno = 20
    
    node_multiple_decorators = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 10
```


# LLM-generated content at query #87
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function definition without decorators
    class SimpleFunction:
        def method(self):
            pass
    
    simple_func_node = SimpleFunction.method.__code__
    # Create a mock AST node
    class MockNode:
        lineno = 10
        decorator_list = []
    
    node = MockNode()
    assert get_first_line_number(node) == 10
    
    # Test with decorators
    class MockDecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    
    decorated_node = MockDecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with no decorator_list attribute
    class MockNodeNoDecorators:
        lineno = 20
    
    node_no_decorators = MockNodeNoDecorators()
    assert get_first_line_number(node_no_decorators) == 20
```


# LLM-generated content at query #88
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 5
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 10
    
    decorator_node = DecoratorNode()
    decorator_node.decorator_list = [type('Decorator', (), {'lineno': 3})()]
    assert get_first_line_number(decorator_node) == 3
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15
```


# LLM-generated content at query #89
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        pass
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [type('Decorator', (), {'lineno': 5})()]
    decorated_node.lineno = 10
    
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node without decorators
    class PlainNode:
        pass
    
    plain_node = PlainNode()
    plain_node.lineno = 20
    
    assert get_first_line_number(plain_node) == 20
    
    # Test with a node that has empty decorator_list
    empty_decorator_node = PlainNode()
    empty_decorator_node.decorator_list = []
    empty_decorator_node.lineno = 30
    
    assert get_first_line_number(empty_decorator_node) == 30
    
    # Test with a node that has no decorator_list attribute
    no_decorator_attr_node = PlainNode()
    no_decorator_attr_node.lineno = 40
    
    assert get_first_line_number(no_decorator_attr_node) == 40
```


# LLM-generated content at query #90
#--------------------------

def test_get_first_line_number():
    # Test with a simple node (no decorators)
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        end_lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 15
        end_lineno = 25
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        lineno = 7
        end_lineno = 12
        decorator_list = []
    
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 7
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttribute:
        lineno = 8
        end_lineno = 13
    
    no_attr_node = NoDecoratorAttribute()
    assert get_first_line_number(no_attr_node) == 8


# LLM-generated content at query #91
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 30
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 30
    
    # Test with multiple decorators - should return first decorator's line
    class MultiDecoratedNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 1})(),
                type('Decorator2', (), {'lineno': 2})()
            ]
            self.lineno = 3
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1
    
    # Test with empty decorator list but node has lineno
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 40
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 40
```


# LLM-generated content at query #92
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [MockDecorator(5)]
    
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with a node without decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = []
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorList:
        def __init__(self):
            self.lineno = 30
    
    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 30
    
    # Test with a real AST node
    import ast
    code = """
@decorator
def func():
    pass
"""
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # line of @decorator
    
    # Test with a regular function (no decorators)
    code2 = """
def func():
    pass
"""
    tree2 = ast.parse(code2)
    func_node2 = tree2.body[0]
    assert get_first_line_number(func_node2) == 2  # line of def
```


# LLM-generated content at query #93
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 20
    decorator_node = DecoratorNode()
    
    # Add a mock decorator with lineno attribute
    class MockDecorator:
        lineno = 5
    decorator_node.decorator_list = [MockDecorator()]
    
    assert get_first_line_number(decorator_node) == 5

    # Test with multiple decorators - should return the first decorator's lineno
    class MultipleDecoratorsNode:
        decorator_list = []
        lineno = 30
    multi_decorator_node = MultipleDecoratorsNode()
    
    class MockDecorator2:
        lineno = 7
    class MockDecorator3:
        lineno = 8
    multi_decorator_node.decorator_list = [MockDecorator2(), MockDecorator3()]
    
    assert get_first_line_number(multi_decorator_node) == 7

    # Test with empty decorator_list attribute
    class EmptyDecoratorListNode:
        decorator_list = []
        lineno = 40
    empty_list_node = EmptyDecoratorListNode()
    assert get_first_line_number(empty_list_node) == 40

    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorListAttrNode:
        lineno = 50
    no_attr_node = NoDecoratorListAttrNode()
    assert get_first_line_number(no_attr_node) == 50
```


# LLM-generated content at query #94
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self, decorator_lineno, node_lineno):
            self.decorator_list = [MockDecorator(decorator_lineno)]
            self.lineno = node_lineno
    
    node_with_decorators = MockNodeWithDecorators(10, 20)
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_without_decorators = MockNodeWithoutDecorators(30)
    assert get_first_line_number(node_without_decorators) == 30
    
    # Test with a node that has an empty decorator_list
    class MockNodeEmptyDecorators:
        def __init__(self, lineno):
            self.decorator_list = []
            self.lineno = lineno
    
    node_empty_decorators = MockNodeEmptyDecorators(40)
    assert get_first_line_number(node_empty_decorators) == 40
```


# LLM-generated content at query #95
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        lineno = 30
        decorator_list = []
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 30
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 40
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 40
```


# LLM-generated content at query #96
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Function without decorators
    class SimpleFunction:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []

    simple_func = SimpleFunction()
    assert get_first_line_number(simple_func) == 10

    # Test case 2: Function with decorators
    class DecoratedFunction:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = [type('Decorator', (), {'lineno': 15})()]

    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 15

    # Test case 3: Class with decorators
    class DecoratedClass:
        def __init__(self):
            self.lineno = 30
            self.decorator_list = [type('Decorator', (), {'lineno': 25})()]

    decorated_class = DecoratedClass()
    assert get_first_line_number(decorated_class) == 25

    # Test case 4: Class without decorators
    class PlainClass:
        def __init__(self):
            self.lineno = 40
            self.decorator_list = []

    plain_class = PlainClass()
    assert get_first_line_number(plain_class) == 40

    # Test case 5: Node with no decorator_list attribute
    class NoDecoratorAttr:
        def __init__(self):
            self.lineno = 50

    no_decorator = NoDecoratorAttr()
    assert get_first_line_number(no_decorator) == 50
```


# LLM-generated content at query #97
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10

    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20

        def __init__(self):
            class Decorator:
                lineno = 5
            self.decorator_list = [Decorator()]

    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5

    # Test with multiple decorators - should return the first one's lineno
    class MultiDecoratedNode:
        decorator_list = []
        lineno = 30

        def __init__(self):
            class Decorator1:
                lineno = 7
            class Decorator2:
                lineno = 8
            self.decorator_list = [Decorator1(), Decorator2()]

    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 7

    # Test with empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 40

    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 40

    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 50

    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 50
```


# LLM-generated content at query #98
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 8
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
        lineno = 10
        end_lineno = 15
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 20
        end_lineno = 25
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 20
    
    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 1})(),
            type('Decorator', (), {'lineno': 2})(),
            type('Decorator', (), {'lineno': 3})()
        ]
        lineno = 10
        end_lineno = 15
    
    multi_decorator_node = MultiDecoratorNode()
    # Should return the lineno of the first decorator
    assert get_first_line_number(multi_decorator_node) == 1
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 30
        end_lineno = 35
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 30


# LLM-generated content at query #99
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 20
        end_lineno = 25
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 30
        end_lineno = 35
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})(),
            type('Decorator3', (), {'lineno': 3})()
        ]
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1
    
    # Test with a node that has decorator_list attribute but is empty
    class EmptyDecoratorListNode:
        lineno = 40
        end_lineno = 45
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorListNode()
    assert get_first_line_number(empty_decorator_node) == 40
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorListAttrNode:
        lineno = 50
        end_lineno = 55
    
    no_decorator_attr_node = NoDecoratorListAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 50
```


# LLM-generated content at query #100
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockNodeWithDecorators:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [
                type('MockDecorator', (), {'lineno': 5})(),
                type('MockDecorator', (), {'lineno': 6})()
            ]

    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 5

    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = []

    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20

    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        def __init__(self):
            self.lineno = 30

    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 30

    # Test with a node that has decorator_list attribute set to None
    class MockNodeNoneDecoratorList:
        def __init__(self):
            self.lineno = 40
            self.decorator_list = None

    node_none_decorator_list = MockNodeNoneDecoratorList()
    assert get_first_line_number(node_none_decorator_list) == 40
```


# LLM-generated content at query #101
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        end_lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 3
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 15
        end_lineno = 25
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 1
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 30
        end_lineno = 35
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 30
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        lineno = 40
        end_lineno = 45
        decorator_list = []
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 40


# LLM-generated content at query #102
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a function node that has decorators
    import ast
    code = '''
@decorator1
@decorator2
def my_function():
    pass
'''
    tree = ast.parse(code)
    function_node = tree.body[0]
    
    # Should return the lineno of the first decorator
    assert get_first_line_number(function_node) == function_node.decorator_list[0].lineno
    assert get_first_line_number(function_node) == 2  # decorator1 is on line 2
    
    # Test with a class node that has decorators
    code2 = '''
@class_decorator
class MyClass:
    pass
'''
    tree2 = ast.parse(code2)
    class_node = tree2.body[0]
    
    # Should return the lineno of the first decorator
    assert get_first_line_number(class_node) == class_node.decorator_list[0].lineno
    assert get_first_line_number(class_node) == 2
    
    # Test with an AST node that has decorator_list attribute but empty
    code3 = '''
def plain_function():
    pass
'''
    tree3 = ast.parse(code3)
    plain_node = tree3.body[0]
    
    # Should return node.lineno when no decorators
    assert get_first_line_number(plain_node) == plain_node.lineno
    assert get_first_line_number(plain_node) == 2
```


# LLM-generated content at query #103
#--------------------------

def test_get_first_line_number():
    # Test with a simple AST node without decorators
    import ast
    code = "x = 1\n"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with a node that has a decorator
    code = "@decorator\ndef foo():\n    pass\n"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with multiple decorators
    code = "@decorator1\n@decorator2\ndef foo():\n    pass\n"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with a class that has a decorator
    code = "@decorator\nclass Foo:\n    pass\n"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with an expression statement
    code = "print('hello')\n"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1


# LLM-generated content at query #104
#--------------------------

def test_get_first_line_number():
    # Test with no decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with decorators
    class DecoratedNode:
        decorator_list = []
    
    decorator1 = type('Decorator', (), {'lineno': 5})()
    decorator2 = type('Decorator', (), {'lineno': 7})()
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(decorated_node) == 5
    
    # Test with node that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 20
    
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 20
    
    # Test edge case with empty decorator list
    class EmptyDecoratorList:
        lineno = 30
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorList()
    assert get_first_line_number(empty_decorator_node) == 30


# LLM-generated content at query #105
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class MockDecorator:
        lineno = 10
    class MockFunctionDef:
        lineno = 20
        decorator_list = [MockDecorator()]
    node = MockFunctionDef()
    assert get_first_line_number(node) == 10

    # Test with a function without decorators
    class MockFunctionDefNoDecorator:
        lineno = 30
        decorator_list = []
    node = MockFunctionDefNoDecorator()
    assert get_first_line_number(node) == 30

    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecoratorList:
        lineno = 40
    node = MockNodeNoDecoratorList()
    assert get_first_line_number(node) == 40

    # Test with multiple decorators
    class MockDecorator1:
        lineno = 5
    class MockDecorator2:
        lineno = 6
    class MockFunctionDefMultipleDecorators:
        lineno = 20
        decorator_list = [MockDecorator1(), MockDecorator2()]
    node = MockFunctionDefMultipleDecorators()
    assert get_first_line_number(node) == 5
```


# LLM-generated content at query #106
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 15
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 3})(),
            type('Decorator', (), {'lineno': 4})(),
            type('Decorator', (), {'lineno': 5})()
        ]
        lineno = 10
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 3
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 8
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 8
    
    # Test with empty decorator_list
    class EmptyDecoratorList:
        decorator_list = []
        lineno = 20
    
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 20
```


# LLM-generated content at query #107
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})]
        lineno = 10
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorsNode:
        decorator_list = []
        lineno = 15
    
    node = EmptyDecoratorsNode()
    assert get_first_line_number(node) == 15


# LLM-generated content at query #108
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Function without decorators
    class SimpleFunc:
        lineno = 10
    
    node = SimpleFunc()
    assert get_first_line_number(node) == 10

    # Test case 2: Function with decorators
    class DecoratedFunc:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    
    node = DecoratedFunc()
    assert get_first_line_number(node) == 5

    # Test case 3: Empty decorator list
    class EmptyDecorators:
        decorator_list = []
        lineno = 20
    
    node = EmptyDecorators()
    assert get_first_line_number(node) == 20

    # Test case 4: Class without decorator_list attribute
    class NoDecorators:
        lineno = 30
    
    node = NoDecorators()
    assert get_first_line_number(node) == 30
```


# LLM-generated content at query #109
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [SimpleNode(10)]
            self.lineno = 20
    
    class SimpleNode:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 10
    
    # Test with a node without decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 30
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 30
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 40
    
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 40
```


# LLM-generated content at query #110
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
    
    decorated_node = DecoratedNode()
    
    # Create a mock decorator with a lineno attribute
    class MockDecorator:
        def __init__(self, line):
            self.lineno = line
    
    decorated_node.decorator_list = [MockDecorator(3), MockDecorator(4)]
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
    
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 20
    
    # Test with a node that has decorator_list as None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 25
    
    none_node = NoneDecoratorNode()
    assert get_first_line_number(none_node) == 25
```


# LLM-generated content at query #111
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        lineno = 10
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 15
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 20
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorList:
        lineno = 25
    
    node_no_decorator_list = MockNodeNoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 25
    
    # Test with multiple decorators - should return first decorator's lineno
    class MockDecorator1:
        lineno = 30
    
    class MockDecorator2:
        lineno = 35
    
    class MockNodeMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 40
    
    node_multiple_decorators = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 30
```


# LLM-generated content at query #112
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})()]
        lineno = 15
    node = DecoratedNode()
    assert get_first_line_number(node) == 10

    # Test with a node that has an empty decorator_list attribute
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 20
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 20

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 25
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 25

    # Test with a method in a class (simulated)
    class MethodNode:
        decorator_list = [type('Decorator', (), {'lineno': 30})()]
        lineno = 35
    node = MethodNode()
    assert get_first_line_number(node) == 30
```


# LLM-generated content at query #113
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        pass
    
    simple_func = SimpleFunction()
    simple_func.lineno = 10
    simple_func.decorator_list = []
    
    assert get_first_line_number(simple_func) == 10
    
    # Test with a function that has decorators
    class DecoratedFunction:
        pass
    
    decorated_func = DecoratedFunction()
    decorated_func.lineno = 20
    
    class Decorator:
        def __init__(self):
            self.lineno = 15
    
    decorated_func.decorator_list = [Decorator()]
    
    assert get_first_line_number(decorated_func) == 15
    
    # Test with an object that doesn't have decorator_list attribute
    class NoDecorators:
        pass
    
    no_decorators = NoDecorators()
    no_decorators.lineno = 30
    
    assert get_first_line_number(no_decorators) == 30
```


# LLM-generated content at query #114
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    import ast
    code = "def foo():\n    pass\n"
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with a function with decorators
    code = "@decorator\ndef foo():\n    pass\n"
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with multiple decorators
    code = "@decorator1\n@decorator2\ndef foo():\n    pass\n"
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with class definition
    code = "class MyClass:\n    pass\n"
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 1

    # Test with decorated class
    code = "@decorator\nclass MyClass:\n    pass\n"
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 1


# LLM-generated content at query #115
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        lineno = 5
        decorator_list = []
    
    simple_func = SimpleFunction()
    assert get_first_line_number(simple_func) == 5
    
    # Test with a function that has decorators
    class DecoratedFunction:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 3
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 7
    
    no_decorator = NoDecoratorAttr()
    assert get_first_line_number(no_decorator) == 7
    
    # Test with empty decorator_list (same as no decorators)
    class EmptyDecorators:
        lineno = 12
        decorator_list = []
    
    empty_decorators = EmptyDecorators()
    assert get_first_line_number(empty_decorators) == 12


# LLM-generated content at query #116
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 15
        end_lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 12})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 12
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 25
        end_lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 12})(),
            type('Decorator2', (), {'lineno': 14})(),
            type('Decorator3', (), {'lineno': 16})()
        ]
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 12
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 35
        end_lineno = 40
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 35


# LLM-generated content at query #117
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 15
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 20
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 20
```


# LLM-generated content at query #118
#--------------------------

def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15


# LLM-generated content at query #119
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10

    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20

    decorated_node = DecoratedNode()
    
    # Create mock decorator objects with lineno attributes
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    decorated_node.decorator_list = [MockDecorator(15), MockDecorator(18)]
    
    # Should return the lineno of the first decorator
    assert get_first_line_number(decorated_node) == 15

    # Test with empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 30

    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 30

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 40

    no_decorator_list = NoDecoratorList()
    assert get_first_line_number(no_decorator_list) == 40
```


# LLM-generated content at query #120
#--------------------------

def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        end_lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 7})]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 7
    
    # Test with a node that has multiple decorators, should return first decorator's lineno
    class MultiDecoratedNode:
        lineno = 20
        end_lineno = 25
        decorator_list = [
            type('Decorator1', (), {'lineno': 12}),
            type('Decorator2', (), {'lineno': 13})
        ]
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 12
    
    # Test edge case: empty decorator_list should return node.lineno
    class EmptyDecoratorNode:
        lineno = 30
        end_lineno = 35
        decorator_list = []
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 30


# LLM-generated content at query #121
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a function that has decorators
    class DecoratorNode:
        decorator_list = [type('Decorator', (), {'lineno': 2})()]
        lineno = 10
    node = DecoratorNode()
    assert get_first_line_number(node) == 2

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 15
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 15

    # Test with empty decorator list explicitly
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 20
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 20

    # Test with multiple decorators
    class MultiDecoratorNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 3})(),
        ]
        lineno = 8
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 1
```


# LLM-generated content at query #122
#--------------------------

```python
def test_get_first_line_number():
    # Test with a plain AST node without decorators
    import ast
    code = "x = 1"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with a function that has decorators
    code = "@decorator\ndef func():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with a decorated class
    code = "@decorator\nclass MyClass:\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with multiple decorators
    code = "@dec1\n@dec2\ndef func():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with decorators on different lines
    code = "@dec1\n@dec2\ndef func():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with node that has a different lineno
    code = "\n\nx = 1"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 3

    # Test with decorated function where decorator is on line 2
    code = "\n@decorator\ndef func():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 2
```


# LLM-generated content at query #123
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = [type('Decorator', (), {'lineno': 2})()]
        lineno = 10
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 2
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 15
    
    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 3})(),
            type('Decorator3', (), {'lineno': 4})()
        ]
        lineno = 20
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 1
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 8
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 8
```


# LLM-generated content at query #124
#--------------------------

```python
def test_get_first_line_number():
    # Test with a plain function that has no decorators
    class SimpleFunction:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = SimpleFunction()
    assert get_first_line_number(node) == 5
    
    # Test with a function that has decorators
    class DecoratedFunction:
        lineno = 10
        end_lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 1})()]
    
    node = DecoratedFunction()
    assert get_first_line_number(node) == 1
    
    # Test with a class that has multiple decorators
    class DecoratedClass:
        lineno = 15
        end_lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 2})(),
            type('Decorator2', (), {'lineno': 3})()
        ]
    
    node = DecoratedClass()
    assert get_first_line_number(node) == 2
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecorators:
        lineno = 25
        end_lineno = 35
    
    node = NoDecorators()
    assert get_first_line_number(node) == 25
    
    # Test with a node that has decorator_list but is empty
    class EmptyDecorators:
        lineno = 30
        end_lineno = 40
        decorator_list = []
    
    node = EmptyDecorators()
    assert get_first_line_number(node) == 30
    
    # Test edge case where decorator_list is None
    class NoneDecorators:
        lineno = 35
        end_lineno = 45
        decorator_list = None
    
    node = NoneDecorators()
    assert get_first_line_number(node) == 35
```


# LLM-generated content at query #125
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 10

    # Test with a function that has decorators
    class DecoratedNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    node = DecoratedNode()
    assert get_first_line_number(node) == 5

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 30
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 30

    # Test with multiple decorators, should return the first one's lineno
    class MultiDecoratedNode:
        lineno = 40
        decorator_list = [
            type('Decorator1', (), {'lineno': 15})(),
            type('Decorator2', (), {'lineno': 25})()
        ]
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 15


# LLM-generated content at query #126
#--------------------------

def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class MockNodeNoDecorators:
        lineno = 5
    
    node = MockNodeNoDecorators()
    assert get_first_line_number(node) == 5
    
    # Test case 2: Node with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self):
            self.decorator_list = [MockDecorator(2), MockDecorator(3)]
            self.lineno = 4
    
    node = MockNodeWithDecorators()
    assert get_first_line_number(node) == 2
    
    # Test case 3: Node with empty decorator list
    class MockNodeEmptyDecorators:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 10
    
    node = MockNodeEmptyDecorators()
    assert get_first_line_number(node) == 10


# LLM-generated content at query #127
#--------------------------

def test_get_first_line_number():
    # Test case 1: Node without decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test case 2: Node with decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})()]
        lineno = 15
    node = DecoratedNode()
    assert get_first_line_number(node) == 10
    
    # Test case 3: Empty decorator list
    class EmptyDecorators:
        decorator_list = []
        lineno = 20
    node = EmptyDecorators()
    assert get_first_line_number(node) == 20
    
    # Test case 4: Multiple decorators - should return first decorator's line
    class MultiDecorated:
        decorator_list = [
            type('Decorator1', (), {'lineno': 100})(),
            type('Decorator2', (), {'lineno': 200})()
        ]
        lineno = 300
    node = MultiDecorated()
    assert get_first_line_number(node) == 100


# LLM-generated content at query #128
#--------------------------

def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = []

    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]

    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5

    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        def __init__(self):
            self.lineno = 30
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 3})(),
                type('Decorator2', (), {'lineno': 7})()
            ]

    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 3


# LLM-generated content at query #129
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        lineno = 10
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 15
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 20
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 25
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 25
    
    # Test edge case with multiple decorators
    class MockDecorator1:
        lineno = 30
    
    class MockDecorator2:
        lineno = 35
    
    class MockNodeMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 40
    
    node_multiple_decorators = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 30
```


# LLM-generated content at query #130
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node having no decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node having decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
    
    node = DecoratedNode()
    node.decorator_list.append(type('Decorator', (), {'lineno': 5})())
    node.decorator_list.append(type('Decorator', (), {'lineno': 6})())
    
    assert get_first_line_number(node) == 5
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 30
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 30
    
    # Test with empty decorator_list
    class EmptyDecoratorList:
        decorator_list = []
        lineno = 40
    
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 40
```


# LLM-generated content at query #131
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self, decorator_lineno, node_lineno):
            self.decorator_list = [MockDecorator(decorator_lineno)]
            self.lineno = node_lineno
    
    node_with_decorators = MockNodeWithDecorators(10, 20)
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_without_decorators = MockNodeWithoutDecorators(30)
    assert get_first_line_number(node_without_decorators) == 30
    
    # Test with empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self, lineno):
            self.decorator_list = []
            self.lineno = lineno
    
    node_with_empty_decorators = MockNodeWithEmptyDecorators(40)
    assert get_first_line_number(node_with_empty_decorators) == 40
```


# LLM-generated content at query #132
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 20
    decorator_node = DecoratorNode()
    decorator_node.decorator_list = [type('Decorator', (), {'lineno': 15})()]
    assert get_first_line_number(decorator_node) == 15

    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 30
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 30

    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 40
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 40
```


# LLM-generated content at query #133
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self, decorator_list, lineno):
            self.decorator_list = decorator_list
            self.lineno = lineno
    
    decorator1 = MockDecorator(10)
    decorator2 = MockDecorator(11)
    node_with_decorators = MockNodeWithDecorators([decorator1, decorator2], 12)
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []
    
    node_without_decorators = MockNodeWithoutDecorators(15)
    assert get_first_line_number(node_without_decorators) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_no_attr = MockNodeNoDecoratorAttr(20)
    assert get_first_line_number(node_no_attr) == 20
```


# LLM-generated content at query #134
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Function without decorators
    class MockNode:
        lineno = 5
        decorator_list = []
    
    node = MockNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Function with decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno

    class MockNodeWithDecorator:
        def __init__(self):
            self.decorator_list = [MockDecorator(3)]
            self.lineno = 10

    node_with_decorator = MockNodeWithDecorator()
    assert get_first_line_number(node_with_decorator) == 3

    # Test case 3: Node without decorator_list attribute
    class MinimalNode:
        lineno = 7

    minimal_node = MinimalNode()
    assert get_first_line_number(minimal_node) == 7

    # Test case 4: Multiple decorators - should return first one
    class MockNodeWithMultipleDecorators:
        def __init__(self):
            self.decorator_list = [MockDecorator(1), MockDecorator(2)]
            self.lineno = 15

    multi_decorator_node = MockNodeWithMultipleDecorators()
    assert get_first_line_number(multi_decorator_node) == 1

    # Test case 5: Empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20

    empty_decorator_node = MockNodeWithEmptyDecorators()
    assert get_first_line_number(empty_decorator_node) == 20
```


# LLM-generated content at query #135
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has a decorator
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 8})(), 
                         type('Decorator', (), {'lineno': 9})()]
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 8
    
    # Test with a node that has no decorators
    class PlainNode:
        lineno = 15
        decorator_list = []
    
    node = PlainNode()
    assert get_first_line_number(node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 25
    
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 25
```


# LLM-generated content at query #136
#--------------------------

def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})()]
        lineno = 15
    node = DecoratorNode()
    assert get_first_line_number(node) == 10

    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 20})(),
            type('Decorator2', (), {'lineno': 25})()
        ]
        lineno = 30
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 20

    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 35
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 35

    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 40
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 40


# LLM-generated content at query #137
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
        decorator_list = []
        lineno = 10
    
    decorated_node = DecoratedNode()
    
    # Create mock decorator objects
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorated_node.decorator_list = [MockDecorator(3), MockDecorator(4)]
    
    assert get_first_line_number(decorated_node) == 3
    
    # Test with empty decorator_list
    decorated_node.decorator_list = []
    assert get_first_line_number(decorated_node) == 10
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttribute:
        lineno = 15
    
    no_decorator_node = NoDecoratorAttribute()
    assert get_first_line_number(no_decorator_node) == 15
```


# LLM-generated content at query #138
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
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
        lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorsNode:
        decorator_list = []
        lineno = 7
    
    empty_decorators_node = EmptyDecoratorsNode()
    assert get_first_line_number(empty_decorators_node) == 7
    
    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
        lineno = 15
    
    multi_decorator_node = MultiDecoratorNode()
    assert get_first_line_number(multi_decorator_node) == 1
```


# LLM-generated content at query #139
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 3
        end_lineno = 8
        decorator_list = [type('Decorator', (), {'lineno': 1})()]
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 1

    # Test with a node that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 7
        end_lineno = 12
    no_decorator_list = NoDecoratorList()
    assert get_first_line_number(no_decorator_list) == 7

    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecorators:
        lineno = 4
        end_lineno = 9
        decorator_list = [
            type('Decorator1', (), {'lineno': 2})(),
            type('Decorator2', (), {'lineno': 3})()
        ]
    multi_decorator = MultipleDecorators()
    assert get_first_line_number(multi_decorator) == 2


# LLM-generated content at query #140
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        end_lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
        end_lineno = 25
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 20

    # Test with multiple decorators
    class MultiDecoratedNode:
        lineno = 30
        end_lineno = 35
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1
```


# LLM-generated content at query #141
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})]
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3

    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 15
        decorator_list = [
            type('Decorator1', (), {'lineno': 7}),
            type('Decorator2', (), {'lineno': 8}),
            type('Decorator3', (), {'lineno': 9})
        ]
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 7

    # Test with a node that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 20
    no_decorator_list_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_list_node) == 20

    # Test with an empty decorator list (explicitly set)
    class EmptyDecoratorList:
        lineno = 25
        decorator_list = []
    empty_decorator_node = EmptyDecoratorList()
    assert get_first_line_number(empty_decorator_node) == 25


# LLM-generated content at query #142
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    node = DecoratedNode()
    assert get_first_line_number(node) == 3

    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 20
        decorator_list = [
            type('Decorator1', (), {'lineno': 7}),
            type('Decorator2', (), {'lineno': 8})
        ]()
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 7


# LLM-generated content at query #143
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
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 3})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 7
    
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 7
    
    # Test with empty decorator list
    class EmptyDecoratorNode:
        lineno = 12
        decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 12


# LLM-generated content at query #144
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 15
    
    node = DecoratedNode()
    node.decorator_list = [type('Decorator', (), {'lineno': 5})()]
    assert get_first_line_number(node) == 5
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 20
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 20


# LLM-generated content at query #145
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        def __init__(self):
            self.lineno = 30
    
    no_decorator_list = NoDecoratorList()
    assert get_first_line_number(no_decorator_list) == 30
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator', (), {'lineno': 1})(),
                type('Decorator', (), {'lineno': 2})()
            ]
            self.lineno = 3
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1
```


# LLM-generated content at query #146
#--------------------------

def test_get_first_line_number():
    # Test with a simple function node (no decorators)
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a function that has decorators
    class DecoratedNode:
        lineno = 15
        
        class Decorator:
            def __init__(self):
                self.lineno = 12
        
        decorator_list = [Decorator()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 12
    
    # Test with a class node that has multiple decorators
    class MultiDecoratedNode:
        lineno = 20
        
        class Decorator1:
            def __init__(self):
                self.lineno = 17
        
        class Decorator2:
            def __init__(self):
                self.lineno = 18
        
        decorator_list = [Decorator1(), Decorator2()]
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 17
    
    # Test with an object that has decorator_list as None
    class NoneDecoratorNode:
        lineno = 25
        decorator_list = None
    
    node = NoneDecoratorNode()
    assert get_first_line_number(node) == 25
    
    # Test with an object that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 30
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 30
    
    # Test with an empty decorator_list
    class EmptyDecoratorNode:
        lineno = 35
        decorator_list = []
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 35


# LLM-generated content at query #147
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        
    def create_decorator(lineno):
        class Decorator:
            pass
        decorator = Decorator()
        decorator.lineno = lineno
        return decorator
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [create_decorator(3), create_decorator(4)]
    decorated_node.lineno = 2
    assert get_first_line_number(decorated_node) == 3
    
    # Test with empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 7
    
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 7
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 9
    
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 9
    
    # Test with multiple decorators, ensuring first one is returned
    class MultipleDecoratorsNode:
        decorator_list = [create_decorator(1), create_decorator(2), create_decorator(3)]
        lineno = 0
    
    multi_node = MultipleDecoratorsNode()
    assert get_first_line_number(multi_node) == 1
```


# LLM-generated content at query #148
#--------------------------

```python
def test_get_first_line_number():
    # Test simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test function with decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5

    # Test node without decorator_list attribute
    class NoDecoratorList:
        lineno = 20
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 20

    # Test with empty decorator_list
    class EmptyDecoratorList:
        lineno = 30
        decorator_list = []
    empty_decorator_node = EmptyDecoratorList()
    assert get_first_line_number(empty_decorator_node) == 30
```


# LLM-generated content at query #149
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecoratorNode:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self, decorator_linenos, node_lineno):
            self.decorator_list = [MockDecoratorNode(l) for l in decorator_linenos]
            self.lineno = node_lineno
    
    node_with_decorators = MockNodeWithDecorators([10, 12], 15)
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []
    
    node_without_decorators = MockNodeWithoutDecorators(20)
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class SimpleMockNode:
        def __init__(self, lineno):
            self.lineno = lineno
    
    simple_node = SimpleMockNode(25)
    assert get_first_line_number(simple_node) == 25
    
    # Test with multiple decorators
    node_multiple_decorators = MockNodeWithDecorators([5, 7, 9], 11)
    assert get_first_line_number(node_multiple_decorators) == 5


# LLM-generated content at query #150
#--------------------------

def test_get_first_line_number():
    # Test case 1: Node with no decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 20
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test case 2: Node with decorators
    class DecoratedNode:
        lineno = 15
        end_lineno = 25
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test case 3: Node without decorator_list attribute
    class NoDecoratorAttr:
        lineno = 30
        end_lineno = 40
    
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 30
    
    # Test case 4: Multiple decorators - should return first decorator's lineno
    class MultipleDecorators:
        lineno = 50
        end_lineno = 60
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    
    node = MultipleDecorators()
    assert get_first_line_number(node) == 1


# LLM-generated content at query #151
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
        decorator_list = [
            type('Decorator', (), {'lineno': 2})(),
            type('Decorator', (), {'lineno': 3})()
        ]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 2
    
    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        lineno = 15
        decorator_list = []
    
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 15
    
    # Test with a node that has decorator_list attribute but no lineno attribute
    class NoLinenoNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 7})()
        ]
    
    no_lineno_node = NoLinenoNode()
    assert get_first_line_number(no_lineno_node) == 7
```


# LLM-generated content at query #152
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 10
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [type('Decorator', (), {'lineno': 3})()]
    assert get_first_line_number(decorated_node) == 3
    
    # Test with multiple decorators - should return first decorator's lineno
    decorated_node.decorator_list = [
        type('Decorator', (), {'lineno': 3})(),
        type('Decorator', (), {'lineno': 4})()
    ]
    assert get_first_line_number(decorated_node) == 3
    
    # Test with empty decorator_list
    decorated_node.decorator_list = []
    assert get_first_line_number(decorated_node) == 10


# LLM-generated content at query #153
#--------------------------

def test_get_first_line_number():
    # Test for a simple function without decorators
    node = type('Node', (), {'lineno': 10, 'decorator_list': []})()
    assert get_first_line_number(node) == 10

    # Test for a function with a single decorator
    decorator = type('Decorator', (), {'lineno': 5})()
    node = type('Node', (), {'lineno': 10, 'decorator_list': [decorator]})()
    assert get_first_line_number(node) == 5

    # Test for a function with multiple decorators
    decorator1 = type('Decorator', (), {'lineno': 3})()
    decorator2 = type('Decorator', (), {'lineno': 5})()
    node = type('Node', (), {'lineno': 10, 'decorator_list': [decorator1, decorator2]})()
    assert get_first_line_number(node) == 3

    # Test for a class without decorators
    node = type('Node', (), {'lineno': 20, 'decorator_list': []})()
    assert get_first_line_number(node) == 20

    # Test for a class with decorators
    decorator = type('Decorator', (), {'lineno': 15})()
    node = type('Node', (), {'lineno': 20, 'decorator_list': [decorator]})()
    assert get_first_line_number(node) == 15

    # Test for a node that doesn't have decorator_list attribute
    node = type('Node', (), {'lineno': 30})()
    assert get_first_line_number(node) == 30


# LLM-generated content at query #154
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test case 2: Node with decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
    
    decorator1 = SimpleNode()
    decorator1.lineno = 5
    decorator2 = SimpleNode()
    decorator2.lineno = 8
    
    node = DecoratedNode()
    node.decorator_list = [decorator1, decorator2]
    node.lineno = 20
    
    assert get_first_line_number(node) == 5
    
    # Test case 3: Node with empty decorator_list
    node = DecoratedNode()
    node.decorator_list = []
    node.lineno = 30
    
    assert get_first_line_number(node) == 30
    
    # Test case 4: Edge case - node without decorator attribute
    class NoDecoratorNode:
        lineno = 42
    
    node = NoDecoratorNode()
    assert get_first_line_number(node) == 42
```


# LLM-generated content at query #155
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 2})(),
            type('Decorator', (), {'lineno': 3})()
        ]
        lineno = 4
        end_lineno = 8
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 2
    
    # Test with empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 7
        end_lineno = 12
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 7
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 3
        end_lineno = 6
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 3
    
    # Test with multiple decorators to ensure it picks the first one
    class MultipleDecoratorsNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 1})(),
            type('Decorator', (), {'lineno': 5})(),
            type('Decorator', (), {'lineno': 9})()
        ]
        lineno = 10
        end_lineno = 15
    
    node = MultipleDecoratorsNode()
    assert get_first_line_number(node) == 1
```


# LLM-generated content at query #156
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    import ast
    code = '''
@decorator
def foo():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    # In Python 3.8+, decorated object's lineno points to def line
    # but we should return the decorator's line
    assert get_first_line_number(func_node) == 1  # line of @decorator
    
    # Test with a function without decorators
    code2 = '''
def bar():
    pass
'''
    tree2 = ast.parse(code2)
    func_node2 = tree2.body[0]
    assert get_first_line_number(func_node2) == 1  # line of def
    
    # Test with a class with decorators
    code3 = '''
@decorator
class MyClass:
    pass
'''
    tree3 = ast.parse(code3)
    class_node = tree3.body[0]
    assert get_first_line_number(class_node) == 1  # line of @decorator
    
    # Test with a class without decorators
    code4 = '''
class MyClass2:
    pass
'''
    tree4 = ast.parse(code4)
    class_node2 = tree4.body[0]
    assert get_first_line_number(class_node2) == 1  # line of class
    
    # Test with multiple decorators - should return first decorator's line
    code5 = '''
@decorator1
@decorator2
def baz():
    pass
'''
    tree5 = ast.parse(code5)
    func_node5 = tree5.body[0]
    assert get_first_line_number(func_node5) == 1  # line of @decorator1
    
    # Test that the function works with different line positions
    code6 = '''
# comment
@decorator
def qux():
    pass
'''
    tree6 = ast.parse(code6)
    func_node6 = tree6.body[0]
    assert get_first_line_number(func_node6) == 2  # line of @decorator (comment is line 1)
```


# LLM-generated content at query #157
#--------------------------

def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
        lineno = 10
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has multiple decorators
    class MultiDecoratedNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 3})(),
            type('Decorator2', (), {'lineno': 4})()
        ]
        lineno = 10
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 3
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 15


# LLM-generated content at query #158
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self, lineno, decorator_linenos):
            self.lineno = lineno
            self.decorator_list = [MockDecorator(l) for l in decorator_linenos]
    
    node_with_decorators = MockNodeWithDecorators(10, [5, 6])
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []
    
    node_without_decorators = MockNodeWithoutDecorators(20)
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorsAttr:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_no_decorators_attr = MockNodeNoDecoratorsAttr(30)
    assert get_first_line_number(node_no_decorators_attr) == 30


# LLM-generated content at query #159
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [SimpleNode(5), SimpleNode(6)]
            self.lineno = 10
    
    class SimpleNode:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorated = DecoratedNode()
    assert get_first_line_number(decorated) == 5
    
    # Test with a node without decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    plain = PlainNode()
    assert get_first_line_number(plain) == 20
    
    # Test with a node that has decorator_list attribute but is empty
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 30
    
    empty_decorator = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator) == 30
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 40
    
    no_attr = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr) == 40
```


# LLM-generated content at query #160
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Function with no decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test case 2: Function with decorators
    class DecoratedNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test case 3: Function with multiple decorators - should return first decorator's lineno
    class MultiDecoratedNode:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 1}),
            type('Decorator2', (), {'lineno': 2}),
            type('Decorator3', (), {'lineno': 3})
        ]
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 1
    
    # Test case 4: Node without decorator_list attribute
    class NoDecoratorAttr:
        lineno = 40
    
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 40
    
    # Test case 5: Edge case - empty decorator list
    class EmptyDecoratorList:
        lineno = 50
        decorator_list = []
    
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 50
```


# LLM-generated content at query #161
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a function with decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 7})()]
    node = DecoratorNode()
    assert get_first_line_number(node) == 7

    # Test with a class without decorators
    class SimpleClass:
        lineno = 3
        decorator_list = []
    node = SimpleClass()
    assert get_first_line_number(node) == 3

    # Test with a class with multiple decorators
    class DecoratedClass:
        lineno = 15
        decorator_list = [
            type('Decorator1', (), {'lineno': 12}),
            type('Decorator2', (), {'lineno': 13})
        ]
    node = DecoratedClass()
    assert get_first_line_number(node) == 12


# LLM-generated content at query #162
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.lineno = 15
            self.end_lineno = 20
            self.decorator_list = [type('Decorator', (), {'lineno': 12})()]
    
    decorator_node = DecoratorNode()
    assert get_first_line_number(decorator_node) == 12
    
    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        def __init__(self):
            self.lineno = 30
            self.end_lineno = 35
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 25})(),
                type('Decorator2', (), {'lineno': 27})()
            ]
    
    multi_dec_node = MultiDecoratorNode()
    assert get_first_line_number(multi_dec_node) == 25
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorListAttribute:
        lineno = 40
        end_lineno = 45
    
    no_dec_list_node = NoDecoratorListAttribute()
    assert get_first_line_number(no_dec_list_node) == 40
```


# LLM-generated content at query #163
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 7
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 2})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 2
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 15
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15
    
    # Test with a node that has decorator_list attribute but it's not a list
    class NonListDecoratorNode:
        def __init__(self):
            self.decorator_list = None
            self.lineno = 20
    
    non_list_node = NonListDecoratorNode()
    assert get_first_line_number(non_list_node) == 20
```


# LLM-generated content at query #164
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10

    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5

    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20

    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 20

    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 30

    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 30

    # Test with multiple decorators (should return first decorator's line)
    class MultipleDecoratorsNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 1})(),
                type('Decorator2', (), {'lineno': 2})()
            ]
            self.lineno = 100

    multiple_decorators_node = MultipleDecoratorsNode()
    assert get_first_line_number(multiple_decorators_node) == 1
```


# LLM-generated content at query #165
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test case 2: Node with decorators
    class DecoratedNode:
        lineno = 10
        end_lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 3
    
    # Test case 3: Node without decorator_list attribute
    class NoDecoratorAttr:
        lineno = 7
        end_lineno = 12
    
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 7
    
    # Test case 4: Multiple decorators - should return first decorator's lineno
    class MultiDecoratorNode:
        lineno = 15
        end_lineno = 25
        decorator_list = [
            type('Decorator', (), {'lineno': 1})(),
            type('Decorator', (), {'lineno': 2})()
        ]
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 1
```


# LLM-generated content at query #166
#--------------------------

```python
def test_get_first_line_number():
    # Test with node that has decorators
    class DecoratedNode:
        def __init__(self, lineno, decorator_list):
            self.lineno = lineno
            self.decorator_list = decorator_list
    
    class Decorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_with_decorators = DecoratedNode(10, [Decorator(5), Decorator(6)])
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test with node that has no decorators
    class SimpleNode:
        def __init__(self, lineno):
            self.lineno = lineno
            self.decorator_list = []
    
    node_without_decorators = SimpleNode(15)
    assert get_first_line_number(node_without_decorators) == 15
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_without_attr = NoDecoratorAttrNode(20)
    assert get_first_line_number(node_without_attr) == 20
    
    # Test with empty decorator list
    node_with_empty_decorators = SimpleNode(25)
    node_with_empty_decorators.decorator_list = []
    assert get_first_line_number(node_with_empty_decorators) == 25


# LLM-generated content at query #167
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test case 2: Node with decorators
    class DecoratorNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 5
    
    # Test case 3: Node with multiple decorators, should return first decorator's lineno
    class MultiDecoratorNode:
        lineno = 10
        decorator_list = [
            type('Decorator1', (), {'lineno': 3})(),
            type('Decorator2', (), {'lineno': 4})(),
            type('Decorator3', (), {'lineno': 5})()
        ]
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 3
```


# LLM-generated content at query #168
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    import ast
    code = "def foo():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with a function that has decorators
    code = "@decorator\ndef foo():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with multiple decorators
    code = "@decorator1\n@decorator2\ndef foo():\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with decorators and preceding code
    code = "x = 1\n@decorator\ndef foo():\n    pass"
    tree = ast.parse(code)
    node = tree.body[1]
    assert get_first_line_number(node) == 2

    # Test with a class without decorators
    code = "class MyClass:\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with a class with decorators
    code = "@decorator\nclass MyClass:\n    pass"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with a node that doesn't have decorator_list attribute
    code = "x = 1"
    tree = ast.parse(code)
    node = tree.body[0]
    assert get_first_line_number(node) == 1

    # Test with a node that has decorator_list but is not a function or class
    # (e.g., a decorated variable - though this is not valid Python, the function should handle it)
    # Create a mock node
    class MockNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    mock_node = MockNode()
    assert get_first_line_number(mock_node) == 5

    # Test with a node that has empty decorator_list
    class MockNode2:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 7
    mock_node2 = MockNode2()
    assert get_first_line_number(mock_node2) == 7


# LLM-generated content at query #169
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorator_list attribute but empty
    class NodeWithEmptyDecorators:
        lineno = 20
        decorator_list = []
    
    node_empty = NodeWithEmptyDecorators()
    assert get_first_line_number(node_empty) == 20
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 30
    
    class Decorator:
        lineno = 5
    
    decorated_node = DecoratorNode()
    decorated_node.decorator_list = [Decorator()]
    assert get_first_line_number(decorated_node) == 5
    
    # Test with multiple decorators - should return first decorator's line
    class MultipleDecoratorNode:
        lineno = 40
    
    decorator1 = Decorator()
    decorator1.lineno = 7
    decorator2 = Decorator()
    decorator2.lineno = 8
    
    multi_decorated = MultipleDecoratorNode()
    multi_decorated.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(multi_decorated) == 7


# LLM-generated content at query #170
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 10
    
    # Create a mock decorator with a lineno attribute
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [MockDecorator(3), MockDecorator(4)]
    assert get_first_line_number(decorated_node) == 3

    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 15

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 20
    no_decorator_attr = NoDecoratorAttr()
    assert get_first_line_number(no_decorator_attr) == 20


# LLM-generated content at query #171
#--------------------------

def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.lineno = 20
            self.end_lineno = 25
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 5
    
    # Test with a function definition with decorators
    def decorated_func():
        pass
    decorated_func = decorator(decorated_func)
    
    # Test with a class definition with decorators
    @decorator
    class DecoratedClass:
        pass
    
    # Test edge case where decorator_list is empty but attribute exists
    class EmptyDecoratorNode:
        lineno = 30
        end_lineno = 35
        decorator_list = []
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 30


# LLM-generated content at query #172
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a function that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 20
    
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 20
    
    # Test with empty decorator list explicitly set
    node_with_empty_decorators = type('Node', (), {'lineno': 15, 'decorator_list': []})()
    assert get_first_line_number(node_with_empty_decorators) == 15
```


# LLM-generated content at query #173
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 5}),
            type('Decorator', (), {'lineno': 7})
        ]
        lineno = 10
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 20
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 30
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 30
    
    # Test with a node that has decorator_list as None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 40
    
    node = NoneDecoratorNode()
    assert get_first_line_number(node) == 40
    
    # Test with multiple decorators to ensure first one is returned
    class MultipleDecoratorsNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 1}),
            type('Decorator', (), {'lineno': 2}),
            type('Decorator', (), {'lineno': 3})
        ]
        lineno = 100
    
    node = MultipleDecoratorsNode()
    assert get_first_line_number(node) == 1
```


# LLM-generated content at query #174
#--------------------------

```python
def test_get_first_line_number():
    # Test when node has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 5
    
    # Test when node has no decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 10
    
    node_without_decorators = PlainNode()
    assert get_first_line_number(node_without_decorators) == 10
    
    # Test when decorator_list attribute doesn't exist
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 10
    
    node_no_attr = NoDecoratorAttrNode()
    assert get_first_line_number(node_no_attr) == 10
```


# LLM-generated content at query #175
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_node = DecoratorNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that has multiple decorators - should return first decorator's line
    class MultiDecoratorNode:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = [
                type('Decorator', (), {'lineno': 7})(),
                type('Decorator', (), {'lineno': 8})()
            ]
    
    multi_decorated_node = MultiDecoratorNode()
    assert get_first_line_number(multi_decorated_node) == 7
    
    # Test with an empty decorator list
    class EmptyDecoratorNode:
        def __init__(self):
            self.lineno = 15
            self.decorator_list = []
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15
```


# LLM-generated content at query #176
#--------------------------

```python
def test_get_first_line_number():
    # Test with a function that has a decorator
    class MockNodeWithDecorator:
        lineno = 10
        decorator_list = [type('MockDecorator', (), {'lineno': 5})()]
    
    node_with_decorator = MockNodeWithDecorator()
    assert get_first_line_number(node_with_decorator) == 5
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorator:
        lineno = 20
        decorator_list = []
    
    node_without_decorator = MockNodeWithoutDecorator()
    assert get_first_line_number(node_without_decorator) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 30
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 30
    
    # Test with multiple decorators - should return first decorator's line
    class MockNodeMultipleDecorators:
        lineno = 40
        decorator_list = [
            type('MockDecorator1', (), {'lineno': 1})(),
            type('MockDecorator2', (), {'lineno': 2})()
        ]
    
    node_multiple_decorators = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 1
```


# LLM-generated content at query #177
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        lineno = 10
        decorator_list = []
    simple_func = SimpleFunction()
    assert get_first_line_number(simple_func) == 10

    # Test with a function that has decorators
    class DecoratedFunction:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    decorated_func = DecoratedFunction()
    assert get_first_line_number(decorated_func) == 15

    # Test with a class that has decorators
    class DecoratedClass:
        lineno = 30
        decorator_list = [type('Decorator', (), {'lineno': 25})()]
    decorated_class = DecoratedClass()
    assert get_first_line_number(decorated_class) == 25

    # Test with an object that has no decorator_list attribute
    class NoDecorators:
        lineno = 40
    no_decorators = NoDecorators()
    assert get_first_line_number(no_decorators) == 40

    # Test with an object that has an empty decorator_list
    class EmptyDecorators:
        lineno = 50
        decorator_list = []
    empty_decorators = EmptyDecorators()
    assert get_first_line_number(empty_decorators) == 50

    # Test with a node that has multiple decorators
    class MultipleDecorators:
        lineno = 60
        decorator_list = [
            type('Decorator1', (), {'lineno': 55}),
            type('Decorator2', (), {'lineno': 56})
        ]
    multiple_decorators = MultipleDecorators()
    assert get_first_line_number(multiple_decorators) == 55
```


# LLM-generated content at query #178
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        pass
    
    # Create a mock node without decorator_list
    class MockNode:
        lineno = 10
    
    node = MockNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has decorator_list
    class DecoratedNode:
        decorator_list = []
        lineno = 15
    
    node = DecoratedNode()
    node.decorator_list = [type('Decorator', (), {'lineno': 5})()]
    assert get_first_line_number(node) == 5
    
    # Test with empty decorator_list
    node = DecoratedNode()
    node.decorator_list = []
    assert get_first_line_number(node) == 15
    
    # Test with multiple decorators (should return first decorator's lineno)
    node = DecoratedNode()
    node.decorator_list = [
        type('Decorator', (), {'lineno': 3})(),
        type('Decorator', (), {'lineno': 7})()
    ]
    assert get_first_line_number(node) == 3
    
    # Test with a real AST node (simulating an ast.FunctionDef)
    import ast
    code = '''
@decorator
def foo():
    pass
'''
    tree = ast.parse(code)
    function_def = tree.body[0]
    assert get_first_line_number(function_def) == 2  # line of @decorator
    
    # Test with a simple function without decorators in real AST
    code = '''
def bar():
    pass
'''
    tree = ast.parse(code)
    function_def = tree.body[0]
    assert get_first_line_number(function_def) == 2  # line of def
    
    # Test edge case: node without lineno attribute
    class NoLineNo:
        pass
    
    node = NoLineNo()
    # Should raise AttributeError
    try:
        get_first_line_number(node)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #179
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 20
    
    decorator_node = DecoratorNode()
    assert get_first_line_number(decorator_node) == 5

    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 30
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 30

    # Test with a node that has decorator_list attribute set to None
    class NoneDecoratorNode:
        def __init__(self):
            self.decorator_list = None
            self.lineno = 40
    
    none_decorator_node = NoneDecoratorNode()
    assert get_first_line_number(none_decorator_node) == 40

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttributeNode:
        lineno = 50
    
    no_decorator_attr_node = NoDecoratorAttributeNode()
    assert get_first_line_number(no_decorator_attr_node) == 50
```


# LLM-generated content at query #180
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10

    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    decorator_node = DecoratorNode()
    assert get_first_line_number(decorator_node) == 5

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 30
    no_decorator_list = NoDecoratorList()
    assert get_first_line_number(no_decorator_list) == 30
```


# LLM-generated content at query #181
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node without decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class SimpleNode:
        def __init__(self):
            self.lineno = 30
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 30
    
    # Test with multiple decorators
    class MultiDecoratedNode:
        def __init__(self):
            self.decorator_list = [
                type('Decorator1', (), {'lineno': 1})(),
                type('Decorator2', (), {'lineno': 2})()
            ]
            self.lineno = 15
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 1
```


# LLM-generated content at query #182
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a function that has decorators
    class DecoratorNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 30
    
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 30
    
    # Test with empty decorator_list
    class EmptyDecoratorNode:
        lineno = 40
        decorator_list = []
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 40
    
    # Test with multiple decorators, should return first one's lineno
    class MultiDecoratorNode:
        lineno = 50
        decorator_list = [
            type('Decorator1', (), {'lineno': 45}),
            type('Decorator2', (), {'lineno': 46}),
            type('Decorator3', (), {'lineno': 47})
        ]
    
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 45
```


# LLM-generated content at query #183
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 10})()]
            self.lineno = 15
    
    node_with_decorators = DecoratedNode()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node without decorators
    class PlainNode:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = []
    
    node_without_decorators = PlainNode()
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorList:
        def __init__(self):
            self.lineno = 25
    
    node_no_decorator_list = NoDecoratorList()
    assert get_first_line_number(node_no_decorator_list) == 25
    
    # Test with a node that has an empty decorator_list
    node_empty_decorators = PlainNode()
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #184
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test case 2: Node with decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 20
        
        def __init__(self):
            self.decorator_list = [
                type('Decorator', (), {'lineno': 5})(),
                type('Decorator', (), {'lineno': 6})()
            ]
    
    decorator_node = DecoratorNode()
    assert get_first_line_number(decorator_node) == 5
    
    # Test case 3: Node with empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 30
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 30
    
    # Test case 4: Node without decorator_list attribute
    class NoDecoratorAttribute:
        lineno = 40
    
    no_decorator_attr = NoDecoratorAttribute()
    assert get_first_line_number(no_decorator_attr) == 40
    
    # Test case 5: Node with decorators but decorator list is None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 50
    
    none_decorator_node = NoneDecoratorNode()
    assert get_first_line_number(none_decorator_node) == 50
```


# LLM-generated content at query #185
#--------------------------

```python
def test_get_first_line_number():
    # Test case with no decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test case with decorators
    class DecoratedNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 10})(),
            type('Decorator', (), {'lineno': 11})()
        ]
        lineno = 12
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 10
    
    # Test case with no decorator_list attribute
    class MinimalNode:
        lineno = 20
    
    node = MinimalNode()
    assert get_first_line_number(node) == 20
    
    # Test case with empty decorator_list
    class EmptyDecoratorsNode:
        decorator_list = []
        lineno = 30
    
    node = EmptyDecoratorsNode()
    assert get_first_line_number(node) == 30
```


# LLM-generated content at query #186
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    import ast
    code = '''
@decorator
def my_function():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # line of decorator
    
    # Test with a non-decorated function
    code = '''
def my_function():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # line of function definition
    
    # Test with a decorated class
    code = '''
@decorator
class MyClass:
    pass
'''
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2  # line of decorator
    
    # Test with a non-decorated class
    code = '''
class MyClass:
    pass
'''
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 2  # line of class definition
    
    # Test with multiple decorators
    code = '''
@decorator1
@decorator2
def my_function():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 2  # line of first decorator
    
    # Test with a node that has no decorator_list attribute
    code = '''
x = 1
'''
    tree = ast.parse(code)
    assign_node = tree.body[0]
    assert get_first_line_number(assign_node) == 2  # line of assignment
```


# LLM-generated content at query #187
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockFunction:
        def __init__(self, lineno, decorator_list=None):
            self.lineno = lineno
            self.decorator_list = decorator_list or []
    
    # Test with decorators - should return first decorator's line
    decorator1 = MockDecorator(10)
    decorator2 = MockDecorator(15)
    node = MockFunction(20, [decorator1, decorator2])
    assert get_first_line_number(node) == 10
    
    # Test with single decorator
    decorator = MockDecorator(5)
    node = MockFunction(8, [decorator])
    assert get_first_line_number(node) == 5
    
    # Test without decorators - should return node's line
    node = MockFunction(30)
    assert get_first_line_number(node) == 30
    
    # Test with empty decorator list
    node = MockFunction(42, [])
    assert get_first_line_number(node) == 42
    
    # Test edge case with decorator list attribute missing
    class MockNodeNoDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node = MockNodeNoDecorators(100)
    assert get_first_line_number(node) == 100
```


# LLM-generated content at query #188
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.lineno = 10
            self.decorator_list = [SimpleNode(5), SimpleNode(6)]
    
    class SimpleNode:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.lineno = 20
            self.decorator_list = []
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 30
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 30
    
    # Test with multiple decorators - should return first decorator's line
    class MultiDecoratedNode:
        def __init__(self):
            self.lineno = 50
            self.decorator_list = [SimpleNode(40), SimpleNode(41), SimpleNode(42)]
    
    multi_decorated_node = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated_node) == 40
```


# LLM-generated content at query #189
#--------------------------

def test_get_first_line_number():
    # Test case 1: Function with decorators
    node = type('Node', (), {
        'decorator_list': [type('Decorator', (), {'lineno': 10})],
        'lineno': 15
    })()
    assert get_first_line_number(node) == 10

    # Test case 2: Function without decorators
    node = type('Node', (), {
        'decorator_list': [],
        'lineno': 20
    })()
    assert get_first_line_number(node) == 20

    # Test case 3: Class with decorators
    node = type('Node', (), {
        'decorator_list': [type('Decorator', (), {'lineno': 5})],
        'lineno': 8
    })()
    assert get_first_line_number(node) == 5

    # Test case 4: Function without decorator_list attribute
    node = type('Node', (), {
        'lineno': 30
    })()
    assert get_first_line_number(node) == 30

    # Test case 5: Multiple decorators
    node = type('Node', (), {
        'decorator_list': [
            type('Decorator', (), {'lineno': 1}),
            type('Decorator', (), {'lineno': 2})
        ],
        'lineno': 3
    })()
    assert get_first_line_number(node) == 1


# LLM-generated content at query #190
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
        lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 8
    
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 8


# LLM-generated content at query #191
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Node without decorators
    class SimpleNode:
        lineno = 5
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test case 2: Node with decorators
    class DecoratedNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 3})(),
            type('Decorator', (), {'lineno': 4})()
        ]
        lineno = 10
    node = DecoratedNode()
    assert get_first_line_number(node) == 3

    # Test case 3: Node with empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 7
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 7

    # Test case 4: Node without decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 12
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 12

    # Test case 5: Multiple decorators, ensure first one is used
    class MultiDecoratorNode:
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})(),
            type('Decorator3', (), {'lineno': 3})()
        ]
        lineno = 20
    node = MultiDecoratorNode()
    assert get_first_line_number(node) == 1
```


# LLM-generated content at query #192
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 30
    
    no_decorator_node = NoDecoratorList()
    assert get_first_line_number(no_decorator_node) == 30
    
    # Test with multiple decorators (should return first decorator's lineno)
    class MultipleDecorators:
        lineno = 40
        decorator_list = [
            type('Decorator', (), {'lineno': 35})(),
            type('Decorator', (), {'lineno': 36})()
        ]
    
    multi_decorator_node = MultipleDecorators()
    assert get_first_line_number(multi_decorator_node) == 35
    
    # Test with empty decorator list
    class EmptyDecorators:
        lineno = 50
        decorator_list = []
    
    empty_decorator_node = EmptyDecorators()
    assert get_first_line_number(empty_decorator_node) == 50
```


# LLM-generated content at query #193
#--------------------------

```python
def test_get_first_line_number():
    # Test with a regular function node (no decorators)
    class FakeNode:
        lineno = 10
        decorator_list = []
    node = FakeNode()
    assert get_first_line_number(node) == 10

    # Test with a function node that has decorators
    class FakeDecorator:
        lineno = 5
    class FakeNodeWithDecorators:
        lineno = 10
        decorator_list = [FakeDecorator()]
    node = FakeNodeWithDecorators()
    assert get_first_line_number(node) == 5

    # Test with a node that doesn't have decorator_list attribute
    class SimpleNode:
        lineno = 20
    node = SimpleNode()
    assert get_first_line_number(node) == 20

    # Test with multiple decorators - should return first decorator's lineno
    class FakeDecorator2:
        lineno = 8
    class FakeNodeMultipleDecorators:
        lineno = 10
        decorator_list = [FakeDecorator(), FakeDecorator2()]
    node = FakeNodeMultipleDecorators()
    assert get_first_line_number(node) == 5
```


# LLM-generated content at query #194
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class MockDecoratedNode:
        def __init__(self):
            self.decorator_list = [MockDecorator(10)]
            self.lineno = 15
        class MockDecorator:
            def __init__(self, line):
                self.lineno = line

    node = MockDecoratedNode()
    assert get_first_line_number(node) == 10

    # Test with a node without decorators
    class MockPlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 20

    plain_node = MockPlainNode()
    assert get_first_line_number(plain_node) == 20

    # Test with a node that doesn't have decorator_list attribute
    class MockNoDecorators:
        def __init__(self):
            self.lineno = 25

    no_decorators_node = MockNoDecorators()
    assert get_first_line_number(no_decorators_node) == 25
```


# LLM-generated content at query #195
#--------------------------

def test_get_first_line_number():
    # Test with a node that has no decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 5})()]
    
    decorator_node = DecoratorNode()
    assert get_first_line_number(decorator_node) == 5
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
    
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 20
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultipleDecoratorsNode:
        lineno = 25
        decorator_list = [
            type('Decorator1', (), {'lineno': 1})(),
            type('Decorator2', (), {'lineno': 2})()
        ]
    
    multi_decorator_node = MultipleDecoratorsNode()
    assert get_first_line_number(multi_decorator_node) == 1


# LLM-generated content at query #196
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    class MockNodeWithDecorators:
        def __init__(self, decorator_lineno, node_lineno):
            self.decorator_list = [MockDecorator(decorator_lineno)]
            self.lineno = node_lineno
    
    node_with_decorators = MockNodeWithDecorators(10, 20)
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has multiple decorators
    class MockNodeWithMultipleDecorators:
        def __init__(self, decorator_lines, node_lineno):
            self.decorator_list = [MockDecorator(line) for line in decorator_lines]
            self.lineno = node_lineno
    
    node_with_multiple_decorators = MockNodeWithMultipleDecorators([5, 6, 7], 20)
    assert get_first_line_number(node_with_multiple_decorators) == 5
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        def __init__(self, lineno):
            self.lineno = lineno
    
    node_without_decorators = MockNodeWithoutDecorators(30)
    assert get_first_line_number(node_without_decorators) == 30
    
    # Test with a node that has an empty decorator list
    class MockNodeWithEmptyDecorators:
        def __init__(self, lineno):
            self.decorator_list = []
            self.lineno = lineno
    
    node_with_empty_decorators = MockNodeWithEmptyDecorators(40)
    assert get_first_line_number(node_with_empty_decorators) == 40
    
    # Test with a real AST node (if ast module is available)
    import ast
    code = "@decorator\ndef func():\n    pass\n"
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1
    
    code_no_decorator = "def func():\n    pass\n"
    tree_no_decorator = ast.parse(code_no_decorator)
    func_node_no_decorator = tree_no_decorator.body[0]
    assert get_first_line_number(func_node_no_decorator) == 1
```


# LLM-generated content at query #197
#--------------------------

```python
def test_get_first_line_number():
    # Test with a node that has decorators
    class MockDecorator:
        lineno = 10
    
    class MockNodeWithDecorators:
        decorator_list = [MockDecorator()]
        lineno = 5
    
    node_with_decorators = MockNodeWithDecorators()
    assert get_first_line_number(node_with_decorators) == 10
    
    # Test with a node that has no decorators
    class MockNodeWithoutDecorators:
        decorator_list = []
        lineno = 7
    
    node_without_decorators = MockNodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 7
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 3
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 3
    
    # Test with multiple decorators
    class MockDecorator1:
        lineno = 1
    
    class MockDecorator2:
        lineno = 2
    
    class MockNodeMultipleDecorators:
        decorator_list = [MockDecorator1(), MockDecorator2()]
        lineno = 4
    
    node_multiple_decorators = MockNodeMultipleDecorators()
    assert get_first_line_number(node_multiple_decorators) == 1
```


# LLM-generated content at query #198
#--------------------------

def test_get_first_line_number():
    # Test with a simple function node
    class SimpleNode:
        lineno = 10
        end_lineno = 15
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a decorated function node
    class DecoratedNode:
        lineno = 20
        end_lineno = 25
        decorator_list = [type('Decorator', (), {'lineno': 18})()]
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 18
    
    # Test with a node that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 30
        end_lineno = 35
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 30
    
    # Test with multiple decorators
    class MultipleDecorators:
        lineno = 40
        end_lineno = 45
        decorator_list = [
            type('Decorator1', (), {'lineno': 35})(),
            type('Decorator2', (), {'lineno': 36})()
        ]
    
    node = MultipleDecorators()
    assert get_first_line_number(node) == 35


# LLM-generated content at query #199
#--------------------------

def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
        end_lineno = 10
        decorator_list = []
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.lineno = 10
            self.end_lineno = 15
            self.decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 3
    
    # Test with a node where decorator_list attribute doesn't exist
    class NoDecoratorAttr:
        lineno = 20
        end_lineno = 25
    
    no_decorator_node = NoDecoratorAttr()
    assert get_first_line_number(no_decorator_node) == 20


# LLM-generated content at query #200
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 10

    # Test with a decorated function
    class DecoratedNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    node = DecoratedNode()
    assert get_first_line_number(node) == 15

    # Test with multiple decorators
    class MultiDecoratedNode:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 22}),
            type('Decorator2', (), {'lineno': 25})
        ]
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 22

    # Test with class that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 40
    node = NoDecoratorList()
    assert get_first_line_number(node) == 40


# LLM-generated content at query #201
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node (no decorators)
    class SimpleNode:
        lineno = 42
        decorator_list = []
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 42

    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 100
    decorated_node = DecoratedNode()
    decorator1 = type('Decorator', (), {'lineno': 50})()
    decorator2 = type('Decorator', (), {'lineno': 55})()
    decorated_node.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(decorated_node) == 50

    # Test with an empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 10
    empty_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_node) == 10

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 7
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 7

    # Test with actual AST nodes
    import ast
    code = "def foo():\n    pass"
    tree = ast.parse(code)
    func_def = tree.body[0]
    assert get_first_line_number(func_def) == 1

    code_with_decorator = "@decorator\ndef foo():\n    pass"
    tree_with_decorator = ast.parse(code_with_decorator)
    decorated_func = tree_with_decorator.body[0]
    assert get_first_line_number(decorated_func) == 1
```


# LLM-generated content at query #202
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 10
    
    # Create a mock decorator with lineno attribute
    class MockDecorator:
        def __init__(self, lineno):
            self.lineno = lineno
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [MockDecorator(3), MockDecorator(4)]
    assert get_first_line_number(decorated_node) == 3
    
    # Test with empty decorator list
    node_with_empty_decorators = SimpleNode()
    node_with_empty_decorators.decorator_list = []
    assert get_first_line_number(node_with_empty_decorators) == 5
    
    # Test with node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 7
    no_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_attr_node) == 7
    
    # Test with a node that has decorator_list attribute set to None
    class NoneDecoratorNode:
        decorator_list = None
        lineno = 8
    none_node = NoneDecoratorNode()
    assert get_first_line_number(none_node) == 8
```


# LLM-generated content at query #203
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        lineno = 5
        decorator_list = []
    
    node = SimpleFunction()
    assert get_first_line_number(node) == 5
    
    # Test with a function with decorators
    class DecoratedFunction:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 7})()]
    
    node = DecoratedFunction()
    assert get_first_line_number(node) == 7
    
    # Test with a function that has multiple decorators
    class MultiDecoratedFunction:
        lineno = 15
        decorator_list = [
            type('Decorator1', (), {'lineno': 8}),
            type('Decorator2', (), {'lineno': 9})
        ]
    
    node = MultiDecoratedFunction()
    assert get_first_line_number(node) == 8
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorList:
        lineno = 20
    
    node = NoDecoratorList()
    assert get_first_line_number(node) == 20


# LLM-generated content at query #204
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    node = SimpleNode()
    assert get_first_line_number(node) == 10

    # Test with a node that has decorators
    class DecoratorNode:
        decorator_list = []
        lineno = 20
        
        class Decorator:
            lineno = 15
    
    node = DecoratorNode()
    node.decorator_list = [DecoratorNode.Decorator()]
    assert get_first_line_number(node) == 15

    # Test with a node that has multiple decorators
    class MultiDecoratorNode:
        decorator_list = []
        lineno = 30
        
        class Decorator1:
            lineno = 22
        
        class Decorator2:
            lineno = 25
    
    node = MultiDecoratorNode()
    node.decorator_list = [MultiDecoratorNode.Decorator1(), MultiDecoratorNode.Decorator2()]
    assert get_first_line_number(node) == 22

    # Test with a node that has empty decorator_list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 40
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 40

    # Test with a node that has no decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 50
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 50
```


# LLM-generated content at query #205
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a function that has decorators
    class DecoratedNode:
        lineno = 15
        decorator_list = [type('Decorator', (), {'lineno': 12})()]
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 12
    
    # Test with a node that doesn't have decorator_list attribute
    class NodeWithoutDecorators:
        lineno = 20
    
    node_without_decorators = NodeWithoutDecorators()
    assert get_first_line_number(node_without_decorators) == 20
    
    # Test with multiple decorators - should return first decorator's lineno
    class MultiDecoratedNode:
        lineno = 30
        decorator_list = [
            type('Decorator1', (), {'lineno': 25})(),
            type('Decorator2', (), {'lineno': 26})()
        ]
    
    multi_decorated = MultiDecoratedNode()
    assert get_first_line_number(multi_decorated) == 25
```


# LLM-generated content at query #206
#--------------------------

```python
def test_get_first_line_number():
    # Test case: node without decorators
    class SimpleNode:
        lineno = 5
    
    node = SimpleNode()
    assert get_first_line_number(node) == 5
    
    # Test case: node with decorators
    class DecoratedNode:
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
        lineno = 10
    
    node = DecoratedNode()
    assert get_first_line_number(node) == 3
    
    # Test case: node with multiple decorators, should return first decorator's lineno
    class MultiDecoratedNode:
        decorator_list = [
            type('Decorator', (), {'lineno': 2})(),
            type('Decorator', (), {'lineno': 4})()
        ]
        lineno = 8
    
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 2
    
    # Test case: node with empty decorator list
    class EmptyDecoratorNode:
        decorator_list = []
        lineno = 15
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 15
    
    # Test case: node without decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 20
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 20
```


# LLM-generated content at query #207
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
    
    node = SimpleNode()
    assert get_first_line_number(node) == 10
    
    # Test with a node that has a decorator_list attribute
    class DecoratorNode:
        lineno = 20
        decorator_list = [type('Decorator', (), {'lineno': 15})()]
    
    node = DecoratorNode()
    assert get_first_line_number(node) == 15
    
    # Test with a node that has an empty decorator_list
    class EmptyDecoratorNode:
        lineno = 30
        decorator_list = []
    
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == 30
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 40
    
    node = NoDecoratorAttrNode()
    assert get_first_line_number(node) == 40
    
    # Test with multiple decorators
    class MultipleDecoratorsNode:
        lineno = 50
        decorator_list = [
            type('Decorator1', (), {'lineno': 45})(),
            type('Decorator2', (), {'lineno': 46})(),
            type('Decorator3', (), {'lineno': 47})()
        ]
    
    node = MultipleDecoratorsNode()
    assert get_first_line_number(node) == 45
```


# LLM-generated content at query #208
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunc:
        lineno = 10
        decorator_list = []
    
    simple_node = SimpleFunc()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a function that has decorators
    class DecoratedFunc:
        decorator_list = [type('Decorator', (), {'lineno': 5})]
        lineno = 10
    
    decorated_node = DecoratedFunc()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a class that has decorators
    class DecoratedClass:
        decorator_list = [type('Decorator', (), {'lineno': 3})]
        lineno = 8
    
    decorated_class = DecoratedClass()
    assert get_first_line_number(decorated_class) == 3
    
    # Test with an object that has no decorator_list attribute
    class NoDecorators:
        lineno = 15
    
    no_decorator_node = NoDecorators()
    assert get_first_line_number(no_decorator_node) == 15
    
    # Test with multiple decorators - should return the first decorator's line
    class MultiDecorated:
        decorator_list = [
            type('Decorator', (), {'lineno': 2}),
            type('Decorator', (), {'lineno': 4})
        ]
        lineno = 10
    
    multi_decorated = MultiDecorated()
    assert get_first_line_number(multi_decorated) == 2


# LLM-generated content at query #209
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class FakeDecorator:
        lineno = 10
    class FakeNode:
        decorator_list = [FakeDecorator()]
        lineno = 15
    node = FakeNode()
    assert get_first_line_number(node) == 10

    # Test with a non-decorated node
    class FakeNodeNoDecorator:
        decorator_list = []
        lineno = 20
    node = FakeNodeNoDecorator()
    assert get_first_line_number(node) == 20

    # Test with a node that has no decorator_list attribute
    class FakeNodeNoDecoratorList:
        lineno = 25
    node = FakeNodeNoDecoratorList()
    assert get_first_line_number(node) == 25
```


# LLM-generated content at query #210
#--------------------------

```python
def test_get_first_line_number():
    # Create a mock node with decorator
    class MockDecorator:
        lineno = 10
    
    class MockNodeWithDecorator:
        decorator_list = [MockDecorator()]
        lineno = 15
    
    node_with_decorator = MockNodeWithDecorator()
    assert get_first_line_number(node_with_decorator) == 10
    
    # Create a mock node without decorator
    class MockNodeWithoutDecorator:
        decorator_list = []
        lineno = 20
    
    node_without_decorator = MockNodeWithoutDecorator()
    assert get_first_line_number(node_without_decorator) == 20
    
    # Test with a node that doesn't have decorator_list attribute
    class MockNodeNoDecoratorAttr:
        lineno = 25
    
    node_no_decorator_attr = MockNodeNoDecoratorAttr()
    assert get_first_line_number(node_no_decorator_attr) == 25
    
    # Test with real AST node
    import ast
    code = """
@decorator
def foo():
    pass
"""
    tree = ast.parse(code)
    function_def = tree.body[0]
    assert get_first_line_number(function_def) == 1
    
    # Test with real AST node without decorator
    code2 = """
def bar():
    pass
"""
    tree2 = ast.parse(code2)
    function_def2 = tree2.body[0]
    assert get_first_line_number(function_def2) == 1
```


# LLM-generated content at query #211
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class MockDecorator:
        lineno = 10
    class MockFunctionDef:
        lineno = 5
        decorator_list = [MockDecorator()]
    
    node = MockFunctionDef()
    assert get_first_line_number(node) == 10
    
    # Test with a non-decorated function
    class MockFunctionDefNoDecorators:
        lineno = 15
        decorator_list = []
    
    node = MockFunctionDefNoDecorators()
    assert get_first_line_number(node) == 15
    
    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecorators:
        lineno = 20
    
    node = MockNodeNoDecorators()
    assert get_first_line_number(node) == 20
    
    # Test with multiple decorators
    class MockDecorator1:
        lineno = 25
    class MockDecorator2:
        lineno = 30
    class MockFunctionDefMultipleDecorators:
        lineno = 22
        decorator_list = [MockDecorator1(), MockDecorator2()]
    
    node = MockFunctionDefMultipleDecorators()
    assert get_first_line_number(node) == 25


# LLM-generated content at query #212
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node (no decorators)
    class SimpleNode:
        lineno = 10
    
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 10
    
    # Test with a node that has decorators
    class DecoratedNode:
        decorator_list = []
        lineno = 20
    
    # Test with decorator present
    class DecoratorMock:
        lineno = 5
    
    decorated_node = DecoratedNode()
    decorated_node.decorator_list = [DecoratorMock()]
    decorated_node.lineno = 20
    
    assert get_first_line_number(decorated_node) == 5
    
    # Test with multiple decorators (should return first decorator's line)
    class DecoratorMock2:
        lineno = 8
    
    decorated_node.decorator_list = [DecoratorMock(), DecoratorMock2()]
    assert get_first_line_number(decorated_node) == 5
    
    # Test with empty decorator list
    empty_decorator_node = DecoratedNode()
    empty_decorator_node.decorator_list = []
    empty_decorator_node.lineno = 15
    assert get_first_line_number(empty_decorator_node) == 15
    
    # Test that get_last_line_number works correctly
    class NodeWithEnd:
        end_lineno = 25
    
    node_with_end = NodeWithEnd()
    assert get_last_line_number(node_with_end) == 25
```


# LLM-generated content at query #213
#--------------------------

def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleNode:
        lineno = 5
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with a function that has decorators
    class DecoratorNode:
        decorator_list = [type('Decorator', (), {'lineno': 10})()]
        lineno = 15
    node = DecoratorNode()
    assert get_first_line_number(node) == 10

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttr:
        lineno = 20
    node = NoDecoratorAttr()
    assert get_first_line_number(node) == 20

    # Test with a node that has decorator_list as empty list but no lineno attribute
    class EmptyDecoratorNode:
        decorator_list = []
    node = EmptyDecoratorNode()
    assert get_first_line_number(node) == node.lineno


# LLM-generated content at query #214
#--------------------------

```python
def test_get_first_line_number():
    # Test with no decorator
    class SimpleNode:
        lineno = 5
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 5

    # Test with one decorator
    class DecoratedNode:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 2})()]
    node = DecoratedNode()
    assert get_first_line_number(node) == 2

    # Test with multiple decorators
    class MultiDecoratedNode:
        lineno = 20
        decorator_list = [
            type('Decorator1', (), {'lineno': 3})(),
            type('Decorator2', (), {'lineno': 4})(),
            type('Decorator3', (), {'lineno': 5})()
        ]
    node = MultiDecoratedNode()
    assert get_first_line_number(node) == 3

    # Test with node that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 30
    node = NoDecoratorList()
    assert get_first_line_number(node) == 30

    # Test with decorator list as None
    class NoneDecoratorList:
        lineno = 40
        decorator_list = None
    node = NoneDecoratorList()
    assert get_first_line_number(node) == 40

    # Test with empty decorator list
    class EmptyDecoratorList:
        lineno = 50
        decorator_list = []
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 50
```


# LLM-generated content at query #215
#--------------------------

def test_get_first_line_number():
    # Test with a node that has decorators
    class DecoratedNode:
        def __init__(self):
            self.decorator_list = [type('Decorator', (), {'lineno': 5})()]
            self.lineno = 10
    
    decorated_node = DecoratedNode()
    assert get_first_line_number(decorated_node) == 5
    
    # Test with a node that has no decorators
    class PlainNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 15
    
    plain_node = PlainNode()
    assert get_first_line_number(plain_node) == 15
    
    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        def __init__(self):
            self.lineno = 20
    
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 20


# LLM-generated content at query #216
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple function without decorators
    class SimpleFunction:
        lineno = 5
        decorator_list = []
    
    simple_function = SimpleFunction()
    assert get_first_line_number(simple_function) == 5
    
    # Test with a decorated function
    class DecoratedFunction:
        lineno = 10
        decorator_list = [type('Decorator', (), {'lineno': 3})()]
    
    decorated_function = DecoratedFunction()
    assert get_first_line_number(decorated_function) == 3
    
    # Test with a class that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 15
    
    no_decorator_list = NoDecoratorList()
    assert get_first_line_number(no_decorator_list) == 15
    
    # Test with multiple decorators - should return first decorator's line
    class MultipleDecorators:
        lineno = 20
        decorator_list = [
            type('Decorator1', (), {'lineno': 7}),
            type('Decorator2', (), {'lineno': 8})
        ]
    
    multiple_decorators = MultipleDecorators()
    assert get_first_line_number(multiple_decorators) == 7
    
    # Test with empty decorator_list
    class EmptyDecoratorList:
        lineno = 25
        decorator_list = []
    
    empty_decorator_list = EmptyDecoratorList()
    assert get_first_line_number(empty_decorator_list) == 25
    
    # Test with decorator_list set to None
    class NoneDecoratorList:
        lineno = 30
        decorator_list = None
    
    none_decorator_list = NoneDecoratorList()
    assert get_first_line_number(none_decorator_list) == 30
```


# LLM-generated content at query #217
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    class MockDecorator:
        lineno = 5
    class MockFunction:
        lineno = 10
        decorator_list = [MockDecorator()]
    node = MockFunction()
    assert get_first_line_number(node) == 5

    # Test with a function without decorators
    class MockFunctionNoDecorator:
        lineno = 15
        decorator_list = []
    node = MockFunctionNoDecorator()
    assert get_first_line_number(node) == 15

    # Test with a node that has no decorator_list attribute
    class MockNodeNoDecorators:
        lineno = 20
    node = MockNodeNoDecorators()
    assert get_first_line_number(node) == 20

    # Test with multiple decorators (should return first decorator's line)
    class MockDecorator1:
        lineno = 2
    class MockDecorator2:
        lineno = 3
    class MockFunctionMultiDecorators:
        lineno = 10
        decorator_list = [MockDecorator1(), MockDecorator2()]
    node = MockFunctionMultiDecorators()
    assert get_first_line_number(node) == 2
```


# LLM-generated content at query #218
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node without decorators
    class SimpleNode:
        lineno = 10
        end_lineno = 15
        decorator_list = []
    node = SimpleNode()
    assert get_first_line_number(node) == 10

    # Test with a node that has decorators
    class DecoratorNode:
        lineno = 20
        end_lineno = 25
        decorator_list = [type('Decorator', (), {'lineno': 18})(), type('Decorator', (), {'lineno': 19})()]
    node = DecoratorNode()
    assert get_first_line_number(node) == 18

    # Test with a node that has no decorator_list attribute
    class NoDecoratorList:
        lineno = 30
        end_lineno = 35
    node = NoDecoratorList()
    assert get_first_line_number(node) == 30

    # Test with a node that has empty decorator_list
    class EmptyDecoratorList:
        lineno = 40
        end_lineno = 45
        decorator_list = []
    node = EmptyDecoratorList()
    assert get_first_line_number(node) == 40

    # Test with a node that has decorator_list but it's None
    class NoneDecoratorList:
        lineno = 50
        end_lineno = 55
        decorator_list = None
    node = NoneDecoratorList()
    assert get_first_line_number(node) == 50
```


# LLM-generated content at query #219
#--------------------------

```python
def test_get_first_line_number():
    # Test with a decorated function
    import ast
    code = '''
@decorator
def foo():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with a function without decorators
    code = '''
def bar():
    pass
'''
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert get_first_line_number(func_node) == 1

    # Test with a class with decorators
    code = '''
@class_decorator
class MyClass:
    pass
'''
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 1

    # Test with a class without decorators
    code = '''
class MyClass2:
    pass
'''
    tree = ast.parse(code)
    class_node = tree.body[0]
    assert get_first_line_number(class_node) == 1
```


# LLM-generated content at query #220
#--------------------------

```python
def test_get_first_line_number():
    # Test case 1: Simple function without decorators
    class SimpleFunction:
        pass
    SimpleFunction.lineno = 10
    SimpleFunction.decorator_list = []
    assert get_first_line_number(SimpleFunction) == 10

    # Test case 2: Function with a single decorator
    class DecoratedFunction:
        pass
    DecoratedFunction.lineno = 20
    DecoratedFunction.decorator_list = [type('Decorator', (), {'lineno': 15})]
    assert get_first_line_number(DecoratedFunction) == 15

    # Test case 3: Function with multiple decorators
    class MultiDecoratedFunction:
        pass
    MultiDecoratedFunction.lineno = 30
    MultiDecoratedFunction.decorator_list = [
        type('Decorator1', (), {'lineno': 21}),
        type('Decorator2', (), {'lineno': 22})
    ]
    assert get_first_line_number(MultiDecoratedFunction) == 21

    # Test case 4: Node without decorator_list attribute
    class NoDecoratorAttr:
        pass
    NoDecoratorAttr.lineno = 40
    assert get_first_line_number(NoDecoratorAttr) == 40
```


# LLM-generated content at query #221
#--------------------------

```python
def test_get_first_line_number():
    # Test with a simple node that has no decorators
    class SimpleNode:
        lineno = 5
    simple_node = SimpleNode()
    assert get_first_line_number(simple_node) == 5

    # Test with a node that has decorators
    class DecoratorNode:
        def __init__(self):
            self.decorator_list = [SimpleNode()]
            self.lineno = 10
    decorator_node = DecoratorNode()
    decorator_node.decorator_list[0].lineno = 3
    assert get_first_line_number(decorator_node) == 3

    # Test with a node that has an empty decorator list
    class EmptyDecoratorNode:
        def __init__(self):
            self.decorator_list = []
            self.lineno = 15
    empty_decorator_node = EmptyDecoratorNode()
    assert get_first_line_number(empty_decorator_node) == 15

    # Test with a node that has decorator_list attribute but is None
    class NoneDecoratorNode:
        def __init__(self):
            self.decorator_list = None
            self.lineno = 20
    none_decorator_node = NoneDecoratorNode()
    assert get_first_line_number(none_decorator_node) == 20

    # Test with a node that doesn't have decorator_list attribute
    class NoDecoratorAttrNode:
        lineno = 25
    no_decorator_attr_node = NoDecoratorAttrNode()
    assert get_first_line_number(no_decorator_attr_node) == 25
```


# LLM-generated content at query #222
#--------------------------

def test_get_first_line_number():
    # Test case for a plain function without decorators
    class FakeNode:
        lineno = 10
        decorator_list = []
    node = FakeNode()
    assert get_first_line_number(node) == 10

    # Test case for a function with decorators
    class FakeDecorator:
        lineno = 5
    class FakeNodeWithDecorators:
        lineno = 10
        decorator_list = [FakeDecorator()]
    node = FakeNodeWithDecorators()
    assert get_first_line_number(node) == 5

    # Test case for a class without decorators
    class FakeClass:
        lineno = 20
        decorator_list = []
    node = FakeClass()
    assert get_first_line_number(node) == 20

    # Test case for a class with decorators
    class FakeClassDecorator:
        lineno = 15
    class FakeClassWithDecorators:
        lineno = 20
        decorator_list = [FakeClassDecorator()]
    node = FakeClassWithDecorators()
    assert get_first_line_number(node) == 15

    # Test case with multiple decorators - should return first decorator's lineno
    class FakeDecorator1:
        lineno = 2
    class FakeDecorator2:
        lineno = 3
    class FakeNodeMultipleDecorators:
        lineno = 10
        decorator_list = [FakeDecorator1(), FakeDecorator2()]
    node = FakeNodeMultipleDecorators()
    assert get_first_line_number(node) == 2


