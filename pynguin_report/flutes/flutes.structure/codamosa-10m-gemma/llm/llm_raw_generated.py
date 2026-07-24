####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure():
    # Test simple scalar
    assert map_structure(lambda x: x + 1, 1) == 2

    # Test list
    assert map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

    # Test nested list
    assert map_structure(lambda x: x + 1, [[1], [2, [3]]]) == [[2], [3, [4]]]

    # Test tuple
    assert map_structure(lambda x: x * 2, (1, 2)) == (2, 4)

    # Test namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    assert map_structure(lambda x: x + 1, p) == Point(2, 3)

    # Test dict
    assert map_structure(lambda x: x * 10, {'a': 1, 'b': 2}) == {'a': 10, 'b': 20}

    # Test nested dict
    nested_dict = {'a': [1, 2], 'b': {'c': 3}}
    expected_dict = {'a': [11, 12], 'b': {'c': 13}}
    assert map_structure(lambda x: x + 10, nested_dict) == expected_dict

    # Test set
    # Note: set order is non-deterministic, so we check content
    s = {1, 2, 3}
    result_s = map_structure(lambda x: x * 2, s)
    assert result_s == {2, 4, 6}

    # Test register_no_map_class
    register_no_map_class(set)
    # After registering set as no-map, it should be treated as a leaf/singleton
    # instead of being traversed.
    s_no_map = {1, 2}
    # If it were traversed, it would be {11, 12}. 
    # Since it's no-map, the function is applied to the set itself.
    assert map_structure(lambda x: len(x), s_no_map) == 2

    # Test no_map_instance
    # Create a custom class instance that should not be traversed
    class MyContainer:
        def __init__(self, val):
            self.val = val
    
    instance = MyContainer([1, 2, 3])
    # Use no_map_instance to prevent traversal of the instance contents
    no_map_inst = no_map_instance(instance)
    # map_structure should call the function on the instance itself
    assert map_structure(lambda x: len(x.val), no_map_inst) == 3

    # Test no_map_instance with an immutable type (triggers _no_map_type)
    # Using a tuple as a proxy for a type that cannot have attributes set
    t = (1, 2)
    no_map_t = no_map_instance(t)
    assert map_structure(lambda x: len(x), no_map_t) == 2
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test case 1: Simple list of scalars
    fn_add = lambda x, y: x + y
    objs1 = [[1, 2], [3, 4]]
    assert map_structure_zip(fn_add, objs1) == [[4, 6]]

    # Test case 2: Nested lists
    objs2 = [[[1], [2]], [[3], [4]]]
    assert map_structure_zip(fn_add, objs2) == [[[4], [6]]]

    # Test case 3: Dictionaries
    fn_dict = lambda x, y: x * y
    objs3 = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    assert map_structure_zip(fn_dict, objs3) == {'a': 3, 'b': 8}

    # Test case 4: Tuples
    objs4 = [(1, 2), (3, 4)]
    assert map_structure_zip(fn_add, objs4) == [(4, 6)]

    # Test case 5: Namedtuples
    Point = namedtuple('Point', ['x', 'y'])
    objs5 = [Point(1, 2), Point(3, 4)]
    # Note: zip logic in code handles the reconstruction via type(obj)(...)
    result5 = map_structure_zip(fn_add, objs5)
    assert isinstance(result5, Point)
    assert result5.x == 4
    assert result5.y == 6

    # Test case 6: Mixed structures (matching)
    objs6 = [{'a': [1, 2]}, {'a': [3, 4]}]
    assert map_structure_zip(fn_add, objs6) == {'a': [[4, 6]]}

    # Test case 7: Single element (scalar)
    objs7 = [1, 2, 3]
    assert map_structure_zip(fn_add, objs7) == 6

    # Test case 8: Using no_map_instance
    # This should treat the list as a single object and apply fn to the list itself
    val = [1, 2]
    obj_no_map = no_map_instance([1, 2])
    objs8 = [obj_no_map, [3, 4]]
    # Since the first object is marked no-map, zip sees it as a scalar
    # map_structure_zip(fn, [scalar, list]) -> fn(scalar, list_element)
    # Because it's a list of lists, it descends. 
    # But the first element is a "singleton".
    # The code says: if obj.__class__ in _NO_MAP_TYPES or hasattr(obj, _NO_MAP_INSTANCE_ATTR): return fn(*objs)
    # Here obj is objs[0] which is [1, 2]. It is a list, but it has the attribute.
    # So it calls fn([1, 2], [3, 4])
    assert map_structure_zip(fn_add, objs8) == [1, 2] + [3, 4] # result of fn([1,2], [3,4]) if fn was sum? 
    # Wait, the lambda is fn_add (x+y). [1,2] + [3,4] = [1,2,3,4]
    assert map_structure_zip(fn_add, objs8) == [1, 2, 3, 4]

    # Test case 9: Error case for sets
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(fn_add, [{1}, {2}])

    # Test case 10: Registering a class as no-map
    class MyContainer(list):
        pass
    
    register_no_map_class(MyContainer)
    objs10 = [MyContainer([1, 2]), MyContainer([3, 4])]
    # Because MyContainer is registered, it doesn't descend into the list
    # It calls fn(MyContainer([1,2]), MyContainer([3,4]))
    assert map_structure_zip(fn_add, objs10) == [1, 2, 3, 4]
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test case 1: Simple list of scalars
    objs1 = [[1, 2], [3, 4]]
    assert map_structure_zip(lambda x, y: x + y, objs1) == [4, 6]

    # Test case 2: Nested lists
    objs2 = [[[1], [2]], [[3], [4]]]
    assert map_structure_zip(lambda x, y: x[0] + y[0], objs2) == [[4], [6]]

    # Test case 3: Dictionaries with matching keys
    objs3 = [{"a": 1, "b": 2}, {"a": 10, "b": 20}]
    assert map_structure, map_structure_zip(lambda x, y: x + y, objs3) == {"a": 11, "b": 22}

    # Test case 4: Tuples
    objs4 = [(1, 2), (3, 4)]
    assert map_structure_zip(lambda x, y: x * y, objs4) == [(3, 8)]

    # Test case 5: Namedtuple
    Point = namedtuple("Point", ["x", "y"])
    objs5 = [Point(1, 2), Point(3, 4)]
    result5 = map_structure_zip(lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y), objs5)
    assert result5 == [Point(4, 6)]

    # Test case 6: Mixed structures (List of dicts)
    objs6 = [{"val": 1}, {"val": 10}]
    assert map_structure_zip(lambda d1, d2: {"val": d1["val"] + d2["val"]}, objs6) == [{"val": 11}]

    # Test case 7: Using no_map_instance
    # The function should treat the object as a single unit and pass its components to fn
    class MyType:
        def __init__(self, val):
            self.val = val
    
    m1 = no_map_instance(MyType(1))
    m2 = no_map_instance(MyType(10))
    assert map_structure_zip(lambda a, b: a.val + b.val, [m1, m2]) == 11

    # Test case 8: Using registered no_map_class
    register_no_map_class(set)
    # Since set is registered as no-map, it should be treated as a single element
    # Note: map_structure_zip logic for set is specialized to raise ValueError, 
    # but if it's a registered type, it hits the first 'if' block.
    objs8 = [{1, 2}, {3, 4}]
    # It should pass the two sets as arguments to the lambda
    assert map_structure_zip(lambda s1, s2: len(s1) + len(s2), objs8) == 4

    # Test case 9: Error case - Set contains unordered structure
    # According to the implementation, if the top level is a set, it raises ValueError
    with pytest.raises(ValueError, match="Structures cannot contain `set` because it's unordered"):
        map_structure_zip(lambda x: x, [{1}, {2}])

    # Test case 10: Single element list (identity-like)
    objs10 = [[1, 2, 3]]
    # zip(*objs10) becomes (1,), (2,), (3,)
    # Since the elements are scalars, it calls fn(1), fn(2), fn(3)
    # But wait, map_structure_zip(fn, [ [1,2,3] ]) -> obj is [1,2,3]. 
    # It iterates through zip(*[[1,2,3]]) which is zip([1,2,3]) -> (1,), (2,), (3,)
    # Then it calls map_structure_zip(fn, (1,)) -> returns fn(1)
    assert map_structure_zip(lambda x: x * 2, [[1, 2, 3]]) == [2, 4, 6]
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure():
    # Test basic list mapping
    assert map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

    # Test nested list mapping
    assert map_tuple_result = map_structure(lambda x: x + 1, [[1], [2, [3]]]) == [[2], [3, [4]]]

    # Test tuple mapping
    assert map_structure(lambda x: x * 2, (1, 2, 3)) == (2, 4, 6)

    # Test dict mapping
    assert map_structure(lambda x: x * 2, {'a': 1, 'b': {'c': 2}}) == {'a': 2, 'b': {'c': 4}}

    # Test set mapping
    assert map_structure(lambda x: x + 1, {1, 2, 3}) == {2, 3, 4}

    # Test namedtuple mapping
    Point = namedtuple('Point', ['x', 'y'])
    assert map_structure(lambda x: x * 10, Point(1, 2)) == Point(10, 20)

    # Test no_map_instance (using a class that can have attributes)
    class MyContainer:
        def __init__(self, val):
            self.val = val
    
    container = MyContainer([1, 2, 3])
    no_map_cont = no_map_instance(container)
    # Should treat container as a singleton, applying fn to the container itself
    assert map_structure(lambda x: x.val, no_map_cont) == [1, 2, 3]

    # Test register_no_map_class
    register_no_map_class(list)
    # Now list is registered as non-mappable. map_structure should apply fn to the list itself.
    assert map_structure(lambda x: len(x), [1, 2, 3]) == 3
    
    # Clean up global state for other tests if necessary (though not strictly required for this single test function)
    _NO_MAP_TYPES.remove(list)

def test_map_structure_edge_cases():
    # Test scalar/atomic types
    assert map_structure(lambda x: x + 1, 5) == 6

    # Test nested structures with mixed types
    input_data = [1, (2, 3), {'a': 4}]
    expected = [2, (4, 6), {'a': 8}]
    assert map_structure(lambda x: x * 2, input_data) == expected

    # Test identity function
    assert map_structure(lambda x: x, [1, [2, {3}]]) == [1, [2, {3}]]
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure():
    # Test simple scalar/leaf mapping
    assert map_structure(lambda x: x + 1, 5) == 6
    
    # Test list mapping
    assert map_tuple_mapping = map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    
    # Test nested list mapping
    assert map_structure(lambda x: x + 1, [[1], [2, [3]]]) == [[2], [3, [4]]]
    
    # Test tuple mapping
    assert map_structure(lambda x: x * 10, (1, 2)) == (10, 20)
    
    # Test namedtuple mapping
    Point = namedtuple("Point", ["x", "y"])
    p = Point(1, 2)
    assert map_structure(lambda x: x + 1, p) == Point(2, 3)
    
    # Test dict mapping
    assert map_structure(lambda x: x * 2, {"a": 1, "b": [2, 3]}) == {"a": 2, "b": [4, 6]}
    
    # Test set mapping
    assert map_structure(lambda x: x + 1, {1, 2, 3}) == {2, 3, 4}
    
    # Test register_no_map_class
    class MyCustomContainer(list):
        pass
    
    register_no_map_class(MyCustomContainer)
    custom_container = MyCustomContainer([1, 2, 3])
    # Since MyCustomContainer is registered as no-map, the function should apply to the container itself
    # rather than its elements.
    assert map_structure(lambda x: len(x), custom_container) == 3
    
    # Test no_map_instance
    # We use a list but mark it as no-map via no_map_instance
    # Note: no_map_instance on a built-in like list creates a subclass
    no_map_list = no_map_instance([1, 2, 3])
    assert map_structure(lambda x: len(x), no_map_list) == 3
    
    # Test complex nested structure
    complex_obj = {
        "a": [1, (2, 3)],
        "b": {"c": 4},
        "d": {5, 6}
    }
    expected_obj = {
        "a": [2, (4, 6)],
        "b": {"c": 8},
        "d": {10, 12}
    }
    # Note: set order is non-deterministic, but for small ints it usually works. 
    # For a robust test, we compare contents.
    result = map_structure(lambda x: x * 2, complex_obj)
    assert result["a"] == [2, (4, 6)]
    assert result["b"] == {"c": 8}
    assert result["d"] == {10, 12}
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure():
    # Test simple scalar mapping
    assert map_structure(lambda x: x + 1, 1) == 2

    # Test list mapping
    assert map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

    # Test nested list mapping
    assert map_structure(lambda x: x + 1, [[1], [2, [3]]]) == [[2], [3, [4]]]

    # Test tuple mapping
    assert map_structure(lambda x: x * 2, (1, 2)) == (2, 4)

    # Test namedtuple mapping
    Point = namedtuple("Point", ["x", "y"])
    assert map_structure(lambda x: x + 10, Point(1, 2)) == Point(11, 12)

    # Test dict mapping (values only)
    assert map_structure(lambda x: x * 3, {"a": 1, "b": 2}) == {"a": 3, "b": 6}

    # Test nested dict mapping
    nested_dict = {"a": [1, 2], "b": {"c": 3}}
    expected_dict = {"a": [2, 4], "b": {"c": 6}}
    assert map_structure(lambda x: x * 2, nested_dict) == expected_dict

    # Test set mapping
    assert map_structure(lambda x: x + 1, {1, 2, 3}) == {2, 3, 4}

    # Test register_no_map_class
    class MyCustomContainer(list):
        pass
    
    register_no_map_class(MyCustomContainer)
    custom_obj = MyCustomContainer([1, 2, 3])
    # Since MyCustomContainer is registered as no-map, it should be treated as a single unit
    # and the function applied to the container itself.
    assert map_structure(lambda x: len(x), custom_obj) == 3

    # Test no_map_instance
    # We use an object that can have attributes set (like a custom class)
    class Wrapper:
        def __init__(self, value):
            self.value = value
    
    wrapped = Wrapper([1, 2, 3])
    no_map_wrapped = no_map_instance(wrapped)
    # map_structure should see the no-map attribute and apply fn to the instance
    assert map_structure(lambda x: x.value, no_map_wrapped) == [1, 2, 3]

    # Test no_map_instance with immutable type (triggers _no_map_type)
    # Using a tuple which is immutable
    immutable_tuple = (1, 2)
    no_map_immutable = no_map_instance(immutable_tuple)
    # The function should be applied to the tuple itself, not its elements
    assert map_structure(lambda x: sum(x), no_map_immutable) == 3
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure():
    # Test basic list mapping
    assert map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

    # Test nested list mapping
    assert map_tuple := map_structure(lambda x: x + 1, [[1], [2, [3]]]) == [[2], [3, [4]]]

    # Test tuple mapping
    assert map_structure(lambda x: str(x), (1, 2, 3)) == ("1", "2", "3")

    # Test dict mapping
    assert map_structure(lambda x: x * 10, {"a": 1, "b": 2}) == {"a": 10, "b": 20}

    # Test nested dict mapping
    nested_dict = {"a": [1, 2], "b": {"c": 3}}
    expected_dict = {"a": [2, 4], "b": {"c": 6}}
    assert map_structure(lambda x: x * 2, nested_dict) == expected_dict

    # Test set mapping (sets are unordered, so we check content)
    res_set = map_structure(lambda x: x + 1, {1, 2, 3})
    assert res_set == {2, 3, 4}

    # Test namedtuple mapping
    Point = namedtuple("Point", ["x", "y"])
    p = Point(1, 2)
    assert map_structure(lambda x: x * 5, p) == Point(5, 10)

    # Test register_no_map_class (treating a class as a leaf)
    class MyCustomContainer(list):
        pass
    
    register_no_map_class(MyCustomContainer)
    custom_obj = MyCustomContainer([1, 2, 3])
    # Should apply fn to the container itself, not its elements
    assert map_structure(lambda x: len(x), custom_obj) == 3

    # Test no_map_instance (treating an instance as a leaf)
    # We use a class that allows setattr to avoid the fallback to _no_map_type for simple tests
    class MappableInstance:
        def __init__(self, value):
            self.value = value
    
    inst = MappulateInstance(10)
    no_map_inst = no_map_instance(inst)
    assert map_structure(lambda x: x.value * 2, no_map_inst) == 20

    # Test leaf nodes (non-containers)
    assert map_structure(lambda x: x + 1, 5) == 6

    # Test complex mixed structure
    complex_obj = [1, {"a": (2, 3)}, {4, 5}]
    # Note: set in map_structure is handled, but order in set is non-deterministic
    result = map_structure(lambda x: x * 2, complex_obj)
    assert result[0] == 2
    assert result[1]["a"] == (4, 6)
    assert set(result[2]) == {8, 10}
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test 1: Basic list of scalars
    objs1 = [[1, 2], [3, 4]]
    fn1 = lambda x, y: x + y
    assert map_structure_tuple_zip_result(fn1, objs1) == [4, 6]

    # Test 2: Nested lists
    objs2 = [[[1], [2]], [[3], [4]]]
    fn2 = lambda x, y: x + y
    assert map_structure_zip_result(fn2, objs2) == [[[4], [6]]]

    # Test 3: Dictionaries
    objs3 = [{"a": 1, "b": 2}, {"a": 10, "b": 20}]
    fn3 = lambda x, y: x + y
    assert map_structure_zip_result(fn3, objs3) == {"a": 11, "b": 22}

    # Test 4: Tuples
    objs4 = [(1, 2), (3, 4)]
    fn4 = lambda x, y: x * y
    assert map_structure_zip_result(fn4, objs4) == [(3, 8)]

    # Test 5: NamedTuples
    Point = namedtuple("Point", ["x", "y"])
    objs5 = [Point(1, 2), Point(3, 4)]
    fn5 = lambda x, y: x + y
    assert map_structure_zip_result(fn5, objs5) == [Point(4, 6)]

    # Test 6: Mixed structures (matching)
    objs6 = [{"a": [1, 2]}, {"a": [3, 4]}]
    fn6 = lambda x, y: x + y
    assert map_structure_zip_result(fn6, objs6) == {"a": [4, 6]}

    # Test 7: Single element/Scalar-like (no traversal)
    objs7 = [1, 2]
    fn7 = lambda x, y: x + y
    assert map_structure_zip_result(fn7, objs7) == [3]

    # Test 8: Error case - set is unordered
    objs8 = [{1, 2}, {3, 4}]
    fn8 = lambda x, y: x + y
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(fn8, objs8)

    # Test 9: No-map registered type
    class MyContainer(list):
        pass
    
    register_no_map_class(MyContainer)
    objs9 = [MyContainer([1, 2]), MyContainer([3, 4])]
    fn9 = lambda x, y: x + y
    # Since MyContainer is registered as no-map, it should treat the instance as a single unit
    # and apply fn to the objects themselves, not their elements.
    # zip(*objs9) results in ([1, 2], [3, 4])
    # The function fn is applied to the elements of the zip, which are the lists.
    assert map_structure_zip_result(fn9, objs9) == [[1, 2, 3, 4]]

# Helper to allow the test to run in the context of the provided snippet
def map_structure_zip_result(fn, objs):
    return map_structure_zip(fn, objs)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test case 1: Simple list of lists (integers)
    fn_add = lambda x, y: x + y
    objs1 = [[1, 2], [3, 4]]
    assert map_structure_zip(fn_add, objs1) == [[4, 6]]

    # Test case 2: Nested lists
    objs2 = [[[1], [2]], [[3], [4]]]
    assert map_structure_zip(fn_add, objs2) == [[[4], [6]]]

    # Test case 3: Dictionaries
    fn_mul = lambda x, y: x * y
    objs3 = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert map_structure_zip(fn_mul, objs3) == {"a": 3, "b": 8}

    # Test case 4: Tuples
    objs4 = [(1, 2), (3, 4)]
    assert map_structure_zip(fn_add, objs4) == [(4, 6)]

    # Test case 5: Namedtuples
    Point = namedtuple("Point", ["x", "y"])
    objs5 = [Point(1, 2), Point(3, 4)]
    result5 = map_structure_zip(fn_add, objs5)
    assert isinstance(result5, Point)
    assert result5.x == 4
    assert result5.y == 6

    # Test case 6: Single elements (not collections)
    objs6 = [1, 2, 3]
    assert map_structure_zip(lambda *args: sum(args), objs6) == 6

    # Test case 7: Using no_map_instance
    # We mark a list as a non-mappable singleton
    special_list = no_map_instance([10, 20])
    objs7 = [special_list, [1, 2]]
    # Since special_list is non-mappable, it is treated as a single argument to fn
    # fn receives (special_list, [1, 2]) -> [10, 20] + [1, 2] is error, so we use a custom fn
    fn_concat = lambda x, y: x + y
    assert map_structure_zip(fn_concat, objs7) == [10, 20, 1, 2]

    # Test case 8: Using register_no_map_class
    register_no_map_class(tuple)
    objs8 = [(1, 2), (3, 4)]
    # Because tuple is registered as no-map, it doesn't descend into the tuple elements
    # It treats the tuple itself as the argument
    assert map_structure_zip(lambda x, y: x[0] + y[0], objs8) == (4,)

    # Test case 9: ValueError for sets
    objs9 = [{1}, {2}]
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(fn_add, objs9)

    # Test case 10: Complex mixed structure
    objs10 = [
        {"a": [1, 2], "b": 3},
        {"a": [10, 20], "b": 7}
    ]
    expected10 = {"a": [11, 22], "b": 10}
    assert map_structure_zip(fn_add, objs10) == expected10
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure():
    # Test basic list mapping
    assert map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

    # Test nested list mapping
    assert map_tuple_nested = map_structure(lambda x: x + 1, [[1], [2, [3]]])
    assert map_tuple_nested == [[2], [3, [4]]]

    # Test tuple mapping
    assert map_structure(lambda x: x * 1.0, (1, 2)) == (1.0, 2.0)

    # Test namedtuple mapping
    Point = namedtuple("Point", ["x", "y"])
    pt = Point(1, 2)
    assert map_structure(lambda x: x + 10, pt) == Point(11, 12)

    # Test dict mapping
    d = {"a": 1, "b": {"c": 2}}
    assert map_structure(lambda x: x * 3, d) == {"a": 3, "b": {"c": 6}}

    # Test set mapping
    s = {1, 2, 3}
    assert map_structure(lambda x: x + 1, s) == {2, 3, 4}

    # Test leaf node (non-collection) mapping
    assert map_structure(lambda x: x + 1, 5) == 6

    # Test register_no_map_class
    class MyCustomContainer(list):
        pass
    
    register_no_map_class(MyCustomContainer)
    custom_list = MyCustomContainer([1, 2, 3])
    # Since MyCustomContainer is registered, it should be treated as a singleton
    # and the function applied to the container itself, not its elements.
    assert map_structure(lambda x: len(x), custom_list) == 3

    # Test no_map_instance
    # We use a simple object that can have attributes
    class SimpleObj:
        pass
    
    obj = SimpleObj()
    obj.val = 10
    no_map_obj = no_map_instance(obj)
    # The function should be applied to the instance itself
    assert map_structure(lambda x: x.val, no_map_obj) == 10

    # Test no_map_instance with immutable type (triggers _no_map_type)
    # Tuples are immutable, so no_map_instance will create a subtype
    tup = (1, 2)
    no_map_tup = no_map_instance(tup)
    assert map_structure(lambda x: len(x), no_map_tup) == 2
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test case 1: Simple list of integers
    fn_add = lambda x, y: x + y
    objs1 = [[1, 2], [3, 4]]
    assert map_structure_zip(fn_add, objs1) == [[4, 6]]

    # Test case 2: Nested lists
    objs2 = [[[1], [2]], [[3], [4]]]
    assert map_structure_zip(fn_add, objs2) == [[[4], [6]]]

    # Test case 3: Dictionaries
    fn_dict = lambda x, y: x * y
    objs3 = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert map_structure_zip(fn_dict, objs3) == {"a": 3, "b": 8}

    # Test case 4: Tuples
    objs4 = [(1, 2), (3, 4)]
    assert map_structure_zip(fn_add, objs4) == [(4, 6)]

    # Test case 5: NamedTuples
    Point = namedtuple("Point", ["x", "y"])
    objs5 = [Point(1, 2), Point(3, 4)]
    assert map_structure_zip(fn_add, objs5) == Point(4, 6)

    # Test case 6: Singletons (non-mappable)
    # Using a custom class that isn't a collection
    class MyObj:
        def __init__(self, val):
            self.val = val
    
    obj_a = MyObj(10)
    obj_b = MyObj(20)
    fn_get_sum = lambda x, y: x.val + y.val
    assert map_structure_zip(fn_get_sum, [obj_a, obj_b]) == 30

    # Test case 7: Error on sets (as specified in docstring)
    objs_set = [{1}, {2}]
    with pytest.raises(ValueError, match="Structures cannot contain `set` because it's unordered"):
        map_structure_zip(fn_add, objs_set)

    # Test case 8: Deeply nested heterogeneous structure
    objs8 = [
        [{"a": 1}, (2, 3)],
        [{"a": 10}, (20, 30)]
    ]
    # For dict, it takes keys from first obj and zips values
    # For tuple, it zips elements
    # For list, it zips elements
    # fn_add will be applied to: 
    # 1. dict values: 1+10=11
    # 2. tuple elements: 2+20=22, 3+30=33
    expected8 = [{"a": 11}, (22, 33)]
    assert map_structure_zip(fn_add, objs8) == expected8
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test 1: Basic list of lists (integers)
    # zips [1, 2], [3, 4] -> [(1, 3), (2, 4)] -> mapped by sum -> [4, 6]
    objs1 = [[1, 2], [3, 4]]
    assert map_structure_zip(lambda x, y: x + y, objs1) == [[4, 6]]

    # Test 2: Nested structures (list of tuples of lists)
    # objs: [ [(1, 2), (3, 4)], [(10, 20), (30, 40)] ]
    # zipped: [ ( (1, 10), (3, 30) ), ( (2, 20), (4, 40) ) ]
    # mapped by sum: [ [(11, 11), (33, 33)], [(22, 22), (44, 44)] ]
    objs2 = [
        [[1, 2], [3, 4]],
        [[10, 20], [30, 40]]
    ]
    # We use a more controlled nested structure for easier verification
    # structure: list of list of tuples
    objs2 = [
        [[ (1, 2) ], [ (3, 4) ]],
        [[ (10, 20) ], [ (30, 40) ]]
    ]
    # zip 1st level: [ [(1, 10), (3, 30)], [(2, 20), (4, 40)] ]
    # zip 2nd level: [ ( (1, 10), (3, 30) ), ( (2, 20), (4, 40) ) ] ... wait, logic is recursive.
    # Let's use a simpler nested structure:
    objs3 = [
        [ (1, 2), (3, 4) ],
        [ (10, 20), (30, 40) ]
    ]
    # zip level 1: [ ( (1, 10), (2, 20) ), ( (3, 30), (4, 40) ) ]
    # mapped by sum: [ ( (11, 22), (33, 44) ) ]
    # The function returns the same structure as objs[0]
    expected3 = [ ( (11, 22), (33, 44) ) ] # This depends on how zip interacts with the nesting
    # Let's re-trace:
    # objs[0] is list. zip(*objs) -> [ ( [1, 2], [10, 20] ), ( [3, 4], [30, 40] ) ]
    # map_structure_zip on first element: fn( [1, 2], [10, 20] ) -> if fn is sum...
    
    # Let's use a concrete, verifiable case:
    # fn = lambda x, y: x + y
    # objs = [ [1, 2], [10, 20] ]
    # zip(*objs) -> (1, 10), (2, 20)
    # result -> [ 1+10, 2+20 ] -> [11, 22]
    assert map_structure_zip(lambda x, y: x + y, [[1, 2], [10, 20]]) == [11, 22]

    # Test 3: Dicts
    # objs = [ {'a': 1, 'b': 2}, {'a': 10, 'b': 20} ]
    # result -> {'a': 11, 'b': 22}
    objs4 = [{'a': 1, 'b': 2}, {'a': 10, 'b': 20}]
    assert map_structure_zip(lambda x, y: x + y, objs4) == {'a': 11, 'b': 22}

    # Test 4: NamedTuples
    Point = namedtuple('Point', ['x', 'y'])
    objs5 = [Point(1, 2), Point(10, 20)]
    # zip(*objs5) -> (1, 10), (2, 20)
    # result -> Point(11, 22)
    # Note: Since objs[0] is a Point, the function returns Point(fn(1,10), fn(2,20))
    # But zip(*objs5) produces an iterator of tuples. 
    # The code says: if isinstance(obj, tuple) and hasattr(obj, '_fields'): 
    # return type(obj)(*[map_structure_zip(fn, xs) for xs in zip(*objs)])
    # This is slightly complex. If objs[0] is Point, it treats it as a tuple.
    # zip(*[Point(1,2), Point(10,20)]) -> (1, 10), (2, 20)
    # map_structure_zip(fn, [(1,10), (2,20)]) -> type(Point)(map_structure_zip(fn, (1,10)), map_structure_zip(fn, (2,20)))
    # Resulting in Point(11, 22)
    result5 = map_structure_zip(lambda x, y: x + y, [Point(1, 2), Point(10, 20)])
    assert result5 == Point(11, 22)

    # Test 5: Singletons / No-map instances
    # If the first object is marked as no-map, it should just apply fn to the args
    class MyType:
        def __init__(self, val): self.val = val
    
    m1 = no_map_instance(MyType(1))
    m2 = no_map_instance(MyType(10))
    # Since m1 is no-map, it returns fn(m1, m2)
    # fn = lambda x, y: x.val + y.val
    assert map_structure_zip(lambda x, y: x.val + y.val, [m1, m2]) == 11

    # Test 6: Error on set
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(lambda x, y: x + y, [{1}, {2}])

    # Test 7: Deeply nested
    # objs = [ [ [1], [2] ], [ [10], [20] ] ]
    # zip -> [ ([1], [10]), ([2], [20]) ]
    # zip -> [ ([1+10], [2+20]) ] -> [ [ [11], [22] ] ] ... actually:
    # zip level 1: [ ( [1], [10] ), ( [2], [20] ) ]
    # zip level 2: [ ( [11], [22] ) ]
    # final: [ [ [11], [22] ] ]
    objs6 = [ [[1], [2]], [[10], [20]] ]
    expected6 = [ [[11], [22]] ]
    # Wait, let's trace carefully:
    # 1. zip(*[[[1], [2]], [[10], [20]]]) -> ( ([1], [10]), ([2], [20]) )
    # 2. map_structure_zip on ([1], [10]) -> zip([1], [10]) -> (1, 10) -> fn(1, 10) -> 11
    # 3. map_structure_zip on ([2], [20]) -> zip([2], [20]) -> (2, 20) -> fn(2, 20) -> 22
    # 4. result is [ [11, 22] ] (if the outer was a list)
    # Let's check the structure: objs[0] is list.
    # The result is [map_structure_zip(fn, xs) for xs in zip(*objs)]
    # xs is ([1], [10]) and ([2], [20])
    # result is [ 11, 22 ] if fn is sum, but wait, the zip is applied to the elements.
    # If fn is lambda x, y: x + y:
    # zip(*objs) -> ( ([1], [10]), ([2], [20]) )
    # For the first element: map_structure_zip(fn, ([1], [10]))
    #   ([1], [10]) is a tuple. zip(*([1], [10])) -> (1, 10)
    #   map_structure_zip(fn, (1, 10)) -> fn(1, 10) -> 11
    # Result: [11, 22]
    assert map_structure_zip(lambda x, y: x + y, [[[1], [2]], [[10], [20]]]) == [11, 22]
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test case 1: Simple list of integers
    fn_add = lambda x, y: x + y
    objs1 = [[1, 2], [3, 4]]
    assert map_structure_zip(fn_add, objs1) == [[4, 6]]

    # Test case 2: Nested lists
    fn_mul = lambda x, y: x * y
    objs2 = [[[1, 2], [3]], [[5, 6], [7]]]
    assert map_tuple_result := map_structure_zip(fn_mul, objs2) == [[[5, 12], [21]]]

    # Test case 3: Dictionary structures
    fn_sub = lambda x, y: x - y
    objs3 = [{"a": 10, "b": 20}, {"a": 1, "b": 2}]
    assert map_structure_zip(fn_sub, objs3) == {"a": 9, "b": 18}

    # Test case 4: Tuple structures
    fn_sum = lambda x, y: x + y
    objs4 = [(1, 2), (10, 20)]
    assert map_structure_zip(fn_sum, objs4) == [(11, 22)]

    # Test case 5: Namedtuple structures
    Point = namedtuple("Point", ["x", "y"])
    fn_point = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    objs5 = [Point(1, 2), Point(3, 4), Point(5, 6)]
    assert map_structure_zip(fn_point, objs5) == Point(9, 12)

    # Test case 6: Single element (no iteration needed)
    fn_identity = lambda x: x
    objs6 = [10, 20]
    assert map_structure_zip(fn_identity, objs6) == 30

    # Test case 7: No-map instance/type handling
    # Using no_map_instance to treat a list as a single unit
    objs7 = [no_map_instance([1, 2]), [3, 4]]
    # Since [1, 2] is no_map, the function receives [1, 2] and [3, 4] as args
    assert map_structure_zip(lambda x, y: sum(x) + sum(y), objs7) == 10

    # Test case 8: Error handling for sets
    objs8 = [{1, 2}, {3, 4}]
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(fn_add, objs8)

    # Test case 9: Complex mixed structure
    # List containing a dict and a tuple
    objs9 = [{"a": [1, 2]}, (3, 4)]
    # Note: zip will align 1st element of obj1 with 1st element of obj2... 
    # But zip(*objs) stops at shortest. 
    # For dicts, we need matching keys.
    objs9_aligned = [{"a": [1]}, {"a": [2]}]
    assert map_structure_zip(lambda x, y: x + y, objs9_aligned) == {"a": [3]}
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test basic list of scalars
    assert map_structure_zip(lambda x, y: x + y, [[1, 2], [3, 4]]) == [[4, 6]]
    
    # Test nested lists
    assert map_structure_zip(lambda x, y: x + y, [[[1], [2]], [[3], [4]]]) == [[[4], [6]]]
    
    # Test lists of tuples
    assert map_tuple := map_structure_zip(lambda x, y: x * y, [(1, 2), (3, 4)], [(10, 20), (30, 40)])
    assert tuple(tuple(t) for t in map_tuple) == [(10, 40), (9, 80)]

    # Test dictionaries
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'a': 10, 'b': 20}
    assert map_structure_zip(lambda x, y: x + y, [dict1, dict2]) == {'a': 11, 'b': 22}

    # Test namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p1 = Point(1, 2)
    p2 = Point(10, 20)
    result = map_structure_zip(lambda x, y: x + y, [p1, p2])
    assert isinstance(result, Point)
    assert result.x == 11
    assert result.y == 22

    # Test with no_map_instance (treating a list as a singleton)
    singleton_list = no_map_instance([1, 2])
    other_list = [3, 4]
    # Since singleton_list is no_map, it should call fn(singleton_list, other_list)
    # We use a reducer to prove it's treating the list as one object
    assert map_structure_zip(lambda x, y: len(x) + len(y), [singleton_list, other_list]) == 4

    # Test with registered no_map_class
    class MyContainer(list):
        pass
    
    register_no_map_class(MyContainer)
    mc1 = MyContainer([1, 2])
    mc2 = MyContainer([3, 4])
    # Because MyContainer is registered as no_map, it should call fn(mc1, mc2)
    # and not traverse the elements.
    result = map_structure_zip(lambda x, y: len(x) + len(y), [mc1, mc2])
    assert result == 4

    # Test ValueError for sets (as per docstring)
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(lambda x, y: x + y, [{1}, {2}])

    # Test deep nesting and mixed types
    obj1 = [1, {'a': 2}, (3,)]
    obj2 = [10, {'a': 20}, (30,)]
    expected = [11, {'a': 22}, (33,)]
    assert map_structure_zip(lambda x, y: x + y, [obj1, obj2]) == expected
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test case 1: Simple flat lists
    fn_add = lambda x, y: x + y
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    assert map_structure_zip(fn_add, [list1, list2]) == [5, 7, 9]

    # Test case 2: Nested lists
    list3 = [[1, 2], [3]]
    list4 = [[10, 20], [30]]
    assert map_structure << [list3, list4] == [[11, 22], [33]] # Correction: check logic
    # Re-verifying logic: zip([1,2], [10,20]) -> (1,10), (2,20) -> [11, 22]
    assert map_structure_zip(fn_add, [list3, list4]) == [[11, 22], [33]]

    # Test case 3: Dictionaries
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'a': 10, 'b': 20}
    assert map_structure_zip(fn_add, [dict1, dict2]) == {'a': 11, 'b': 22}

    # Test case 4: Tuples
    tuple1 = (1, 2)
    tuple2 = (3, 4)
    assert map_structure_zip(fn_add, [tuple1, tuple2]) == (4, 6)

    # Test case 5: Namedtuples
    Point = namedtuple('Point', ['x', 'y'])
    p1 = Point(1, 2)
    p2 = Point(10, 20)
    result_nt = map_structure_zip(fn_add, [p1, p2])
    assert isinstance(result_nt, Point)
    assert result_nt.x == 11
    assert result_nt.y == 22

    # Test case 6: Scalar/Singleton values (no iteration possible)
    # If the first object is not a collection, it calls fn(*objs)
    assert map_structure_zip(fn_add, [1, 2]) == 3

    # Test case 7: Using no_map_instance
    # If the first object is marked as no_map, it should not traverse
    val1 = no_map_instance([1, 2])
    val2 = [10, 20]
    # fn should receive the containers themselves as arguments
    assert map_structure_zip(lambda x, y: (x, y), [val1, val2]) == ([1, 2], [10, 20])

    # Test case 8: Error on sets (as per docstring)
    set1 = {1, 2}
    set2 = {3, 4}
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(fn_add, [set1, set2])

    # Test case 9: Deeply nested heterogeneous structures
    struct1 = [1, {'a': (2, 3)}, 4]
    struct2 = [10, {'a': (20, 30)}, 40]
    expected = [11, {'a': (22, 33)}, 44]
    assert map_structure_zip(fn_add, [struct1, struct2]) == expected
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test 1: Basic mapping of lists (integers)
    fn_add = lambda x, y: x + y
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    assert map_structure_zip(fn_add, [list1, list2]) == [5, 7, 9]

    # Test 2: Nested lists
    list_nested1 = [[1, 2], [3, 4]]
    list_nested2 = [[10, 20], [30, 40]]
    assert map_structure_zip(fn_add, [list_nested1, list_nested2]) == [[11, 22], [33, 44]]

    # Test 3: Dictionary structures
    dict1 = {'a': 1, 'b': {'c': 2}}
    dict2 = {'a': 10, 'b': {'c': 20}}
    assert map_structure_zip(fn_add, [dict1, dict2]) == {'a': 11, 'b': {'c': 22}}

    # Test 4: Tuple structures
    tuple1 = (1, (2, 3))
    tuple2 = (10, (20, 30))
    assert map_structure_zip(fn_add, [tuple1, tuple2]) == (11, (22, 33))

    # Test 5: Namedtuple structures
    Point = namedtuple('Point', ['x', 'y'])
    p1 = Point(1, 2)
    p2 = Point(10, 20)
    assert map_structure_zip(fn_add, [p1, p2]) == Point(11, 22)

    # Test 6: No-map instance (treating object as singleton)
    class MyCustomClass:
        def __init__(self, val):
            self.val = val
    
    obj1 = MyCustomClass(5)
    obj2 = MyCustomClass(10)
    # Since MyCustomClass is not registered and not a standard container, 
    # map_structure_zip should pass the objects as arguments to the function.
    assert map_structure_zip(lambda x, y: x.val + y.val, [obj1, obj2]) == 15

    # Test 7: Registered no-map type
    register_no_map_class(list)
    # If list is registered as no-map, the function receives the list objects themselves
    # rather than iterating through them.
    assert map_structure_zip(lambda x, y: len(x) + len(y), [[1, 2], [3, 4, 5]]) == 5
    
    # Cleanup for other tests (resetting the global state)
    _NO_MAP_TYPES.remove(list)

    # Test 8: ValueError for sets (as per docstring)
    set1 = {1, 2}
    set2 = {3, 4}
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(fn_add, [set1, set2])

    # Test 9: Single element/Scalar (not a collection)
    assert map_structure_zip(fn_add, [1, 2]) == 3
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test 1: Simple lists (flat)
    fn_add = lambda x, y: x + y
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    assert map_structure_zip(fn_add, [list1, list2]) == [5, 7, 9]

    # Test 2: Nested lists
    list_nested1 = [[1, 2], [3]]
    list_nested2 = [[10, 20], [30]]
    assert map_structure_zip(fn_add, [list_nested1, list_nested2]) == [[11, 22], [33]]

    # Test 3: Dictionaries
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'a': 10, 'b': 20}
    assert map_structure_zip(fn_add, [dict1, dict2]) == {'a': 11, 'b': 22}

    # Test 4: Tuples
    tuple1 = (1, 2)
    tuple2 = (3, 4)
    assert map_structure_zip(fn_add, [tuple1, tuple2]) == (4, 6)

    # Test 5: NamedTuples
    Point = namedtuple('Point', ['x', 'y'])
    p1 = Point(1, 2)
    p2 = Point(10, 20)
    # Note: map_structure_zip for namedtuple will reconstruct it with the result of fn
    # Since fn_add takes 2 args, result will be Point(11, 22)
    assert map_structure_zip(fn_add, [p1, p2]) == Point(11, 22)

    # Test 6: Mixed structures (Lists containing dicts)
    struct1 = [{'a': 1}, [2, 3]]
    struct2 = [{'a': 10}, [20, 30]]
    expected = [{'a': 11}, [22, 33]]
    assert map_structure_zip(fn_add, [struct1, struct2]) == expected

    # Test 7: Scalar/Singleton objects (No structure to traverse)
    # When the first object is not a container, it treats all as args to fn
    assert map_structure_zip(lambda x, y, z: x + y + z, [1, 2, 3]) == 6

    # Test 8: Registering no-map class
    class MyCustomContainer(list):
        pass
    
    register_no_map_class(MyCustomContainer)
    custom1 = MyCustomContainer([1, 2])
    custom2 = MyCustom                        # Error in prompt logic? No, use second list
    custom2 = MyCustomContainer([10, 20])
    # Because MyCustomContainer is registered, it should NOT be traversed
    # It should call fn(custom1, custom2)
    assert map_structure_zip(fn_add, [custom1, custom2]) == 11 # This is a bit tricky
    # Actually, if no-map is triggered, it calls fn(*objs). 
    # Since objs is [custom1, custom2], it calls fn(custom1, custom2).
    # custom1 + custom2 (list addition) = [1, 2, 10, 20]
    # However, the fn passed is fn_add(x, y) -> x + y.
    # If fn_add is used on lists, it performs list concatenation.
    assert map_structure_zip(lambda x, y: x + y, [custom1, custom2]) == [1, 2, 10, 20]

    # Test 9: Error case for sets
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(fn_add, [{1}, {2}])

    # Test 10: no_map_instance
    inst1 = no_map_instance([1, 2])
    inst2 = no_map_instance([10, 20])
    # Should treat [1, 2] as a single element, not traverse it
    assert map_structure_zip(fn_add, [inst1, inst2]) == [11, 22]
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test case 1: Basic list of integers
    fn_add = lambda *args: sum(args)
    objs1 = [[1, 2], [3, 4], [5, 6]]
    # zip(*objs1) -> (1, 3, 5), (2, 4, 6)
    # sum(1,3,5)=9, sum(2,4,6)=12
    assert map_structure_zip(fn_add, objs1) == [[9], [12]] # Wait, let's re-evaluate logic
    # Re-evaluating zip logic in code:
    # objs1 is [[1,2], [3,4], [5,6]]
    # obj = [1,2]. Not no-map. Is list.
    # returns [map_structure_zip(fn, xs) for xs in zip([1,2], [3,4], [5,6])]
    # zip yields (1,3,5) and (2,4,6)
    # Next level: map_structure_zip(fn, (1,3,5)) -> fn(1,3,5) -> 9
    # Result: [9, 12]
    assert map_structure_zip(fn_add, [[1, 2], [3, 4], [5, 6]]) == [9, 12]

    # Test case 2: Nested structures (List of Dicts)
    fn_mul = lambda *args: args[0] * args[1]
    objs2 = [{"a": 1, "b": 2}, {"a": 3, "bo": 4}] # Note: code assumes identical structure (keys)
    # Let's use identical keys to avoid KeyError in the dict comprehension logic
    objs2 = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    # obj = {"a": 1, "b": 2}. Is dict.
    # returns type(obj)((k, map_structure_zip(fn, [o[k] for o in objs])) for k in obj.keys())
    # k='a': map_structure_zip(fn, [1, 3]) -> fn(1, 3) -> 3
    # k='b': map_structure_zip(fn, [2, 4]) -> fn(2, 4) -> 8
    # Result: {'a': 3, 'b': 8}
    assert map_structure_zip(fn_mul, [{"a": 1, "b": 2}, {"a": 3, "b": 4}]) == {"a": 3, "b": 8}

    # Test case 3: Namedtuple
    Point = namedtuple("Point", ["x", "y"])
    objs3 = [Point(1, 2), Point(3, 4)]
    # obj is Point. Is tuple. Has _fields.
    # returns type(obj)(*[map_structure_zip(fn, xs) for xs in zip((1,2), (3,4))])
    # zip yields (1, 3) and (2, 4)
    # map_structure_zip(fn, (1,3)) -> 4
    # map_structure_zip(fn, (2,4)) -> 6
    # Result: Point(4, 6)
    assert map_structure:
        res = map_structure_zip(fn_add, [Point(1, 2), Point(3, 4)])
        assert res == Point(4, 6)

    # Test case 4: No-map instance (Singleton)
    # Using a class that is registered as no-map
    class MyNoMap:
        pass
    register_no_map_class(MyNoMap)
    obj_no_map = MyNoMap()
    objs4 = [obj_no_map, obj_no_map]
    # obj is MyNoMap. In _NO_MAP_TYPES.
    # returns fn(*objs) -> fn(obj_no_map, obj_no_map)
    fn_id = lambda x, y: x
    assert map_structure_zip(fn_id, [obj_no_map, obj_no_map]) == obj_no_map

    # Test case 5: ValueError for sets
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(fn_add, [{1, 2}, {3, 4}])

    # Test case 6: Nested Lists and Tuples
    objs6 = [ [[1], [2]], [[3], [4]] ]
    # Level 0: List. zip -> ([1], [3]), ([2], [4])
    # Level 1: List. zip -> (1, 3), (2, 4)
    # Level 2: Int. fn(1, 3) -> 4, fn(2, 4) -> 6
    # Result: [[4], [6]]
    assert map_structure_zip(fn_add, [[[1], [2]], [[3], [4]]]) == [[4], [6]]

    # Test case 7: Tuple (standard)
    objs7 = [(1, 2), (3, 4)]
    # obj is tuple. No _fields.
    # zip -> (1, 3), (2, 4)
    # map_structure_zip(fn, (1,3)) -> 4
    # map_structure_zip(fn, (2,4)) -> 6
    # Result: (4, 6)
    assert map_structure_zip(fn_add, [(1, 2), (3, 4)]) == (4, 6)
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test basic list of lists
    fn_add = lambda x, y: x + y
    list1 = [[1, 2], [3]]
    list2 = [[10, 20], [30]]
    assert map_structure_zip(fn_add, [list1, list2]) == [[11, 22], [33]]

    # Test basic list of tuples
    fn_mul = lambda x, y: x * y
    tuple1 = [(1, 2), (3,)]
    tuple2 = [(10, 20), (30,)]
    assert map_structure_zip(fn_mul, [tuple1, tuple2]) == [(10, 40), (90,)]

    # Test dicts with matching keys
    fn_sub = lambda x, y: x - y
    dict1 = {'a': 10, 'b': {'c': 20}}
    dict2 = {'a': 1, 'b': {'c': 2}}
    assert map_structure_zip(fn_sub, [dict1, dict2]) == {'a': 9, 'b': {'c': 18}}

    # Test namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    pt1 = Point(1, 2)
    pt2 = Point(10, 20)
    pt3 = Point(100, 200)
    # Note: map_structure_zip treats the first object's structure as template
    # If first is namedtuple, it maps fn over the elements of the zipped tuple
    # Here objs[0] is pt1 (a tuple with _fields)
    # zip(*[pt1, pt2, pt3]) -> (1, 10, 100), (2, 20, 200)
    # result should be Point(fn(1,10,100), fn(2,20,200))
    fn_sum_three = lambda x, y, z: x + y + z
    assert map_structure_zip(fn_sum_three, [pt1, pt2, pt3]) == Point(111, 222)

    # Test no_map_instance / singleton behavior
    # If the first object is marked as no_map, it should just call fn(*objs)
    class MyType:
        def __init__(self, val):
            self.val = val
    
    obj_no_map = no_map_instance(MyType(5))
    obj_other = MyType(10)
    fn_concat = lambda x, y: f"{x.val}_{y.val}"
    assert map_structure_zip(fn_concat, [obj_no_map, obj_other]) == "5_10"

    # Test register_no_map_class
    register_no_map_class(MyType)
    # Now instances of MyType should be treated as singletons
    assert map_structure_zip(fn_concat, [MyType(1), MyType(2)]) == "1_2"

    # Test ValueError for sets
    with pytest.raises(ValueError, match="Structures cannot contain `set` because it's unordered"):
        map_structure_zip(lambda x, y: x, [{1}, {2}])

    # Test deep nesting
    nested1 = [1, [2, (3,)]]
    nested2 = [10, [20, (30,)]]
    nested3 = [100, [200, (300,)]]
    assert map_structure_zip(lambda x, y, z: x + y + z, [nested1, nested2, nested3]) == [111, [222, (333,)]]
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure():
    # Test simple integer mapping
    assert map_structure(lambda x: x * 2, 5) == 10

    # Test list mapping
    assert map_structure(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]

    # Test nested list mapping
    assert map_structure(lambda x: x * 2, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]

    # Test tuple mapping
    assert map_structure(lambda x: x * 2, (1, 2, 3)) == (2, 4, 6)

    # Test namedtuple mapping
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    assert map_structure(lambda x: x + 1, p) == Point(2, 3)

    # Test dict mapping
    d = {'a': 1, 'b': [2, 3]}
    expected_d = {'a': 2, 'b': [3, 4]}
    assert map_structure(lambda x: x + 1, d) == expected_d

    # Test set mapping
    s = {1, 2, 3}
    result_s = map_structure(lambda x: x * 10, s)
    assert result_s == {10, 20, 30}

    # Test no_map_instance (treating object as singleton)
    class MyClass:
        def __init__(self, val):
            self.val = val
    
    obj = MyClass(5)
    # If we wrap it in no_map_instance, the function should receive the object itself
    wrapped_obj = no_map_instance(obj)
    assert map_structure(lambda x: x.val + 1, wrapped_obj) == 6

    # Test register_no_map_class
    # We'll use a custom class to simulate a registered type
    class MyCustomContainer:
        def __init__(self, items):
            self.items = items
    
    register_no_map_class(MyCustomContainer)
    container = MyCustomContainer([1, 2, 3])
    # Because it is registered, map_structure should apply fn to the container itself, not its items
    assert map_structure(lambda x: len(x.items), container) == 3

    # Test deep nesting
    nested = [1, (2, {"a": 3}), [4, {5}]]
    # Increment all numbers
    expected_nested = [2, (3, {"a": 4}), [5, {6}]]
    assert map_structure(lambda x: x + 1, nested) == expected_nested

    # Test type preservation for dict subclasses (e.g., dict-like behavior)
    from collections import OrderedDict
    od = OrderedDict([('a', 1), ('b', 2)])
    result_od = map_structure(lambda x: x * 2, od)
    assert isinstance(result_od, OrderedDict)
    assert result_od['a'] == 2
    assert result_od['b'] == 4
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test basic list of lists
    fn_add = lambda x, y: x + y
    list1 = [[1, 2], [3]]
    list2 = [[10, 20], [30]]
    assert map_structure_zip(fn_add, [list1, list2]) == [[11, 22], [33]]

    # Test list of tuples
    fn_mul = lambda x, y: x * y
    tuple1 = [(1, 2), (3, 4)]
    tuple2 = [(5, 6), (7, 8)]
    assert map_structure_zip(fn_mul, [tuple1, tuple2]) == [(5, 12), (21, 32)]

    # Test list of dicts
    fn_sub = lambda x, y: x - y
    dict1 = {'a': 10, 'b': 20}
    dict2 = {'a': 1, 'b': 2}
    assert map_structure_zip(fn_sub, [dict1, dict2]) == {'a': 9, 'b': 18}

    # Test nested structures (list of dict of lists)
    fn_sum = lambda x, y: x + y
    struct1 = [{'val': [1, 2]}]
    struct2 = [{'val': [10, 20]}]
    expected = [{'val': [11, 22]}]
    assert map_structure_zip(fn_sum, [struct1, struct2]) == expected

    # Test namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p1 = Point(1, 2)
    p2 = Point(10, 20)
    fn_concat = lambda x, y: f"{x}-{y}"
    # Note: map_structure_zip treats the sequence of objects as the primary structure
    # If passed a list containing namedtuples:
    assert map_structure_zip(fn_concat, [[p1], [p2]]) == [[f"{p1.x}-{p2.x}", f"{p1.y}-{p2.y}"]] # This depends on zip behavior on inner elements
    # Correct way to test namedtuple as the container:
    assert map_structure_zip(fn_add, [p1, p2]) == Point(11, 22)

    # Test no_map_instance/class behavior
    class MyCustom:
        def __init__(self, val):
            self.val = val
    
    obj1 = MyCustom(1)
    obj2 = MyCustom(2)
    assert map_structure_zip(fn_add, [obj1, obj2]) == 3

    # Test error on set
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        set1 = {1, 2}
        set2 = {3, 4}
        map_structure_zip(fn_add, [set1, set2])

    # Test single element (base case)
    assert map_structure_zip(lambda x, y: x + y, [5, 5]) == 10
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test basic list of integers
    assert map_structure_zip(lambda x, y: x + y, [[1, 2], [3, 4]]) == [[4, 6]]
    
    # Test nested lists
    assert map_structure_zip(lambda x, y: x + y, [[[1], [2]], [[3], [4]]]) == [[[4], [6]]]
    
    # Test tuples
    assert map_structure_zip(lambda x, y: x * y, [(1, 2), (3, 4)]) == [(3, 8)]
    
    # Test dicts
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'a': 10, 'tuple': 20} # Note: key 'b' is missing in dict2, but map_structure_zip uses obj[k]
    # The implementation uses obj[k] for k in obj.keys(), so dict2 must have keys of dict1
    dict2 = {'a': 10, 'b': 20}
    assert map_structure_zip(lambda x, y: x + y, [dict1, dict2]) == {'a': 11, 'b': 22}

    # Test namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    result = map_structure_zip(lambda x, y: x + y, [p1, p2])
    assert isinstance(result, Point)
    assert result.x == 4
    assert result.y == 6

    # Test no_map_instance (treating a list as a single object)
    no_map_list = no_map_instance([1, 2])
    other_list = [3, 4]
    # Since no_map_list is treated as a singleton, it should call fn(no_map_list, other_list)
    assert map_structure_zip(lambda x, y: len(x) + len(y), [no_map_list, other_list]) == 4

    # Test register_no_map_class
    register_no_map_class(list)
    # Now all lists are treated as singletons
    assert map_structure_zip(lambda x, y: x + y, [[1], [2]]) == [3]

    # Test ValueError for sets
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(lambda x, y: x + y, [{1}, {2}])

    # Test scalar/leaf elements
    assert map_structure_zip(lambda x, y: x + y, [1, 2]) == 3
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure_zip():
    # Test simple scalars (treated as singletons)
    assert map_structure_zip(lambda x, y: x + y, [1, 2]) == 3
    assert map_structure_zip(lambda x, y: x * y, [10, 20]) == 200

    # Test lists
    list_objs = [[1, 2], [3, 4]]
    assert map_structure_zip(lambda x, y: x + y, list_objs) == [4, 6]

    # Test nested lists
    nested_objs = [[[1], [2]], [[3], [4]]]
    assert map_structure_zip(lambda x, y: x + y, nested_objs) == [[[4], [6]]]

    # Test tuples
    tuple_objs = [(1, 2), (3, 4)]
    assert map_structure_zip(lambda x, y: x + y, tuple_objs) == (4, 6)

    # Test namedtuples
    Point = namedtuple("Point", ["x", "y"])
    nt_objs = [Point(1, 2), Point(3, 4)]
    result_nt = map_structure_zip(lambda x, y: x + y, nt_objs)
    assert isinstance(result_nt[0], Point)
    assert result_nt[0] == Point(4, 6)
    assert result_nt[1] == Point(6, 8)

    # Test dicts
    dict_objs = [{"a": 1, "b": 2}, {"a": 10, "b": 20}]
    assert map_structure_zip(lambda x, y: x + y, dict_objs) == {"a": 11, "b": 22}

    # Test mixed structures (matching structure)
    mixed_objs = [
        [{"a": 1}, (2, 3)],
        [{"a": 10}, (20, 30)]
    ]
    expected = [
        {"a": 11},
        (22, 33)
    ]
    assert map_structure_zip(lambda x, y: x + y, mixed_objs) == expected

    # Test no_map_instance/type registration
    # We use a custom class to simulate a non-mappable container
    class NonMappable:
        def __init__(self, val):
            self.val = val
    
    nm_obj1 = NonMappable(1)
    nm_obj2 = NonMappable(10)
    # Since no_map_instance adds an attribute, map_structure_zip should treat it as a singleton
    assert map_structure_zip(lambda x, y: x.val + y.val, [nm_obj1, nm_obj2]) == 11

    # Test ValueError for sets
    set_objs = [{1, 2}, {3, 4}]
    with pytest.raises(ValueError, match="Structures cannot contain `set`"):
        map_structure_zip(lambda x, y: x + y, set_objs)

    # Test register_no_map_class
    class MySpecialContainer(list):
        pass
    
    register_no_map_class(MySpecialContainer)
    special_objs = [MySpecialContainer([1, 2]), MySpecialContainer([3, 4])]
    # Because MySpecialContainer is registered, it shouldn't recurse into its elements
    # It should treat the container itself as the unit to be passed to fn
    # Note: map_structure_zip(fn, [obj1, obj2]) calls fn(element_from_obj1, element_from_obj2)
    # For lists, it zips the elements. For registered types, it passes the objects themselves.
    assert map_structure_zip(lambda x, y: len(x) + len(y), special_objs) == 4
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from collections import namedtuple, OrderedDict

def test_map_structure():
    # Test basic list mapping
    assert map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

    # Test nested list mapping
    assert map_structure(lambda x: x + 1, [[1], [2, [3]]]) == [[2], [3, [4]]]

    # Test tuple mapping
    assert map_structure(lambda x: str(x), (1, 2, 3)) == ("1", "2", "tuple") # Wait, logic check
    assert map_structure(lambda x: str(x), (1, 2)) == ("1", "2")

    # Test namedtuple mapping
    Point = namedtuple("Point", ["x", "y"])
    p = Point(1, 2)
    assert map_structure(lambda x: x * 10, p) == Point(10, 20)

    # Test dict mapping
    d = {"a": 1, "b": [2, 3]}
    assert map_structure(lambda x: x + 1, d) == {"a": 2, "b": [3, 4]}

    # Test OrderedDict mapping
    od = OrderedDict([("a", 1), ("b", 2)])
    assert map_structure(lambda x: x * 2, od) == OrderedDict([("a", 2), ("b", 4)])

    # Test set mapping
    s = {1, 2, 3}
    # Sets are unordered, so we check membership
    result_s = map_structure(lambda x: x + 1, s)
    assert result_s == {2, 3, 4}

    # Test leaf nodes (non-collections)
    assert map_structure(lambda x: x + 5, 10) == 15

    # Test register_no_map_class
    register_no_map_class(set)
    # Since set is now registered as no-map, it should be treated as a leaf
    # map_structure will call fn(set_obj) instead of iterating
    s_no_map = {1, 2}
    # The function passed should receive the set itself
    assert map_structure(lambda x: len(x), s_no_map) == 2
    
    # Resetting global state for other tests if necessary is hard without 
    # access to _NO_MAP_TYPES, but for this unit test we assume clean environment
    # or that the behavior is intended.

    # Test no_map_instance
    # We use a custom class that allows setattr to simulate a mutable container
    class MyContainer:
        def __init__(self, val):
            self.val = val
    
    c = MyContainer([1, 2])
    # Wrap instance to be non-mappable
    no_map_c = no_map_instance(c)
    # map_structure should see the attribute and not traverse
    assert map_structure(lambda x: x.val, no_map_c) == [1, 2]
    # But it should call fn(no_map_c) and return the result of fn
    # Actually, the logic is: if hasattr(obj, _NO_MAP_INSTANCE_ATTR): return fn(obj)
    # So it should return the result of fn(no_map_c)
    assert map_structure(lambda x: "found", no_map_c) == "found"

    # Test deep nesting mix
    complex_obj = [1, {"a": (2, 3)}, {4, 5}]
    # Note: if set is registered as no_map, {4, 5} becomes a leaf
    # Let's assume set is NOT registered for the complex test
    # To make this test robust, we'd ideally use a fresh process, 
    # but here we test the logic provided.
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure():
    # Test basic scalar mapping
    assert map_structure(lambda x: x + 1, 1) == 2

    # Test list mapping
    assert map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

    # Test nested list mapping
    assert map_structure(lambda x: x + 1, [[1], [2, [3]]]) == [[2], [3, [4]]]

    # Test tuple mapping
    assert map_structure(lambda x: x * 2, (1, 2)) == (2, 4)

    # Test namedtuple mapping
    Point = namedtuple("Point", ["x", "y"])
    assert map_structure(lambda x: x + 1, Point(1, 2)) == Point(2, 3)

    # Test dict mapping (values only)
    assert map_structure(lambda x: x * 10, {"a": 1, "b": 2}) == {"a": 10, "b": 20}

    # Test dict with nested structures
    nested_dict = {"a": [1, 2], "b": {"c": 3}}
    expected_dict = {"a": [2, 3], "b": {"c": 4}}
    assert map_structure(lambda x: x + 1, nested_dict) == expected_dict

    # Test set mapping
    assert map_structure(lambda x: x * 2, {1, 2, 3}) == {2, 4, 6}

    # Test registered no-map class
    class MyContainer(list):
        pass
    
    register_no_map_class(MyContainer)
    container_instance = MyContainer([1, 2, 3])
    # Since MyContainer is registered as no-map, it should apply fn to the object itself
    assert map_structure(lambda x: len(x), container_instance) == 3

    # Test no_map_instance
    # We use an object that allows attribute setting (like a simple class)
    class MockObj:
        def __init__(self, val):
            self.val = val
    
    mock_obj = MockObj(10)
    # no_map_instance marks the instance as non-traversable
    no_map_val = no_map_instance(mock_obj)
    # map_structure should see the attribute and return fn(obj)
    assert map_structure(lambda x: x.val + 5, no_map_val) == 15

    # Test no_map_instance with immutable type (triggering _no_map_type)
    # Using a tuple which is immutable and cannot have attributes set
    immutable_tuple = (1, 2)
    no_map_tuple = no_map_instance(immutable_tuple)
    # It should treat the tuple as a single unit and not traverse it
    assert map_structure(lambda x: len(x), no_map_tuple) == 2
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from collections import namedtuple, OrderedDict

def test_map_structure():
    # Test basic mapping on list
    assert map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]

    # Test nested list
    assert map_structure(lambda x: x + 1, [[1], [2, [3]]]) == [[2], [3, [4]]]

    # Test tuple
    assert map_structure(lambda x: str(x), (1, 2)) == ("1", "2")

    # Test namedtuple
    Point = namedtuple("Point", ["x", "y"])
    assert map_structure(lambda x: x * 10, Point(1, 2)) == Point(10, 20)

    # Test dict
    assert map_structure(lambda x: x * 2, {"a": 1, "b": 2}) == {"a": 2, "b": 4}

    # Test OrderedDict
    od = OrderedDict([("a", 1), ("b", 2)])
    result_od = map_structure(lambda x: x + 1, od)
    assert isinstance(result_od, OrderedDict)
    assert list(result_od.items()) == [("a", 2), ("b", 3)]

    # Test set
    # Note: set order is not guaranteed, so we check content
    assert map_structure(lambda x: x * 2, {1, 2}) == {2, 4}

    # Test leaf nodes (non-collections)
    assert map_structure(lambda x: x + 5, 10) == 15

    # Test register_no_map_class
    class MyCustomContainer(list):
        pass

    register_no_map_class(MyCustomContainer)
    # Because MyCustomContainer is registered, it should be treated as a single unit
    # The function fn should be applied to the instance itself, not its elements
    assert map_structure(lambda x: len(x), MyCustomContainer([1, 2, 3])) == 3

    # Test no_map_instance
    # Create a list and mark it as no_map
    no_map_list = no_map_instance([1, 2, 3])
    # map_structure should see the attribute and apply fn to the list itself
    assert map_structure(lambda x: len(x), no_map_list) == 3

    # Test deeply nested mixed structures
    complex_obj = [
        {"a": (1, 2)},
        {"b": [3, 4]},
        (5, 6)
    ]
    expected_obj = [
        {"a": (2, 4)},
        {"b": [6, 8]},
        (10, 12)
    ]
    assert map_structure(lambda x: x * 2, complex_obj) == expected_obj

    # Test no_map_type via subclassing behavior
    # We can't easily check the private _no_map_type without knowing its name, 
    # but we can verify that if we pass a type that was processed by _no_map_type,
    # it behaves as a leaf.
    class MockType(list):
        pass
    
    # Manually trigger the creation of a no-map type via the internal logic
    # Since we can't easily access _no_map_type without importing, we rely on 
    # the fact that no_map_instance uses it.
    no_map_tuple = no_map_instance((1, 2))
    assert map_structure(lambda x: len(x), no_map_tuple) == 2
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure():
    # Test simple scalar/leaf node
    assert map_structure(lambda x: x + 1, 1) == 2
    
    # Test list mapping
    assert map_tuple_list = map_structure(lambda x: x * 2, [1, 2, 3])
    assert map_tuple_list == [2, 4, 6]
    
    # Test nested list mapping
    assert map_structure(lambda x: x + 1, [[1, 2], [3]]) == [[2, 3], [4]]
    
    # Test tuple mapping
    assert map_structure(lambda x: x * 2, (1, 2)) == (2, 4)
    
    # Test namedtuple mapping
    Point = namedtuple("Point", ["x", "y"])
    assert map_structure(lambda x: x + 10, Point(1, 2)) == Point(11, 12)
    
    # Test dict mapping (values only)
    assert map_structure(lambda x: x * 3, {"a": 1, "b": 2}) == {"a": 3, "b": 6}
    
    # Test nested dict mapping
    nested_dict = {"a": [1, 2], "b": {"c": 3}}
    expected_dict = {"a": [2, 3], "b": {"c": 4}}
    assert map_structure(lambda x: x + 1, nested_dict) == expected_dict
    
    # Test set mapping
    assert map_structure(lambda x: x * 2, {1, 2, 3}) == {2, 4, 6}
    
    # Test no_map_instance (treating a list as a singleton)
    no_map_list = no_map_instance([1, 2, 3])
    # The function should apply to the list object itself, not its elements
    # Since we can't easily check the identity of the returned list without 
    # knowing if it's a new list or the same, we check the behavior:
    # If it's no_map, it returns fn(obj), so [1, 2, 3] + [1] = [1, 2, 3, 1]
    assert map_structure(lambda x: x + [4], no_map_list) == [1, 2, 3, 4]
    
    # Test register_no_map_class
    register_no_map_class(tuple)
    # Now all tuples should be treated as singletons (not traversed)
    assert map_structure(lambda x: x + 1, (1, 2)) == (1, 2) + (1,) # This is an error in logic, 
    # actually: fn(obj) where obj is (1, 2) -> (1, 2) + 1 is impossible.
    # Let's use a compatible function for the registered type.
    assert map_structure(lambda x: len(x), (1, 2)) == 2
    
    # Resetting the global state for other tests is hard without refactoring, 
    # but for this specific test scope, we verify the behavior of the registered type.
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from collections import namedtuple

def test_map_structure():
    # Test basic list mapping
    assert map_structure(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]

    # Test nested list mapping
    assert map_tuple_nested = map_structure(lambda x: x * 2, [[1, 2], [3, [4]]])
    assert map_tuple_nested == [[2, 4], [6, [8]]]

    # Test tuple mapping
    assert map_structure(lambda x: x * 2, (1, 2, 3)) == (2, 4, 6)

    # Test namedtuple mapping
    Point = namedtuple("Point", ["x", "y"])
    p = Point(1, 2)
    assert map_structure(lambda x: x + 1, p) == Point(2, 3)

    # Test dict mapping
    d = {"a": 1, "b": {"c": 2}}
    assert map_structure(lambda x: x + 1, d) == {"a": 2, "b": {"c": 3}}

    # Test set mapping
    s = {1, 2, 3}
    assert map_structure(lambda x: x * 10, s) == {10, 20, 30}

    # Test singleton/leaf elements (non-collections)
    assert map_structure(lambda x: x + 1, 5) == 6

    # Test register_no_map_class
    class MyCustomContainer(list):
        pass

    register_no_map_class(MyCustomContainer)
    custom_list = MyCustomContainer([1, 2, 3])
    # Should treat the whole list as one element and apply fn to it
    # Since fn(obj) is called, and obj is the list, 
    # we check if it returns the list itself (if fn is identity) 
    # or a transformed version. 
    # Note: map_structure returns fn(obj) if type is in _NO_MAP_TYPES.
    assert map_structure(lambda x: len(x), custom_list) == 3

    # Test no_map_instance
    # We use a class that allows setattr to demonstrate no_map_instance
    class MutableBox:
        def __init__(self, val):
            self.val = val
    
    box = MutableBox(10)
    no_mapped_box = no_map_instance(box)
    # map_structure should see the attribute and not traverse
    assert map_structure(lambda x: x.val + 1, no_mapped_box) == 11

    # Test no_map_instance with immutable type (triggers _no_map_type)
    # Tuples are immutable, so it creates a subclass
    t = (1, 2)
    no_mapped_t = no_map_instance(t)
    assert map_structure(lambda x: len(x), no_mapped_t) == 2
    assert isinstance(no_mapped_t, tuple)
```


