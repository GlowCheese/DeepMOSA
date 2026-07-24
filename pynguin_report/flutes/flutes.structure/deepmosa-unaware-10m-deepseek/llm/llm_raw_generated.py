####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    objs = [[1, 2, 3], [4, 5, 6]]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == [5, 7, 9]

    # Test with nested lists
    objs = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = map_structure_zip(lambda x, y: x * y, objs)
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    objs = [((1, 2), (3, 4)), ((5, 6), (7, 8))]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == ((6, 8), (10, 12))

    # Test with dicts
    objs = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    result = map_structure_zip(lambda x, y: x - y, objs)
    assert result == {'a': -2, 'b': -2}

    # Test with mixed structures
    objs = [{'a': [1, 2], 'b': (3, 4)}, {'a': [5, 6], 'b': (7, 8)}]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == {'a': [6, 8], 'b': (10, 12)}

    # Test with three collections
    objs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = map_structure_zip(lambda x, y, z: x + y + z, objs)
    assert result == [12, 15, 18]

    # Test with no_map_instance
    import sys
    no_map_list = no_map_instance([1, 2, 3])
    objs = [no_map_list, no_map_list]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2])
    objs = [custom_list, custom_list]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert isinstance(result, list)
    assert result == [1, 2, 1, 2]

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    objs = [Point(1, 2), Point(3, 4)]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == Point(4, 6)

    # Test with set should raise ValueError
    objs = [{1, 2}, {3, 4}]
    try:
        map_structure_zip(lambda x, y: x + y, objs)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with empty structure
    objs = [[], []]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == []

    # Test with single collection
    objs = [[1, 2, 3]]
    result = map_structure_zip(lambda x: x * 2, objs)
    assert result == [2, 4, 6]


# LLM-generated content at query #2
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: f"{x}{y}", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert result == {"a": "13", "b": "24"}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": [6, 8], "b": (10, 12)}

    # Test with single input
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with no_map_instance
    import collections
    Point = collections.namedtuple("Point", ["x", "y"])
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    result = map_structure_zip(lambda x, y: Point(x.x + y.x, x.y + y.y), [p1, p2])
    assert result == Point(4, 6)

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom1 = CustomList([1, 2, 3])
    custom2 = CustomList([4, 5, 6])
    result = map_structure_zip(lambda x, y: str(x) + str(y), [custom1, custom2])
    assert isinstance(result, str)

    # Test with no_map_instance
    normal_list = [1, 2, 3]
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [normal_list, no_map_list])
    assert isinstance(result, int)

    # Test error with sets
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    result = map_structure_zip(lambda x, y: x + y, [{}, {}])
    assert result == {}

    # Test with three inputs
    result = map_structure_zip(lambda x, y, z: x + y + z, [[1, 2], [3, 4], [5, 6]])
    assert result == [9, 12]


# LLM-generated content at query #3
#--------------------------

```python
def test_map_structure_zip():
    # Test with simple flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]
    
    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]
    
    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    result = map_structure_zip(lambda a, b: a + b, [Point(1, 2), Point(3, 4)])
    assert result == Point(4, 6)
    
    # Test with dictionaries
    result = map_structure_zip(lambda x, y: x - y, [{'a': 10, 'b': 20}, {'a': 3, 'b': 5}])
    assert result == {'a': 7, 'b': 15}
    
    # Test with mixed structures
    obj1 = {'list': [1, 2], 'tuple': (3, 4), 'nested': {'a': 5}}
    obj2 = {'list': [6, 7], 'tuple': (8, 9), 'nested': {'a': 10}}
    result = map_structure_zip(lambda x, y: x * y, [obj1, obj2])
    expected = {'list': [6, 14], 'tuple': (24, 36), 'nested': {'a': 50}}
    assert result == expected
    
    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: str(x) + str(y), [no_map_list, no_map_list])
    assert result == '[1, 2, 3][1, 2, 3]'
    
    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom_list, custom_list])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]
    
    # Test with single argument (should still work)
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]
    
    # Test with three arguments
    result = map_structure_zip(lambda x, y, z: x + y + z, [[1, 2], [3, 4], [5, 6]])
    assert result == [9, 12]
    
    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []
    
    # Test that sets raise ValueError
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError for sets"
    except ValueError:
        pass
    
    # Test with deeply nested structure
    complex_obj1 = {'a': [1, (2, {'b': 3})], 'c': {'d': [4, 5]}}
    complex_obj2 = {'a': [6, (7, {'b': 8})], 'c': {'d': [9, 10]}}
    result = map_structure_zip(lambda x, y: x - y, [complex_obj1, complex_obj2])
    expected = {'a': [-5, (-5, {'b': -5})], 'c': {'d': [-5, -5]}}
    assert result == expected


# LLM-generated content at query #4
#--------------------------

```python
def test_map_structure_zip():
    # Test with simple flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested structures
    obj1 = [1, {"a": 2, "b": 3}, (4, 5)]
    obj2 = [10, {"a": 20, "b": 30}, (40, 50)]
    result = map_structure_zip(lambda x, y: x + y, [obj1, obj2])
    assert result == [11, {"a": 22, "b": 33}, (44, 55)]

    # Test with three collections
    result = map_structure_zip(lambda x, y, z: x + y + z, [[1, 2], [3, 4], [5, 6]])
    assert result == [9, 12]

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    result = map_structure_zip(lambda x, y: x + y, [p1, p2])
    assert result == Point(4, 6)

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom1 = CustomList([1, 2])
    custom2 = CustomList([3, 4])
    result = map_structure_zip(lambda x, y: str(x) + str(y), [custom1, custom2])
    assert result == "[1, 2][3, 4]"

    # Test with dict (including OrderedDict-like)
    from collections import OrderedDict
    d1 = OrderedDict([("a", 1), ("b", 2)])
    d2 = OrderedDict([("a", 3), ("b", 4)])
    result = map_structure_zip(lambda x, y: x * y, [d1, d2])
    assert list(result.items()) == [("a", 3), ("b", 8)]

    # Test with set should raise ValueError
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with mixed structures
    mixed1 = {"a": [1, 2], "b": {"c": 3, "d": 4}}
    mixed2 = {"a": [5, 6], "b": {"c": 7, "d": 8}}
    result = map_structure_zip(lambda x, y: x - y, [mixed1, mixed2])
    assert result == {"a": [-4, -4], "b": {"c": -4, "d": -4}}

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]


# LLM-generated content at query #5
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}
    
    # Test with nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1}, "b": {"y": 2}})
    assert result == {"a": {"x": 11}, "b": {"y": 12}}
    
    # Test with set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}
    
    # Test with mixed nested structure
    obj = {"a": [1, 2, {"x": 3}], "b": (4, 5)}
    result = map_structure(lambda x: x * 2, obj)
    assert result == {"a": [2, 4, {"x": 6}], "b": (8, 10)}
    
    # Test with non-container (leaf node)
    result = map_structure(lambda x: x + 1, 5)
    assert result == 6
    
    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [1, 2, 3] * 2
    
    # Test with registered no_map_class
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == CustomList([1, 2, 3]) * 2
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x + 10, point)
    assert result == Point(11, 12)
    
    # Test with empty structures
    result = map_structure(lambda x: x, [])
    assert result == []
    
    result = map_structure(lambda x: x, {})
    assert result == {}
    
    result = map_structure(lambda x: x, ())
    assert result == ()
    
    # Test function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ["1", "2", "3"]
    
    # Test with complex nested structure
    obj = {
        "list": [1, 2, {"a": 3}],
        "tuple": (4, [5, 6]),
        "set": {7, 8},
        "nested": {"x": {"y": [9, 10]}}
    }
    expected = {
        "list": [2, 4, {"a": 6}],
        "tuple": (8, [10, 12]),
        "set": {14, 16},
        "nested": {"x": {"y": [18, 20]}}
    }
    result = map_structure(lambda x: x * 2, obj)
    assert result == expected


# LLM-generated content at query #6
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: f"{x}{y}", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert result == {"a": "13", "b": "24"}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": [6, 8], "b": (10, 12)}

    # Test with no_map_instance
    import sys
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom, custom])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(lambda x, y, z: x + y + z, [[1, 2], [3, 4], [5, 6]])
    assert result == [9, 12]

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test with sets (should raise ValueError)
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    result = map_structure_zip(
        lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y),
        [Point(1, 2), Point(3, 4)]
    )
    assert result == Point(4, 6)
    assert isinstance(result, Point)


# LLM-generated content at query #7
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}
    
    # Test with nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1}, "b": {"y": 2}})
    assert result == {"a": {"x": 11}, "b": {"y": 12}}
    
    # Test with set
    result = map_structure(lambda x: x * 2, {1, 2, 3})
    assert result == {2, 4, 6}
    
    # Test with mixed nested structure
    obj = {"a": [1, 2, {"x": 3}], "b": (4, 5)}
    result = map_structure(lambda x: x + 1, obj)
    assert result == {"a": [2, 3, {"x": 4}], "b": (5, 6)}
    
    # Test with non-container (leaf node)
    result = map_structure(lambda x: x * 2, 5)
    assert result == 10
    
    # Test with registered no-map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == [2, 4, 6]  # The entire list gets doubled, not elements
    
    # Test with no-map instance
    normal_list = [1, 2, 3]
    no_map_list = no_map_instance(normal_list)
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [2, 4, 6]  # The entire list gets doubled
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 2, point)
    assert result == Point(2, 4)
    
    # Test with empty structures
    result = map_structure(lambda x: x * 2, [])
    assert result == []
    
    result = map_structure(lambda x: x * 2, {})
    assert result == {}
    
    result = map_structure(lambda x: x * 2, set())
    assert result == set()
    
    # Test with function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ["1", "2", "3"]
    
    # Test with deep nesting
    obj = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = map_structure(lambda x: x - 1, obj)
    assert result == [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]


# LLM-generated content at query #8
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ('a', 'b', 'c'))
    assert result == ('A', 'B', 'C')
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x + 10, {'a': 1, 'b': 2})
    assert result == {'a': 11, 'b': 12}
    
    # Test with nested dict
    result = map_structure(lambda x: len(x), {'a': 'hi', 'b': 'hello'})
    assert result == {'a': 2, 'b': 5}
    
    # Test with set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}
    
    # Test with mixed structure
    obj = {'a': [1, 2, 3], 'b': (4, 5, 6)}
    result = map_structure(lambda x: x * 2, obj)
    assert result == {'a': [2, 4, 6], 'b': (8, 10, 12)}
    
    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [1, 2, 3] * 2
    
    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == [1, 2, 3] * 2
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    point = Point(1, 2)
    result = map_structure(lambda x: x + 10, point)
    assert result == Point(11, 12)
    
    # Test with simple value (non-collection)
    result = map_structure(lambda x: x + 5, 10)
    assert result == 15
    
    # Test with empty structures
    result = map_structure(lambda x: x, [])
    assert result == []
    
    result = map_structure(lambda x: x, {})
    assert result == {}
    
    # Test with function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ['1', '2', '3']
    
    # Test with deeply nested structure
    obj = {'a': [{'b': (1, 2)}, {'c': [3, 4]}]}
    result = map_structure(lambda x: x * 2, obj)
    expected = {'a': [{'b': (2, 4)}, {'c': [6, 8]}]}
    assert result == expected


# LLM-generated content at query #9
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}
    
    # Test with nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1}, "b": {"y": 2}})
    assert result == {"a": {"x": 11}, "b": {"y": 12}}
    
    # Test with set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}
    
    # Test with mixed nested structure
    obj = {"a": [1, 2, 3], "b": (4, 5, 6), "c": {7, 8}}
    result = map_structure(lambda x: x * 2, obj)
    assert result == {"a": [2, 4, 6], "b": (8, 10, 12), "c": {14, 16}}
    
    # Test with non-container (leaf node)
    result = map_structure(lambda x: x + 1, 5)
    assert result == 6
    
    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [1, 2, 3] * 2
    
    # Test with registered no_map_class
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == [1, 2, 3] * 2
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 2, point)
    assert result == Point(2, 4)
    
    # Test with empty structures
    result = map_structure(lambda x: x, [])
    assert result == []
    
    result = map_structure(lambda x: x, {})
    assert result == {}
    
    # Test with function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ["1", "2", "3"]


# LLM-generated content at query #10
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: x - y, [{"a": 10, "b": 20}, {"a": 3, "b": 5}])
    assert result == {"a": 7, "b": 15}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": [6, 8], "b": (10, 12)}

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(
        lambda x, y, z: f"{x}{y}{z}",
        [["a", "b"], ["c", "d"], ["e", "f"]]
    )
    assert result == ["ace", "bdf"]

    # Test with no_map_instance
    import sys
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class SpecialList(list):
        pass
    
    register_no_map_class(SpecialList)
    special = SpecialList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [special, special])
    assert isinstance(result, list)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    result = map_structure_zip(
        lambda a, b: Point(a.x + b.x, a.y + b.y),
        [Point(1, 2), Point(3, 4)]
    )
    assert result == Point(4, 6)
    assert result.x == 4
    assert result.y == 6

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    result = map_structure_zip(lambda x, y: x + y, [{}, {}])
    assert result == {}

    # Test with set should raise ValueError
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with different dict types
    from collections import OrderedDict
    od1 = OrderedDict([("a", 1), ("b", 2)])
    od2 = OrderedDict([("a", 3), ("b", 4)])
    result = map_structure_zip(lambda x, y: x + y, [od1, od2])
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [("a", 4), ("b", 6)]


# LLM-generated content at query #11
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}
    
    # Test with nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1}, "b": {"y": 2}})
    assert result == {"a": {"x": 11}, "b": {"y": 12}}
    
    # Test with set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}
    
    # Test with mixed nested structure
    obj = [{"a": (1, 2), "b": [3, 4]}, {"c": {5, 6}}]
    result = map_structure(lambda x: x + 1, obj)
    expected = [{"a": (2, 3), "b": [4, 5]}, {"c": {6, 7}}]
    assert result == expected
    
    # Test with non-mappable registered type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom)
    assert result == [2, 4, 6]  # Should apply fn to the whole object
    
    # Test with no_map_instance
    lst = [1, 2, 3]
    no_map_lst = no_map_instance(lst)
    result = map_structure(lambda x: x * 2, no_map_lst)
    assert result == [2, 4, 6]  # Should apply fn to the whole object
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    p = Point(1, 2)
    result = map_structure(lambda x: x * 2, p)
    assert result == Point(2, 4)
    
    # Test with empty structures
    result = map_structure(lambda x: x, [])
    assert result == []
    
    result = map_structure(lambda x: x, {})
    assert result == {}
    
    result = map_structure(lambda x: x, set())
    assert result == set()
    
    # Test with non-collection
    result = map_structure(lambda x: x * 2, 5)
    assert result == 10
    
    # Test with string (should be treated as non-collection)
    result = map_structure(lambda x: x + "!", "hello")
    assert result == "hello!"
    
    # Test with nested no_map_instance
    obj = [no_map_instance([1, 2]), [3, 4]]
    result = map_structure(lambda x: str(x), obj)
    assert result == ["[1, 2]", [3, 4]]


# LLM-generated content at query #12
#--------------------------

```python
def test_map_structure_zip():
    # Test with simple flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]
    
    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]
    
    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)
    
    # Test with dictionaries
    result = map_structure_zip(lambda x, y: f"{x}{y}", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert result == {"a": "13", "b": "24"}
    
    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": [6, 8], "b": (10, 12)}
    
    # Test with registered no-map type
    class NoMapList(list):
        pass
    
    register_no_map_class(NoMapList)
    no_map_obj = NoMapList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_obj, no_map_obj])
    assert isinstance(result, NoMapList)
    assert result == [1, 2, 3, 1, 2, 3]
    
    # Test with no-map instance
    normal_list = [1, 2, 3]
    no_map_instance_list = no_map_instance([4, 5, 6])
    result = map_structure_zip(lambda x, y: x + y, [normal_list, no_map_instance_list])
    assert result == [1, 2, 3, 4, 5, 6]
    
    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]
    
    # Test with three collections
    result = map_structure_zip(lambda x, y, z: x + y + z, [[1, 2], [3, 4], [5, 6]])
    assert result == [9, 12]
    
    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []
    
    # Test that sets raise ValueError
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError for sets"
    except ValueError:
        pass
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    result = map_structure_zip(
        lambda a, b: Point(a.x + b.x, a.y + b.y),
        [Point(1, 2), Point(3, 4)]
    )
    assert result == Point(4, 6)
    assert isinstance(result, Point)


# LLM-generated content at query #13
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}
    
    # Test with nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1, "y": 2}, "b": 3})
    assert result == {"a": {"x": 11, "y": 12}, "b": 13}
    
    # Test with set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}
    
    # Test with mixed nested structure
    obj = {"a": [1, 2, {"x": 3}], "b": (4, 5)}
    result = map_structure(lambda x: x * 2, obj)
    assert result == {"a": [2, 4, {"x": 6}], "b": (8, 10)}
    
    # Test with non-container leaf
    result = map_structure(lambda x: x + 1, 5)
    assert result == 6
    
    # Test with registered no-map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == [2, 4, 6]  # The list itself is passed to fn
    
    # Test with no-map instance
    normal_list = [1, 2, 3]
    no_map_list = no_map_instance(normal_list)
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [2, 4, 6]  # The list itself is passed to fn
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 2, point)
    assert result == Point(2, 4)
    
    # Test with empty structures
    result = map_structure(lambda x: x * 2, [])
    assert result == []
    
    result = map_structure(lambda x: x * 2, {})
    assert result == {}
    
    # Test with function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ["1", "2", "3"]
    
    # Test with deeply nested structure
    obj = {"a": [{"b": (1, 2)}, [3, 4]], "c": 5}
    result = map_structure(lambda x: x + 10, obj)
    assert result == {"a": [{"b": (11, 12)}, [13, 14]], "c": 15}


# LLM-generated content at query #14
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: f"{x}{y}", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert result == {"a": "13", "b": "24"}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": [6, 8], "b": (10, 12)}

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    result = map_structure_zip(
        lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y),
        [Point(1, 2), Point(3, 4)]
    )
    assert result == Point(4, 6)
    assert isinstance(result, Point)

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom_list, custom_list])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(
        lambda x, y, z: x + y + z,
        [[1, 2], [3, 4], [5, 6]]
    )
    assert result == [9, 12]

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test with set should raise ValueError
    import pytest
    with pytest.raises(ValueError):
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])


# LLM-generated content at query #15
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    result = map_structure_zip(lambda a, b: Point(a.x + b.x, a.y + b.y), 
                               [Point(1, 2), Point(3, 4)])
    assert result == Point(4, 6)

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: x + y, 
                               [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
    assert result == {'a': 4, 'b': 6}

    # Test with mixed nested structures
    result = map_structure_zip(lambda x, y: x + y, 
                               [[{'a': 1}, {'b': 2}], [{'a': 3}, {'b': 4}]])
    assert result == [{'a': 4}, {'b': 6}]

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom_list, custom_list])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with single argument (should still work)
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three arguments
    result = map_structure_zip(lambda x, y, z: x + y + z, 
                               [[1, 2], [3, 4], [5, 6]])
    assert result == [9, 12]

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test that set raises ValueError
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with deep nesting
    structure = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = map_structure_zip(lambda x, y: x - y, [structure, structure])
    expected = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    assert result == expected


# LLM-generated content at query #16
#--------------------------

```python
def test_map_structure():
    # Test basic mapping with list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}
    
    # Test nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1}, "b": {"y": 2}})
    assert result == {"a": {"x": 11}, "b": {"y": 12}}
    
    # Test set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}
    
    # Test mixed structure
    obj = [{"a": (1, 2), "b": [3, 4]}, 5]
    result = map_structure(lambda x: x * 2, obj)
    assert result == [{"a": (2, 4), "b": [6, 8]}, 10]
    
    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [1, 2, 3] * 2
    
    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == [1, 2, 3] * 2
    
    # Test namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x + 10, point)
    assert result == Point(11, 12)
    
    # Test scalar (non-collection)
    result = map_structure(lambda x: x * 2, 5)
    assert result == 10
    
    # Test string (non-collection)
    result = map_structure(lambda x: x.upper(), "hello")
    assert result == "HELLO"
    
    # Test empty structures
    result = map_structure(lambda x: x * 2, [])
    assert result == []
    
    result = map_structure(lambda x: x * 2, {})
    assert result == {}
    
    # Test function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ["1", "2", "3"]


# LLM-generated content at query #17
#--------------------------

```python
def test_map_structure():
    # Test basic mapping with simple types
    def add_one(x):
        return x + 1

    assert map_structure(add_one, 5) == 6
    assert map_structure(add_one, [1, 2, 3]) == [2, 3, 4]
    assert map_structure(add_one, (1, 2, 3)) == (2, 3, 4)
    assert map_structure(add_one, {1: 2, 3: 4}) == {1: 3, 3: 5}
    assert map_structure(add_one, {1, 2, 3}) == {2, 3, 4}

    # Test nested structures
    nested_list = [[1, 2], [3, 4]]
    assert map_structure(add_one, nested_list) == [[2, 3], [4, 5]]

    nested_dict = {"a": {"x": 1, "y": 2}, "b": {"z": 3}}
    assert map_structure(add_one, nested_dict) == {"a": {"x": 2, "y": 3}, "b": {"z": 4}}

    mixed = {"a": [1, 2, 3], "b": (4, 5), "c": {6, 7}}
    result = map_structure(add_one, mixed)
    assert result["a"] == [2, 3, 4]
    assert result["b"] == (5, 6)
    assert result["c"] == {7, 8}

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    assert map_structure(add_one, no_map_list) == [2, 3, 4]

    # Test with registered no_map types
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    assert map_structure(add_one, custom_list) == 4

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(add_one, point)
    assert result.x == 2
    assert result.y == 3
    assert isinstance(result, Point)

    # Test with string function
    def repeat(x):
        return str(x) * 2

    assert map_structure(repeat, [1, "a", True]) == ["11", "aa", "TrueTrue"]
    assert map_structure(repeat, {"key": "value"}) == {"key": "valuevalue"}

    # Test with empty structures
    assert map_structure(add_one, []) == []
    assert map_structure(add_one, {}) == {}
    assert map_structure(add_one, set()) == set()
    assert map_structure(add_one, ()) == ()

    # Test with complex nested structure
    complex_structure = {
        "list": [1, {"inner": 2}, (3, 4)],
        "tuple": (5, [6, 7]),
        "set": {8, 9}
    }
    result = map_structure(add_one, complex_structure)
    assert result["list"] == [2, {"inner": 3}, (4, 5)]
    assert result["tuple"] == (6, [7, 8])
    assert result["set"] == {9, 10}

    # Test that original structure is preserved
    original = {"a": [1, 2, 3], "b": {"c": 4, "d": 5}}
    result = map_structure(add_one, original)
    assert isinstance(result, dict)
    assert isinstance(result["a"], list)
    assert isinstance(result["b"], dict)
    assert result["a"] == [2, 3, 4]
    assert result["b"] == {"c": 5, "d": 6}


# LLM-generated content at query #18
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}
    
    # Test with nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1}, "b": {"y": 2}})
    assert result == {"a": {"x": 11}, "b": {"y": 12}}
    
    # Test with set
    result = map_structure(lambda x: x * 2, {1, 2, 3})
    assert result == {2, 4, 6}
    
    # Test with mixed nested structure
    obj = {"a": [1, 2, {"x": 3}], "b": (4, 5)}
    result = map_structure(lambda x: x * 2, obj)
    expected = {"a": [2, 4, {"x": 6}], "b": (8, 10)}
    assert result == expected
    
    # Test with non-mappable registered type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom)
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3] * 2  # The whole list is passed to fn
    
    # Test with no_map_instance
    lst = [1, 2, 3]
    no_map_lst = no_map_instance(lst)
    result = map_structure(lambda x: x * 2, no_map_lst)
    assert result == [1, 2, 3] * 2  # The whole list is passed to fn
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    p = Point(1, 2)
    result = map_structure(lambda x: x + 10, p)
    assert result == Point(11, 12)
    
    # Test with scalar (non-collection)
    result = map_structure(lambda x: x * 3, 5)
    assert result == 15
    
    # Test with empty structures
    result = map_structure(lambda x: x, [])
    assert result == []
    
    result = map_structure(lambda x: x, {})
    assert result == {}
    
    result = map_structure(lambda x: x, ())
    assert result == ()
    
    # Test with function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ["1", "2", "3"]
    
    # Test with deeply nested structure
    obj = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = map_structure(lambda x: x - 1, obj)
    expected = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    assert result == expected


# LLM-generated content at query #19
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with dictionaries
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    )
    assert result == {"a": 4, "b": 6}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: f"{x}{y}",
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": ["15", "26"], "b": ("37", "48")}

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom_list, custom_list])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with single argument (should still work)
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    result = map_structure_zip(
        lambda a, b: Point(a.x + b.x, a.y + b.y),
        [Point(1, 2), Point(3, 4)]
    )
    assert result == Point(4, 6)

    # Test error case with set
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_map_structure_zip():
    # Test with simple flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with nested tuples
    result = map_structure_zip(lambda x, y: f"{x}{y}", [((1, 2), (3, 4)), (("a", "b"), ("c", "d"))])
    assert result == (("1a", "2b"), ("3c", "4d"))

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: x - y, [{"a": 10, "b": 20}, {"a": 3, "b": 5}])
    assert result == {"a": 7, "b": 15}

    # Test with nested dictionaries
    result = map_structure_zip(lambda x, y: x.upper() + y, [{"a": {"b": "hello"}}, {"a": {"b": "world"}}])
    assert result == {"a": {"b": "HELLOworld"}}

    # Test with mixed structures
    result = map_structure_zip(lambda x, y: x + str(y), [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}])
    assert result == {"a": ["15", "26"], "b": ("37", "48")}

    # Test with no_map_instance
    import collections
    Point = collections.namedtuple("Point", ["x", "y"])
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    result = map_structure_zip(lambda x, y: Point(x[0] + y[0], x[1] + y[1]), [no_map_instance(p1), no_map_instance(p2)])
    assert result.x == 4 and result.y == 6

    # Test with registered no_map type
    class SpecialList(list):
        pass
    
    register_no_map_class(SpecialList)
    special1 = SpecialList([1, 2, 3])
    special2 = SpecialList([4, 5, 6])
    result = map_structure_zip(lambda x, y: x + y, [special1, special2])
    assert isinstance(result, SpecialList)
    assert result == [1, 2, 3, 4, 5, 6]

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(lambda x, y, z: x + y + z, [[1, 2], [3, 4], [5, 6]])
    assert result == [9, 12]

    # Test error case with set
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot contain `set`" in str(e)

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test with empty dict
    result = map_structure_zip(lambda x, y: None, [{}, {}])
    assert result == {}


# LLM-generated content at query #21
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}
    
    # Test with nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1, "y": 2}, "b": 3})
    assert result == {"a": {"x": 11, "y": 12}, "b": 13}
    
    # Test with set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}
    
    # Test with mixed nested structure
    obj = {"a": [1, 2, {"x": 3}], "b": (4, 5)}
    result = map_structure(lambda x: x * 2, obj)
    assert result == {"a": [2, 4, {"x": 6}], "b": (8, 10)}
    
    # Test with non-container (leaf node)
    result = map_structure(lambda x: x + 1, 5)
    assert result == 6
    
    # Test with registered no-map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == [2, 4, 6]  # The list itself is passed to fn
    
    # Test with no-map instance
    normal_list = [1, 2, 3]
    no_map_list = no_map_instance(normal_list)
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [2, 4, 6]  # The list itself is passed to fn
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 2, point)
    assert result == Point(2, 4)
    
    # Test with empty structures
    result = map_structure(lambda x: x, [])
    assert result == []
    
    result = map_structure(lambda x: x, {})
    assert result == {}
    
    result = map_structure(lambda x: x, set())
    assert result == set()
    
    # Test that function is applied to leaves
    def add_one(x):
        return x + 1
    
    structure = [{"a": 1, "b": [2, 3]}, 4]
    expected = [{"a": 2, "b": [3, 4]}, 5]
    result = map_structure(add_one, structure)
    assert result == expected


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x + 10, {"a": 1, "b": 2})
    assert result == {"a": 11, "b": 12}
    
    # Test with nested dict
    result = map_structure(lambda x: len(x), {"a": "hi", "b": "hello"})
    assert result == {"a": 2, "b": 5}
    
    # Test with set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}
    
    # Test with mixed nested structure
    obj = {"a": [1, 2, 3], "b": (4, 5, 6), "c": {"d": 7, "e": 8}}
    result = map_structure(lambda x: x * 2, obj)
    expected = {"a": [2, 4, 6], "b": (8, 10, 12), "c": {"d": 14, "e": 16}}
    assert result == expected
    
    # Test with non-container leaf
    result = map_structure(lambda x: x + 5, 10)
    assert result == 15
    
    # Test with registered no-map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert isinstance(result, CustomList)
    assert result == [2, 4, 6]  # Should apply function to the whole object
    
    # Test with no-map instance
    normal_list = [1, 2, 3]
    no_map_list = no_map_instance(normal_list)
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [2, 4, 6]  # Should apply function to the whole object
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 2, point)
    assert result == Point(2, 4)
    assert isinstance(result, Point)
    
    # Test with empty structures
    result = map_structure(lambda x: x, [])
    assert result == []
    
    result = map_structure(lambda x: x, {})
    assert result == {}
    
    result = map_structure(lambda x: x, ())
    assert result == ()
    
    # Test with function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ["1", "2", "3"]
    
    # Test with deeply nested structure
    obj = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = map_structure(lambda x: x - 1, obj)
    expected = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    assert result == expected


# LLM-generated content at query #2
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    assert map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    
    # Test with nested list
    assert map_structure(lambda x: x + 1, [[1, 2], [3, 4]]) == [[2, 3], [4, 5]]
    
    # Test with tuple
    assert map_structure(lambda x: x.upper(), ("a", "b", "c")) == ("A", "B", "C")
    
    # Test with nested tuple
    assert map_structure(lambda x: x * 3, ((1, 2), (3, 4))) == ((3, 6), (9, 12))
    
    # Test with dict
    assert map_structure(lambda x: x * 2, {"a": 1, "b": 2}) == {"a": 2, "b": 4}
    
    # Test with nested dict
    assert map_structure(lambda x: x + 10, {"a": {"x": 1}, "b": {"y": 2}}) == {"a": {"x": 11}, "b": {"y": 12}}
    
    # Test with set
    result = map_structure(lambda x: x * 2, {1, 2, 3})
    assert isinstance(result, set)
    assert result == {2, 4, 6}
    
    # Test with mixed structure
    obj = {"a": [1, 2, 3], "b": (4, 5, 6), "c": {7, 8, 9}}
    expected = {"a": [2, 4, 6], "b": (8, 10, 12), "c": {14, 16, 18}}
    result = map_structure(lambda x: x * 2, obj)
    assert result["a"] == expected["a"]
    assert result["b"] == expected["b"]
    assert result["c"] == expected["c"]
    
    # Test with non-collection (leaf node)
    assert map_structure(lambda x: x * 2, 5) == 10
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    p = Point(1, 2)
    result = map_structure(lambda x: x * 2, p)
    assert result == Point(2, 4)
    assert isinstance(result, Point)
    
    # Test with registered no-map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert isinstance(result, CustomList)
    assert result == CustomList([1, 2, 3]) * 2
    
    # Test with no-map instance
    normal_list = [1, 2, 3]
    no_map_list = no_map_instance(normal_list)
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [1, 2, 3] * 2
    
    # Test with empty structures
    assert map_structure(lambda x: x * 2, []) == []
    assert map_structure(lambda x: x * 2, {}) == {}
    assert map_structure(lambda x: x * 2, ()) == ()
    
    # Test with function that changes type
    assert map_structure(str, [1, 2, 3]) == ["1", "2", "3"]
    
    # Test with deeply nested structure
    obj = {"a": [{"x": (1, 2)}, {"y": [3, 4]}]}
    expected = {"a": [{"x": (2, 4)}, {"y": [6, 8]}]}
    assert map_structure(lambda x: x * 2, obj) == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: f"{x}{y}", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert result == {"a": "13", "b": "24"}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": [6, 8], "b": (10, 12)}

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    result = map_structure_zip(
        lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y),
        [Point(1, 2), Point(3, 4)]
    )
    assert result == Point(4, 6)

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom_list, custom_list])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(
        lambda x, y, z: x + y + z,
        [[1, 2], [3, 4], [5, 6]]
    )
    assert result == [9, 12]

    # Test error case with set
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    result = map_structure_zip(lambda x, y: x + y, [{}, {}])
    assert result == {}


# LLM-generated content at query #4
#--------------------------

```python
def test_map_structure_zip():
    # Test with simple flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with nested tuples
    result = map_structure_zip(lambda x, y: f"{x}{y}", [((1, 2), (3, 4)), (("a", "b"), ("c", "d"))])
    assert result == (("1a", "2b"), ("3c", "4d"))

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: x - y, [{"a": 10, "b": 20}, {"a": 3, "b": 5}])
    assert result == {"a": 7, "b": 15}

    # Test with nested dictionaries
    result = map_structure_zip(lambda x, y: x.upper() + y, [{"a": {"b": "hello"}}, {"a": {"b": "world"}}])
    assert result == {"a": {"b": "HELLOworld"}}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": [6, 8], "b": (10, 12)}

    # Test with no_map_instance
    import sys
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: (x, y), [no_map_list, [4, 5, 6]])
    assert result == (no_map_list, [4, 5, 6])

    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom, [4, 5, 6]])
    assert isinstance(result, tuple)
    assert result[0] is custom
    assert result[1] == [4, 5, 6]

    # Test with single collection (should still work)
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(lambda x, y, z: x + y + z, [[1, 2], [3, 4], [5, 6]])
    assert result == [9, 12]

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test with sets (should raise ValueError)
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    result = map_structure_zip(lambda a, b: a + b, [Point(1, 2), Point(3, 4)])
    assert result == Point(4, 6)
    assert isinstance(result, Point)

    # Test with deep nesting
    complex_obj = [{"a": (1, [2, 3]), "b": {"c": 4}}, {"a": (5, [6, 7]), "b": {"c": 8}}]
    result = map_structure_zip(lambda x, y: x * y, complex_obj)
    expected = {"a": (5, [12, 21]), "b": {"c": 32}}
    assert result == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]

    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]

    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")

    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))

    # Test with dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}

    # Test with nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1}, "b": {"y": 2}})
    assert result == {"a": {"x": 11}, "b": {"y": 12}}

    # Test with set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}

    # Test with mixed nested structure
    obj = [{"a": (1, 2), "b": [3, 4]}, 5]
    result = map_structure(lambda x: x * 2, obj)
    assert result == [{"a": (2, 4), "b": [6, 8]}, 10]

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [1, 2, 3] * 2

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == [1, 2, 3] * 2

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 2, point)
    assert result == Point(2, 4)

    # Test with scalar (non-collection)
    result = map_structure(lambda x: x + 5, 10)
    assert result == 15

    # Test with empty structures
    result = map_structure(lambda x: x, [])
    assert result == []

    result = map_structure(lambda x: x, {})
    assert result == {}

    result = map_structure(lambda x: x, ())
    assert result == ()

    # Test with function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ["1", "2", "3"]

    # Test with deep nesting
    obj = {"a": [1, (2, {"b": 3})], "c": 4}
    result = map_structure(lambda x: x * 2, obj)
    assert result == {"a": [2, (4, {"b": 6})], "c": 8}


# LLM-generated content at query #6
#--------------------------

```python
def test_map_structure():
    # Test basic mapping on simple types
    assert map_structure(lambda x: x * 2, 5) == 10
    assert map_structure(lambda x: x.upper(), "hello") == "HELLO"
    
    # Test mapping on list
    assert map_structure(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
    
    # Test mapping on nested list
    assert map_structure(lambda x: x * 2, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]
    
    # Test mapping on tuple
    assert map_structure(lambda x: x.upper(), ("a", "b", "c")) == ("A", "B", "C")
    
    # Test mapping on nested tuple
    assert map_structure(lambda x: x + 10, ((1, 2), (3, 4))) == ((11, 12), (13, 14))
    
    # Test mapping on dict
    assert map_structure(lambda x: x * 2, {"a": 1, "b": 2}) == {"a": 2, "b": 4}
    
    # Test mapping on nested dict
    result = map_structure(lambda x: x.upper(), {"a": {"b": "hello"}, "c": "world"})
    assert result == {"a": {"b": "HELLO"}, "c": "WORLD"}
    
    # Test mapping on set
    result = map_structure(lambda x: x * 2, {1, 2, 3})
    assert isinstance(result, set)
    assert result == {2, 4, 6}
    
    # Test mixed nested structures
    obj = {"a": [1, 2, {"b": 3}], "c": (4, 5)}
    expected = {"a": [2, 4, {"b": 6}], "c": (8, 10)}
    assert map_structure(lambda x: x * 2, obj) == expected
    
    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    assert map_structure(lambda x: x * 2, no_map_list) == [2, 4, 6]
    
    # Test with registered no_map_class
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    assert map_structure(lambda x: x * 2, custom_list) == CustomList([1, 2, 3])
    
    # Test namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 2, point)
    assert isinstance(result, Point)
    assert result.x == 2
    assert result.y == 4
    
    # Test empty structures
    assert map_structure(lambda x: x * 2, []) == []
    assert map_structure(lambda x: x * 2, {}) == {}
    assert map_structure(lambda x: x * 2, set()) == set()
    
    # Test function that changes type
    assert map_structure(str, [1, 2, 3]) == ["1", "2", "3"]
    
    # Test with complex nested structure
    complex_obj = {
        "list": [1, {"a": 2, "b": 3}, (4, 5)],
        "tuple": (6, [7, 8]),
        "set": {9, 10}
    }
    result = map_structure(lambda x: x + 100, complex_obj)
    assert result["list"][0] == 101
    assert result["list"][1]["a"] == 102
    assert result["list"][1]["b"] == 103
    assert result["list"][2] == (104, 105)
    assert result["tuple"][0] == 106
    assert result["tuple"][1] == [107, 108]
    assert result["set"] == {109, 110}


# LLM-generated content at query #7
#--------------------------

```python
def test_map_structure():
    # Test basic mapping with simple list
    def add_one(x):
        return x + 1
    
    result = map_structure(add_one, [1, 2, 3])
    assert result == [2, 3, 4]
    
    # Test nested list
    nested = [[1, 2], [3, 4]]
    result = map_structure(add_one, nested)
    assert result == [[2, 3], [4, 5]]
    
    # Test dictionary
    def square(x):
        return x * x
    
    d = {'a': 2, 'b': 3}
    result = map_structure(square, d)
    assert result == {'a': 4, 'b': 9}
    
    # Test nested dictionary
    nested_dict = {'a': {'x': 1, 'y': 2}, 'b': 3}
    result = map_structure(add_one, nested_dict)
    assert result == {'a': {'x': 2, 'y': 3}, 'b': 4}
    
    # Test tuple
    t = (1, 2, 3)
    result = map_structure(add_one, t)
    assert result == (2, 3, 4)
    
    # Test namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    result = map_structure(add_one, p)
    assert result == Point(2, 3)
    
    # Test set
    s = {1, 2, 3}
    result = map_structure(add_one, s)
    assert result == {2, 3, 4}
    
    # Test mixed structure
    mixed = {'a': [1, 2, 3], 'b': (4, 5), 'c': {'x': 6, 'y': 7}}
    result = map_structure(lambda x: x * 2, mixed)
    assert result == {'a': [2, 4, 6], 'b': (8, 10), 'c': {'x': 12, 'y': 14}}
    
    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure(add_one, no_map_list)
    assert result == [1, 2, 3]  # Should apply fn to the whole list, not elements
    
    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom = CustomList([1, 2, 3])
    result = map_structure(add_one, custom)
    assert result == [1, 2, 3]  # Should apply fn to whole CustomList
    
    # Test string (non-collection)
    result = map_structure(str.upper, "hello")
    assert result == "HELLO"
    
    # Test integer (non-collection)
    result = map_structure(lambda x: x * 10, 5)
    assert result == 50
    
    # Test empty structures
    result = map_structure(add_one, [])
    assert result == []
    
    result = map_structure(add_one, {})
    assert result == {}
    
    # Test with function that changes type
    def to_str(x):
        return str(x)
    
    result = map_structure(to_str, [1, 2, 3])
    assert result == ['1', '2', '3']


# LLM-generated content at query #8
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    assert map_structure(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    
    # Test with nested list
    assert map_structure(lambda x: x + 1, [[1, 2], [3, 4]]) == [[2, 3], [4, 5]]
    
    # Test with tuple
    assert map_structure(str, (1, 2, 3)) == ('1', '2', '3')
    
    # Test with nested tuple
    assert map_structure(lambda x: x * 3, ((1, 2), (3, 4))) == ((3, 6), (9, 12))
    
    # Test with dict
    assert map_structure(lambda x: x.upper(), {'a': 'hello', 'b': 'world'}) == {'a': 'HELLO', 'b': 'WORLD'}
    
    # Test with nested dict
    assert map_structure(len, {'a': ['hello', 'world'], 'b': ['test']}) == {'a': 2, 'b': 1}
    
    # Test with set
    assert map_structure(lambda x: x ** 2, {1, 2, 3}) == {1, 4, 9}
    
    # Test with mixed structure
    obj = {'a': [1, 2, 3], 'b': (4, 5), 'c': {6, 7}}
    result = map_structure(lambda x: x * 2, obj)
    assert result == {'a': [2, 4, 6], 'b': (8, 10), 'c': {12, 14}}
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    result = map_structure(lambda x: x + 10, p)
    assert result == Point(11, 12)
    
    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    assert map_structure(lambda x: x * 2, no_map_list) == [1, 2, 3, 1, 2, 3]
    
    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    assert map_structure(lambda x: x * 2, custom_list) == [1, 2, 3, 1, 2, 3]
    
    # Test with primitive type
    assert map_structure(lambda x: x * 2, 5) == 10
    
    # Test with string
    assert map_structure(lambda x: x + '!', 'hello') == 'hello!'
    
    # Test with empty structures
    assert map_structure(lambda x: x, []) == []
    assert map_structure(lambda x: x, {}) == {}
    assert map_structure(lambda x: x, ()) == ()
    assert map_structure(lambda x: x, set()) == set()
    
    # Test with deeply nested structure
    obj = {'a': [{'b': (1, 2)}, {'c': {3, 4}}]}
    result = map_structure(lambda x: x + 1, obj)
    assert result == {'a': [{'b': (2, 3)}, {'c': {4, 5}}]}


# LLM-generated content at query #9
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with nested tuples
    result = map_structure_zip(lambda x, y: f"{x}{y}", [((1, 2), (3, 4)), (("a", "b"), ("c", "d"))])
    assert result == (("1a", "2b"), ("3c", "4d"))

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: x - y, [{"a": 10, "b": 20}, {"a": 3, "b": 5}])
    assert result == {"a": 7, "b": 15}

    # Test with nested dictionaries
    result = map_structure_zip(lambda x, y: x.upper() + y, [{"a": {"b": "hello"}}, {"a": {"b": "world"}}])
    assert result == {"a": {"b": "HELLOworld"}}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": [6, 8], "b": (10, 12)}

    # Test with three collections
    result = map_structure_zip(lambda x, y, z: x * y * z, [[1, 2], [3, 4], [5, 6]])
    assert result == [15, 48]

    # Test with no_map_instance
    import collections
    Point = collections.namedtuple("Point", ["x", "y"])
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    no_map_p1 = no_map_instance(p1)
    result = map_structure_zip(lambda x, y: (x, y), [no_map_p1, p2])
    assert result == (no_map_p1, p2)

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom1 = CustomList([1, 2, 3])
    custom2 = CustomList([4, 5, 6])
    result = map_structure_zip(lambda x, y: x + y, [custom1, custom2])
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] is custom1
    assert result[1] is custom2

    # Test with set should raise ValueError
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    result = map_structure_zip(lambda x, y: x + y, [{}, {}])
    assert result == {}


# LLM-generated content at query #10
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}
    
    # Test with nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1}, "b": {"y": 2}})
    assert result == {"a": {"x": 11}, "b": {"y": 12}}
    
    # Test with set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}
    
    # Test with mixed structure
    obj = {"a": [1, 2, 3], "b": (4, 5), "c": {6, 7}}
    result = map_structure(lambda x: x - 1, obj)
    assert result == {"a": [0, 1, 2], "b": (3, 4), "c": {5, 6}}
    
    # Test with non-collection (leaf node)
    result = map_structure(lambda x: x.upper(), "hello")
    assert result == "HELLO"
    
    # Test with registered no-map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert isinstance(result, CustomList)
    assert result == [2, 4, 6]
    
    # Test with no-map instance
    normal_list = [1, 2, 3]
    no_map_list = no_map_instance(normal_list)
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [2, 4, 6]
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 10, point)
    assert result == Point(10, 20)
    assert isinstance(result, Point)
    
    # Test with empty structures
    result = map_structure(lambda x: x, [])
    assert result == []
    
    result = map_structure(lambda x: x, {})
    assert result == {}
    
    result = map_structure(lambda x: x, ())
    assert result == ()
    
    # Test with identity function
    obj = {"a": [1, 2], "b": (3, 4)}
    result = map_structure(lambda x: x, obj)
    assert result == obj
    
    # Test with function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ["1", "2", "3"]


# LLM-generated content at query #11
#--------------------------

```python
def test_map_structure():
    # Test basic mapping with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]

    # Test nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]

    # Test tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")

    # Test nested tuple
    result = map_structure(lambda x: x * 3, (1, (2, 3), 4))
    assert result == (3, (6, 9), 12)

    # Test dict
    result = map_structure(lambda x: x + 10, {"a": 1, "b": 2})
    assert result == {"a": 11, "b": 12}

    # Test nested dict
    result = map_structure(lambda x: len(x), {"a": "hi", "b": {"c": "hello"}})
    assert result == {"a": 2, "b": {"c": 5}}

    # Test set
    result = map_structure(lambda x: x**2, {1, 2, 3})
    assert result == {1, 4, 9}

    # Test mixed structure
    obj = {"a": [1, 2, 3], "b": (4, 5), "c": {"d": 6}}
    result = map_structure(lambda x: x - 1, obj)
    assert result == {"a": [0, 1, 2], "b": (3, 4), "c": {"d": 5}}

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [1, 2, 3] * 2

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == [1, 2, 3] * 2

    # Test namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 2, point)
    assert result == Point(2, 4)

    # Test function applied to non-collection leaf
    result = map_structure(lambda x: x.upper(), "hello")
    assert result == "HELLO"

    # Test empty structures
    result = map_structure(lambda x: x, [])
    assert result == []

    result = map_structure(lambda x: x, {})
    assert result == {}

    result = map_structure(lambda x: x, ())
    assert result == ()

    # Test identity function
    obj = [1, {"a": 2}, (3, 4)]
    result = map_structure(lambda x: x, obj)
    assert result == obj


# LLM-generated content at query #12
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with nested tuples
    result = map_structure_zip(lambda x, y: f"{x}{y}", [((1, 2), (3, 4)), (("a", "b"), ("c", "d"))])
    assert result == (("1a", "2b"), ("3c", "4d"))

    # Test with dicts
    result = map_structure_zip(lambda x, y: x - y, [{"a": 10, "b": 20}, {"a": 3, "b": 5}])
    assert result == {"a": 7, "b": 15}

    # Test with nested dicts
    result = map_structure_zip(lambda x, y: x.upper() + y, [{"k1": {"k2": "hello"}}, {"k1": {"k2": "world"}}])
    assert result == {"k1": {"k2": "HELLOworld"}}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"list": [1, 2], "tuple": (3, 4)}, {"list": [5, 6], "tuple": (7, 8)}]
    )
    assert result == {"list": [6, 8], "tuple": (10, 12)}

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(lambda x, y, z: x + y + z, [[1, 2], [3, 4], [5, 6]])
    assert result == [9, 12]

    # Test with no_map_instance
    import sys
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom_list, custom_list])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with sets should raise ValueError
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    result = map_structure_zip(lambda a, b: a + b, [Point(1, 2), Point(3, 4)])
    assert result == Point(4, 6)
    assert isinstance(result, Point)

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    result = map_structure_zip(lambda x, y: x + y, [{}, {}])
    assert result == {}

    result = map_structure_zip(lambda x, y: x + y, [(), ()])
    assert result == ()


# LLM-generated content at query #13
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    objs = [[1, 2, 3], [4, 5, 6]]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == [5, 7, 9]

    # Test with nested lists
    objs = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == [[6, 8], [10, 12]]

    # Test with tuples
    objs = [(1, 2, 3), (4, 5, 6)]
    result = map_structure_zip(lambda x, y: x * y, objs)
    assert result == (4, 10, 18)

    # Test with nested tuples
    objs = [((1, 2), (3, 4)), ((5, 6), (7, 8))]
    result = map_structure_zip(lambda x, y: x - y, objs)
    assert result == ((-4, -4), (-4, -4))

    # Test with namedtuples
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    objs = [Point(1, 2), Point(3, 4)]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == Point(4, 6)

    # Test with dictionaries
    objs = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    result = map_structure_zip(lambda x, y: x * y, objs)
    assert result == {'a': 3, 'b': 8}

    # Test with nested dictionaries
    objs = [{'a': {'x': 1}, 'b': 2}, {'a': {'x': 3}, 'b': 4}]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == {'a': {'x': 4}, 'b': 6}

    # Test with mixed structures
    objs = [{'a': [1, 2], 'b': (3, 4)}, {'a': [5, 6], 'b': (7, 8)}]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == {'a': [6, 8], 'b': (10, 12)}

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    objs = [no_map_list, no_map_list]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    objs = [custom_list, custom_list]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with single object
    objs = [[1, 2, 3]]
    result = map_structure_zip(lambda x: x * 2, objs)
    assert result == [2, 4, 6]

    # Test with three objects
    objs = [[1, 2], [3, 4], [5, 6]]
    result = map_structure_zip(lambda x, y, z: x + y + z, objs)
    assert result == [9, 12]

    # Test with empty structures
    objs = [[], []]
    result = map_structure_zip(lambda x, y: x + y, objs)
    assert result == []

    # Test with sets should raise ValueError
    objs = [{1, 2}, {3, 4}]
    try:
        map_structure_zip(lambda x, y: x + y, objs)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with different dict types (OrderedDict-like)
    from collections import OrderedDict
    objs = [OrderedDict([('a', 1), ('b', 2)]), OrderedDict([('a', 3), ('b', 4)])]
    result = map_structure_zip(lambda x, y: x * y, objs)
    assert isinstance(result, OrderedDict)
    assert list(result.items()) == [('a', 3), ('b', 8)]


# LLM-generated content at query #14
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(lambda x: x.upper(), ("a", "b", "c"))
    assert result == ("A", "B", "C")
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 3, ((1, 2), (3, 4)))
    assert result == ((3, 6), (9, 12))
    
    # Test with dict
    result = map_structure(lambda x: x * 2, {"a": 1, "b": 2})
    assert result == {"a": 2, "b": 4}
    
    # Test with nested dict
    result = map_structure(lambda x: x + 10, {"a": {"x": 1}, "b": {"y": 2}})
    assert result == {"a": {"x": 11}, "b": {"y": 12}}
    
    # Test with set
    result = map_structure(lambda x: x * 2, {1, 2, 3})
    assert result == {2, 4, 6}
    
    # Test with mixed nested structure
    obj = {"a": [1, 2, 3], "b": (4, 5), "c": {"d": 6}}
    result = map_structure(lambda x: x + 1, obj)
    expected = {"a": [2, 3, 4], "b": (5, 6), "c": {"d": 7}}
    assert result == expected
    
    # Test with non-mappable registered type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == [2, 4, 6]  # The list itself is passed to fn
    
    # Test with no_map_instance
    normal_list = [1, 2, 3]
    no_map_list = no_map_instance(normal_list)
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [2, 4, 6]  # The entire list is passed to fn
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 2, point)
    assert result == Point(2, 4)
    
    # Test with scalar (non-collection)
    result = map_structure(lambda x: x * 2, 5)
    assert result == 10
    
    # Test with empty structures
    result = map_structure(lambda x: x * 2, [])
    assert result == []
    
    result = map_structure(lambda x: x * 2, {})
    assert result == {}
    
    # Test with function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ["1", "2", "3"]
    
    # Test with deep nesting
    obj = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = map_structure(lambda x: x - 1, obj)
    expected = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    assert result == expected


# LLM-generated content at query #15
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    result = map_structure_zip(lambda a, b: a.x + b.x, [(Point(1, 2), Point(3, 4)), (Point(5, 6), Point(7, 8))])
    assert result == Point(6, 10)

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: x - y, [{'a': 10, 'b': 20}, {'a': 5, 'b': 8}])
    assert result == {'a': 5, 'b': 12}

    # Test with mixed structures
    obj1 = {'a': [1, 2], 'b': (3, 4)}
    obj2 = {'a': [5, 6], 'b': (7, 8)}
    result = map_structure_zip(lambda x, y: x * y, [obj1, obj2])
    assert result == {'a': [5, 12], 'b': (21, 32)}

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: str(x) + str(y), [no_map_list, no_map_list])
    assert result == '[1, 2, 3][1, 2, 3]'

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom = CustomList([1, 2])
    result = map_structure_zip(lambda x, y: x + y, [custom, custom])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 1, 2]

    # Test with single object (should still work)
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three objects
    result = map_structure_zip(lambda x, y, z: x + y + z, [[1, 2], [3, 4], [5, 6]])
    assert result == [9, 12]

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test with scalars
    result = map_structure_zip(lambda x, y: x + y, [5, 10])
    assert result == 15

    # Test error with sets
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cannot contain `set`" in str(e)


# LLM-generated content at query #16
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with dicts
    result = map_structure_zip(lambda x, y: x - y, [{'a': 10, 'b': 20}, {'a': 3, 'b': 5}])
    assert result == {'a': 7, 'b': 15}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: f"{x}{y}",
        [{'a': [1, 2], 'b': (3, 4)}, {'a': [5, 6], 'b': (7, 8)}]
    )
    assert result == {'a': ['15', '26'], 'b': ('37', '48')}

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom, custom])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(
        lambda x, y, z: x + y + z,
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )
    assert result == [12, 15, 18]

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    result = map_structure_zip(
        lambda a, b: Point(a.x + b.x, a.y + b.y),
        [Point(1, 2), Point(3, 4)]
    )
    assert result == Point(4, 6)

    # Test that set raises ValueError
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with nested no_map instances
    no_map_inner = no_map_instance([10, 20])
    result = map_structure_zip(
        lambda x, y: x + y,
        [[no_map_inner, [1, 2]], [[30, 40], [3, 4]]]
    )
    assert result == [[10, 20, 30, 40], [4, 6]]


# LLM-generated content at query #17
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: x - y, [{'a': 10, 'b': 20}, {'a': 3, 'b': 8}])
    assert result == {'a': 7, 'b': 12}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: f"{x}{y}",
        [{'a': [1, 2], 'b': (3, 4)}, {'a': [5, 6], 'b': (7, 8)}]
    )
    assert result == {'a': ['15', '26'], 'b': ('37', '48')}

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom, custom])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(
        lambda x, y, z: x + y + z,
        [[1, 2], [3, 4], [5, 6]]
    )
    assert result == [9, 12]

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test with set should raise ValueError
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    result = map_structure_zip(
        lambda a, b: Point(a.x + b.x, a.y + b.y),
        [Point(1, 2), Point(3, 4)]
    )
    assert result == Point(4, 6)
    assert isinstance(result, Point)


# LLM-generated content at query #18
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with dictionaries
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    )
    assert result == {"a": 4, "b": 6}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: f"{x}{y}",
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": ["15", "26"], "b": ("37", "48")}

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class CustomList(list):
        pass

    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [custom_list, custom_list])
    assert isinstance(result, CustomList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(
        lambda x, y, z: x + y + z,
        [[1, 2], [3, 4], [5, 6]]
    )
    assert result == [9, 12]

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    # Test with sets should raise ValueError
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    result = map_structure_zip(
        lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y),
        [Point(1, 2), Point(3, 4)]
    )
    assert result == Point(4, 6)
    assert isinstance(result, Point)


# LLM-generated content at query #19
#--------------------------

```python
def test_map_structure():
    # Test with simple list
    result = map_structure(lambda x: x * 2, [1, 2, 3])
    assert result == [2, 4, 6]
    
    # Test with nested list
    result = map_structure(lambda x: x + 1, [[1, 2], [3, 4]])
    assert result == [[2, 3], [4, 5]]
    
    # Test with tuple
    result = map_structure(str, (1, 2, 3))
    assert result == ('1', '2', '3')
    
    # Test with nested tuple
    result = map_structure(lambda x: x * 2, ((1, 2), (3, 4)))
    assert result == ((2, 4), (6, 8))
    
    # Test with dict
    result = map_structure(lambda x: x.upper(), {'a': 'hello', 'b': 'world'})
    assert result == {'a': 'HELLO', 'b': 'WORLD'}
    
    # Test with nested dict
    result = map_structure(lambda x: x * 2, {'a': [1, 2], 'b': {'c': 3}})
    assert result == {'a': [2, 4], 'b': {'c': 6}}
    
    # Test with set
    result = map_structure(lambda x: x ** 2, {1, 2, 3})
    assert result == {1, 4, 9}
    
    # Test with mixed structure
    obj = {'a': [1, 2, 3], 'b': (4, 5), 'c': {'d': 6}}
    result = map_structure(lambda x: x + 10, obj)
    assert result == {'a': [11, 12, 13], 'b': (14, 15), 'c': {'d': 16}}
    
    # Test with non-mappable registered type
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert result == [2, 4, 6]
    
    # Test with no_map_instance
    normal_list = [1, 2, 3]
    no_map_list = no_map_instance(normal_list.copy())
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [2, 4, 6]
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])
    point = Point(1, 2)
    result = map_structure(lambda x: x * 3, point)
    assert result == Point(3, 6)
    
    # Test with scalar value
    result = map_structure(lambda x: x + 5, 10)
    assert result == 15
    
    # Test with string
    result = map_structure(lambda x: x + '!', 'hello')
    assert result == 'hello!'
    
    # Test with empty structures
    result = map_structure(lambda x: x, [])
    assert result == []
    
    result = map_structure(lambda x: x, {})
    assert result == {}
    
    result = map_structure(lambda x: x, ())
    assert result == ()
    
    # Test with function that changes type
    result = map_structure(str, [1, 2, 3])
    assert result == ['1', '2', '3']


# LLM-generated content at query #20
#--------------------------

```python
def test_map_structure():
    # Test basic mapping with simple types
    assert map_structure(lambda x: x * 2, 5) == 10
    assert map_structure(lambda x: x.upper(), "hello") == "HELLO"
    
    # Test with list
    assert map_structure(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
    
    # Test with nested list
    assert map_structure(lambda x: x * 2, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]
    
    # Test with tuple
    assert map_structure(lambda x: x.upper(), ("a", "b", "c")) == ("A", "B", "C")
    
    # Test with nested tuple
    assert map_structure(lambda x: x + 10, ((1, 2), (3, 4))) == ((11, 12), (13, 14))
    
    # Test with dict
    assert map_structure(lambda x: x * 2, {"a": 1, "b": 2}) == {"a": 2, "b": 4}
    
    # Test with nested dict
    assert map_structure(lambda x: x + "!", {"a": {"b": "hello"}, "c": "world"}) == {"a": {"b": "hello!"}, "c": "world!"}
    
    # Test with set
    assert map_structure(lambda x: x * 2, {1, 2, 3}) == {2, 4, 6}
    
    # Test with mixed nested structure
    obj = {"a": [1, 2, {"b": 3}], "c": (4, 5)}
    result = map_structure(lambda x: x * 2, obj)
    expected = {"a": [2, 4, {"b": 6}], "c": (8, 10)}
    assert result == expected
    
    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    p = Point(1, 2)
    result = map_structure(lambda x: x * 2, p)
    assert result == Point(2, 4)
    
    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure(lambda x: x * 2, no_map_list)
    assert result == [2, 4, 6]  # The entire list is passed to fn
    
    # Test with registered no_map_class
    class CustomList(list):
        pass
    
    register_no_map_class(CustomList)
    custom_list = CustomList([1, 2, 3])
    result = map_structure(lambda x: x * 2, custom_list)
    assert isinstance(result, CustomList)
    assert result == [2, 4, 6]  # The entire list is passed to fn
    
    # Test that non-container types are passed directly to fn
    assert map_structure(lambda x: type(x).__name__, 42) == "int"
    assert map_structure(lambda x: type(x).__name__, "test") == "str"
    
    # Test with empty structures
    assert map_structure(lambda x: x, []) == []
    assert map_structure(lambda x: x, {}) == {}
    assert map_structure(lambda x: x, set()) == set()
    assert map_structure(lambda x: x, ()) == ()
    
    # Test with complex nested structure
    complex_obj = {
        "list": [1, {"a": 2, "b": 3}, (4, 5)],
        "tuple": (6, [7, 8]),
        "set": {9, 10}
    }
    result = map_structure(lambda x: x + 100, complex_obj)
    expected = {
        "list": [101, {"a": 102, "b": 103}, (104, 105)],
        "tuple": (106, [107, 108]),
        "set": {109, 110}
    }
    assert result == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_map_structure_zip():
    # Test with flat lists
    result = map_structure_zip(lambda x, y: x + y, [[1, 2, 3], [4, 5, 6]])
    assert result == [5, 7, 9]

    # Test with nested lists
    result = map_structure_zip(lambda x, y: x * y, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert result == [[5, 12], [21, 32]]

    # Test with tuples
    result = map_structure_zip(lambda x, y, z: x + y + z, [(1, 2), (3, 4), (5, 6)])
    assert result == (9, 12)

    # Test with dictionaries
    result = map_structure_zip(lambda x, y: f"{x}{y}", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert result == {"a": "13", "b": "24"}

    # Test with mixed structures
    result = map_structure_zip(
        lambda x, y: x + y,
        [{"a": [1, 2], "b": (3, 4)}, {"a": [5, 6], "b": (7, 8)}]
    )
    assert result == {"a": [6, 8], "b": (10, 12)}

    # Test with no_map_instance
    no_map_list = no_map_instance([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [no_map_list, no_map_list])
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with registered no_map type
    class SpecialList(list):
        pass

    register_no_map_class(SpecialList)
    special = SpecialList([1, 2, 3])
    result = map_structure_zip(lambda x, y: x + y, [special, special])
    assert isinstance(result, SpecialList)
    assert result == [1, 2, 3, 1, 2, 3]

    # Test with namedtuple
    from collections import namedtuple
    Point = namedtuple("Point", ["x", "y"])
    result = map_structure_zip(
        lambda a, b: Point(a.x + b.x, a.y + b.y),
        [Point(1, 2), Point(3, 4)]
    )
    assert result == Point(4, 6)
    assert isinstance(result, Point)

    # Test with single collection
    result = map_structure_zip(lambda x: x * 2, [[1, 2, 3]])
    assert result == [2, 4, 6]

    # Test with three collections
    result = map_structure_zip(
        lambda x, y, z: x + y + z,
        [[1, 2], [3, 4], [5, 6]]
    )
    assert result == [9, 12]

    # Test error with set
    try:
        map_structure_zip(lambda x, y: x + y, [{1, 2}, {3, 4}])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test with empty structures
    result = map_structure_zip(lambda x, y: x + y, [[], []])
    assert result == []

    result = map_structure_zip(lambda x, y: x + y, [{}, {}])
    assert result == {}


