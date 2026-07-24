####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_maybe_parse_user_type_single_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_single_string():
    assert maybe_parse_user_type("int") == ["int"]

def test_maybe_parse_user_type_list_of_types():
    assert maybe_parse_user_type([int, str]) == (int, str)

def test_maybe_parse_user_type_nested_list():
    assert maybe_parse_user_type([[int], str]) == (int, str)

def test_maybe_parse_user_type_single_element_list():
    assert maybe_parse_user_type([int]) == (int,)

def test_maybe_parse_user_type_invalid_input_raises_error():
    try:
        maybe_parse_user_type(123)
    except TypeError:
        assert True
    else:
        raise AssertionError("Should have raised TypeError")
```


# LLM-generated content at query #2
#--------------------------

```python
def test_invariant_errors_empty_invariants():
    assert _invariant_errors(10, []) == []

def test_invariant_errors_all_valid():
    assert _invariant_errors(10, [lambda x: (True, None), lambda x: (True, "error")]) == []

def test_invariant_errors_one_invalid():
    assert _invariant_errors(10, [lambda x: (True, None), lambda x: (False, "error_msg")]) == ["error_msg"]

def test_invariant_errors_multiple_invalid():
    assert _invariant_errors(10, [
        lambda x: (False, "err1"),
        lambda x: (True, None),
        lambda x: (False, "err2")
    ]) == ["err1", "err2"]

def test_invariant_errors_with_complex_logic():
    assert _invariant_errors(
        5, 
        [
            lambda x: (x > 0, "must be positive"),
            lambda x: (x < 10, "must be less than 10"),
            lambda x: (x % 2 == 0, "must be even")
        ]
    ) == ["must be even"]
```


# LLM-generated content at query #3
#--------------------------

```python
def test_maybe_parse_user_type_single_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_single_string():
    assert maybe_parse_user_type("int") == ["int"]

def test_maybe_parse_user_type_list_of_types():
    assert maybe_parse_user_type([int, str]) == (int, str)

def test_maybe_parse_user_type_nested_list():
    assert maybe_parse_user_type([[int], str]) == (int, str)

def test_maybe_parse_user_type_single_tuple():
    assert maybe_parse_user_type((float,)) == (float,)

def test_maybe_parse_user_type_invalid_input_raises_error():
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)

def test_maybe_parse_user_type_preserved_type():
    # Assuming _preserved_iterable_types contains something like list or tuple
    # If the function logic depends on this global, we test the path
    # This test assumes the environment allows the execution of the provided snippet
    assert maybe_parse_user_type(list) == [list]
```


# LLM-generated content at query #4
#--------------------------

```python
def test_maybe_parse_user_type_single_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_single_string():
    assert maybe_parse_user_type("int") == ["int"]

def test_maybe_parse_user_type_list_of_types():
    assert maybe_parse_user_type([int, str]) == (int, str)

def test_maybe_parse_user_type_nested_list():
    assert maybe_parse_user_type([[int], str]) == (int, str)

def test_maybe_parse_user_type_tuple_of_strings():
    assert maybe_parse_user_type(("float", "bool")) == ("float", "bool")

def test_maybe_parse_tuple_of_types():
    assert maybe_parse_user_type((list, dict)) == (list, dict)

def test_maybe_parse_user_type_invalid_input_raises_error():
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)
```


# LLM-generated content at query #5
#--------------------------

```python
def test_maybe_parse_user_type_preserves_iterable_type():
    _preserved_iterable_types = (list, tuple)
    from collections.abc import Iterable
    
    # To make is_preserved True, t must be a subclass of _preserved_iterable_types and be a type.
    # Since list is a type and is a subclass of itself (and is in our dummy _preserved_iterable_types).
    t = list
    
    # The predicate is: is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    # We need to ensure the function is called in an environment where _preserved_iterable_types is defined.
    # Since I cannot modify the source, I assume the environment has the necessary globals.
    # For the purpose of this test case, we simulate the logic.
    
    result = maybe_parse_user_type(t)
    assert result == [list]
```


# LLM-generated content at query #6
#--------------------------

```python
def test_check_types_valid_input():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            super().__init__(msg)

    class DummySource:
        pass

    it = [1, "a", 2.5]
    expected_types = [int, str, float]
    source_class = DummySource
    
    _check_types(it, expected_types, source_class)

def test_check_types_invalid_input_raises_error():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            super().__init__(msg)

    class DummySource:
        pass

    it = [1, "a", True]  # True is bool, which is not in [int, str]
    expected_types = [int, str]
    source_class = DummySource
    
    # This should raise CheckedValueTypeError because True is not an instance of int or str 
    # (Note: bool is a subclass of int, but if we strictly check types or if we use a different type)
    # Let's use a type that is definitely not in the list
    it = [1, "a", [1, 2]] 
    
    # We expect an exception
    try:
        _check_types(it, expected_types, source_class, exception_type=CheckedValueTypeError)
    except CheckedValueTypeError as e:
        assert "Type DummySource can only be used with ('int', 'str'), not list" in str(e)
    else:
        raise AssertionError("CheckedValueTypeError was not raised")

def test_check_types_empty_expected_types_does_nothing():
    class CheckedValueTypeError(Exception):
        pass

    class DummySource:
        pass

    it = [1, "a", [1, 2]]
    expected_types = []
    source_class = DummySource
    
    _check_types(it, expected_types, source_class)

def test_check_types_empty_iterator_does_nothing():
    class CheckedValueTypeError(Exception):
        pass

    class DummySource:
        pass

    it = []
    expected_types = [int]
    source_class = DummySource
    
    _check_types(it, expected_types, source_class)
```


# LLM-generated content at query #7
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.5}
    mapping = IntToFloatMap(initial_data)
    assert isinstance(mapping, IntToFloatMap)
    assert mapping[1] == 1.0
    assert mapping[2] == 2.5
    assert len(mapping) == 2

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ and __value_type__ = int, float

    initial_data = {1: 1.0}
    # Using a specific size triggers the super().__new__ path
    mapping = IntToFloatMap(initial_data, size=10)
    assert isinstance(mapping, IntToFloatMap)
    assert mapping[1] == 1.0
    # Note: The actual size implementation depends on the underlying PMap/Buckets implementation

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    # 1: 1.5 fails because int(1.5) is 1, but we check against 2. 
    # Wait, the example says (int(v) == k). If v=1.5, int(v)=1. If k=1, it passes.
    # Let's use a failing one: k=1, v=2.0 -> int(2.0) is 2, 2 != 1.
    invalid_data = {1: 2.0}
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert len(e.error_codes) > 0

def test_checkedpmap_constructor_type_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    invalid_data = {"not_an_int": 1.0}
    try:
        IntToFloatMap(invalid_data)
    except CheckedKeyTypeError:
        assert True
```


# LLM-generated content at query #8
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.5}
    instance = IntToFloatMap(initial_data)
    assert isinstance(instance, IntToFloat
    assert dict(instance) == initial_data

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    instance = IntToFloatMap(initial_data, size=10)
    assert instance.size() == 10
    assert instance[1] == 1.5

def test_checkedpmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    invalid_data = {1: 1.5}
    # The constructor uses Evolver.set which checks invariants.
    # If invariant fails, persistent() raises InvariantException.
    # Note: In the provided code, the error is collected and raised in .persistent()
    # which is called at the end of __new__.
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert 'Invalid mapping' in str(e)

def test_checkedpmap_constructor_with_type_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    invalid_data = {"not_an_int": 1.5}
    try:
        IntToFloatMap(invalid_data)
    except CheckedKeyTypeError:
        assert True
```


# LLM-generated content at query #9
#--------------------------

```python
def test_store_invariants_success():
    class Base:
        def invariant_source(self):
            return True, (1,)

    class Derived(Base):
        pass

    target_dct = {}
    store_invariants(target_dct, (Derived,), "target_inv", "invariant_source")
    
    assert "target_inv" in target_dct
    assert len(target_dct["target_inv"]) == 2
    
    # Test execution of wrapped invariant from base
    # The wrapper returns result directly if first element is bool
    # In this case, invariant_source returns (True, (1,))
    # result[0] is True (bool), so it returns (True, (1,))
    result = target_dct["target_inv"][0]()
    assert result == (True, (1,))

def test_store_invariants_merging_logic():
    def multi_test(x):
        return [(True, "a"), (False, "b"), (True, "c")]
    
    class Base:
        def inv_src(self):
            return multi_test(None)

    target_dct = {}
    store_invariants(target_dct, (Base,), "target_inv", "inv_src")
    
    # wrap_invariant calls _merge_invariant_results if result[0] is not bool
    # Here multi_test returns a list, but the wrapper checks result[0]
    # result[0] is (True, "a"), which is a tuple, not a bool.
    # Therefore it triggers _merge_invariant_results.
    # _merge_invariant_results looks for False verdicts.
    # In our list, (False, "b") is the only one where verd is False.
    
    result = target_dct["target_inv"][0]()
    assert result == (False, ("b",))

def test_store_invariants_type_error():
    class Base:
        invariant_source = "not a callable"

    target_dct = {}
    try:
        store_invariants(target_dct, (Base,), "target_inv", "invariant_source")
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        raise AssertionError("TypeError not raised")

def test_store_invariants_inheritance_chain():
    class GrandParent:
        def inv_src(self):
            return True

    class Parent(GrandParent):
        pass

    class Child(Parent):
        def inv_src(self):
            return False, "child_data"

    target_dct = {}
    store_invariants(target_dct, (Child,), "target_inv", "inv_src")
    
    # Should contain both GrandParent's and Child's invariants
    assert len(target_dct["target_inv"]) == 2
    
    # Check Child's wrapped invariant
    # Child's result is (False, "child_data"). result[0] is False (bool).
    # Wrapper returns result as is.
    assert target_dct["target_inv"][1]() == (False, "child_data")
    
    # Check GrandParent's wrapped invariant
    # GrandParent's result is True. result[0] is True (bool).
    assert target_dct["target_inv"][0]() == True
```


# LLM-generated content at query #10
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.5}
    mapping = IntToFloatMap(initial_data)
    assert dict(mapping) == initial_data
    assert isinstance(mapping, IntToFlagMap)

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    # Using a specific size to trigger the super().__new__ path
    mapping = IntToFloatMap(initial_data, size=16)
    assert dict(mapping) == initial_data

def test_checkedpmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    # The constructor uses Evolver.set, which accumulates errors.
    # persistent() is called at the end of __new__, which raises InvariantException.
    invalid_data = {1: 1.5} # 1.5 cast to int is 1, but 1.5 != 1? No, int(1.5) is 1.
    # Let's use a clearer violation:
    class StrictIntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k == v, 'Mismatch')
    
    try:
        StrictIntMap({1: 2})
    except InvariantException as e:
        assert len(e.error_codes) > 0
```


# LLM-generated content at query #11
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "a"), (True, "b"), (True, "c")]
    assert _merge_invariant_results(result) == (True, ())

def test_merge_invariant_results_all_false():
    result = [(False, "a"), (False, "b"), (False, "c")]
    assert _merge_invariant_results(result) == (False, ("a", "b", "c"))

def test_merge_invariant_results_mixed():
    result = [(True, "a"), (False, "b"), (True, "c"), (False, "d")]
    assert _merge_invariant_results(result) == (False, ("b", "d"))

def test_merge_invariant_results_empty():
    result = []
    assert _merge_invariant_results(result) == (True, ())
```


# LLM-generated content at query #12
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    initial_data = {1: 1.0, 2: 2.0}
    instance = IntToFloatMap(initial_data)
    assert isinstance(instance, IntToFloatMap)
    assert instance[1] == 1.0
    assert instance[2] == 2.0
    assert len(instance) == 2

def test_checkedpmap_constructor_with_invalid_invariant_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ is int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    # The constructor uses Evolver.set which collects errors, 
    # and persistent() which raises InvariantException if errors exist.
    # Note: The provided __new__ implementation calls evolver.set and then returns evolver.persistent().
    with pytest.raises(InvariantException):
        IntToFloatMap({1: 1.5})

def test_checkedpmap_constructor_with_type_mismatch_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    with pytest.raises(CheckedKeyTypeError):
        IntToFloatMap({"not_an_int": 1.0})

def test_checkedpmap_constructor_with_explicit_size():
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    # Testing the branch: if size is not _UNDEFINED_CHECKED_PMAP_SIZE
    # We assume _UNDEFINED_CHECKED_PMAP_SIZE is a specific constant (e.g. -1)
    # Since we cannot see the value, we test the functionality via the logic provided.
    instance = SimpleMap(initial={1: 10}, size=10)
    assert instance[1] == 10
```


# LLM-generated content at query #13
#--------------------------

```python
def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (True, None)

    initial_data = {1: 1.5, 2: 2.5}
    pmap_instance = IntToFloatMap(initial_data)
    
    assert isinstance(pmap_instance, IntToFloatMap)
    assert pmap_instance[1] == 1.5
    assert pmap_instance[2] == 2.5
    assert len(pmap_instance) == 2

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (True, None)

    initial_data = {1: 1.5}
    # Assuming _UNDEFINED_CHECKED_PMAP_SIZE is a known constant from the context
    # and that providing a specific size triggers the super().__new__ path
    pmap_instance = IntToFloatMap(initial_data, size=10)
    
    assert isinstance(pmap_instance, IntToFloatMap)
    assert pmap_instance[1] == 1.5

def test_checked_pmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (v == k, 'Value must equal key')

    # The constructor uses Evolver.set which accumulates errors, 
    # and persistent() which raises InvariantException if errors exist.
    invalid_data = {1: 2.0}
    
    try:
        IntToFloatMap(invalid_data)
        raise AssertionError("Should have raised InvariantException")
    except InvariantException as e:
        assert 'Value must equal key' in e.error_codes
```


# LLM-generated content at query #14
#--------------------------

```python
def test_checked_pvector_constructor_from_iterable():
    class IntVector(CheckedPVector):
        __type__ = int
    
    vector = IntVector([1, 2, 3])
    assert vector.tolist() == [1, 2, 3]
    assert isinstance(vector, IntVector)

def test_checked_pvector_constructor_from_existing_pvector():
    class IntVector(CheckedPVector):
        __type__ = int
    
    base_vector = python_pvector([10, 20])
    vector = IntVector(base_vector)
    assert vector.tolist() == [10, 20]
    assert isinstance(vector, IntVector)

def test_checked_pvector_constructor_empty():
    class IntVector(CheckedPVector):
        __type__ = int
    
    vector = IntVector()
    assert vector.tolist() == []
    assert isinstance(vector, IntVector)

def test_checked_pvector_constructor_type_validation_fails():
    class IntVector(CheckedPVector):
        __type__ = int
    
    # The constructor uses .extend() via Evolver, which calls _check
    # _check calls _check_types which raises TypeError for invalid types
    import pytest
    with pytest.raises(TypeError):
        IntVector(["not an int"])
```


# LLM-generated content at query #15
#--------------------------

```python
def test_store_invariants_success():
    def invariant_1():
        return True, (1,)
    
    def invariant_2():
        return False, ("fail",)

    class Base:
        source = invariant_1

    class Derived(Base):
        pass

    target_dict = {}
    store_invariants(target_dict, (Derived,), "dest", "source")
    
    assert "dest" in target_dict
    assert len(target_dict["dest"]) == 2
    # Check that the wrapped functions return expected values
    # Note: wrap_invariant returns the result of invariant if first element is bool
    # or the merged result.
    assert target_dict["dest"][0]() == (True, (1,))
    assert target_dict["dest"][1]() == (True, (1,)) # Inherited from Base

def test_store_invariants_merging_logic():
    def invariant_multi():
        return (True, ("a",)), (False, ("b",))

    class Base:
        source = invariant_multi

    target_dict = {}
    store_inents = []
    
    # We need to test the merged result specifically
    # wrap_invariant returns result if result[0] is bool, else merged
    # In invariant_multi, result[0] is (True, ('a',)), which is not a bool.
    # Therefore it calls _merge_invariant_results
    
    store_invariants(target_dict, (Base,), "dest", "source")
    
    # result[0] is (True, ('a',)), which is a tuple, not a bool.
    # _merge_invariant_results iterates over result.
    # result is ((True, ('a',)), (False, ('b',)))
    # Loop 1: verd=True, dat=('a',). verdict stays True.
    # Loop 2: verd=False, dat=('b',). verdict becomes False, data=[('b',)]
    # Returns (False, (('b',),))
    
    result_val = target_dict["dest"][0]()
    assert result_val == (False, (('b',),))

def test_store_invariants_type_error():
    class Base:
        source = "not a callable"

    target_dict = {}
    
    try:
        store_invariants(target_dict, (Base,), "dest", "source")
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        raise AssertionError("TypeError not raised")

def test_store_invariants_inheritance_chain():
    def inv_a(): return True
    def inv_b(): return False

    class GrandParent:
        source = inv_a
    
    class Parent(GrandParent):
        source = inv_b

    class Child(Parent):
        pass

    target_dict = {}
    store_invariants(target_dict, (Child,), "dest", "source")
    
    # Should find inv_b (from Parent) and inv_a (from GrandParent)
    # order: Child's dict, then Parent's dict, then GrandParent's dict
    assert len(target_dict["dest"]) == 2
    # Check that the functions are wrapped and callable
    assert target_dict["dest"][0]() == (False,)
    assert target_dict["dest"][1]() == (True,)
```


# LLM-generated content at query #16
#--------------------------

```python
def test_maybe_parse_user_type_evaluates_is_type_and_not_is_iterable():
    class MySimpleType:
        pass
    
    # To satisfy the predicate at line 18:
    # is_type must be True (isinstance(t, type))
    # is_iterable must be False (not isinstance(t, Iterable))
    # t must not be a string (to avoid line 17)
    # t must not be a preserved iterable type (to avoid line 15)
    
    result = maybe_parse_user_type(MySimpleType)
    assert result == [MySimpleType]
```


# LLM-generated content at query #17
#--------------------------

```python
def test_checked_type_instantiation_fails_due_to_abstractmethod():
    from abc import ABC
    class ConcreteCheckedType(CheckedType, ABC):
        pass
    
    with Exception:
        instance = ConcreteCheckedType()

def test_checked_type_slots_is_empty():
    assert CheckedType.__slots__ == ()

def test_checked_type_is_instance_of_object():
    assert isinstance(CheckedType(), object)
```


# LLM-generated content at query #18
#--------------------------

```python
def test_store_types_single_source():
    dct = {}
    class Base:
        source = int
    bases = []
    _store_types(dct, bases, "dest", "source")
    assert dct["dest"] == [int]

def test_store_types_multiple_bases_inheritance():
    dct = {}
    class Base1:
        source = str
    class Base2:
        source = float
    bases = [Base1, Base2]
    _store_types(dct, bases, "dest", "source")
    assert dct["dest"] == [str, float]

def test_store_types_with_string_input():
    dct = {}
    class Base:
        source = "MyType"
    bases = []
    _store_types(dct, bases, "dest", "source")
    assert dct["dest"] == ["MyType"]

def test_store_types_with_iterable_input():
    dct = {}
    class Base:
        source = (int, str)
    bases = []
    _import_iterable = type('Iterable', (), {'__iter__': lambda self: iter([])})
    # Note: Assuming _preserved_iterable_types is defined such that tuple is handled
    # Since we can't define new functions, we rely on the existing logic for tuple
    _store_types(dct, bases, "dest", "source")
    assert dct["dest"] == (int, str)

def test_store_types_overwriting_existing_key():
    dct = {"dest": ["old"]}
    class Base:
        source = bool
    bases = []
    _store_types(dct, bases, "dest", "source")
    assert dct["dest"] == [bool]

def test_store_types_no_source_found():
    dct = {}
    class Base:
        other = int
    bases = [Base]
    _store_types(dct, bases, "dest", "source")
    assert "dest" not in dct
```


# LLM-generated content at query #19
#--------------------------

```python
def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.5}
    pmap_instance = IntToFloatMap(initial_data)
    assert dict(pmap_instance) == initial_data
    assert isinstance(pmap_instance, IntToASSERT_MAP_TYPE_PLACEHOLDER) # Note: In real scenario, check type equality

def test_checked_pmap_constructor_with_size_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    # Testing the branch where size is not _UNDEFINED_CHECKED_PMAP_SIZE
    # This assumes _UNDEFINED_CHECKED_PMAP_SIZE is a specific sentinel value
    pmap_instance = IntToFloatMap(initial_data, size=10)
    assert dict(pmap_instance) == initial_data

def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    # The constructor uses Evolver.set which checks invariants.
    # If invariant fails, persistent() raises InvariantException.
    # We test that providing invalid data during construction (via initial dict) 
    # triggers the error during the persistent() call inside __new__.
    try:
        IntToFloatMap({1: 1.5}) # 1.5 is not 1 (int(1.5) == 1, so this actually passes)
        # Let's use a clearly failing one:
        IntToFloatMap({1: 2.5}) # int(2.5) is 2, 2 != 1
    except InvariantException as e:
        assert len(e.error_codes) > 0
```


# LLM-generated content at query #20
#--------------------------

```python
def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (True, None)
    
    initial_data = {1: 1.5, 2: 2.25}
    pmap_instance = IntToFloatMap(initial_data)
    assert isinstance(pmap_instance, IntToFloatMap)
    assert dict(pmap_instance) == initial_data

def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ and __value_type__ = int, float
        __invariant__ = lambda k, v: (v == k, 'Invalid mapping')
    
    initial_data = {1: 2.0}
    try:
        IntToFloatMap(initial_data)
    except InvariantException as e:
        assert 'Invalid mapping' in str(e)

def test_checked_pmap_constructor_with_explicit_size():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (True, None)
    
    initial_data = {1: 1.5}
    pmap_instance = IntToFloatMap(initial_data, size=10)
    assert isinstance(pmap_instance, IntToFloatMap)
    assert dict(pmap_instance) == initial_data

def test_checked_pmap_constructor_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (True, None)
    
    pmap_instance = IntToFloatMap({})
    assert len(pmap_instance) == 0
```


# LLM-generated content at query #21
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (True, None)

    initial_data = {1: 1.5, 2: 2.5}
    result = IntToFloatMap(initial_data)
    assert isinstance(result, IntToFloatElseMap)
    assert result[1] == 1.5
    assert result[2] == 2.5
    assert len(result) == 2

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (True, None)

    initial_data = {1: 1.5}
    # Note: This test assumes _UNDEFINED_CHECKED_PMAP_SIZE is a specific sentinel value 
    # and that the superclass PMap handles the size parameter correctly.
    result = IntToFloatMap(initial_data, size=10)
    assert result[1] == 1.5
    assert len(result) == 1

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (v == k, 'Value must equal key')

    # The constructor uses the Evolver which accumulates errors and raises InvariantException on .persistent()
    # In the provided __new__, .persistent() is called at the end.
    try:
        IntToFloatMap({1: 2.0})
    except InvariantException as e:
        assert 'Value must equal key' in str(e)
```


# LLM-generated content at query #22
#--------------------------

```python
def test_restore_pickle_calls_create_with_correct_arguments():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return f"result_{data}_{_factory_fields}"

    data = {"key": "value"}
    expected_factory_fields = set()
    
    result = _restore_pickle(MockClass, data)
    
    assert result == f"result_{data}_{expected_factory_fields}"
```


# LLM-generated content at query #23
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (True, None)

    initial_data = {1: 1.5, 2: 2.5}
    mapping = IntToFloatMap(initial_data)
    assert isinstance(mapping, IntToFloatelseMap)
    assert mapping[1] == 1.5
    assert mapping[2] == 2.5
    assert len(mapping) == 2

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (True, None)

    # Note: size parameter triggers super().__new__ directly
    # This assumes _UNDEFINED_CHECKED_PMAP_SIZE is a specific sentinel
    mapping = IntToFloatMap({}, size=10)
    assert len(mapping) == 0

def test_checkedpmap_constructor_with_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (v == k, 'Value must equal key')

    # The constructor uses Evolver.set internally, which accumulates errors.
    # persistent() is called at the end of __new__.
    # Therefore, initializing with invalid data should raise InvariantException.
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Value must equal key' in str(e.error_codes)

def test_checkedpmap_constructor_type_error_on_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (True, None)

    try:
        IntToFloatMap({"not_an_int": 1.0})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        assert True
```


# LLM-generated content at query #24
#--------------------------

```python
def test_invariant_errors_predicate_evaluates_to_true():
    def invariant_fail(elem):
        return False

    def invariant_pass(elem):
        return True

    elem = None
    invariants = [invariant_fail, invariant_pass]
    
    # The predicate at line 1 is: (invariant(elem) for invariant in invariants)
    # To ensure the predicate (the generator expression) evaluates to True for the 
    # specific element being checked for failure, we verify the result contains 
    # the data from the failed invariant.
    
    # In the context of the list comprehension, the 'valid' variable 
    # evaluates the result of invariant(elem).
    # We assert that the logic correctly identifies the failure.
    
    assert _invariant_errors(elem, [lambda x: False]) == [None] # If we define data as the result
    # Re-evaluating based on the provided code structure:
    # The code returns [data for valid, data in (invariant(elem) for invariant in invariants) if not valid]
    # This implies 'invariant(elem)' must return a tuple (bool, data).
    
    def invariant_true_tuple(elem):
        return (True, "success")
    
    def invariant_false_tuple(elem):
        return (False, "failure")

    assert _invariant_errors(None, [invariant_false_tuple]) == ["failure"]
    assert _invariant_errors(None, [invariant_true_tuple]) == []
```


# LLM-generated content at query #25
#--------------------------

```python
def test_store_invariants_predicate_true():
    def mock_all_dicts(bases):
        return []
    
    # Injecting mock into global scope for the function to find
    import builtins
    builtins._all_dicts = mock_all_dicts
    
    # Setup data where the predicate (all(callable(invariant) for invariant in invariants)) is True
    # We need a source_name that exists and points to a callable
    source_key = "some_key"
    dest_key = "some_dest"
    
    # Mocking wrap_invariant to avoid dependency issues
    builtins.wrap_invariant = lambda x: x
    
    test_dict = {source_key: lambda: True}
    bases = []
    
    # This should execute without raising TypeError
    store_invariants(test_dict, bases, dest_key, source_key)
    
    # Verification that the logic reached the assignment
    assert dest_key in test_dict
    assert callable(test_dict[dest_key][0])
```


# LLM-generated content at query #26
#--------------------------

```python
def test_checkedpset_constructor_with_list():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pset = Positives([1, 2, 3])
    assert isinstance(pset, Positives)
    assert 1 in pset
    assert 2 in pset
    assert 3 in pset
    assert len(pset) == 3

def test_checkedpset_constructor_with_empty_iterable():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pset = Positives([])
    assert len(pset) == 0

def test_checkedpset_constructor_with_invalid_type_raises_error():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, 'string'])
    except Exception as e:
        assert isinstance(e, (TypeError, Exception))

def test_checkedpset_constructor_with_invariant_violation_raises_error():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([-1])
    except Exception as e:
        assert 'Negative' in str(e) or isinstance(e, Exception)

def test_checkedpset_constructor_with_pmap_direct_initialization():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    initial_map = PMap({1: True, 2: True})
    pset = Positives(initial_map)
    assert 1 in pset
    assert 2 in pset
    assert len(pset) == 2
```


# LLM-generated content at query #27
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict_and_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.5}
    # Note: 2.5 will fail the invariant (int(2.5) != 2)
    # We test a valid initial state
    valid_data = {1: 1.0, 2: 2.0}
    mapping = IntToFloatMap(valid_data)
    
    assert isinstance(mapping, IntToCacheMap)
    assert mapping[1] == 1.0
    assert mapping[2] == 2.0
    assert str(mapping) == "IntToFloatMap({1: 1.0, 2: 2.0})"

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    invalid_data = {1: 1.0, 2: 2.5}
    # The constructor uses Evolver.set which collects errors, 
    # then persistent() raises InvariantException
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert 'Invalid mapping' in str(e)

def test_checkedpmap_constructor_type_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    # Providing a string key where int is expected should trigger type check in Evolver.set
    try:
        IntToFloatMap({"not_an_int": 1.0})
    except CheckedKeyTypeError:
        pass
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    pmap_instance = IntToFloatMap(initial_data)
    assert dict(pmap_instance) == initial_data
    assert isinstance(pmap_instance, IntToFloatMap)

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    size_param = 10
    pmap_instance = IntToFloatMap(initial_data, size=size_param)
    assert dict(pmap_instance) == initial_data
    assert pmap_instance._size == size_param

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ and __value_type__ = int, float
        __checked_invariants__ = [(lambda k, v: (v > k, 'Value must be greater than key'),)]
    
    invalid_data = {5: 2.0}
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert len(e.error_codes) > 0

def test_checkedpmap_constructor_type_error_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    invalid_data = {"not_an_int": 1.5}
    try:
        IntToFloatMap(invalid_data)
    except CheckedKeyTypeError:
        assert True

def test_checkedpmap_constructor_type_error_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    invalid_data = {1: "not_a_float"}
    try:
        IntToFloatMap(invalid_data)
    except TypeError:
        assert True
```


# LLM-generated content at query #2
#--------------------------

```python
def test_serialize_raises_not_implemented_error():
    class ConcreteCheckedType(CheckedType):
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
        def serialize(self, format=None):
            return super().serialize(format)

    instance = ConcreteCheckedType()
    with pytest.raises(NotImplementedError):
        instance.serialize()

def test_serialize_with_argument_passing_to_subclass():
    class MockCheckedType(CheckedType):
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
        def serialize(self, format=None):
            return format

    instance = MockCheckedType()
    assert instance.serialize(format="json") == "json"
    assert instance.serialize(format="xml") == "xml"
    assert instance.serialize() is None
```


# LLM-generated content at query #3
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.5}
    result = IntToFloatMap(initial_data)
    assert isinstance(result, IntToFloatElseMap)
    assert dict(result) == initial_data

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    # Testing the branch where size is not _UNDEFINED_CHECKED_PMAP_SIZE
    # Note: Assuming _UNDEFINED_CHECKED_PMAP_SIZE is a specific constant
    result = IntToFloatMap(initial_data, size=10)
    assert dict(result) == initial_data

def test_checkedpmap_constructor_with_invariant_violation_during_evolution():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    # The constructor uses an Evolver and calls set() for each item in initial.
    # If an invariant fails, persistent() is called at the end of __new__.
    # We provide data that violates the invariant: 1 should map to 1.x, not 2.x
    invalid_data = {1: 2.5}
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert 'Invalid mapping' in str(e)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_serialize_basic_dict_conversion():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    pmap_instance = IntToFloatMap(initial_data)
    serialized_data = pmap_instance.serialize()
    
    assert serialized_data == {1: 1.5, 2: 2.25}
    assert isinstance(serialized_data, dict)

def test_serialize_empty_map():
    class EmptyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    pmap_instance = EmptyMap({})
    serialized_data = pmap_instance.serialize()
    
    assert serialized_data == {}
    assert isinstance(serialized_data, dict)

def test_serialize_with_custom_serializer_logic():
    # Note: This test assumes the existence of a custom serializer 
    # as implied by the 'serializer(format, k, v)' call in the source.
    # Since we cannot define new functions, we test the default behavior 
    # where __serializer__ is the default identity-like behavior for PMap.
    class StringMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = str
    
    pmap_instance = StringMap({"a": "1", "b": "2"})
    serialized_data = pmap_instance.serialize(format="string")
    
    assert serialized_data == {"a": "1", "b": "2"}
```


# LLM-generated content at query #5
#--------------------------

```python
def test_maybe_parse_user_type_single_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_single_string():
    assert maybe_parse_user_type("int") == ["int"]

def test_maybe_parse_user_type_list_of_types():
    assert maybe_parse_user_type([int, str]) == (int, str)

def test_maybe_parse_user_type_nested_list():
    assert maybe_parse_user_type([[int], str]) == (int, str)

def test_maybe_parse_user_type_string_in_list():
    assert maybe_parse_user_type(["int", "str"]) == ("int", "str")

def test_maybe_parse_user_type_invalid_input_raises_error():
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)

def test_maybe_parse_user_type_preserved_type():
    # Assuming _preserved_iterable_types contains list or tuple
    # This test depends on the global definition of _preserved_iterable_types
    assert maybe_parse_user_type(list) == [list]
```


# LLM-generated content at query #6
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.5}
    mapping = IntToFloatMap(initial_data)
    assert dict(mapping) == initial_data
    assert isinstance(mapping, IntToFloatMap)

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    # size is passed to super(CheckedPMap, cls).__new__
    # Assuming _UNDEFINED_CHECKED_PMAP_SIZE is a specific constant
    mapping = IntToFloatableMap(initial_data, size=10)
    assert dict(mapping) == initial_data

def test_checkedpmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    # This should raise InvariantException because 1.5 cast to int is 1, 
    # but the logic in Evolver.set checks the invariant during the loop.
    # In the provided __new__, the evolver.set is called for each item.
    # If 1.5 is passed, int(1.5) == 1. If 2.2 is passed, int(2.2) == 2.
    # To trigger an error, we need a value where int(v) != k.
    invalid_data = {1: 2.5} 
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert len(e.error_codes) > 0
```


# LLM-generated content at query #7
#--------------------------

```python
def test_checkedpset_constructor_with_list():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pset = Positives([1, 2, 3])
    assert list(pset) == [1, 2, 3]
    assert isinstance(pset, Positives)

def test_checkedpset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    with Exception:
        Positives([1, "string", 3])

def test_checkedpset_constructor_with_invalid_invariant():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    with Exception:
        Positives([-1])

def test_checkedpset_constructor_empty():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pset = Positives([])
    assert len(pset) == 0
```


# LLM-generated content at query #8
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    mapping = IntToFloatMap(initial_data)
    assert dict(mapping) == initial_data
    assert isinstance(mapping, IntToDummyPMap)

def test_checkedpmap_constructor_with_size_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    size = 10
    mapping = IntToFloatMap(initial_data, size=size)
    assert mapping[1] == 1.5
    assert mapping.size == size

def test_checkedpmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    invalid_data = {1: 1.5} # int(1.5) is 1, so this is valid
    valid_mapping = IntToFloatMap(invalid_data)
    assert valid_mapping[1] == 1.5

    invalid_data_error = {1: 2.5} # int(2.5) is 2, not 1
    try:
        IntToFloatMap(invalid_data_error)
    except InvariantException as e:
        assert len(e.error_codes) > 0

def test_checkedpmap_constructor_with_type_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    invalid_data = {"not_an_int": 1.5}
    try:
        IntToFloatMap(invalid_data)
    except CheckedKeyTypeError:
        assert True
```


# LLM-generated content at query #9
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.0}
    instance = IntToFloatMap(initial_data)
    assert isinstance(instance, IntToFloatMap)
    assert instance[1] == 1.0
    assert instance[2] == 2.0
    assert len(instance) == 2

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ and __value_type__ = int, float

    initial_data = {1: 1.0}
    size = 10
    instance = IntToFloatMap(initial_data, size=size)
    assert instance[1] == 1.0
    assert instance.size == size

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    invalid_data = {1: 2.5}
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert 'Invalid mapping' in str(e)
    else:
        raise AssertionError("InvariantException not raised")

def test_checkedpmap_constructor_type_error_on_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    invalid_data = {"not_an_int": 1.0}
    try:
        IntToFloatMap(invalid_data)
    except CheckedKeyTypeError:
        pass
    else:
        raise AssertionError("CheckedKeyTypeError not raised for invalid key type")

def test_checkedpmap_constructor_type_error_on_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    invalid_data = {1: "not_a_float"}
    try:
        IntToFloatMap(invalid_data)
    except TypeError:
        pass
    else:
        raise AssertionError("TypeError not raised for invalid value type")
```


# LLM-generated content at query #10
#--------------------------

```python
def test_check_types_valid_input():
    class CheckedValueTypeError(Exception):
        pass
    class SourceClass:
        pass
    it = [1, 2, 3]
    expected_types = [int, float]
    source_class = SourceClass
    _check_types(it, expected_types, source_class)

def test_check_types_invalid_input_raises_error():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.msg = msg
            super().__init__(msg)
    class SourceClass:
        pass
    it = [1, "string", 3]
    expected_types = [int]
    source_class = SourceClass
    
    import pytest
    with pytest.raises(CheckedValueTypeError) as excinfo:
        _check_types(it, expected_types, source_class)
    assert "Type SourceClass can only be used with ('int'), not str" in str(excinfo.value)

def test_check_types_empty_expected_types_does_nothing():
    class CheckedValueTypeError(Exception):
        pass
    class SourceClass:
        pass
    it = ["a", "b", "c"]
    expected_types = []
    source_class = SourceClass
    _check_types(it, expected_types, source_class)

def test_check_types_empty_iterator_does_nothing():
    class CheckedValueTypeError(Exception):
        pass
    class SourceClass:
        pass
    it = []
    expected_types = [int]
    source_class = SourceClass
    _check_types(it, expected_types, source_class)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_checked_type_slots_empty():
    assert CheckedType.__slots__ == ()

def test_checked_type_is_abstract_and_raises_not_implemented_on_create():
    class ConcreteCheckedType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
        def serialize(self, format=None):
            return {}

    instance = ConcreteCheckedType.create({})
    
    try:
        CheckedType.create({})
    except NotImplementedError:
        assert True
    else:
        assert False

def test_checked_type_raises_not_implemented_on_serialize():
    class ConcreteCheckedType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
        def serialize(self, format=None):
            raise NotImplementedError()

    instance = ConcreteCheckedType.create({})
    
    try:
        instance.serialize()
    except NotImplementedError:
        assert True
    else:
        assert False
```


# LLM-generated content at query #12
#--------------------------

```python
def test_store_invariants_success():
    class Base:
        def invariant_source(self):
            return True, (1,)

    class Derived(Base):
        pass

    target_dct = {}
    store_invariants(target_dct, (Base,), "dest", "invariant_source")
    
    assert "dest" in target_dct
    assert len(target_dct["dest"]) == 1
    
    # Test execution of the wrapped invariant
    wrapped_inv = target_dct["dest"][0]
    result_verdict, result_data = wrapped_inv()
    assert result_verdict is True
    assert result_data == (1,)

def test_store_invariants_inheritance_and_merging():
    def inv1():
        return True, (1,)
    
    def inv2():
        return False, (2,)

    class Base:
        invariant_source = inv1

    class Sub(Base):
        invariant_source = inv2

    target_dct = {}
    # We pass Sub as the main dict and Base as base
    # The function looks at target_dct, then all_dicts(bases)
    store_insets_params = (target_dct, (Base,), "dest", "invariant_source")
    store_invariants(*store_insets_params)

    # It should find inv2 from Sub (via target_dct) and inv1 from Base (via bases)
    # Note: store_invariants logic: [dct] + _all_dicts(bases)
    # If we use target_dct as the dict, it checks target_dct[source_name]
    # Then it checks Base.__dict__
    
    # Let's redefine for a clearer inheritance test
    class Parent:
        invariant_source = inv1

    class Child(Parent):
        pass

    target_dct = {}
    store_invariants(target_dct, (Child,), "dest", "invariant_source")
    
    # target_dct has nothing, so it looks at Child and Parent
    # Child does not have invariant_source, Parent does.
    assert len(target_dct["dest"]) == 1
    assert target_dct["dest"][0]() == (True, (1,))

def test_store_invariants_type_error():
    class Base:
        invariant_source = "not_callable"

    target_dct = {}
    try:
        store_invariants(target_dct, (Base,), "dest", "invariant_source")
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        raise AssertionError("TypeError not raised")

def test_store_invariants_multiple_invariants_merging():
    def inv_pass():
        return True, (1,)
    
    def inv_fail():
        return False, (2,)

    class Base:
        invariant_source = inv_pass

    class Child(Base):
        invariant_source = inv_fail

    target_dct = {}
    # The implementation: for ns in [dct] + list(_all_dicts(bases)):
    # If dct is Child, and bases is (Base,)
    # ns will be Child, then Base.
    # If Child has inv_fail and Base has inv_pass
    # The wrapped invariants are (wrap(inv_fail), wrap(inv_pass))
    
    # We need to simulate a scenario where the result of the wrapped function 
    # is checked. The function returns a tuple of wrapped functions.
    # The requirement is to test store_invariants itself.
    
    store_invariants(target_dct, (Base,), "dest", "invariant_source")
    
    # Check that the wrapped function, when called, works
    # Since we only have one source name, it finds the one in Base
    wrapped = target_dct["dest"][0]
    assert wrapped() == (True, (1,))
```


# LLM-generated content at query #13
#--------------------------

```python
def test_checked_type_create_returns_instance_if_already_correct_type():
    class DummyType:
        pass
    instance = DummyType()
    result = _checked_type_create(DummyType, instance)
    assert result is instance

def test_checked_type_create_wraps_data_in_new_instance_for_simple_type():
    class SimpleType:
        def __init__(self, data):
            self.data = data
    source_data = [1, 2, 3]
    result = _checked_type_create(SimpleType, source_data)
    assert isinstance(result, SimpleType)
    assert result.data == [1, 2, 3]

class CheckedType:
    @classmethod
    def create(cls, data, ignore_extra=False):
        return f"processed_{data}"

class WrapperType(CheckedType):
    _checked_types = [CheckedType]
    def __init__(self, items):
        self.items = items

class NestedType(CheckedType):
    _checked_types = [WrapperType]
    def __init__(self, items):
        self.items = items

def test_checked_type_create_uses_checked_type_recursion():
    source_data = [["a", "b"], ["c"]]
    result = _checked_type_create(NestedType, source_data)
    assert isinstance(result, NestedType)
    assert isinstance(result.items[0], WrapperType)
    assert isinstance(result.items[0].items[0], str)
    assert result.items[0].items[0] == "a"

def test_checked_type_create_skips_checked_type_creation_if_data_already_matches_type_in_list():
    class InnerChecked(CheckedType):
        pass
    
    class OuterChecked(CheckedType):
        _checked_types = [InnerChecked]
        def __init__(self, data):
            self.data = data
            
    inner_instance = InnerChecked()
    source_data = [inner_instance, "raw_string"]
    
    result = _checked_type_create(OuterChecked, source_data)
    assert isinstance(result, OuterChecked)
    assert result.data[0] is inner_instance
    assert result.data[1] == "processed_raw_string"
```


# LLM-generated content at query #14
#--------------------------

```python
def test_maybe_parse_user_type_single_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_single_string():
    assert maybe_parse_user_type("int") == ["int"]

def test_maybe_parse_user_type_list_of_types():
    assert maybe_parse_user_type([int, str]) == (int, str)

def test_maybe_parse_user_type_nested_list_of_types():
    assert maybe_parse_user_type([[int], str]) == (int, str)

def test_maybe_parse_user_type_tuple_of_strings():
    assert maybe_parse_user_type(("int", "str")) == ("int", "str)

def test_maybe_parse_user_type_error_on_invalid_type():
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)

def test_maybe_parse_user_type_preserved_type():
    # Assuming _preserved_iterable_types contains list or similar
    # This test depends on the global definition of _preserved_iterable_types
    # If list is in _preserved_iterable_types, it should return [list] instead of flattening
    assert maybe_parse_user_type(list) == [list]
```


# LLM-generated content at query #15
#--------------------------

```python
def test_store_invariants_success():
    class Base:
        def inv_base(self):
            return True, (1,)

    class Derived(Base):
        def inv_derived(self):
            return False, (2,)

    target_dict = {}
    store_invariants(target_dict, (Derived,), "dest", "source")
    
    # Check if destination exists
    assert "dest" in target_dict
    # Check if it contains two wrapped invariants (from Derived and Base)
    assert len(target_dict["dest"]) == 2
    
    # Test first invariant (from Derived)
    # wrap_invariant returns the result directly if first element is bool
    # inv_derived returns (False, (2,)) -> first element is False (bool)
    # Wait, wrap_invariant logic: if isinstance(result[0], bool): return result
    # inv_derived returns (False, (2,)). result[0] is False. Returns (False, (2,))
    res_derived = target_dict["dest"][0]()
    assert res_derived == (False, (2,))

    # Test second invariant (from Base)
    # inv_base returns (True, (1,)). result[0] is True. Returns (True, (1,))
    res_base = target_dict["dest"][1]()
    assert res_base == (True, (1,))

def test_store_invariants_type_error():
    class Base:
        not_callable = "I am a string"

    target_dict = {}
    # This should raise TypeError because 'not_callable' is not callable
    # Note: store_invariants looks for source_name in bases. 
    # If we point source_name to a non-callable, it should fail.
    try:
        store_invariants(target_dict, (Base,), "dest", "not_callable")
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        assert False, "Expected TypeError"

def test_store_invariants_merging_logic():
    # Testing the interaction with wrap_invariant's merging capability
    # We need an invariant that returns a non-bool first element to trigger merge
    # Example: returns ( [True, False], (data,) ) -> This is not what the code does.
    # The code checks: if isinstance(result[0], bool): return result
    # To trigger _merge_invariant_results, result[0] must NOT be a bool.
    
    class ComplexInvariant:
        def complex_inv(self):
            # result[0] is a list, not a bool. This triggers merge.
            # result = [(True, 'a'), (False, 'b')]
            return [(True, 'a'), (False, 'b')]

    class Target:
        pass

    target_dict = {}
    # We need to mock the source_name to point to a function that returns a list of tuples
    # But store_invariants looks for source_name in the class dict.
    # Let's create a class where source_name is the complex_inv.
    
    class SourceClass:
        def source_func(self):
            return [(True, 'a'), (False, 'b')]

    store_invariants(target_dict, (SourceClass,), "dest", "source_func")
    
    # The wrapped function will call source_func(). 
    # result = [(True, 'a'), (False, 'b')]
    # result[0] is (True, 'a'), which is a tuple, not a bool.
    # So it calls _merge_invariant_results([(True, 'a'), (False, 'b')])
    # _merge_invariant_results iterates:
    # 1. (True, 'a'): verd=True, dat='a'. Nothing happens.
    # 2. (False, 'a'): verd=False, dat='b'. verdict=False, data=['b']
    # Returns (False, ('b',))
    
    res = target_dict["dest"][0]()
    assert res == (False, ('b',))
```


# LLM-generated content at query #16
#--------------------------

```python
def test_maybe_parse_user_type_evaluates_true_at_line_18():
    class IntType:
        pass
    
    # To trigger line 18:
    # is_preserved must be False (not in _preserved_iterable_types)
    # is_string must be False (not a str)
    # is_type must be True (is a type)
    # is_iterable must be False (not an Iterable)
    
    # We assume _preserved_iterable_types does not contain IntType
    # and IntType is not an Iterable.
    
    # Mocking/Setting up the environment for the test context
    import collections.abc
    global _preserved_iterable_types
    _preserved_iterable_types = (list, tuple, set) 
    
    result = maybe_parse_user_type(IntType)
    assert result == [IntType]
```


# LLM-generated content at query #17
#--------------------------

```python
def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.5}
    mapping = IntToFloatMap(initial_data)
    assert dict(mapping) == initial_data
    assert isinstance(mapping, IntToFloatMap)

def test_checked_pmap_constructor_with_size_and_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    mapping = IntToFloatMap(initial_data, size=10)
    assert dict(mapping) == initial_data
    # Note: In the provided implementation, the size is passed to super().__new__
    # which is handled by the underlying PMap implementation.

def test_checked_pmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ and __value_type__ = int, float
        __invariants__ = [(lambda k, v: (v > k, "Value must be greater than key"),)]

    class InvariantException(Exception): pass

    # The constructor uses Evolver.set which triggers invariant checks.
    # If the invariant fails during the loop in __new__, persistent() raises InvariantException.
    # We use a subclass that specifically triggers the error via the initial dict.
    class InvariantErrorMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __checked_invariants__ = [(lambda k, v: (v > k, 'Invalid mapping'),)]

    import pytest
    with pytest.raises(Exception): # InvariantException
        InvariantErrorMap({1: 0})

def test_checked_pmap_constructor_type_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    import pytest
    with pytest.raises(Exception): # CheckedKeyTypeError or TypeError from _check_types
        IntToFloatMap({ "not_an_int": 1.5 })
```


# LLM-generated content at query #18
#--------------------------

```python
def test_checkedpset_constructor_with_list():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    initial_elements = [1, 2, 3]
    pset = Positives(initial_elements)
    assert len(pset) == 3
    assert 1 in pset
    assert 2 in pset
    assert 3 in pset

def test_checkedpset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    initial_elements = [1, "not_an_int"]
    try:
        Positives(initial_elements)
    except (TypeError, Exception):
        assert True
    else:
        assert False

def test_checkedpset_constructor_with_invariant_failure():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    initial_elements = [1, -5]
    try:
        Positives(initial_elements)
    except Exception as e:
        assert "Negative" in str(e)
    else:
        assert False

def test_checkedpset_constructor_with_empty_input():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pset = Positives([])
    assert len(pset) == 0
```


# LLM-generated content at query #19
#--------------------------

```python
def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.0}
    pmap_instance = IntToFloatMap(initial_data)
    assert dict(pmap_instance) == initial_data
    assert isinstance(pmap_instance, IntToFloatMap)

def test_checked_pmap_constructor_with_size_parameter():
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int

    initial_data = {1: 10}
    size = 10
    pmap_instance = SimpleMap(initial_data, size=size)
    assert dict(pmap_instance) == initial_data
    # Note: Internal size check depends on the underlying PMap implementation behavior

def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    # 1: 1.5 should fail because int(1.5) is 1, but the logic in Evolver.set 
    # checks the invariant during the loop of the constructor.
    # If the constructor uses evolver.set for each item in initial, 
    # an invalid pair will trigger InvariantException.
    invalid_data = {1: 1.5} 
    # Since 1.5 is not int(1.5) == 1? Wait, int(1.5) is 1. 
    # Let's use 1: 2.5 -> int(2.5) is 2, 2 != 1.
    invalid_data = {1: 2.5}
    
    try:
        IntToFloatMap(invalid_data)
        raise AssertionError("Should have raised InvariantException")
    except InvariantException as e:
        assert len(e.error_codes) > 0

def test_checked_pmap_constructor_type_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    invalid_types = {"not_an_int": 1.0}
    try:
        IntToFloatMap(invalid_types)
        raise AssertionError("Should have raised CheckedKeyTypeError")
    except CheckedKeyTypeError:
        pass
```


# LLM-generated content at query #20
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "a"), (True, "b"), (True, "c")]
    assert _merge_invariant_results(result) == (True, ())

def test_merge_invariant_results_all_false():
    result = [(False, "a"), (False, "b"), (False, "c")]
    assert _merge_invariant_results(result) == (False, ("a", "b", "c"))

def test_merge_invariant_results_mixed_values():
    result = [(True, "a"), (False, "b"), (True, "c"), (False, "d")]
    assert _merge_invariant_results(result) == (False, ("b", "d"))

def test_merge_invariant_results_empty_input():
    result = []
    assert _merge_invariant_results(result) == (True, ())

def test_merge_invariant_results_single_true():
    result = [(True, "only_one")]
    assert _merge_invariant_results(result) == (True, ())

def test_merge_invariant_results_single_false():
    result = [(False, "only_one")]
    assert _merge_invariant_results(result) == (False, ("only_one",))
```


# LLM-generated content at query #21
#--------------------------

```python
def test_checked_pvector_constructor_with_iterable():
    class IntVector(CheckedPVector):
        __type__ = int
    
    v = IntVector([1, 2, 3])
    assert v.tolist() == [1, 2, 3]
    assert isinstance(v, IntVector)

def test_checked_pvector_constructor_with_empty():
    class EmptyVector(CheckedPVector):
        __type__ = int
    
    v = EmptyVector()
    assert v.tolist() == []
    assert len(v) == 0

def test_checked_pvector_constructor_with_existing_pvector():
    class FloatVector(CheckedPVector):
        __type__ = float
    
    base_vector = python_pvector([1.1, 2.2])
    v = FloatVector(base_vector)
    assert v.tolist() == [1.1, 2.2]
    assert isinstance(v, FloatVector)

def test_checked_pvector_constructor_type_validation():
    class IntVector(CheckedPVector):
        __type__ = int
    
    with pytest.raises(TypeError):
        IntVector(["not", "an", "int"])
```


# LLM-generated content at query #22
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    initial_data = {1: 10, 2: 20}
    mapping = IntMap(initial_data)
    assert mapping[1] == 10
    assert mapping[2] == 20
    assert len(mapping) == 2

def test_checkedpmap_constructor_with_size_parameter():
    class IntMap(CheckedPMap):
        __key_type__ and __value_type__ = int, int
    
    # Note: This assumes the underlying PMap implementation handles size
    # and that _UNDEFINED_CHECKED_PMAP_SIZE is a known constant.
    initial_data = {1: 10}
    mapping = IntMap(initial_data, size=10)
    assert mapping[1] == 10

def test_checkedpmap_constructor_invariant_validation_success():
    class ValidatingMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > k, 'Value must be greater than key')
    
    mapping = ValidatingMap({1: 5, 2: 10})
    assert mapping[1] == 5
    assert mapping[2] == 10

def test_checkedpmap_constructor_invariant_validation_failure():
    class ValidatingMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > k, 'Value must be greater than key')
    
    # The Evolver.set method is called during __new__ loop. 
    # If invariant fails, persistent() raises InvariantException.
    try:
        ValidatingMap({1: 0})
    except InvariantException as e:
        assert len(e.error_codes) > 0

def test_checkedpmap_constructor_type_error_on_invalid_key():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int

    try:
        IntMap({"string_key": 10})
    except CheckedKeyTypeError:
        assert True
```


# LLM-generated content at query #23
#--------------------------

```python
def test_invariant_errors_empty_invariants():
    assert _invariant_errors(10, []) == []

def test_invariant_errors_all_valid():
    assert _invariant_errors(10, [lambda x: (True, None), lambda x: (True, "error")]) == []

def test_invariant_errors_single_invalid():
    assert _invariant_errors(10, [lambda x: (False, "error_1"), lambda x: (True, "error_2")]) == ["error_1"]

def test_invariant_errors_multiple_invalid():
    assert _invariant_errors(5, [
        lambda x: (False, "error_A"),
        lambda x: (True, "skip"),
        lambda x: (False, "error_B"),
        lambda x: (True, None)
    ]) == ["error_A", "error_B"]

def test_invariant_errors_with_complex_data():
    assert _invariant_errors({"a": 1}, [
        lambda x: (x.get("a") > 0, "must be positive"),
        lambda x: (isinstance(x, dict), "not a dict")
    ]) == []

def test_invariant_errors_with_complex_data_failure():
    assert _invariant_errors({"a": -1}, [
        lambda x: (x.get("a") > 0, "must be positive"),
        lambda x: (isinstance(x, dict), "not a dict")
    ]) == ["must be positive"]
```


# LLM-generated content at query #24
#--------------------------

```python
def test_checked_type_create_returns_source_data_when_is_instance():
    class MockType:
        pass

    source_data = MockType()
    result = _checked_type_create(MockType, source_data)
    assert result is source_data
```


# LLM-generated content at query #25
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    initial_data = {1: 10, 2: 20}
    result = IntMap(initial_data)
    assert isinstance(result, IntMap)
    assert dict(result) == initial_data

def test_checkedpmap_constructor_with_size():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    initial_data = {1: 10}
    # Note: The implementation uses super(CheckedPMap, cls).__new__(cls, size, initial)
    # which assumes the underlying PMap implementation supports a size argument.
    result = IntMap(initial_data, size=10)
    assert isinstance(result, IntMap)
    assert dict(result) == initial_data

def test_checkedpmap_constructor_with_invariants_success():
    class ValidatedMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > k, "Value must be greater than key")
    
    result = ValidatedMap({1: 5, 2: 10})
    assert dict(result) == {1: 5, 2: 10}

def test_checkedpmap_constructor_with_invariants_failure():
    class ValidatedMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > k, "Value must be greater than key")
    
    # The constructor uses an Evolver. If an invariant fails during 'set', 
    # the error is collected and raised in .persistent().
    # Since __new__ calls evolver.persistent(), it should raise InvariantException.
    try:
        ValidatedMap({1: 0})
    except Exception as e:
        assert "Value must be greater than key" in str(e)
```


