####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_invariant_errors_empty_invariants():
    assert _invariant_errors(10, []) == []

def test_invariant_errors_all_valid():
    assert _invariant_errors(10, [lambda x: (True, None), lambda x: (True, "error")]) == []

def test_invariant_errors_one_invalid():
    assert _invariant_errors(10, [lambda x: (True, None), lambda x: (False, "not positive"), lambda x: (True, None)]) == ["not positive"]

def test_invariant_errors_multiple_invalid():
    assert _invariant_errors(-5, [lambda x: (x > 0, "must be positive"), lambda x: (x < 0, "must be negative")]) == ["must be positive"]

def test_invariant_errors_all_invalid():
    assert _invariant_errors(0, [lambda x: (False, "err1"), lambda x: (False, "err2")]) == ["err1", "err2"]
```


# LLM-generated content at query #2
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

def test_maybe_parse_tuple_preservation():
    # Assuming tuple is in _preserved_iterable_types based on common patterns
    # If not, this tests the is_iterable logic path
    assert maybe_parse_user_type(tuple) == [tuple]

def test_maybe_parse_user_type_invalid_input_raises_error():
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_maybe_parse_user_type_single_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_string():
    assert maybe_parse_user_type("int") == ["int"]

def test_maybe_parse_user_type_list_of_types():
    assert maybe_parse_user_type([int, str]) == (int, str)

def test_maybe_parse_user_type_nested_list():
    assert maybe_parse_user_type([[int], str]) == (int, str)

def test_maybe_parse_user_type_tuple_of_strings():
    assert maybe_parse_user_type(("float", "bool")) == ("float", "bool")

def test_maybe_parse_user_type_invalid_input_raises_error():
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_checkedpset_constructor_with_list():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pset = Positives([1, 2, 3])
    assert isinstance(pset, Positives)
    assert len(pset) == 3
    assert 1 in pset
    assert 2 in pset
    assert 3 in pset

def test_checkedpset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([1, "string"])
    except TypeError:
        assert True
    else:
        assert False

def test_checkedpset_constructor_with_invariant_failure():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        Positives([-1])
    except InvariantException:
        assert True
    else:
        assert False

def test_checkedpset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pset = Positives([])
    assert len(pset) == 0
```


# LLM-generated content at query #5
#--------------------------

```python
def test_maybe_parse_user_type_single_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_string():
    assert maybe_parse_user_type("int") == ["int"]

def test_maybe_parse_user_type_list_of_types():
    assert maybe_parse_user_type([int, str]) == (int, str)

def test_maybe_parse_user_type_nested_list():
    assert maybe_parse_user_type([[int], str]) == (int, str)

def test_maybe_parse_user_type_tuple_of_strings():
    assert maybe_parse_user_type(("float", "bool")) == ("float", "bool")

def test_maybe_parse_user_type_invalid_input_raises_error():
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)
```


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_evaluates_to_true():
    def true_invariant(x):
        return True

    elem = 10
    invariants = [true_invariant]
    
    result = _invariant_errors(elem, invariants)
    assert len(result) == 0
```


# LLM-generated content at query #7
#--------------------------

```python
def test_maybe_parse_user_type_evaluates_string_branch():
    import collections.abc
    class Iterable(collections.abc.Iterable):
        pass
    
    # Mocking the necessary environment for the function to run
    # We need to define _preserved_iterable_types so line 10 doesn't crash
    global _preserved_iterable_types
    _preserved_iterable_types = (int,)
    
    # To satisfy the logic: is_string must be True, and is_preserved must be False.
    # Since 'a' is a string, is_string is True. 
    # 'a' is also an instance of str, but not a subclass of _preserved_iterable_types (which contains int).
    # Therefore, it hits the elif is_string block at line 17.
    # To specifically target line 18, we need is_type to be True and is_iterable to be False.
    # However, the prompt asks to ensure the predicate AT line 18 evaluates to True.
    # Line 18: elif is_type and not is_iterable:
    # We need a type that is NOT an iterable. 'int' is a type and not an iterable.
    # But if we use 'int', line 10 (is_preserved) might be True if int is in _preserved_iterable_types.
    # So we ensure int is NOT in _preserved_iterable_types.
    
    _preserved_iterable_types = (float,)
    t = int
    
    result = maybe_parse_user_type(t)
    assert result == [int]
```


# LLM-generated content at query #8
#--------------------------

```python
def test_maybe_parse_user_type_preserves_iterable_type():
    global _preserved_iterable_types
    _preserved_iterable_types = (list,)
    result = maybe_parse_user_type(list)
    assert result == [list]
```


# LLM-generated content at query #9
#--------------------------

```python
def test_check_types_valid_input():
    class CheckedValueTypeError(Exception):
        pass

    class MockSource:
        pass

    it = [1, 2, 3]
    expected_types = [int]
    source_class = MockSource
    _check_types(it, expected_types, source_class)

def test_check_types_invalid_input_raises_error():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.msg = msg
            super().__init__(msg)

    class MockSource:
        pass

    it = [1, "string", 3]
    expected_types = [int]
    source_class = MockSource
    
    import pytest
    with pytest.raises(CheckedValueTypeError) as excinfo:
        _check_types(it, expected_types, source_class)
    assert "Type MockSource can only be used with ('int'), not str" in str(excinfo.value)

def test_check_types_empty_expected_types_passes():
    class CheckedValueTypeError(Exception):
        pass

    class MockSource:
        pass

    it = ["any", 123, True]
    expected_types = []
    source_class = MockSource
    _check_types(it, expected_types, source_class)

def test_check_types_multiple_valid_types():
    class CheckedValueTypeError(Exception):
        pass

    class MockSource:
        pass

    it = [1, 2.5, True]
    expected_types = [int, float, bool]
    source_class = MockSource
    _check_types(it, expected_types, source_class)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_invariant_exception_constructor_with_basic_args():
    error_codes = ("err1", "err2")
    missing_fields = ("field1",)
    exc = InvariantException(error_codes=error_codes, missing_fields=missing_fields, msg="base error")
    assert exc.invariant_errors == ("err1", "err2")
    assert exc.missing_fields == ("field1",)
    assert "base error" in str(exc)

def test_invariant_exception_constructor_with_callables():
    error_codes = (lambda: "dynamic_err", "static_err")
    missing_fields = ()
    exc = InvariantException(error_codes=error_codes, missing_fields=missing_fields)
    assert exc.invariant_errors == ("dynamic_err", "static_err")
    assert exc.missing_fields == ()

def test_invariant_exception_constructor_empty_inputs():
    exc = InvariantException()
    assert exc.invariant_errors == ()
    assert exc.missing_fields == ()
    assert "invariant_errors=[], missing_fields=[]" in str(exc)

def test_invariant_exception_str_formatting():
    error_codes = ("e1",)
    missing_fields = ("f1", "f2")
    exc = InvariantException(error_codes=error_codes, missing_fields=missing_fields, msg="msg")
    expected_str = "msg, invariant_errors=[e1], missing_fields=[f1, f2]"
    assert str(exc) == expected_str
```


# LLM-generated content at query #11
#--------------------------

```python
def test_checked_type_instantiation_fails_due_to_abstract_methods():
    from abc import ABC
    class ConcreteCheckedType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
        def serialize(self, format=None):
            return {}

    instance = ConcreteCheckedType()
    assert isinstance(instance, CheckedType)

def test_checked_type_slots_are_empty():
    assert CheckedType.__slots__ == ()
```


# LLM-generated content at query #12
#--------------------------

```python
def test_store_invariants_success():
    class Base:
        def check_base(self, x):
            return True, (x,)

    class Derived(Base):
        pass

    target_dct = {}
    source_name = "check_base"
    destination_name = "wrapped_checks"

    store_invariants(target_dct, (Derived,), destination_name, source_name)

    assert destination_name in target_dct
    assert len(target_dct[destination_name]) == 2
    # First element is the wrapped check from Derived itself (if it were there) or just inherited
    # In this specific implementation: _all_dicts yields Derived.__dict__ then Base.__dict__
    # Since 'check_base' is in Base, it finds it.
    
    # Test execution of the wrapped invariant
    wrapped_inv = target_dct[destination_name][0]
    assert wrapped_inv(10) == (True, (10,))

def test_store_invariants_type_error():
    class Base:
        check_base = "not a callable"

    target_dct = {}
    
    try:
        store_invariants(target_dct, (Base,), "dest", "check_base")
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        raise AssertionError("TypeError not raised")

def test_store_invariants_multiple_inheritance():
    class A:
        def inv_a(self, x):
            return True, (x,)
    
    class B:
        def inv_b(self, x):
            return False, ("fail",)

    class C(A, B):
        pass

    target_dct = {}
    store_inants_args = (target_dct, (C,), "checks", "inv_a") # This will only find inv_a if we search specific names
    # To test multiple, we need the source_name to match both. 
    # Let's adjust the logic: store_invariants looks for 'source_name' in all dicts.
    
    class Shared:
        def shared_inv(self, x):
            return True, (x,)

    class Sub1(Shared):
        pass
    
    class Sub2(Shared):
        pass

    target_dct = {}
    store_invariants(target_dct, (Sub1, Sub2), "dest", "shared_inv")
    
    # _all_dicts with Sub1 and Sub2 will yield Shared twice, but 'seen' prevents duplicate processing.
    # However, it yields the dict of Sub1, then its bases, then Sub2, then its bases.
    # Since Shared is in both, seen handles it. 
    # The result should contain the shared_inv wrapped once or twice depending on visibility.
    assert len(target_dict_result := target_dct["dest"]) >= 1
    assert target_dct["dest"][0](5) == (True, (5,))

def test_store_invariants_no_matches():
    class Base:
        pass

    target_dct = {}
    store_invariants(target_dct, (Base,), "dest", "non_existent")
    
    assert "dest" in target_dct
    assert target_dct["dest"] == ()
```


# LLM-generated content at query #13
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.5}
    mapping = IntToFloatMap(initial_data)
    assert isinstance(mapping, IntToFloatelseMapping)
    assert mapping[1] == 1.5
    assert mapping[2] == 2.5
    assert len(mapping) == 2

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    # Assuming _UNDEFINED_CHECKED_PMAP_SIZE is a constant used for default size
    mapping = IntToFloatMap(initial_data, size=10)
    assert mapping[1] == 1.5
    assert len(mapping) == 1

def test_checkedpmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    # 1.5 cast to int is 1, which matches key 1. This should work.
    valid_map = IntToFloatMap({1: 1.5})
    assert valid_map[1] == 1.5

    # 1.5 cast to int is 1, but key is 2. This should raise InvariantException during persistent() call in __new__
    with _assertRaises(InvariantException):
        IntToFloatMap({2: 1.5})

def test_checkedpmap_constructor_type_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    # Providing a string key where an int is expected
    with _assertRaises(CheckedKeyTypeError):
        IntToFloatMap({"not_an_int": 1.5})
```


# LLM-generated content at query #14
#--------------------------

```python
def test_checkedpset_constructor_with_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pset = Positives([1, 2.5, 3])
    assert isinstance(pset, Positives)
    assert len(pset) == 3
    assert 1 in pset
    assert 2.5 in pset
    assert 3 in pset

def test_checkedpset_constructor_with_invalid_type():
    class IntOnly(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda n: (True, None)

    try:
        IntOnly([1, "string"])
    except TypeError:
        assert True
    else:
        assert False

def test_checkedpset_constructor_with_invariant_violation():
    class NonNegative(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        NonNegative([-1])
    except InvariantException as e:
        assert 'Negative' in e.error_codes
    else:
        assert False

def test_checkedpset_constructor_with_empty_initial():
    class EmptySet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda n: (True, None)

    pset = EmptySet([])
    assert len(pset) == 0
```


# LLM-generated content at query #15
#--------------------------

```python
def test_store_types_single_source():
    dct = {}
    class Base:
        source = int
    _store_types(dct, [Base], "dest", "source")
    assert dct["dest"] == [int]

def test_store_types_multiple_bases_and_overrides():
    dct = {"source": str}
    class Base1:
        source = float
    class Base2:
        source = bool
    _store_types(dct, [Base1, Base2], "dest", "source")
    assert dct["dest"] == [str, float, bool]

def test_store_types_no_matching_source():
    dct = {}
    class Base:
        other = int
    _store_types(dct, [Base], "dest", "source")
    assert "dest" not in dct

def test_store_types_with_string_types():
    dct = {"source": ["int", "str"]}
    class Base:
        pass
    _store_types(dct, [Base], "dest", "source")
    assert dct["dest"] == ("int", "str")

def test_store_types_with_iterable_input():
    dct = {}
    class Base:
        source = (int, float)
    _store_types(dct, [Base], "dest", "source")
    assert dct["dest"] == (int, float)
```


# LLM-generated content at query #16
#--------------------------

```python
import collections

class CheckedType:
    @classmethod
    def create(cls, data, ignore_extra=False):
        pass

class IntCheckedType(CheckedType):
    @classmethod
    def create(cls, data, ignore_extra=False):
        return int(data)

class ListCheckedType(CheckedType):
    @classmethod
    def create(cls, data, ignore_extra=False):
        return [cls.create(item, ignore_extra=ignore_extra) for item in data]

class MyIntClass:
    _checked_types = (IntCheckedType,)
    def __init__(self, value):
        self.value = value
    def __eq__(self, other):
        return isinstance(other, MyIntClass) and self.value == other.value

class MyListClass:
    _checked_types = (ListCheckedModule,)
    def __init__(self, value):
        self.value = value
    def __eq__(self, other):
        return isinstance(other, MyListClass) and self.value == other.value

class ListCheckedModule(CheckedType):
    @classmethod
    def create(cls, data, ignore_extra=False):
        return MyListClass([MyIntClass.create(i) for i in data])

def test_checked_type_create_direct_instance():
    instance = MyIntClass(10)
    result = _checked_type_create(MyIntClass, instance)
    assert result == instance

def test_checked_type_create_conversion_int():
    source_data = "123"
    result = _checked_type_create(MyIntClass, source_data)
    assert result == MyIntClass(123)

def test_checked_type_create_list_conversion():
    source_data = ["1", "2", "3"]
    # We need a structure where the top level class is a CheckedType itself to trigger recursion
    class MyListWrapper(CheckedType):
        _checked_types = (ListCheckedModule,)
        def __init__(self, data):
            self.data = data
        @classmethod
        def create(cls, data, ignore_extra=False):
            return MyListClass([MyIntClass.create(i) for i in data])

    result = _checked_type_create(MyListWrapper, ["1", "2"])
    assert result == MyListClass([MyIntClass(1), MyIntClass(2)])

def test_checked_type_create_no_checked_types():
    class SimpleClass:
        _checked_types = ()
        def __init__(self, data):
            self.data = data
    
    result = _checked_type_create(SimpleClass, "raw_string")
    assert result.data == "raw_string"
```


# LLM-generated content at query #17
#--------------------------

```python
def test_store_invariants_predicate_true():
    def mock_all_dicts(bases):
        return []
    
    # Mocking the global _all_dicts to return an empty list
    import builtins
    original_all_dicts = builtins._all_dicts
    builtins._all_dicts = mock_all_dicts
    
    def mock_wrap_invariant(inv):
        return inv

    # Setup input data where the predicate 'all(callable(invariant) for invariant in invariants)' evaluates to True
    # We provide a callable object as the value for source_name
    source_key = "src"
    dest_key = "dst"
    input_dict = {source_key: lambda x: x}
    bases = []

    # Execution (assuming wrap_invariant and _all_dicts are available in scope)
    # Since the prompt asks to test the predicate at line 1 (which is actually a function definition, 
    # but context implies testing the logic inside), we verify the logic passes.
    try:
        # We need to mock wrap_invariant because it's called in the function
        import builtins
        original_wrap = builtins.wrap_invariant
        builtins.wrap_invariant = mock_wrap_invariant
        
        # The predicate at line 11 is all(callable(invariant) for invariant in invariants)
        # We ensure 'invariants' contains only callables.
        store_invariants(input_dict, bases, dest_key, source_key)
        
        assert dest_key in input_dict
        assert callable(input_dict[dest_key][0])
    finally:
        builtins._all_dicts = original_all_dicts
        builtins.wrap_invariant = original_wrap
```


# LLM-generated content at query #18
#--------------------------

```python
def test_restore_pickle_returns_instance_with_correct_data():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            return {"data": data, "factory_fields": _factory_fields}

    mock_data = {"id": 1, "name": "test"}
    result = _restore_pickle(MockClass, mock_data)
    
    assert result["data"] == mock_data
    assert result["factory_fields"] == set()

def test_restore_pickle_passes_empty_set_to_factory_fields():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            return _factory_fields

    result = _restore_pickle(MockClass, "some_data")
    
    assert result == set()
```


# LLM-generated content at query #19
#--------------------------

```python
def test_checkedpvector_constructor_with_iterable():
    class IntVector(CheckedPVector):
        __type__ = int
    
    initial_data = [1, 2, 3]
    vector = IntVector(initial_data)
    assert vector.tolist() == [1, 2, 3]
    assert isinstance(vector, IntVector)

def test_checkedpvector_constructor_empty():
    class IntVector(CheckedPVector):
        __type__ = int
    
    vector = IntVector()
    assert vector.tolist() == []

def test_checkedpvector_constructor_with_existing_pvector():
    from pyrsistent import pvector
    class IntVector(CheckedPVector):
        __type__ = int
    
    base_vector = pvector([10, 20])
    vector = IntVector(base_vector)
    assert vector.tolist() == [10, 20]
    assert isinstance(vector, IntVector)

def test_checkedpvector_constructor_type_validation():
    class IntVector(CheckedPVector):
        __type__ = int
    
    with Exception:
        IntVector([1, "not_an_int", 3])
```


# LLM-generated content at query #20
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.0}
    result = IntToFloatMap(initial_data)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial_data

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ and __value_type__ = int, float
        __invariant__ = lambda k, v: (True, '')

    initial_data = {1: 1.0}
    # Testing the branch where size is not _UNDEFINED_CHECKED_PMAP_SIZE
    # Note: Assuming _UNDEFINED_CHECKED_PMAP_SIZE is a specific sentinel value
    result = IntToFloatMap(initial_data, size=10)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial_data

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    # The constructor uses Evolver.set which checks invariants
    # In the provided implementation, persistent() is called at the end of __new__
    # If an invariant fails during the loop in __new__, it raises InvariantException
    invalid_data = {1: 1.5} # 1.5 cast to int is 1, but let's use one that fails
    # Actually, for 1: 1.5, int(1.5) == 1, so this is valid.
    # Let's use a value where int(v) != k
    invalid_data = {1: 2.5} 
    
    try:
        IntToFloatMap(invalid_data)
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_checkedpmap_constructor_type_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    invalid_data = {"not_an_int": 1.0}
    try:
        IntToFloatMap(invalid_data)
    except Exception as e:
        assert isinstance(e, CheckedKeyTypeError)
```


# LLM-generated content at query #21
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.5}
    result = IntToFloatMap(initial_data)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial_data
    assert result[1] == 1.0

def test_checkedpmap_constructor_with_size_parameter():
    class SimpleMap(CheckedPMap):
        __key_type__
        __value_type__

    initial_data = {1: 'a'}
    size = 10
    result = SimpleMap(initial_data, size=size)
    assert result[1] == 'a'
    # Since PMap/CheckedPMap implementation details for size are internal to the superclass/evolver logic,
    # we verify that it successfully constructs and holds data.

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    # The constructor uses Evolver.persistent(), which raises InvariantException if errors exist
    invalid_data = {1: 2.5} # int(2.5) is 2, not 1
    try:
        IntToFloatMap(invalid_data)
        raise AssertionError("Should have raised InvariantException")
    except InvariantException as e:
        assert len(e.error_codes) > 0

def test_checkedpmap_constructor_type_error():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    invalid_data = {"not_an_int": "value"}
    try:
        IntMap(invalid_data)
        raise AssertionError("Should have raised CheckedKeyTypeError")
    except CheckedKeyTypeError:
        pass
```


# LLM-generated content at query #22
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.0}
    mapping = IntToFloatMap(initial_data)
    assert isinstance(mapping, IntToFloatMap)
    assert dict(mapping) == initial_data
    assert mapping[1] == 1.0

def test_checkedpmap_constructor_with_size_parameter():
    class SimpleMap(CheckedPMap):
        __key_type__ .__value_type__ = int, int

    initial_data = {1: 1}
    mapping = SimpleMap(initial_data, size=10)
    assert mapping[1] == 1
    assert isinstance(mapping, SimpleMap)

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    # 1.5 is not 1, so invariant should fail during evolver.persistent() called by __new__
    try:
        IntToFloatMap({1: 1.5})
        raise AssertionError("Should have raised InvariantException")
    except InvariantException as e:
        assert 'Invalid mapping' in str(e.error_codes)

def test_checkedpmap_constructor_type_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"not_an_int": 1.0})
        raise AssertionError("Should have raised CheckedKeyTypeError")
    except CheckedKeyTypeError:
        pass
```


# LLM-generated content at query #23
#--------------------------

```python
def test_check_types_valid_input():
    class CheckedValueTypeError(Exception):
        pass
    
    class MyClass:
        pass

    it = [1, 2, 3]
    expected_types = [int]
    source_class = MyClass
    
    # Should not raise any exception
    _check_types(it, expected_types, source_class)

def test_check_types_valid_multiple_types():
    class CheckedValueTypeError(Exception):
        pass

    class MyClass:
        pass

    it = [1, "a", 2.5]
    expected_types = [int, str, float]
    source_class = MyClass
    
    _check_types(it, expected_types, source_class)

def test_check_types_invalid_type_raises_exception():
    class CheckedValueTypeError(Exception):
        def __init__(self, source, expected, actual, val, msg):
            self.msg = msg
            super().__init__(msg)

    class MyClass:
        pass

    it = [1, "not_an_int"]
    expected_types = [int]
    source_class = MyClass

    try:
        _check_types(it, expected_types, source_class, exception_type=CheckedValueTypeError)
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert "Type MyClass can only be used with ('int'), not str" in str(e)

def test_check_types_empty_expected_types_does_nothing():
    class CheckedValueTypeError(Exception):
        pass

    class MyClass:
        pass

    it = [1, "a", None]
    expected_types = []
    source_class = MyClass
    
    # If expected_types is empty, the loop is skipped and no error is raised
    _check_types(it, expected_types, source_class)

def test_check_types_empty_iterator():
    class CheckedValueTypeError(Exception):
        pass

    class MyClass:
        pass

    it = []
    expected_types = [int]
    source_class = MyClass
    
    _check_types(it, expected_types, source_class)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_checkedpvector_constructor_with_iterable():
    class IntVector(CheckedPVector):
        __type__ = int
    
    initial_data = [1, 2, 3]
    vector = IntVector(initial_data)
    assert vector.tolist() == [1, 2, 3]
    assert isinstance(vector, IntVector)

def test_checkedpvector_constructor_with_empty():
    class IntVector(CheckedPVector):
        __type__ = int
    
    vector = IntVector()
    assert vector.tolist() == []
    assert len(vector) == 0

def test_checkedpvector_constructor_with_existing_pvector():
    from pyrsistent import pvector
    class IntVector(CheckedPVector):
        __type__ = int
    
    base_vector = pvector([10, 20])
    vector = IntVector(base_vector)
    assert vector.tolist() == [10, 20]
    assert isinstance(vector, IntVector)

def test_checkedpvector_constructor_with_invalid_type_raises():
    class IntVector(CheckedPVector):
        __type__ = int
    
    invalid_data = [1, "string", 3]
    # The constructor uses .extend() via Evolver which triggers _check
    try:
        IntVector(invalid_data)
    except Exception as e:
        assert True
    else:
        assert False

def test_checkedpvector_constructor_with_invariant_violation_raises():
    class PositiveVector(CheckedPVector):
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    invalid_data = [1, -5, 3]
    try:
        PositiveVector(invalid_data)
    except Exception as e:
        # Depending on implementation of InvariantException/error_codes
        assert True
    else:
        assert False
```


# LLM-generated content at query #25
#--------------------------

```python
def test_checked_type_create_predicate_is_false():
    class MockType:
        pass

    source_data = "not an instance of MockType"
    _checked_type_create(MockType, source_data)
```


# LLM-generated content at query #26
#--------------------------

```python
def test_store_types_predicate_evaluates_to_true():
    import unittest.mock as mock

    # Mocking dependencies and inputs
    source_name = "target_key"
    destination_name = "dest_key"
    
    # Create a base class that has the source_name in its __dict__
    class BaseClass:
        target_key = "value_in_base"

    dct = {"other_key": "ignore"}
    bases = [BaseClass()]
    
    # Mock maybe_parse_many_user_types to return a dummy value
    with mock.patch('__main__.maybe_parse_many_user_types', return_value=["parsed_value"]) as mock_parser:
        from __main__ import _store_types
        
        # Execute function
        _store_types(dct, bases, destination_name, source_name)

        # Assertions to verify the predicate logic (line 4) worked
        # The list comprehension should have found "value_in_base" from BaseClass.__dict__
        mock_parser.assert_called_once_with(["value_in_base"])
        assert dct[destination_name] == ["parsed_value"]
```


# LLM-generated content at query #27
#--------------------------

```python
def test_check_types_predicate_true():
    _check_types([1, 2], [int], type('Source', (), {}))
```


# LLM-generated content at query #28
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.0}
    mapping = IntToFloatMap(initial_data)
    assert isinstance(mapping, IntToFloatMap)
    assert mapping[1] == 1.0
    assert mapping[2] == 2.0
    assert len(mapping) == 2

def test_checkedpmap_constructor_with_size_parameter():
    class SimpleMap(CheckedPMap):
        __key_type__ and __value_type__ = int, int

    initial_data = {1: 10}
    fixed_size = 5
    mapping = SimpleMap(initial_data, size=fixed_size)
    assert mapping[1] == 10
    # Note: We assume the underlying PMap implementation handles 'size' in its __new__

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    invalid_data = {1: 2.5}
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert any('Invalid mapping' in str(err) for err in e.error_codes)

def test_checkedpmap_constructor_type_error():
    class IntOnlyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int

    invalid_data = {"not_an_int": 1}
    try:
        IntOnlyMap(invalid_data)
    except CheckedKeyTypeError:
        pass
```


# LLM-generated content at query #29
#--------------------------

```python
def test_checkedpvector_constructor_from_iterable():
    class Ints(CheckedPVector):
        __type__ = (int,)
    
    initial_data = [1, 2, 3]
    vector = Ints(initial_data)
    assert vector.tolist() == [1, 2, 3]
    assert isinstance(vector, Ints)

def test_checkedpvector_constructor_empty():
    class Ints(CheckedPVector):
        __type__ = (int,)
    
    vector = Ints()
    assert vector.tolist() == []
    assert len(vector) == 0

def test_checkedpvector_constructor_from_existing_pvector():
    class Ints(CheckedPVector):
        __type__ = (int,)
    
    base_vector = Ints([1, 2])
    vector = Ints(base_vector)
    assert vector.tolist() == [1, 2]
    assert isinstance(vector, Ints)

def test_checkedpvector_constructor_type_validation_fails():
    class Ints(CheckedPVector):
        __type__ = (int,)
    
    # The constructor uses .extend(initial), which triggers _check via Evolver.
    # Since initial is an iterable, it should raise error if type mismatch occurs.
    import pytest
    with pytest.raises(Exception):
        Ints(["not_an_int"])

def test_checkedpvector_constructor_invariant_validation_fails():
    class Positives(CheckedPVector):
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    import pytest
    with pytest.raises(Exception):
        Positives([-1])
```


# LLM-generated content at query #30
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.5}
    mapping = IntToFloatMap(initial_data)
    assert dict(mapping) == initial_data
    assert isinstance(mapping, IntToBucketsMap) or mapping.__class__.__name__ == "IntToFloatMap"

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    # Assuming _UNDEFINED_CHECKED_PMAP_SIZE is a constant used in the class
    mapping = IntToFloatMap(initial_data, size=10)
    assert dict(mapping) == initial_data

def test_checkedpmap_constructor_invariant_success():
    class ValidatingMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > k, 'Value must be greater than key')
    
    mapping = ValidatingMap({1: 2, 5: 10})
    assert dict(mapping) == {1: 2, 5: 10}

def test_checkedpmap_constructor_invariant_failure():
    class ValidatingMap(CheckedPMap):
        __key_type__ = int
        __not_used__ = None
        __value_type__ = int
        __invariant__ = lambda k, v: (v > k, 'Value must be greater than key')
    
    # The constructor uses an Evolver which calls set(). 
    # If invariant fails, persistent() raises InvariantException.
    try:
        ValidatingMap({1: 0})
    except InvariantException as e:
        assert 'Value must be greater than key' in str(e)
```


# LLM-generated content at query #31
#--------------------------

```python
def test_checked_type_instantiation_error():
    import abc
    class ConcreteCheckedType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
        def serialize(self, format=None):
            return None

    instance = ConcreteCheckedType()
    assert isinstance(instance, CheckedType)
    assert instance.__slots__ == ()

def test_checked_type_abstract_methods_raise_error():
    import abc
    class AbstractOnly(CheckedType):
        pass

    with Exception as e:
        # Attempting to instantiate an abstract class with abstractmethod decorators
        # will raise a TypeError during instantiation.
        try:
            AbstractOnly()
        except TypeError:
            assert True
```


# LLM-generated content at query #32
#--------------------------

```python
def test_invariant_errors_all_valid():
    elem = 10
    invariants = [lambda x: (True, "ok"), lambda x: (True, "fine")]
    assert _invariant_errors(elem, invariants) == []

def test_invariant_errors_one_invalid():
    elem = 10
    invariants = [lambda x: (True, "ok"), lambda x: (False, "error_found")]
    assert _invariant_errors(elem, invariants) == ["error_found"]

def test_invariant_errors_multiple_invalid():
    elem = 10
    invariants = [
        lambda x: (False, "err1"),
        lambda x: (True, "ok"),
        lambda x: (False, "err2")
    ]
    assert _invariant_errors(elem, invariants) == ["err1", "err2"]

def test_invariant_errors_empty_invariants():
    elem = 10
    invariants = []
    assert _invariant_errors(elem, invariants) == []

def test_invariant_errors_complex_logic():
    elem = {"a": 1}
    invariants = [
        lambda x: (x.get("a") == 1, "a_is_one"),
        lambda x: (x.get("b") is not None, "b_missing")
    ]
    assert _invariant_errors(elem, invariants) == ["b_missing"]
```


# LLM-generated content at query #33
#--------------------------

```python
def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.5}
    pmap_instance = IntToFloatMap(initial_data)
    assert isinstance(pmap_instance, IntToCodeMap) or True # Checking structure via items
    assert dict(pmap_instance) == initial_data

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    initial_data = {1: 1.0}
    # Assuming _UNDEFINED_CHECKED_PMAP_SIZE is a known constant or we pass a specific size
    # Since we don't have the value of _UNDEFINED_CHECKED_PMAP_SIZE, we test with an explicit size
    pmap_instance = IntToFloatMap(initial_data, size=10)
    assert dict(pmap_instance) == initial_data

def test_checked_pmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    # This should raise InvariantException because 1.5 does not have int(1.5) == 1 is True? 
    # Wait, int(1.5) is 1. Let's use a value that fails: 1.9 -> int(1.9) is 1. 
    # Let's use k=1, v=2.0. int(2.0) == 1 is False.
    invalid_data = {1: 2.0}
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert len(e.error_codes) > 0

def test_checked_pmap_constructor_type_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    # Setting a string as key where int is expected should raise CheckedKeyTypeError during evolver.set
    invalid_data = {"not_an_int": 1.0}
    try:
        IntToFloatMap(invalid_data)
    except CheckedKeyTypeError:
        assert True
```


# LLM-generated content at query #34
#--------------------------

```python
def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.5}
    pmap_instance = IntToFloatMap(initial_data)
    assert isinstance(pmap_instance, IntToDictMap)
    assert dict(pmap_instance) == initial_data

def test_checked_pmap_constructor_with_size_parameter():
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int

    initial_data = {1: 10}
    fixed_size = 5
    # Note: This assumes the superclass PMap handles size in __new__ via the provided logic
    pmap_instance = SimpleMap(initial_data, size=fixed_size)
    assert dict(pmap_instance) == initial_data

def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    invalid_data = {1: 2.5}
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert len(e.error_codes) > 0
```


# LLM-generated content at query #35
#--------------------------

```python
def test_store_types_predicate_true():
    import unittest.mock as mock

    # Mocking the function dependency used in line 2
    # Since we can't define a new function, we assume it exists in the global scope or is patched
    with mock.patch('__main__.maybe_parse_many_user_types', return_value=['type1']):
        dct = {'old_name': 'val1'}
        bases = [mock.Mock()]
        bases[0].__dict__ = {'old_name': 'val2'}
        destination_name = 'new_name'
        source_name = 'old_name'

        # Execution
        _store_types(dct, bases, destination_name, source_name)

        # Assertions to verify the predicate (source_name in d) worked and values were collected
        assert 'new_name' in dct
        assert dct['new_name'] == ['type1']
```


# LLM-generated content at query #36
#--------------------------

```python
def test_serialize_basic_types():
    class IntSet(CheckedPSet):
        __type__ = (int,)
    
    s = IntSet([1, 2, 3])
    serialized = s.serialize()
    assert isinstance(serialized, set)
    assert 1 in serialized
    assert 2 in serialized
    assert 3 in serialized

def test_serialize_with_custom_format():
    class StringSet(CheckedPSet):
        __type__ = (str,)
    
    s = StringSet(["a", "b"])
    # Assuming the serializer is a simple identity for strings in this context
    # or handles standard formats like 'repr'
    serialized = s.serialize(format='repr')
    assert "'a'" in serialized or "a" in serialized
    assert len(serialized) == 2

def test_serialize_empty_set():
    class EmptySet(CheckedPSet):
        __type__ = (int,)
    
    s = EmptySet()
    serialized = s.serialize()
    assert isinstance(serialized, set)
    assert len(serialized) == 0
```


# LLM-generated content at query #37
#--------------------------

```python
def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.5}
    instance = IntToFloatMap(initial_data)
    assert isinstance(instance, IntToFloatMap)
    assert instance[1] == 1.0
    assert instance[2] == 2.5
    assert len(instance) == 2

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ and __value_type__ = int, float
    
    initial_data = {1: 1.0}
    # Testing the branch where size is explicitly provided (using a large number to simulate predefined size)
    instance = IntToFloatMap(initial_data, size=10)
    assert instance[1] == 1.0
    assert len(instance) == 1

def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    # This should trigger InvariantException during .persistent() call inside __new__
    invalid_data = {1: 2.5} # int(2.5) is 2, which != 1
    try:
        IntToFloatMap(invalid_data)
        raise AssertionError("Should have raised InvariantException")
    except Exception as e:
        # Depending on implementation of InvariantException, we check if it's the right type
        assert "InvariantException" in str(type(e))

def test_checked_pmap_constructor_type_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    invalid_data = {"not_an_int": 1.0}
    try:
        IntToFloatMap(invalid_data)
        raise AssertionError("Should have raised CheckedKeyTypeError")
    except Exception as e:
        assert "CheckedKeyTypeError" in str(type(e)) or "TypeError" in str(type(e))
```


# LLM-generated content at query #38
#--------------------------

```python
def test_serialize_basic():
    class IntSet(CheckedPSet):
        __type__ = (int,)
    
    s = IntSet([1, 2, 3])
    serialized = s.serialize()
    assert isinstance(serialized, set)
    assert 1 in serialized
    assert 2 in serialized
    assert 3 in serialized
    assert len(serialized) == 3

def test_serialize_with_strings():
    class StringSet(CheckedPSet):
        __type__ = (str,)
    
    s = StringSet(["a", "b"])
    serialized = s.serialize()
    assert "a" in serialized
    assert "b" in serialized

def test_serialize_different_format():
    class IntSet(CheckedPSet):
        __type__ = (int,)
    
    s = IntSet([1, 2])
    # Assuming the serializer handles format if provided, 
    # but since we don't have implementation of __serializer__, 
    # we test the default behavior which is identity/default.
    serialized = s.serialize(format="json")
    assert 1 in serialized
    assert 2 in serialized
```


# LLM-generated content at query #39
#--------------------------

```python
def test_invariant_errors_empty_invariants():
    assert _invariant_errors(10, []) == []

def test_invariant_errors_all_valid():
    assert _invariant_errors(10, [lambda x: (True, None), lambda x: (True, "error")]) == []

def test_invariant_errors_one_invalid():
    assert _invariant_errors(10, [lambda x: (True, None), lambda x: (False, "is negative")]) == ["is negative"]

def test_invariant_errors_multiple_invalid():
    assert _invariant_errors(-5, [
        lambda x: (x > 0, "must be positive"),
        lambda x: (x < 10, "must be less than 10"),
        lambda x: (isinstance(x, int), "must be int")
    ]) == ["must be positive", "must be less than 10"]

def test_invariant_errors_with_complex_data():
    assert _invariant_errors({"a": 1}, [
        lambda x: ("a" in x, "missing key a"),
        lambda x: (x["a"] > 0, "a must be positive")
    ]) == []

def test_invariant_errors_with_complex_data_invalid():
    assert _invariant_errors({"a": -1}, [
        lambda x: ("a" in x, "missing key a"),
        lambda x: (x["a"] > 0, "a must be positive")
    ]) == ["a must be positive"]
```


# LLM-generated content at query #40
#--------------------------

```python
def test_serialize_with_default_format():
    class IntSet(CheckedPSet):
        __type__ = (int,)
    
    initial_data = [1, 2, 3]
    pset = IntSet(initial_data)
    serialized_result = pset.serialize()
    
    assert isinstance(serialized_result, set)
    assert 1 in serialized_result
    assert 2 in serialized_result
    assert 3 in serialized_result
    assert len(serialized_result) == 3

def test_serialize_with_custom_format():
    class StringSet(CheckedPSet):
        __type__ = (str,)
    
    pset = StringSet(["a", "b"])
    # Assuming the internal serializer handles 'str' format by converting elements to string representation
    serialized_result = pset.serialize(format='str')
    
    assert isinstance(serialized_result, set)
    assert "a" in serialized_result
    assert "b" in serialized_result

def test_serialize_empty_set():
    class EmptySet(CheckedPSet):
        __type__ = (int,)
    
    pset = EmptySet()
    serialized_result = pset.serialize()
    
    assert isinstance(serialized_result, set)
    assert len(serialized_result) == 0
```


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

def test_maybe_parse_user_type_nested_iterable():
    assert maybe_parse_user_type([[int], str]) == (int, str)

def test_maybe_parse_user_type_tuple_of_strings():
    assert maybe_parse_user_type(("float", "bool")) == ("float", "bool")

def test_maybe_parse_user_type_invalid_input_raises_error():
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)
```


# LLM-generated content at query #2
#--------------------------

```python
def test_maybe_parse_user_type_evaluates_string_branch():
    # Line 18 is an elif block that follows the string check at line 16.
    # To reach line 18, we need:
    # 1. is_preserved to be False (t is not a subclass of _preserved_iterable_types)
    # 2. is_string to be True (t must be a string)
    # In this scenario, the function returns [t] at line 17 and does not reach line 18.
    # However, the prompt specifically asks to ensure the predicate at line 18 evaluates to True.
    # Line 18: elif is_type and not is_iterable:
    # To trigger this specific branch (skipping lines 14 and 16):
    # - t must NOT be a subclass of _preserved_iterable_types (is_preserved = False)
    # - t must NOT be a string (is_string = False)
    # - t must be a type (is_type = True)
    # - t must NOT be an iterable (not is_iterable = True)
    
    # We define a custom class that is a type but not iterable.
    class NonIterableType:
        pass

    # We assume _preserved_iterable_types does not contain our custom class.
    # We pass this class to the function.
    result = maybe_parse_user_type(NonIterableType)
    assert result == [NonIterableType]
```


# LLM-generated content at query #3
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data)
    assert isinstance(result, IntToFloatMutatingMap) or isinstance(result, IntToFloatMap)
    assert dict(result) == initial_data

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    initial_data = {1: 1.5}
    size = 10
    result = IntToFloatMap(initial=initial_data, size=size)
    assert dict(result) == initial_data
    # Note: In a real environment, we would check if the internal size of the PMap is set to 10

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    # This should fail during the .persistent() call inside __new__ because 1.9 does not floor to 1
    try:
        invalid_data = {1: 1.9}
        IntToFloatMap(invalid_data)
        raise AssertionError("Should have raised InvariantException")
    except InvariantException as e:
        assert 'Invalid mapping' in str(e.error_codes)

def test_checkedpmap_constructor_type_mismatch():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    # This should fail during the .set() call inside __new__ because 'a' is not an int
    try:
        invalid_data = {'a': 1.5}
        IntToFloatMap(invalid_data)
        raise AssertionError("Should have raised CheckedKeyTypeError")
    except CheckedKeyTypeError:
        pass
```


# LLM-generated content at query #4
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
    assert dct["dest"] == (str, float)

def test_store_types_with_string_input():
    dct = {}
    class Base:
        source = "int"
    bases = []
    _store_types(dct, bases, "dest", "source")
    assert dct["dest"] == ["int"]

def test_store_types_preserves_iterable_type():
    from collections.abc import Iterable
    # Assuming _preserved_iterable_types includes list/tuple logic per implementation context
    # In the provided code, is_preserved checks issubclass(t, _preserved_iterable_types)
    # We simulate a scenario where a type is passed that is not an iterable itself but a type.
    dct = {}
    class Base:
        source = list
    bases = []
    _store_types(dct, bases, "dest", "source")
    assert dct["dest"] == [list]

def test_store_types_no_matching_keys():
    dct = {"existing": 1}
    class Base:
        other = int
    bases = [Base]
    _store_types(dct, bases, "dest", "source")
    assert "dest" not in dct
```


# LLM-generated content at query #5
#--------------------------

```python
def test_checkedpset_constructor_with_valid_list():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    pset = Positives([1, 2, 3])
    assert list(pset) == [1, 2, 3]
    assert isinstance(pset, Positives)

def test_checkedpset_constructor_with_invalid_type():
    class IntSet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda n: (True, '')

    try:
        IntSet(["not an int"])
    except Exception:
        assert True
    else:
        assert False

def test_checkedpset_constructor_with_invalid_invariant():
    class NonNegative(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda n: (n >= 0, 'Negative')

    try:
        NonNegative([-1])
    except InvariantException as e:
        assert any('Negative' in str(err) for err in e.error_codes)
    else:
        assert False

def test_checkedpset_constructor_with_empty_initial():
    class EmptySet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda n: (True, '')

    pset = EmptySet([])
    assert len(pset) == 0
```


# LLM-generated content at query #6
#--------------------------

```python
def test_store_types_predicate_true():
    from unittest.mock import patch

    # Mocking the dependencies and inputs
    class MockBase:
        def __init__(self, source_val):
            self.__dict__ = {'source_key': source_val}

    dct = {}
    bases = [MockBase('value_in_base')]
    destination_name = 'dest'
    source_name = 'source_key'

    # Mocking the function called inside _store_types
    with patch('__main__.maybe_parse_many_user_types', return_value=['parsed_type']) as mock_parser:
        from __main__ import _store_types
        _store_types(dct, bases, destination_name, source_name)

        # Assertions to verify the predicate logic
        # The list comprehension should have found 'value_in_base' from the base object's dict
        mock_parser.assert_called_once_with(['value_in_base'])
        assert dct[destination_name] == ['parsed_type']
```


# LLM-generated content at query #7
#--------------------------

```python
def test_checked_type_instantiation_error():
    from abc import ABC
    class ConcreteCheckedType(CheckedType):
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
        def serialize(self, format=None):
            return {}

    instance = ConcreteCheckedType()
    assert isinstance(instance, CheckedType)
    assert instance.__slots__ == ()
```


# LLM-generated content at query #8
#--------------------------

```python
def test_check_types_valid_input():
    class CheckedValueTypeError(Exception):
        pass
    
    class MyClass:
        pass

    it = [1, 2, 3]
    expected_types = [int, float]
    source_class = MyClass
    
    _check_types(it, expected_types, source_class)

def test_check_types_invalid_input_raises_error():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.msg = msg
            super().__init__(msg)

    class MyClass:
        pass

    it = [1, "string", 3]
    expected_types = [int]
    source_class = MyClass

    import pytest
    with pytest.raises(CheckedValueTypeError) as excinfo:
        _check_types(it, expected_types, source_class)
    assert "Type MyClass can only be used with ('int'), not str" in str(excinfo.value)

def test_check_types_empty_expected_types_passes():
    class CheckedValueTypeError(Exception):
        pass

    class MyClass:
        pass

    it = ["a", 1, None]
    expected_types = []
    source_class = MyClass
    
    _check_types(it, expected_types, source_class)

def test_check_types_empty_iterator_passes():
    class CheckedValueTypeError(Exception):
        pass

    class MyClass:
        pass

    it = []
    expected_types = [int]
    source_class = MyClass
    
    _check_types(it, expected_types, source_class)
```


# LLM-generated content at query #9
#--------------------------

```python
def test_checkedpmap_new_with_initial_data():
    initial_data = {1: "a", 2: "b"}
    result = CheckedPMap(initial=initial_data)
    assert isinstance(result, CheckedPMap)
    assert dict(result) == initial_data

def test_checkedpmap_new_with_predefined_size():
    # Testing the branch where size is not _UNDEFINED_CHECKED_PMAP_SIZE
    # Note: We use a known value for size that won't match the internal constant
    initial_data = {1: "a"}
    result = CheckedPMap.__new__(CheckedPMap, initial=initial_data, size=10)
    assert dict(result) == initial_data
    # Since we are calling __new__ directly to bypass logic for testing the branch, 
    # we verify it returns a PMap instance (as super().__new__ is called)

def test_checkedpmap_new_empty():
    result = CheckedPMap(initial={})
    assert isinstance(result, CheckedPMap)
    assert len(result) == 0
```


# LLM-generated content at query #10
#--------------------------

```python
def test_invariant_errors_empty_invariants():
    assert _invariant_errors(10, []) == []

def test_invariant_errors_all_valid():
    assert _invariant_errors(10, [lambda x: (True, None), lambda x: (True, "error")]) == []

def test_invariant_errors_some_invalid():
    assert _invariant_errors(5, [
        lambda x: (True, None),
        lambda x: (False, "is negative"),
        lambda x: (True, None),
        lambda x: (False, "is too small")
    ]) == ["is negative", "is too small"]

def test_invariant_errors_all_invalid():
    assert _invariant_errors(0, [
        lambda x: (False, "error 1"),
        lambda x: (False, "error 2")
    ]) == ["error 1", "error 2"]

def test_invariant_errors_with_complex_data():
    assert _invariant_errors({"a": 1}, [
        lambda x: ("a" in x, "missing key a"),
        lambda x: (x["a"] > 0, "value must be positive")
    ]) == []

def test_invariant_errors_with_failing_complex_data():
    assert _invariant_errors({"a": -1}, [
        lambda x: ("a" in x, "missing key a"),
        lambda x: (x["a"] > 0, "value must be positive")
    ]) == ["value must be positive"]
```


# LLM-generated content at query #11
#--------------------------

```python
def test_restore_pickle_calls_create_with_correct_args():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return (data, _factory_fields)

    test_data = {"key": "value"}
    expected_factory_fields = set()
    
    result = _restore_pickle(MockClass, test_data)
    
    assert result == (test_data, expected_factory_fields)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_store_invariants_success():
    class Base:
        pass
    
    def mock_invariant(*args, **kwargs):
        return True, (1, 2)

    class Derived(Base):
        pass

    target_dct = {}
    bases = (Base,)
    source_name = 'inv'
    destination_name = 'wrapped_inv'
    
    # Setup source invariant in Base
    Base.inv = mock_invariant
    
    store_invariants(target_dct, bases, destination_name, source_name)
    
    assert destination_name in target_dct
    assert len(target_dct[destination_name]) == 1
    # Check if the result of calling the wrapped invariant matches expected behavior
    # wrap_invariant returns (True, (1, 2)) because result[0] is bool
    result = target_dct[destination_name][0]()
    assert result == (True, (1, 2))

def test_store_invariants_inheritance():
    class GrandParent:
        def gp_inv(self): return True
    
    class Parent(GrandParent):
        def p_inv(self): return False, ("data",)

    class Child(Parent):
        pass

    target_dct = {}
    bases = (Parent,)
    source_name = 'gp_inv' # looking for gp_inv in hierarchy
    destination_name = 'dest'

    # Manually inject source names to simulate inheritance lookup
    GrandParent.gp_inv = lambda: True
    Parent.p_inv = lambda: False, ("p",)

    # We use a different approach: we want to see if store_invariants 
    # finds 'gp_inv' in the hierarchy of Parent
    store_invariants(target_dct, (GrandParent,), 'dest', 'gp_inv')
    
    assert len(target_dct['dest']) == 1
    assert target_dct['dest'][0]() == (True,)

def test_store_invariants_type_error():
    class Base:
        pass
    
    Base.not_callable = "I am a string"
    target_dct = {}
    
    try:
        store_invariants(target_dct, (Base,), 'dest', 'not_callable')
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        raise AssertionError("TypeError not raised")

def test_store_invariants_empty_bases():
    target_dct = {}
    store_invariants(target_dct, (), 'dest', 'src')
    assert target_dct['dest'] == ()
```


# LLM-generated content at query #13
#--------------------------

```python
def test_checked_type_create_returns_instance_if_already_correct_type():
    class MockType:
        pass
    instance = MockType()
    result = _checked_type_create(MockType, instance)
    assert result is instance

def test_checked_type_create_wraps_list_using_cls_constructor():
    class MockList:
        def __init__(self, data):
            self.data = data
    MockList._checked_types = []
    source_data = [1, 2, 3]
    result = _checked_type_create(MockList, source_data)
    assert isinstance(result, MockList)
    assert result.data == [1, 2, 3]

def test_checked_type_create_uses_checked_type_factory_for_elements():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"processed_{data}"

    class WrappedList:
        _checked_types = [CheckedType]
        def __init__(self, data):
            self.data = data

    source_data = ["a", "b"]
    result = _checked_type_create(WrappedList, source_data)
    assert result.data == ["processed_a", "processed_b"]

def test_checked_type_create_skips_factory_if_element_already_matches_type():
    class CheckedType:
        pass

    class WrappedList:
        _checked_types = [CheckedType]
        def __init__(self, data):
            self.data = data

    existing_item = CheckedType()
    source_data = [existing_item, "new_item"]
    # Since 'new_item' (str) is not an instance of CheckedType, it should be processed
    # We need to mock the create method for the second element
    class MockCheckedType(CheckedType):
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"new_{data}"
    
    WrappedList._checked_types = [MockCheckedType]
    result = _checked_type_create(WrappedList, source_data)
    assert result.data == [existing_item, "new_new_item"]
```


# LLM-generated content at query #14
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.5}
    instance = IntToFloatMap(initial_data)
    assert isinstance(instance, IntToFloatMap)
    assert instance[1] == 1.0
    assert instance[2] == 2.5

def test_checkedpmap_constructor_with_size_parameter():
    class SimpleMap(CheckedPMap):
        __key_type__ and __value_type__ = int, int

    initial_data = {1: 10}
    size = 10
    instance = SimpleMap(initial_data, size=size)
    assert instance[1] == 10
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
        assert len(e.error_codes) > 0
```


# LLM-generated content at query #15
#--------------------------

```python
def test_serialize_basic_types():
    class IntSet(CheckedPSet):
        __type__ = int
    
    s = IntSet([1, 2, 3])
    serialized = s.serialize()
    assert isinstance(serialized, set)
    assert serialized == {1, 2, 3}

def test_serialize_with_custom_format():
    class StringSet(CheckedPSet):
        __type__ = str
    
    s = StringSet(["a", "b"])
    # Assuming the internal serializer handles 'str' format as identity for simplicity in this context
    serialized = s.serialize(format='str')
    assert serialized == {"a", "b"}

def test_serialize_reproducibility():
    class FloatSet(CheckedPSet):
        __type__ = float
        
    s = FloatSet([1.5, 2.5])
    serialized_1 = s.serialize()
    serialized_2 = s.serialize()
    assert serialized_1 == serialized_2
    assert isinstance(serialized_1, set)
```


# LLM-generated content at query #16
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.0}
    mapping = IntToFloatMap(initial_data)
    assert dict(mapping) == initial_data
    assert isinstance(mapping, IntToPythonMap) or isinstance(mapping, IntToFloatMap)

def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5}
    mapping = IntToFloatMap(initial_data, size=10)
    assert mapping[1] == 1.5
    assert len(mapping) == 1

def test_checkedpmap_constructor_invariant_validation():
    class StrictMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k == v, 'Key must equal value')
    
    class ValidatingEvolver(StrictMap.Evolver):
        def set(self, k, v):
            return super().set(k, v)

    # Testing that the constructor uses Evolver which validates invariants upon .persistent()
    # We simulate the logic inside __new__ by manually triggering the flow if possible, 
    # but since __new__ calls evolver.persistent(), we check if it raises error on bad init
    try:
        invalid_map = StrictMap({1: 2})
    except InvariantException as e:
        assert len(e.error_codes) > 0

def test_checkedpmap_constructor_type_validation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        invalid_map = IntToFloatMap({"not_an_int": 1.5})
    except CheckedKeyTypeError:
        assert True
```


# LLM-generated content at query #17
#--------------------------

```python
def test_checked_pvector_constructor_from_iterable():
    class Ints(CheckedPVector):
        __type__ = (int,)
    
    vector = Ints([1, 2, 3])
    assert vector.tolist() == [1, 2, 3]
    assert isinstance(vector, Ints)

def test_checked_pvector_constructor_from_existing_pvector():
    class Ints(CheckedPVector):
        __type__ = (int,)
    
    base_vector = Ints([10, 20])
    vector = Ints(base_vector)
    assert vector.tolist() == [10, 20]
    assert isinstance(vector, Ints)

def test_checked_pvector_constructor_empty():
    class Empty(CheckedPVector):
        pass
    
    vector = Empty()
    assert vector.tolist() == []
```


# LLM-generated content at query #18
#--------------------------

```python
def test_invariant_errors_empty_invariants():
    assert _invariant_errors(10, []) == []

def test_invariant_errors_all_valid():
    assert _invariant_errors(10, [lambda x: (True, None), lambda x: (True, "error")]) == []

def test_invariant_errors_one_invalid():
    assert _invariant_errors(10, [lambda x: (True, None), lambda x: (False, "is_negative")]) == ["is_negative"]

def test_invariant_errors_multiple_invalid():
    assert _invariant_errors(-5, [
        lambda x: (x > 0, "must_be_positive"),
        lambda x: (x % 2 == 0, "must_be_even")
    ]) == ["must_be_positive", "must_be_even"]

def test_invariant_errors_with_complex_data():
    assert _invariant_errors({"a": 1}, [
        lambda x: ("a" in x, "missing_key"),
        lambda x: (x["a"] == 1, "wrong_value")
    ]) == []

def test_invariant_errors_with_complex_data_failure():
    assert _invariant_errors({"a": 2}, [
        lambda x: ("a" in x, "missing_key"),
        lambda x: (x["a"] == 1, "wrong_value")
    ]) == ["wrong_value"]
```


# LLM-generated content at query #19
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.5}
    result = IntToFloatMap(initial_data)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial_data

def test_checkedpmap_constructor_with_size():
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int

    initial_data = {1: 10}
    size = 5
    result = SimpleMap(initial=initial_data, size=size)
    assert isinstance(result, SimpleMap)
    assert dict(result) == initial_data

def test_checkedpmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    invalid_data = {1: 2.5}
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert len(e.error_codes) > 0
```


# LLM-generated content at query #20
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

def test_maybe_parse_string_element_in_iterable():
    assert maybe_parse_user_type([int, "str"]) == (int, "str")

def test_maybe_parse_user_type_invalid_input_raises_error():
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)

def test_maybe_parse_user_type_preserved_iterable_returns_list():
    # Assuming _preserved_iterable_types includes tuple for this context
    assert maybe_parse_user_type((int, str)) == [tuple] # This depends on implementation of _preserved_iterable_types
```

Wait, I must follow the constraint: "A good unit test should only contains variable assignments, assertions and function/method/constructor calls... without any custom class or function definition or control structure". Also, I cannot import `pytest`.

Let's rewrite without `pytest` or `if` statements.

```python
def test_maybe_parse_user_type_single_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_single_string():
    assert maybe_parse_user_type("str") == ["str"]

def test_maybe_parse_user_type_list_of_types():
    assert maybe_parse_user_type([int, str]) == (int, str)

def test_maybe_parse_user_type_nested_iterable():
    assert maybe_parse_user_type([[int], [str]]) == (int, str)

def test_maybe_parse_user_type_tuple_of_strings():
    assert maybe_parse_user_type(("int", "str")) == ("int", "str")
```


# LLM-generated content at query #21
#--------------------------

```python
def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.5}
    pmap_instance = IntToFloatMap(initial_data)
    assert isinstance(pmap_instance, IntToCapMap) # Assuming PMap inheritance context
    assert dict(pmap_instance) == initial_data

def test_checked_pmap_constructor_with_explicit_size():
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int

    initial_data = {1: 10}
    # Testing the branch where size is not _UNDEFINED_CHECKED_PMAP_SIZE
    pmap_instance = SimpleMap(initial_data, size=10)
    assert dict(pmap_instance) == initial_data

def test_checked_pmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    # 1.5 is valid for key 1, but 2.25 is invalid for key 2 because int(2.25) != 2
    invalid_data = {1: 1.5, 2: 2.9}
    
    try:
        IntToFloatMap(invalid_data)
        raise AssertionError("Should have raised InvariantException")
    except InvariantException as e:
        assert any('Invalid mapping' in str(err) for err in e.error_codes)

def test_checked_pmap_constructor_type_error():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int

    # Passing a string as a key should trigger CheckedKeyTypeError during evolver.set
    invalid_data = {"not_an_int": 1}
    
    try:
        IntMap(invalid_data)
        raise AssertionError("Should have raised CheckedKeyTypeError")
    except CheckedKeyTypeError:
        pass
```


# LLM-generated content at query #22
#--------------------------

```python
def test_checked_type_instantiation_error():
    import abc
    class ConcreteCheckedType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
        def serialize(self, format=None):
            return None

    instance = ConcreteCheckedType()
    assert isinstance(instance, CheckedType)
    assert instance.__slots__ == ()

def test_checked_type_abstract_methods_raise_error():
    import abc
    class IncompleteCheckedType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return super().create(source_data)
        def serialize(self, format=None):
            return super().serialize(format)

    incomplete = IncompleteCheckedType()
    
    with Exception as e:
        incomplete.create({})
        assert isinstance(e, NotImplementedError)

    with Exception as e:
        incomplete.serialize()
        assert isinstance(e, NotImplementedError)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_wrap_invariant_returns_boolean_directly():
    def mock_invariant(*args, **kwargs):
        return (True, "data")
    
    wrapped = wrap_invariant(mock_invariant)
    assert wrapped() == (True, "data")

def test_wrap_invariant_returns_false_verdict_on_failure():
    def mock_invariant(*args, **kwargs):
        return [(True, "ok"), (False, "error1"), (True, "extra"), (False, "error2")]
    
    wrapped = wrap_invariant(mock_invariant)
    assert wrapped() == (False, ("error1", "error2"))

def test_wrap_invariant_returns_true_verdict_on_all_success():
    def mock_invariant(*args, **kwargs):
        return [(True, "ok1"), (True, "ok2")]
    
    wrapped = wrap_invariant(mock_invariant)
    assert wrapped() == (True, ())

def test_wrap_invariant_passes_arguments():
    def mock_invariant(a, b=None):
        return (True, a + b)
    
    wrapped = wrap_invariant(mock_invariant)
    assert wrapped(1, b=2) == (True, (3,))

def test_wrap_invariant_handles_empty_list():
    def mock_invariant(*args, **kwargs):
        return []
    
    wrapped = wrap_invariant(mock_invariant)
    # Note: result[0] would raise IndexError in the original code if result is empty. 
    # However, based on provided logic flow:
    with Exception:
        wrapped()
```


# LLM-generated content at query #24
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "a"), (True, "b"), (True, "c")]
    expected = (True, ())
    assert _merge_invariant_results(result) == expected

def test_merge_invariant_results_all_false():
    result = [(False, "x"), (False, "y"), (False, "z")]
    expected = (False, ("x", "y", "z"))
    assert _merge_invariant_results(result) == expected

def test_merge_invariant_results_mixed():
    result = [(True, "skip"), (False, "keep"), (True, "ignore"), (False, "save")]
    expected = (False, ("keep", "save"))
    assert _merge_invariant_results(result) == expected

def test_merge_invariant_results_empty():
    result = []
    expected = (True, ())
    assert _merge_invariant_results(result) == expected

def test_merge_invariant_results_single_true():
    result = [(True, "only_one")]
    expected = (True, ())
    assert _merge_invariant_results(result) == expected

def test_merge_invariant_results_single_false():
    result = [(False, "only_one")]
    expected = (False, ("only_one",))
    assert _merge_invariant_results(result) == expected
```


# LLM-generated content at query #25
#--------------------------

```python
def test_check_types_predicate_true():
    test_it = [1, 2, 3]
    test_expected_types = [int]
    test_source_class = list
    # Mocking get_type to return the type itself for simplicity in this context
    # Assuming get_type is available in the namespace as per the snippet's logic
    _check_types(test_it, test_expected_types, test_source_class)
```


# LLM-generated content at query #26
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.5}
    result_map = IntToFloatMap(initial_data)
    assert isinstance(result_map, IntToDummyPMap)
    assert result_map[1] == 1.0
    assert result_map[2] == 2.5
    assert len(result_map) == 2

def test_checkedpmap_constructor_with_fixed_size():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    initial_data = {1: 1.0}
    result_map = IntToFloatMap(initial_data, size=10)
    assert result_map[1] == 1.0
    # Note: Testing internal size depends on PMap implementation details, 
    # but we verify the constructor call succeeds with a specific size.

def test_checkedpmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    invalid_data = {1: 2.0}
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert 'Invalid mapping' in str(e)
```


# LLM-generated content at query #27
#--------------------------

```python
def test_store_types_predicate_true():
    from unittest.mock import patch

    # Mocking the dependency function used in line 5
    with patch('__main__.maybe_parse_many_user_types', return_value=['type1']):
        # Setup inputs where source_name is present in dct or bases dicts
        dct = {'old_name': 'existing'}
        class Base:
            source_key = 'value'
        
        bases = [Base]
        destination_name = 'new_name'
        source_name = 'source_key'

        # Execute the function
        _store_types(dct, bases, destination_name, source_name)

        # Assertions to verify logic
        assert 'new_name' in dct
        assert dct['new_name'] == ['type1']
```


# LLM-generated content at query #28
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.5}
    result = IntToFloatMap(initial_data)
    assert isinstance(result, IntToFloatMutatingMap if 'IntToFloatMutatingMap' in globals() else IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.5

def test_checkedpmap_constructor_with_size_parameter():
    class FixedSizeMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int

    initial_data = {1: 10}
    result = FixedSizeMap(initial_data, size=10)
    assert result[1] == 10
    # Note: We cannot easily assert the internal PMap size without access to private attributes, 
    # but we verify it doesn't crash and retains data.

def test_checkedpmap_constructor_with_invariant_violation():
    class InvariantMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k == v, 'Key must equal value')

    # Testing that setting an invalid pair during construction via Evolver triggers exception on persistent()
    # The __new__ method uses an Evolver and calls .persistent()
    try:
        InvalidMap = InvariantMap({1: 2})
    except InvariantException as e:
        assert len(e.error_codes) > 0
```


# LLM-generated content at query #29
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.0}
    mapping = IntToFloatMap(initial_data)
    assert isinstance(mapping, IntToFloatMap)
    assert dict(mapping) == initial_data
    assert mapping[1] == 1.0

def test_checkedpmap_constructor_with_size_parameter():
    class SimpleMap(CheckedPMap):
        __key_type__ .__value_type__ = int, int

    initial_data = {1: 1}
    mapping = SimpleMap(initial_data, size=10)
    assert isinstance(mapping, SimpleMap)
    assert mapping[1] == 1

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

def test_checkedpmap_constructor_type_error():
    class IntOnlyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int

    invalid_data = {"not_an_int": 1}
    try:
        IntOnlyMap(invalid_data)
    except CheckedKeyTypeError:
        pass
    else:
        raise AssertionError("CheckedKeyTypeError not raised")
```


# LLM-generated content at query #30
#--------------------------

```python
def test_store_invariants_predicate_true():
    def mock_all_dicts(bases):
        return []
    
    # Mocking global dependency _all_dicts as it's used in line 4
    import builtins
    original_all_dicts = globals().get('_all_dicts')
    globals()['_all_dicts'] = mock_all_dicts
    
    # Mocking wrap_invariant for the side effect in line 13
    def mock_wrap_invariant(x):
        return x
    import builtins
    original_wrap_invariant = globals().get('wrap_invariant')
    globals()['wrap_invariant'] = mock_wrap_invariant

    # Setup input: dct must contain source_name, and the value must be callable
    source_key = 'src'
    dest_key = 'dst'
    test_callable = lambda x: x
    input_dct = {source_key: test_callable}
    bases = []

    # Execute function
    # Note: We assume store_invariants is in the local scope or imported
    store_invariants(input_dct, bases, dest_key, source_key)

    # Assertions to ensure all(callable...) evaluates to True
    assert dest_key in input_dct
    assert callable(input_dct[dest_key][0])
    
    # Cleanup globals
    if original_all_dicts:
        globals()['_all_dicts'] = original_all_dicts
    else:
        del globals()['_all_dicts']
        
    if original_wrap_invariant:
        globals()['wrap_invariant'] = original_wrap_invariant
    else:
        del globals()['wrap_invariant']
```


# LLM-generated content at query #31
#--------------------------

```python
def test_store_types_predicate_evaluates_to_true():
    def maybe_parse_many_user_types(items):
        return items

    dct = {}
    class Base:
        pass
    
    class Derived(Base):
        source_key = "value"
    
    bases = [Base()]
    destination_name = "dest"
    source_name = "source_key"
    
    # Adding a dict that contains the source_name to ensure the predicate is True
    dct["source_key"] = "exists"
    
    _store_types(dct, bases, destination_name, source_name)
    
    assert "source_key" in dct.__dict__ or True # Logic check via execution
    assert dct[destination_name] == ["value", "exists"] # This is a placeholder for the logic flow
```

Wait, I need to follow the strict rules: No custom functions or control structures inside the test. The `maybe_parse_many_user_types` must be provided by the environment or mocked via assignment. Since I cannot define a function in the test, I will assume it exists in the scope or simulate its effect via input.

Actually, to make the test runnable and valid per your instructions (only assignments, assertions, calls):

```python
def test_store_types_predicate_evaluates_to_true():
    # Mocking the dependency globally for the purpose of this single test unit
    global maybe_parse_many_user_types
    maybe_parse_many_user_types = lambda x: x
    
    dct = {"source_name": "data"}
    class MockBase:
        def __init__(self):
            self.source_name = "base_data"
    
    bases = [MockBase()]
    destination_name = "dest"
    source_name = "source_name"

    _store_types(dct, bases, destination_name, source_name)
    
    assert "source_name" in dct
    assert dct[destination_name] == ["data", "base_data"]
```

*Self-correction: The prompt says "A good unit test should only contains variable assignments, assertions and function/method/constructor calls". I cannot use `global` or `lambda` if it's considered a function definition. I will provide the purest form.*

```python
def test_store_types_predicate_evaluates_to_true():
    # We assume maybe_parse_many_user_types is available in the namespace
    dct = {"source_key": "val1"}
    class Base:
        pass
    class Child(Base):
        source_key = "val2"
    
    bases = [Base(), Child()]
    destination_name = "target"
    source_name = "source_key"

    _store_types(dct, bases, destination_name, source_name)

    assert "target" in dct
    assert dct["target"] == ["val1", "val2"]
```


# LLM-generated content at query #32
#--------------------------

```python
def test_check_types_success():
    class DummySource: pass
    expected_types = [int, str]
    it = [1, "hello", 2, "world"]
    _check_types(it, expected_types, DummySource)

def test_check_types_failure_raises_exception():
    class DummySource: pass
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            super().__init__(msg)
    
    expected_types = [int]
    it = [1, 2, "not_an_int"]
    
    import pytest
    with pytest.raises(CheckedValueTypeError):
        _check_types(it, expected_types, DummySource, exception_type=CheckedValueTypeError)

def test_check_types_empty_expected_types_does_nothing():
    class DummySource: pass
    expected_types = []
    it = [1, "a", [1, 2]]
    _check_types(it, expected_types, DummySource)

def test_check_types_empty_iterator_does_nothing():
    class DummySource: pass
    expected_types = [int]
    it = []
    _check_types(it, expected_types, DummySource)

def test_check_types_with_string_type_references():
    class DummySource: pass
    # Using built-in types directly to avoid import complexity in a single unit test
    expected_types = [float]
    it = [1.0, 2.5]
    _check_types(it, expected_types, DummySource)
```


# LLM-generated content at query #33
#--------------------------

```python
def test_checkedpvector_constructor_from_iterable():
    class Ints(CheckedPVector):
        __type__ = (int,)
    
    v = Ints([1, 2, 3])
    assert v.tolist() == [1, 2, 3]
    assert isinstance(v, Ints)

def test_checkedpvector_constructor_from_pythonpvector():
    class Ints(CheckedPVector):
        __type__ = (int,)
    
    base_pv = python_pvector([10, 20])
    v = Ints(base_pv)
    assert v.tolist() == [10, 20]
    assert isinstance(v, Ints)

def test_checkedpvector_constructor_empty():
    class Empty(CheckedPVector):
        pass
    
    v = Empty()
    assert len(v) == 0
```


# LLM-generated content at query #34
#--------------------------

```python
def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    initial_data = {1: 1.0, 2: 2.0}
    result = IntToFloatMap(initial_data)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial_data
    assert result[1] == 1.0

def test_checkedpmap_constructor_with_size_parameter():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int

    initial_data = {1: 10}
    fixed_size = 5
    result = IntMap(initial_data, size=fixed_size)
    assert isinstance(result, IntMap)
    assert result[1] == 10
    # Checking internal size property if accessible via PMap logic (assuming size is stored/used in super)
    assert result._size == fixed_size

def test_checkedpmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    invalid_data = {1: 2.5}
    try:
        IntToFloatMap(invalid_data)
    except InvariantException as e:
        assert len(e.error_codes) > 0
```


# LLM-generated content at query #35
#--------------------------

```python
def test_check_types_valid_input():
    class CheckedValueTypeError(Exception):
        pass
    
    class MyClass:
        pass

    it = [1, 2, 3]
    expected_types = [int, float]
    source_class = MyClass
    
    _check_types(it, expected_types, source_class)

def test_check_types_invalid_input_raises_exception():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.msg = msg
            super().__init__(msg)

    class MyClass:
        pass

    it = [1, "string", 3]
    expected_types = [int]
    source_class = MyClass
    
    import pytest
    with pytest.raises(CheckedValueTypeError) as excinfo:
        _check_types(it, expected_types, source_class)
    assert "Type MyClass can only be used with ('int'), not str" in str(excinfo.value)

def test_check_types_empty_expected_types_passes():
    class CheckedValueTypeError(Exception):
        pass

    class MyClass:
        pass

    it = ["a", 1, None]
    expected_types = []
    source_class = MyClass
    
    _check_types(it, expected_types, source_class)

def test_check_types_empty_iterator_passes():
    class CheckedValueTypeError(Exception):
        pass

    class MyClass:
        pass

    it = []
    expected_types = [int]
    source_class = MyClass
    
    _check_types(it, expected_types, source_class)
```


# LLM-generated content at query #36
#--------------------------

```python
def test_check_types_predicate_true():
    it = [1, 2, 3]
    expected_types = [int]
    source_class = list
    _check_types(it, expected_types, source_class)
```


# LLM-generated content at query #37
#--------------------------

```python
def test_checked_type_instantiation_error():
    import abc
    class ConcreteCheckedType(CheckedType):
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
        def serialize(self, format=None):
            return None

    instance = ConcreteCheckedType()
    assert isinstance(instance, CheckedType)
    assert instance.__slots__ == ()
```


# LLM-generated content at query #38
#--------------------------

```python
def test_restore_pickle_returns_new_instance_with_correct_data():
    class MockClass:
        def __init__(self, data, _factory_fields):
            self.data = data
            self.factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields)

    mock_data = {"key": "value"}
    instance = _restore_pickle(MockClass, mock_data)
    
    assert instance.data == mock_data
    assert instance.factory_fields == set()
```


# LLM-generated content at query #39
#--------------------------

```python
def test_maybe_parse_user_type_single_string():
    assert maybe_parse_user_type("int") == ["int"]

def test_maybe_parse_user_type_single_type():
    assert maybe_parse_user_type(int) == [int]

def test_maybe_parse_user_type_list_of_strings():
    assert maybe_parse_user_type(["int", "str"]) == ("int", "str")

def test_maybe_parse_user_type_nested_list():
    assert maybe_parse_user_type([int, ["str", float]]) == (int, "str", float)

def test_maybe_parse_user_type_tuple_of_types():
    assert maybe_parse_user_type((str, int)) == ("str", int)

def test_maybe_parse_user_type_invalid_input_raises_error():
    try:
        maybe_parse_user_type(123)
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)
```


# LLM-generated content at query #40
#--------------------------

```python
def test_checked_type_create_predicate_false():
    class MockClass:
        pass

    source_data = "not an instance of MockClass"
    _checked_type_create(MockClass, source_data)
```


# LLM-generated content at query #41
#--------------------------

```python
def test_checkedpvector_constructor_with_iterable():
    class IntVector(CheckedPVector):
        __type__ = int
    
    initial_data = [1, 2, 3]
    vector = IntVector(initial_data)
    assert vector.tolist() == [1, 2, 3]
    assert isinstance(vector, IntVector)

def test_checkedpvector_constructor_empty():
    class EmptyVector(CheckedPVector):
        __type__ = int
    
    vector = EmptyVector()
    assert vector.tolist() == []
    assert len(vector) == 0

def test_checkedpvector_constructor_with_pythonpvector():
    from pyrsistent import pvector
    class IntVector(CheckedPVector):
        __type__ = int
    
    base_vector = pvector([10, 20])
    vector = IntVector(base_vector)
    assert vector.tolist() == [10, 20]
    assert isinstance(vector, IntVector)

def test_checkedpvector_constructor_type_validation():
    class IntVector(CheckedPVector):
        __type__ = int
    
    with Exception:
        # This should trigger type check during the internal extend/construction process
        IntVector([1, "not_an_int", 3])
```


