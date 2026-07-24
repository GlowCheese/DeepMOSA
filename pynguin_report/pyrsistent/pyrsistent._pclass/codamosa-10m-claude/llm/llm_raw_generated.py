####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method __reduce__ of class PClass
def test_PClass___reduce__():```python


# LLM-generated content at query #2
#--------------------------

# Unit test for method set of class PClass
def test_PClass_set():```python


# LLM-generated content at query #3
#--------------------------

# Unit test for method set of class PClass
def test_PClass_set():```python


# LLM-generated content at query #4
#--------------------------

# Unit test for method remove of class _PClassEvolver
def test__PClassEvolver_remove():```python


# LLM-generated content at query #5
#--------------------------

# Unit test for method set of class PClass
def test_PClass_set():```python


# LLM-generated content at query #6
#--------------------------

# Unit test for method __new__ of class PClass
def test_PClass___new__():```python


# LLM-generated content at query #7
#--------------------------

# Unit test for method __reduce__ of class PClass
def test_PClass___reduce__():```python


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_PClass___reduce__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    # Test basic reduce functionality
    obj = TestClass(x=1, y=2, z=3)
    reduce_result = obj.__reduce__()
    
    # Should return a tuple of (callable, args)
    assert isinstance(reduce_result, tuple)
    assert len(reduce_result) == 2
    
    restore_func, (cls, data) = reduce_result
    
    # Check the restoration function
    assert restore_func == _restore_pickle
    
    # Check the class is correct
    assert cls == TestClass
    
    # Check the data dictionary contains all fields
    assert data == {'x': 1, 'y': 2, 'z': 3}
    
    # Test that we can restore from the pickle data
    restored = _restore_pickle(cls, data)
    assert restored == obj
    assert restored.x == 1
    assert restored.y == 2
    assert restored.z == 3


def test_PClass___reduce__with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=42)
        z = field()
    
    # Test with only some fields set
    obj = TestClass(x=1, z=3)
    reduce_result = obj.__reduce__()
    
    restore_func, (cls, data) = reduce_result
    
    # Should only include fields that are actually set
    assert 'x' in data
    assert 'z' in data
    assert 'y' in data  # Has initial value
    assert data['x'] == 1
    assert data['z'] == 3
    assert data['y'] == 42


def test_PClass___reduce__with_complex_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test with complex values
    obj = TestClass(x=[1, 2, 3], y={'a': 'b'})
    reduce_result = obj.__reduce__()
    
    restore_func, (cls, data) = reduce_result
    
    assert data['x'] == [1, 2, 3]
    assert data['y'] == {'a': 'b'}
    
    # Verify restoration works
    restored = _restore_pickle(cls, data)
    assert restored == obj


def test_PClass___reduce__pickling_roundtrip():
    import pickle
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=10, y=20)
    
    # Pickle and unpickle
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    # Should be equal
    assert unpickled == obj
    assert unpickled.x == 10
    assert unpickled.y == 20


# LLM-generated content at query #2
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field(initial=10)
    
    # Test basic set with kwargs
    obj = TestClass(x=1, y=2)
    obj2 = obj.set(x=5)
    assert obj.x == 1
    assert obj2.x == 5
    assert obj2.y == 2
    assert obj2.z == 10
    
    # Test set with args (field name as string)
    obj3 = obj.set('y', 20)
    assert obj.y == 2
    assert obj3.y == 20
    assert obj3.x == 1
    
    # Test multiple field updates
    obj4 = obj.set(x=100, y=200)
    assert obj4.x == 100
    assert obj4.y == 200
    assert obj.x == 1
    assert obj.y == 2
    
    # Test that original object is unmodified
    original_x = obj.x
    original_y = obj.y
    obj5 = obj.set(x=999, y=888)
    assert obj.x == original_x
    assert obj.y == original_y
    
    # Test setting field with initial value
    obj6 = obj.set(z=50)
    assert obj6.z == 50
    assert obj.z == 10
    
    # Test set returns new instance of same class
    obj7 = obj.set(x=7)
    assert isinstance(obj7, TestClass)
    assert obj7 is not obj
    
    # Test args and kwargs together
    obj8 = obj.set('x', 11, y=22)
    assert obj8.x == 11
    assert obj8.y == 22


# LLM-generated content at query #3
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    class AClass(PClass):
        x = field()
        y = field()
    
    # Test basic set with keyword argument
    a = AClass(x=1, y=2)
    a2 = a.set(x=2)
    assert a.x == 1
    assert a.y == 2
    assert a2.x == 2
    assert a2.y == 2
    
    # Test set with positional arguments (field name and value)
    a3 = a.set('x', 3)
    assert a.x == 1
    assert a3.x == 3
    assert a3.y == 2
    
    # Test multiple fields at once
    a4 = a.set(x=10, y=20)
    assert a.x == 1
    assert a.y == 2
    assert a4.x == 10
    assert a4.y == 20
    
    # Test that original is unmodified
    assert a.x == 1
    assert a.y == 2
    
    # Test setting field that doesn't exist raises AttributeError
    with pytest.raises(AttributeError):
        a.set(z=5)
    
    # Test that returned object is same class
    assert isinstance(a2, AClass)
    assert isinstance(a3, AClass)
    assert isinstance(a4, AClass)
    
    # Test with optional fields
    class BClass(PClass):
        a = field()
        b = field()
        c = field()
    
    b = BClass(a=1)
    b2 = b.set(b=2, c=3)
    assert b2.a == 1
    assert b2.b == 2
    assert b2.c == 3


# LLM-generated content at query #4
#--------------------------

```python
def test__PClassEvolver_remove():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    # Test successful removal of an existing item
    instance = TestClass(x=1, y=2, z=3)
    evolver = instance.evolver()
    
    result = evolver.remove('x')
    assert result is evolver  # Should return self
    assert 'x' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'x' in evolver._factory_fields
    
    # Test removal of another field
    evolver.remove('y')
    assert 'y' not in evolver._pclass_evolver_data
    assert 'y' in evolver._factory_fields
    
    # Test removal of non-existent item raises AttributeError
    with pytest.raises(AttributeError) as exc_info:
        evolver.remove('nonexistent')
    assert 'nonexistent' in str(exc_info.value)
    
    # Test removal of already removed item raises AttributeError
    with pytest.raises(AttributeError):
        evolver.remove('x')
    
    # Test persistent() after removal returns new instance without removed fields
    persistent_instance = evolver.persistent()
    assert hasattr(persistent_instance, 'z')
    assert not hasattr(persistent_instance, 'x')
    assert not hasattr(persistent_instance, 'y')
    
    # Test that original instance is unchanged
    assert hasattr(instance, 'x')
    assert instance.x == 1
    
    # Test __delitem__ calls remove
    instance2 = TestClass(x=10, y=20)
    evolver2 = instance2.evolver()
    del evolver2['x']
    assert 'x' not in evolver2._pclass_evolver_data
    assert evolver2._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #5
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    # Test basic set with kwargs
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    obj2 = obj.set(x=10)
    
    assert obj.x == 1
    assert obj.y == 2
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj is not obj2
    
    # Test set with positional args (field name, value)
    obj3 = obj.set('y', 20)
    assert obj.y == 2
    assert obj3.y == 20
    assert obj3.x == 1
    
    # Test set with multiple kwargs
    obj4 = obj.set(x=100, y=200)
    assert obj4.x == 100
    assert obj4.y == 200
    assert obj.x == 1
    assert obj.y == 2
    
    # Test set returns same class type
    assert isinstance(obj2, TestClass)
    assert isinstance(obj3, TestClass)
    assert isinstance(obj4, TestClass)
    
    # Test set with field having initial value
    class TestClassWithInitial(PClass):
        a = field(initial=5)
        b = field()
    
    obj5 = TestClassWithInitial(b=10)
    obj6 = obj5.set(a=15)
    assert obj5.a == 5
    assert obj6.a == 15
    assert obj6.b == 10
    
    # Test set preserves frozen state
    obj7 = obj2.set(x=999)
    try:
        obj7.x = 1000
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass
    
    # Test set with same value
    obj8 = obj.set(x=1)
    assert obj8.x == 1
    assert obj8.y == 2
    
    # Test positional args takes precedence in kwargs
    obj9 = obj.set('x', 50, x=60)
    assert obj9.x == 50


# LLM-generated content at query #6
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3
    
    # Test inequality with different x value
    obj4 = TestClass(x=2, y=2)
    assert obj1 != obj4
    
    # Test equality with same object
    assert obj1 == obj1
    
    # Test inequality with different class
    class AnotherClass(PClass):
        x = field()
        y = field()
    
    obj5 = AnotherClass(x=1, y=2)
    assert (obj1 == obj5) is NotImplemented or obj1 != obj5
    
    # Test inequality with non-PClass object
    assert (obj1 == "not a pclass") is NotImplemented or obj1 != "not a pclass"
    assert (obj1 == 42) is NotImplemented or obj1 != 42
    assert (obj1 == None) is NotImplemented or obj1 != None
    
    # Test with missing fields
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj6 = OptionalClass(x=1)
    obj7 = OptionalClass(x=1, y=None)
    assert obj6 == obj7
    
    obj8 = OptionalClass(x=1, y=2)
    assert obj6 != obj8
    
    # Test with nested PClass
    class InnerClass(PClass):
        a = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner1 = InnerClass(a=1)
    inner2 = InnerClass(a=1)
    outer1 = OuterClass(inner=inner1)
    outer2 = OuterClass(inner=inner2)
    assert outer1 == outer2
    
    inner3 = InnerClass(a=2)
    outer3 = OuterClass(inner=inner3)
    assert outer1 != outer3


# LLM-generated content at query #7
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field
    import pickle
    
    class TestPClass(PClass):
        x = field()
        y = field()
        z = field(initial=42)
    
    # Test basic __reduce__ functionality
    obj = TestPClass(x=1, y="hello")
    reduced = obj.__reduce__()
    
    # __reduce__ should return a tuple of (callable, args)
    assert isinstance(reduced, tuple)
    assert len(reduced) == 2
    
    restore_func, args = reduced
    assert callable(restore_func)
    assert isinstance(args, tuple)
    assert len(args) == 2
    
    # First arg should be the class, second should be the data dict
    restored_class, data_dict = args
    assert restored_class is TestPClass
    assert isinstance(data_dict, dict)
    
    # Data dict should contain the fields that were set
    assert data_dict['x'] == 1
    assert data_dict['y'] == "hello"
    assert data_dict['z'] == 42
    
    # Test that pickling and unpickling works
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    assert unpickled == obj
    assert unpickled.x == 1
    assert unpickled.y == "hello"
    assert unpickled.z == 42
    
    # Test with only some fields set
    obj2 = TestPClass(x=10, y=20)
    reduced2 = obj2.__reduce__()
    restored_class2, data_dict2 = reduced2
    
    assert restored_class2 is TestPClass
    assert data_dict2['x'] == 10
    assert data_dict2['y'] == 20
    assert data_dict2['z'] == 42
    
    # Test pickling round-trip
    pickled2 = pickle.dumps(obj2)
    unpickled2 = pickle.loads(pickled2)
    assert unpickled2 == obj2
    
    # Test with complex nested values
    class NestedPClass(PClass):
        name = field()
        value = field()
    
    nested_obj = NestedPClass(name="test", value=[1, 2, 3])
    reduced_nested = nested_obj.__reduce__()
    restored_class_nested, data_dict_nested = reduced_nested
    
    assert restored_class_nested is NestedPClass
    assert data_dict_nested['name'] == "test"
    assert data_dict_nested['value'] == [1, 2, 3]
    
    pickled_nested = pickle.dumps(nested_obj)
    unpickled_nested = pickle.loads(pickled_nested)
    assert unpickled_nested == nested_obj


# LLM-generated content at query #8
#--------------------------

```python
def test_PClassMeta___new__():
    """Test PClassMeta.__new__ creates proper class structure with fields and slots"""
    from pyrsistent import PClass, field
    
    # Test basic class creation
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Verify _pclass_fields are set
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    
    # Verify __slots__ are properly configured
    assert hasattr(TestClass, '__slots__')
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'x' in TestClass.__slots__
    assert 'y' in TestClass.__slots__
    assert '__weakref__' in TestClass.__slots__
    
    # Test that instances can be created
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2
    
    # Test with invariants
    class TestClassWithInvariant(PClass):
        value = field()
        
        @staticmethod
        def __invariant__(obj):
            assert obj.value > 0, "Value must be positive"
    
    assert hasattr(TestClassWithInvariant, '_pclass_invariants')
    
    # Test nested inheritance
    class BaseClass(PClass):
        a = field()
    
    class DerivedClass(BaseClass):
        b = field()
    
    assert 'a' in DerivedClass._pclass_fields
    assert 'b' in DerivedClass._pclass_fields
    assert 'a' in DerivedClass.__slots__
    assert 'b' in DerivedClass.__slots__
    
    # Test __weakref__ only in top-level PClass
    derived_instance = DerivedClass(a=1, b=2)
    assert hasattr(derived_instance, '__weakref__')
    
    # Verify metaclass is PClassMeta
    assert type(TestClass) is PClassMeta


# LLM-generated content at query #9
#--------------------------

```python
def test_PClass___reduce__():
    """Test pickling support for PClass via __reduce__ method."""
    from pyrsistent import PClass, field
    import pickle
    
    # Test basic PClass pickling
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y=2)
    reduce_result = obj.__reduce__()
    
    # Verify __reduce__ returns the expected format
    assert len(reduce_result) == 2
    assert reduce_result[0] == _restore_pickle
    assert len(reduce_result[1]) == 2
    assert reduce_result[1][0] == SimpleClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}
    
    # Test that pickling and unpickling works correctly
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj
    assert unpickled.x == 1
    assert unpickled.y == 2


def test_PClass___reduce__with_missing_optional_fields():
    """Test __reduce__ with optional fields that are not set."""
    from pyrsistent import PClass, field
    import pickle
    
    class OptionalFieldClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj = OptionalFieldClass(x=10)
    reduce_result = obj.__reduce__()
    
    # Only set fields should be in the data
    assert reduce_result[1][1] == {'x': 10, 'y': None}
    
    # Verify round-trip
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj
    assert unpickled.x == 10
    assert unpickled.y is None


def test_PClass___reduce__with_complex_types():
    """Test __reduce__ with complex field types."""
    from pyrsistent import PClass, field
    import pickle
    
    class ComplexClass(PClass):
        name = field()
        values = field()
        mapping = field()
    
    obj = ComplexClass(
        name="test",
        values=[1, 2, 3],
        mapping={'a': 1, 'b': 2}
    )
    reduce_result = obj.__reduce__()
    
    assert reduce_result[0] == _restore_pickle
    assert reduce_result[1][0] == ComplexClass
    assert reduce_result[1][1]['name'] == "test"
    assert reduce_result[1][1]['values'] == [1, 2, 3]
    assert reduce_result[1][1]['mapping'] == {'a': 1, 'b': 2}
    
    # Verify round-trip
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj


def test_PClass___reduce__empty_class():
    """Test __reduce__ with a PClass that has no fields set."""
    from pyrsistent import PClass, field
    import pickle
    
    class EmptyClass(PClass):
        x = field(initial=0)
    
    obj = EmptyClass()
    reduce_result = obj.__reduce__()
    
    assert reduce_result[1][0] == EmptyClass
    assert reduce_result[1][1] == {'x': 0}
    
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj


# LLM-generated content at query #10
#--------------------------

```python
def test_PClass___eq__():
    """Test PClass __eq__ method"""
    from pyrsistent import PClass, field
    
    # Define a test PClass
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality with identical values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3
    
    # Test inequality with different field values
    obj4 = TestClass(x=2, y=2)
    assert obj1 != obj4
    
    # Test equality with same instance
    assert obj1 == obj1
    
    # Test inequality with different types
    assert obj1 != {"x": 1, "y": 2}
    assert (obj1 == {"x": 1, "y": 2}) is NotImplemented or (obj1 != {"x": 1, "y": 2})
    
    # Test inequality with string
    assert obj1 != "not a PClass"
    
    # Test with optional fields
    class TestClassOptional(PClass):
        x = field()
        y = field(initial=10)
    
    obj5 = TestClassOptional(x=1)
    obj6 = TestClassOptional(x=1, y=10)
    assert obj5 == obj6
    
    obj7 = TestClassOptional(x=1, y=20)
    assert obj5 != obj7
    
    # Test inequality with different PClass subclasses
    class AnotherTestClass(PClass):
        x = field()
        y = field()
    
    obj8 = AnotherTestClass(x=1, y=2)
    assert obj1 != obj8
    
    # Test with missing optional fields
    class TestClassMissing(PClass):
        x = field()
        y = field(initial=None)
    
    obj9 = TestClassMissing(x=1)
    obj10 = TestClassMissing(x=1, y=None)
    assert obj9 == obj10
    
    # Test ne method
    assert not (obj1 != obj2)
    assert obj1 != obj3


# LLM-generated content at query #11
#--------------------------

```python
def test_PClass___reduce__():
    """Test pickling support via __reduce__ method"""
    import pickle
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Create an instance
    obj = TestClass(x=1, y=2)
    
    # Call __reduce__
    reduce_result = obj.__reduce__()
    
    # __reduce__ should return a tuple of (callable, args)
    assert isinstance(reduce_result, tuple)
    assert len(reduce_result) == 2
    
    restore_func, args = reduce_result
    
    # First element should be _restore_pickle
    assert restore_func is _restore_pickle
    
    # Second element should be a tuple of (class, data_dict)
    assert isinstance(args, tuple)
    assert len(args) == 2
    assert args[0] is TestClass
    assert isinstance(args[1], dict)
    assert args[1] == {'x': 1, 'y': 2}
    
    # Test that pickle can serialize and deserialize
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    assert unpickled == obj
    assert unpickled.x == 1
    assert unpickled.y == 2


def test_PClass___reduce__with_missing_fields():
    """Test __reduce__ only includes fields that are set"""
    import pickle
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj = TestClass(x=5)
    
    reduce_result = obj.__reduce__()
    restore_func, args = reduce_result
    
    data_dict = args[1]
    # Should only include x and y (y has initial value)
    assert 'x' in data_dict
    assert 'y' in data_dict
    assert data_dict['x'] == 5
    assert data_dict['y'] == 10
    
    # Verify pickle round-trip
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj


def test_PClass___reduce__empty_object():
    """Test __reduce__ with object having only default fields"""
    import pickle
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    
    obj = TestClass()
    
    reduce_result = obj.__reduce__()
    restore_func, args = reduce_result
    
    data_dict = args[1]
    assert data_dict == {'x': 1, 'y': 2}
    
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj


# LLM-generated content at query #12
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test basic serialization
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y="hello")
    serialized = obj.serialize()
    assert serialized == {'x': 1, 'y': "hello"}
    
    # Test serialization with missing fields
    class OptionalFieldClass(PClass):
        a = field()
        b = field(initial=None)
    
    obj2 = OptionalFieldClass(a=42)
    serialized2 = obj2.serialize()
    assert serialized2 == {'a': 42, 'b': None}
    
    # Test serialization with custom serializer
    class CustomSerializerClass(PClass):
        value = field(serializer=lambda x, fmt: x * 2)
    
    obj3 = CustomSerializerClass(value=5)
    serialized3 = obj3.serialize()
    assert serialized3 == {'value': 10}
    
    # Test serialization with format parameter
    class FormattedClass(PClass):
        data = field(serializer=lambda x, fmt: str(x) if fmt == 'str' else x)
    
    obj4 = FormattedClass(data=123)
    serialized4_default = obj4.serialize()
    serialized4_str = obj4.serialize(format='str')
    assert serialized4_default == {'data': 123}
    assert serialized4_str == {'data': '123'}
    
    # Test serialization with nested objects
    class NestedClass(PClass):
        inner = field()
    
    class OuterClass(PClass):
        nested = field()
        value = field()
    
    inner_obj = NestedClass(inner=10)
    outer_obj = OuterClass(nested=inner_obj, value=20)
    serialized5 = outer_obj.serialize()
    assert 'nested' in serialized5
    assert 'value' in serialized5
    assert serialized5['value'] == 20
    
    # Test empty serialization
    class EmptyClass(PClass):
        x = field(initial=None)
    
    obj6 = EmptyClass()
    serialized6 = obj6.serialize()
    assert serialized6 == {'x': None}


# LLM-generated content at query #13
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test basic hash functionality
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)
    
    # Same values should produce same hash
    assert hash(obj1) == hash(obj2)
    
    # Different values should (likely) produce different hash
    assert hash(obj1) != hash(obj3)
    
    # Hash should be consistent across multiple calls
    hash1 = hash(obj1)
    hash2 = hash(obj1)
    assert hash1 == hash2
    
    # Hash should work with sets
    s = {obj1, obj2, obj3}
    assert len(s) == 2  # obj1 and obj2 are equal, so only 2 unique items
    
    # Hash should work with dicts
    d = {obj1: 'value1', obj3: 'value3'}
    assert d[obj2] == 'value1'  # obj2 should map to same key as obj1
    
    # Test with missing values
    class TestClassOptional(PClass):
        x = field()
        y = field(initial=None)
    
    obj4 = TestClassOptional(x=1)
    obj5 = TestClassOptional(x=1, y=None)
    
    # Both should be hashable
    assert isinstance(hash(obj4), int)
    assert isinstance(hash(obj5), int)
    
    # Test hash with nested structures
    class Inner(PClass):
        a = field()
    
    class Outer(PClass):
        inner = field()
        b = field()
    
    inner1 = Inner(a=1)
    inner2 = Inner(a=1)
    outer1 = Outer(inner=inner1, b=2)
    outer2 = Outer(inner=inner2, b=2)
    
    # Nested objects with same values should have same hash
    assert hash(outer1) == hash(outer2)


# LLM-generated content at query #14
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    class AClass(PClass):
        x = field()
        y = field()
    
    # Test basic set with keyword argument
    a = AClass(x=1, y=2)
    a2 = a.set(x=2)
    assert a2.x == 2
    assert a2.y == 2
    assert a.x == 1
    assert a.y == 2
    
    # Test set with positional arguments (field name and value)
    a3 = a.set('x', 3)
    assert a3.x == 3
    assert a3.y == 2
    assert a.x == 1
    
    # Test set with multiple keyword arguments
    a4 = a.set(x=10, y=20)
    assert a4.x == 10
    assert a4.y == 20
    assert a.x == 1
    assert a.y == 2
    
    # Test that set returns a new instance
    assert a is not a2
    assert a is not a3
    assert a is not a4
    
    # Test set preserves other fields
    class BClass(PClass):
        a = field()
        b = field()
        c = field()
    
    b = BClass(a=1, b=2, c=3)
    b2 = b.set(b=100)
    assert b2.a == 1
    assert b2.b == 100
    assert b2.c == 3
    
    # Test set with optional fields
    class CClass(PClass):
        x = field()
        y = field(initial=5)
    
    c = CClass(x=1)
    c2 = c.set(y=10)
    assert c2.x == 1
    assert c2.y == 10
    assert c.y == 5
    
    # Test set with field having no initial value set
    c3 = c.set(x=2)
    assert c3.x == 2
    assert c3.y == 5


# LLM-generated content at query #15
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test basic serialization without custom serializer
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y=2)
    result = obj.serialize()
    assert result == {'x': 1, 'y': 2}
    
    # Test serialization with missing optional fields
    class OptionalClass(PClass):
        a = field()
        b = field(initial=None)
    
    obj2 = OptionalClass(a=10)
    result2 = obj2.serialize()
    assert result2 == {'a': 10, 'b': None}
    
    # Test serialization with custom serializer
    class CustomSerializerClass(PClass):
        value = field(serializer=lambda v, _: v * 2)
        name = field()
    
    obj3 = CustomSerializerClass(value=5, name="test")
    result3 = obj3.serialize()
    assert result3 == {'value': 10, 'name': "test"}
    
    # Test serialization with format parameter
    class FormattedClass(PClass):
        timestamp = field(serializer=lambda v, fmt: str(v) if fmt == 'string' else v)
    
    obj4 = FormattedClass(timestamp=12345)
    result4 = obj4.serialize(format='string')
    assert result4 == {'timestamp': '12345'}
    
    result5 = obj4.serialize(format='numeric')
    assert result5 == {'timestamp': 12345}
    
    # Test empty PClass serialization
    class EmptyClass(PClass):
        pass
    
    obj5 = EmptyClass()
    result6 = obj5.serialize()
    assert result6 == {}
    
    # Test serialization with nested structures
    class NestedClass(PClass):
        data = field()
    
    obj6 = NestedClass(data={'nested': 'value'})
    result7 = obj6.serialize()
    assert result7 == {'data': {'nested': 'value'}}


# LLM-generated content at query #16
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field
    
    # Test basic repr with single field
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=42)
    assert repr(obj) == "SimpleClass(x=42)"
    
    # Test repr with multiple fields
    class MultiFieldClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = MultiFieldClass(a=1, b="hello", c=3.14)
    repr_str = repr(obj)
    assert "MultiFieldClass" in repr_str
    assert "a=1" in repr_str
    assert "b='hello'" in repr_str
    assert "c=3.14" in repr_str
    
    # Test repr with string values (should show quotes)
    class StringClass(PClass):
        name = field()
    
    obj = StringClass(name="test")
    assert repr(obj) == "StringClass(name='test')"
    
    # Test repr with nested PClass
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner = InnerClass(value=10)
    outer = OuterClass(inner=inner)
    repr_str = repr(outer)
    assert "OuterClass" in repr_str
    assert "InnerClass(value=10)" in repr_str
    
    # Test repr with empty PClass (only optional fields)
    class OptionalFieldClass(PClass):
        x = field(initial=None)
    
    obj = OptionalFieldClass()
    assert repr(obj) == "OptionalFieldClass(x=None)"
    
    # Test repr with boolean and None values
    class MixedTypeClass(PClass):
        flag = field()
        nothing = field()
    
    obj = MixedTypeClass(flag=True, nothing=None)
    repr_str = repr(obj)
    assert "flag=True" in repr_str
    assert "nothing=None" in repr_str
    
    # Test repr with list/dict values
    class CollectionClass(PClass):
        items = field()
    
    obj = CollectionClass(items=[1, 2, 3])
    assert "items=[1, 2, 3]" in repr(obj)


# LLM-generated content at query #17
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test basic serialization without custom serializers
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y='hello')
    result = obj.serialize()
    assert result == {'x': 1, 'y': 'hello'}
    
    # Test serialization with missing optional fields
    class PartialClass(PClass):
        a = field()
        b = field(initial=None)
    
    obj2 = PartialClass(a=42)
    result2 = obj2.serialize()
    assert result2 == {'a': 42, 'b': None}
    
    # Test serialization with custom serializer
    class CustomSerializerClass(PClass):
        value = field(serializer=lambda v, f: v * 2 if v else None)
    
    obj3 = CustomSerializerClass(value=5)
    result3 = obj3.serialize()
    assert result3 == {'value': 10}
    
    # Test serialization with format parameter
    class FormatClass(PClass):
        data = field(serializer=lambda v, f: f"{v}_{f}" if f else str(v))
    
    obj4 = FormatClass(data='test')
    result4 = obj4.serialize(format='json')
    assert result4 == {'data': 'test_json'}
    
    # Test serialization with nested objects
    class NestedClass(PClass):
        inner = field()
    
    class OuterClass(PClass):
        nested = field()
        value = field()
    
    inner = NestedClass(inner=99)
    outer = OuterClass(nested=inner, value=10)
    result5 = outer.serialize()
    assert result5['value'] == 10
    assert result5['nested'] == inner
    
    # Test empty PClass serialization
    class EmptyClass(PClass):
        pass
    
    obj6 = EmptyClass()
    result6 = obj6.serialize()
    assert result6 == {}
    
    # Test serialization with None values
    class NullableClass(PClass):
        nullable = field(initial=None)
        required = field()
    
    obj7 = NullableClass(required='value')
    result7 = obj7.serialize()
    assert result7 == {'nullable': None, 'required': 'value'}


# LLM-generated content at query #18
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test that hash is computed from fields
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert hash(obj1) == hash(obj2)
    
    # Test that different values produce different hashes
    obj3 = TestClass(x=1, y=3)
    assert hash(obj1) != hash(obj3)
    
    # Test that hash is consistent across multiple calls
    hash1 = hash(obj1)
    hash2 = hash(obj1)
    assert hash1 == hash2
    
    # Test with missing optional fields
    class TestClassOptional(PClass):
        x = field()
        y = field(initial=None)
    
    obj4 = TestClassOptional(x=1)
    obj5 = TestClassOptional(x=1, y=None)
    assert hash(obj4) == hash(obj5)
    
    # Test hashability allows use in sets and dicts
    obj_set = {obj1, obj2, obj3}
    assert len(obj_set) == 2  # obj1 and obj2 have same hash and are equal
    
    obj_dict = {obj1: 'value1', obj3: 'value2'}
    assert len(obj_dict) == 2
    assert obj_dict[obj2] == 'value1'  # obj2 should map to same key as obj1
    
    # Test with nested field values
    class Inner(PClass):
        value = field()
    
    class Outer(PClass):
        inner = field()
    
    inner1 = Inner(value=42)
    inner2 = Inner(value=42)
    outer1 = Outer(inner=inner1)
    outer2 = Outer(inner=inner2)
    assert hash(outer1) == hash(outer2)


# LLM-generated content at query #19
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field(initial=10)
    
    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    obj2 = obj.set(x=5)
    assert obj.x == 1
    assert obj2.x == 5
    assert obj2.y == 2
    assert obj2.z == 10
    
    # Test set with positional arguments (field name and value)
    obj3 = obj.set('y', 20)
    assert obj.y == 2
    assert obj3.y == 20
    assert obj3.x == 1
    
    # Test multiple fields at once
    obj4 = obj.set(x=100, y=200)
    assert obj4.x == 100
    assert obj4.y == 200
    assert obj4.z == 10
    
    # Test that original object is not modified
    assert obj.x == 1
    assert obj.y == 2
    
    # Test set returns a new instance
    assert obj is not obj2
    assert obj is not obj3
    assert obj is not obj4
    
    # Test that the returned object is of the same class
    assert isinstance(obj2, TestClass)
    assert isinstance(obj3, TestClass)
    assert isinstance(obj4, TestClass)
    
    # Test set with field that has initial value
    obj5 = obj.set(z=50)
    assert obj5.z == 50
    assert obj.z == 10
    
    # Test set preserves all fields
    obj6 = obj.set(x=7)
    assert hasattr(obj6, 'x')
    assert hasattr(obj6, 'y')
    assert hasattr(obj6, 'z')


# LLM-generated content at query #20
#--------------------------

```python
def test_PClass___new__():
    """Test PClass.__new__ method"""
    from pyrsistent import PClass, field
    
    # Test basic instantiation with valid fields
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2
    assert obj._pclass_frozen is True
    
    # Test with single field
    class SingleField(PClass):
        name = field()
    
    obj2 = SingleField(name="test")
    assert obj2.name == "test"
    
    # Test with mandatory field missing raises InvariantException
    class MandatoryField(PClass):
        required = field(mandatory=True)
    
    with pytest.raises(InvariantException) as exc_info:
        MandatoryField()
    assert 'MandatoryField.required' in str(exc_info.value)
    
    # Test with initial value
    class WithInitial(PClass):
        value = field(initial=42)
    
    obj3 = WithInitial()
    assert obj3.value == 42
    
    # Test with callable initial value
    class WithCallableInitial(PClass):
        items = field(initial=list)
    
    obj4 = WithCallableInitial()
    assert obj4.items == []
    
    # Test extra kwargs raise AttributeError
    class NoExtra(PClass):
        x = field()
    
    with pytest.raises(AttributeError) as exc_info:
        NoExtra(x=1, z=2)
    assert "z" in str(exc_info.value)
    assert "not among the specified fields" in str(exc_info.value)
    
    # Test ignore_extra parameter
    obj5 = NoExtra.create({'x': 1, 'z': 2}, ignore_extra=True)
    assert obj5.x == 1
    
    # Test field factory is called
    class WithFactory(PClass):
        items = field(factory=list)
    
    obj6 = WithFactory(items=(1, 2, 3))
    assert obj6.items == [1, 2, 3]
    
    # Test field type checking
    class TypeChecked(PClass):
        num = field(type=int)
    
    with pytest.raises(TypeError):
        TypeChecked(num="not an int")
    
    # Test object is frozen after creation
    class Frozen(PClass):
        x = field()
    
    obj7 = Frozen(x=1)
    with pytest.raises(AttributeError) as exc_info:
        obj7.x = 2
    assert "Can't set attribute" in str(exc_info.value)
    
    # Test multiple fields with mixed initial and mandatory
    class Mixed(PClass):
        required = field(mandatory=True)
        optional = field(initial=10)
        with_default = field(initial="default")
    
    obj8 = Mixed(required="value")
    assert obj8.required == "value"
    assert obj8.optional == 10
    assert obj8.with_default == "default"
    
    # Test field invariant violations
    class WithInvariant(PClass):
        positive = field(invariant=lambda x: (x > 0, "must be positive"))
    
    with pytest.raises(InvariantException):
        WithInvariant(positive=-1)
    
    obj9 = WithInvariant(positive=5)
    assert obj9.positive == 5
    
    # Test _factory_fields parameter
    class FactoryFieldsTest(PClass):
        x = field(factory=int)
        y = field()
    
    obj10 = FactoryFieldsTest(_factory_fields={'x'}, x="42", y="hello")
    assert obj10.x == 42
    assert obj10.y == "hello"


# LLM-generated content at query #21
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    class AClass(PClass):
        x = field()
        y = field()
    
    # Test basic set with keyword arguments
    a = AClass(x=1, y=2)
    a2 = a.set(x=2)
    assert a.x == 1
    assert a.y == 2
    assert a2.x == 2
    assert a2.y == 2
    
    # Test set with positional arguments (field name and value)
    a3 = a.set('x', 3)
    assert a.x == 1
    assert a3.x == 3
    assert a3.y == 2
    
    # Test set with multiple keyword arguments
    a4 = a.set(x=4, y=5)
    assert a.x == 1
    assert a.y == 2
    assert a4.x == 4
    assert a4.y == 5
    
    # Test that original instance is unchanged
    assert a.x == 1
    assert a.y == 2
    
    # Test set returns new instance
    assert a is not a2
    assert a is not a3
    assert a is not a4
    
    # Test set with optional field
    class BClass(PClass):
        a = field()
        b = field(initial=10)
    
    b = BClass(a=1)
    b2 = b.set(a=2)
    assert b.a == 1
    assert b.b == 10
    assert b2.a == 2
    assert b2.b == 10
    
    # Test set with factory fields
    class CClass(PClass):
        items = field()
    
    c = CClass(items=[1, 2, 3])
    c2 = c.set(items=[4, 5, 6])
    assert c.items == [1, 2, 3]
    assert c2.items == [4, 5, 6]
    
    # Test that set preserves all fields
    class DClass(PClass):
        x = field()
        y = field()
        z = field()
    
    d = DClass(x=1, y=2, z=3)
    d2 = d.set(x=10)
    assert d2.x == 10
    assert d2.y == 2
    assert d2.z == 3


# LLM-generated content at query #22
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality between two instances with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality when field values differ
    obj3 = TestClass(x=1, y=3)
    assert not (obj1 == obj3)
    
    # Test inequality when different number of fields are set
    class TestClass2(PClass):
        x = field()
        y = field(initial=10)
    
    obj4 = TestClass2(x=1)
    obj5 = TestClass2(x=1, y=10)
    assert obj4 == obj5
    
    # Test inequality with different class types
    class OtherClass(PClass):
        x = field()
        y = field()
    
    obj6 = OtherClass(x=1, y=2)
    assert not (obj1 == obj6)
    
    # Test comparison with non-PClass object
    assert (obj1 == "not a pclass") == NotImplemented
    assert (obj1 == 42) == NotImplemented
    assert (obj1 == None) == NotImplemented
    
    # Test with missing values
    class TestClass3(PClass):
        x = field()
        y = field(initial=None)
    
    obj7 = TestClass3(x=1)
    obj8 = TestClass3(x=1, y=None)
    assert obj7 == obj8
    
    # Test self equality
    assert obj1 == obj1
    
    # Test with complex field values
    class TestClass4(PClass):
        data = field()
    
    obj9 = TestClass4(data=[1, 2, 3])
    obj10 = TestClass4(data=[1, 2, 3])
    assert obj9 == obj10
    
    obj11 = TestClass4(data=[1, 2, 4])
    assert not (obj9 == obj11)


# LLM-generated content at query #23
#--------------------------

```python
def test_PClass_set():
    """Test PClass.set() method for setting field values and returning new instances."""
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field(initial=0)
    
    # Test basic set with keyword argument
    obj1 = TestClass(x=1, y=2)
    obj2 = obj1.set(x=10)
    assert obj1.x == 1
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj2.z == 0
    
    # Test set with positional arguments
    obj3 = obj1.set('y', 20)
    assert obj1.y == 2
    assert obj3.y == 20
    assert obj3.x == 1
    
    # Test set multiple fields at once
    obj4 = obj1.set(x=100, y=200)
    assert obj1.x == 1
    assert obj1.y == 2
    assert obj4.x == 100
    assert obj4.y == 200
    
    # Test that original object is unmodified
    original_x = obj1.x
    original_y = obj1.y
    obj1.set(x=999, y=999)
    assert obj1.x == original_x
    assert obj1.y == original_y
    
    # Test set returns new instance
    obj5 = obj1.set(x=5)
    assert obj5 is not obj1
    assert isinstance(obj5, TestClass)
    
    # Test set with all fields
    obj6 = obj1.set(x=11, y=22, z=33)
    assert obj6.x == 11
    assert obj6.y == 22
    assert obj6.z == 33
    
    # Test set preserves fields not being changed
    obj7 = TestClass(x=1, y=2, z=3)
    obj8 = obj7.set(x=100)
    assert obj8.y == 2
    assert obj8.z == 3


# LLM-generated content at query #24
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test basic serialization without custom serializer
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y=2)
    result = obj.serialize()
    assert result == {'x': 1, 'y': 2}
    
    # Test serialization with missing optional fields
    class PartialClass(PClass):
        a = field()
        b = field(initial=None)
    
    obj2 = PartialClass(a=10)
    result2 = obj2.serialize()
    assert result2 == {'a': 10, 'b': None}
    
    # Test serialization with custom serializer function
    def custom_serializer(value, format=None):
        return str(value).upper() if isinstance(value, str) else value
    
    class CustomSerializerClass(PClass):
        name = field(serializer=custom_serializer)
        count = field()
    
    obj3 = CustomSerializerClass(name='hello', count=5)
    result3 = obj3.serialize()
    assert result3 == {'name': 'HELLO', 'count': 5}
    
    # Test serialization with format parameter
    def format_aware_serializer(value, format=None):
        if format == 'json':
            return str(value)
        return value
    
    class FormatAwareClass(PClass):
        data = field(serializer=format_aware_serializer)
    
    obj4 = FormatAwareClass(data=42)
    result4 = obj4.serialize(format='json')
    assert result4 == {'data': '42'}
    
    # Test empty PClass
    class EmptyClass(PClass):
        pass
    
    obj5 = EmptyClass()
    result5 = obj5.serialize()
    assert result5 == {}
    
    # Test serialization preserves nested structures
    class NestedClass(PClass):
        items = field()
    
    obj6 = NestedClass(items=[1, 2, 3])
    result6 = obj6.serialize()
    assert result6 == {'items': [1, 2, 3]}


# LLM-generated content at query #25
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality with identical values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert not (obj1 == obj3)
    
    # Test inequality when comparing with different class
    class OtherClass(PClass):
        x = field()
        y = field()
    
    obj4 = OtherClass(x=1, y=2)
    assert (obj1 == obj4) is NotImplemented
    
    # Test equality with missing fields
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj5 = OptionalClass(x=1)
    obj6 = OptionalClass(x=1, y=None)
    assert obj5 == obj6
    
    # Test inequality with different missing fields
    obj7 = OptionalClass(x=2)
    assert not (obj5 == obj7)
    
    # Test equality with None values
    obj8 = TestClass(x=None, y=None)
    obj9 = TestClass(x=None, y=None)
    assert obj8 == obj9
    
    # Test inequality with one having None and other having value
    obj10 = TestClass(x=1, y=None)
    obj11 = TestClass(x=1, y=2)
    assert not (obj10 == obj11)
    
    # Test self equality
    assert obj1 == obj1
    
    # Test with complex nested values
    obj12 = TestClass(x=[1, 2, 3], y={'a': 1})
    obj13 = TestClass(x=[1, 2, 3], y={'a': 1})
    assert obj12 == obj13
    
    # Test with different complex values
    obj14 = TestClass(x=[1, 2, 3], y={'a': 2})
    assert not (obj12 == obj14)
    
    # Test comparison with non-PClass object
    assert (obj1 == "not a pclass") is NotImplemented
    assert (obj1 == 42) is NotImplemented
    assert (obj1 == None) is NotImplemented


# LLM-generated content at query #26
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    # Test basic set with keyword argument
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    obj2 = obj.set(x=10)
    
    assert obj.x == 1
    assert obj.y == 2
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj is not obj2
    
    # Test set with positional arguments (field name and value)
    obj3 = obj.set('y', 20)
    assert obj.y == 2
    assert obj3.y == 20
    assert obj3.x == 1
    
    # Test set with multiple keyword arguments
    obj4 = obj.set(x=100, y=200)
    assert obj.x == 1
    assert obj.y == 2
    assert obj4.x == 100
    assert obj4.y == 200
    
    # Test that original object is unmodified
    assert obj.x == 1
    assert obj.y == 2
    
    # Test set on object with optional fields
    class TestClass2(PClass):
        a = field()
        b = field(initial=5)
    
    obj5 = TestClass2(a=1)
    obj6 = obj5.set(a=2)
    
    assert obj5.a == 1
    assert obj5.b == 5
    assert obj6.a == 2
    assert obj6.b == 5
    
    # Test set preserves fields not being modified
    obj7 = obj5.set(b=10)
    assert obj7.a == 1
    assert obj7.b == 10
    
    # Test set with no arguments returns new instance
    obj8 = obj.set()
    assert obj8.x == obj.x
    assert obj8.y == obj.y
    assert obj8 is not obj


# LLM-generated content at query #27
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test basic serialization
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.serialize()
    assert result == {'x': 1, 'y': 2}
    
    # Test serialization with missing optional field
    class TestClassOptional(PClass):
        x = field()
        y = field(initial=None)
    
    obj2 = TestClassOptional(x=5)
    result2 = obj2.serialize()
    assert result2 == {'x': 5, 'y': None}
    
    # Test serialization with custom serializer
    def custom_serializer(format, value):
        return str(value).upper()
    
    class TestClassCustom(PClass):
        name = field(serializer=custom_serializer)
    
    obj3 = TestClassCustom(name='hello')
    result3 = obj3.serialize()
    assert result3 == {'name': 'HELLO'}
    
    # Test serialization with format parameter
    def format_aware_serializer(format, value):
        if format == 'json':
            return str(value)
        return value
    
    class TestClassFormat(PClass):
        data = field(serializer=format_aware_serializer)
    
    obj4 = TestClassFormat(data=42)
    result4 = obj4.serialize(format='json')
    assert result4 == {'data': '42'}
    
    result5 = obj4.serialize(format='other')
    assert result5 == {'data': 42}
    
    # Test serialization with nested PClass
    class Inner(PClass):
        value = field()
    
    class Outer(PClass):
        inner = field()
    
    inner_obj = Inner(value=10)
    outer_obj = Outer(inner=inner_obj)
    result6 = outer_obj.serialize()
    assert result6 == {'inner': inner_obj}
    
    # Test empty PClass serialization
    class EmptyClass(PClass):
        pass
    
    empty_obj = EmptyClass()
    result7 = empty_obj.serialize()
    assert result7 == {}


# LLM-generated content at query #28
#--------------------------

```python
def test_PClassMeta___new__():
    """Test PClassMeta.__new__ creates class with proper fields and slots."""
    from pyrsistent import PClass, field
    
    # Test basic class creation
    class TestClass(PClass):
        x = field()
        y = field()
    
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    
    # Test slots are created correctly
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'x' in TestClass.__slots__
    assert 'y' in TestClass.__slots__
    
    # Test __weakref__ is added to base PClass
    assert '__weakref__' in TestClass.__slots__
    
    # Test that instances cannot have arbitrary attributes due to slots
    instance = TestClass(x=1, y=2)
    with pytest.raises(AttributeError):
        instance.z = 3
    
    # Test invariants are stored
    assert hasattr(TestClass, '_pclass_invariants')
    
    # Test class with inheritance
    class BaseClass(PClass):
        a = field()
    
    class DerivedClass(BaseClass):
        b = field()
    
    assert 'a' in DerivedClass._pclass_fields
    assert 'b' in DerivedClass._pclass_fields
    assert '_pclass_frozen' in DerivedClass.__slots__
    assert 'b' in DerivedClass.__slots__
    
    # Test that __weakref__ is only added once in hierarchy
    assert DerivedClass.__slots__.count('__weakref__') == 1
    
    # Test multiple fields
    class ComplexClass(PClass):
        field1 = field()
        field2 = field()
        field3 = field()
    
    assert len(ComplexClass._pclass_fields) == 3
    assert all(f in ComplexClass.__slots__ for f in ['field1', 'field2', 'field3', '_pclass_frozen'])


# LLM-generated content at query #29
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field
    
    # Test basic hash functionality
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)
    
    # Same values should produce same hash
    assert hash(obj1) == hash(obj2)
    
    # Different values should (likely) produce different hashes
    assert hash(obj1) != hash(obj3)
    
    # Hash should be consistent across multiple calls
    hash1 = hash(obj1)
    hash2 = hash(obj1)
    assert hash1 == hash2
    
    # Test with optional fields
    class TestClassOptional(PClass):
        a = field()
        b = field(initial=None)
    
    obj4 = TestClassOptional(a=1)
    obj5 = TestClassOptional(a=1, b=None)
    
    # Objects with same effective state should have same hash
    assert hash(obj4) == hash(obj5)
    
    # Test with missing values
    class TestClassMissing(PClass):
        name = field()
        value = field(initial=10)
    
    obj6 = TestClassMissing(name='test')
    obj7 = TestClassMissing(name='test', value=10)
    
    assert hash(obj6) == hash(obj7)
    
    # Test that hashed objects can be used in sets and dicts
    obj_set = {obj1, obj2, obj3}
    assert len(obj_set) == 2  # obj1 and obj2 should be considered same in set
    
    obj_dict = {obj1: 'first', obj2: 'second'}
    assert len(obj_dict) == 1  # obj1 and obj2 should map to same key
    assert obj_dict[obj1] == 'second'
    
    # Test with nested structures
    class Inner(PClass):
        val = field()
    
    class Outer(PClass):
        inner = field()
    
    inner1 = Inner(val=5)
    inner2 = Inner(val=5)
    outer1 = Outer(inner=inner1)
    outer2 = Outer(inner=inner2)
    
    assert hash(outer1) == hash(outer2)
    
    # Test with different field types
    class MixedClass(PClass):
        integer = field()
        string = field()
        boolean = field()
    
    mixed1 = MixedClass(integer=42, string='hello', boolean=True)
    mixed2 = MixedClass(integer=42, string='hello', boolean=True)
    mixed3 = MixedClass(integer=42, string='hello', boolean=False)
    
    assert hash(mixed1) == hash(mixed2)
    assert hash(mixed1) != hash(mixed3)


# LLM-generated content at query #30
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field(initial=10)
    
    # Test basic set with kwargs
    obj = TestClass(x=1, y=2)
    obj2 = obj.set(x=5)
    assert obj.x == 1
    assert obj2.x == 5
    assert obj2.y == 2
    assert obj2.z == 10
    
    # Test set with args (field name and value)
    obj3 = obj.set('y', 20)
    assert obj.y == 2
    assert obj3.y == 20
    assert obj3.x == 1
    
    # Test set with multiple kwargs
    obj4 = obj.set(x=100, y=200)
    assert obj4.x == 100
    assert obj4.y == 200
    assert obj.x == 1
    assert obj.y == 2
    
    # Test that original object is unchanged
    original_x = obj.x
    obj.set(x=999)
    assert obj.x == original_x
    
    # Test set returns new instance
    obj5 = obj.set(x=50)
    assert obj5 is not obj
    assert isinstance(obj5, TestClass)
    
    # Test set with field that has initial value
    obj6 = obj.set(z=30)
    assert obj6.z == 30
    assert obj.z == 10
    
    # Test set with args takes precedence in kwargs
    obj7 = obj.set('x', 7)
    assert obj7.x == 7


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test basic serialization without custom serializer
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y=2)
    result = obj.serialize()
    assert result == {'x': 1, 'y': 2}
    
    # Test serialization with missing optional fields
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj2 = OptionalClass(x=10)
    result2 = obj2.serialize()
    assert result2 == {'x': 10, 'y': None}
    
    # Test serialization with custom serializer function
    def double_serializer(format, value):
        return value * 2 if value is not None else None
    
    class CustomSerializerClass(PClass):
        x = field(serializer=double_serializer)
        y = field()
    
    obj3 = CustomSerializerClass(x=5, y=10)
    result3 = obj3.serialize()
    assert result3 == {'x': 10, 'y': 10}
    
    # Test serialization with format parameter
    def format_aware_serializer(format, value):
        if format == 'json':
            return str(value)
        return value
    
    class FormatAwareClass(PClass):
        x = field(serializer=format_aware_serializer)
    
    obj4 = FormatAwareClass(x=42)
    result4_default = obj4.serialize()
    result4_json = obj4.serialize(format='json')
    assert result4_default == {'x': 42}
    assert result4_json == {'x': '42'}
    
    # Test serialization with nested PClass
    class InnerClass(PClass):
        a = field()
    
    class OuterClass(PClass):
        inner = field()
        b = field()
    
    inner = InnerClass(a=1)
    outer = OuterClass(inner=inner, b=2)
    result5 = outer.serialize()
    assert result5['b'] == 2
    assert isinstance(result5['inner'], InnerClass)
    
    # Test serialization with empty PClass
    class EmptyClass(PClass):
        pass
    
    obj6 = EmptyClass()
    result6 = obj6.serialize()
    assert result6 == {}
    
    # Test serialization preserves all fields
    class AllFieldsClass(PClass):
        field1 = field()
        field2 = field()
        field3 = field()
    
    obj7 = AllFieldsClass(field1='a', field2='b', field3='c')
    result7 = obj7.serialize()
    assert len(result7) == 3
    assert result7 == {'field1': 'a', 'field2': 'b', 'field3': 'c'}


# LLM-generated content at query #2
#--------------------------

def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality with identical values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3
    
    # Test inequality with different field values
    obj4 = TestClass(x=2, y=2)
    assert obj1 != obj4
    
    # Test equality with same instance
    assert obj1 == obj1
    
    # Test inequality with different classes
    class AnotherClass(PClass):
        x = field()
        y = field()
    
    obj5 = AnotherClass(x=1, y=2)
    assert obj1 != obj5
    
    # Test inequality with non-PClass object
    assert (obj1 == "not a pclass") == NotImplemented
    assert (obj1 == 42) == NotImplemented
    assert (obj1 == None) == NotImplemented
    
    # Test with missing optional fields
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj6 = OptionalClass(x=1)
    obj7 = OptionalClass(x=1, y=None)
    assert obj6 == obj7
    
    # Test with different optional field values
    obj8 = OptionalClass(x=1, y=5)
    assert obj6 != obj8
    
    # Test with nested PClass
    class InnerClass(PClass):
        a = field()
    
    class OuterClass(PClass):
        inner = field()
        b = field()
    
    inner1 = InnerClass(a=10)
    inner2 = InnerClass(a=10)
    outer1 = OuterClass(inner=inner1, b=20)
    outer2 = OuterClass(inner=inner2, b=20)
    assert outer1 == outer2
    
    # Test inequality with nested PClass
    inner3 = InnerClass(a=11)
    outer3 = OuterClass(inner=inner3, b=20)
    assert outer1 != outer3


# LLM-generated content at query #3
#--------------------------

```python
def test__PClassEvolver_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    # Create an instance
    obj = TestClass(x=1, y=2, z=3)
    evolver = obj.evolver()
    
    # Test setting a new value
    result = evolver.set('x', 10)
    assert result is evolver  # Should return self for chaining
    assert evolver._pclass_evolver_data['x'] == 10
    assert 'x' in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True
    
    # Test setting another field
    evolver.set('y', 20)
    assert evolver._pclass_evolver_data['y'] == 20
    assert 'y' in evolver._factory_fields
    
    # Test setting to the same value (should not mark as dirty if already set)
    evolver2 = obj.evolver()
    evolver2.set('x', 1)  # Same as original
    assert 'x' in evolver2._factory_fields
    assert evolver2._pclass_evolver_data_is_dirty is True
    
    # Test that persistent() creates new object with updated values
    persistent_obj = evolver.persistent()
    assert persistent_obj.x == 10
    assert persistent_obj.y == 20
    assert persistent_obj.z == 3
    assert persistent_obj is not obj
    
    # Test that original object is unchanged
    assert obj.x == 1
    assert obj.y == 2
    
    # Test setting with same value as current in evolver
    evolver3 = obj.evolver()
    evolver3.set('x', 1)
    evolver3.set('x', 1)  # Set to same value again
    assert evolver3._pclass_evolver_data_is_dirty is True
    assert 'x' in evolver3._factory_fields
    
    # Test chaining multiple sets
    evolver4 = obj.evolver()
    result = evolver4.set('x', 100).set('y', 200).set('z', 300)
    assert result is evolver4
    assert evolver4._pclass_evolver_data == {'x': 100, 'y': 200, 'z': 300}
    assert evolver4._factory_fields == {'x', 'y', 'z'}


# LLM-generated content at query #4
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3
    
    # Test inequality with different field values
    obj4 = TestClass(x=2, y=2)
    assert obj1 != obj4
    
    # Test equality with same object
    assert obj1 == obj1
    
    # Test inequality with different types
    assert (obj1 == "not a PClass") == NotImplemented
    assert (obj1 == 42) == NotImplemented
    assert (obj1 == None) == NotImplemented
    
    # Test inequality with different PClass types
    class AnotherClass(PClass):
        x = field()
        y = field()
    
    obj5 = AnotherClass(x=1, y=2)
    assert obj1 != obj5
    
    # Test with missing optional fields
    class OptionalFieldClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj6 = OptionalFieldClass(x=1)
    obj7 = OptionalFieldClass(x=1, y=None)
    assert obj6 == obj7
    
    obj8 = OptionalFieldClass(x=1, y=5)
    assert obj6 != obj8
    
    # Test with nested PClass
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner1 = InnerClass(value=10)
    inner2 = InnerClass(value=10)
    outer1 = OuterClass(inner=inner1)
    outer2 = OuterClass(inner=inner2)
    assert outer1 == outer2
    
    inner3 = InnerClass(value=20)
    outer3 = OuterClass(inner=inner3)
    assert outer1 != outer3


# LLM-generated content at query #5
#--------------------------

```python
def test__PClassEvolver_remove():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    # Test removing an existing item
    obj = TestClass(x=1, y=2, z=3)
    evolver = obj.evolver()
    
    result = evolver.remove('x')
    assert result is evolver  # Should return self for chaining
    assert 'x' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'x' in evolver._factory_fields
    
    # Test removing another item
    evolver.remove('y')
    assert 'y' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    
    # Verify z is still there
    assert 'z' in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data['z'] == 3
    
    # Test removing non-existent item raises AttributeError
    evolver2 = obj.evolver()
    try:
        evolver2.remove('nonexistent')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert str(e) == 'nonexistent'
    
    # Test that persistent() creates a new object without removed fields
    obj2 = TestClass(x=1, y=2, z=3)
    evolver3 = obj2.evolver()
    evolver3.remove('x')
    persistent_obj = evolver3.persistent()
    
    assert not hasattr(persistent_obj, 'x') or getattr(persistent_obj, 'x', None) is None
    assert persistent_obj.y == 2
    assert persistent_obj.z == 3
    
    # Test removing from empty evolver data
    obj3 = TestClass(x=1)
    evolver4 = obj3.evolver()
    evolver4.remove('x')
    assert 'x' not in evolver4._pclass_evolver_data
    assert evolver4._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #6
#--------------------------

```python
def test_PClass___new__():
    """Test PClass.__new__ method"""
    from pyrsistent import PClass, field
    
    # Test basic instantiation with valid fields
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2
    assert obj._pclass_frozen is True
    
    # Test with mandatory fields
    class MandatoryClass(PClass):
        required = field(mandatory=True)
        optional = field()
    
    obj = MandatoryClass(required=10)
    assert obj.required == 10
    assert not hasattr(obj, 'optional')
    
    # Test missing mandatory field raises InvariantException
    with pytest.raises(InvariantException) as exc_info:
        MandatoryClass()
    assert 'MandatoryClass.required' in exc_info.value.missing_fields
    
    # Test extra fields raise AttributeError
    with pytest.raises(AttributeError) as exc_info:
        SimpleClass(x=1, y=2, z=3)
    assert 'z' in str(exc_info.value)
    assert 'not among the specified fields' in str(exc_info.value)
    
    # Test with initial values
    class InitialClass(PClass):
        x = field(initial=5)
        y = field(initial=lambda: 10)
    
    obj = InitialClass()
    assert obj.x == 5
    assert obj.y == 10
    
    # Test with initial values overridden
    obj = InitialClass(x=20)
    assert obj.x == 20
    assert obj.y == 10
    
    # Test field type checking
    class TypedClass(PClass):
        x = field(type=int)
    
    obj = TypedClass(x=5)
    assert obj.x == 5
    
    with pytest.raises(TypeError):
        TypedClass(x="not an int")
    
    # Test field invariant
    class InvariantClass(PClass):
        x = field(invariant=lambda x: (x > 0, 'must be positive'))
    
    obj = InvariantClass(x=5)
    assert obj.x == 5
    
    with pytest.raises(InvariantException) as exc_info:
        InvariantClass(x=-1)
    assert 'must be positive' in exc_info.value.invariant_errors
    
    # Test factory fields parameter
    class FactoryClass(PClass):
        x = field()
    
    obj = FactoryClass(_factory_fields={'x'}, x=1)
    assert obj.x == 1
    
    # Test ignore_extra parameter
    obj = FactoryClass(x=1, ignore_extra=True, extra_field=999)
    assert obj.x == 1
    assert not hasattr(obj, 'extra_field')
    
    # Test multiple missing fields
    class MultiMandatoryClass(PClass):
        a = field(mandatory=True)
        b = field(mandatory=True)
        c = field(mandatory=True)
    
    with pytest.raises(InvariantException) as exc_info:
        MultiMandatoryClass()
    assert len(exc_info.value.missing_fields) == 3
    
    # Test instance is frozen after creation
    class FrozenClass(PClass):
        x = field()
    
    obj = FrozenClass(x=1)
    with pytest.raises(AttributeError) as exc_info:
        obj.x = 2
    assert "Can't set attribute" in str(exc_info.value)
    
    # Test with empty PClass
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    assert obj._pclass_frozen is True


# LLM-generated content at query #7
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert not (obj1 == obj3)
    
    # Test equality with self
    assert obj1 == obj1
    
    # Test inequality with different types
    assert (obj1 == "not a PClass") == NotImplemented
    assert (obj1 == 42) == NotImplemented
    assert (obj1 == None) == NotImplemented
    
    # Test inequality with different PClass types
    class OtherClass(PClass):
        x = field()
        y = field()
    
    obj4 = OtherClass(x=1, y=2)
    assert (obj1 == obj4) == NotImplemented
    
    # Test with missing fields (optional fields)
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj5 = OptionalClass(x=1)
    obj6 = OptionalClass(x=1, y=None)
    assert obj5 == obj6
    
    # Test with different optional field values
    obj7 = OptionalClass(x=1, y=5)
    assert not (obj5 == obj7)
    
    # Test with complex values
    class ComplexClass(PClass):
        data = field()
    
    obj8 = ComplexClass(data=[1, 2, 3])
    obj9 = ComplexClass(data=[1, 2, 3])
    assert obj8 == obj9
    
    obj10 = ComplexClass(data=[1, 2, 4])
    assert not (obj8 == obj10)


# LLM-generated content at query #8
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field
    
    # Test basic serialization without custom serializer
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y='test')
    result = obj.serialize()
    assert result == {'x': 1, 'y': 'test'}
    
    # Test serialization with missing optional field
    class PartialClass(PClass):
        a = field()
        b = field(initial=None)
    
    obj2 = PartialClass(a=42)
    result2 = obj2.serialize()
    assert result2 == {'a': 42, 'b': None}
    
    # Test serialization with custom serializer
    def custom_serializer(value, format=None):
        if format == 'uppercase':
            return value.upper()
        return value.lower()
    
    class CustomClass(PClass):
        name = field(serializer=custom_serializer)
        count = field()
    
    obj3 = CustomClass(name='Hello', count=5)
    result3 = obj3.serialize(format='uppercase')
    assert result3 == {'name': 'HELLO', 'count': 5}
    
    result4 = obj3.serialize(format='lowercase')
    assert result4 == {'name': 'hello', 'count': 5}
    
    # Test serialization with nested objects
    class Inner(PClass):
        value = field()
    
    class Outer(PClass):
        inner = field()
        name = field()
    
    inner_obj = Inner(value=10)
    outer_obj = Outer(inner=inner_obj, name='test')
    result5 = outer_obj.serialize()
    assert result5['name'] == 'test'
    assert isinstance(result5['inner'], Inner)
    
    # Test empty PClass serialization
    class EmptyClass(PClass):
        pass
    
    empty_obj = EmptyClass()
    result6 = empty_obj.serialize()
    assert result6 == {}
    
    # Test serialization with multiple fields
    class MultiField(PClass):
        field1 = field()
        field2 = field()
        field3 = field()
        field4 = field(initial=0)
    
    obj7 = MultiField(field1=1, field2=2, field3=3)
    result7 = obj7.serialize()
    assert result7 == {'field1': 1, 'field2': 2, 'field3': 3, 'field4': 0}


# LLM-generated content at query #9
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    # Test basic set with keyword arguments
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2
    
    # Test set with keyword argument
    obj2 = obj.set(x=10)
    assert obj2.x == 10
    assert obj2.y == 2
    assert obj.x == 1  # Original unchanged
    
    # Test set with positional arguments (field name, value)
    obj3 = obj.set('y', 20)
    assert obj3.x == 1
    assert obj3.y == 20
    assert obj.y == 2  # Original unchanged
    
    # Test set with multiple keyword arguments
    obj4 = obj.set(x=100, y=200)
    assert obj4.x == 100
    assert obj4.y == 200
    assert obj.x == 1
    assert obj.y == 2
    
    # Test that returned object is a new instance
    obj5 = obj.set(x=5)
    assert obj5 is not obj
    assert isinstance(obj5, TestClass)
    
    # Test set with optional field
    class TestClassOptional(PClass):
        a = field()
        b = field(initial=None)
    
    obj_opt = TestClassOptional(a=1)
    obj_opt2 = obj_opt.set(b=42)
    assert obj_opt2.a == 1
    assert obj_opt2.b == 42
    
    # Test set preserves all fields
    class ComplexClass(PClass):
        field1 = field()
        field2 = field()
        field3 = field()
    
    complex_obj = ComplexClass(field1=1, field2=2, field3=3)
    complex_obj2 = complex_obj.set(field2=20)
    assert complex_obj2.field1 == 1
    assert complex_obj2.field2 == 20
    assert complex_obj2.field3 == 3


# LLM-generated content at query #10
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3
    
    # Test inequality with different field values
    obj4 = TestClass(x=2, y=2)
    assert obj1 != obj4
    
    # Test equality with self
    assert obj1 == obj1
    
    # Test inequality with different class
    class AnotherClass(PClass):
        x = field()
        y = field()
    
    obj5 = AnotherClass(x=1, y=2)
    assert obj1 != obj5
    
    # Test inequality with non-PClass object
    assert (obj1 == "not a pclass") == NotImplemented
    assert (obj1 == 42) == NotImplemented
    assert (obj1 == None) == NotImplemented
    
    # Test with missing field values
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj6 = OptionalClass(x=1)
    obj7 = OptionalClass(x=1, y=None)
    assert obj6 == obj7
    
    # Test with callable initial values
    def get_default():
        return 10
    
    class CallableInitialClass(PClass):
        x = field()
        y = field(initial=get_default)
    
    obj8 = CallableInitialClass(x=1)
    obj9 = CallableInitialClass(x=1, y=10)
    assert obj8 == obj9
    
    # Test inequality when one has value and other doesn't
    obj10 = OptionalClass(x=1, y=5)
    obj11 = OptionalClass(x=1)
    assert obj10 != obj11


# LLM-generated content at query #11
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field(initial=10)
    
    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    obj2 = obj.set(x=5)
    assert obj2.x == 5
    assert obj2.y == 2
    assert obj2.z == 10
    assert obj.x == 1  # Original unchanged
    
    # Test set with positional arguments (field name, value)
    obj3 = obj.set('y', 20)
    assert obj3.x == 1
    assert obj3.y == 20
    assert obj3.z == 10
    assert obj.y == 2  # Original unchanged
    
    # Test set with multiple keyword arguments
    obj4 = obj.set(x=100, y=200)
    assert obj4.x == 100
    assert obj4.y == 200
    assert obj4.z == 10
    assert obj.x == 1  # Original unchanged
    assert obj.y == 2  # Original unchanged
    
    # Test set preserves fields not being modified
    obj5 = obj.set(x=7)
    assert obj5.x == 7
    assert obj5.y == 2
    assert obj5.z == 10
    
    # Test set with field that has initial value
    obj6 = obj.set(z=99)
    assert obj6.x == 1
    assert obj6.y == 2
    assert obj6.z == 99
    
    # Test set returns new instance
    obj7 = obj.set(x=999)
    assert obj7 is not obj
    assert isinstance(obj7, TestClass)
    
    # Test set with all fields
    obj8 = obj.set(x=11, y=22, z=33)
    assert obj8.x == 11
    assert obj8.y == 22
    assert obj8.z == 33
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 10


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from pyrsistent import PClass, field


def test_PClass_set():
    """Test PClass.set() method for setting field values."""
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field(initial=10)
    
    # Create initial instance
    obj = TestClass(x=1, y=2)
    
    # Test set with keyword arguments
    obj2 = obj.set(x=5)
    assert obj2.x == 5
    assert obj2.y == 2
    assert obj2.z == 10
    assert obj.x == 1  # Original unchanged
    
    # Test set with positional arguments (field name, value)
    obj3 = obj.set('y', 3)
    assert obj3.x == 1
    assert obj3.y == 3
    assert obj3.z == 10
    assert obj.y == 2  # Original unchanged
    
    # Test set with multiple keyword arguments
    obj4 = obj.set(x=7, y=8, z=9)
    assert obj4.x == 7
    assert obj4.y == 8
    assert obj4.z == 9
    assert obj.x == 1  # Original unchanged
    assert obj.y == 2
    assert obj.z == 10
    
    # Test that set returns a new instance
    assert obj2 is not obj
    assert obj3 is not obj
    assert obj4 is not obj
    
    # Test set with only positional arguments
    obj5 = obj.set('x', 11)
    assert obj5.x == 11
    assert obj5.y == 2
    assert obj5.z == 10
    
    # Test set preserves all fields when only updating one
    obj6 = obj.set(x=100)
    assert obj6.x == 100
    assert obj6.y == 2
    assert obj6.z == 10
    
    # Test set with field that has initial value
    obj7 = obj.set(z=20)
    assert obj7.x == 1
    assert obj7.y == 2
    assert obj7.z == 20


def test_PClass_set_with_missing_fields():
    """Test PClass.set() handles fields not yet set."""
    
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    obj = TestClass(x=1)
    obj2 = obj.set(x=2)
    
    assert obj2.x == 2
    assert obj2.y == 5


def test_PClass_set_returns_correct_type():
    """Test that set() returns instance of correct class."""
    
    class CustomClass(PClass):
        value = field()
    
    obj = CustomClass(value=1)
    obj2 = obj.set(value=2)
    
    assert isinstance(obj2, CustomClass)
    assert type(obj2) is type(obj)


def test_PClass_set_immutability():
    """Test that set() doesn't modify original instance."""
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    original_x = obj.x
    original_y = obj.y
    
    obj.set(x=10, y=20)
    
    assert obj.x == original_x
    assert obj.y == original_y


# LLM-generated content at query #13
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field
    
    class AClass(PClass):
        x = field()
        y = field()
    
    # Test basic set with kwargs
    a = AClass(x=1, y=2)
    a2 = a.set(x=2)
    assert a2.x == 2
    assert a2.y == 2
    assert a.x == 1
    assert a.y == 2
    
    # Test set with positional args
    a3 = a.set('x', 3)
    assert a3.x == 3
    assert a3.y == 2
    assert a.x == 1
    
    # Test set multiple fields
    a4 = a.set(x=10, y=20)
    assert a4.x == 10
    assert a4.y == 20
    assert a.x == 1
    assert a.y == 2
    
    # Test that original instance is not modified
    original_x = a.x
    original_y = a.y
    a.set(x=100, y=200)
    assert a.x == original_x
    assert a.y == original_y
    
    # Test set returns a new instance
    a5 = a.set(x=5)
    assert a5 is not a
    assert isinstance(a5, AClass)
    
    # Test set with optional fields
    class BClass(PClass):
        x = field()
        y = field(initial=0)
    
    b = BClass(x=1)
    b2 = b.set(x=2)
    assert b2.x == 2
    assert b2.y == 0
    
    # Test set preserves unmodified fields
    class CClass(PClass):
        a = field()
        b = field()
        c = field()
    
    c = CClass(a=1, b=2, c=3)
    c2 = c.set(b=20)
    assert c2.a == 1
    assert c2.b == 20
    assert c2.c == 3


# LLM-generated content at query #14
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    # Test equality with same values
    obj1 = SimpleClass(x=1, y=2)
    obj2 = SimpleClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = SimpleClass(x=1, y=3)
    assert obj1 != obj3
    
    # Test inequality with different x value
    obj4 = SimpleClass(x=2, y=2)
    assert obj1 != obj4
    
    # Test equality with itself
    assert obj1 == obj1
    
    # Test with missing optional fields
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj5 = OptionalClass(x=1)
    obj6 = OptionalClass(x=1)
    assert obj5 == obj6
    
    obj7 = OptionalClass(x=1, y=None)
    assert obj5 == obj7
    
    # Test inequality with different types
    obj8 = SimpleClass(x=1, y=2)
    assert (obj8 == "not a pclass") == NotImplemented
    assert (obj8 == 42) == NotImplemented
    assert (obj8 == None) == NotImplemented
    
    # Test with different PClass types
    class AnotherClass(PClass):
        x = field()
        y = field()
    
    obj9 = AnotherClass(x=1, y=2)
    obj10 = SimpleClass(x=1, y=2)
    assert obj9 != obj10
    
    # Test with empty PClass
    class EmptyClass(PClass):
        pass
    
    obj11 = EmptyClass()
    obj12 = EmptyClass()
    assert obj11 == obj12
    
    # Test with nested PClass
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner1 = InnerClass(value=5)
    inner2 = InnerClass(value=5)
    outer1 = OuterClass(inner=inner1)
    outer2 = OuterClass(inner=inner2)
    assert outer1 == outer2
    
    inner3 = InnerClass(value=6)
    outer3 = OuterClass(inner=inner3)
    assert outer1 != outer3


# LLM-generated content at query #15
#--------------------------

def test_PClass___reduce__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test basic pickling
    obj = TestClass(x=1, y="hello")
    restore_func, args = obj.__reduce__()
    
    assert restore_func == _restore_pickle
    assert len(args) == 2
    assert args[0] == TestClass
    assert args[1] == {'x': 1, 'y': "hello"}
    
    # Test with missing optional fields
    class TestClass2(PClass):
        a = field()
        b = field(initial=42)
    
    obj2 = TestClass2(a=10)
    restore_func2, args2 = obj2.__reduce__()
    
    assert restore_func2 == _restore_pickle
    assert args2[0] == TestClass2
    assert args2[1] == {'a': 10, 'b': 42}
    
    # Test with all fields present
    obj3 = TestClass2(a=5, b=100)
    restore_func3, args3 = obj3.__reduce__()
    
    assert restore_func3 == _restore_pickle
    assert args3[0] == TestClass2
    assert args3[1] == {'a': 5, 'b': 100}
    
    # Test with nested PClass
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
        name = field()
    
    inner_obj = InnerClass(value=42)
    outer_obj = OuterClass(inner=inner_obj, name="test")
    restore_func4, args4 = outer_obj.__reduce__()
    
    assert restore_func4 == _restore_pickle
    assert args4[0] == OuterClass
    assert args4[1]['name'] == "test"
    assert isinstance(args4[1]['inner'], InnerClass)
    assert args4[1]['inner'].value == 42


# LLM-generated content at query #16
#--------------------------

```python
def test_PClass___reduce__():
    """Test pickling support via __reduce__ method"""
    import pickle
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field(initial=42)
    
    # Test basic __reduce__ with all fields
    obj = TestClass(x=1, y="hello", z=99)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0] == _restore_pickle
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': "hello", 'z': 99}
    
    # Test __reduce__ with missing optional fields
    obj2 = TestClass(x=10, y="world")
    reduce_result2 = obj2.__reduce__()
    
    assert reduce_result2[0] == _restore_pickle
    assert reduce_result2[1][0] == TestClass
    assert reduce_result2[1][1] == {'x': 10, 'y': "world", 'z': 42}
    
    # Test that pickling and unpickling works correctly
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    assert unpickled == obj
    assert unpickled.x == 1
    assert unpickled.y == "hello"
    assert unpickled.z == 99
    
    # Test with minimal fields
    class MinimalClass(PClass):
        a = field()
    
    obj3 = MinimalClass(a=100)
    reduce_result3 = obj3.__reduce__()
    
    assert reduce_result3[1][0] == MinimalClass
    assert reduce_result3[1][1] == {'a': 100}
    
    pickled3 = pickle.dumps(obj3)
    unpickled3 = pickle.loads(pickled3)
    assert unpickled3.a == 100


# LLM-generated content at query #17
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3
    
    # Test inequality with different field values
    obj4 = TestClass(x=2, y=2)
    assert obj1 != obj4
    
    # Test equality with same instance
    assert obj1 == obj1
    
    # Test inequality with different class
    class AnotherClass(PClass):
        x = field()
        y = field()
    
    obj5 = AnotherClass(x=1, y=2)
    assert (obj1 == obj5) is NotImplemented or obj1 != obj5
    
    # Test inequality with non-PClass object
    assert (obj1 == "not a pclass") is NotImplemented or obj1 != "not a pclass"
    assert (obj1 == 42) is NotImplemented or obj1 != 42
    assert (obj1 == None) is NotImplemented or obj1 != None
    
    # Test with optional fields (missing values)
    class OptionalClass(PClass):
        x = field()
        y = field()
    
    obj6 = OptionalClass(x=1)
    obj7 = OptionalClass(x=1)
    assert obj6 == obj7
    
    obj8 = OptionalClass(x=1, y=2)
    assert obj6 != obj8
    
    # Test with all fields missing
    obj9 = OptionalClass()
    obj10 = OptionalClass()
    assert obj9 == obj10


# LLM-generated content at query #18
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3
    
    # Test inequality with different field values
    obj4 = TestClass(x=2, y=2)
    assert obj1 != obj4
    
    # Test equality with self
    assert obj1 == obj1
    
    # Test inequality with different class
    class AnotherClass(PClass):
        x = field()
        y = field()
    
    obj5 = AnotherClass(x=1, y=2)
    assert (obj1 == obj5) is NotImplemented or obj1 != obj5
    
    # Test inequality with non-PClass object
    assert (obj1 == {'x': 1, 'y': 2}) is NotImplemented
    assert (obj1 == 42) is NotImplemented
    assert (obj1 == "test") is NotImplemented
    
    # Test equality with optional fields
    class OptionalClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj6 = OptionalClass(x=1)
    obj7 = OptionalClass(x=1, y=10)
    assert obj6 == obj7
    
    obj8 = OptionalClass(x=1, y=20)
    assert obj6 != obj8
    
    # Test equality with missing values
    class MissingFieldClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj9 = MissingFieldClass(x=1)
    obj10 = MissingFieldClass(x=1, y=None)
    assert obj9 == obj10


# LLM-generated content at query #19
#--------------------------

```python
def test_PClass___reduce__():
    import pickle
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Create an instance
    obj = TestClass(x=1, y=2)

    # Call __reduce__
    result = obj.__reduce__()

    # Verify the result is a tuple with 2 elements
    assert isinstance(result, tuple)
    assert len(result) == 2

    # First element should be _restore_pickle function
    restore_func, (cls, data) = result
    assert restore_func == _restore_pickle
    assert cls is TestClass
    assert isinstance(data, dict)
    assert data == {'x': 1, 'y': 2}

    # Test pickling and unpickling round-trip
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj
    assert unpickled.x == 1
    assert unpickled.y == 2


def test_PClass___reduce___with_optional_fields():
    import pickle
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(initial=None)

    # Create instance with only required field
    obj = TestClass(x=5)

    result = obj.__reduce__()
    restore_func, (cls, data) = result

    assert restore_func == _restore_pickle
    assert cls is TestClass
    assert data == {'x': 5, 'y': None}

    # Verify pickle round-trip
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj


def test_PClass___reduce___empty_class():
    import pickle
    from pyrsistent import PClass

    class EmptyClass(PClass):
        pass

    obj = EmptyClass()
    result = obj.__reduce__()
    restore_func, (cls, data) = result

    assert restore_func == _restore_pickle
    assert cls is EmptyClass
    assert data == {}

    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj


def test_PClass___reduce___with_complex_values():
    import pickle
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    obj = TestClass(x=[1, 2, 3], y={'a': 1}, z=(1, 2))
    result = obj.__reduce__()
    restore_func, (cls, data) = result

    assert data == {'x': [1, 2, 3], 'y': {'a': 1}, 'z': (1, 2)}

    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj
    assert unpickled.x == [1, 2, 3]
    assert unpickled.y == {'a': 1}
    assert unpickled.z == (1, 2)


# LLM-generated content at query #20
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field
    
    # Test basic hash functionality
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj1 = SimpleClass(x=1, y=2)
    obj2 = SimpleClass(x=1, y=2)
    obj3 = SimpleClass(x=1, y=3)
    
    # Equal objects should have equal hashes
    assert hash(obj1) == hash(obj2)
    
    # Different objects should (likely) have different hashes
    assert hash(obj1) != hash(obj3)
    
    # Hash should be consistent across multiple calls
    hash1 = hash(obj1)
    hash2 = hash(obj1)
    assert hash1 == hash2
    
    # Test with missing values
    class OptionalFieldClass(PClass):
        a = field()
        b = field(initial=None)
    
    opt1 = OptionalFieldClass(a=5)
    opt2 = OptionalFieldClass(a=5, b=None)
    
    # Objects with same field values should have same hash
    assert hash(opt1) == hash(opt2)
    
    # Test with nested PClass
    class InnerClass(PClass):
        val = field()
    
    class OuterClass(PClass):
        inner = field()
        name = field()
    
    inner1 = InnerClass(val=10)
    inner2 = InnerClass(val=10)
    
    outer1 = OuterClass(inner=inner1, name="test")
    outer2 = OuterClass(inner=inner2, name="test")
    
    assert hash(outer1) == hash(outer2)
    
    # Test that hash can be used in sets and dicts
    obj_set = {obj1, obj2, obj3}
    assert len(obj_set) == 2  # obj1 and obj2 are equal
    
    obj_dict = {obj1: "value1"}
    obj_dict[obj2] = "value2"
    assert len(obj_dict) == 1  # obj1 and obj2 are the same key
    assert obj_dict[obj1] == "value2"
    
    # Test hash with various field types
    class MixedTypesClass(PClass):
        int_field = field()
        str_field = field()
        tuple_field = field()
    
    mixed1 = MixedTypesClass(int_field=42, str_field="hello", tuple_field=(1, 2, 3))
    mixed2 = MixedTypesClass(int_field=42, str_field="hello", tuple_field=(1, 2, 3))
    
    assert hash(mixed1) == hash(mixed2)


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from pyrsistent import PClass, field, InvariantException


def test_PClass___new__():
    """Test PClass.__new__ method with various scenarios"""
    
    # Test 1: Basic PClass creation with required fields
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2
    assert obj._pclass_frozen is True
    
    # Test 2: PClass with default initial values
    class ClassWithDefaults(PClass):
        x = field(initial=10)
        y = field(initial=20)
    
    obj = ClassWithDefaults()
    assert obj.x == 10
    assert obj.y == 20
    
    # Test 3: PClass with callable initial values
    class ClassWithCallableDefaults(PClass):
        items = field(initial=list)
    
    obj1 = ClassWithCallableDefaults()
    obj2 = ClassWithCallableDefaults()
    assert obj1.items == []
    assert obj2.items == []
    assert obj1.items is not obj2.items  # Different list instances
    
    # Test 4: Missing mandatory field raises InvariantException
    class ClassWithMandatory(PClass):
        required = field()
        optional = field(initial=None)
    
    with pytest.raises(InvariantException) as exc_info:
        ClassWithMandatory(optional=5)
    assert 'ClassWithMandatory.required' in str(exc_info.value)
    
    # Test 5: Extra unknown fields raise AttributeError
    class StrictClass(PClass):
        x = field()
    
    with pytest.raises(AttributeError) as exc_info:
        StrictClass(x=1, unknown_field=2)
    assert 'unknown_field' in str(exc_info.value)
    assert 'not among the specified fields' in str(exc_info.value)
    
    # Test 6: Field type checking
    class TypeCheckedClass(PClass):
        x = field(type=int)
    
    with pytest.raises(TypeError):
        TypeCheckedClass(x="not an int")
    
    obj = TypeCheckedClass(x=42)
    assert obj.x == 42
    
    # Test 7: ignore_extra parameter
    class IgnoreExtraClass(PClass):
        x = field()
    
    obj = IgnoreExtraClass.create({'x': 1, 'extra': 2}, ignore_extra=True)
    assert obj.x == 1
    assert not hasattr(obj, 'extra')
    
    # Test 8: Field with factory function
    class ClassWithFactory(PClass):
        x = field(factory=int)
    
    obj = ClassWithFactory(x="42")
    assert obj.x == 42
    
    # Test 9: Multiple fields with mixed initialization
    class MixedClass(PClass):
        a = field()
        b = field(initial=100)
        c = field(initial=lambda: [])
    
    obj = MixedClass(a=1)
    assert obj.a == 1
    assert obj.b == 100
    assert obj.c == []
    
    # Test 10: PClass is frozen after creation
    class FrozenTestClass(PClass):
        x = field()
    
    obj = FrozenTestClass(x=1)
    with pytest.raises(AttributeError) as exc_info:
        obj.x = 2
    assert "Can't set attribute" in str(exc_info.value)
    
    # Test 11: Field invariant validation
    class ClassWithInvariant(PClass):
        x = field(invariant=lambda v: (v > 0, "x must be positive"))
    
    with pytest.raises(InvariantException):
        ClassWithInvariant(x=-1)
    
    obj = ClassWithInvariant(x=5)
    assert obj.x == 5
    
    # Test 12: Empty PClass
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    assert obj._pclass_frozen is True
    
    # Test 13: Partial field initialization with defaults
    class PartialClass(PClass):
        a = field()
        b = field(initial=2)
        c = field(initial=3)
    
    obj = PartialClass(a=1, c=30)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 30


# LLM-generated content at query #22
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field
    import pickle
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Create an instance
    obj = TestClass(x=1, y=2)
    
    # Test __reduce__ returns correct structure
    result = obj.__reduce__()
    assert len(result) == 2
    assert result[0] == _restore_pickle
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}
    
    # Test pickling and unpickling
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    assert unpickled == obj
    assert unpickled.x == 1
    assert unpickled.y == 2
    
    # Test with missing optional fields
    class TestClassOptional(PClass):
        x = field()
        y = field(initial=None)
    
    obj2 = TestClassOptional(x=5)
    result2 = obj2.__reduce__()
    assert result2[1][0] == TestClassOptional
    assert result2[1][1] == {'x': 5, 'y': None}
    
    pickled2 = pickle.dumps(obj2)
    unpickled2 = pickle.loads(pickled2)
    assert unpickled2 == obj2
    assert unpickled2.x == 5


# LLM-generated content at query #23
#--------------------------

```python
def test_PClassMeta___new__():
    """Test PClassMeta.__new__ method"""
    from pyrsistent import PClass, field
    
    # Test basic PClass creation with metaclass
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Verify that _pclass_fields was set correctly
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    
    # Verify that __slots__ was set correctly
    assert hasattr(TestClass, '__slots__')
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'x' in TestClass.__slots__
    assert 'y' in TestClass.__slots__
    
    # Verify __weakref__ is in slots for direct PClass subclass
    assert '__weakref__' in TestClass.__slots__
    
    # Test that invariants are stored
    assert hasattr(TestClass, '_pclass_invariants')
    
    # Test nested inheritance - __weakref__ should not be duplicated
    class DerivedClass(TestClass):
        z = field()
    
    assert hasattr(DerivedClass, '_pclass_fields')
    assert 'x' in DerivedClass._pclass_fields
    assert 'y' in DerivedClass._pclass_fields
    assert 'z' in DerivedClass._pclass_fields
    
    # __weakref__ should only be in the direct PClass subclass, not derived classes
    assert '__weakref__' in TestClass.__slots__
    assert '__weakref__' not in DerivedClass.__slots__
    
    # Verify slots contain all fields
    assert 'z' in DerivedClass.__slots__
    assert '_pclass_frozen' in DerivedClass.__slots__
    
    # Test that instances can be created
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2
    
    # Test with custom invariant
    def check_positive(val):
        return (val > 0, 'must be positive')
    
    class PositiveClass(PClass):
        value = field(invariant=check_positive)
    
    assert hasattr(PositiveClass, '_pclass_invariants')
    
    # Test that the metaclass is properly set
    assert type(TestClass) == PClassMeta
    assert isinstance(TestClass, PClassMeta)


# LLM-generated content at query #24
#--------------------------

def test_PClass___eq__():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2
    
    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3
    
    # Test inequality with different field values
    obj4 = TestClass(x=2, y=2)
    assert obj1 != obj4
    
    # Test equality with self
    assert obj1 == obj1
    
    # Test inequality with different class
    class AnotherClass(PClass):
        x = field()
        y = field()
    
    obj5 = AnotherClass(x=1, y=2)
    assert (obj1 == obj5) is NotImplemented or obj1 != obj5
    
    # Test with missing optional fields
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj6 = OptionalClass(x=1)
    obj7 = OptionalClass(x=1, y=None)
    assert obj6 == obj7
    
    # Test inequality when one has value and other doesn't
    obj8 = OptionalClass(x=1, y=5)
    assert obj6 != obj8
    
    # Test with non-PClass object
    result = obj1 == "not a pclass"
    assert result is NotImplemented or result is False
    
    # Test with None
    result = obj1 == None
    assert result is NotImplemented or result is False
    
    # Test with multiple fields having different values
    class MultiFieldClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj9 = MultiFieldClass(a=1, b=2, c=3)
    obj10 = MultiFieldClass(a=1, b=2, c=3)
    obj11 = MultiFieldClass(a=1, b=2, c=4)
    assert obj9 == obj10
    assert obj9 != obj11


# LLM-generated content at query #25
#--------------------------

```python
def test_PClass___reduce__():
    import pickle
    
    class TestPClass(PClass):
        x = field()
        y = field()
    
    # Create an instance
    obj = TestPClass(x=1, y=2)
    
    # Test __reduce__ returns the correct format
    result = obj.__reduce__()
    
    # Should return a tuple of (callable, args)
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    # First element should be _restore_pickle function
    assert result[0] == _restore_pickle
    
    # Second element should be a tuple containing the class and data dict
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestPClass
    assert result[1][1] == {'x': 1, 'y': 2}
    
    # Test pickling and unpickling round-trip
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)
    
    assert unpickled == obj
    assert unpickled.x == 1
    assert unpickled.y == 2
    
    # Test with only some fields set
    class TestPClass2(PClass):
        a = field()
        b = field(initial=10)
    
    obj2 = TestPClass2(a=5)
    result2 = obj2.__reduce__()
    
    assert result2[0] == _restore_pickle
    assert result2[1][0] == TestPClass2
    # Should only include fields that are set
    assert 'a' in result2[1][1]
    assert result2[1][1]['a'] == 5
    assert 'b' in result2[1][1]
    assert result2[1][1]['b'] == 10
    
    # Verify round-trip works
    pickled2 = pickle.dumps(obj2)
    unpickled2 = pickle.loads(pickled2)
    assert unpickled2 == obj2


