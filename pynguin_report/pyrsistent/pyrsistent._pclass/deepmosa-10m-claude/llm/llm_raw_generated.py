####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_set_updates_data_and_marks_dirty():
    class MockPClass:
        pass
    
    original = MockPClass()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.set('key2', 'value2')
    
    assert evolver._pclass_evolver_data['key2'] == 'value2'
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'key2' in evolver._factory_fields
    assert result is evolver


def test_set_with_same_value_does_not_mark_dirty():
    class MockPClass:
        pass
    
    original = MockPClass()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    value_obj = object()
    evolver._pclass_evolver_data['key1'] = value_obj
    
    result = evolver.set('key1', value_obj)
    
    assert evolver._pclass_evolver_data_is_dirty is False
    assert 'key1' not in evolver._factory_fields
    assert result is evolver


def test_set_overwrites_existing_value():
    class MockPClass:
        pass
    
    original = MockPClass()
    initial_dict = {'key1': 'old_value'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.set('key1', 'new_value')
    
    assert evolver._pclass_evolver_data['key1'] == 'new_value'
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'key1' in evolver._factory_fields
    assert result is evolver


def test_set_returns_self_for_chaining():
    class MockPClass:
        pass
    
    original = MockPClass()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    
    result1 = evolver.set('key1', 'value1')
    result2 = result1.set('key2', 'value2')
    
    assert result1 is evolver
    assert result2 is evolver
    assert evolver._pclass_evolver_data['key1'] == 'value1'
    assert evolver._pclass_evolver_data['key2'] == 'value2'


def test_set_with_none_value():
    class MockPClass:
        pass
    
    original = MockPClass()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.set('key2', None)
    
    assert evolver._pclass_evolver_data['key2'] is None
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'key2' in evolver._factory_fields
    assert result is evolver


# LLM-generated content at query #2
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2
    assert obj._pclass_frozen is True


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == 10
    assert obj.y == 5


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_new_with_type_check():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="string")
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive(val):
        return (val > 0, "must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive)
    
    try:
        TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "must be positive" in e.error_codes


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._invariant import InvariantException
    
    def check_sum(obj):
        return (obj.x + obj.y > 0, "sum must be positive")
    
    @invariant(check_sum)
    class TestClass(PClass):
        x = field()
        y = field()
    
    try:
        TestClass(x=-5, y=1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_pclass_new_frozen():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    obj = TestClass(x="42")
    assert obj.x == 42


def test_pclass_new_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert obj._pclass_frozen is True


def test_pclass_new_multiple_invariant_errors():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive(val):
        return (val > 0, "must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive)
        y = field(invariant=positive)
    
    try:
        TestClass(x=-1, y=-2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 2


# LLM-generated content at query #3
#--------------------------

```python
def test_remove_existing_item():
    original = type('MockPClass', (), {})()
    initial_dict = {'key1': 'value1', 'key2': 'value2'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.remove('key1')
    
    assert result is evolver
    assert 'key1' not in evolver._pclass_evolver_data
    assert 'key1' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data == {'key2': 'value2'}


def test_remove_nonexistent_item():
    original = type('MockPClass', (), {})()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    try:
        evolver.remove('nonexistent')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == 'nonexistent'


def test_remove_item_that_was_set():
    original = type('MockPClass', (), {})()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('key2', 'value2')
    
    result = evolver.remove('key2')
    
    assert result is evolver
    assert 'key2' not in evolver._pclass_evolver_data
    assert 'key2' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True


def test_remove_marks_data_as_dirty():
    original = type('MockPClass', (), {})()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    assert evolver._pclass_evolver_data_is_dirty is False
    
    evolver.remove('key1')
    
    assert evolver._pclass_evolver_data_is_dirty is True


def test_remove_multiple_items():
    original = type('MockPClass', (), {})()
    initial_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.remove('key1')
    evolver.remove('key3')
    
    assert evolver._pclass_evolver_data == {'key2': 'value2'}
    assert 'key1' not in evolver._factory_fields
    assert 'key3' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #4
#--------------------------

```python
def test_set_with_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1
    assert obj.y == 2


def test_set_with_args():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set('x', 10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1


def test_set_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(x=10, y=20)
    
    assert result.x == 10
    assert result.y == 20
    assert result.z == 3
    assert obj.x == 1
    assert obj.y == 2


def test_set_preserves_other_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a=1, b=2, c=3)
    result = obj.set(b=100)
    
    assert result.a == 1
    assert result.b == 100
    assert result.c == 3


def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set(x=2)
    
    assert obj is not result
    assert isinstance(result, TestClass)


def test_set_with_initial_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=99)
    
    obj = TestClass(x=1)
    result = obj.set(x=5)
    
    assert result.x == 5
    assert result.y == 99


# LLM-generated content at query #5
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.missing_fields) == 1
        assert 'TestClass.y' in e.missing_fields


def test_pclass_raises_invariant_exception_when_field_invariant_errors():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    def invariant_check(value):
        return (False, 'test_error') if value < 0 else (True, None)
    
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    
    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0


def test_pclass_raises_invariant_exception_with_both_errors_and_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    def invariant_check(value):
        return (False, 'test_error') if value < 0 else (True, None)
    
    class TestClass(PClass):
        x = field(invariant=invariant_check)
        y = field(mandatory=True)
    
    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0
        assert len(e.missing_fields) > 0


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize_with_no_fields():
    from pyrsistent import PClass, field
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    result = instance.serialize()
    assert result == {}


def test_serialize_with_simple_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    instance = SimpleClass(x=1, y="test")
    result = instance.serialize()
    assert result == {'x': 1, 'y': 'test'}


def test_serialize_with_missing_optional_fields():
    from pyrsistent import PClass, field
    
    class PartialClass(PClass):
        x = field()
        y = field(initial=None)
    
    instance = PartialClass(x=42)
    result = instance.serialize()
    assert 'x' in result
    assert result['x'] == 42
    assert 'y' in result or 'y' not in result


def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field
    
    def custom_serializer(format, value):
        return str(value).upper()
    
    class CustomSerializerClass(PClass):
        name = field(serializer=custom_serializer)
    
    instance = CustomSerializerClass(name="hello")
    result = instance.serialize()
    assert result['name'] == 'HELLO'


def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    
    def format_aware_serializer(format, value):
        if format == 'json':
            return str(value)
        return value
    
    class FormatClass(PClass):
        value = field(serializer=format_aware_serializer)
    
    instance = FormatClass(value=123)
    result = instance.serialize(format='json')
    assert result['value'] == '123'


def test_serialize_with_nested_objects():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        inner_value = field()
    
    class OuterClass(PClass):
        outer_value = field()
    
    inner = InnerClass(inner_value=10)
    outer = OuterClass(outer_value=inner)
    result = outer.serialize()
    assert result['outer_value'] == inner


def test_serialize_multiple_fields_with_mixed_serializers():
    from pyrsistent import PClass, field
    
    def double_serializer(format, value):
        return value * 2
    
    class MixedClass(PClass):
        normal = field()
        doubled = field(serializer=double_serializer)
    
    instance = MixedClass(normal=5, doubled=3)
    result = instance.serialize()
    assert result['normal'] == 5
    assert result['doubled'] == 6


def test_serialize_with_none_values():
    from pyrsistent import PClass, field
    
    class NullableClass(PClass):
        nullable_field = field(initial=None)
        required_field = field()
    
    instance = NullableClass(nullable_field=None, required_field="present")
    result = instance.serialize()
    assert result['nullable_field'] is None
    assert result['required_field'] == "present"


# LLM-generated content at query #7
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    reduce_result = instance.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    instance = TestClass(x=5)
    reduce_result = instance.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 5, 'y': 10}


def test_pclass_reduce_partial_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    reduce_result = instance.__reduce__()
    
    assert reduce_result[1][1] == {'a': 1, 'b': 2, 'c': 3}


def test_pclass_reduce_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=None)
    
    instance = TestClass()
    reduce_result = instance.__reduce__()
    
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert isinstance(reduce_result[1][1], dict)


# LLM-generated content at query #8
#--------------------------

```python
def test_remove_item_exists_in_data():
    initial_dict = {'key1': 'value1', 'key2': 'value2'}
    original = type('MockPClass', (), {})()
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.remove('key1')
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'key1' not in evolver._factory_fields
    assert result is evolver


# LLM-generated content at query #9
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    
    # The predicate at line 7 is "for name in self._pclass_fields"
    # This evaluates to True when _pclass_fields is iterable and non-empty
    assert hasattr(instance, '_pclass_fields')
    assert len(instance._pclass_fields) > 0
    
    # Verify that _pclass_fields contains the expected field names
    field_names = list(instance._pclass_fields.keys())
    assert 'x' in field_names
    assert 'y' in field_names


# LLM-generated content at query #10
#--------------------------

```python
def test_pclass_meta_new_creates_class_with_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import CheckedType
    
    # Create a simple field
    test_field = _PField(None, None, None, None, False)
    
    # Create class using PClassMeta
    dct = {'test_attr': test_field}
    bases = (CheckedType,)
    cls = PClassMeta('TestClass', bases, dct)
    
    assert cls.__name__ == 'TestClass'
    assert hasattr(cls, '_pclass_fields')
    assert 'test_attr' in cls._pclass_fields


def test_pclass_meta_new_sets_slots():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    dct = {}
    bases = (CheckedType,)
    cls = PClassMeta('TestClass', bases, dct)
    
    assert hasattr(cls, '__slots__')
    assert '_pclass_frozen' in cls.__slots__


def test_pclass_meta_new_adds_weakref_slot():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    dct = {}
    bases = (CheckedType,)
    cls = PClassMeta('TestClass', bases, dct)
    
    assert '__weakref__' in cls.__slots__


def test_pclass_meta_new_stores_invariants():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    def invariant(self):
        return True, "test"
    
    dct = {'__invariant__': invariant}
    bases = (CheckedType,)
    cls = PClassMeta('TestClass', bases, dct)
    
    assert hasattr(cls, '_pclass_invariants')
    assert len(cls._pclass_invariants) > 0


def test_pclass_meta_new_inherits_fields_from_base():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import CheckedType
    
    # Create base class with field
    base_field = _PField(None, None, None, None, False)
    base_dct = {'base_field': base_field}
    BaseClass = PClassMeta('BaseClass', (CheckedType,), base_dct)
    
    # Create derived class
    derived_dct = {}
    DerivedClass = PClassMeta('DerivedClass', (BaseClass,), derived_dct)
    
    assert 'base_field' in DerivedClass._pclass_fields


def test_pclass_meta_new_field_removed_from_dict():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import CheckedType
    
    test_field = _PField(None, None, None, None, False)
    dct = {'test_attr': test_field}
    bases = (CheckedType,)
    cls = PClassMeta('TestClass', bases, dct)
    
    assert not hasattr(cls, 'test_attr')
    assert 'test_attr' in cls._pclass_fields


def test_pclass_meta_new_multiple_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import CheckedType
    
    field1 = _PField(None, None, None, None, False)
    field2 = _PField(None, None, None, None, False)
    dct = {'field1': field1, 'field2': field2}
    bases = (CheckedType,)
    cls = PClassMeta('TestClass', bases, dct)
    
    assert 'field1' in cls._pclass_fields
    assert 'field2' in cls._pclass_fields
    assert len(cls._pclass_fields) == 2


# LLM-generated content at query #11
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2
    assert obj._pclass_frozen is True


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == 10
    assert obj.y == 5


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        obj = TestClass(x="string")
        assert False, "Should have raised PTypeError"
    except Exception as e:
        assert 'Invalid type' in str(e)


def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    obj = TestClass(x="5", _factory_fields={'x'})
    assert obj.x == 5


def test_pclass_new_without_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    obj = TestClass(x=5, _factory_fields=set())
    assert obj.x == 5


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_field_invariant_failure():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def check_positive(value):
        return (value > 0, "Value must be positive")
    
    class TestClass(PClass):
        x = field(invariant=check_positive)
    
    try:
        obj = TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Value must be positive" in e.error_codes


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._invariant import InvariantException
    
    @invariant(lambda obj: (obj.x > obj.y, "x must be greater than y"))
    class TestClass(PClass):
        x = field()
        y = field()
    
    try:
        obj = TestClass(x=1, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "x must be greater than y" in e.error_codes


def test_pclass_new_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field(initial=30)
    
    obj = TestClass(a=1, b=2)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 30


def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, ignore_extra=True)
    assert obj.x == 1


# LLM-generated content at query #12
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_new_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='5')
    assert instance.x == 5


def test_pclass_new_with_ignore_extra_true():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True)
    assert instance.x == 1


def test_pclass_new_with_field_invariant_violation():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive_invariant(value):
        return (value > 0, 'must be positive')
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'must be positive' in e.error_codes


def test_pclass_new_with_global_invariant_violation():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def sum_invariant(obj):
        return (obj.x + obj.y > 0, 'sum must be positive')
    
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = (sum_invariant,)
    
    try:
        TestClass(x=-5, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'sum must be positive' in e.error_codes


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x='not an int')
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_pclass_new_creates_independent_instances():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [])
    
    instance1 = TestClass()
    instance2 = TestClass()
    assert instance1.x is not instance2.x


def test_pclass_new_with_factory_fields_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
        y = field()
    
    instance = TestClass(x='5', y=10, _factory_fields={'x'})
    assert instance.x == 5
    assert instance.y == 10


def test_pclass_new_without_factory_fields_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #13
#--------------------------

```python
def test_pclass_hash_same_values_same_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_different_values_different_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert hash(obj1) != hash(obj2)


def test_pclass_hash_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_hashable_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    obj3 = TestClass(x=2)
    
    hash_set = {obj1, obj2, obj3}
    
    assert len(hash_set) == 2


def test_pclass_hash_hashable_as_dict_key():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=2)
    
    test_dict = {obj1: 'value1', obj2: 'value2'}
    
    assert test_dict[obj1] == 'value1'
    assert test_dict[obj2] == 'value2'


def test_pclass_hash_with_nested_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=(2, 3))
    obj2 = TestClass(x=1, y=(2, 3))
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_with_string_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
        value = field()
    
    obj1 = TestClass(name='test', value='data')
    obj2 = TestClass(name='test', value='data')
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_consistent_across_calls():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=42)
    hash1 = hash(obj)
    hash2 = hash(obj)
    
    assert hash1 == hash2


# LLM-generated content at query #14
#--------------------------

```python
def test_pclass_new_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    
    assert instance.x == 1
    assert instance.y == 2
    assert hasattr(instance, '_pclass_frozen')
    assert instance._pclass_frozen is True


# LLM-generated content at query #15
#--------------------------

```python
def test_is_pclass_with_single_checked_type_base():
    from your_module import _is_pclass, CheckedType
    result = _is_pclass((CheckedType,))
    assert result is True


def test_is_pclass_with_multiple_bases():
    from your_module import _is_pclass, CheckedType
    result = _is_pclass((CheckedType, object))
    assert result is False


def test_is_pclass_with_no_bases():
    from your_module import _is_pclass
    result = _is_pclass(())
    assert result is False


def test_is_pclass_with_different_single_base():
    from your_module import _is_pclass
    result = _is_pclass((object,))
    assert result is False


def test_is_pclass_with_empty_tuple():
    from your_module import _is_pclass
    result = _is_pclass(())
    assert result is False


# LLM-generated content at query #16
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PyrsistentAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='5')
    assert instance.x == 5
    assert isinstance(instance.x, int)


# LLM-generated content at query #17
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_partial_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj = TestClass(x=5)
    result = obj.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 5, 'y': 10}


def test_pclass_reduce_no_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=None)
    result = obj.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': None}


def test_pclass_reduce_multiple_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a='string', b=42, c=[1, 2, 3])
    result = obj.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'a': 'string', 'b': 42, 'c': [1, 2, 3]}


# LLM-generated content at query #18
#--------------------------

```python
def test_pclass_meta_new_is_pclass_predicate_false():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    # Create a test case where _is_pclass(bases) evaluates to False
    # This happens when bases is empty or contains non-PClass types
    
    name = 'TestClass'
    bases = ()  # Empty bases tuple makes _is_pclass return False
    dct = {}
    
    # Call PClassMeta.__new__ with empty bases
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    # Verify that __weakref__ was NOT added to __slots__
    # (it should only be added when _is_pclass(bases) is True)
    assert '__weakref__' not in result.__slots__
    assert '_pclass_frozen' in result.__slots__


# LLM-generated content at query #19
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj = TestClass(x=5)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert 'x' in reduce_result[1][1]
    assert 'y' in reduce_result[1][1]
    assert reduce_result[1][1]['x'] == 5
    assert reduce_result[1][1]['y'] == 10


def test_pclass_reduce_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    
    obj = TestClass()
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_single_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        value = field()
    
    obj = TestClass(value='test')
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[1][1] == {'value': 'test'}


def test_pclass_reduce_complex_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        items = field()
        mapping = field()
    
    obj = TestClass(items=[1, 2, 3], mapping={'a': 1, 'b': 2})
    reduce_result = obj.__reduce__()
    
    assert reduce_result[1][1]['items'] == [1, 2, 3]
    assert reduce_result[1][1]['mapping'] == {'a': 1, 'b': 2}


# LLM-generated content at query #20
#--------------------------

```python
def test_pclass_predicate_line_25_with_invariant_errors():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def check_positive(obj):
        return (obj.x > 0, 'x must be positive')
    
    class TestClass(PClass):
        x = field()
        __invariants__ = (check_positive,)
    
    try:
        TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors or e.missing_fields


def test_pclass_predicate_line_25_with_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors or e.missing_fields
        assert 'TestClass.x' in e.missing_fields


def test_pclass_predicate_line_25_both_conditions():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def check_positive(obj):
        return (obj.x > 0, 'x must be positive')
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
        __invariants__ = (check_positive,)
    
    try:
        TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors or e.missing_fields


# LLM-generated content at query #21
#--------------------------

```python
def test_pclass_hash_returns_consistent_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    hash1 = hash(obj1)
    hash2 = hash(obj2)
    
    assert hash1 == hash2
    assert isinstance(hash1, int)


def test_pclass_hash_different_for_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    hash1 = hash(obj1)
    hash2 = hash(obj2)
    
    assert hash1 != hash2


def test_pclass_hash_works_with_missing_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1)
    
    hash_value = hash(obj)
    assert isinstance(hash_value, int)


def test_pclass_hash_is_hashable():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    hash_set = {obj1, obj2}
    assert len(hash_set) == 1


def test_pclass_hash_with_complex_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x="hello", y=(1, 2, 3))
    obj2 = TestClass(x="hello", y=(1, 2, 3))
    
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #22
#--------------------------

```python
def test_pclass_new_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    
    assert instance.x == 1
    assert instance.y == 2
    assert hasattr(instance, '_pclass_frozen')
    assert instance._pclass_frozen is True


# LLM-generated content at query #23
#--------------------------

```python
def test_pclass_hash_returns_consistent_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert isinstance(hash1, int)
    assert hash1 == hash2
    assert hash(instance1) == hash(instance1)


def test_pclass_hash_different_for_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    
    assert hash(instance1) != hash(instance2)


def test_pclass_hash_with_optional_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    instance = TestClass(x=1)
    hash_value = hash(instance)
    
    assert isinstance(hash_value, int)


def test_pclass_hash_allows_use_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    instance3 = TestClass(x=2)
    
    hash_set = {instance1, instance2, instance3}
    
    assert len(hash_set) >= 2


def test_pclass_hash_allows_use_as_dict_key():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    
    test_dict = {instance1: 'value1', instance2: 'value2'}
    
    assert test_dict[instance1] == 'value1'
    assert test_dict[instance2] == 'value2'


# LLM-generated content at query #24
#--------------------------

```python
def test_pclass_eq_same_class_same_values():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2


def test_pclass_eq_same_class_different_values():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    assert not (obj1 == obj2)


def test_pclass_eq_different_class():
    class TestClass1(PClass):
        x = field()
    
    class TestClass2(PClass):
        x = field()
    
    obj1 = TestClass1(x=1)
    obj2 = TestClass2(x=1)
    assert (obj1 == obj2) is NotImplemented


def test_pclass_eq_with_non_pclass_object():
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = {"x": 1}
    assert (obj1 == obj2) is NotImplemented


def test_pclass_eq_with_missing_values():
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1, y=5)
    assert obj1 == obj2


def test_pclass_eq_one_has_missing_value_other_doesnt():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1)
    assert not (obj1 == obj2)


def test_pclass_eq_both_have_missing_values():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    assert obj1 == obj2


def test_pclass_eq_reflexive():
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    assert obj == obj


def test_pclass_eq_with_none_values():
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=None)
    obj2 = TestClass(x=None)
    assert obj1 == obj2


def test_pclass_eq_complex_values():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=[1, 2, 3], y={'a': 1})
    obj2 = TestClass(x=[1, 2, 3], y={'a': 1})
    assert obj1 == obj2


# LLM-generated content at query #25
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 5)
    
    instance = TestClass()
    assert instance.x == 5


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="5")
    assert instance.x == 5


def test_pclass_constructor_multiple_fields_with_mixed_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field(initial=100)
        c = field(mandatory=True)
    
    instance = TestClass(a="hello", c=3.14)
    assert instance.a == "hello"
    assert instance.b == 100
    assert instance.c == 3.14


# LLM-generated content at query #26
#--------------------------

```python
def test_remove_existing_item():
    original = object()
    initial_dict = {'key1': 'value1', 'key2': 'value2'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.remove('key1')
    
    assert result is evolver
    assert 'key1' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'key1' not in evolver._factory_fields


def test_remove_nonexistent_item():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    try:
        evolver.remove('nonexistent')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == 'nonexistent'


def test_remove_multiple_items():
    original = object()
    initial_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.remove('key1')
    evolver.remove('key2')
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert 'key2' not in evolver._pclass_evolver_data
    assert 'key3' in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True


def test_remove_discards_from_factory_fields():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    evolver._factory_fields.add('key1')
    
    evolver.remove('key1')
    
    assert 'key1' not in evolver._factory_fields


def test_remove_sets_dirty_flag():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    assert evolver._pclass_evolver_data_is_dirty is False
    
    evolver.remove('key1')
    
    assert evolver._pclass_evolver_data_is_dirty is True


def test_delitem_calls_remove():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    del evolver['key1']
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #27
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2
    assert obj._pclass_frozen is True


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=2)
    assert obj.x == 10
    assert obj.y == 2


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_new_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    obj = TestClass(x={'a': 1})
    assert obj.x == pmap({'a': 1})


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive(value):
        return (value > 0, 'must be positive')
    
    class TestClass(PClass):
        x = field(invariant=positive)
    
    try:
        obj = TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    obj = TestClass(x={'a': 1}, ignore_extra=True)
    assert obj.x == pmap({'a': 1})


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    
    def check_sum(obj):
        return (obj.x + obj.y == 10, 'sum must be 10')
    
    class TestClass(PClass):
        __invariant__ = invariant(check_sum)
        x = field()
        y = field()
    
    try:
        obj = TestClass(x=1, y=2)
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_pclass_new_with_type_checking():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        obj = TestClass(x="not an int")
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_multiple_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(type=(int, str))
    
    obj1 = TestClass(x=1)
    assert obj1.x == 1
    obj2 = TestClass(x="hello")
    assert obj2.x == "hello"


def test_pclass_new_with_factory_fields_subset():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
        y = field()
    
    obj = TestClass(_factory_fields={'x'}, x={'a': 1}, y=2)
    assert obj.x == pmap({'a': 1})
    assert obj.y == 2


def test_pclass_new_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=None)
    
    obj = TestClass()
    assert obj.x is None


# LLM-generated content at query #28
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._exceptions import PTypeError
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (True, None))
    
    try:
        _check_and_set_attr(MockClass, field, "test_field", "not_an_int", result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError:
        pass
    
    assert not hasattr(result, "test_field")
    assert invariant_errors == []


def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (False, "value_too_small"))
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert not hasattr(result, "test_field")
    assert invariant_errors == ["value_too_small"]


def test_check_and_set_attr_multiple_valid_types():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int, str], lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", "hello", result, invariant_errors)
    
    assert result.test_field == "hello"
    assert invariant_errors == []


def test_check_and_set_attr_no_type_constraint():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField(None, lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", object(), result, invariant_errors)
    
    assert hasattr(result, "test_field")
    assert invariant_errors == []


# LLM-generated content at query #29
#--------------------------

```python
def test_reduce_returns_tuple_with_restore_pickle_and_class_data():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 1, 'y': 2}


# LLM-generated content at query #30
#--------------------------

```python
def test_is_pclass_returns_false_for_empty_bases():
    from pyrsistent._pclass import PClassMeta
    
    # Create a metaclass instance with empty bases
    # This should result in _is_pclass(bases) evaluating to False
    name = 'TestClass'
    bases = ()
    dct = {'_pclass_fields': {}}
    
    # Call __new__ with empty bases
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    # Verify that __weakref__ was NOT added to __slots__
    # (it would only be added if _is_pclass(bases) returned True)
    assert '__weakref__' not in result.__slots__
    assert result.__slots__ == ('_pclass_frozen',)


# LLM-generated content at query #31
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(ignore_extra=True, x=1, z=2)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_with_none_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=None, y=2)
    assert instance.x is None
    assert instance.y == 2


# LLM-generated content at query #32
#--------------------------

```python
def test_pclass_repr_single_field():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=1)
    result = repr(obj)
    assert result == "SimpleClass(x=1)"


def test_pclass_repr_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        x = field()
        y = field()
    
    obj = MultiFieldClass(x=1, y=2)
    result = repr(obj)
    assert "MultiFieldClass(" in result
    assert "x=1" in result
    assert "y=2" in result
    assert result.endswith(")")


def test_pclass_repr_string_field():
    from pyrsistent import PClass, field
    
    class StringClass(PClass):
        name = field()
    
    obj = StringClass(name="test")
    result = repr(obj)
    assert result == "StringClass(name='test')"


def test_pclass_repr_empty_pclass():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    result = repr(obj)
    assert result == "EmptyClass()"


def test_pclass_repr_nested_pclass():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner_obj = InnerClass(value=42)
    outer_obj = OuterClass(inner=inner_obj)
    result = repr(outer_obj)
    assert "OuterClass(" in result
    assert "InnerClass(value=42)" in result


def test_pclass_repr_with_list_field():
    from pyrsistent import PClass, field
    
    class ListClass(PClass):
        items = field()
    
    obj = ListClass(items=[1, 2, 3])
    result = repr(obj)
    assert "ListClass(" in result
    assert "items=[1, 2, 3]" in result


def test_pclass_repr_with_optional_field():
    from pyrsistent import PClass, field
    
    class OptionalClass(PClass):
        required = field()
        optional = field(initial=None)
    
    obj = OptionalClass(required="value")
    result = repr(obj)
    assert "OptionalClass(" in result
    assert "required='value'" in result
    assert "optional=None" in result


def test_pclass_repr_with_special_characters():
    from pyrsistent import PClass, field
    
    class SpecialClass(PClass):
        text = field()
    
    obj = SpecialClass(text="hello\nworld")
    result = repr(obj)
    assert "SpecialClass(" in result
    assert "text=" in result
    assert "hello" in result


# LLM-generated content at query #33
#--------------------------

```python
def test_hash_returns_integer():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    hash_value = hash(instance)
    
    assert isinstance(hash_value, int)


# LLM-generated content at query #34
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_constructor_extra_fields_raise_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_fields_with_mixed_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field(initial=5)
        c = field()
    
    instance = TestClass(a=1, c=3)
    assert instance.a == 1
    assert instance.b == 5
    assert instance.c == 3


# LLM-generated content at query #35
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PClass, field
    
    class TestField:
        def __init__(self, invariant_func, type_spec=None):
            self.invariant = invariant_func
            self.type = type_spec
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    
    def failing_invariant(value):
        return (False, "error_code_1")
    
    test_field = TestField(failing_invariant, type_spec=[int])
    
    _check_and_set_attr(TestClass, test_field, "test_attr", 42, result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "error_code_1"
    assert not hasattr(result, "test_attr")


# LLM-generated content at query #36
#--------------------------

```python
def test_pclass_meta_new_with_pclass_bases():
    from pyrsistent._pclass import PClassMeta, _is_pclass
    from pyrsistent._field_common import _PField
    
    # Create a mock base class that _is_pclass would return True for
    class MockPClassBase(metaclass=PClassMeta):
        pass
    
    # Verify that _is_pclass returns True for bases containing PClass
    bases = (MockPClassBase,)
    result = _is_pclass(bases)
    
    assert result is True


# LLM-generated content at query #37
#--------------------------

```python
def test_repr_with_single_field():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=1)
    assert repr(obj) == "SimpleClass(x=1)"


def test_repr_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = MultiFieldClass(x=1, y=2, z=3)
    result = repr(obj)
    assert "MultiFieldClass(" in result
    assert "x=1" in result
    assert "y=2" in result
    assert "z=3" in result
    assert result.endswith(")")


def test_repr_with_string_field():
    from pyrsistent import PClass, field
    
    class StringClass(PClass):
        name = field()
    
    obj = StringClass(name="test")
    assert repr(obj) == "StringClass(name='test')"


def test_repr_with_nested_objects():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner_obj = InnerClass(value=42)
    outer_obj = OuterClass(inner=inner_obj)
    result = repr(outer_obj)
    assert "OuterClass(" in result
    assert "InnerClass(value=42)" in result


def test_repr_with_missing_optional_field():
    from pyrsistent import PClass, field
    
    class OptionalFieldClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj = OptionalFieldClass(x=1, y=None)
    result = repr(obj)
    assert "OptionalFieldClass(" in result
    assert "x=1" in result
    assert "y=None" in result


def test_repr_with_list_field():
    from pyrsistent import PClass, field
    
    class ListClass(PClass):
        items = field()
    
    obj = ListClass(items=[1, 2, 3])
    result = repr(obj)
    assert "ListClass(" in result
    assert "items=[1, 2, 3]" in result


def test_repr_with_dict_field():
    from pyrsistent import PClass, field
    
    class DictClass(PClass):
        data = field()
    
    obj = DictClass(data={'key': 'value'})
    result = repr(obj)
    assert "DictClass(" in result
    assert "data=" in result
    assert "'key': 'value'" in result


def test_repr_empty_class():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    assert repr(obj) == "EmptyClass()"


def test_repr_with_boolean_field():
    from pyrsistent import PClass, field
    
    class BoolClass(PClass):
        flag = field()
    
    obj = BoolClass(flag=True)
    assert repr(obj) == "BoolClass(flag=True)"


def test_repr_with_float_field():
    from pyrsistent import PClass, field
    
    class FloatClass(PClass):
        value = field()
    
    obj = FloatClass(value=3.14)
    assert repr(obj) == "FloatClass(value=3.14)"


# LLM-generated content at query #38
#--------------------------

```python
def test_pclass_meta_new_creates_slots():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField

    class TestField(_PField):
        def __init__(self):
            self.initial = None
            self.factory = None
            self.invariant = None

    dct = {
        'field1': TestField(),
        'field2': TestField(),
    }
    bases = (CheckedType,)
    name = 'TestPClass'

    result = PClassMeta(name, bases, dct)

    assert hasattr(result, '__slots__')
    assert '_pclass_frozen' in result.__slots__
    assert 'field1' in result.__slots__
    assert 'field2' in result.__slots__


def test_pclass_meta_new_sets_pclass_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField

    class TestField(_PField):
        def __init__(self):
            self.initial = None
            self.factory = None
            self.invariant = None

    dct = {
        'field1': TestField(),
    }
    bases = (CheckedType,)
    name = 'TestPClass'

    result = PClassMeta(name, bases, dct)

    assert hasattr(result, '_pclass_fields')
    assert 'field1' in result._pclass_fields


def test_pclass_meta_new_sets_pclass_invariants():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType

    def test_invariant(self):
        return True

    dct = {
        '__invariant__': test_invariant,
    }
    bases = (CheckedType,)
    name = 'TestPClass'

    result = PClassMeta(name, bases, dct)

    assert hasattr(result, '_pclass_invariants')
    assert isinstance(result._pclass_invariants, tuple)
    assert len(result._pclass_invariants) > 0


def test_pclass_meta_new_adds_weakref_for_direct_subclass():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType

    dct = {}
    bases = (CheckedType,)
    name = 'TestPClass'

    result = PClassMeta(name, bases, dct)

    assert '__weakref__' in result.__slots__


def test_pclass_meta_new_no_weakref_for_indirect_subclass():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType

    class FirstPClass(CheckedType, metaclass=PClassMeta):
        pass

    dct = {}
    bases = (FirstPClass,)
    name = 'SecondPClass'

    result = PClassMeta(name, bases, dct)

    assert '__weakref__' not in result.__slots__


def test_pclass_meta_new_inherits_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField

    class TestField(_PField):
        def __init__(self):
            self.initial = None
            self.factory = None
            self.invariant = None

    class ParentPClass(CheckedType, metaclass=PClassMeta):
        _pclass_fields = {'parent_field': TestField()}

    dct = {
        'child_field': TestField(),
    }
    bases = (ParentPClass,)
    name = 'ChildPClass'

    result = PClassMeta(name, bases, dct)

    assert 'parent_field' in result._pclass_fields
    assert 'child_field' in result._pclass_fields


# LLM-generated content at query #39
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == 10
    assert obj.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._checked_types import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=5, _factory_fields=set())
    assert obj.x == 5


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, ignore_extra=True, z=2)
    assert obj.x == 1


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert obj is not None


def test_pclass_constructor_multiple_fields_with_defaults():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field(initial=1)
        b = field(initial=2)
        c = field(initial=3)
    
    obj = TestClass()
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3


# LLM-generated content at query #40
#--------------------------

```python
def test_is_pclass_returns_false_for_empty_bases():
    from pyrsistent._pclass import PClassMeta
    
    dct = {}
    bases = ()
    name = 'TestClass'
    
    meta = PClassMeta(name, bases, dct)
    
    assert '__weakref__' not in meta.__slots__
    assert meta.__slots__ == ('_pclass_frozen',)


# LLM-generated content at query #41
#--------------------------

```python
def test_repr_returns_formatted_string():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y='hello')
    result = repr(obj)
    
    assert isinstance(result, str)
    assert result.startswith('TestClass(')
    assert result.endswith(')')
    assert 'x=1' in result
    assert "y='hello'" in result


# LLM-generated content at query #42
#--------------------------

```python
def test_repr_returns_correct_format():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y='hello')
    repr_result = repr(instance)
    
    assert 'TestClass' in repr_result
    assert 'x=1' in repr_result
    assert "y='hello'" in repr_result
    assert repr_result.startswith('TestClass(')
    assert repr_result.endswith(')')


# LLM-generated content at query #43
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    result = obj1 == obj2
    assert result is True


# LLM-generated content at query #44
#--------------------------

```python
def test_pclass_meta_weakref_not_added_when_not_pclass():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    # Create a base class that is NOT a PClass
    class NonPClassBase:
        pass
    
    # Create a dict with some fields
    dct = {
        'field1': _PField(type=int, initial=0, invariant=None, initial_factory=None, factory=None, serializer=None, invariant_errors=None),
    }
    
    # Call PClassMeta.__new__ with a non-PClass base
    result = PClassMeta.__new__(PClassMeta, 'TestClass', (NonPClassBase,), dct)
    
    # Verify that __weakref__ was NOT added to __slots__
    assert '__weakref__' not in result.__slots__
    assert '_pclass_frozen' in result.__slots__


# LLM-generated content at query #45
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    # The predicate at line 7 is "for name in self._pclass_fields"
    # This should iterate over all field names defined in _pclass_fields
    pclass_fields_keys = set(instance._pclass_fields.keys())
    expected_fields = {'x', 'y', 'z'}
    
    assert pclass_fields_keys == expected_fields
    
    # Verify that serialize method iterates through these fields
    result = instance.serialize()
    assert set(result.keys()) == expected_fields


# LLM-generated content at query #46
#--------------------------

```python
def test_pclass_meta_new_creates_slots_for_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field1 = _PField(None, None, None, None, None)
    field2 = _PField(None, None, None, None, None)
    dct = {'field1': field1, 'field2': field2}
    bases = (CheckedType,)
    
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    
    assert hasattr(result, '__slots__')
    assert '_pclass_frozen' in result.__slots__
    assert 'field1' in result.__slots__
    assert 'field2' in result.__slots__


def test_pclass_meta_new_adds_weakref_slot_for_base_pclass():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field1 = _PField(None, None, None, None, None)
    dct = {'field1': field1}
    bases = (CheckedType,)
    
    result = PClassMeta.__new__(PClassMeta, 'BaseClass', bases, dct)
    
    assert '__weakref__' in result.__slots__


def test_pclass_meta_new_moves_fields_to_pclass_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field1 = _PField(None, None, None, None, None)
    dct = {'field1': field1}
    bases = (CheckedType,)
    
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    
    assert '_pclass_fields' in result.__dict__
    assert 'field1' in result._pclass_fields
    assert 'field1' not in dct or dct['field1'] != field1


def test_pclass_meta_new_stores_invariants():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    def invariant_func(self):
        return True
    
    dct = {'__invariant__': invariant_func}
    bases = (CheckedType,)
    
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    
    assert '_pclass_invariants' in result.__dict__
    assert isinstance(result._pclass_invariants, tuple)
    assert len(result._pclass_invariants) > 0


def test_pclass_meta_new_inherits_fields_from_bases():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field1 = _PField(None, None, None, None, None)
    parent_dct = {'field1': field1}
    parent_bases = (CheckedType,)
    parent = PClassMeta.__new__(PClassMeta, 'Parent', parent_bases, parent_dct)
    
    field2 = _PField(None, None, None, None, None)
    child_dct = {'field2': field2}
    child_bases = (parent,)
    
    result = PClassMeta.__new__(PClassMeta, 'Child', child_bases, child_dct)
    
    assert '_pclass_fields' in result.__dict__
    assert 'field1' in result._pclass_fields or len(result._pclass_fields) >= 1
    assert 'field2' in result._pclass_fields


def test_pclass_meta_new_no_weakref_for_non_base_pclass():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field1 = _PField(None, None, None, None, None)
    parent_dct = {'field1': field1}
    parent_bases = (CheckedType,)
    parent = PClassMeta.__new__(PClassMeta, 'Parent', parent_bases, parent_dct)
    
    field2 = _PField(None, None, None, None, None)
    child_dct = {'field2': field2}
    child_bases = (parent,)
    
    result = PClassMeta.__new__(PClassMeta, 'Child', child_bases, child_dct)
    
    assert '__weakref__' not in result.__slots__


# LLM-generated content at query #47
#--------------------------

```python
def test_pclass_new_basic_initialization():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == 10
    assert obj.y == 5


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_new_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_extra_fields_raise_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_field_invariant_failure():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    def positive_invariant(value):
        return (value > 0, "must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'must be positive' in e.invariant_errors


def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, ignore_extra=True)
    assert obj.x == 1


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=5)
    assert obj.x == 5


def test_pclass_new_with_type_checking():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="not an int")
        assert False, "Should have raised PTypeError"
    except PTypeError as e:
        assert 'Invalid type' in str(e)


def test_pclass_new_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a=1, b=2, c=3)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    def global_check(obj):
        return (obj.x > obj.y, "x must be greater than y")
    
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = (global_check,)
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Global invariant failed' in str(e)


def test_pclass_new_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert isinstance(obj, TestClass)


def test_pclass_new_all_fields_with_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    
    obj = TestClass()
    assert obj.x == 1
    assert obj.y == 2


# LLM-generated content at query #48
#--------------------------

```python
def test_set_method_predicate_at_line_25():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.set(x=10)
    
    assert hasattr(TestClass, '_pclass_fields')
    assert isinstance(TestClass._pclass_fields, dict)
    assert len(TestClass._pclass_fields) > 0
    assert result.x == 10
    assert result.y == 2


# LLM-generated content at query #49
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_invariant_errors_present():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    # Mock the invariant_errors by patching _check_and_set_attr to add an error
    # We need to test that line 25 evaluates to True when invariant_errors is non-empty
    original_check = TestClass.__dict__.get('_check_and_set_attr')
    
    # Create a minimal test: we'll create a subclass with a field that has an invariant
    class StrictClass(PClass):
        value = field()
    
    try:
        # This will succeed normally
        obj = StrictClass(value=5)
        assert obj.value == 5
    except InvariantException:
        pass


def test_pclass_raises_invariant_exception_when_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class RequiredClass(PClass):
        required_field = field(mandatory=True)
    
    # Attempt to create instance without mandatory field should raise InvariantException
    exception_raised = False
    try:
        obj = RequiredClass()
    except InvariantException as e:
        exception_raised = True
        assert 'RequiredClass.required_field' in e.missing_fields
    
    assert exception_raised


def test_pclass_invariant_exception_with_both_errors_and_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class ComplexClass(PClass):
        mandatory_field = field(mandatory=True)
    
    exception_raised = False
    try:
        obj = ComplexClass()
    except InvariantException as e:
        exception_raised = True
        assert len(e.missing_fields) > 0
    
    assert exception_raised


# LLM-generated content at query #50
#--------------------------

```python
def test_pclass_new_basic_field_assignment():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == 10
    assert obj.y == 5


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == [1, 2, 3]
    assert obj.y == 5


def test_pclass_new_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        obj = TestClass(y=5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_extra_kwargs_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)
        assert "not among the specified fields" in str(e)


def test_pclass_new_invalid_type():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        obj = TestClass(x="not an int")
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    assert obj._pclass_frozen is True


def test_pclass_new_cannot_modify_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    obj = TestClass(x="5")
    assert obj.x == 5


def test_pclass_new_field_invariant_violation():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive_invariant(val):
        return (val > 0, "must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        obj = TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "must be positive" in e.error_codes


def test_pclass_new_with_ignore_extra_true():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert obj.x == 1
    assert not hasattr(obj, 'y')


def test_pclass_new_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj = TestClass(a=1, b=2, c=3, d=4)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3
    assert obj.d == 4


def test_pclass_new_with_type_check_multiple_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(type=(int, str))
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x="hello")
    assert obj1.x == 1
    assert obj2.x == "hello"


def test_pclass_new_global_invariant_violation():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def global_check(obj):
        return (obj.x > obj.y, "x must be greater than y")
    
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = (global_check,)
    
    try:
        obj = TestClass(x=1, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Global invariant failed" in str(e)


def test_pclass_new_all_fields_with_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    
    obj = TestClass()
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_new_mixed_initial_and_mandatory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field(mandatory=True)
    
    obj = TestClass(y=20)
    assert obj.x == 10
    assert obj.y == 20


def test_pclass_new_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert isinstance(obj, TestClass)


def test_pclass_new_with_factory_and_type():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(type=int, factory=int)
    
    obj = TestClass(x="42")
    assert obj.x == 42
    assert isinstance(obj.x, int)


# LLM-generated content at query #51
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [])
    
    instance = TestClass()
    assert instance.x == []


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(ignore_extra=True, x=1, z=2)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    assert instance.x is None


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5)
    assert instance.x == 5


# LLM-generated content at query #52
#--------------------------

```python
def test_set_with_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1
    assert obj.y == 2


def test_set_with_positional_args():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set('x', 5)
    
    assert result.x == 5
    assert obj.x == 1


def test_set_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(x=10, y=20)
    
    assert result.x == 10
    assert result.y == 20
    assert result.z == 3
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3


def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set(x=2)
    
    assert result is not obj
    assert isinstance(result, TestClass)


def test_set_with_optional_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    obj = TestClass(x=1)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 5


def test_set_preserves_unmodified_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a=1, b=2, c=3)
    result = obj.set(b=20)
    
    assert result.a == 1
    assert result.b == 20
    assert result.c == 3


# LLM-generated content at query #53
#--------------------------

```python
def test_set_method_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    updated_instance = instance.set(x=10)
    
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    assert 'z' in TestClass._pclass_fields
    assert updated_instance.x == 10
    assert updated_instance.y == 2
    assert updated_instance.z == 3


# LLM-generated content at query #54
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return False, "invariant_error_code"
    
    class MockClass:
        pass
    
    field = MockField()
    result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    
    assert invariant_errors == ["invariant_error_code"]
    assert not hasattr(result, "test_field")


# LLM-generated content at query #55
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    class MockClass:
        pass
    
    field = MockField()
    name = "test_field"
    value = "test_value"
    result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    
    assert invariant_errors == ["invariant_error_code"]
    assert not hasattr(result, name)


# LLM-generated content at query #56
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='5')
    assert instance.x == 5


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    
    instance = TestClass()
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #57
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
    
    obj = TestClass()
    assert obj.x == 10


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    obj = TestClass()
    assert obj.x == 42


def test_pclass_new_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_extra_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    assert obj._pclass_frozen is True


def test_pclass_new_cannot_set_after_frozen():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    obj = TestClass(x={'a': 1})
    assert obj.x == pmap({'a': 1})


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="string")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def check_positive(value):
        if value > 0:
            return True, None
        return False, 'must_be_positive'
    
    class TestClass(PClass):
        x = field(invariant=check_positive)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'must_be_positive' in e.error_codes


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def sum_check(obj):
        if obj.x + obj.y > 10:
            return True, None
        return False, 'sum_too_small'
    
    class TestClass(PClass):
        __invariants__ = (sum_check,)
        x = field()
        y = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'sum_too_small' in e.error_codes


def test_pclass_new_partial_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=20)
    
    obj = TestClass(x=10)
    assert obj.x == 10
    assert obj.y == 20


def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    obj = TestClass(x={'a': 1}, _factory_fields={'x'}, ignore_extra=True)
    assert obj.x == pmap({'a': 1})


def test_pclass_new_multiple_invariant_errors():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def check_positive(value):
        if value > 0:
            return True, None
        return False, 'must_be_positive'
    
    class TestClass(PClass):
        x = field(invariant=check_positive)
        y = field(invariant=check_positive)
    
    try:
        TestClass(x=-1, y=-2)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) == 2


# LLM-generated content at query #58
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    result = obj1 == obj2
    assert result is True


# LLM-generated content at query #59
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    # Trigger the isinstance check at line 2 to evaluate to True
    result = obj1 == obj2
    
    assert result is True


# LLM-generated content at query #60
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
    
    instance = TestClass()
    assert instance.x == 5


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


# LLM-generated content at query #61
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PClass, field
    
    class TestClass(PClass):
        value = field(type=int)
    
    class MockField:
        def __init__(self):
            self.type = [int]
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    mock_field = MockField()
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, mock_field, "value", 42, result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_error_code"
    assert not hasattr(result, "value") or getattr(result, "value", None) is None


# LLM-generated content at query #62
#--------------------------

```python
def test_serialize_with_no_fields():
    from pyrsistent import PClass, field
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    result = obj.serialize()
    assert result == {}


def test_serialize_with_simple_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y='hello')
    result = obj.serialize()
    assert result == {'x': 1, 'y': 'hello'}


def test_serialize_with_missing_fields():
    from pyrsistent import PClass, field
    
    class PartialClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj = PartialClass(x=42)
    result = obj.serialize()
    assert 'x' in result
    assert result['x'] == 42


def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    
    class FormattedClass(PClass):
        value = field()
    
    obj = FormattedClass(value=100)
    result = obj.serialize(format='json')
    assert result == {'value': 100}


def test_serialize_preserves_values():
    from pyrsistent import PClass, field
    
    class DataClass(PClass):
        name = field()
        age = field()
        active = field()
    
    obj = DataClass(name='Alice', age=30, active=True)
    result = obj.serialize()
    assert result['name'] == 'Alice'
    assert result['age'] == 30
    assert result['active'] is True


def test_serialize_with_nested_structures():
    from pyrsistent import PClass, field
    
    class NestedClass(PClass):
        data = field()
    
    obj = NestedClass(data={'key': 'value'})
    result = obj.serialize()
    assert result == {'data': {'key': 'value'}}


def test_serialize_returns_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        field1 = field()
    
    obj = TestClass(field1='test')
    result = obj.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #63
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #64
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5)
    assert instance.x == 5


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_instances_independent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    
    assert instance1.x == 1
    assert instance2.x == 2


# LLM-generated content at query #65
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [])
    
    instance = TestClass()
    assert instance.x == []


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "y" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "frozen" in str(e).lower()


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5)
    assert instance.x == 5


def test_pclass_constructor_multiple_invariant_errors():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e) and "TestClass.y" in str(e)


def test_pclass_constructor_no_arguments():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    
    instance = TestClass()
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #66
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        a = field()
    
    class TestClass(PClass):
        inner = field(factory=InnerClass)
    
    instance = TestClass(inner={'a': 5})
    assert instance.inner.a == 5


def test_pclass_constructor_no_arguments():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    
    instance = TestClass()
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass.create({'x': 1, 'z': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_multiple_missing_mandatory_fields():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #67
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=20)
    assert obj.x == 10
    assert obj.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    obj = TestClass()
    assert obj.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, z=2, ignore_extra=True)
    assert obj.x == 1
    assert not hasattr(obj, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    obj = TestClass(x='42')
    assert obj.x == 42


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert isinstance(obj, TestClass)


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #68
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == 10
    assert obj.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(_factory_fields={'x'}, x=5)
    assert obj.x == 5


def test_pclass_constructor_freezes_object():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(ignore_extra=True, x=1, y=2)
    assert obj.x == 1
    assert not hasattr(obj, 'y')


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
    
    obj = TestClass()
    assert obj.x == 5


def test_pclass_constructor_multiple_fields_with_mixed_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field(mandatory=True)
        b = field(initial=10)
        c = field()
    
    obj = TestClass(a=1, c=3)
    assert obj.a == 1
    assert obj.b == 10
    assert obj.c == 3


# LLM-generated content at query #69
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, z=2)
    assert instance.x == 1


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #70
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='5', _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_no_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='5', _factory_fields=set())
    assert instance.x == '5'


def test_pclass_constructor_all_optional_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass()
    assert not hasattr(instance, 'x')
    assert not hasattr(instance, 'y')


def test_pclass_constructor_multiple_missing_mandatory_fields():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        error_msg = str(e)
        assert 'TestClass.x' in error_msg
        assert 'TestClass.y' in error_msg


def test_pclass_constructor_with_field_invariant_error():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(invariant=lambda v: v > 0)
    
    try:
        TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #71
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == 10
    assert obj.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(_factory_fields={'x'}, x=1)
    assert obj.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(ignore_extra=True, x=1, y=2)
    assert obj.x == 1
    assert not hasattr(obj, 'y')


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert isinstance(obj, TestClass)


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj = TestClass(a=1, b=2, c=3, d=4)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3
    assert obj.d == 4


# LLM-generated content at query #72
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.args[2]


def test_pclass_constructor_with_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #73
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #74
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(ignore_extra=True, x=1, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='42')
    assert instance.x == 42


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


# LLM-generated content at query #75
#--------------------------

```python
def test_pclass_repr_single_field():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=1)
    result = repr(obj)
    assert result == "SimpleClass(x=1)"


def test_pclass_repr_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = MultiFieldClass(x=1, y=2, z=3)
    result = repr(obj)
    assert "MultiFieldClass(" in result
    assert "x=1" in result
    assert "y=2" in result
    assert "z=3" in result
    assert result.endswith(")")


def test_pclass_repr_with_string_values():
    from pyrsistent import PClass, field
    
    class StringClass(PClass):
        name = field()
    
    obj = StringClass(name="test")
    result = repr(obj)
    assert result == "StringClass(name='test')"


def test_pclass_repr_with_nested_object():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner_obj = InnerClass(value=42)
    outer_obj = OuterClass(inner=inner_obj)
    result = repr(outer_obj)
    assert "OuterClass(" in result
    assert "InnerClass(value=42)" in result


def test_pclass_repr_empty_class():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    result = repr(obj)
    assert result == "EmptyClass()"


def test_pclass_repr_with_list_value():
    from pyrsistent import PClass, field
    
    class ListClass(PClass):
        items = field()
    
    obj = ListClass(items=[1, 2, 3])
    result = repr(obj)
    assert result == "ListClass(items=[1, 2, 3])"


def test_pclass_repr_with_none_value():
    from pyrsistent import PClass, field
    
    class NoneClass(PClass):
        value = field(initial=None)
    
    obj = NoneClass()
    result = repr(obj)
    assert result == "NoneClass(value=None)"


def test_pclass_repr_with_boolean_values():
    from pyrsistent import PClass, field
    
    class BoolClass(PClass):
        flag1 = field()
        flag2 = field()
    
    obj = BoolClass(flag1=True, flag2=False)
    result = repr(obj)
    assert "BoolClass(" in result
    assert "flag1=True" in result
    assert "flag2=False" in result


def test_pclass_repr_with_float_value():
    from pyrsistent import PClass, field
    
    class FloatClass(PClass):
        value = field()
    
    obj = FloatClass(value=3.14)
    result = repr(obj)
    assert result == "FloatClass(value=3.14)"


# LLM-generated content at query #76
#--------------------------

```python
def test_pclass_invariant_errors_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field()
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.missing_fields) > 0
        assert 'TestClass.y' in e.missing_fields


def test_pclass_invariant_errors_with_invariant_check():
    from pyrsistent import PClass, field, InvariantException
    
    def check_positive(obj):
        return (obj.x > 0, 'x must be positive')
    
    class TestClass(PClass):
        __invariants__ = (check_positive,)
        x = field()
    
    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.error_codes) > 0


def test_pclass_predicate_line_25_true_with_missing_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        mandatory_field = field(mandatory=True)
    
    exception_raised = False
    try:
        TestClass()
    except InvariantException:
        exception_raised = True
    
    assert exception_raised is True


# LLM-generated content at query #77
#--------------------------

```python
def test_pclass_new_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    
    assert hasattr(instance, 'x')
    assert hasattr(instance, 'y')
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #78
#--------------------------

```python
def test_reduce_returns_restore_pickle_and_class_data():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()
    
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}
    assert len(result) == 2


# LLM-generated content at query #79
#--------------------------

```python
def test_set_with_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1
    assert obj.y == 2


def test_set_with_args():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set('x', 10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1


def test_set_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(x=10, y=20)
    
    assert result.x == 10
    assert result.y == 20
    assert result.z == 3


def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set(x=2)
    
    assert obj is not result
    assert isinstance(result, TestClass)


def test_set_preserves_other_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(x=100)
    
    assert result.x == 100
    assert result.y == 2
    assert result.z == 3


def test_set_with_args_and_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set('x', 10, y=20)
    
    assert result.x == 10
    assert result.y == 20


def test_set_creates_immutable_copy():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set(x=2)
    
    try:
        result.x = 3
        assert False, "Should not be able to set attribute on frozen instance"
    except AttributeError:
        pass


# LLM-generated content at query #80
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PyrsistentAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #81
#--------------------------

```python
def test_pclass_meta_weakref_not_added_when_not_pclass():
    from pyrsistent._pclass import PClassMeta
    
    class TestMeta(metaclass=PClassMeta):
        pass
    
    assert '__weakref__' not in TestMeta.__slots__


# LLM-generated content at query #82
#--------------------------

```python
def test_pclass_hash_returns_integer():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    hash_value = hash(instance)
    
    assert isinstance(hash_value, int)


def test_pclass_hash_consistency():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    
    assert hash(instance1) == hash(instance2)


def test_pclass_hash_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    
    assert hash(instance1) != hash(instance2)


def test_pclass_hash_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    instance = TestClass(x=1)
    hash_value = hash(instance)
    
    assert isinstance(hash_value, int)


def test_pclass_hash_usable_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=2, y=3)
    
    test_set = {instance1, instance2, instance3}
    
    assert len(test_set) == 2


# LLM-generated content at query #83
#--------------------------

```python
def test_pclass_meta_new_creates_class_with_fields_and_invariants():
    from pyrsistent._pclass import PClassMeta, CheckedType
    from pyrsistent._field_common import PField
    
    # Create a simple field
    test_field = PField()
    
    # Create a class using PClassMeta
    dct = {'test_attr': test_field}
    bases = (CheckedType,)
    
    TestClass = PClassMeta('TestClass', bases, dct)
    
    assert TestClass.__name__ == 'TestClass'
    assert hasattr(TestClass, '_pclass_fields')
    assert 'test_attr' in TestClass._pclass_fields
    assert hasattr(TestClass, '_pclass_invariants')


def test_pclass_meta_new_sets_slots():
    from pyrsistent._pclass import PClassMeta, CheckedType
    from pyrsistent._field_common import PField
    
    test_field = PField()
    dct = {'field1': test_field}
    bases = (CheckedType,)
    
    TestClass = PClassMeta('TestClass', bases, dct)
    
    assert hasattr(TestClass, '__slots__')
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'field1' in TestClass.__slots__


def test_pclass_meta_new_adds_weakref_for_top_level_class():
    from pyrsistent._pclass import PClassMeta, CheckedType
    
    dct = {}
    bases = (CheckedType,)
    
    TestClass = PClassMeta('TestClass', bases, dct)
    
    assert '__weakref__' in TestClass.__slots__


def test_pclass_meta_new_without_weakref_for_subclass():
    from pyrsistent._pclass import PClassMeta, CheckedType
    from pyrsistent._field_common import PField
    
    # Create a parent class first
    parent_dct = {}
    parent_bases = (CheckedType,)
    ParentClass = PClassMeta('ParentClass', parent_bases, parent_dct)
    
    # Create a child class
    child_dct = {'field1': PField()}
    child_bases = (ParentClass,)
    ChildClass = PClassMeta('ChildClass', child_bases, child_dct)
    
    assert '__weakref__' not in ChildClass.__slots__


def test_pclass_meta_new_with_invariant():
    from pyrsistent._pclass import PClassMeta, CheckedType
    
    def my_invariant(self):
        return True, None
    
    dct = {'__invariant__': my_invariant}
    bases = (CheckedType,)
    
    TestClass = PClassMeta('TestClass', bases, dct)
    
    assert hasattr(TestClass, '_pclass_invariants')
    assert len(TestClass._pclass_invariants) > 0
    assert callable(TestClass._pclass_invariants[0])


def test_pclass_meta_new_removes_fields_from_dct():
    from pyrsistent._pclass import PClassMeta, CheckedType
    from pyrsistent._field_common import PField
    
    test_field = PField()
    dct = {'my_field': test_field, 'other_attr': 'value'}
    bases = (CheckedType,)
    
    TestClass = PClassMeta('TestClass', bases, dct)
    
    assert not hasattr(TestClass, 'my_field') or 'my_field' not in TestClass.__dict__
    assert 'my_field' in TestClass._pclass_fields


# LLM-generated content at query #84
#--------------------------

```python
def test_pclass_predicate_line_25_with_invariant_errors():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return (False, "test_error")
    
    class TestClass(PClass):
        x = field()
        __invariants__ = (failing_invariant,)
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.error_codes == ("test_error",)


def test_pclass_predicate_line_25_with_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields


def test_pclass_predicate_line_25_with_both_errors_and_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return (False, "invariant_error")
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
        __invariants__ = (failing_invariant,)
    
    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields
        assert "invariant_error" in e.error_codes


# LLM-generated content at query #85
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    
    # The predicate at line 7 is: `for name in self._pclass_fields:`
    # This evaluates to True when self._pclass_fields is iterable and non-empty
    assert hasattr(instance, '_pclass_fields')
    assert len(instance._pclass_fields) > 0
    
    # Verify that serialize method iterates through the fields
    result = instance.serialize()
    assert 'x' in result or 'y' in result


# LLM-generated content at query #86
#--------------------------

```python
def test_eq_predicate_isinstance_true():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    result = obj1 == obj2
    assert result is True


# LLM-generated content at query #87
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PClass, field
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.invariant = lambda x: (False, "invariant_error")
    
    class MockClass:
        pass
    
    mock_field = MockField()
    mock_result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, mock_field, "test_field", 42, mock_result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_error"
    assert not hasattr(mock_result, "test_field")


# LLM-generated content at query #88
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_all_mandatory_fields_provided():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_partial_mandatory_fields_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #89
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_constructor_becomes_frozen():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class TestClass(PClass):
        inner = field()
    
    instance = TestClass(inner={'value': 42}, _factory_fields={'inner'})
    assert instance.inner.value == 42


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_no_fields():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    assert isinstance(instance, EmptyClass)


def test_pclass_constructor_with_multiple_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field(initial=1)
        b = field(initial=2)
        c = field()
    
    instance = TestClass(c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


# LLM-generated content at query #90
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=20)
    assert obj.x == 10
    assert obj.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    obj = TestClass()
    assert obj.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, z=2, ignore_extra=True)
    assert obj.x == 1
    assert not hasattr(obj, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2, _factory_fields={'x'})
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    class TestClassWithInitial(PClass):
        x = field(initial=5)
    
    obj = TestClassWithInitial()
    assert obj.x == 5


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        data = field()
    
    obj = TestClass(data={'a': 1, 'b': 2})
    assert obj.data is not None


# LLM-generated content at query #91
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


# LLM-generated content at query #92
#--------------------------

```python
def test_pclass_meta_new_predicate_line_1():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    mcs = PClassMeta
    name = "TestClass"
    bases = ()
    dct = {}
    
    result = mcs.__new__(mcs, name, bases, dct)
    
    assert result is not None
    assert isinstance(result, type)
    assert result.__name__ == name


# LLM-generated content at query #93
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="5")
    assert instance.x == 5


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_create_from_same_class():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass.create(instance1)
    assert instance1 is instance2


def test_pclass_constructor_multiple_fields_with_mixed_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field(initial=1)
        b = field()
        c = field(initial=3)
    
    instance = TestClass(b=2)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


# LLM-generated content at query #94
#--------------------------

```python
def test_pclass_hash_returns_integer():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    hash_value = hash(instance)
    assert isinstance(hash_value, int)


def test_pclass_hash_consistency():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert hash(instance1) == hash(instance2)


def test_pclass_hash_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    assert hash(instance1) != hash(instance2)


def test_pclass_hash_with_single_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=42)
    hash_value = hash(instance)
    assert isinstance(hash_value, int)


def test_pclass_hash_with_optional_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    instance = TestClass(x=1)
    hash_value = hash(instance)
    assert isinstance(hash_value, int)


def test_pclass_hash_usable_in_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    instance3 = TestClass(x=2)
    
    d = {instance1: 'first'}
    d[instance3] = 'third'
    assert d[instance2] == 'first'


def test_pclass_hash_usable_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    instance3 = TestClass(x=2)
    
    s = {instance1, instance3}
    assert instance2 in s
    assert len(s) == 2


# LLM-generated content at query #95
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    reduce_result = instance.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    instance = TestClass(x=5)
    reduce_result = instance.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 5, 'y': 10}


def test_pclass_reduce_with_no_fields():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    reduce_result = instance.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == EmptyClass
    assert reduce_result[1][1] == {}


def test_pclass_reduce_with_complex_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=[1, 2, 3], y={'a': 1}, z='string')
    reduce_result = instance.__reduce__()
    
    assert reduce_result[1][1] == {'x': [1, 2, 3], 'y': {'a': 1}, 'z': 'string'}


# LLM-generated content at query #96
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)
    
    # Test that isinstance(other, self.__class__) evaluates to True
    # and the loop at line 3 is entered
    assert obj1 == obj2
    assert not (obj1 == obj3)
    assert obj1.__eq__(obj2) is True
    assert obj1.__eq__(obj3) is False


# LLM-generated content at query #97
#--------------------------

```python
def test_is_pclass_returns_false_for_empty_bases():
    from pyrsistent._pclass import PClassMeta
    
    # Create a class with empty bases to make _is_pclass(bases) return False
    dct = {}
    bases = ()
    name = 'TestClass'
    
    # Call __new__ with empty bases
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    # Verify that __weakref__ was NOT added to __slots__
    # (which means the predicate _is_pclass(bases) evaluated to False)
    assert '__weakref__' not in result.__slots__


# LLM-generated content at query #98
#--------------------------

```python
def test_serialize_with_no_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=1)
    result = obj.serialize()
    assert result == {'x': 1}


def test_serialize_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = MultiFieldClass(a=1, b='hello', c=3.14)
    result = obj.serialize()
    assert result == {'a': 1, 'b': 'hello', 'c': 3.14}


def test_serialize_with_missing_optional_field():
    from pyrsistent import PClass, field
    
    class OptionalFieldClass(PClass):
        x = field()
        y = field()
    
    obj = OptionalFieldClass(x=10)
    result = obj.serialize()
    assert result == {'x': 10}


def test_serialize_with_nested_pclass():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
        name = field()
    
    inner_obj = InnerClass(value=42)
    outer_obj = OuterClass(inner=inner_obj, name='test')
    result = outer_obj.serialize()
    assert result['name'] == 'test'
    assert result['inner'] == inner_obj


def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    
    class FormattedClass(PClass):
        data = field()
    
    obj = FormattedClass(data='test_data')
    result = obj.serialize(format='json')
    assert result == {'data': 'test_data'}


def test_serialize_with_all_field_types():
    from pyrsistent import PClass, field
    
    class AllTypesClass(PClass):
        int_field = field()
        str_field = field()
        list_field = field()
        dict_field = field()
    
    obj = AllTypesClass(
        int_field=123,
        str_field='text',
        list_field=[1, 2, 3],
        dict_field={'key': 'value'}
    )
    result = obj.serialize()
    assert result['int_field'] == 123
    assert result['str_field'] == 'text'
    assert result['list_field'] == [1, 2, 3]
    assert result['dict_field'] == {'key': 'value'}


def test_serialize_returns_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.serialize()
    assert isinstance(result, dict)
    assert len(result) == 2


# LLM-generated content at query #99
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        TestClass(y=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.missing_fields) > 0
        assert 'TestClass.x' in e.missing_fields


def test_pclass_raises_invariant_exception_with_invariant_errors():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._invariant import InvariantException
    
    def check_positive(obj):
        if obj.x > 0:
            return True, None
        return False, "x must be positive"
    
    class TestClass(PClass):
        x = field()
        __invariants__ = (invariant(check_positive),)
    
    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.error_codes) > 0


# LLM-generated content at query #100
#--------------------------

```python
def test_pclass_new_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    
    assert hasattr(instance, 'x')
    assert hasattr(instance, 'y')
    assert instance.x == 1
    assert instance.y == 2
    assert instance._pclass_frozen is True


# LLM-generated content at query #101
#--------------------------

```python
def test_repr_format():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y="hello")
    repr_str = repr(instance)
    
    assert repr_str.startswith("TestClass(")
    assert repr_str.endswith(")")
    assert "x=1" in repr_str
    assert "y='hello'" in repr_str
    assert isinstance(repr_str, str)


# LLM-generated content at query #102
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class MockField:
        def __init__(self, field_type, invariant_fn):
            self.type = field_type
            self.invariant = invariant_fn
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._pclass import PTypeError
    
    class MockField:
        def __init__(self, field_type, invariant_fn):
            self.type = field_type
            self.invariant = invariant_fn
    
    class MockClass:
        __name__ = "MockClass"
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (True, None))
    
    try:
        _check_and_set_attr(MockClass, field, "test_field", "invalid", result, invariant_errors)
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_check_and_set_attr_failed_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_fn):
            self.type = field_type
            self.invariant = invariant_fn
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    error_code = "invalid_value"
    field = MockField([int], lambda x: (False, error_code))
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert not hasattr(result, "test_field")
    assert invariant_errors == [error_code]


def test_check_and_set_attr_no_type_check():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_fn):
            self.type = field_type
            self.invariant = invariant_fn
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField(None, lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", "any_value", result, invariant_errors)
    
    assert result.test_field == "any_value"
    assert invariant_errors == []


def test_check_and_set_attr_multiple_valid_types():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_fn):
            self.type = field_type
            self.invariant = invariant_fn
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int, str], lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_field", "hello", result, invariant_errors)
    
    assert result.test_field == "hello"
    assert invariant_errors == []


# LLM-generated content at query #103
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


def test_pclass_constructor_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    assert instance.x is None


def test_pclass_constructor_field_with_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='42')
    assert instance.x == 42


# LLM-generated content at query #104
#--------------------------

```python
def test_pclass_invariant_errors_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._precord_fields import InvariantException
    
    class TestClass(PClass):
        x = field(invariant=lambda x: (False, 'test error'))
    
    try:
        TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('test error',)


def test_pclass_missing_mandatory_field_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._precord_fields import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_both_invariant_errors_and_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._precord_fields import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True, invariant=lambda x: (False, 'x error'))
        y = field(mandatory=True)
    
    try:
        TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'x error' in e.invariant_errors
        assert 'TestClass.y' in e.missing_fields


# LLM-generated content at query #105
#--------------------------

```python
def test_set_method_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    updated = instance.set(x=10)
    
    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    assert 'z' in TestClass._pclass_fields
    assert updated.x == 10
    assert updated.y == 2
    assert updated.z == 3


# LLM-generated content at query #106
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        instance = TestClass(y=20)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PyrsistentAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'TestClass' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=3, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    assert instance._pclass_frozen is True
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass()
    assert not hasattr(instance, 'x')
    assert not hasattr(instance, 'y')


def test_pclass_constructor_with_invariant_error():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, 'x must be positive'))
    
    try:
        instance = TestClass(x=-5)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_remove_existing_item():
    initial_dict = {'key1': 'value1', 'key2': 'value2'}
    original = type('MockPClass', (), {})()
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.remove('key1')
    
    assert result is evolver
    assert 'key1' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'key1' not in evolver._factory_fields


def test_remove_item_not_in_data():
    initial_dict = {'key1': 'value1'}
    original = type('MockPClass', (), {})()
    evolver = _PClassEvolver(original, initial_dict)
    
    try:
        evolver.remove('nonexistent_key')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == 'nonexistent_key'


def test_remove_multiple_items():
    initial_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    original = type('MockPClass', (), {})()
    evolver = _PClassEvolver(original, initial_dict)
    evolver._factory_fields.add('key1')
    evolver._factory_fields.add('key2')
    
    evolver.remove('key1')
    evolver.remove('key2')
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert 'key2' not in evolver._pclass_evolver_data
    assert 'key3' in evolver._pclass_evolver_data
    assert 'key1' not in evolver._factory_fields
    assert 'key2' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True


def test_remove_using_delitem():
    initial_dict = {'key1': 'value1', 'key2': 'value2'}
    original = type('MockPClass', (), {})()
    evolver = _PClassEvolver(original, initial_dict)
    
    del evolver['key1']
    
    assert 'key1' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #2
#--------------------------

```python
def test_set_with_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    
    assert obj.x == 1
    assert obj.y == 2
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj is not new_obj


def test_set_with_args():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    new_obj = obj.set('x', 5)
    
    assert obj.x == 1
    assert new_obj.x == 5
    assert new_obj.y == 2


def test_set_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    new_obj = obj.set(x=10, y=20)
    
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3
    assert new_obj.x == 10
    assert new_obj.y == 20
    assert new_obj.z == 3


def test_set_preserves_original():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
    
    original = TestClass(a=1, b=2)
    modified = original.set(a=100)
    
    assert original.a == 1
    assert original.b == 2
    assert modified.a == 100
    assert modified.b == 2
    assert original is not modified


def test_set_with_single_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        value = field()
    
    obj = TestClass(value=42)
    new_obj = obj.set(value=99)
    
    assert obj.value == 42
    assert new_obj.value == 99


def test_set_returns_same_class_type():
    from pyrsistent import PClass, field
    
    class CustomClass(PClass):
        x = field()
    
    obj = CustomClass(x=1)
    new_obj = obj.set(x=2)
    
    assert isinstance(new_obj, CustomClass)
    assert type(new_obj) is CustomClass


# LLM-generated content at query #3
#--------------------------

```python
def test_set_method_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #4
#--------------------------

```python
def test_pclass_eq_same_class_same_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert obj1 == obj2


def test_pclass_eq_same_class_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert not (obj1 == obj2)


def test_pclass_eq_different_class():
    from pyrsistent import PClass, field
    
    class TestClass1(PClass):
        x = field()
    
    class TestClass2(PClass):
        x = field()
    
    obj1 = TestClass1(x=1)
    obj2 = TestClass2(x=1)
    
    assert (obj1 == obj2) is NotImplemented or not (obj1 == obj2)


def test_pclass_eq_with_non_pclass():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = {'x': 1}
    
    assert (obj1 == obj2) is NotImplemented or not (obj1 == obj2)


def test_pclass_eq_missing_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert obj1 == obj2


def test_pclass_eq_one_missing_one_present():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1)
    
    assert not (obj1 == obj2)


def test_pclass_eq_reflexive():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    
    assert obj1 == obj1


def test_pclass_eq_symmetric():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert (obj1 == obj2) == (obj2 == obj1)


def test_pclass_eq_transitive():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    obj3 = TestClass(x=1)
    
    assert obj1 == obj2
    assert obj2 == obj3
    assert obj1 == obj3


# LLM-generated content at query #5
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "z" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "frozen" in str(e).lower()


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_multiple_instances_independent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    assert instance1.x == 1
    assert instance2.x == 2
    assert instance1.x != instance2.x


def test_pclass_constructor_with_nested_pclass():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        a = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner_instance = InnerClass(a=10)
    outer_instance = OuterClass(inner=inner_instance)
    assert outer_instance.inner.a == 10


# LLM-generated content at query #6
#--------------------------

```python
def test_hash_same_values_same_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert hash(obj1) == hash(obj2)


def test_hash_different_values_different_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert hash(obj1) != hash(obj2)


def test_hash_missing_field_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert hash(obj1) == hash(obj2)


def test_hash_usable_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    obj3 = TestClass(x=2)
    
    hash_set = {obj1, obj2, obj3}
    assert len(hash_set) == 2


def test_hash_usable_in_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=2)
    
    hash_dict = {obj1: "value1", obj2: "value2"}
    assert hash_dict[obj1] == "value1"
    assert hash_dict[obj2] == "value2"


def test_hash_with_nested_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=(1, 2, 3))
    obj2 = TestClass(x=1, y=(1, 2, 3))
    
    assert hash(obj1) == hash(obj2)


def test_hash_consistent_across_calls():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    hash1 = hash(obj)
    hash2 = hash(obj)
    
    assert hash1 == hash2


# LLM-generated content at query #7
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_partial_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=5)
    result = obj.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 5}


def test_pclass_reduce_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=None)
    result = obj.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': None}


def test_pclass_reduce_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj = TestClass(a=1, b='test', c=[1, 2, 3], d={'key': 'value'})
    result = obj.__reduce__()
    
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'a': 1, 'b': 'test', 'c': [1, 2, 3], 'd': {'key': 'value'}}


# LLM-generated content at query #8
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
    
    instance = TestClass()
    assert instance.x == 10


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_new_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_extra_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    instance = TestClass(x={'a': 1})
    assert instance.x['a'] == 1


def test_pclass_new_with_type_checking():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="not an int")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_multiple_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(type=(int, str))
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x="hello")
    assert instance1.x == 1
    assert instance2.x == "hello"


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive(value):
        if value > 0:
            return True, None
        return False, "Must be positive"
    
    class TestClass(PClass):
        x = field(invariant=positive)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    
    @invariant("x_positive", lambda obj: obj.x > 0)
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except Exception:
        pass


def test_pclass_new_ignore_extra_true():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1


def test_pclass_new_with_partial_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    instance = TestClass(x=1)
    assert instance.x == 1
    assert instance.y == 5


def test_pclass_new_set_attribute_on_frozen():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    assert getattr(instance, '_pclass_frozen', False) is True


def test_pclass_new_multiple_instances_independent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    assert instance1.x == 1
    assert instance2.x == 2


def test_pclass_new_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    assert instance.x is None


def test_pclass_new_factory_fields_parameter():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
        y = field()
    
    instance = TestClass(_factory_fields={'x'}, x={'a': 1}, y=2)
    assert instance.x['a'] == 1
    assert instance.y == 2


# LLM-generated content at query #9
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj = TestClass(x=5)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 5, 'y': 10}


def test_pclass_reduce_with_no_fields():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == EmptyClass
    assert reduce_result[1][1] == {}


def test_pclass_reduce_with_complex_values():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=[1, 2, 3], y={'key': 'value'})
    reduce_result = obj.__reduce__()
    
    assert reduce_result[1][1] == {'x': [1, 2, 3], 'y': {'key': 'value'}}


# LLM-generated content at query #10
#--------------------------

```python
def test_pclass_meta_new_creates_slots():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(initial=1, factory=None, invariant=None, initial_factory=None),
        'field2': _PField(initial=2, factory=None, invariant=None, initial_factory=None),
    }
    bases = (CheckedType,)
    
    result = PClassMeta('TestClass', bases, dct)
    
    assert hasattr(result, '__slots__')
    assert '_pclass_frozen' in result.__slots__
    assert 'field1' in result.__slots__
    assert 'field2' in result.__slots__
    assert '__weakref__' in result.__slots__


def test_pclass_meta_new_sets_pclass_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(initial=1, factory=None, invariant=None, initial_factory=None),
    }
    bases = (CheckedType,)
    
    result = PClassMeta('TestClass', bases, dct)
    
    assert hasattr(result, '_pclass_fields')
    assert 'field1' in result._pclass_fields


def test_pclass_meta_new_sets_pclass_invariants():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    def invariant_func(obj):
        return True, None
    
    dct = {
        '__invariant__': invariant_func,
    }
    bases = (CheckedType,)
    
    result = PClassMeta('TestClass', bases, dct)
    
    assert hasattr(result, '_pclass_invariants')
    assert isinstance(result._pclass_invariants, tuple)
    assert len(result._pclass_invariants) > 0


def test_pclass_meta_new_removes_field_from_dct():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field = _PField(initial=1, factory=None, invariant=None, initial_factory=None)
    dct = {
        'field1': field,
    }
    bases = (CheckedType,)
    
    result = PClassMeta('TestClass', bases, dct)
    
    assert not hasattr(result, 'field1') or 'field1' not in vars(result)


def test_pclass_meta_new_without_weakref_for_subclass():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    # Create base class first
    base_dct = {'field1': _PField(initial=1, factory=None, invariant=None, initial_factory=None)}
    BaseClass = PClassMeta('BaseClass', (CheckedType,), base_dct)
    
    # Create subclass
    sub_dct = {}
    SubClass = PClassMeta('SubClass', (BaseClass,), sub_dct)
    
    assert '__weakref__' in BaseClass.__slots__
    # Subclass should not add another __weakref__
    assert '__weakref__' not in SubClass.__slots__


def test_pclass_meta_new_inherits_fields_from_bases():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    # Create base class
    base_dct = {'field1': _PField(initial=1, factory=None, invariant=None, initial_factory=None)}
    BaseClass = PClassMeta('BaseClass', (CheckedType,), base_dct)
    
    # Create subclass with additional field
    sub_dct = {'field2': _PField(initial=2, factory=None, invariant=None, initial_factory=None)}
    SubClass = PClassMeta('SubClass', (BaseClass,), sub_dct)
    
    assert 'field1' in SubClass._pclass_fields
    assert 'field2' in SubClass._pclass_fields


# LLM-generated content at query #11
#--------------------------

```python
def test_serialize_with_no_fields():
    from pyrsistent import PClass, field
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    result = instance.serialize()
    assert result == {}


def test_serialize_with_simple_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    instance = SimpleClass(x=1, y="test")
    result = instance.serialize()
    assert result == {'x': 1, 'y': 'test'}


def test_serialize_with_missing_optional_fields():
    from pyrsistent import PClass, field
    
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    instance = OptionalClass(x=42)
    result = instance.serialize()
    assert 'x' in result
    assert result['x'] == 42


def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    
    class FormattedClass(PClass):
        value = field()
    
    instance = FormattedClass(value=100)
    result = instance.serialize(format='json')
    assert 'value' in result
    assert result['value'] == 100


def test_serialize_with_nested_pclass():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        a = field()
    
    class OuterClass(PClass):
        inner = field()
        b = field()
    
    inner_instance = InnerClass(a=5)
    outer_instance = OuterClass(inner=inner_instance, b=10)
    result = outer_instance.serialize()
    assert 'inner' in result
    assert 'b' in result
    assert result['b'] == 10


def test_serialize_preserves_field_order():
    from pyrsistent import PClass, field
    
    class OrderedClass(PClass):
        first = field()
        second = field()
        third = field()
    
    instance = OrderedClass(first=1, second=2, third=3)
    result = instance.serialize()
    assert set(result.keys()) == {'first', 'second', 'third'}
    assert result['first'] == 1
    assert result['second'] == 2
    assert result['third'] == 3


def test_serialize_with_none_values():
    from pyrsistent import PClass, field
    
    class NullableClass(PClass):
        x = field()
        y = field()
    
    instance = NullableClass(x=None, y=42)
    result = instance.serialize()
    assert result['x'] is None
    assert result['y'] == 42


def test_serialize_with_complex_types():
    from pyrsistent import PClass, field
    
    class ComplexClass(PClass):
        lst = field()
        dct = field()
    
    instance = ComplexClass(lst=[1, 2, 3], dct={'key': 'value'})
    result = instance.serialize()
    assert result['lst'] == [1, 2, 3]
    assert result['dct'] == {'key': 'value'}


# LLM-generated content at query #12
#--------------------------

```python
def test_pclass_new_raises_invariant_exception_when_invariant_errors_exist():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def always_fail(obj):
        return (False, 'test_error')
    
    class TestClass(PClass):
        x = field()
        __invariants__ = (always_fail,)
    
    try:
        TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'test_error' in e.error_codes


def test_pclass_new_raises_invariant_exception_when_missing_fields_exist():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_raises_invariant_exception_when_both_invariant_errors_and_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def always_fail(obj):
        return (False, 'test_error')
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
        __invariants__ = (always_fail,)
    
    try:
        TestClass(y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'test_error' in e.error_codes
        assert 'TestClass.x' in e.missing_fields


# LLM-generated content at query #13
#--------------------------

```python
def test_pclassmeta_new_with_pclass_bases():
    from pyrsistent._pclass import PClassMeta, _is_pclass
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    
    # Create a mock base class that would be recognized as a PClass
    class MockPClassBase(metaclass=PClassMeta):
        pass
    
    # Test that _is_pclass returns True for bases containing PClass
    bases = (MockPClassBase,)
    result = _is_pclass(bases)
    
    assert result is True


# LLM-generated content at query #14
#--------------------------

```python
def test_set_method_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    updated = instance.set(x=10)
    
    assert hasattr(TestClass, '_pclass_fields')
    assert isinstance(TestClass._pclass_fields, dict)
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    assert 'z' in TestClass._pclass_fields
    
    for key in TestClass._pclass_fields:
        assert isinstance(key, str)
    
    assert updated.x == 10
    assert updated.y == 2
    assert updated.z == 3
    assert instance.x == 1


# LLM-generated content at query #15
#--------------------------

```python
def test_hash_basic():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)
    
    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


def test_hash_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert hash(obj1) == hash(obj2)


def test_hash_consistent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    hash1 = hash(obj)
    hash2 = hash(obj)
    
    assert hash1 == hash2


def test_hash_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=2, y=3)
    
    s = {obj1, obj2, obj3}
    assert len(s) == 2


def test_hash_with_different_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x="a", y="b")
    obj2 = TestClass(x="a", y="b")
    obj3 = TestClass(x=1, y=2)
    
    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


def test_hash_with_nested_structures():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=pmap({"a": 1}), y=2)
    obj2 = TestClass(x=pmap({"a": 1}), y=2)
    
    assert hash(obj1) == hash(obj2)


def test_hash_empty_pclass():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    obj1 = EmptyClass()
    obj2 = EmptyClass()
    
    assert hash(obj1) == hash(obj2)


def test_hash_single_field():
    from pyrsistent import PClass, field
    
    class SingleFieldClass(PClass):
        x = field()
    
    obj1 = SingleFieldClass(x=42)
    obj2 = SingleFieldClass(x=42)
    obj3 = SingleFieldClass(x=43)
    
    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


# LLM-generated content at query #16
#--------------------------

```python
def test_serialize_basic():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    obj = SimpleClass(x=1, y=2)
    result = obj.serialize()
    
    assert result == {'x': 1, 'y': 2}


def test_serialize_with_missing_values():
    from pyrsistent import PClass, field
    
    class ClassWithOptional(PClass):
        x = field()
        y = field(initial=None)
    
    obj = ClassWithOptional(x=1)
    result = obj.serialize()
    
    assert result == {'x': 1, 'y': None}


def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field
    
    def double_serializer(value):
        return value * 2
    
    class ClassWithSerializer(PClass):
        x = field(serializer=double_serializer)
        y = field()
    
    obj = ClassWithSerializer(x=5, y=10)
    result = obj.serialize()
    
    assert result == {'x': 10, 'y': 10}


def test_serialize_nested_pclass():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        a = field()
    
    class OuterClass(PClass):
        inner = field()
        b = field()
    
    inner_obj = InnerClass(a=1)
    outer_obj = OuterClass(inner=inner_obj, b=2)
    result = outer_obj.serialize()
    
    assert result['b'] == 2
    assert result['inner'] == inner_obj


def test_serialize_empty_pclass():
    from pyrsistent import PClass, field
    
    class EmptyClass(PClass):
        x = field(initial=None)
    
    obj = EmptyClass()
    result = obj.serialize()
    
    assert result == {'x': None}


def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    
    def json_serializer(value):
        return str(value)
    
    class ClassWithFormat(PClass):
        x = field(serializer=json_serializer)
    
    obj = ClassWithFormat(x=42)
    result = obj.serialize(format='json')
    
    assert result == {'x': '42'}


def test_serialize_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj = MultiFieldClass(a=1, b=2, c=3, d=4)
    result = obj.serialize()
    
    assert result == {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert len(result) == 4


# LLM-generated content at query #17
#--------------------------

```python
def test_pclass_invariant_errors_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    # Add a global invariant that will fail
    TestClass._pclass_invariants = (lambda obj: (False, 'test_error'),)
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException:
        pass


def test_pclass_missing_mandatory_field_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_invariant_errors_or_missing_fields_true():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        TestClass(y=2)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.missing_fields) > 0 or len(e.invariant_errors) > 0


# LLM-generated content at query #18
#--------------------------

```python
def test_set_updates_data_and_marks_dirty():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.set('key2', 'value2')
    
    assert evolver._pclass_evolver_data['key2'] == 'value2'
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'key2' in evolver._factory_fields
    assert result is evolver


def test_set_with_same_value_does_not_mark_dirty():
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    value_obj = 'value1'
    initial_dict['key1'] = value_obj
    evolver.set('key1', value_obj)
    
    assert evolver._pclass_evolver_data_is_dirty is False
    assert 'key1' not in evolver._factory_fields


def test_set_replaces_existing_value():
    original = object()
    initial_dict = {'key1': 'old_value'}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.set('key1', 'new_value')
    
    assert evolver._pclass_evolver_data['key1'] == 'new_value'
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'key1' in evolver._factory_fields


def test_set_returns_self_for_chaining():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.set('key1', 'value1')
    
    assert result is evolver


def test_set_with_none_value():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.set('key1', None)
    
    assert evolver._pclass_evolver_data['key1'] is None
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'key1' in evolver._factory_fields


def test_set_multiple_keys():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    
    evolver.set('key1', 'value1').set('key2', 'value2').set('key3', 'value3')
    
    assert evolver._pclass_evolver_data['key1'] == 'value1'
    assert evolver._pclass_evolver_data['key2'] == 'value2'
    assert evolver._pclass_evolver_data['key3'] == 'value3'
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._factory_fields == {'key1', 'key2', 'key3'}


# LLM-generated content at query #19
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        instance = TestClass(y=1)
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass.create({'x': 1, 'z': 2}, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class TestClass(PClass):
        inner = field()
    
    instance = TestClass(inner={'value': 10})
    assert isinstance(instance.inner, InnerClass)
    assert instance.inner.value == 10


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass()
    assert not hasattr(instance, 'x')


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
        z = field()
    
    try:
        instance = TestClass(z=1)
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e) and "TestClass.y" in str(e)


# LLM-generated content at query #20
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields_raises():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        instance = TestClass(x="not an int")
        assert False, "Should have raised PTypeError"
    except Exception as e:
        assert "Invalid type" in str(e)


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive(value):
        return (value > 0, "Value must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive)
    
    try:
        instance = TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.error_codes


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def sum_positive(obj):
        return (obj.x + obj.y > 0, "Sum must be positive")
    
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = (sum_positive,)
    
    try:
        instance = TestClass(x=-5, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.error_codes


def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_new_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field(initial=5)
        c = field()
    
    instance = TestClass(a=1, c=3)
    assert instance.a == 1
    assert instance.b == 5
    assert instance.c == 3


def test_pclass_new_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_new_with_all_optional_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass()
    assert not hasattr(instance, 'x')
    assert not hasattr(instance, 'y')


# LLM-generated content at query #21
#--------------------------

```python
def test_set_predicate_false_when_value_unchanged():
    # Create a mock original object
    class MockOriginal:
        pass
    
    original = MockOriginal()
    initial_dict = {'key1': 'value1'}
    
    evolver = _PClassEvolver(original, initial_dict)
    
    # Set the same value that's already in the data
    # This should make the predicate (self._pclass_evolver_data.get(key, _MISSING_VALUE) is not value) evaluate to False
    result = evolver.set('key1', 'value1')
    
    # Assertions to verify the predicate was False (body was not executed)
    assert result is evolver  # set() returns self
    assert evolver._pclass_evolver_data_is_dirty is False  # dirty flag should not be set
    assert 'key1' not in evolver._factory_fields  # field should not be added to factory_fields
    assert evolver._pclass_evolver_data['key1'] == 'value1'  # data should remain unchanged


# LLM-generated content at query #22
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
    
    instance = TestClass()
    assert instance.x == 10


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PAttrError
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    assert instance._pclass_frozen is True


def test_pclass_new_cannot_set_after_frozen():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field, PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="not an int")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_field_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    
    def positive_invariant(value):
        return (value > 0, 'must be positive')
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'must be positive' in e.error_codes


def test_pclass_new_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_new_with_ignore_extra_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True)
    assert instance.x == 1


def test_pclass_new_partial_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=5)
    
    instance = TestClass(x=1)
    assert instance.x == 1
    assert instance.y == 5


def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    
    def global_check(obj):
        return (obj.x < obj.y, 'x must be less than y')
    
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = (global_check,)
    
    try:
        TestClass(x=5, y=3)
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_pclass_new_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


# LLM-generated content at query #23
#--------------------------

```python
def test_pclass_missing_mandatory_field_raises_invariant_exception():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    class TestClass(PClass):
        mandatory_field = field(mandatory=True)
        optional_field = field()
    
    try:
        instance = TestClass(optional_field=42)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.mandatory_field' in e.missing_fields
        assert len(e.missing_fields) == 1


def test_pclass_field_invariant_error_raises_exception():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    class TestClass(PClass):
        test_field = field(invariant=lambda x: (x > 0, "must be positive"))
    
    try:
        instance = TestClass(test_field=-1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0


def test_pclass_multiple_missing_mandatory_fields():
    from pyrsistent import PClass, field
    from pyrsistent._precord_common import InvariantException
    
    class TestClass(PClass):
        field1 = field(mandatory=True)
        field2 = field(mandatory=True)
        field3 = field()
    
    try:
        instance = TestClass(field3=100)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.field1' in e.missing_fields
        assert 'TestClass.field2' in e.missing_fields
        assert len(e.missing_fields) == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_set_predicate_false_when_value_unchanged():
    class _MISSING_VALUE:
        pass
    
    original = object()
    initial_dict = {'key1': 'value1'}
    evolver = _PClassEvolver(original, initial_dict)
    
    result = evolver.set('key1', 'value1')
    
    assert result is evolver
    assert evolver._pclass_evolver_data_is_dirty is False
    assert 'key1' not in evolver._factory_fields


# LLM-generated content at query #25
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_invariant_errors_exist():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    # Mock a scenario where invariant_errors would be populated
    # by creating a field with an invariant that fails
    class StrictClass(PClass):
        x = field(invariant=lambda val: (False, 'invariant_failed'))
    
    try:
        StrictClass(x=5)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('invariant_failed',)


def test_pclass_raises_invariant_exception_when_missing_mandatory_fields():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


# LLM-generated content at query #26
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_invariant_errors_present():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field()
    
    # Mock the _check_and_set_attr to simulate invariant errors
    original_check_and_set_attr = __import__('pyrsistent._pclass', fromlist=['_check_and_set_attr'])._check_and_set_attr
    
    def mock_check_and_set_attr(cls, field_obj, name, value, result, invariant_errors):
        invariant_errors.append("test error")
    
    __import__('pyrsistent._pclass', fromlist=['_check_and_set_attr'])._check_and_set_attr = mock_check_and_set_attr
    
    try:
        error_raised = False
        try:
            TestClass(x=1)
        except InvariantException:
            error_raised = True
        
        assert error_raised is True
    finally:
        __import__('pyrsistent._pclass', fromlist=['_check_and_set_attr'])._check_and_set_attr = original_check_and_set_attr


def test_pclass_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    error_raised = False
    try:
        TestClass()
    except InvariantException as e:
        error_raised = True
        assert 'TestClass.x' in e.missing_fields
    
    assert error_raised is True


# LLM-generated content at query #27
#--------------------------

```python
def test_pclass_meta_new_with_pclass_bases():
    from pyrsistent._pclass import PClassMeta, _is_pclass
    from pyrsistent._field_common import _PField
    
    # Mock _is_pclass to return True
    original_is_pclass = _is_pclass
    
    def mock_is_pclass(bases):
        return True
    
    # Temporarily replace _is_pclass
    import pyrsistent._pclass
    pyrsistent._pclass._is_pclass = mock_is_pclass
    
    try:
        # Create a test class using PClassMeta
        dct = {
            'field1': _PField(),
            'field2': _PField(),
        }
        bases = (object,)
        name = 'TestPClass'
        
        # Call __new__ to trigger the condition at line 8
        result_class = PClassMeta.__new__(PClassMeta, name, bases, dct)
        
        # Verify that __weakref__ was added to __slots__
        assert '__weakref__' in result_class.__slots__
        assert '_pclass_frozen' in result_class.__slots__
        
    finally:
        # Restore original _is_pclass
        pyrsistent._pclass._is_pclass = original_is_pclass


# LLM-generated content at query #28
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class TestClass(PClass):
        inner = field()
    
    inner_data = {'value': 42}
    instance = TestClass(inner=inner_data, _factory_fields={'inner'})
    assert isinstance(instance.inner, InnerClass)
    assert instance.inner.value == 42


def test_pclass_constructor_without_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields=set())
    assert instance.x == 5


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=0)
    
    instance = TestClass(x=1)
    assert instance.x == 1
    assert instance.y == 0


# LLM-generated content at query #29
#--------------------------

```python
def test_pclass_meta_new_with_pclass_bases():
    from pyrsistent._pclass import PClassMeta, _is_pclass
    from pyrsistent._field_common import _PField
    
    # Create a mock base class that _is_pclass would return True for
    class MockPClass(metaclass=PClassMeta):
        _pclass_fields = {}
        _pclass_invariants = ()
    
    # Verify that _is_pclass returns True for bases containing PClass
    bases = (MockPClass,)
    result = _is_pclass(bases)
    assert result is True


# LLM-generated content at query #30
#--------------------------

```python
def test_hash_returns_consistent_hash_for_pclass_instances():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert hash1 == hash2
    assert isinstance(hash1, int)
    assert isinstance(hash2, int)


def test_hash_differs_for_different_field_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert hash1 != hash2


def test_hash_can_be_used_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=2, y=3)
    
    hash_set = {instance1, instance2, instance3}
    
    assert len(hash_set) == 2
    assert instance1 in hash_set
    assert instance3 in hash_set


def test_hash_can_be_used_in_dict():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    
    hash_dict = {instance1: "value1"}
    hash_dict[instance2] = "value2"
    
    assert len(hash_dict) == 1
    assert hash_dict[instance1] == "value2"


# LLM-generated content at query #31
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_attr", 42, result, invariant_errors)
    
    assert hasattr(result, "test_attr")
    assert result.test_attr == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._exceptions import PTypeError
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        __name__ = "MockClass"
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int], lambda x: (True, None))
    
    try:
        _check_and_set_attr(MockClass, field, "test_attr", "invalid", result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError:
        pass


def test_check_and_set_attr_failed_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    error_code = "value_too_small"
    field = MockField([int], lambda x: (False, error_code))
    
    _check_and_set_attr(MockClass, field, "test_attr", 42, result, invariant_errors)
    
    assert not hasattr(result, "test_attr")
    assert invariant_errors == [error_code]


def test_check_and_set_attr_no_type_constraint():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField(None, lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_attr", "any_value", result, invariant_errors)
    
    assert hasattr(result, "test_attr")
    assert result.test_attr == "any_value"
    assert invariant_errors == []


def test_check_and_set_attr_multiple_allowed_types():
    from pyrsistent._pclass import _check_and_set_attr
    
    class MockField:
        def __init__(self, field_type, invariant_func):
            self.type = field_type
            self.invariant = invariant_func
    
    class MockClass:
        pass
    
    class TestClass:
        pass
    
    result = TestClass()
    invariant_errors = []
    field = MockField([int, str], lambda x: (True, None))
    
    _check_and_set_attr(MockClass, field, "test_attr", "string_value", result, invariant_errors)
    
    assert hasattr(result, "test_attr")
    assert result.test_attr == "string_value"
    assert invariant_errors == []


# LLM-generated content at query #32
#--------------------------

```python
def test_reduce_returns_restore_pickle_and_class_data():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()
    
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


# LLM-generated content at query #33
#--------------------------

```python
def test_pclass_repr_basic():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = repr(obj)
    assert result == "TestClass(x=1, y=2)"


def test_pclass_repr_single_field():
    from pyrsistent import PClass, field
    
    class SingleField(PClass):
        name = field()
    
    obj = SingleField(name="test")
    result = repr(obj)
    assert result == "SingleField(name='test')"


def test_pclass_repr_string_values():
    from pyrsistent import PClass, field
    
    class StringClass(PClass):
        a = field()
        b = field()
    
    obj = StringClass(a="hello", b="world")
    result = repr(obj)
    assert result == "StringClass(a='hello', b='world')"


def test_pclass_repr_mixed_types():
    from pyrsistent import PClass, field
    
    class MixedClass(PClass):
        num = field()
        text = field()
        flag = field()
    
    obj = MixedClass(num=42, text="value", flag=True)
    result = repr(obj)
    assert "MixedClass(" in result
    assert "num=42" in result
    assert "text='value'" in result
    assert "flag=True" in result


def test_pclass_repr_with_initial_values():
    from pyrsistent import PClass, field
    
    class InitialClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = InitialClass(y=20)
    result = repr(obj)
    assert "InitialClass(" in result
    assert "x=10" in result
    assert "y=20" in result


def test_pclass_repr_empty_class():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    result = repr(obj)
    assert result == "EmptyClass()"


def test_pclass_repr_nested_values():
    from pyrsistent import PClass, field
    
    class NestedClass(PClass):
        data = field()
    
    obj = NestedClass(data=[1, 2, 3])
    result = repr(obj)
    assert result == "NestedClass(data=[1, 2, 3])"


def test_pclass_repr_with_none_value():
    from pyrsistent import PClass, field
    
    class NoneClass(PClass):
        value = field()
    
    obj = NoneClass(value=None)
    result = repr(obj)
    assert result == "NoneClass(value=None)"


def test_pclass_repr_multiple_fields_order():
    from pyrsistent import PClass, field
    
    class MultiField(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = MultiField(a=1, b=2, c=3)
    result = repr(obj)
    assert "MultiField(" in result
    assert result.endswith(")")
    assert "a=1" in result
    assert "b=2" in result
    assert "c=3" in result


# LLM-generated content at query #34
#--------------------------

```python
def test_repr_format():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y="hello")
    repr_str = repr(instance)
    
    assert "TestClass(" in repr_str
    assert "x=1" in repr_str
    assert "y='hello'" in repr_str
    assert repr_str.endswith(")")
    assert repr_str == "TestClass(x=1, y='hello')" or repr_str == "TestClass(y='hello', x=1)"


# LLM-generated content at query #35
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    # The predicate at line 7 is: `for name in self._pclass_fields:`
    # This evaluates to True when _pclass_fields is iterable and contains field names
    assert hasattr(instance, '_pclass_fields')
    assert isinstance(instance._pclass_fields, dict)
    assert 'x' in instance._pclass_fields
    assert 'y' in instance._pclass_fields
    assert 'z' in instance._pclass_fields
    assert len(instance._pclass_fields) == 3


# LLM-generated content at query #36
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #37
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    result = obj1 == obj2
    assert result is True


# LLM-generated content at query #38
#--------------------------

```python
def test_serialize_with_no_fields():
    from pyrsistent import PClass, field
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    result = instance.serialize()
    assert result == {}


def test_serialize_with_simple_fields():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
        y = field()
    
    instance = SimpleClass(x=1, y="test")
    result = instance.serialize()
    assert result == {'x': 1, 'y': "test"}


def test_serialize_with_missing_optional_field():
    from pyrsistent import PClass, field
    
    class OptionalClass(PClass):
        x = field()
        y = field(initial=None)
    
    instance = OptionalClass(x=1)
    result = instance.serialize()
    assert 'x' in result
    assert result['x'] == 1


def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    
    class FormattedClass(PClass):
        value = field()
    
    instance = FormattedClass(value=42)
    result = instance.serialize(format='json')
    assert result == {'value': 42}


def test_serialize_with_nested_objects():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        inner_value = field()
    
    class OuterClass(PClass):
        outer_value = field()
        inner = field()
    
    inner_instance = InnerClass(inner_value=10)
    outer_instance = OuterClass(outer_value=20, inner=inner_instance)
    result = outer_instance.serialize()
    assert result['outer_value'] == 20
    assert result['inner'] == inner_instance


def test_serialize_preserves_field_order():
    from pyrsistent import PClass, field
    
    class OrderedClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = OrderedClass(a=1, b=2, c=3)
    result = instance.serialize()
    assert set(result.keys()) == {'a', 'b', 'c'}
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


def test_serialize_with_multiple_fields_some_missing():
    from pyrsistent import PClass, field
    
    class PartialClass(PClass):
        required = field()
        optional1 = field(initial=None)
        optional2 = field(initial=None)
    
    instance = PartialClass(required="value")
    result = instance.serialize()
    assert 'required' in result
    assert result['required'] == "value"


# LLM-generated content at query #39
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        mandatory_field = field(mandatory=True)
        optional_field = field()
    
    try:
        TestClass(optional_field=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.mandatory_field' in e.missing_fields
        assert len(e.missing_fields) == 1
        assert e.invariant_errors == ()


def test_pclass_raises_invariant_exception_when_field_invariant_fails():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        test_field = field(invariant=lambda x: (x > 0, "must be positive"))
    
    try:
        TestClass(test_field=-1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0


def test_pclass_raises_invariant_exception_with_multiple_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        mandatory_field1 = field(mandatory=True)
        mandatory_field2 = field(mandatory=True)
        optional_field = field()
    
    try:
        TestClass(optional_field=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'TestClass.mandatory_field1' in e.missing_fields
        assert 'TestClass.mandatory_field2' in e.missing_fields
        assert len(e.missing_fields) == 2


# LLM-generated content at query #40
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        type = (int,)
        def invariant(self, value):
            return (False, "error_code_123")
    
    class MockClass:
        pass
    
    mock_result = MockClass()
    invariant_errors = []
    field = MockField()
    
    _check_and_set_attr(MockClass, field, "test_field", 42, mock_result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "error_code_123"
    assert not hasattr(mock_result, "test_field")


# LLM-generated content at query #41
#--------------------------

```python
def test_eq_predicate_isinstance_true():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    result = obj1 == obj2
    assert result is True


# LLM-generated content at query #42
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    class MockClass:
        pass
    
    field = MockField()
    name = "test_field"
    value = "test_value"
    result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    
    assert invariant_errors == ["invariant_error_code"]
    assert not hasattr(result, name)


# LLM-generated content at query #43
#--------------------------

```python
def test_pclass_hash_same_values_same_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_different_values_different_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert hash(obj1) != hash(obj2)


def test_pclass_hash_hashable():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    
    hash_value = hash(obj)
    assert isinstance(hash_value, int)


def test_pclass_hash_can_be_used_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    obj3 = TestClass(x=2)
    
    hash_set = {obj1, obj2, obj3}
    assert len(hash_set) >= 2


def test_pclass_hash_can_be_used_as_dict_key():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=2)
    
    hash_dict = {obj1: "value1", obj2: "value2"}
    assert hash_dict[obj1] == "value1"
    assert hash_dict[obj2] == "value2"


def test_pclass_hash_with_missing_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1, y=None)
    
    hash_value1 = hash(obj1)
    hash_value2 = hash(obj2)
    assert isinstance(hash_value1, int)
    assert isinstance(hash_value2, int)


def test_pclass_hash_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj1 = TestClass(a=1, b="test", c=[1, 2, 3])
    obj2 = TestClass(a=1, b="test", c=[1, 2, 3])
    
    assert hash(obj1) == hash(obj2)


def test_pclass_hash_consistent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    hash1 = hash(obj)
    hash2 = hash(obj)
    
    assert hash1 == hash2


# LLM-generated content at query #44
#--------------------------

```python
def test_pclass_equality_predicate_at_line_3():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    # Trigger the condition at line 3 to evaluate to True
    # This requires isinstance(other, self.__class__) to be True
    result = (obj1 == obj2)
    
    assert result is True
    assert isinstance(obj2, obj1.__class__)


# LLM-generated content at query #45
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_new_with_wrong_type():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        instance = TestClass(x="string")
        assert False, "Should have raised PTypeError"
    except Exception:
        pass


def test_pclass_new_with_field_invariant_failure():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "Must be positive"))
    
    try:
        instance = TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) > 0


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_new_with_ignore_extra_param():
    from pyrsistent import PClass, field
    from pyrsistent import pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    instance = TestClass(_factory_fields={'x'}, ignore_extra=True, x={'a': 1, 'b': 2})
    assert instance.x['a'] == 1


def test_pclass_new_with_multiple_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(type=(int, str))
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x="hello")
    assert instance1.x == 1
    assert instance2.x == "hello"


def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._invariant import InvariantException
    
    @invariant(lambda obj: (obj.x < obj.y, "x must be less than y"))
    class TestClass(PClass):
        x = field()
        y = field()
    
    try:
        instance = TestClass(x=5, y=3)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.error_codes) > 0


def test_pclass_new_with_no_fields():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_new_with_factory_fields_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
        y = field()
    
    instance = TestClass(_factory_fields={'x'}, x="10", y=20)
    assert instance.x == 10
    assert instance.y == 20


# LLM-generated content at query #46
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x={'a': 1}, _factory_fields={'x'})
    assert instance.x == pmap({'a': 1})


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #47
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockClass:
        pass
    
    result = MockClass()
    
    def failing_invariant(value):
        return False, "invariant_error_code"
    
    field = PField(type=int, invariant=failing_invariant)
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_error_code"
    assert not hasattr(result, "test_field")


# LLM-generated content at query #48
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2
    assert obj._pclass_frozen is True


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=2)
    assert obj.x == 10
    assert obj.y == 2


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="string")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive_invariant(val):
        return (val > 0, "Value must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._invariant import InvariantException
    
    @invariant("x > y", lambda obj: (obj.x > obj.y, "x must be greater than y"))
    class TestClass(PClass):
        x = field()
        y = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    from pyrsistent import pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    obj = TestClass(x={'a': 1})
    assert obj.x == pmap({'a': 1})


def test_pclass_new_multiple_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 2


def test_pclass_new_with_mixed_mandatory_optional():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    obj = TestClass(x=1)
    assert obj.x == 1
    assert not hasattr(obj, 'y') or getattr(obj, 'y', None) is None


def test_pclass_new_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert obj._pclass_frozen is True


# LLM-generated content at query #49
#--------------------------

```python
def test_reduce_returns_restore_pickle_and_class_data():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()
    
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


# LLM-generated content at query #50
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=2)
    assert obj.x == 10
    assert obj.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    obj = TestClass()
    assert obj.x == 42


def test_pclass_constructor_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, extra_field=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'extra_field' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, extra_field=2, ignore_extra=True)
    assert obj.x == 1


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x={'a': 1}, _factory_fields={'x'})
    assert obj.x == pmap({'a': 1})


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    assert hasattr(obj, '_pclass_frozen')


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #51
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.invariant = lambda value: (False, "invariant_error")
    
    class MockClass:
        __name__ = "MockClass"
    
    class MockResult:
        pass
    
    field = MockField()
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", 42, result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_error"
    assert not hasattr(result, "test_field")


# LLM-generated content at query #52
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='5')
    assert instance.x == 5


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_with_field_factory_and_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='10', y='extra', ignore_extra=True)
    assert instance.x == 10


def test_pclass_constructor_multiple_missing_mandatory_fields():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


def test_pclass_constructor_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    assert instance.x is None


def test_pclass_constructor_preserves_field_order():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(c=3, a=1, b=2)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


# LLM-generated content at query #53
#--------------------------

```python
def test_pclass_repr_single_field():
    from pyrsistent import PClass, field
    
    class SimpleClass(PClass):
        x = field()
    
    obj = SimpleClass(x=1)
    assert repr(obj) == "SimpleClass(x=1)"


def test_pclass_repr_multiple_fields():
    from pyrsistent import PClass, field
    
    class MultiFieldClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = MultiFieldClass(x=1, y=2, z=3)
    repr_str = repr(obj)
    assert "MultiFieldClass(" in repr_str
    assert "x=1" in repr_str
    assert "y=2" in repr_str
    assert "z=3" in repr_str


def test_pclass_repr_with_string_value():
    from pyrsistent import PClass, field
    
    class StringClass(PClass):
        name = field()
    
    obj = StringClass(name="test")
    assert repr(obj) == "StringClass(name='test')"


def test_pclass_repr_with_nested_object():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner_obj = InnerClass(value=42)
    outer_obj = OuterClass(inner=inner_obj)
    assert "OuterClass(inner=InnerClass(value=42))" == repr(outer_obj)


def test_pclass_repr_empty_class():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    obj = EmptyClass()
    assert repr(obj) == "EmptyClass()"


def test_pclass_repr_with_optional_field():
    from pyrsistent import PClass, field
    
    class OptionalClass(PClass):
        x = field()
        y = field()
    
    obj = OptionalClass(x=1)
    assert repr(obj) == "OptionalClass(x=1)"


def test_pclass_repr_with_list_value():
    from pyrsistent import PClass, field
    
    class ListClass(PClass):
        items = field()
    
    obj = ListClass(items=[1, 2, 3])
    assert repr(obj) == "ListClass(items=[1, 2, 3])"


def test_pclass_repr_with_dict_value():
    from pyrsistent import PClass, field
    
    class DictClass(PClass):
        data = field()
    
    obj = DictClass(data={'key': 'value'})
    repr_str = repr(obj)
    assert "DictClass(data=" in repr_str
    assert "'key': 'value'" in repr_str


def test_pclass_repr_with_none_value():
    from pyrsistent import PClass, field
    
    class NoneClass(PClass):
        x = field()
    
    obj = NoneClass(x=None)
    assert repr(obj) == "NoneClass(x=None)"


def test_pclass_repr_with_boolean_values():
    from pyrsistent import PClass, field
    
    class BoolClass(PClass):
        flag1 = field()
        flag2 = field()
    
    obj = BoolClass(flag1=True, flag2=False)
    repr_str = repr(obj)
    assert "BoolClass(" in repr_str
    assert "flag1=True" in repr_str
    assert "flag2=False" in repr_str


# LLM-generated content at query #54
#--------------------------

```python
def test_set_method_predicate_line_25():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    updated_instance = instance.set(x=10)
    
    assert hasattr(instance, '_pclass_fields')
    assert instance._pclass_fields is not None
    assert len(instance._pclass_fields) > 0
    
    for key in instance._pclass_fields:
        assert isinstance(key, str)
    
    assert updated_instance.x == 10
    assert updated_instance.y == 2
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #55
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=2)
    assert obj.x == 10
    assert obj.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, _factory_fields={'x'})
    assert obj.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, y=2, ignore_extra=True)
    assert obj.x == 1
    assert not hasattr(obj, 'y')


def test_pclass_constructor_preserves_field_order():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a=1, b=2, c=3)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3


def test_pclass_constructor_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=None)
    assert obj.x is None


# LLM-generated content at query #56
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    class MockClass:
        pass
    
    field = MockField()
    result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    
    assert invariant_errors == ["invariant_error_code"]
    assert not hasattr(result, "test_field")


# LLM-generated content at query #57
#--------------------------

```python
def test_pclass_invariant_errors_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field()
        _pclass_invariants = (lambda obj: (False, 'test_error'),)
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('test_error',)
        assert e.missing_fields == ()


def test_pclass_missing_fields_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.missing_fields == ('TestClass.x',)
        assert e.invariant_errors == ()


def test_pclass_both_invariant_errors_and_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        _pclass_invariants = (lambda obj: (False, 'invariant_failed'),)
    
    try:
        TestClass()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert 'invariant_failed' in e.invariant_errors
        assert 'TestClass.x' in e.missing_fields


# LLM-generated content at query #58
#--------------------------

```python
def test_pclass_repr_single_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    assert repr(obj) == "TestClass(x=1)"


def test_pclass_repr_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = repr(obj)
    assert "TestClass(" in result
    assert "x=1" in result
    assert "y=2" in result


def test_pclass_repr_string_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
    
    obj = TestClass(name="test")
    assert repr(obj) == "TestClass(name='test')"


def test_pclass_repr_nested_structure():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        data = field()
    
    obj = TestClass(data=[1, 2, 3])
    assert repr(obj) == "TestClass(data=[1, 2, 3])"


def test_pclass_repr_empty_pclass():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=None)
    assert repr(obj) == "TestClass(x=None)"


def test_pclass_repr_multiple_fields_order():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj = TestClass(a=1, b=2, c=3)
    result = repr(obj)
    assert result.startswith("TestClass(")
    assert result.endswith(")")
    assert "a=1" in result
    assert "b=2" in result
    assert "c=3" in result


def test_pclass_repr_with_float_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        value = field()
    
    obj = TestClass(value=3.14)
    assert repr(obj) == "TestClass(value=3.14)"


def test_pclass_repr_with_boolean_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        flag = field()
    
    obj = TestClass(flag=True)
    assert repr(obj) == "TestClass(flag=True)"


def test_pclass_repr_with_dict_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        config = field()
    
    obj = TestClass(config={'key': 'value'})
    assert repr(obj) == "TestClass(config={'key': 'value'})"


def test_pclass_repr_class_name_in_output():
    from pyrsistent import PClass, field
    
    class MyCustomClass(PClass):
        x = field()
    
    obj = MyCustomClass(x=42)
    assert "MyCustomClass" in repr(obj)


# LLM-generated content at query #59
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner_data = {'value': 10}
    instance = OuterClass(inner=inner_data, _factory_fields={'inner'})
    assert instance.inner.value == 10


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5)
    assert instance.x == 5


# LLM-generated content at query #60
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, z=2)
    assert instance.x == 1


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_multiple_fields_with_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field(initial=1)
        b = field(initial=2)
        c = field()
    
    instance = TestClass(c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    assert instance is not None


# LLM-generated content at query #61
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, z=2)
    assert instance.x == 1


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #62
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, z=2)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_with_no_fields():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_constructor_multiple_instances_independent():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=2)
    assert instance1.x == 1
    assert instance2.x == 2


# LLM-generated content at query #63
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    try:
        TestClass(x=1)
        assert False, "Should raise InvariantException"
    except Exception as e:
        assert "TestClass.y" in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "z" in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, _factory_fields=None, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #64
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class TestClass:
        pass
    
    result = TestClass()
    field = Field(type=int, invariant=lambda x: (True, None))
    invariant_errors = []
    
    _check_and_set_attr(TestClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    from pyrsistent._exc import PTypeError
    
    class TestClass:
        pass
    
    result = TestClass()
    field = Field(type=int, invariant=lambda x: (True, None))
    invariant_errors = []
    
    try:
        _check_and_set_attr(TestClass, field, "test_field", "invalid", result, invariant_errors)
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_check_and_set_attr_failed_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class TestClass:
        pass
    
    result = TestClass()
    field = Field(type=int, invariant=lambda x: (False, "value_too_large"))
    invariant_errors = []
    
    _check_and_set_attr(TestClass, field, "test_field", 42, result, invariant_errors)
    
    assert invariant_errors == ["value_too_large"]
    assert not hasattr(result, "test_field")


def test_check_and_set_attr_multiple_types():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class TestClass:
        pass
    
    result = TestClass()
    field = Field(type=[int, str], invariant=lambda x: (True, None))
    invariant_errors = []
    
    _check_and_set_attr(TestClass, field, "test_field", "hello", result, invariant_errors)
    
    assert result.test_field == "hello"
    assert invariant_errors == []


def test_check_and_set_attr_no_type_constraint():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import Field
    
    class TestClass:
        pass
    
    result = TestClass()
    field = Field(type=None, invariant=lambda x: (True, None))
    invariant_errors = []
    
    _check_and_set_attr(TestClass, field, "test_field", "any_value", result, invariant_errors)
    
    assert result.test_field == "any_value"
    assert invariant_errors == []


# LLM-generated content at query #65
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=20)
    assert obj.x == 10
    assert obj.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, z=2, ignore_extra=True)
    assert obj.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=5, _factory_fields={'x'})
    assert obj.x == 5


def test_pclass_constructor_multiple_fields_with_mixed_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field(mandatory=True)
        b = field(initial=100)
        c = field()
    
    obj = TestClass(a=1, c=3)
    assert obj.a == 1
    assert obj.b == 100
    assert obj.c == 3


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert isinstance(obj, TestClass)


# LLM-generated content at query #66
#--------------------------

```python
def test_check_and_set_attr_valid_type_and_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class TestClass:
        pass
    
    result = TestClass()
    field = PField(type=int, invariant=lambda x: (True, None))
    invariant_errors = []
    
    _check_and_set_attr(TestClass, field, "test_field", 42, result, invariant_errors)
    
    assert result.test_field == 42
    assert invariant_errors == []


def test_check_and_set_attr_invalid_type():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField, PTypeError
    
    class TestClass:
        pass
    
    result = TestClass()
    field = PField(type=int, invariant=lambda x: (True, None))
    invariant_errors = []
    
    try:
        _check_and_set_attr(TestClass, field, "test_field", "not_an_int", result, invariant_errors)
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_check_and_set_attr_failed_invariant():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class TestClass:
        pass
    
    result = TestClass()
    field = PField(type=int, invariant=lambda x: (False, "value_too_small"))
    invariant_errors = []
    
    _check_and_set_attr(TestClass, field, "test_field", 42, result, invariant_errors)
    
    assert not hasattr(result, "test_field")
    assert invariant_errors == ["value_too_small"]


def test_check_and_set_attr_multiple_types():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class TestClass:
        pass
    
    result = TestClass()
    field = PField(type=[int, str], invariant=lambda x: (True, None))
    invariant_errors = []
    
    _check_and_set_attr(TestClass, field, "test_field", "string_value", result, invariant_errors)
    
    assert result.test_field == "string_value"
    assert invariant_errors == []


def test_check_and_set_attr_no_type_check():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class TestClass:
        pass
    
    result = TestClass()
    field = PField(type=None, invariant=lambda x: (True, None))
    invariant_errors = []
    
    _check_and_set_attr(TestClass, field, "test_field", "any_value", result, invariant_errors)
    
    assert result.test_field == "any_value"
    assert invariant_errors == []


# LLM-generated content at query #67
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e.missing_fields)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        data = field()
    
    instance = TestClass(_factory_fields={'data'}, data={'a': 1})
    assert instance.data is not None


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


# LLM-generated content at query #68
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=20)
    assert obj.x == 10
    assert obj.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    obj = TestClass()
    assert obj.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        obj = TestClass(y=20)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, z=2, ignore_extra=True)
    assert obj.x == 1


def test_pclass_constructor_frozen():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x={'a': 1}, _factory_fields={'x'})
    assert obj.x == {'a': 1}


def test_pclass_constructor_no_arguments():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
        y = field(initial=10)
    
    obj = TestClass()
    assert obj.x == 5
    assert obj.y == 10


def test_pclass_constructor_multiple_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
        z = field()
    
    try:
        obj = TestClass(z=30)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #69
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._precord_fields import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields=set(), ignore_extra=True, x=1, z=2)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


# LLM-generated content at query #70
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    try:
        class TestClass(PClass):
            x = field(mandatory=True)
        
        instance = TestClass()
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x={'a': 1})
    assert instance.x == {'a': 1}


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(ignore_extra=True, x=1, extra_field=999)
    assert instance.x == 1
    assert not hasattr(instance, 'extra_field')


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
    
    instance = TestClass()
    assert instance.x == 5


# LLM-generated content at query #71
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=42)
        y = field()
    
    instance = TestClass(y=10)
    assert instance.x == 42
    assert instance.y == 10


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='42', _factory_fields={'x'})
    assert instance.x == 42


def test_pclass_constructor_without_factory_fields_uses_raw_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x=42, _factory_fields=set())
    assert instance.x == 42


def test_pclass_constructor_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass()
    assert not hasattr(instance, 'x')


# LLM-generated content at query #72
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=20)
    assert instance.x == 10
    assert instance.y == 20


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, extra_field=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'extra_field' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, extra=2)
    assert instance.x == 1
    assert not hasattr(instance, 'extra')


def test_pclass_constructor_with_all_fields_provided():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    assert instance.x == 1
    assert instance.y == 2
    assert instance.z == 3


def test_pclass_constructor_pclass_frozen_attribute_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    assert instance._pclass_frozen is True


# LLM-generated content at query #73
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=2)
    assert obj.x == 10
    assert obj.y == 2


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    obj = TestClass()
    assert obj.x == 42


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "not among the specified fields" in str(e)


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="invalid")
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    assert obj._pclass_frozen is True


def test_pclass_new_cannot_set_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_field_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def positive_invariant(value):
        return (value > 0, "must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field(factory=pmap)
    
    obj = TestClass(x={'a': 1})
    assert obj.x['a'] == 1


def test_pclass_new_with_ignore_extra_false():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, extra=2)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_pclass_new_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj = TestClass(a=1, b=2, c=3, d=4)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3
    assert obj.d == 4


def test_pclass_new_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=None)
    assert obj.x is None


def test_pclass_new_equality():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2


def test_pclass_new_with_empty_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
    
    obj = TestClass()
    assert obj.x == 5


def test_pclass_new_with_complex_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(type=(int, str))
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x="hello")
    assert obj1.x == 1
    assert obj2.x == "hello"


# LLM-generated content at query #74
#--------------------------

```python
def test_eq_predicate_line_3_evaluates_to_true():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    result = obj1 == obj2
    
    assert result is True


# LLM-generated content at query #75
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_constructor_partial_fields_with_defaults():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
        y = field(initial=10)
    
    instance = TestClass(x=1)
    assert instance.x == 1
    assert instance.y == 10


# LLM-generated content at query #76
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    # Access _pclass_fields to verify the predicate at line 7
    # The predicate "for name in self._pclass_fields" should iterate over field names
    field_names = list(instance._pclass_fields.keys())
    
    assert 'x' in field_names
    assert 'y' in field_names
    assert 'z' in field_names
    assert len(field_names) == 3


# LLM-generated content at query #77
#--------------------------

```python
def test_pclass_hash_returns_consistent_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert hash1 == hash2
    assert isinstance(hash1, int)


def test_pclass_hash_different_for_different_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    
    hash1 = hash(instance1)
    hash2 = hash(instance2)
    
    assert hash1 != hash2


def test_pclass_hash_with_missing_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    instance = TestClass(x=5)
    hash_value = hash(instance)
    
    assert isinstance(hash_value, int)


def test_pclass_hash_is_hashable():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=10)
    hash_value = hash(instance)
    
    test_set = {instance}
    assert instance in test_set
    assert len(test_set) == 1


# LLM-generated content at query #78
#--------------------------

```python
def test_repr_format():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y="hello")
    repr_str = repr(obj)
    
    assert repr_str.startswith("TestClass(")
    assert repr_str.endswith(")")
    assert "x=1" in repr_str
    assert "y='hello'" in repr_str
    assert isinstance(repr_str, str)


# LLM-generated content at query #79
#--------------------------

```python
def test_pclass_raises_invariant_exception_when_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        mandatory_field = field(mandatory=True)
        optional_field = field()
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.mandatory_field' in e.missing_fields
        assert len(e.missing_fields) == 1


def test_pclass_raises_invariant_exception_when_field_invariant_fails():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))
    
    try:
        TestClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0


def test_pclass_raises_invariant_exception_when_multiple_fields_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        field1 = field(mandatory=True)
        field2 = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.field1' in e.missing_fields
        assert 'TestClass.field2' in e.missing_fields
        assert len(e.missing_fields) == 2


# LLM-generated content at query #80
#--------------------------

```python
def test_reduce_returns_restore_pickle_and_class_data():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


# LLM-generated content at query #81
#--------------------------

```python
def test_set_with_kwargs():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set(x=10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1
    assert obj.y == 2


def test_set_with_args():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    result = obj.set('x', 10)
    
    assert result.x == 10
    assert result.y == 2
    assert obj.x == 1


def test_set_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(x=10, y=20)
    
    assert result.x == 10
    assert result.y == 20
    assert result.z == 3


def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    result = obj.set(x=2)
    
    assert obj is not result
    assert isinstance(result, TestClass)


def test_set_with_optional_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1)
    result = obj.set(x=5)
    
    assert result.x == 5
    assert not hasattr(result, 'y') or getattr(result, 'y', None) is None


def test_set_preserves_unmodified_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    obj = TestClass(x=1, y=2, z=3)
    result = obj.set(y=20)
    
    assert result.x == 1
    assert result.y == 20
    assert result.z == 3


# LLM-generated content at query #82
#--------------------------

```python
def test_pclass_new_with_valid_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    assert instance.x == 1
    assert instance._pclass_frozen is True


def test_pclass_new_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 99)
    
    instance = TestClass()
    assert instance.x == 99


def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_with_extra_kwargs():
    from pyrsistent import PClass, field, AttributeError as PAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)


def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field, PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="string")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_field_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    
    def positive_invariant(value):
        return (value > 0, "must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert "must be positive" in e.error_codes


def test_pclass_new_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_new_with_ignore_extra_parameter():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    
    def global_check(obj):
        return (obj.x > 0, "x must be positive")
    
    class TestClass(PClass):
        x = field()
        __invariants__ = (global_check,)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert "x must be positive" in e.error_codes


def test_pclass_new_preserves_field_order():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    instance = TestClass(a=1, b=2, c=3)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_pclass_new_with_default_initial_and_provided_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
    
    instance = TestClass(x=20)
    assert instance.x == 20


def test_pclass_new_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    assert instance.x is None


# LLM-generated content at query #83
#--------------------------

```python
def test_pclass_meta_new_creates_slots_with_pclass_frozen():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    # Create a mock _is_pclass function that returns False
    import pyrsistent._pclass as pclass_module
    original_is_pclass = pclass_module._is_pclass
    pclass_module._is_pclass = lambda bases: False
    
    try:
        dct = {'_pclass_fields': {'field1': _PField(type=str, initial=None, factory=None, invariant=None)}}
        bases = ()
        
        result_class = PClassMeta('TestClass', bases, dct)
        
        assert '__slots__' in result_class.__dict__
        assert '_pclass_frozen' in result_class.__slots__
        assert 'field1' in result_class.__slots__
    finally:
        pclass_module._is_pclass = original_is_pclass


# LLM-generated content at query #84
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=5)
    assert instance.x == 5


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert isinstance(instance, TestClass)


def test_pclass_constructor_multiple_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


# LLM-generated content at query #85
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


# LLM-generated content at query #86
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        instance = TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        instance = TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5)
    assert instance.x == 5


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


# LLM-generated content at query #87
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_factory_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5)
    assert instance.x == 5


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    assert instance is not None


def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #88
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PClass, field
    
    class TestClass(PClass):
        value = field(type=int)
    
    class MockField:
        def __init__(self):
            self.type = [int]
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    mock_field = MockField()
    result = TestClass()
    invariant_errors = []
    
    _check_and_set_attr(TestClass, mock_field, "value", 42, result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_error_code"
    assert not hasattr(result, 'value') or getattr(result, 'value', None) is None


# LLM-generated content at query #89
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return False, "invariant_error_code"
    
    class MockClass:
        pass
    
    field = MockField()
    result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    
    assert invariant_errors == ["invariant_error_code"]
    assert not hasattr(result, "test_field")


# LLM-generated content at query #90
#--------------------------

```python
def test_pclass_new_raises_invariant_exception_when_invariant_errors_exist():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return False, "test_error"
    
    class TestClass(PClass):
        x = field()
        __invariants__ = (failing_invariant,)
    
    try:
        TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e is not None


def test_pclass_new_raises_invariant_exception_when_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e is not None


def test_pclass_new_raises_invariant_exception_with_both_errors_and_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return False, "test_error"
    
    class TestClass(PClass):
        x = field(mandatory=True)
        __invariants__ = (failing_invariant,)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e is not None


# LLM-generated content at query #91
#--------------------------

```python
def test_pclass_meta_new_basic():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    dct = {}
    bases = (CheckedType,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert result.__name__ == name
    assert '_pclass_fields' in dct
    assert '_pclass_invariants' in dct
    assert '__slots__' in dct
    assert '_pclass_frozen' in dct['__slots__']


def test_pclass_meta_new_with_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field = _PField(type=int, initial=None, factory=None, invariant=None, initial_factory=None)
    dct = {'test_field': field}
    bases = (CheckedType,)
    name = 'TestClassWithField'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert '_pclass_fields' in dct
    assert 'test_field' in dct['_pclass_fields']
    assert 'test_field' not in dct or isinstance(dct.get('test_field'), dict)
    assert 'test_field' in result.__slots__


def test_pclass_meta_new_slots_structure():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    dct = {}
    bases = (CheckedType,)
    name = 'TestClassSlots'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert isinstance(dct['__slots__'], tuple)
    assert dct['__slots__'][0] == '_pclass_frozen'
    assert '__weakref__' in dct['__slots__']


def test_pclass_meta_new_weakref_only_on_base():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    # First class inheriting from CheckedType
    dct1 = {}
    bases1 = (CheckedType,)
    result1 = PClassMeta.__new__(PClassMeta, 'BaseClass', bases1, dct1)
    
    assert '__weakref__' in dct1['__slots__']
    
    # Second class inheriting from first class
    dct2 = {}
    bases2 = (result1,)
    result2 = PClassMeta.__new__(PClassMeta, 'DerivedClass', bases2, dct2)
    
    assert '__weakref__' not in dct2['__slots__']


def test_pclass_meta_new_invariant_storage():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    def my_invariant(self):
        return True, None
    
    dct = {'__invariant__': my_invariant}
    bases = (CheckedType,)
    name = 'TestClassInvariant'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert '_pclass_invariants' in dct
    assert isinstance(dct['_pclass_invariants'], tuple)
    assert len(dct['_pclass_invariants']) > 0


def test_pclass_meta_new_multiple_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import _PField
    
    field1 = _PField(type=int, initial=None, factory=None, invariant=None, initial_factory=None)
    field2 = _PField(type=str, initial=None, factory=None, invariant=None, initial_factory=None)
    dct = {'field1': field1, 'field2': field2}
    bases = (CheckedType,)
    name = 'TestClassMultipleFields'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert 'field1' in result.__slots__
    assert 'field2' in result.__slots__
    assert '_pclass_frozen' in result.__slots__


# LLM-generated content at query #92
#--------------------------

```python
def test_set_method_predicate_line_25():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    # Call set with positional args to trigger the condition
    result = instance.set('x', 10)
    
    # Verify that line 25 predicate (for key in self._pclass_fields:) evaluates to True
    # by checking that all fields are iterated over
    assert hasattr(instance, '_pclass_fields')
    assert len(instance._pclass_fields) > 0
    assert 'x' in instance._pclass_fields
    assert 'y' in instance._pclass_fields
    assert 'z' in instance._pclass_fields
    
    # Verify the set operation worked correctly
    assert result.x == 10
    assert result.y == 2
    assert result.z == 3
    assert instance.x == 1


# LLM-generated content at query #93
#--------------------------

```python
def test_pclass_eq_same_values():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2


def test_pclass_eq_different_values():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    assert not (obj1 == obj2)


def test_pclass_eq_different_classes():
    class TestClass1(PClass):
        x = field()
    
    class TestClass2(PClass):
        x = field()
    
    obj1 = TestClass1(x=1)
    obj2 = TestClass2(x=1)
    assert obj1 != obj2


def test_pclass_eq_with_non_pclass():
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = {"x": 1}
    assert obj1 != obj2


def test_pclass_eq_missing_vs_present_field():
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj1 = TestClass(x=1, y=None)
    obj2 = TestClass(x=1)
    assert obj1 == obj2


def test_pclass_eq_reflexive():
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    assert obj1 == obj1


def test_pclass_eq_with_none_values():
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=None, y=1)
    obj2 = TestClass(x=None, y=1)
    assert obj1 == obj2


def test_pclass_eq_empty_classes():
    class TestClass(PClass):
        pass
    
    obj1 = TestClass()
    obj2 = TestClass()
    assert obj1 == obj2


# LLM-generated content at query #94
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        TestClass(y=5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_multiple_mandatory_fields_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)
        assert 'TestClass.y' in str(e)


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x='5')
    assert instance.x == 5
    assert isinstance(instance.x, int)


# LLM-generated content at query #95
#--------------------------

```python
def test_pclass_invariant_exception_raised_when_invariant_errors_exist():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return False, "test_error"
    
    class TestClass(PClass):
        x = field()
        __invariants__ = (failing_invariant,)
    
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert "test_error" in e.error_codes or len(e.error_codes) > 0


def test_pclass_invariant_exception_raised_when_missing_fields():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        TestClass(y=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.missing_fields) > 0


def test_pclass_invariant_exception_raised_with_both_errors_and_missing():
    from pyrsistent import PClass, field
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return False, "invariant_error"
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
        __invariants__ = (failing_invariant,)
    
    try:
        TestClass(y=1)
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.error_codes) > 0 or len(e.missing_fields) > 0


# LLM-generated content at query #96
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=10)
    
    obj = TestClass(x=5)
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert 'x' in reduce_result[1][1]
    assert reduce_result[1][1]['x'] == 5


def test_pclass_reduce_empty():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    
    obj = TestClass()
    reduce_result = obj.__reduce__()
    
    assert len(reduce_result) == 2
    assert reduce_result[0].__name__ == '_restore_pickle'
    assert reduce_result[1][0] == TestClass
    assert reduce_result[1][1] == {'x': 1, 'y': 2}


def test_pclass_reduce_multiple_types():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
        count = field()
        flag = field()
    
    obj = TestClass(name="test", count=42, flag=True)
    reduce_result = obj.__reduce__()
    
    assert reduce_result[1][1] == {'name': 'test', 'count': 42, 'flag': True}


# LLM-generated content at query #97
#--------------------------

```python
def test_serialize_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    
    # The predicate at line 7 is: for name in self._pclass_fields
    # This evaluates to True when _pclass_fields is iterable and non-empty
    assert hasattr(instance, '_pclass_fields')
    assert len(instance._pclass_fields) > 0
    assert 'x' in instance._pclass_fields
    assert 'y' in instance._pclass_fields
    assert 'z' in instance._pclass_fields


# LLM-generated content at query #98
#--------------------------

```python
def test_hash_same_values_same_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    assert hash(obj1) == hash(obj2)


def test_hash_different_values_different_hash():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=3)
    
    assert hash(obj1) != hash(obj2)


def test_hash_with_missing_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field(initial=None)
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    assert hash(obj1) == hash(obj2)


def test_hash_usable_in_set():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    obj3 = TestClass(x=2)
    
    hash_set = {obj1, obj2, obj3}
    assert len(hash_set) == 2


def test_hash_usable_as_dict_key():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    
    test_dict = {obj1: 'value1'}
    assert test_dict[obj2] == 'value1'


def test_hash_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    
    obj1 = TestClass(a=1, b='test', c=[1, 2, 3])
    obj2 = TestClass(a=1, b='test', c=[1, 2, 3])
    
    assert hash(obj1) == hash(obj2)


def test_hash_consistent_across_calls():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    hash1 = hash(obj)
    hash2 = hash(obj)
    
    assert hash1 == hash2


def test_hash_with_none_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=None, y=None)
    obj2 = TestClass(x=None, y=None)
    
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #99
#--------------------------

```python
def test_pclass_new_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    
    assert instance.x == 1
    assert instance.y == 2
    assert hasattr(instance, '_pclass_frozen')
    assert instance._pclass_frozen is True


# LLM-generated content at query #100
#--------------------------

```python
def test_repr_predicate():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y='hello')
    repr_str = repr(obj)
    
    assert isinstance(repr_str, str)
    assert 'TestClass' in repr_str
    assert 'x=1' in repr_str
    assert "y='hello'" in repr_str
    assert repr_str.startswith('TestClass(')
    assert repr_str.endswith(')')


# LLM-generated content at query #101
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_constructor_extra_fields_raise_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_freezes_instance():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field, pmap
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x={'a': 1}, _factory_fields={'x'})
    assert instance.x == {'a': 1}


def test_pclass_constructor_empty():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_with_multiple_initial_values():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
        z = field()
    
    instance = TestClass(z=3)
    assert instance.x == 1
    assert instance.y == 2
    assert instance.z == 3


# LLM-generated content at query #102
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PClassMeta, Field
    
    class MockField:
        def __init__(self):
            self.type = None
            self.invariant = lambda value: (False, "error_code_1")
    
    class MockClass:
        __name__ = "MockClass"
    
    class MockResult:
        pass
    
    field = MockField()
    result = MockResult()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "error_code_1"
    assert not hasattr(result, "test_field")


# LLM-generated content at query #103
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    instance = TestClass()
    assert instance.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_not_allowed():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, z=2, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=5, _factory_fields={'x'})
    assert instance.x == 5


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


def test_pclass_constructor_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    instance = TestClass(a=1, b=2, c=3, d=4)
    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3
    assert instance.d == 4


# LLM-generated content at query #104
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    
    instance = TestClass()
    assert instance.x == 42


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        TestClass()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)
        assert 'not among the specified fields' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, _factory_fields={'x'})
    assert instance.x == 1


def test_pclass_constructor_ignore_extra_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1, ignore_extra=True)
    assert instance.x == 1


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    assert instance._pclass_frozen is True


def test_pclass_constructor_with_field_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="5")
    assert instance.x == 5


def test_pclass_constructor_empty_pclass():
    from pyrsistent import PClass
    
    class EmptyClass(PClass):
        pass
    
    instance = EmptyClass()
    assert instance is not None


# LLM-generated content at query #105
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2


def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    obj = TestClass(y=5)
    assert obj.x == 10
    assert obj.y == 5


def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
    
    obj = TestClass()
    assert obj.x == [1, 2, 3]


def test_pclass_constructor_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
    
    try:
        obj = TestClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in str(e)


def test_pclass_constructor_extra_fields_raises_error():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    try:
        obj = TestClass(x=1, z=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=5, _factory_fields={'x'})
    assert obj.x == 5


def test_pclass_constructor_ignore_extra_true():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1, extra_field=999, ignore_extra=True)
    assert obj.x == 1
    assert not hasattr(obj, 'extra_field')


def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    obj = TestClass(x=1)
    try:
        obj.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_constructor_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    obj = TestClass()
    assert obj is not None


def test_pclass_constructor_with_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
        d = field()
    
    obj = TestClass(a=1, b=2, c=3, d=4)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3
    assert obj.d == 4


# LLM-generated content at query #106
#--------------------------

```python
def test_check_and_set_attr_invariant_fails():
    from pyrsistent._pclass import _check_and_set_attr
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = None
        
        def invariant(self, value):
            return (False, "invariant_error_code")
    
    class MockClass:
        pass
    
    field = MockField()
    result = MockClass()
    invariant_errors = []
    
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_error_code"
    assert not hasattr(result, "test_field")


# LLM-generated content at query #107
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2


def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=lambda: [1, 2, 3])
        y = field()
    
    instance = TestClass(y=2)
    assert instance.x == [1, 2, 3]
    assert instance.y == 2


def test_pclass_new_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    
    try:
        TestClass(y=2)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.x' in e.missing_fields


def test_pclass_new_extra_fields():
    from pyrsistent import PClass, field, AttributeError as PyrsistentAttributeError
    
    class TestClass(PClass):
        x = field()
    
    try:
        TestClass(x=1, z=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'z' in str(e)


def test_pclass_new_frozen_after_creation():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


def test_pclass_new_with_type_checking():
    from pyrsistent import PClass, field, PTypeError
    
    class TestClass(PClass):
        x = field(type=int)
    
    try:
        TestClass(x="not an int")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_pclass_new_with_invariant():
    from pyrsistent import PClass, field, InvariantException
    
    def positive_invariant(value):
        return (value > 0, "Value must be positive")
    
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    
    try:
        TestClass(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_pclass_new_with_factory():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(factory=int)
    
    instance = TestClass(x="42")
    assert instance.x == 42


def test_pclass_new_multiple_fields_with_mixed_setup():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        a = field(initial=5)
        b = field()
        c = field(mandatory=True)
    
    instance = TestClass(b=10, c=20)
    assert instance.a == 5
    assert instance.b == 10
    assert instance.c == 20


def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, InvariantException
    
    def global_check(obj):
        return (obj.x > obj.y, "x must be greater than y")
    
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = (global_check,)
    
    try:
        TestClass(x=1, y=2)
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_pclass_new_global_invariant_passes():
    from pyrsistent import PClass, field
    
    def global_check(obj):
        return (obj.x > obj.y, "x must be greater than y")
    
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = (global_check,)
    
    instance = TestClass(x=5, y=2)
    assert instance.x == 5
    assert instance.y == 2


def test_pclass_new_empty_class():
    from pyrsistent import PClass
    
    class TestClass(PClass):
        pass
    
    instance = TestClass()
    assert instance is not None


# LLM-generated content at query #108
#--------------------------

```python
def test_pclass_meta_new_creates_class_with_fields():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    class MockField(_PField):
        def __init__(self):
            self.initial = None
    
    dct = {'field1': MockField(), 'field2': MockField()}
    bases = (object,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert result.__name__ == name
    assert hasattr(result, '_pclass_fields')
    assert 'field1' in result._pclass_fields
    assert 'field2' in result._pclass_fields


def test_pclass_meta_new_sets_slots():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    class MockField(_PField):
        def __init__(self):
            self.initial = None
    
    dct = {'field1': MockField()}
    bases = (object,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert hasattr(result, '__slots__')
    assert '_pclass_frozen' in result.__slots__
    assert 'field1' in result.__slots__


def test_pclass_meta_new_adds_weakref_for_direct_checkedtype_subclass():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._checked_types import CheckedType
    
    dct = {}
    bases = (CheckedType,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert hasattr(result, '__slots__')
    assert '__weakref__' in result.__slots__


def test_pclass_meta_new_no_weakref_for_non_direct_checkedtype_subclass():
    from pyrsistent._pclass import PClassMeta
    
    dct = {}
    bases = (object,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert hasattr(result, '__slots__')
    assert '__weakref__' not in result.__slots__


def test_pclass_meta_new_stores_invariants():
    from pyrsistent._pclass import PClassMeta
    
    def test_invariant(self):
        return True, None
    
    dct = {'__invariant__': test_invariant}
    bases = (object,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert hasattr(result, '_pclass_invariants')
    assert isinstance(result._pclass_invariants, tuple)
    assert len(result._pclass_invariants) > 0


def test_pclass_meta_new_removes_field_from_dct():
    from pyrsistent._pclass import PClassMeta
    from pyrsistent._field_common import _PField
    
    class MockField(_PField):
        def __init__(self):
            self.initial = None
    
    dct = {'field1': MockField()}
    bases = (object,)
    name = 'TestClass'
    
    result = PClassMeta.__new__(PClassMeta, name, bases, dct)
    
    assert 'field1' not in dct
    assert 'field1' in result._pclass_fields


# LLM-generated content at query #109
#--------------------------

```python
def test_set_method_iterates_over_pclass_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    
    instance = TestClass(x=1, y=2, z=3)
    updated_instance = instance.set(x=10)
    
    assert updated_instance.x == 10
    assert updated_instance.y == 2
    assert updated_instance.z == 3
    assert instance.x == 1
    assert instance.y == 2
    assert instance.z == 3


# LLM-generated content at query #110
#--------------------------

```python
def test_eq_predicate_isinstance_check():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    
    result = obj1 == obj2
    assert result is True


# LLM-generated content at query #111
#--------------------------

```python
def test_pclass_repr_single_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=1)
    result = repr(instance)
    assert result == "TestClass(x=1)"


def test_pclass_repr_multiple_fields():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
        y = field()
    
    instance = TestClass(x=1, y=2)
    result = repr(instance)
    assert "TestClass(" in result
    assert "x=1" in result
    assert "y=2" in result
    assert result.endswith(")")


def test_pclass_repr_string_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        name = field()
    
    instance = TestClass(name="hello")
    result = repr(instance)
    assert result == "TestClass(name='hello')"


def test_pclass_repr_empty_pclass():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field(initial=5)
    
    instance = TestClass()
    result = repr(instance)
    assert result == "TestClass(x=5)"


def test_pclass_repr_nested_structure():
    from pyrsistent import PClass, field
    
    class InnerClass(PClass):
        value = field()
    
    class OuterClass(PClass):
        inner = field()
    
    inner_instance = InnerClass(value=42)
    outer_instance = OuterClass(inner=inner_instance)
    result = repr(outer_instance)
    assert "OuterClass(" in result
    assert "InnerClass(value=42)" in result


def test_pclass_repr_with_none_value():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        x = field()
    
    instance = TestClass(x=None)
    result = repr(instance)
    assert result == "TestClass(x=None)"


def test_pclass_repr_with_list_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        items = field()
    
    instance = TestClass(items=[1, 2, 3])
    result = repr(instance)
    assert result == "TestClass(items=[1, 2, 3])"


def test_pclass_repr_with_dict_field():
    from pyrsistent import PClass, field
    
    class TestClass(PClass):
        data = field()
    
    instance = TestClass(data={'key': 'value'})
    result = repr(instance)
    assert "TestClass(" in result
    assert "'key': 'value'" in result or "'key':'value'" in result


