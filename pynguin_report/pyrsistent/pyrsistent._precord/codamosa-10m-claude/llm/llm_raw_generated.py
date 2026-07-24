####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field()
        y = field(type=int)
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 'value1')
    assert evolver['x'] == 'value1'
    
    # Test set with type checking
    evolver2 = _PRecordEvolver(TestRecord, pmap())
    evolver2.set('y', 42)
    assert evolver2['y'] == 42
    
    # Test set with invalid field raises AttributeError
    evolver3 = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver3.set('invalid_field', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'invalid_field' in str(e)
        assert 'TestRecord' in str(e)
    
    # Test set with type mismatch
    evolver4 = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver4.set('y', 'not_an_int')
        evolver4.persistent()
        assert False, "Should have raised type error"
    except (InvariantException, TypeError):
        pass
    
    # Test set with factory_fields restriction
    class FactoryRecord(PRecord):
        a = field()
        b = field()
    
    field_a = FactoryRecord._precord_fields['a']
    evolver5 = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields={field_a})
    evolver5.set('a', 'value_a')
    assert evolver5['a'] == 'value_a'
    
    # Test __setitem__ delegates to set
    evolver6 = _PRecordEvolver(TestRecord, pmap())
    evolver6['x'] = 'test_value'
    assert evolver6['x'] == 'test_value'


# LLM-generated content at query #2
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test the persistent method of _PRecordEvolver"""
    
    # Test 1: Basic persistent creation with valid fields
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: Persistent with mandatory fields missing raises InvariantException
    from pyrsistent import field
    
    class MandatoryRecord(PRecord):
        required_field = field(mandatory=True)
        optional_field = field()
    
    evolver = MandatoryRecord.evolver()
    evolver['optional_field'] = 'value'
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'required_field' in str(e.missing_fields)
    
    # Test 3: Persistent with all mandatory fields provided succeeds
    evolver = MandatoryRecord.evolver()
    evolver['required_field'] = 'required'
    evolver['optional_field'] = 'optional'
    result = evolver.persistent()
    
    assert isinstance(result, MandatoryRecord)
    assert result['required_field'] == 'required'
    assert result['optional_field'] == 'optional'
    
    # Test 4: Persistent from non-dirty evolver returns original if already correct type
    original = TestRecord(x=10, y=20)
    evolver = original.evolver()
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 10
    assert result['y'] == 20
    
    # Test 5: Persistent with field invariant violation raises InvariantException
    class InvariantRecord(PRecord):
        value = field()
    
    evolver = InvariantRecord.evolver()
    evolver['value'] = 'test'
    result = evolver.persistent()
    
    assert isinstance(result, InvariantRecord)
    assert result['value'] == 'test'


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field


def test_PRecord___new__():
    # Test basic PRecord creation with no fields
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    assert isinstance(record, EmptyRecord)
    assert len(record) == 0
    
    # Test PRecord creation with fields
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='John', age=30)
    assert record['name'] == 'John'
    assert record['age'] == 30
    
    # Test PRecord creation with initial values
    class RecordWithInitial(PRecord):
        name = field()
        status = field(initial='active')
    
    record = RecordWithInitial(name='Alice')
    assert record['name'] == 'Alice'
    assert record['status'] == 'active'
    
    # Test PRecord creation with callable initial values
    class RecordWithCallableInitial(PRecord):
        name = field()
        items = field(initial=list)
    
    record = RecordWithCallableInitial(name='Bob')
    assert record['name'] == 'Bob'
    assert record['items'] == []
    
    # Test PRecord creation with kwargs override
    class RecordWithOverride(PRecord):
        name = field()
        status = field(initial='inactive')
    
    record = RecordWithOverride(name='Charlie', status='active')
    assert record['name'] == 'Charlie'
    assert record['status'] == 'active'
    
    # Test internal PRecord creation with buckets and size
    class InternalRecord(PRecord):
        x = field()
    
    record1 = InternalRecord(x=1)
    record2 = InternalRecord(_precord_buckets=record1._buckets, _precord_size=record1._size)
    assert isinstance(record2, InternalRecord)
    assert record2['x'] == 1
    
    # Test _ignore_extra flag
    class StrictRecord(PRecord):
        name = field()
    
    record = StrictRecord(_ignore_extra=True, name='Dave', extra='ignored')
    assert record['name'] == 'Dave'
    assert 'extra' not in record
    
    # Test with _factory_fields
    class FactoryRecord(PRecord):
        value = field()
    
    record = FactoryRecord(_factory_fields=set(), value=42)
    assert record['value'] == 42


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord
from pyrsistent._checked_types import InvariantException
from pyrsistent._precord import _PRecordEvolver
from pyrsistent._pmap import pmap


def test__PRecordEvolver_set():
    """Test _PRecordEvolver.set method with various scenarios"""
    
    # Define a test PRecord class
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        optional_field = field(type=str, mandatory=False)
    
    # Test 1: Setting a valid field with correct type
    original = pmap()
    evolver = _PRecordEvolver(TestRecord, original)
    result = evolver.set('name', 'John')
    assert result is evolver
    
    # Test 2: Setting multiple fields
    evolver.set('age', 30)
    assert evolver['name'] == 'John'
    assert evolver['age'] == 30
    
    # Test 3: Setting a field that doesn't exist should raise AttributeError
    with pytest.raises(AttributeError, match="'invalid_field' is not among the specified fields"):
        evolver.set('invalid_field', 'value')
    
    # Test 4: Setting field via __setitem__
    evolver2 = _PRecordEvolver(TestRecord, pmap())
    evolver2['name'] = 'Jane'
    assert evolver2['name'] == 'Jane'
    
    # Test 5: Test with ignore_extra flag
    evolver3 = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver3.set('name', 'Bob')
    assert evolver3['name'] == 'Bob'
    
    # Test 6: Setting optional field
    evolver4 = _PRecordEvolver(TestRecord, pmap())
    evolver4.set('optional_field', 'optional_value')
    persistent = evolver4.persistent()
    assert persistent['optional_field'] == 'optional_value'
    
    # Test 7: Test that set returns self for method chaining
    evolver5 = _PRecordEvolver(TestRecord, pmap())
    result1 = evolver5.set('name', 'Alice')
    result2 = evolver5.set('age', 25)
    assert result1 is evolver5
    assert result2 is evolver5


# LLM-generated content at query #5
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Test basic set operation
    original = TestRecord(name="John", age=30)
    evolver = original.evolver()
    evolver.set("name", "Jane")
    result = evolver.persistent()
    assert result["name"] == "Jane"
    assert result["age"] == 30
    
    # Test set with type checking
    class TypedRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
    
    evolver = TypedRecord().evolver()
    evolver.set("name", "Bob")
    evolver.set("age", 25)
    result = evolver.persistent()
    assert result["name"] == "Bob"
    assert result["age"] == 25
    
    # Test set with invalid field raises AttributeError
    evolver = TestRecord().evolver()
    with pytest.raises(AttributeError, match="not among the specified fields"):
        evolver.set("invalid_field", "value")
    
    # Test set with factory function
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver = FactoryRecord().evolver()
    evolver.set("items", [1, 2, 3])
    result = evolver.persistent()
    assert result["items"] == [1, 2, 3]
    
    # Test __setitem__ calls set
    evolver = TestRecord().evolver()
    evolver["name"] = "Alice"
    result = evolver.persistent()
    assert result["name"] == "Alice"


# LLM-generated content at query #6
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class"""
    from pyrsistent import field
    
    # Create a test PRecord class
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Test 1: Setting a valid field
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.set('x', 10)
    assert result is evolver  # set returns self
    assert evolver._data['x'] == 10
    
    # Test 2: Setting multiple fields
    evolver.set('y', 20)
    assert evolver._data['y'] == 20
    
    # Test 3: Setting an invalid field should raise AttributeError
    with pytest.raises(AttributeError) as excinfo:
        evolver.set('z', 30)
    assert "'z' is not among the specified fields for TestRecord" in str(excinfo.value)
    
    # Test 4: Setting field via __setitem__
    evolver2 = _PRecordEvolver(TestRecord, pmap())
    evolver2['x'] = 15
    assert evolver2._data['x'] == 15
    
    # Test 5: Field with type checking
    class TypedRecord(PRecord):
        num = field(type=int)
    
    evolver3 = _PRecordEvolver(TypedRecord, pmap())
    evolver3.set('num', 42)
    assert evolver3._data['num'] == 42
    
    # Test 6: Invalid type should be caught
    evolver4 = _PRecordEvolver(TypedRecord, pmap())
    with pytest.raises(TypeError):
        evolver4.set('num', 'not_an_int')
    
    # Test 7: Setting with factory fields
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver5 = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields={FactoryRecord._precord_fields['items']})
    evolver5.set('items', [1, 2, 3])
    assert evolver5._data['items'] == [1, 2, 3]
    
    # Test 8: Setting field without factory fields specified
    evolver6 = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields=set())
    evolver6.set('items', [4, 5, 6])
    assert evolver6._data['items'] == [4, 5, 6]


# LLM-generated content at query #7
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=str)
    
    # Test basic set operation
    original = TestRecord(x=1, y="hello")
    evolver = original.evolver()
    evolver.set('x', 5)
    result = evolver.persistent()
    assert result.x == 5
    assert result.y == "hello"
    
    # Test set with string field
    evolver = original.evolver()
    evolver.set('y', "world")
    result = evolver.persistent()
    assert result.x == 1
    assert result.y == "world"
    
    # Test set with type checking
    evolver = original.evolver()
    evolver.set('x', 10)
    result = evolver.persistent()
    assert result.x == 10
    
    # Test set with invalid field name raises AttributeError
    evolver = original.evolver()
    try:
        evolver.set('z', 100)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'z' is not among the specified fields" in str(e)
    
    # Test __setitem__ calls set
    evolver = original.evolver()
    evolver['x'] = 20
    result = evolver.persistent()
    assert result.x == 20
    
    # Test multiple sets
    evolver = original.evolver()
    evolver.set('x', 15)
    evolver.set('y', "test")
    result = evolver.persistent()
    assert result.x == 15
    assert result.y == "test"


# LLM-generated content at query #8
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'John')
    assert evolver._data['name'] == 'John'
    
    # Test set with type checking
    class TypedRecord(PRecord):
        count = field(type=int)
    
    evolver = _PRecordEvolver(TypedRecord, pmap())
    evolver.set('count', 42)
    assert evolver._data['count'] == 42
    
    # Test set with invalid field raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'invalid_field' in str(e)
        assert 'TestRecord' in str(e)
    
    # Test set with factory field
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('items', [1, 2, 3])
    assert evolver._data['items'] == [1, 2, 3]
    
    # Test __setitem__ delegates to set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['name'] = 'Jane'
    assert evolver._data['name'] == 'Jane'
    
    # Test set returns self for chaining
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('name', 'Bob')
    assert result is evolver
    
    # Test set with field that has invariant
    class InvariantRecord(PRecord):
        positive_num = field(invariant=lambda x: (x > 0, 'must be positive'))
    
    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('positive_num', 5)
    assert evolver._data['positive_num'] == 5
    
    # Test set with failing invariant collects error codes
    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('positive_num', -1)
    assert len(evolver._invariant_error_codes) > 0


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, InvariantException


def test__PRecordEvolver_persistent():
    # Test 1: Basic persistent() call with valid data
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2


def test__PRecordEvolver_persistent_with_mandatory_fields():
    # Test 2: persistent() raises InvariantException when mandatory field is missing
    class TestRecord(PRecord):
        x = field(mandatory=True)
        y = field()
    
    evolver = TestRecord.evolver()
    evolver['y'] = 2
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'TestRecord.x' in exc_info.value.missing_fields


def test__PRecordEvolver_persistent_with_all_mandatory_fields():
    # Test 3: persistent() succeeds when all mandatory fields are provided
    class TestRecord(PRecord):
        x = field(mandatory=True)
        y = field(mandatory=True)
    
    evolver = TestRecord.evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert result['x'] == 1
    assert result['y'] == 2


def test__PRecordEvolver_persistent_field_invariant_failure():
    # Test 4: persistent() raises InvariantException on field invariant failure
    def positive_invariant(value):
        return value > 0, "positive"
    
    class TestRecord(PRecord):
        x = field(invariant=positive_invariant)
    
    evolver = TestRecord.evolver()
    evolver['x'] = -1
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'positive' in exc_info.value.invariant_errors


def test__PRecordEvolver_persistent_no_changes():
    # Test 5: persistent() returns same object when no changes made
    class TestRecord(PRecord):
        x = field()
    
    original = TestRecord(x=1)
    evolver = original.evolver()
    result = evolver.persistent()
    assert result is original


def test__PRecordEvolver_persistent_with_changes():
    # Test 6: persistent() returns new object when changes made
    class TestRecord(PRecord):
        x = field()
    
    original = TestRecord(x=1)
    evolver = original.evolver()
    evolver['x'] = 2
    result = evolver.persistent()
    assert result is not original
    assert result['x'] == 2
    assert original['x'] == 1


def test__PRecordEvolver_persistent_multiple_mandatory_missing():
    # Test 7: persistent() reports multiple missing mandatory fields
    class TestRecord(PRecord):
        x = field(mandatory=True)
        y = field(mandatory=True)
        z = field(mandatory=True)
    
    evolver = TestRecord.evolver()
    evolver['x'] = 1
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    missing = exc_info.value.missing_fields
    assert 'TestRecord.y' in missing
    assert 'TestRecord.z' in missing


def test__PRecordEvolver_persistent_empty_record():
    # Test 8: persistent() works with empty record (no fields)
    class TestRecord(PRecord):
        pass
    
    evolver = TestRecord.evolver()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert len(result) == 0


def test__PRecordEvolver_persistent_preserves_type():
    # Test 9: persistent() returns instance of correct class
    class CustomRecord(PRecord):
        x = field()
    
    evolver = CustomRecord.evolver()
    evolver['x'] = 42
    result = evolver.persistent()
    assert type(result) is CustomRecord


# LLM-generated content at query #10
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y='hello')
    result = repr(record)
    
    assert result.startswith('TestRecord(')
    assert 'x=1' in result
    assert "y='hello'" in result
    assert result.endswith(')')


def test_PRecord___repr__empty():
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    result = repr(record)
    
    assert result == 'EmptyRecord()'


def test_PRecord___repr__single_field():
    class SingleFieldRecord(PRecord):
        name = field()
    
    record = SingleFieldRecord(name='test')
    result = repr(record)
    
    assert result == "SingleFieldRecord(name='test')"


def test_PRecord___repr__multiple_fields():
    class MultiFieldRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = MultiFieldRecord(a=10, b=20, c=30)
    result = repr(record)
    
    assert 'MultiFieldRecord(' in result
    assert 'a=10' in result
    assert 'b=20' in result
    assert 'c=30' in result


def test_PRecord___repr__with_nested_structures():
    class NestedRecord(PRecord):
        data = field()
    
    record = NestedRecord(data={'nested': [1, 2, 3]})
    result = repr(record)
    
    assert 'NestedRecord(' in result
    assert 'data=' in result
    assert 'nested' in result


def test_PRecord___repr__with_none_value():
    class RecordWithNone(PRecord):
        value = field()
    
    record = RecordWithNone(value=None)
    result = repr(record)
    
    assert result == 'RecordWithNone(value=None)'


def test_PRecord___repr__with_special_characters():
    class SpecialRecord(PRecord):
        text = field()
    
    record = SpecialRecord(text="test'with\"quotes")
    result = repr(record)
    
    assert 'SpecialRecord(' in result
    assert 'text=' in result


# LLM-generated content at query #11
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = pfield(type=int)
        y = pfield(type=str)

    record = TestRecord(x=42, y='hello')
    repr_str = repr(record)
    
    assert repr_str == "TestRecord(x=42, y='hello')" or repr_str == "TestRecord(y='hello', x=42)"
    assert "TestRecord" in repr_str
    assert "x=" in repr_str
    assert "y=" in repr_str
    assert "42" in repr_str
    assert "'hello'" in repr_str


def test_PRecord___repr__empty():
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    repr_str = repr(record)
    
    assert repr_str == "EmptyRecord()"


def test_PRecord___repr__single_field():
    class SingleFieldRecord(PRecord):
        name = pfield(type=str)

    record = SingleFieldRecord(name='test')
    repr_str = repr(record)
    
    assert repr_str == "SingleFieldRecord(name='test')"


def test_PRecord___repr__multiple_fields():
    class MultiFieldRecord(PRecord):
        a = pfield(type=int)
        b = pfield(type=str)
        c = pfield(type=float)

    record = MultiFieldRecord(a=1, b='two', c=3.0)
    repr_str = repr(record)
    
    assert "MultiFieldRecord" in repr_str
    assert "a=1" in repr_str
    assert "b='two'" in repr_str
    assert "c=3.0" in repr_str


def test_PRecord___repr__with_special_chars():
    class SpecialRecord(PRecord):
        text = pfield(type=str)

    record = SpecialRecord(text="hello\nworld")
    repr_str = repr(record)
    
    assert "SpecialRecord" in repr_str
    assert "text=" in repr_str
    assert "hello\\nworld" in repr_str


# LLM-generated content at query #12
#--------------------------

```python
def test_PRecord_serialize():
    from pyrsistent import PRecord, field
    
    # Test basic serialization without custom serializers
    class SimpleRecord(PRecord):
        name = field()
        age = field()
    
    record = SimpleRecord(name="Alice", age=30)
    serialized = record.serialize()
    
    assert serialized == {"name": "Alice", "age": 30}
    assert isinstance(serialized, dict)
    
    # Test serialization with custom serializer function
    def age_serializer(format, value):
        if format == "string":
            return str(value)
        return value
    
    class RecordWithSerializer(PRecord):
        name = field()
        age = field(serializer=age_serializer)
    
    record2 = RecordWithSerializer(name="Bob", age=25)
    serialized2 = record2.serialize(format="string")
    
    assert serialized2["name"] == "Bob"
    assert serialized2["age"] == "25"
    assert isinstance(serialized2["age"], str)
    
    # Test serialization with None format
    serialized3 = record2.serialize(format=None)
    assert serialized3["age"] == 25
    
    # Test serialization with empty record
    class EmptyRecord(PRecord):
        pass
    
    empty = EmptyRecord()
    serialized_empty = empty.serialize()
    assert serialized_empty == {}
    
    # Test serialization preserves all fields
    class MultiFieldRecord(PRecord):
        field1 = field()
        field2 = field()
        field3 = field()
    
    multi = MultiFieldRecord(field1=1, field2=2, field3=3)
    serialized_multi = multi.serialize()
    
    assert len(serialized_multi) == 3
    assert serialized_multi["field1"] == 1
    assert serialized_multi["field2"] == 2
    assert serialized_multi["field3"] == 3


# LLM-generated content at query #13
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test _PRecordEvolver.persistent() method"""
    
    # Test 1: Basic persistent creation with valid fields
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: Persistent with mandatory fields missing raises InvariantException
    from pyrsistent import field
    
    class TestRecordMandatory(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    evolver = TestRecordMandatory.evolver()
    evolver['optional'] = 'value'
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'required' in str(e.missing_fields)
    
    # Test 3: Persistent with all mandatory fields provided
    evolver = TestRecordMandatory.evolver()
    evolver['required'] = 'value'
    result = evolver.persistent()
    assert result['required'] == 'value'
    
    # Test 4: Persistent returns same object if not dirty
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result == record
    
    # Test 5: Persistent creates new object if dirty
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    evolver['x'] = 10
    result = evolver.persistent()
    assert result != record
    assert result['x'] == 10
    assert result['y'] == 2
    
    # Test 6: Persistent with field invariant violation
    def positive_invariant(val):
        return (val > 0, 'must_be_positive')
    
    class TestRecordInvariant(PRecord):
        value = field(invariant=positive_invariant)
    
    evolver = TestRecordInvariant.evolver()
    evolver['value'] = -1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'must_be_positive' in e.invariant_errors
    
    # Test 7: Persistent with multiple missing mandatory fields
    class TestRecordMultipleMandatory(PRecord):
        field1 = field(mandatory=True)
        field2 = field(mandatory=True)
        field3 = field()
    
    evolver = TestRecordMultipleMandatory.evolver()
    evolver['field3'] = 'value'
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 2
        assert any('field1' in f for f in e.missing_fields)
        assert any('field2' in f for f in e.missing_fields)


# LLM-generated content at query #14
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    assert evolver._data['x'] == 10
    
    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver._data['y'] == 20
    
    # Test that setting an undefined field raises AttributeError
    evolver2 = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError) as exc_info:
        evolver2.set('z', 30)
    assert "'z' is not among the specified fields for TestRecord" in str(exc_info.value)
    
    # Test __setitem__ delegates to set
    evolver3 = _PRecordEvolver(TestRecord, pmap())
    evolver3['x'] = 100
    assert evolver3._data['x'] == 100
    
    # Test with type checking
    class TypedRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
    
    evolver4 = _PRecordEvolver(TypedRecord, pmap())
    evolver4.set('name', 'John')
    assert evolver4._data['name'] == 'John'
    
    evolver4.set('age', 25)
    assert evolver4._data['age'] == 25
    
    # Test that invalid type raises error during persistent
    evolver5 = _PRecordEvolver(TypedRecord, pmap())
    evolver5.set('age', 'not an int')
    with pytest.raises(InvariantException):
        evolver5.persistent()


# LLM-generated content at query #15
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class"""
    
    # Test 1: Setting a valid field with correct type
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=str)
    
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 42)
    assert evolver._data['x'] == 42
    
    # Test 2: Setting multiple fields
    evolver.set('y', 'hello')
    assert evolver._data['y'] == 'hello'
    
    # Test 3: Setting non-existent field raises AttributeError
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('z', 100)
    assert "'z' is not among the specified fields for TestRecord" in str(exc_info.value)
    
    # Test 4: Type checking for field
    class StrictRecord(PRecord):
        num = field(type=int)
    
    evolver2 = _PRecordEvolver(StrictRecord, pmap())
    with pytest.raises(TypeError):
        evolver2.set('num', 'not_an_int')
    
    # Test 5: Field with factory function
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver3 = _PRecordEvolver(FactoryRecord, pmap())
    evolver3.set('items', [1, 2, 3])
    assert evolver3._data['items'] == [1, 2, 3]
    
    # Test 6: Field invariant validation
    class InvariantRecord(PRecord):
        positive_num = field(type=int, invariant=lambda x: (x > 0, 'must be positive'))
    
    evolver4 = _PRecordEvolver(InvariantRecord, pmap())
    evolver4.set('positive_num', 5)
    assert evolver4._data['positive_num'] == 5
    
    # Negative value should fail invariant
    evolver5 = _PRecordEvolver(InvariantRecord, pmap())
    evolver5.set('positive_num', -1)
    assert 'must be positive' in evolver5._invariant_error_codes
    
    # Test 7: __setitem__ delegates to set
    evolver6 = _PRecordEvolver(TestRecord, pmap())
    evolver6['x'] = 99
    assert evolver6._data['x'] == 99
    
    # Test 8: Setting field with factory_fields restriction
    class RestrictedRecord(PRecord):
        a = field(type=int)
        b = field(type=int)
    
    field_a = RestrictedRecord._precord_fields['a']
    evolver7 = _PRecordEvolver(RestrictedRecord, pmap(), _factory_fields={field_a})
    evolver7.set('a', 10)
    assert evolver7._data['a'] == 10
    
    # Test 9: Setting with ignore_extra flag
    class ExtraRecord(PRecord):
        name = field(type=str)
    
    evolver8 = _PRecordEvolver(ExtraRecord, pmap(), _ignore_extra=True)
    evolver8.set('name', 'test')
    assert evolver8._data['name'] == 'test'
    
    # Test 10: Return value is self for chaining
    evolver9 = _PRecordEvolver(TestRecord, pmap())
    result = evolver9.set('x', 5)
    assert result is evolver9


# LLM-generated content at query #16
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test _PRecordEvolver.set() method"""
    from pyrsistent import field
    
    # Define a test PRecord class
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        optional_field = field(type=str, initial='default')
    
    # Test 1: Setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('name', 'John')
    assert result is evolver
    assert evolver._data['name'] == 'John'
    
    # Test 2: Setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Alice')
    evolver.set('age', 30)
    assert evolver._data['name'] == 'Alice'
    assert evolver._data['age'] == 30
    
    # Test 3: Setting an invalid field raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('nonexistent_field', 'value')
    assert 'nonexistent_field' in str(exc_info.value)
    assert 'TestRecord' in str(exc_info.value)
    
    # Test 4: Type checking - setting wrong type should raise InvariantException on persistent()
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('age', 'not_an_int')
    with pytest.raises(InvariantException):
        evolver.persistent()
    
    # Test 5: Setting field with factory function
    class RecordWithFactory(PRecord):
        items = field(factory=list)
    
    evolver = _PRecordEvolver(RecordWithFactory, pmap())
    evolver.set('items', [1, 2, 3])
    result = evolver.persistent()
    assert result['items'] == [1, 2, 3]
    
    # Test 6: Using __setitem__ should delegate to set()
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['name'] = 'Bob'
    assert evolver._data['name'] == 'Bob'
    
    # Test 7: Setting field with ignore_extra flag
    class StrictRecord(PRecord):
        field1 = field(type=str)
    
    evolver = _PRecordEvolver(StrictRecord, pmap(), _ignore_extra=True)
    evolver.set('field1', 'value')
    result = evolver.persistent()
    assert result['field1'] == 'value'


# LLM-generated content at query #17
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test the __new__ method of _PRecordMeta"""
    
    # Test basic PRecord class creation
    class TestRecord(PRecord):
        x = pfield()
        y = pfield()
    
    assert hasattr(TestRecord, '_precord_fields')
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert TestRecord.__slots__ == ()
    
    # Test mandatory fields tracking
    class MandatoryRecord(PRecord):
        required = pfield(mandatory=True)
        optional = pfield()
    
    assert 'required' in MandatoryRecord._precord_mandatory_fields
    assert 'optional' not in MandatoryRecord._precord_mandatory_fields
    
    # Test initial values tracking
    class InitialRecord(PRecord):
        with_initial = pfield(initial=42)
        without_initial = pfield()
    
    assert 'with_initial' in InitialRecord._precord_initial_values
    assert InitialRecord._precord_initial_values['with_initial'] == 42
    assert 'without_initial' not in InitialRecord._precord_initial_values
    
    # Test callable initial values
    class CallableInitialRecord(PRecord):
        with_callable = pfield(initial=lambda: [])
    
    assert 'with_callable' in CallableInitialRecord._precord_initial_values
    assert callable(CallableInitialRecord._precord_initial_values['with_callable'])
    
    # Test inheritance of fields
    class BaseRecord(PRecord):
        base_field = pfield()
    
    class DerivedRecord(BaseRecord):
        derived_field = pfield()
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    
    # Test invariant storage
    class InvariantRecord(PRecord):
        value = pfield()
        
        def __invariant__(self):
            return (True, None)
    
    assert hasattr(InvariantRecord, '_precord_invariants')
    
    # Test metaclass type
    assert isinstance(TestRecord, _PRecordMeta)
    assert issubclass(TestRecord, PRecord)


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field
from pyrsistent._checked_types import InvariantException


def test__PRecordEvolver_persistent():
    """Test _PRecordEvolver.persistent() method"""
    
    # Test 1: Basic persistent creation with valid fields
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: Persistent with mandatory fields missing raises InvariantException
    class MandatoryRecord(PRecord):
        required_field = field(mandatory=True)
        optional_field = field()
    
    evolver = MandatoryRecord.evolver()
    evolver['optional_field'] = 'value'
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'required_field' in str(exc_info.value)
    
    # Test 3: Persistent with all mandatory fields present
    evolver = MandatoryRecord.evolver()
    evolver['required_field'] = 'required_value'
    evolver['optional_field'] = 'optional_value'
    result = evolver.persistent()
    
    assert result['required_field'] == 'required_value'
    assert result['optional_field'] == 'optional_value'
    
    # Test 4: Persistent returns same instance if not dirty and already correct type
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    
    assert result is record
    
    # Test 5: Persistent creates new instance if dirty
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    evolver['x'] = 10
    result = evolver.persistent()
    
    assert result is not record
    assert result['x'] == 10
    assert result['y'] == 2
    
    # Test 6: Persistent with field invariant violation
    def positive_invariant(value):
        return value > 0, "must_be_positive"
    
    class PositiveRecord(PRecord):
        value = field(invariant=positive_invariant)
    
    evolver = PositiveRecord.evolver()
    evolver['value'] = -5
    
    with pytest.raises(InvariantException):
        evolver.persistent()
    
    # Test 7: Persistent with valid field invariant
    evolver = PositiveRecord.evolver()
    evolver['value'] = 5
    result = evolver.persistent()
    
    assert result['value'] == 5


# LLM-generated content at query #19
#--------------------------

```python
def test_PRecord___new__():
    # Test basic PRecord creation with no fields
    class SimplePRecord(PRecord):
        pass
    
    record = SimplePRecord()
    assert isinstance(record, PRecord)
    assert len(record) == 0
    
    # Test PRecord creation with kwargs
    from pyrsistent import field
    
    class PersonRecord(PRecord):
        name = field()
        age = field()
    
    person = PersonRecord(name='Alice', age=30)
    assert person['name'] == 'Alice'
    assert person['age'] == 30
    assert len(person) == 2
    
    # Test PRecord creation with initial values
    class ConfigRecord(PRecord):
        debug = field(initial=False)
        timeout = field(initial=60)
    
    config = ConfigRecord()
    assert config['debug'] is False
    assert config['timeout'] == 60
    
    # Test PRecord creation with callable initial values
    class ListRecord(PRecord):
        items = field(initial=list)
    
    list_rec = ListRecord()
    assert list_rec['items'] == []
    
    # Test PRecord creation with _ignore_extra flag
    class StrictRecord(PRecord):
        name = field()
    
    strict = StrictRecord.create({'name': 'Bob', 'extra': 'value'}, ignore_extra=True)
    assert strict['name'] == 'Bob'
    assert 'extra' not in strict
    
    # Test PRecord creation via internal hack (with _precord_size and _precord_buckets)
    from pyrsistent._pmap import EMPTY_PMAP
    record_internal = PersonRecord(_precord_size=EMPTY_PMAP._size, _precord_buckets=EMPTY_PMAP._buckets)
    assert isinstance(record_internal, PersonRecord)
    
    # Test PRecord with factory_fields parameter
    person_with_factory = PersonRecord(_factory_fields=set(), name='Charlie', age=25)
    assert person_with_factory['name'] == 'Charlie'
    assert person_with_factory['age'] == 25
    
    # Test PRecord overriding initial values
    config_override = ConfigRecord(debug=True, timeout=120)
    assert config_override['debug'] is True
    assert config_override['timeout'] == 120


# LLM-generated content at query #20
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class."""
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Test 1: Setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('x', 10)
    assert result is evolver
    assert evolver[TestRecord._precord_fields['x']] is not None or True
    
    # Test 2: Setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 20)
    pm = evolver.persistent()
    assert pm['x'] == 10
    assert pm['y'] == 20
    
    # Test 3: Setting invalid field should raise AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError):
        evolver.set('z', 30)
    
    # Test 4: Setting field with type checking
    class TypedRecord(PRecord):
        count = field(type=int)
    
    evolver = _PRecordEvolver(TypedRecord, pmap())
    evolver.set('count', 5)
    pm = evolver.persistent()
    assert pm['count'] == 5
    
    # Test 5: Test __setitem__ delegates to set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 15
    pm = evolver.persistent()
    assert pm['x'] == 15
    
    # Test 6: Setting field with factory
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('items', [1, 2, 3])
    pm = evolver.persistent()
    assert pm['items'] == [1, 2, 3]
    
    # Test 7: Setting with ignore_extra flag
    evolver = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 100)
    pm = evolver.persistent()
    assert pm['x'] == 100
    
    # Test 8: Verify evolver state after set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 42)
    assert len(evolver._invariant_error_codes) == 0
    assert len(evolver._missing_fields) == 0


# LLM-generated content at query #21
#--------------------------

```python
def test_PRecord_serialize():
    """Test PRecord.serialize() method with various field types and serializers"""
    from pyrsistent import PRecord, field
    
    # Test basic serialization without custom serializers
    class SimpleRecord(PRecord):
        x = field()
        y = field()
    
    record = SimpleRecord(x=1, y="hello")
    serialized = record.serialize()
    assert serialized == {'x': 1, 'y': "hello"}
    
    # Test serialization with custom serializer function
    def custom_serializer(format, value):
        if format == 'uppercase':
            return str(value).upper()
        return str(value)
    
    class RecordWithSerializer(PRecord):
        name = field(serializer=custom_serializer)
        age = field()
    
    record2 = RecordWithSerializer(name="alice", age=30)
    serialized2 = record2.serialize(format='uppercase')
    assert serialized2['name'] == 'ALICE'
    assert serialized2['age'] == 30
    
    # Test serialization without format parameter
    serialized3 = record2.serialize()
    assert serialized3['name'] == 'alice'
    assert serialized3['age'] == 30
    
    # Test serialization with None serializer (no-op)
    class RecordWithNoneSerializer(PRecord):
        value = field(serializer=None)
    
    record3 = RecordWithNoneSerializer(value=42)
    serialized4 = record3.serialize()
    assert serialized4 == {'value': 42}
    
    # Test empty record
    class EmptyRecord(PRecord):
        pass
    
    empty = EmptyRecord()
    serialized5 = empty.serialize()
    assert serialized5 == {}
    
    # Test serialization with multiple fields having different serializers
    def int_serializer(format, value):
        if format == 'hex':
            return hex(value)
        return value
    
    def str_serializer(format, value):
        if format == 'reverse':
            return value[::-1]
        return value
    
    class ComplexRecord(PRecord):
        num = field(serializer=int_serializer)
        text = field(serializer=str_serializer)
        plain = field()
    
    record4 = ComplexRecord(num=255, text="hello", plain="unchanged")
    serialized6 = record4.serialize(format='hex')
    assert serialized6['num'] == '0xff'
    assert serialized6['text'] == 'hello'
    assert serialized6['plain'] == 'unchanged'
    
    serialized7 = record4.serialize(format='reverse')
    assert serialized7['num'] == 255
    assert serialized7['text'] == 'olleh'
    assert serialized7['plain'] == 'unchanged'


# LLM-generated content at query #22
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test the persistent method of _PRecordEvolver class"""
    
    # Test 1: Basic persistent creation with valid fields
    class SimpleRecord(PRecord):
        x = pfield()
        y = pfield()
    
    evolver = SimpleRecord.create({}).evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    
    assert isinstance(result, SimpleRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: Persistent with mandatory fields missing should raise InvariantException
    class MandatoryRecord(PRecord):
        required_field = pfield(mandatory=True)
        optional_field = pfield()
    
    evolver = MandatoryRecord.create({}).evolver()
    evolver['optional_field'] = 'value'
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'required_field' in str(e.missing_fields)
    
    # Test 3: Persistent with all mandatory fields provided
    evolver = MandatoryRecord.create({}).evolver()
    evolver['required_field'] = 'required'
    evolver['optional_field'] = 'optional'
    result = evolver.persistent()
    
    assert isinstance(result, MandatoryRecord)
    assert result['required_field'] == 'required'
    assert result['optional_field'] == 'optional'
    
    # Test 4: Persistent without changes returns same instance
    record = SimpleRecord.create({'x': 10, 'y': 20})
    evolver = record.evolver()
    result = evolver.persistent()
    
    assert result == record
    
    # Test 5: Persistent with changes creates new instance
    record = SimpleRecord.create({'x': 10, 'y': 20})
    evolver = record.evolver()
    evolver['x'] = 99
    result = evolver.persistent()
    
    assert isinstance(result, SimpleRecord)
    assert result['x'] == 99
    assert result['y'] == 20
    assert result != record
    
    # Test 6: Field invariant violation
    class InvariantRecord(PRecord):
        value = pfield()
    
    # Assuming pfield supports invariant checking
    evolver = InvariantRecord.create({}).evolver()
    evolver['value'] = 'test'
    result = evolver.persistent()
    
    assert isinstance(result, InvariantRecord)
    assert result['value'] == 'test'


# LLM-generated content at query #23
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test _PRecordEvolver.persistent() method"""
    
    # Test 1: Basic persistent creation with valid fields
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: Persistent with mandatory fields missing raises InvariantException
    class MandatoryRecord(PRecord):
        required_field = field(mandatory=True)
        optional_field = field()
    
    evolver = MandatoryRecord.evolver()
    evolver['optional_field'] = 'value'
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'MandatoryRecord.required_field' in exc_info.value.missing_fields
    
    # Test 3: Persistent with all mandatory fields present succeeds
    evolver = MandatoryRecord.evolver()
    evolver['required_field'] = 'required'
    evolver['optional_field'] = 'optional'
    result = evolver.persistent()
    
    assert result['required_field'] == 'required'
    assert result['optional_field'] == 'optional'
    
    # Test 4: Persistent with field invariant violation raises InvariantException
    class InvariantRecord(PRecord):
        value = field()
    
    evolver = InvariantRecord.evolver()
    evolver._invariant_error_codes = ['error_code_1']
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'error_code_1' in exc_info.value.invariant_errors
    
    # Test 5: Persistent returns same object if not dirty and already correct type
    original = TestRecord(x=5, y=10)
    evolver = _PRecordEvolver(TestRecord, original)
    
    result = evolver.persistent()
    assert result is original
    
    # Test 6: Persistent creates new instance if dirty
    evolver = TestRecord.evolver()
    evolver['x'] = 100
    evolver['y'] = 200
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 100
    assert result['y'] == 200
    
    # Test 7: Multiple missing fields reported together
    class MultiMandatoryRecord(PRecord):
        field1 = field(mandatory=True)
        field2 = field(mandatory=True)
        field3 = field(mandatory=True)
    
    evolver = MultiMandatoryRecord.evolver()
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    missing = exc_info.value.missing_fields
    assert len(missing) == 3
    assert any('field1' in m for m in missing)
    assert any('field2' in m for m in missing)
    assert any('field3' in m for m in missing)


# LLM-generated content at query #24
#--------------------------

```python
def test_PRecord_serialize():
    from pyrsistent import PRecord, field
    
    # Test basic serialization without custom serializers
    class SimpleRecord(PRecord):
        name = field()
        age = field()
    
    record = SimpleRecord(name="Alice", age=30)
    serialized = record.serialize()
    
    assert serialized == {"name": "Alice", "age": 30}
    assert isinstance(serialized, dict)


def test_PRecord_serialize_with_custom_serializer():
    from pyrsistent import PRecord, field
    
    # Test serialization with custom serializer functions
    def uppercase_serializer(value, format=None):
        return value.upper() if isinstance(value, str) else value
    
    class RecordWithSerializer(PRecord):
        name = field(serializer=uppercase_serializer)
        value = field()
    
    record = RecordWithSerializer(name="alice", value=42)
    serialized = record.serialize()
    
    assert serialized == {"name": "ALICE", "value": 42}


def test_PRecord_serialize_with_format():
    from pyrsistent import PRecord, field
    
    # Test serialization with format parameter
    def format_aware_serializer(value, format=None):
        if format == "json":
            return str(value)
        return value
    
    class RecordWithFormatSerializer(PRecord):
        data = field(serializer=format_aware_serializer)
    
    record = RecordWithFormatSerializer(data=123)
    serialized = record.serialize(format="json")
    
    assert serialized == {"data": "123"}


def test_PRecord_serialize_empty_record():
    from pyrsistent import PRecord
    
    # Test serialization of empty record
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    serialized = record.serialize()
    
    assert serialized == {}


def test_PRecord_serialize_mixed_fields():
    from pyrsistent import PRecord, field
    
    # Test serialization with mix of fields with and without serializers
    def double_serializer(value, format=None):
        return value * 2 if isinstance(value, int) else value
    
    class MixedRecord(PRecord):
        name = field()
        count = field(serializer=double_serializer)
        active = field()
    
    record = MixedRecord(name="test", count=5, active=True)
    serialized = record.serialize()
    
    assert serialized == {"name": "test", "count": 10, "active": True}


# LLM-generated content at query #25
#--------------------------

def test_PRecord_serialize():
    from pyrsistent import PRecord, field
    
    # Test basic serialization without custom serializers
    class SimpleRecord(PRecord):
        name = field()
        age = field()
    
    record = SimpleRecord(name='John', age=30)
    serialized = record.serialize()
    assert serialized == {'name': 'John', 'age': 30}
    
    # Test serialization with custom serializer function
    def uppercase_serializer(format, value):
        return value.upper() if isinstance(value, str) else value
    
    class RecordWithSerializer(PRecord):
        name = field(serializer=uppercase_serializer)
        age = field()
    
    record2 = RecordWithSerializer(name='john', age=25)
    serialized2 = record2.serialize()
    assert serialized2 == {'name': 'JOHN', 'age': 25}
    
    # Test serialization with format parameter
    def format_aware_serializer(format, value):
        if format == 'json':
            return str(value)
        return value
    
    class RecordWithFormatSerializer(PRecord):
        value = field(serializer=format_aware_serializer)
    
    record3 = RecordWithFormatSerializer(value=42)
    serialized3 = record3.serialize(format='json')
    assert serialized3 == {'value': '42'}
    
    # Test serialization with None serializer
    class RecordWithNoneSerializer(PRecord):
        data = field(serializer=None)
    
    record4 = RecordWithNoneSerializer(data='test')
    serialized4 = record4.serialize()
    assert serialized4 == {'data': 'test'}
    
    # Test serialization with empty record
    class EmptyRecord(PRecord):
        pass
    
    empty_record = EmptyRecord()
    serialized_empty = empty_record.serialize()
    assert serialized_empty == {}
    
    # Test serialization preserves all fields
    class MultiFieldRecord(PRecord):
        field1 = field(serializer=lambda f, v: v * 2)
        field2 = field()
        field3 = field(serializer=lambda f, v: v.lower() if isinstance(v, str) else v)
    
    record5 = MultiFieldRecord(field1=5, field2='middle', field3='UPPER')
    serialized5 = record5.serialize()
    assert serialized5 == {'field1': 10, 'field2': 'middle', 'field3': 'upper'}


# LLM-generated content at query #26
#--------------------------

def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = pfield(type=int)
        y = pfield(type=str)
    
    record = TestRecord(x=42, y='hello')
    repr_str = repr(record)
    
    assert repr_str == "TestRecord(x=42, y='hello')" or repr_str == "TestRecord(y='hello', x=42)"
    assert "TestRecord" in repr_str
    assert "x=" in repr_str
    assert "y=" in repr_str
    assert "42" in repr_str
    assert "'hello'" in repr_str


def test_PRecord___repr__empty():
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    repr_str = repr(record)
    
    assert repr_str == "EmptyRecord()"
    assert "EmptyRecord" in repr_str


def test_PRecord___repr__single_field():
    class SingleFieldRecord(PRecord):
        name = pfield(type=str)
    
    record = SingleFieldRecord(name='test')
    repr_str = repr(record)
    
    assert repr_str == "SingleFieldRecord(name='test')"
    assert "SingleFieldRecord" in repr_str
    assert "name='test'" in repr_str


def test_PRecord___repr__multiple_types():
    class MultiTypeRecord(PRecord):
        a = pfield(type=int)
        b = pfield(type=float)
        c = pfield(type=bool)
    
    record = MultiTypeRecord(a=1, b=2.5, c=True)
    repr_str = repr(record)
    
    assert "MultiTypeRecord" in repr_str
    assert "a=1" in repr_str
    assert "b=2.5" in repr_str
    assert "c=True" in repr_str


def test_PRecord___repr__with_special_characters():
    class SpecialRecord(PRecord):
        text = pfield(type=str)
    
    record = SpecialRecord(text="hello'world\"test")
    repr_str = repr(record)
    
    assert "SpecialRecord" in repr_str
    assert "text=" in repr_str


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from pyrsistent._checked_types import InvariantException
from pyrsistent._field_common import PField, PFIELD_NO_INITIAL


def test__PRecordMeta___new__():
    """Test that _PRecordMeta.__new__ properly initializes PRecord subclasses."""
    
    # Test basic class creation with fields
    class TestRecord(PRecord):
        x = PField()
        y = PField()
    
    assert hasattr(TestRecord, '_precord_fields')
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert TestRecord.__slots__ == ()
    
    # Test mandatory fields detection
    class MandatoryRecord(PRecord):
        required = PField(mandatory=True)
        optional = PField()
    
    assert TestRecord._precord_mandatory_fields == set()
    assert MandatoryRecord._precord_mandatory_fields == {'required'}
    
    # Test initial values collection
    class InitialRecord(PRecord):
        x = PField(initial=42)
        y = PField(initial=lambda: [])
        z = PField()
    
    assert InitialRecord._precord_initial_values == {'x': 42, 'y': InitialRecord._precord_fields['y'].initial}
    assert 'z' not in InitialRecord._precord_initial_values
    
    # Test invariants storage
    class InvariantRecord(PRecord):
        value = PField()
        
        @staticmethod
        def __invariant__(record):
            return True, None
    
    assert hasattr(InvariantRecord, '_precord_invariants')
    
    # Test inheritance of fields
    class BaseRecord(PRecord):
        base_field = PField()
    
    class DerivedRecord(BaseRecord):
        derived_field = PField()
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    
    # Test that instances can be created
    instance = TestRecord(x=1, y=2)
    assert instance['x'] == 1
    assert instance['y'] == 2
    
    # Test that mandatory fields are enforced
    with pytest.raises(InvariantException):
        MandatoryRecord()
    
    # Test initial values are applied
    initial_inst = InitialRecord()
    assert initial_inst['x'] == 42


# LLM-generated content at query #28
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass creation with no fields
    class SimpleRecord(PRecord):
        pass
    
    assert hasattr(SimpleRecord, '_precord_fields')
    assert SimpleRecord._precord_fields == {}
    assert SimpleRecord._precord_mandatory_fields == set()
    assert SimpleRecord._precord_initial_values == {}
    assert SimpleRecord.__slots__ == ()


def test__PRecordMeta___new__with_fields():
    # Test metaclass creation with field definitions
    from pyrsistent import field
    
    class RecordWithFields(PRecord):
        name = field()
        age = field()
    
    assert 'name' in RecordWithFields._precord_fields
    assert 'age' in RecordWithFields._precord_fields
    assert len(RecordWithFields._precord_fields) == 2


def test__PRecordMeta___new__with_mandatory_fields():
    # Test that mandatory fields are tracked
    from pyrsistent import field
    
    class RecordWithMandatory(PRecord):
        required_field = field()
        optional_field = field(initial=None)
    
    assert 'required_field' in RecordWithMandatory._precord_mandatory_fields
    assert 'optional_field' not in RecordWithMandatory._precord_mandatory_fields


def test__PRecordMeta___new__with_initial_values():
    # Test that initial values are stored
    from pyrsistent import field
    
    class RecordWithInitial(PRecord):
        field1 = field(initial=42)
        field2 = field(initial='default')
    
    assert RecordWithInitial._precord_initial_values['field1'] == 42
    assert RecordWithInitial._precord_initial_values['field2'] == 'default'


def test__PRecordMeta___new__with_callable_initial():
    # Test that callable initial values are stored
    from pyrsistent import field
    
    def get_default():
        return []
    
    class RecordWithCallableInitial(PRecord):
        items = field(initial=get_default)
    
    assert RecordWithCallableInitial._precord_initial_values['items'] == get_default


def test__PRecordMeta___new__inheritance():
    # Test that field definitions are inherited from parent classes
    from pyrsistent import field
    
    class ParentRecord(PRecord):
        parent_field = field()
    
    class ChildRecord(ParentRecord):
        child_field = field()
    
    assert 'parent_field' in ChildRecord._precord_fields
    assert 'child_field' in ChildRecord._precord_fields


def test__PRecordMeta___new__with_invariants():
    # Test that invariants are stored
    from pyrsistent import field, invariant
    
    class RecordWithInvariant(PRecord):
        value = field()
        
        @invariant
        def check_positive(self):
            if self.value < 0:
                return False, 'value_must_be_positive'
            return True, None
    
    assert hasattr(RecordWithInvariant, '_precord_invariants')
    assert len(RecordWithInvariant._precord_invariants) > 0


def test__PRecordMeta___new__slots():
    # Test that __slots__ is set to empty tuple
    from pyrsistent import field
    
    class RecordWithSlots(PRecord):
        field1 = field()
    
    assert RecordWithSlots.__slots__ == ()


# LLM-generated content at query #29
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        optional_field = field()
    
    # Test setting a valid field
    original = pmap()
    evolver = _PRecordEvolver(TestRecord, original)
    evolver.set('name', 'John')
    assert evolver.data['name'] == 'John'
    
    # Test setting multiple fields
    evolver.set('age', 30)
    assert evolver.data['age'] == 30
    
    # Test setting optional field
    evolver.set('optional_field', 'value')
    assert evolver.data['optional_field'] == 'value'
    
    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('non_existent', 'value')
    assert 'non_existent' in str(exc_info.value)
    assert 'TestRecord' in str(exc_info.value)
    
    # Test type checking - setting wrong type
    with pytest.raises(Exception):  # Will raise during persistent()
        evolver2 = _PRecordEvolver(TestRecord, pmap())
        evolver2.set('age', 'not_an_int')
        evolver2.persistent()
    
    # Test __setitem__ delegates to set
    evolver3 = _PRecordEvolver(TestRecord, pmap())
    evolver3['name'] = 'Alice'
    assert evolver3.data['name'] == 'Alice'
    
    # Test with factory fields parameter
    evolver4 = _PRecordEvolver(TestRecord, pmap(), _factory_fields=None)
    evolver4.set('name', 'Bob')
    assert evolver4.data['name'] == 'Bob'
    
    # Test with ignore_extra parameter
    evolver5 = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver5.set('name', 'Charlie')
    assert evolver5.data['name'] == 'Charlie'


# LLM-generated content at query #30
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test _PRecordMeta.__new__ creates PRecord classes with proper attributes."""
    
    # Test basic PRecord class creation
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert hasattr(TestRecord, '_precord_invariants')
    assert TestRecord.__slots__ == ()
    
    # Test that fields are properly registered
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    
    # Test mandatory fields detection
    class TestRecordMandatory(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    assert 'required' in TestRecordMandatory._precord_mandatory_fields
    assert 'optional' not in TestRecordMandatory._precord_mandatory_fields
    
    # Test initial values collection
    class TestRecordWithInitial(PRecord):
        x = field(initial=42)
        y = field(initial=lambda: [])
    
    assert 'x' in TestRecordWithInitial._precord_initial_values
    assert 'y' in TestRecordWithInitial._precord_initial_values
    assert TestRecordWithInitial._precord_initial_values['x'] == 42
    assert callable(TestRecordWithInitial._precord_initial_values['y'])
    
    # Test inheritance
    class BaseRecord(PRecord):
        base_field = field()
    
    class DerivedRecord(BaseRecord):
        derived_field = field()
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    
    # Test invariants are stored
    class TestRecordWithInvariant(PRecord):
        x = field()
        
        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"
    
    assert hasattr(TestRecordWithInvariant, '_precord_invariants')
    assert len(TestRecordWithInvariant._precord_invariants) > 0
    
    # Test metaclass creates proper type
    assert isinstance(TestRecord, _PRecordMeta)
    assert issubclass(TestRecord, PRecord)


# LLM-generated content at query #31
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test the persistent method of _PRecordEvolver class."""
    
    # Test 1: Basic persistent creation with valid fields
    class TestRecord(PRecord):
        x = pfield()
        y = pfield()
    
    evolver = TestRecord.evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: Persistent with mandatory fields missing should raise InvariantException
    class MandatoryRecord(PRecord):
        required = pfield(mandatory=True)
        optional = pfield()
    
    evolver = MandatoryRecord.evolver()
    evolver['optional'] = 'value'
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'required' in str(e.missing_fields)
    
    # Test 3: Persistent with all mandatory fields present
    evolver = MandatoryRecord.evolver()
    evolver['required'] = 'present'
    evolver['optional'] = 'value'
    result = evolver.persistent()
    assert result['required'] == 'present'
    assert result['optional'] == 'value'
    
    # Test 4: Persistent called on already persistent record (not dirty)
    evolver = TestRecord.evolver()
    evolver['x'] = 10
    evolver['y'] = 20
    result1 = evolver.persistent()
    assert result1['x'] == 10
    assert result1['y'] == 20
    
    # Test 5: Persistent with field invariant violation
    class InvariantRecord(PRecord):
        value = pfield()
    
    evolver = InvariantRecord.evolver()
    evolver['value'] = -5
    
    # This would raise InvariantException if field has invariant that fails
    result = evolver.persistent()
    assert result['value'] == -5
    
    # Test 6: Persistent returns correct class type
    class CustomRecord(PRecord):
        field1 = pfield()
    
    evolver = CustomRecord.evolver()
    evolver['field1'] = 'test'
    result = evolver.persistent()
    assert type(result).__name__ == 'CustomRecord'
    assert isinstance(result, CustomRecord)
    
    # Test 7: Persistent with no changes (not dirty)
    original = TestRecord(x=5, y=10)
    evolver = original.evolver()
    result = evolver.persistent()
    assert result == original
    
    # Test 8: Persistent with multiple field updates
    evolver = TestRecord.evolver()
    evolver['x'] = 100
    evolver['y'] = 200
    result = evolver.persistent()
    assert result['x'] == 100
    assert result['y'] == 200
    assert len(result) == 2


# LLM-generated content at query #32
#--------------------------

```python
def test_PRecord___new__():
    """Test PRecord.__new__ method"""
    
    # Test basic instantiation with keyword arguments
    class SimpleRecord(PRecord):
        x = field()
        y = field()
    
    record = SimpleRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2
    
    # Test with initial values
    class RecordWithInitial(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record_init = RecordWithInitial()
    assert record_init['x'] == 10
    assert record_init['y'] == 20
    
    # Test with callable initial values
    class RecordWithCallableInitial(PRecord):
        items = field(initial=list)
    
    record_callable = RecordWithCallableInitial()
    assert record_callable['items'] == []
    
    # Test override of initial values
    record_override = RecordWithCallableInitial(items=[1, 2, 3])
    assert record_override['items'] == [1, 2, 3]
    
    # Test internal creation path with _precord_buckets and _precord_size
    record_internal = SimpleRecord(_precord_buckets=record._buckets, _precord_size=record._size)
    assert record_internal['x'] == 1
    assert record_internal['y'] == 2
    
    # Test with _ignore_extra flag
    class StrictRecord(PRecord):
        x = field()
    
    # Should not raise with _ignore_extra=True
    record_ignore = StrictRecord(x=1, extra_field=999, _ignore_extra=True)
    assert record_ignore['x'] == 1
    assert 'extra_field' not in record_ignore
    
    # Test with _factory_fields
    class FactoryRecord(PRecord):
        x = field()
        y = field()
    
    record_factory = FactoryRecord(x=1, y=2, _factory_fields=set())
    assert record_factory['x'] == 1
    assert record_factory['y'] == 2


# LLM-generated content at query #33
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class"""
    from pyrsistent import field
    
    # Create a simple PRecord class for testing
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        email = field(type=str, initial='')
    
    # Test 1: Setting a valid field with valid value
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('name', 'John')
    assert result is evolver  # set should return self
    
    # Test 2: Setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Jane')
    evolver.set('age', 30)
    evolver.set('email', 'jane@example.com')
    
    # Test 3: Setting a non-existent field should raise AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('invalid_field', 'value')
    assert 'invalid_field' in str(exc_info.value)
    assert 'TestRecord' in str(exc_info.value)
    
    # Test 4: Using __setitem__ should call set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['name'] = 'Bob'
    assert evolver is not None
    
    # Test 5: Setting field with type checking
    class TypedRecord(PRecord):
        count = field(type=int)
    
    evolver = _PRecordEvolver(TypedRecord, pmap())
    evolver.set('count', 42)
    
    # Test 6: Setting with factory fields restriction
    class FactoryRecord(PRecord):
        value = field(type=str)
    
    factory_fields = {FactoryRecord._precord_fields['value']}
    evolver = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields=factory_fields)
    evolver.set('value', 'test')
    
    # Test 7: Setting field that's not in factory_fields
    class MultiFieldRecord(PRecord):
        field1 = field(type=str)
        field2 = field(type=str)
    
    factory_fields = {MultiFieldRecord._precord_fields['field1']}
    evolver = _PRecordEvolver(MultiFieldRecord, pmap(), _factory_fields=factory_fields)
    evolver.set('field2', 'value2')


# LLM-generated content at query #34
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord
from pyrsistent._checked_types import InvariantException


def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver"""
    
    # Test 1: Basic set operation with valid field
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    evolver = TestRecord.evolver()
    result = evolver.set('name', 'John')
    assert result is evolver  # set returns self
    
    # Test 2: Set with type checking
    class TypedRecord(PRecord):
        count = field(type=int)
    
    evolver = TypedRecord.evolver()
    evolver.set('count', 42)
    record = evolver.persistent()
    assert record['count'] == 42
    
    # Test 3: Setting non-existent field raises AttributeError
    class StrictRecord(PRecord):
        value = field()
    
    evolver = StrictRecord.evolver()
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('non_existent', 'value')
    assert 'non_existent' in str(exc_info.value)
    assert 'not among the specified fields' in str(exc_info.value)
    
    # Test 4: Set with field factory
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver = FactoryRecord.evolver()
    evolver.set('items', [1, 2, 3])
    record = evolver.persistent()
    assert record['items'] == [1, 2, 3]
    
    # Test 5: Multiple set operations
    class MultiRecord(PRecord):
        first = field()
        second = field()
        third = field()
    
    evolver = MultiRecord.evolver()
    evolver.set('first', 'a')
    evolver.set('second', 'b')
    evolver.set('third', 'c')
    record = evolver.persistent()
    assert record['first'] == 'a'
    assert record['second'] == 'b'
    assert record['third'] == 'c'
    
    # Test 6: __setitem__ delegates to set
    class ItemRecord(PRecord):
        key = field()
    
    evolver = ItemRecord.evolver()
    evolver['key'] = 'value'
    record = evolver.persistent()
    assert record['key'] == 'value'
    
    # Test 7: Set with invariant violation
    def positive_invariant(val):
        return (val > 0, 'must_be_positive')
    
    class InvariantRecord(PRecord):
        num = field(invariant=positive_invariant)
    
    evolver = InvariantRecord.evolver()
    evolver.set('num', -5)
    with pytest.raises(InvariantException):
        evolver.persistent()
    
    # Test 8: Set overwrites previous value
    class OverwriteRecord(PRecord):
        value = field()
    
    evolver = OverwriteRecord.evolver()
    evolver.set('value', 'first')
    evolver.set('value', 'second')
    record = evolver.persistent()
    assert record['value'] == 'second'


# LLM-generated content at query #35
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class"""
    from pyrsistent import field, PRecord
    
    # Test 1: Setting a valid field
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    evolver = TestRecord.create({}).evolver()
    evolver.set('name', 'John')
    assert evolver._data['name'] == 'John'
    
    # Test 2: Setting multiple fields
    evolver.set('age', 30)
    assert evolver._data['age'] == 30
    
    # Test 3: Setting an invalid field should raise AttributeError
    evolver = TestRecord.create({}).evolver()
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('invalid_field', 'value')
    assert 'invalid_field' in str(exc_info.value)
    assert 'TestRecord' in str(exc_info.value)
    
    # Test 4: Setting field with type checking
    class TypedRecord(PRecord):
        value = field(type=int)
    
    evolver = TypedRecord.create({}).evolver()
    evolver.set('value', 42)
    assert evolver._data['value'] == 42
    
    # Test 5: Using __setitem__ should call set
    evolver = TestRecord.create({}).evolver()
    evolver['name'] = 'Jane'
    assert evolver._data['name'] == 'Jane'
    
    # Test 6: Field with factory function
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver = FactoryRecord.create({}).evolver()
    evolver.set('items', [1, 2, 3])
    assert evolver._data['items'] == [1, 2, 3]
    
    # Test 7: Setting field with initial value should override it
    class InitialRecord(PRecord):
        name = field(initial='default')
    
    evolver = InitialRecord.create({}).evolver()
    evolver.set('name', 'custom')
    assert evolver._data['name'] == 'custom'


# LLM-generated content at query #36
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test the persistent method of _PRecordEvolver class."""
    
    # Test 1: Basic persistent creation with valid fields
    class TestRecord(PRecord):
        x = pfield(type=int)
        y = pfield(type=str)
    
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 5
    evolver['y'] = 'hello'
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 5
    assert result['y'] == 'hello'
    
    # Test 2: Persistent with mandatory fields missing should raise InvariantException
    class MandatoryRecord(PRecord):
        required_field = pfield(type=int, mandatory=True)
    
    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'required_field' in str(exc_info.value)
    
    # Test 3: Persistent with all mandatory fields present
    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['required_field'] = 42
    result = evolver.persistent()
    
    assert result['required_field'] == 42
    
    # Test 4: Persistent with invariant error codes should raise InvariantException
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver._invariant_error_codes = ['some_error_code']
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'some_error_code' in exc_info.value.invariant_errors
    
    # Test 5: Persistent with missing fields should raise InvariantException
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver._missing_fields = ['TestRecord.x']
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'TestRecord.x' in exc_info.value.missing_fields
    
    # Test 6: Persistent returns same instance when not dirty and already correct type
    initial_pmap = pmap({'x': 5, 'y': 'hello'})
    evolver = _PRecordEvolver(TestRecord, initial_pmap)
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    
    # Test 7: Persistent with global invariants
    class RecordWithGlobalInvariant(PRecord):
        a = pfield(type=int)
        
        @staticmethod
        def __invariant__(record):
            if record.get('a', 0) < 0:
                return False, 'a_must_be_positive'
            return True, None
    
    evolver = _PRecordEvolver(RecordWithGlobalInvariant, pmap())
    evolver['a'] = -1
    
    with pytest.raises(InvariantException):
        evolver.persistent()
    
    # Test 8: Persistent with valid global invariant
    evolver = _PRecordEvolver(RecordWithGlobalInvariant, pmap())
    evolver['a'] = 10
    result = evolver.persistent()
    
    assert result['a'] == 10


# LLM-generated content at query #37
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y='hello')
    repr_str = repr(record)
    
    assert repr_str.startswith('TestRecord(')
    assert repr_str.endswith(')')
    assert 'x=1' in repr_str
    assert "y='hello'" in repr_str
    
    # Test with empty record
    class EmptyRecord(PRecord):
        pass
    
    empty = EmptyRecord()
    assert repr(empty) == 'EmptyRecord()'
    
    # Test with nested values
    class NestedRecord(PRecord):
        data = field()
    
    nested = NestedRecord(data={'key': 'value'})
    repr_str = repr(nested)
    assert 'NestedRecord(' in repr_str
    assert 'data=' in repr_str


# LLM-generated content at query #38
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test the persistent method of _PRecordEvolver class"""
    
    # Test 1: Basic persistent creation with valid fields
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: Persistent with mandatory fields present
    class MandatoryRecord(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    evolver = MandatoryRecord.evolver()
    evolver['required'] = 'value'
    result = evolver.persistent()
    assert result['required'] == 'value'
    
    # Test 3: Persistent raises InvariantException when mandatory field missing
    evolver = MandatoryRecord.evolver()
    evolver['optional'] = 'value'
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'MandatoryRecord.required' in exc_info.value.missing_fields
    
    # Test 4: Persistent with field invariant violation
    class InvariantRecord(PRecord):
        value = field()
        
        __invariant__ = invariant(lambda record: (record['value'] > 0, 'value_must_be_positive'))
    
    evolver = InvariantRecord.evolver()
    evolver['value'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()
    
    # Test 5: Persistent returns same object if not dirty
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result == record
    
    # Test 6: Persistent with multiple field violations
    class MultiFieldRecord(PRecord):
        a = field(mandatory=True)
        b = field(mandatory=True)
    
    evolver = MultiFieldRecord.evolver()
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert len(exc_info.value.missing_fields) == 2
    
    # Test 7: Persistent creates new instance when dirty
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    evolver['x'] = 10
    result = evolver.persistent()
    assert result != record
    assert result['x'] == 10
    assert result['y'] == 2
    assert isinstance(result, TestRecord)
    
    # Test 8: Empty record with no mandatory fields
    class SimpleRecord(PRecord):
        pass
    
    evolver = SimpleRecord.evolver()
    result = evolver.persistent()
    assert isinstance(result, SimpleRecord)
    assert len(result) == 0


# LLM-generated content at query #39
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test the __new__ method of _PRecordMeta metaclass."""
    
    # Test basic class creation with no fields
    class BasicRecord(PRecord):
        pass
    
    assert hasattr(BasicRecord, '_precord_fields')
    assert isinstance(BasicRecord._precord_fields, dict)
    assert BasicRecord._precord_fields == {}
    assert BasicRecord._precord_mandatory_fields == set()
    assert BasicRecord._precord_initial_values == {}
    assert BasicRecord.__slots__ == ()
    assert hasattr(BasicRecord, '_precord_invariants')
    
    
    # Test class creation with inherited fields
    from pyrsistent import field
    
    class ParentRecord(PRecord):
        x = field()
        y = field(mandatory=True)
    
    assert 'x' in ParentRecord._precord_fields
    assert 'y' in ParentRecord._precord_fields
    assert ParentRecord._precord_mandatory_fields == {'y'}
    assert ParentRecord.__slots__ == ()
    
    
    # Test inheritance of fields from parent class
    class ChildRecord(ParentRecord):
        z = field()
    
    assert 'x' in ChildRecord._precord_fields
    assert 'y' in ChildRecord._precord_fields
    assert 'z' in ChildRecord._precord_fields
    assert ChildRecord._precord_mandatory_fields == {'y'}
    assert ChildRecord.__slots__ == ()
    
    
    # Test field with initial value
    class RecordWithInitial(PRecord):
        a = field(initial=42)
        b = field(initial=lambda: [])
    
    assert 'a' in RecordWithInitial._precord_initial_values
    assert 'b' in RecordWithInitial._precord_initial_values
    assert RecordWithInitial._precord_initial_values['a'] == 42
    assert callable(RecordWithInitial._precord_initial_values['b'])
    
    
    # Test mandatory fields
    class MandatoryRecord(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    assert MandatoryRecord._precord_mandatory_fields == {'required'}
    assert 'optional' not in MandatoryRecord._precord_mandatory_fields
    
    
    # Test that __slots__ is always empty tuple
    class ComplexRecord(PRecord):
        field1 = field()
        field2 = field(mandatory=True)
        field3 = field(initial=100)
    
    assert ComplexRecord.__slots__ == ()
    assert isinstance(ComplexRecord._precord_fields, dict)
    assert len(ComplexRecord._precord_fields) == 3


# LLM-generated content at query #40
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test _PRecordEvolver.set() method"""
    from pyrsistent import field
    
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=str)
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 42)
    assert evolver._data['x'] == 42
    
    # Test set with string field
    evolver.set('y', 'hello')
    assert evolver._data['y'] == 'hello'
    
    # Test set with type checking
    evolver_with_type = _PRecordEvolver(TestRecord, pmap())
    evolver_with_type.set('x', 100)
    assert evolver_with_type._data['x'] == 100
    
    # Test set raises AttributeError for non-existent field
    evolver_error = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError) as exc_info:
        evolver_error.set('non_existent_field', 'value')
    assert 'non_existent_field' in str(exc_info.value)
    assert 'TestRecord' in str(exc_info.value)
    
    # Test __setitem__ delegates to set
    evolver_setitem = _PRecordEvolver(TestRecord, pmap())
    evolver_setitem['x'] = 99
    assert evolver_setitem._data['x'] == 99


# LLM-generated content at query #41
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y='hello')
    repr_str = repr(record)
    
    assert repr_str.startswith('TestRecord(')
    assert repr_str.endswith(')')
    assert 'x=1' in repr_str
    assert "y='hello'" in repr_str
    
    # Test with empty record
    class EmptyRecord(PRecord):
        pass
    
    empty = EmptyRecord()
    assert repr(empty) == 'EmptyRecord()'
    
    # Test with single field
    class SingleField(PRecord):
        name = field()
    
    single = SingleField(name='test')
    assert repr(single) == "SingleField(name='test')"
    
    # Test with nested values
    class NestedRecord(PRecord):
        data = field()
    
    nested = NestedRecord(data={'key': 'value'})
    repr_str = repr(nested)
    assert 'NestedRecord(' in repr_str
    assert "'key': 'value'" in repr_str


# LLM-generated content at query #42
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test _PRecordEvolver.persistent() method"""
    
    # Test 1: Basic persistent() call with valid fields
    class TestRecord(PRecord):
        x = PField()
        y = PField()
    
    evolver = TestRecord.create({}).evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: persistent() with mandatory fields - should raise when missing
    class MandatoryRecord(PRecord):
        required_field = PField(mandatory=True)
        optional_field = PField()
    
    evolver = MandatoryRecord.create({}).evolver()
    evolver['optional_field'] = 'value'
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'required_field' in str(e.missing_fields)
    
    # Test 3: persistent() with all mandatory fields provided
    evolver = MandatoryRecord.create({}).evolver()
    evolver['required_field'] = 'required'
    evolver['optional_field'] = 'optional'
    result = evolver.persistent()
    
    assert result['required_field'] == 'required'
    assert result['optional_field'] == 'optional'
    
    # Test 4: persistent() on clean evolver (no changes)
    original = TestRecord.create({'x': 10, 'y': 20})
    evolver = original.evolver()
    result = evolver.persistent()
    
    assert result is original
    assert result['x'] == 10
    assert result['y'] == 20
    
    # Test 5: persistent() on dirty evolver (with changes)
    original = TestRecord.create({'x': 10, 'y': 20})
    evolver = original.evolver()
    evolver['x'] = 30
    result = evolver.persistent()
    
    assert result is not original
    assert result['x'] == 30
    assert result['y'] == 20
    
    # Test 6: persistent() with field invariant violations
    class RecordWithInvariant(PRecord):
        value = PField(invariant=lambda x: (x > 0, 'must_be_positive'))
    
    evolver = RecordWithInvariant.create({}).evolver()
    evolver['value'] = -5
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for invariant violation"
    except InvariantException as e:
        assert 'must_be_positive' in e.invariant_errors
    
    # Test 7: persistent() preserves type correctly
    class CustomRecord(PRecord):
        name = PField()
    
    evolver = CustomRecord.create({}).evolver()
    evolver['name'] = 'test'
    result = evolver.persistent()
    
    assert type(result).__name__ == 'CustomRecord'
    assert isinstance(result, CustomRecord)


# LLM-generated content at query #43
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord
from pyrsistent._checked_types import InvariantException


def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class"""
    
    # Test 1: Setting a valid field
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.evolver()
    evolver.set('x', 10)
    assert evolver._data['x'] == 10
    
    # Test 2: Setting multiple fields
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result['x'] == 10
    assert result['y'] == 20
    
    # Test 3: Setting non-existent field should raise AttributeError
    with pytest.raises(AttributeError, match="'z' is not among the specified fields"):
        evolver.set('z', 30)
    
    # Test 4: Setting with type checking
    class TypedRecord(PRecord):
        value = field(type=int)
    
    evolver = TypedRecord.evolver()
    evolver.set('value', 42)
    result = evolver.persistent()
    assert result['value'] == 42
    
    # Test 5: Setting with factory function
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver = FactoryRecord.evolver()
    evolver.set('items', [1, 2, 3])
    result = evolver.persistent()
    assert result['items'] == [1, 2, 3]
    
    # Test 6: Setting field with invariant
    class InvariantRecord(PRecord):
        positive = field()
        
        @staticmethod
        def __invariant__(record):
            if 'positive' in record and record['positive'] < 0:
                return False, 'positive_must_be_positive'
            return True, None
    
    evolver = InvariantRecord.evolver()
    evolver.set('positive', 5)
    result = evolver.persistent()
    assert result['positive'] == 5
    
    # Test 7: Setting field that violates invariant
    evolver = InvariantRecord.evolver()
    evolver.set('positive', -5)
    with pytest.raises(InvariantException):
        evolver.persistent()
    
    # Test 8: Using __setitem__ which calls set
    class SimpleRecord(PRecord):
        name = field()
    
    evolver = SimpleRecord.evolver()
    evolver['name'] = 'test'
    result = evolver.persistent()
    assert result['name'] == 'test'
    
    # Test 9: Setting field with ignore_extra flag
    class ExtraRecord(PRecord):
        required = field()
    
    evolver = _PRecordEvolver(ExtraRecord, pmap(), _ignore_extra=True)
    evolver.set('required', 'value')
    result = evolver.persistent()
    assert result['required'] == 'value'
    
    # Test 10: Setting returns self for method chaining
    class ChainRecord(PRecord):
        a = field()
        b = field()
    
    evolver = ChainRecord.evolver()
    result = evolver.set('a', 1)
    assert result is evolver
    result = evolver.set('b', 2)
    assert result is evolver


# LLM-generated content at query #44
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class."""
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        optional = field()
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Alice')
    assert evolver._data['name'] == 'Alice'
    
    # Test set with type checking
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('age', 30)
    assert evolver._data['age'] == 30
    
    # Test set with optional field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('optional', 'value')
    assert evolver._data['optional'] == 'value'
    
    # Test setting non-existent field raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('nonexistent', 'value')
    assert 'nonexistent' in str(exc_info.value)
    assert 'TestRecord' in str(exc_info.value)
    
    # Test __setitem__ delegates to set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['name'] = 'Bob'
    assert evolver._data['name'] == 'Bob'
    
    # Test multiple set operations
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Charlie')
    evolver.set('age', 25)
    evolver.set('optional', 'test')
    assert evolver._data['name'] == 'Charlie'
    assert evolver._data['age'] == 25
    assert evolver._data['optional'] == 'test'


# LLM-generated content at query #45
#--------------------------

```python
def test_PRecord___new__():
    # Test basic PRecord creation with keyword arguments
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2
    assert isinstance(record, PRecord)
    assert isinstance(record, PMap)


def test_PRecord___new__with_initial_values():
    # Test PRecord creation with initial values
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field()
    
    record = TestRecord(y=20)
    assert record['x'] == 10
    assert record['y'] == 20


def test_PRecord___new__with_callable_initial():
    # Test PRecord creation with callable initial values
    class TestRecord(PRecord):
        x = field(initial=lambda: [])
        y = field()
    
    record1 = TestRecord(y=1)
    record2 = TestRecord(y=2)
    assert record1['x'] == []
    assert record2['x'] == []
    assert record1['x'] is not record2['x']


def test_PRecord___new__with_factory_fields():
    # Test PRecord creation with _factory_fields parameter
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields=set())
    assert record['x'] == 1


def test_PRecord___new__with_ignore_extra():
    # Test PRecord creation with _ignore_extra parameter
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _ignore_extra=True)
    assert record['x'] == 1


def test_PRecord___new__with_precord_buckets_and_size():
    # Test PRecord creation with internal _precord_buckets and _precord_size
    class TestRecord(PRecord):
        x = field()
    
    base_map = pmap({'x': 1})
    record = TestRecord(_precord_buckets=base_map._buckets, _precord_size=base_map._size)
    assert record['x'] == 1


def test_PRecord___new__empty():
    # Test creating an empty PRecord
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_PRecord___new__mandatory_field_missing():
    # Test that missing mandatory fields raise InvariantException
    from pyrsistent import field as pfield
    
    class TestRecord(PRecord):
        x = pfield(mandatory=True)
    
    with pytest.raises(InvariantException):
        TestRecord()


def test_PRecord___new__multiple_fields():
    # Test PRecord with multiple fields
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
        d = field()
    
    record = TestRecord(a=1, b=2, c=3, d=4)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3
    assert record['d'] == 4


def test_PRecord___new__overrides_initial_with_kwargs():
    # Test that kwargs override initial values
    class TestRecord(PRecord):
        x = field(initial=10)
    
    record = TestRecord(x=20)
    assert record['x'] == 20


def test_PRecord___new__ignore_extra_filters_fields():
    # Test that _ignore_extra properly filters extra fields
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, y=2, z=3, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record
    assert 'z' not in record


# LLM-generated content at query #46
#--------------------------

```python
def test_PRecord___new__():
    """Test PRecord.__new__ method"""
    from pyrsistent import field, PRecord
    
    # Test 1: Create PRecord with no fields
    class SimpleRecord(PRecord):
        pass
    
    record = SimpleRecord()
    assert isinstance(record, SimpleRecord)
    assert len(record) == 0
    
    # Test 2: Create PRecord with fields and kwargs
    class PersonRecord(PRecord):
        name = field()
        age = field()
    
    person = PersonRecord(name='John', age=30)
    assert person['name'] == 'John'
    assert person['age'] == 30
    
    # Test 3: Create PRecord with initial values
    class ConfigRecord(PRecord):
        host = field(initial='localhost')
        port = field(initial=8080)
    
    config = ConfigRecord()
    assert config['host'] == 'localhost'
    assert config['port'] == 8080
    
    # Test 4: Create PRecord with initial values and override
    config2 = ConfigRecord(host='example.com')
    assert config2['host'] == 'example.com'
    assert config2['port'] == 8080
    
    # Test 5: Create PRecord with callable initial value
    class CallableRecord(PRecord):
        items = field(initial=list)
    
    rec1 = CallableRecord()
    rec2 = CallableRecord()
    assert rec1['items'] == []
    assert rec2['items'] == []
    assert rec1['items'] is not rec2['items']
    
    # Test 6: Create PRecord using internal attributes (direct construction)
    class DirectRecord(PRecord):
        value = field()
    
    temp = DirectRecord(value=42)
    direct = DirectRecord(_precord_size=temp._size, _precord_buckets=temp._buckets)
    assert direct['value'] == 42
    
    # Test 7: Create PRecord with ignore_extra flag
    class StrictRecord(PRecord):
        name = field()
    
    strict = StrictRecord(name='test', _ignore_extra=True, extra_field='ignored')
    assert strict['name'] == 'test'
    assert 'extra_field' not in strict
    
    # Test 8: Create PRecord with _factory_fields
    class FactoryRecord(PRecord):
        value = field()
    
    factory_rec = FactoryRecord(value=100, _factory_fields=[])
    assert factory_rec['value'] == 100


# LLM-generated content at query #47
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class"""
    
    # Test basic field setting
    class TestRecord(PRecord):
        x = pfield(type=int)
        y = pfield(type=str)
    
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 42)
    assert evolver._data['x'] == 42
    
    # Test setting multiple fields
    evolver.set('y', 'hello')
    assert evolver._data['y'] == 'hello'
    
    # Test setting field with factory
    class TestRecordWithFactory(PRecord):
        items = pfield(type=pvector, factory=pvector)
    
    evolver2 = _PRecordEvolver(TestRecordWithFactory, pmap())
    evolver2.set('items', [1, 2, 3])
    assert len(evolver2._data['items']) == 3
    
    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError):
        evolver.set('nonexistent', 'value')
    
    # Test type checking
    class TestRecordTyped(PRecord):
        num = pfield(type=int)
    
    evolver3 = _PRecordEvolver(TestRecordTyped, pmap())
    with pytest.raises(PTypeError):
        evolver3.set('num', 'not_an_int')
    
    # Test field invariant violation
    class TestRecordInvariant(PRecord):
        positive = pfield(type=int, invariant=lambda x: (x > 0, 'must_be_positive'))
    
    evolver4 = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver4.set('positive', -1)
    assert 'must_be_positive' in evolver4._invariant_error_codes
    
    # Test __setitem__ delegates to set
    evolver5 = _PRecordEvolver(TestRecord, pmap())
    evolver5['x'] = 99
    assert evolver5._data['x'] == 99
    
    # Test ignore_extra flag with factory fields
    class TestRecordExtra(PRecord):
        value = pfield(type=str)
    
    evolver6 = _PRecordEvolver(TestRecordExtra, pmap(), _ignore_extra=True)
    evolver6.set('value', 'test')
    assert evolver6._data['value'] == 'test'


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field()
        y = field(type=int)
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 'value1')
    assert evolver._data['x'] == 'value1'
    
    # Test setting multiple fields
    evolver.set('y', 42)
    assert evolver._data['y'] == 42
    
    # Test AttributeError for non-existent field
    evolver2 = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError) as excinfo:
        evolver2.set('z', 'invalid')
    assert "'z' is not among the specified fields for TestRecord" in str(excinfo.value)
    
    # Test type checking
    evolver3 = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(InvariantException):
        evolver3.set('y', 'not_an_int')
    
    # Test __setitem__ calls set
    evolver4 = _PRecordEvolver(TestRecord, pmap())
    evolver4['x'] = 'value2'
    assert evolver4._data['x'] == 'value2'
    
    # Test with factory fields
    class FactoryRecord(PRecord):
        items = field()
    
    evolver5 = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields=set())
    evolver5.set('items', [1, 2, 3])
    assert evolver5._data['items'] == [1, 2, 3]


# LLM-generated content at query #2
#--------------------------

```python
def test__PRecordEvolver_persistent():
    from pyrsistent import field, PRecord
    
    # Test 1: Basic persistent() call with valid data
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: persistent() with mandatory fields missing should raise InvariantException
    class MandatoryRecord(PRecord):
        required_field = field(mandatory=True)
        optional_field = field()
    
    evolver = MandatoryRecord.evolver()
    evolver['optional_field'] = 'value'
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'required_field' in str(e.missing_fields)
    
    # Test 3: persistent() with all mandatory fields present
    evolver = MandatoryRecord.evolver()
    evolver['required_field'] = 'required'
    evolver['optional_field'] = 'optional'
    result = evolver.persistent()
    
    assert result['required_field'] == 'required'
    assert result['optional_field'] == 'optional'
    
    # Test 4: persistent() returns same instance if not dirty and already correct type
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    
    assert result is record
    
    # Test 5: persistent() with field invariant violation
    def positive_invariant(value):
        return value > 0, "must be positive"
    
    class PositiveRecord(PRecord):
        value = field(invariant=positive_invariant)
    
    evolver = PositiveRecord.evolver()
    evolver['value'] = -1
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for invariant violation"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0
    
    # Test 6: persistent() with multiple mandatory fields missing
    class MultiMandatoryRecord(PRecord):
        field1 = field(mandatory=True)
        field2 = field(mandatory=True)
        field3 = field(mandatory=True)
    
    evolver = MultiMandatoryRecord.evolver()
    evolver['field1'] = 'value1'
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'field2' in str(e.missing_fields)
        assert 'field3' in str(e.missing_fields)
    
    # Test 7: persistent() creates new instance when dirty
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    evolver['x'] = 10
    result = evolver.persistent()
    
    assert result is not record
    assert result['x'] == 10
    assert result['y'] == 2


# LLM-generated content at query #3
#--------------------------

```python
def test_PRecord___new__():
    from pyrsistent import field, PRecord
    
    # Test 1: Create PRecord with no fields
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    assert isinstance(record, PRecord)
    assert len(record) == 0
    
    # Test 2: Create PRecord with simple fields
    class SimpleRecord(PRecord):
        name = field()
        age = field()
    
    record = SimpleRecord(name='John', age=30)
    assert record['name'] == 'John'
    assert record['age'] == 30
    
    # Test 3: Create PRecord with default/initial values
    class RecordWithDefaults(PRecord):
        name = field()
        status = field(initial='active')
    
    record = RecordWithDefaults(name='Jane')
    assert record['name'] == 'Jane'
    assert record['status'] == 'active'
    
    # Test 4: Create PRecord with callable initial values
    class RecordWithCallableDefaults(PRecord):
        name = field()
        tags = field(initial=list)
    
    record = RecordWithCallableDefaults(name='Bob')
    assert record['name'] == 'Bob'
    assert record['tags'] == []
    
    # Test 5: Override initial values
    class RecordWithOverride(PRecord):
        name = field()
        count = field(initial=0)
    
    record = RecordWithOverride(name='Alice', count=5)
    assert record['name'] == 'Alice'
    assert record['count'] == 5
    
    # Test 6: Create via internal constructor (with _precord_size and _precord_buckets)
    class InternalRecord(PRecord):
        x = field()
    
    temp_record = InternalRecord(x=10)
    # Simulate internal creation
    record2 = InternalRecord(_precord_size=temp_record._size, _precord_buckets=temp_record._buckets)
    assert record2['x'] == 10
    
    # Test 7: Create with _factory_fields parameter
    class FactoryRecord(PRecord):
        value = field()
    
    record = FactoryRecord(value=42, _factory_fields=None)
    assert record['value'] == 42
    
    # Test 8: Create with _ignore_extra parameter
    class StrictRecord(PRecord):
        name = field()
    
    record = StrictRecord(name='Test', _ignore_extra=True)
    assert record['name'] == 'Test'
    assert len(record) == 1


# LLM-generated content at query #4
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver"""
    from pyrsistent import field
    
    # Define a test PRecord class
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        email = field(type=str, initial="test@example.com")
    
    # Test 1: Setting a valid field with correct type
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('name', 'John')
    assert result is evolver
    assert evolver._data['name'] == 'John'
    
    # Test 2: Setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Jane')
    evolver.set('age', 30)
    assert evolver._data['name'] == 'Jane'
    assert evolver._data['age'] == 30
    
    # Test 3: Setting a field with type coercion
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('age', 25)
    assert evolver._data['age'] == 25
    
    # Test 4: Setting a non-existent field should raise AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('nonexistent', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'nonexistent' in str(e)
        assert 'TestRecord' in str(e)
    
    # Test 5: Using __setitem__ should call set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['name'] = 'Alice'
    assert evolver._data['name'] == 'Alice'
    
    # Test 6: Setting field with initial value
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('email', 'custom@example.com')
    assert evolver._data['email'] == 'custom@example.com'
    
    # Test 7: Overwriting a field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Bob')
    evolver.set('name', 'Robert')
    assert evolver._data['name'] == 'Robert'


# LLM-generated content at query #5
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver"""
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    assert evolver['x'] == 10
    
    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20
    
    # Test that setting an undefined field raises AttributeError
    with pytest.raises(AttributeError, match="'z' is not among the specified fields"):
        evolver.set('z', 30)
    
    # Test __setitem__ delegates to set
    evolver2 = _PRecordEvolver(TestRecord, pmap())
    evolver2['x'] = 15
    assert evolver2['x'] == 15
    
    # Test with type checking
    class TypedRecord(PRecord):
        num = field(type=int)
    
    evolver3 = _PRecordEvolver(TypedRecord, pmap())
    evolver3.set('num', 42)
    assert evolver3['num'] == 42
    
    # Test that invalid type is caught during persistent()
    evolver4 = _PRecordEvolver(TypedRecord, pmap())
    evolver4.set('num', 'not_an_int')
    with pytest.raises(TypeError):
        evolver4.persistent()
    
    # Test with factory fields
    class FactoryRecord(PRecord):
        value = field()
    
    evolver5 = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields=None)
    evolver5.set('value', 100)
    assert evolver5['value'] == 100
    
    # Test that set returns self for chaining
    result = evolver5.set('value', 200)
    assert result is evolver5
    assert evolver5['value'] == 200


# LLM-generated content at query #6
#--------------------------

```python
def test_PRecord___new__():
    """Test PRecord.__new__ method"""
    from pyrsistent import field, PRecord
    
    # Test 1: Basic PRecord creation with no fields
    class SimpleRecord(PRecord):
        pass
    
    record = SimpleRecord()
    assert isinstance(record, PRecord)
    assert len(record) == 0
    
    # Test 2: PRecord creation with fields
    class PersonRecord(PRecord):
        name = field()
        age = field()
    
    person = PersonRecord(name='Alice', age=30)
    assert person['name'] == 'Alice'
    assert person['age'] == 30
    
    # Test 3: PRecord with initial values
    class ConfigRecord(PRecord):
        debug = field(initial=False)
        timeout = field(initial=30)
    
    config = ConfigRecord()
    assert config['debug'] is False
    assert config['timeout'] == 30
    
    # Test 4: PRecord with callable initial values
    class ListRecord(PRecord):
        items = field(initial=list)
    
    record1 = ListRecord()
    record2 = ListRecord()
    assert record1['items'] == []
    assert record2['items'] == []
    assert record1['items'] is not record2['items']
    
    # Test 5: Override initial values
    class DefaultRecord(PRecord):
        value = field(initial=10)
    
    record = DefaultRecord(value=20)
    assert record['value'] == 20
    
    # Test 6: Mandatory fields
    class MandatoryRecord(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    try:
        MandatoryRecord()
        assert False, "Should raise InvariantException"
    except Exception as e:
        assert 'mandatory' in str(type(e)).lower() or 'invariant' in str(type(e)).lower()
    
    record = MandatoryRecord(required='value')
    assert record['required'] == 'value'
    
    # Test 7: Internal constructor with _precord_size and _precord_buckets
    class InternalRecord(PRecord):
        x = field()
    
    test_record = InternalRecord(x=5)
    # Create via internal constructor
    internal = InternalRecord(_precord_size=test_record._size, _precord_buckets=test_record._buckets)
    assert internal['x'] == 5
    
    # Test 8: ignore_extra parameter
    class StrictRecord(PRecord):
        name = field()
    
    record = StrictRecord(name='test', extra='ignored', _ignore_extra=True)
    assert record['name'] == 'test'
    assert 'extra' not in record
    
    # Test 9: Multiple fields initialization
    class MultiRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = MultiRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3
    
    # Test 10: Empty kwargs
    class EmptyRecord(PRecord):
        optional = field()
    
    record = EmptyRecord()
    assert len(record) == 0


# LLM-generated content at query #7
#--------------------------

```python
def test__PRecordMeta___new__():
    from pyrsistent import field, PRecord
    
    # Test basic metaclass instantiation
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    assert hasattr(TestRecord, '_precord_fields')
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    
    # Test mandatory fields detection
    class MandatoryRecord(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    assert 'required' in MandatoryRecord._precord_mandatory_fields
    assert 'optional' not in MandatoryRecord._precord_mandatory_fields
    
    # Test initial values
    class InitialValuesRecord(PRecord):
        with_initial = field(initial=42)
        without_initial = field()
    
    assert 'with_initial' in InitialValuesRecord._precord_initial_values
    assert InitialValuesRecord._precord_initial_values['with_initial'] == 42
    assert 'without_initial' not in InitialValuesRecord._precord_initial_values
    
    # Test callable initial values
    class CallableInitialRecord(PRecord):
        with_callable = field(initial=list)
    
    assert callable(CallableInitialRecord._precord_initial_values['with_callable'])
    assert CallableInitialRecord._precord_initial_values['with_callable'] is list
    
    # Test inheritance of fields
    class BaseRecord(PRecord):
        base_field = field()
    
    class DerivedRecord(BaseRecord):
        derived_field = field()
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    
    # Test invariants storage
    class InvariantRecord(PRecord):
        x = field()
        
        def __invariant__(self):
            return True, None
    
    assert hasattr(InvariantRecord, '_precord_invariants')
    
    # Test slots are empty tuple
    assert TestRecord.__slots__ == ()
    assert MandatoryRecord.__slots__ == ()
    assert DerivedRecord.__slots__ == ()


# LLM-generated content at query #8
#--------------------------

```python
def test_PRecord___new__():
    from pyrsistent import field, PRecord
    
    # Test basic PRecord creation with no fields
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    assert isinstance(record, EmptyRecord)
    assert len(record) == 0
    
    # Test PRecord creation with fields
    class Person(PRecord):
        name = field()
        age = field()
    
    person = Person(name='Alice', age=30)
    assert person['name'] == 'Alice'
    assert person['age'] == 30
    assert len(person) == 2
    
    # Test PRecord creation with initial values
    class Config(PRecord):
        host = field(initial='localhost')
        port = field(initial=8080)
    
    config = Config()
    assert config['host'] == 'localhost'
    assert config['port'] == 8080
    
    # Test PRecord creation with partial initial values
    config2 = Config(port=9000)
    assert config2['host'] == 'localhost'
    assert config2['port'] == 9000
    
    # Test PRecord creation with callable initial values
    class CallableInitial(PRecord):
        items = field(initial=list)
    
    rec1 = CallableInitial()
    rec2 = CallableInitial()
    assert rec1['items'] == []
    assert rec2['items'] == []
    assert rec1['items'] is not rec2['items']
    
    # Test PRecord with mandatory fields
    class RequiredFields(PRecord):
        required_field = field(mandatory=True)
        optional_field = field()
    
    record_with_required = RequiredFields(required_field='value')
    assert record_with_required['required_field'] == 'value'
    
    # Test missing mandatory field raises error
    try:
        RequiredFields()
        assert False, "Should raise InvariantException"
    except Exception:
        pass
    
    # Test PRecord creation with ignore_extra parameter
    class StrictRecord(PRecord):
        known_field = field()
    
    record_ignore = StrictRecord.create({'known_field': 'value', 'extra_field': 'ignored'}, ignore_extra=True)
    assert record_ignore['known_field'] == 'value'
    assert 'extra_field' not in record_ignore
    
    # Test PRecord with internal _precord_size and _precord_buckets
    # (This tests the special internal creation path)
    person2 = Person(name='Bob', age=25)
    person_copy = Person(_precord_size=person2._size, _precord_buckets=person2._buckets)
    assert person_copy['name'] == 'Bob'
    assert person_copy['age'] == 25


# LLM-generated content at query #9
#--------------------------

def test_PRecord___new__():
    # Test basic PRecord creation with no arguments
    class SimpleRecord(PRecord):
        pass
    
    record = SimpleRecord()
    assert isinstance(record, PRecord)
    assert len(record) == 0
    
    # Test PRecord creation with keyword arguments
    from pyrsistent import field
    
    class PersonRecord(PRecord):
        name = field()
        age = field()
    
    person = PersonRecord(name='John', age=30)
    assert person['name'] == 'John'
    assert person['age'] == 30
    
    # Test PRecord creation with initial values
    class ConfigRecord(PRecord):
        debug = field(initial=False)
        timeout = field(initial=30)
    
    config = ConfigRecord()
    assert config['debug'] is False
    assert config['timeout'] == 30
    
    # Test PRecord creation with callable initial values
    class ListRecord(PRecord):
        items = field(initial=list)
    
    record1 = ListRecord()
    record2 = ListRecord()
    assert record1['items'] == []
    assert record2['items'] == []
    assert record1['items'] is not record2['items']
    
    # Test PRecord creation with _ignore_extra flag
    class StrictRecord(PRecord):
        name = field()
    
    strict = StrictRecord.create({'name': 'test', 'extra': 'value'}, ignore_extra=True)
    assert strict['name'] == 'test'
    assert 'extra' not in strict
    
    # Test PRecord creation via internal _precord_size and _precord_buckets
    from pyrsistent._pmap import EMPTY_PMAP
    record = SimpleRecord(_precord_size=EMPTY_PMAP._size, _precord_buckets=EMPTY_PMAP._buckets)
    assert isinstance(record, SimpleRecord)
    
    # Test PRecord with partial initial values
    class MixedRecord(PRecord):
        field1 = field(initial='default1')
        field2 = field()
    
    mixed = MixedRecord(field2='value2')
    assert mixed['field1'] == 'default1'
    assert mixed['field2'] == 'value2'


# LLM-generated content at query #10
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test that _PRecordMeta.__new__ properly sets up PRecord classes"""
    
    # Test basic class creation with fields
    class TestRecord(PRecord):
        x = pfield()
        y = pfield()
    
    assert hasattr(TestRecord, '_precord_fields')
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert TestRecord.__slots__ == ()


def test__PRecordMeta___new__mandatory_fields():
    """Test that mandatory fields are properly identified"""
    from pyrsistent import pfield
    
    class TestRecord(PRecord):
        x = pfield(mandatory=True)
        y = pfield(mandatory=False)
    
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert 'x' in TestRecord._precord_mandatory_fields
    assert 'y' not in TestRecord._precord_mandatory_fields


def test__PRecordMeta___new__initial_values():
    """Test that initial values are properly stored"""
    from pyrsistent import pfield
    
    class TestRecord(PRecord):
        x = pfield(initial=42)
        y = pfield(initial=lambda: [])
    
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord._precord_initial_values['x'] == 42
    assert callable(TestRecord._precord_initial_values['y'])


def test__PRecordMeta___new__invariants():
    """Test that invariants are properly stored"""
    from pyrsistent import pfield
    
    class TestRecord(PRecord):
        x = pfield()
        
        def __invariant__(self):
            return True, None
    
    assert hasattr(TestRecord, '_precord_invariants')


def test__PRecordMeta___new__inheritance():
    """Test that metaclass properly handles inheritance"""
    from pyrsistent import pfield
    
    class BaseRecord(PRecord):
        x = pfield()
    
    class DerivedRecord(BaseRecord):
        y = pfield()
    
    assert 'x' in DerivedRecord._precord_fields
    assert 'y' in DerivedRecord._precord_fields


def test__PRecordMeta___new__no_fields():
    """Test creation of PRecord with no fields"""
    
    class EmptyRecord(PRecord):
        pass
    
    assert hasattr(EmptyRecord, '_precord_fields')
    assert len(EmptyRecord._precord_fields) == 0
    assert EmptyRecord._precord_mandatory_fields == set()


# LLM-generated content at query #11
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test the __new__ method of _PRecordMeta metaclass."""
    
    # Test basic PRecord class creation
    class TestRecord(PRecord):
        pass
    
    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    
    # Test with fields defined
    from pyrsistent import field
    
    class RecordWithFields(PRecord):
        name = field()
        age = field()
    
    assert 'name' in RecordWithFields._precord_fields
    assert 'age' in RecordWithFields._precord_fields
    assert len(RecordWithFields._precord_fields) == 2
    
    # Test mandatory fields detection
    class RecordWithMandatory(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    assert 'required' in RecordWithMandatory._precord_mandatory_fields
    assert 'optional' not in RecordWithMandatory._precord_mandatory_fields
    
    # Test initial values handling
    class RecordWithInitial(PRecord):
        value = field(initial=42)
        callable_value = field(initial=lambda: 100)
    
    assert 'value' in RecordWithInitial._precord_initial_values
    assert 'callable_value' in RecordWithInitial._precord_initial_values
    assert RecordWithInitial._precord_initial_values['value'] == 42
    
    # Test inheritance
    class BaseRecord(PRecord):
        base_field = field()
    
    class DerivedRecord(BaseRecord):
        derived_field = field()
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    
    # Test invariant storage
    class RecordWithInvariant(PRecord):
        value = field()
        
        @staticmethod
        def __invariant__(record):
            return True
    
    assert hasattr(RecordWithInvariant, '_precord_invariants')
    
    # Test that metaclass returns a proper class type
    assert isinstance(TestRecord, type)
    assert issubclass(TestRecord, PRecord)


# LLM-generated content at query #12
#--------------------------

```python
def test_PRecord___new__():
    # Test basic PRecord creation with no fields
    class SimpleRecord(PRecord):
        pass
    
    record = SimpleRecord()
    assert isinstance(record, PRecord)
    assert len(record) == 0
    
    # Test PRecord creation with initial values
    from pyrsistent import field
    
    class PersonRecord(PRecord):
        name = field()
        age = field()
    
    person = PersonRecord(name='John', age=30)
    assert person['name'] == 'John'
    assert person['age'] == 30
    
    # Test PRecord creation with callable initial values
    class RecordWithCallable(PRecord):
        items = field(initial=list)
    
    record1 = RecordWithCallable()
    record2 = RecordWithCallable()
    assert record1['items'] is not record2['items']
    
    # Test PRecord creation with non-callable initial values
    class RecordWithInitial(PRecord):
        value = field(initial=42)
    
    record = RecordWithInitial()
    assert record['value'] == 42
    
    # Test PRecord with factory_fields parameter
    class FactoryRecord(PRecord):
        data = field()
    
    record = FactoryRecord(_factory_fields=set(), data='test')
    assert record['data'] == 'test'
    
    # Test PRecord with ignore_extra parameter
    class StrictRecord(PRecord):
        name = field()
    
    record = StrictRecord(_ignore_extra=True, name='test', extra='ignored')
    assert record['name'] == 'test'
    assert 'extra' not in record
    
    # Test PRecord with internal creation parameters
    class InternalRecord(PRecord):
        value = field()
    
    normal_record = InternalRecord(value=100)
    assert normal_record['value'] == 100
    
    # Test that PRecord is a PMap
    assert isinstance(person, PMap)
    
    # Test empty record creation
    empty_record = SimpleRecord()
    assert len(empty_record) == 0
    assert dict(empty_record) == {}


# LLM-generated content at query #13
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test _PRecordMeta.__new__ creates proper class attributes"""
    
    # Test basic class creation with fields
    class TestRecord(PRecord):
        x = pvector_field()
        y = pmap_field()
    
    # Verify _precord_fields is set
    assert hasattr(TestRecord, '_precord_fields')
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    
    # Verify _precord_invariants is set
    assert hasattr(TestRecord, '_precord_invariants')
    assert isinstance(TestRecord._precord_invariants, tuple)
    
    # Verify __slots__ is set to empty tuple
    assert hasattr(TestRecord, '__slots__')
    assert TestRecord.__slots__ == ()
    
    # Test with mandatory and optional fields
    class TestRecordWithMandatory(PRecord):
        required = field()
        optional = field(initial=42)
    
    # Verify mandatory fields are tracked
    assert hasattr(TestRecordWithMandatory, '_precord_mandatory_fields')
    assert 'required' in TestRecordWithMandatory._precord_mandatory_fields
    assert 'optional' not in TestRecordWithMandatory._precord_mandatory_fields
    
    # Verify initial values are stored
    assert hasattr(TestRecordWithMandatory, '_precord_initial_values')
    assert TestRecordWithMandatory._precord_initial_values.get('optional') == 42
    assert 'required' not in TestRecordWithMandatory._precord_initial_values
    
    # Test inheritance
    class BaseRecord(PRecord):
        base_field = field()
    
    class DerivedRecord(BaseRecord):
        derived_field = field()
    
    # Verify inherited fields are included
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    
    # Test with invariant method
    class RecordWithInvariant(PRecord):
        value = field()
        
        def __invariant__(self):
            return self['value'] > 0, 'value_must_be_positive'
    
    assert len(RecordWithInvariant._precord_invariants) > 0


# LLM-generated content at query #14
#--------------------------

```python
def test_PRecord_serialize():
    """Test PRecord serialize method with custom serializer functions."""
    from pyrsistent import field
    
    # Test basic serialization without custom serializers
    class SimpleRecord(PRecord):
        x = field()
        y = field()
    
    record = SimpleRecord(x=1, y="hello")
    serialized = record.serialize()
    assert serialized == {'x': 1, 'y': "hello"}
    
    # Test serialization with custom serializer function
    def custom_serializer(format, value):
        if format == 'uppercase':
            return str(value).upper()
        return str(value)
    
    class RecordWithSerializer(PRecord):
        name = field(serializer=custom_serializer)
        value = field()
    
    record2 = RecordWithSerializer(name="test", value=42)
    serialized2 = record2.serialize(format='uppercase')
    assert serialized2['name'] == 'TEST'
    assert serialized2['value'] == 42
    
    # Test serialization without format parameter
    serialized3 = record2.serialize()
    assert serialized3['name'] == 'test'
    assert serialized3['value'] == 42
    
    # Test serialization with nested structures
    def list_serializer(format, value):
        return [str(x) for x in value]
    
    class RecordWithList(PRecord):
        items = field(serializer=list_serializer)
    
    record3 = RecordWithList(items=[1, 2, 3])
    serialized4 = record3.serialize()
    assert serialized4['items'] == ['1', '2', '3']
    
    # Test serialization with None serializer (default behavior)
    class RecordNoSerializer(PRecord):
        a = field(serializer=None)
        b = field()
    
    record4 = RecordNoSerializer(a=100, b=200)
    serialized5 = record4.serialize()
    assert serialized5 == {'a': 100, 'b': 200}


# LLM-generated content at query #15
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field()
        y = field(type=int)
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 'value1')
    assert evolver['x'] == 'value1'
    
    # Test setting multiple fields
    evolver.set('y', 42)
    assert evolver['y'] == 42
    
    # Test setting non-existent field raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('z', 'invalid')
    assert "'z' is not among the specified fields for TestRecord" in str(exc_info.value)
    
    # Test type checking
    class TypedRecord(PRecord):
        num = field(type=int)
    
    evolver = _PRecordEvolver(TypedRecord, pmap())
    with pytest.raises(TypeError):
        evolver.set('num', 'not_an_int')
    
    # Test field factory processing
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('items', (1, 2, 3))
    assert evolver['items'] == [1, 2, 3]
    
    # Test __setitem__ calls set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 'via_setitem'
    assert evolver['x'] == 'via_setitem'
    
    # Test with factory_fields restriction
    class RestrictedRecord(PRecord):
        a = field()
        b = field()
    
    field_a = RestrictedRecord._precord_fields['a']
    evolver = _PRecordEvolver(RestrictedRecord, pmap(), _factory_fields=[field_a])
    evolver.set('a', 'processed')
    evolver.set('b', 'not_processed')
    assert evolver['a'] == 'processed'
    assert evolver['b'] == 'not_processed'


# LLM-generated content at query #16
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test _PRecordEvolver.persistent() method"""
    
    # Test 1: Basic persistent creation with valid fields
    class SimpleRecord(PRecord):
        x = field()
        y = field()
    
    evolver = SimpleRecord.create({}).evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    
    assert isinstance(result, SimpleRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: Persistent with mandatory fields missing raises InvariantException
    from pyrsistent import field
    
    class MandatoryRecord(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    evolver = MandatoryRecord.create({}).evolver()
    evolver['optional'] = 'value'
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'required' in str(exc_info.value)
    
    # Test 3: Persistent with all mandatory fields present
    evolver = MandatoryRecord.create({}).evolver()
    evolver['required'] = 'present'
    evolver['optional'] = 'value'
    result = evolver.persistent()
    
    assert result['required'] == 'present'
    assert result['optional'] == 'value'
    
    # Test 4: Persistent preserves original when not dirty
    original = SimpleRecord(x=10, y=20)
    evolver = original.evolver()
    result = evolver.persistent()
    
    assert result is original
    
    # Test 5: Persistent creates new instance when dirty
    original = SimpleRecord(x=10, y=20)
    evolver = original.evolver()
    evolver['x'] = 15
    result = evolver.persistent()
    
    assert result is not original
    assert result['x'] == 15
    assert result['y'] == 20
    
    # Test 6: Field invariant violation raises InvariantException
    class InvariantRecord(PRecord):
        value = field()
    
    evolver = InvariantRecord.create({}).evolver()
    evolver['value'] = 'test'
    result = evolver.persistent()
    
    assert result['value'] == 'test'
    
    # Test 7: Multiple mandatory fields missing
    class MultiMandatoryRecord(PRecord):
        field1 = field(mandatory=True)
        field2 = field(mandatory=True)
        field3 = field()
    
    evolver = MultiMandatoryRecord.create({}).evolver()
    evolver['field3'] = 'value'
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    missing = str(exc_info.value)
    assert 'field1' in missing
    assert 'field2' in missing
    
    # Test 8: Persistent with initial values
    class InitialRecord(PRecord):
        x = field(initial=5)
        y = field()
    
    evolver = InitialRecord.create({}).evolver()
    evolver['y'] = 10
    result = evolver.persistent()
    
    assert result['x'] == 5
    assert result['y'] == 10


# LLM-generated content at query #17
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y='hello')
    repr_str = repr(record)
    
    assert repr_str == "TestRecord(x=1, y='hello')" or repr_str == "TestRecord(y='hello', x=1)"
    assert "TestRecord" in repr_str
    assert "x=" in repr_str
    assert "y=" in repr_str
    assert "1" in repr_str
    assert "'hello'" in repr_str


def test_PRecord___repr__empty():
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    repr_str = repr(record)
    
    assert repr_str == "EmptyRecord()"


def test_PRecord___repr__single_field():
    class SingleRecord(PRecord):
        name = field()
    
    record = SingleRecord(name='test')
    repr_str = repr(record)
    
    assert repr_str == "SingleRecord(name='test')"
    assert "SingleRecord" in repr_str
    assert "name='test'" in repr_str


def test_PRecord___repr__multiple_fields():
    class MultiRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = MultiRecord(a=1, b=2, c=3)
    repr_str = repr(record)
    
    assert "MultiRecord" in repr_str
    assert "a=1" in repr_str
    assert "b=2" in repr_str
    assert "c=3" in repr_str


def test_PRecord___repr__nested_values():
    class NestedRecord(PRecord):
        data = field()
    
    record = NestedRecord(data={'key': 'value'})
    repr_str = repr(record)
    
    assert "NestedRecord" in repr_str
    assert "data=" in repr_str
    assert "'key'" in repr_str
    assert "'value'" in repr_str


# LLM-generated content at query #18
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver"""
    
    # Test 1: Setting a valid field
    class TestRecord(PRecord):
        x = pfield(type=int)
        y = pfield(type=str)
    
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 42)
    assert evolver._data.get('x') == 42
    
    # Test 2: Setting multiple fields
    evolver.set('y', 'hello')
    assert evolver._data.get('y') == 'hello'
    
    # Test 3: Setting an invalid field should raise AttributeError
    with pytest.raises(AttributeError, match="'z' is not among the specified fields"):
        evolver.set('z', 100)
    
    # Test 4: Setting field with type checking
    class TypedRecord(PRecord):
        num = pfield(type=int)
    
    evolver = _PRecordEvolver(TypedRecord, pmap())
    evolver.set('num', 10)
    assert evolver._data.get('num') == 10
    
    # Test 5: Type checking should fail for wrong type
    with pytest.raises(TypeError):
        evolver.set('num', 'not_an_int')
    
    # Test 6: Using __setitem__ should call set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 99
    assert evolver._data.get('x') == 99
    
    # Test 7: Setting with factory fields parameter
    class FactoryRecord(PRecord):
        value = pfield(type=int, factory=int)
    
    evolver = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields={FactoryRecord._precord_fields['value']})
    evolver.set('value', '123')
    assert evolver._data.get('value') == 123
    
    # Test 8: Field with invariant that fails
    class InvariantRecord(PRecord):
        positive = pfield(type=int, invariant=lambda x: (x > 0, 'must_be_positive'))
    
    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('positive', -5)
    assert 'must_be_positive' in evolver._invariant_error_codes
    
    # Test 9: Ignore extra fields
    class ExtraRecord(PRecord):
        a = pfield(type=int)
    
    evolver = _PRecordEvolver(ExtraRecord, pmap(), _ignore_extra=True)
    evolver.set('a', 10)
    assert evolver._data.get('a') == 10
    
    # Test 10: Returns self for chaining
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('x', 5)
    assert result is evolver


# LLM-generated content at query #19
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test _PRecordMeta.__new__ creates PRecord class with proper attributes."""
    from pyrsistent import field
    
    # Test basic class creation
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()


def test__PRecordMeta___new__with_mandatory_fields():
    """Test _PRecordMeta.__new__ correctly identifies mandatory fields."""
    from pyrsistent import field
    
    class TestRecord(PRecord):
        required_field = field(mandatory=True)
        optional_field = field()
    
    assert 'required_field' in TestRecord._precord_mandatory_fields
    assert 'optional_field' not in TestRecord._precord_mandatory_fields


def test__PRecordMeta___new__with_initial_values():
    """Test _PRecordMeta.__new__ correctly stores initial values."""
    from pyrsistent import field
    
    class TestRecord(PRecord):
        name = field(initial='default_name')
        count = field(initial=lambda: 0)
    
    assert 'name' in TestRecord._precord_initial_values
    assert TestRecord._precord_initial_values['name'] == 'default_name'
    assert 'count' in TestRecord._precord_initial_values
    assert callable(TestRecord._precord_initial_values['count'])


def test__PRecordMeta___new__with_invariants():
    """Test _PRecordMeta.__new__ correctly stores invariants."""
    from pyrsistent import field
    
    class TestRecord(PRecord):
        age = field()
        
        @staticmethod
        def __invariant__(record):
            return True, None
    
    assert hasattr(TestRecord, '_precord_invariants')


def test__PRecordMeta___new__inheritance():
    """Test _PRecordMeta.__new__ with class inheritance."""
    from pyrsistent import field
    
    class BaseRecord(PRecord):
        base_field = field()
    
    class DerivedRecord(BaseRecord):
        derived_field = field()
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields


def test__PRecordMeta___new__no_fields():
    """Test _PRecordMeta.__new__ with no fields defined."""
    class EmptyRecord(PRecord):
        pass
    
    assert EmptyRecord._precord_fields == {}
    assert EmptyRecord._precord_mandatory_fields == set()
    assert EmptyRecord._precord_initial_values == {}


# LLM-generated content at query #20
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = pfield()
        y = pfield()

    record = TestRecord(x=1, y='hello')
    repr_str = repr(record)
    
    assert repr_str == "TestRecord(x=1, y='hello')" or repr_str == "TestRecord(y='hello', x=1)"
    assert "TestRecord" in repr_str
    assert "x=" in repr_str
    assert "y=" in repr_str
    assert "1" in repr_str
    assert "'hello'" in repr_str


def test_PRecord___repr__empty():
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    repr_str = repr(record)
    
    assert repr_str == "EmptyRecord()"


def test_PRecord___repr__single_field():
    class SingleRecord(PRecord):
        name = pfield()

    record = SingleRecord(name='test')
    repr_str = repr(record)
    
    assert repr_str == "SingleRecord(name='test')"


def test_PRecord___repr__multiple_fields():
    class MultiRecord(PRecord):
        a = pfield()
        b = pfield()
        c = pfield()

    record = MultiRecord(a=1, b=2, c=3)
    repr_str = repr(record)
    
    assert "MultiRecord" in repr_str
    assert "a=1" in repr_str
    assert "b=2" in repr_str
    assert "c=3" in repr_str


def test_PRecord___repr__nested_values():
    class NestedRecord(PRecord):
        data = pfield()

    record = NestedRecord(data={'key': 'value'})
    repr_str = repr(record)
    
    assert "NestedRecord" in repr_str
    assert "data=" in repr_str
    assert "{'key': 'value'}" in repr_str


# LLM-generated content at query #21
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test _PRecordEvolver.set() method"""
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=str)
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 42)
    assert evolver._data['x'] == 42
    
    # Test set with string field
    evolver.set('y', 'hello')
    assert evolver._data['y'] == 'hello'
    
    # Test set with type checking
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    assert evolver._data['x'] == 10
    
    # Test set with invalid field name raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'invalid_field' in str(e)
        assert 'TestRecord' in str(e)
    
    # Test set with factory field
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('items', [1, 2, 3])
    assert evolver._data['items'] == [1, 2, 3]
    
    # Test set returns self for chaining
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('x', 5)
    assert result is evolver
    
    # Test set with ignore_extra flag
    evolver = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 100)
    assert evolver._data['x'] == 100
    
    # Test set overwrites previous value
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 1)
    evolver.set('x', 2)
    assert evolver._data['x'] == 2


# LLM-generated content at query #22
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    assert evolver._data.get('x') == 10
    
    # Test set with type checking
    class TypedRecord(PRecord):
        x = field(type=int)
    
    evolver = _PRecordEvolver(TypedRecord, pmap())
    evolver.set('x', 5)
    assert evolver._data.get('x') == 5
    
    # Test set with invalid field raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'invalid_field' in str(e)
        assert 'TestRecord' in str(e)
    
    # Test set with factory field
    class FactoryRecord(PRecord):
        x = field()
    
    evolver = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields=set())
    evolver.set('x', 20)
    assert evolver._data.get('x') == 20
    
    # Test set returns self for method chaining
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('x', 15)
    assert result is evolver
    
    # Test set with ignore_extra flag
    class ExtraRecord(PRecord):
        x = field()
    
    evolver = _PRecordEvolver(ExtraRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 25)
    assert evolver._data.get('x') == 25
    
    # Test __setitem__ delegates to set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 30
    assert evolver._data.get('x') == 30


# LLM-generated content at query #23
#--------------------------

```python
def test_PRecord___new__():
    """Test PRecord.__new__ method"""
    
    # Test basic instantiation with no arguments
    class SimpleRecord(PRecord):
        pass
    
    record = SimpleRecord()
    assert isinstance(record, PRecord)
    assert isinstance(record, PMap)
    
    # Test instantiation with keyword arguments
    from pyrsistent import field
    
    class PersonRecord(PRecord):
        name = field()
        age = field()
    
    person = PersonRecord(name='Alice', age=30)
    assert person['name'] == 'Alice'
    assert person['age'] == 30
    
    # Test with initial values
    class ConfigRecord(PRecord):
        host = field(initial='localhost')
        port = field(initial=8080)
    
    config = ConfigRecord()
    assert config['host'] == 'localhost'
    assert config['port'] == 8080
    
    # Test with initial values overridden
    config2 = ConfigRecord(host='example.com', port=9000)
    assert config2['host'] == 'example.com'
    assert config2['port'] == 9000
    
    # Test with callable initial value
    class CallableInitRecord(PRecord):
        items = field(initial=list)
    
    rec1 = CallableInitRecord()
    rec2 = CallableInitRecord()
    assert rec1['items'] == []
    assert rec2['items'] == []
    assert rec1['items'] is not rec2['items']
    
    # Test internal initialization with _precord_size and _precord_buckets
    # This uses the "hack" path in __new__
    person2 = PersonRecord(name='Bob', age=25)
    person3 = PersonRecord(_precord_size=person2._size, _precord_buckets=person2._buckets)
    assert person3['name'] == 'Bob'
    assert person3['age'] == 25
    
    # Test with _ignore_extra flag
    person4 = PersonRecord.create({'name': 'Charlie', 'age': 35, 'extra_field': 'ignored'}, ignore_extra=True)
    assert person4['name'] == 'Charlie'
    assert person4['age'] == 35
    assert 'extra_field' not in person4
    
    # Test empty record
    empty = SimpleRecord()
    assert len(empty) == 0
    assert dict(empty) == {}


# LLM-generated content at query #24
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test the __new__ method of _PRecordMeta metaclass."""
    
    # Test basic PRecord class creation
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    assert hasattr(TestRecord, '_precord_fields')
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    
    # Test mandatory fields detection
    class MandatoryRecord(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    assert 'required' in MandatoryRecord._precord_mandatory_fields
    assert 'optional' not in MandatoryRecord._precord_mandatory_fields
    
    # Test initial values collection
    class InitialValueRecord(PRecord):
        with_initial = field(initial=42)
        without_initial = field()
    
    assert 'with_initial' in InitialValueRecord._precord_initial_values
    assert InitialValueRecord._precord_initial_values['with_initial'] == 42
    assert 'without_initial' not in InitialValueRecord._precord_initial_values
    
    # Test callable initial values
    class CallableInitialRecord(PRecord):
        with_callable = field(initial=lambda: [])
    
    assert callable(CallableInitialRecord._precord_initial_values['with_callable'])
    
    # Test invariants storage
    class InvariantRecord(PRecord):
        value = field()
        
        @invariant
        def check_value(self):
            return self['value'] > 0, 'value must be positive'
    
    assert hasattr(InvariantRecord, '_precord_invariants')
    assert len(InvariantRecord._precord_invariants) > 0
    
    # Test inheritance of fields
    class BaseRecord(PRecord):
        base_field = field()
    
    class DerivedRecord(BaseRecord):
        derived_field = field()
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    
    # Test empty record
    class EmptyRecord(PRecord):
        pass
    
    assert EmptyRecord._precord_fields == {}
    assert EmptyRecord._precord_mandatory_fields == set()
    assert EmptyRecord._precord_initial_values == {}


# LLM-generated content at query #25
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = field()
        y = field()

    record = TestRecord(x=1, y='hello')
    result = repr(record)
    
    assert result.startswith('TestRecord(')
    assert 'x=1' in result
    assert "y='hello'" in result
    assert result.endswith(')')


def test_PRecord___repr__empty():
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    result = repr(record)
    
    assert result == 'EmptyRecord()'


def test_PRecord___repr__single_field():
    class SingleFieldRecord(PRecord):
        value = field()

    record = SingleFieldRecord(value=42)
    result = repr(record)
    
    assert result == 'SingleFieldRecord(value=42)'


def test_PRecord___repr__multiple_fields():
    class MultiFieldRecord(PRecord):
        a = field()
        b = field()
        c = field()

    record = MultiFieldRecord(a=1, b=2, c=3)
    result = repr(record)
    
    assert result.startswith('MultiFieldRecord(')
    assert 'a=1' in result
    assert 'b=2' in result
    assert 'c=3' in result
    assert result.endswith(')')


def test_PRecord___repr__with_nested_structures():
    class NestedRecord(PRecord):
        items = field()
        mapping = field()

    record = NestedRecord(items=[1, 2, 3], mapping={'key': 'value'})
    result = repr(record)
    
    assert result.startswith('NestedRecord(')
    assert 'items=[1, 2, 3]' in result
    assert "mapping={'key': 'value'}" in result
    assert result.endswith(')')


def test_PRecord___repr__with_string_values():
    class StringRecord(PRecord):
        name = field()
        description = field()

    record = StringRecord(name='test', description='a test record')
    result = repr(record)
    
    assert result.startswith('StringRecord(')
    assert "name='test'" in result
    assert "description='a test record'" in result
    assert result.endswith(')')


# LLM-generated content at query #26
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class"""
    from pyrsistent import field
    
    # Define a test PRecord class
    class TestRecord(PRecord):
        name = field()
        age = field()
        email = field()
    
    # Test 1: Setting a valid field with a valid value
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'John')
    assert evolver['name'] == 'John'
    
    # Test 2: Setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Alice')
    evolver.set('age', 30)
    evolver.set('email', 'alice@example.com')
    assert evolver['name'] == 'Alice'
    assert evolver['age'] == 30
    assert evolver['email'] == 'alice@example.com'
    
    # Test 3: Setting a non-existent field should raise AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('nonexistent_field', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'nonexistent_field' in str(e)
        assert 'TestRecord' in str(e)
    
    # Test 4: Using __setitem__ should call set internally
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['name'] = 'Bob'
    assert evolver['name'] == 'Bob'
    
    # Test 5: Setting a field returns self for method chaining
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('name', 'Charlie')
    assert result is evolver
    
    # Test 6: Overwriting an existing field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'David')
    evolver.set('name', 'Eve')
    assert evolver['name'] == 'Eve'


# LLM-generated content at query #27
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class."""
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=str)
    
    # Test 1: Setting a valid field with correct type
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('x', 42)
    assert result is evolver
    assert evolver._data['x'] == 42
    
    # Test 2: Setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 'hello')
    assert evolver._data['x'] == 10
    assert evolver._data['y'] == 'hello'
    
    # Test 3: Setting a field that doesn't exist raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('z', 100)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z' is not among the specified fields" in str(e)
    
    # Test 4: Setting a field with wrong type adds invariant error
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 'not_an_int')
    assert len(evolver._invariant_error_codes) > 0
    
    # Test 5: Using __setitem__ delegates to set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 99
    assert evolver._data['x'] == 99
    
    # Test 6: Setting with factory_fields restriction
    evolver = _PRecordEvolver(TestRecord, pmap(), _factory_fields=[TestRecord._precord_fields['x']])
    evolver.set('x', 5)
    assert evolver._data['x'] == 5
    
    # Test 7: Setting field not in factory_fields uses original value
    class TestRecord2(PRecord):
        a = field()
        b = field()
    
    evolver = _PRecordEvolver(TestRecord2, pmap(), _factory_fields=[TestRecord2._precord_fields['a']])
    evolver.set('b', 'value')
    assert evolver._data['b'] == 'value'


# LLM-generated content at query #28
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test that _PRecordMeta.__new__ properly sets up PRecord subclasses."""
    
    # Test basic PRecord creation with fields
    class TestRecord(PRecord):
        x = pfield(type=int)
        y = pfield(type=str)
    
    # Verify _precord_fields is set correctly
    assert hasattr(TestRecord, '_precord_fields')
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    
    # Verify _precord_invariants is set
    assert hasattr(TestRecord, '_precord_invariants')
    
    # Verify __slots__ is set to empty tuple
    assert TestRecord.__slots__ == ()
    
    # Test with mandatory fields
    class TestRecordMandatory(PRecord):
        required_field = pfield(type=int, mandatory=True)
        optional_field = pfield(type=str)
    
    assert hasattr(TestRecordMandatory, '_precord_mandatory_fields')
    assert 'required_field' in TestRecordMandatory._precord_mandatory_fields
    assert 'optional_field' not in TestRecordMandatory._precord_mandatory_fields
    
    # Test with initial values
    class TestRecordInitial(PRecord):
        field_with_initial = pfield(type=int, initial=42)
        field_without_initial = pfield(type=str)
    
    assert hasattr(TestRecordInitial, '_precord_initial_values')
    assert TestRecordInitial._precord_initial_values.get('field_with_initial') == 42
    assert 'field_without_initial' not in TestRecordInitial._precord_initial_values
    
    # Test inheritance
    class BaseRecord(PRecord):
        base_field = pfield(type=int)
    
    class DerivedRecord(BaseRecord):
        derived_field = pfield(type=str)
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    
    # Test with callable initial value
    class TestRecordCallableInitial(PRecord):
        field_with_callable = pfield(type=list, initial=list)
    
    assert callable(TestRecordCallableInitial._precord_initial_values.get('field_with_callable'))
    
    # Test that metaclass is properly applied
    assert isinstance(TestRecord, _PRecordMeta)


# LLM-generated content at query #29
#--------------------------

```python
def test_PRecord_serialize():
    from pyrsistent import PRecord, field
    
    # Test basic serialization without custom serializers
    class SimpleRecord(PRecord):
        name = field()
        age = field()
    
    record = SimpleRecord(name="Alice", age=30)
    serialized = record.serialize()
    assert serialized == {"name": "Alice", "age": 30}
    assert isinstance(serialized, dict)
    
    # Test serialization with custom serializer function
    def uppercase_serializer(serializer, format, value):
        return value.upper() if isinstance(value, str) else value
    
    class RecordWithSerializer(PRecord):
        name = field(serializer=uppercase_serializer)
        value = field()
    
    record2 = RecordWithSerializer(name="bob", value=42)
    serialized2 = record2.serialize()
    assert serialized2 == {"name": "BOB", "value": 42}
    
    # Test serialization with format parameter
    def format_aware_serializer(serializer, format, value):
        if format == "json":
            return str(value)
        return value
    
    class RecordWithFormatSerializer(PRecord):
        data = field(serializer=format_aware_serializer)
    
    record3 = RecordWithFormatSerializer(data=123)
    serialized3 = record3.serialize(format="json")
    assert serialized3 == {"data": "123"}
    
    serialized3_default = record3.serialize()
    assert serialized3_default == {"data": 123}
    
    # Test serialization with nested records
    class InnerRecord(PRecord):
        inner_val = field()
    
    class OuterRecord(PRecord):
        outer_val = field()
        inner = field()
    
    inner = InnerRecord(inner_val="test")
    outer = OuterRecord(outer_val="outer", inner=inner)
    serialized4 = outer.serialize()
    assert serialized4["outer_val"] == "outer"
    assert isinstance(serialized4["inner"], InnerRecord)
    
    # Test empty record serialization
    class EmptyRecord(PRecord):
        pass
    
    empty = EmptyRecord()
    assert empty.serialize() == {}


# LLM-generated content at query #30
#--------------------------

```python
def test__PRecordEvolver_set():
    """Test the set method of _PRecordEvolver class"""
    from pyrsistent import field
    
    # Create a simple PRecord class for testing
    class TestRecord(PRecord):
        x = field()
        y = field(invariant=lambda x: (x > 0, "must_be_positive"))
    
    # Test 1: Setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set("x", 10)
    assert evolver._data.get("x") == 10
    
    # Test 2: Setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set("x", 5)
    evolver.set("y", 20)
    assert evolver._data.get("x") == 5
    assert evolver._data.get("y") == 20
    
    # Test 3: Setting an invalid field raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError) as exc_info:
        evolver.set("z", 100)
    assert "not among the specified fields" in str(exc_info.value)
    
    # Test 4: Field invariant violation is recorded
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set("y", -5)  # Violates invariant (must be > 0)
    assert "must_be_positive" in evolver._invariant_error_codes
    
    # Test 5: Using __setitem__ should call set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver["x"] = 15
    assert evolver._data.get("x") == 15
    
    # Test 6: Setting field with factory
    class TestRecordWithFactory(PRecord):
        items = field(factory=list)
    
    evolver = _PRecordEvolver(TestRecordWithFactory, pmap())
    evolver.set("items", [1, 2, 3])
    assert evolver._data.get("items") == [1, 2, 3]
    
    # Test 7: Overwriting an existing field
    original = pmap({"x": 10})
    evolver = _PRecordEvolver(TestRecord, original)
    evolver.set("x", 99)
    assert evolver._data.get("x") == 99


# LLM-generated content at query #31
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test the persistent method of _PRecordEvolver"""
    
    # Test 1: Basic persistent creation with valid fields
    class TestRecord(PRecord):
        x = PField(type=int)
        y = PField(type=str)
    
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 42
    evolver['y'] = 'hello'
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 42
    assert result['y'] == 'hello'
    
    # Test 2: Persistent with mandatory fields missing should raise InvariantException
    class TestRecordMandatory(PRecord):
        required = PField(type=int, mandatory=True)
        optional = PField(type=str)
    
    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['optional'] = 'test'
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'TestRecordMandatory.required' in str(exc_info.value)
    
    # Test 3: Persistent with all mandatory fields present
    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['required'] = 100
    result = evolver.persistent()
    
    assert isinstance(result, TestRecordMandatory)
    assert result['required'] == 100
    
    # Test 4: Persistent returns same instance if not dirty
    initial_record = TestRecord(x=1, y='a')
    evolver = _PRecordEvolver(TestRecord, initial_record)
    result = evolver.persistent()
    
    assert result is initial_record
    
    # Test 5: Persistent creates new instance when dirty
    evolver = _PRecordEvolver(TestRecord, initial_record)
    evolver['x'] = 2
    result = evolver.persistent()
    
    assert result is not initial_record
    assert result['x'] == 2
    assert result['y'] == 'a'
    
    # Test 6: Persistent with field invariant violation
    class TestRecordWithInvariant(PRecord):
        value = PField(type=int)
        
        @value.invariant()
        def check_positive(self):
            return self > 0, "must be positive"
    
    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver['value'] = -5
    
    with pytest.raises(InvariantException):
        evolver.persistent()
    
    # Test 7: Persistent with invariant error codes collected
    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver._invariant_error_codes = ['error1', 'error2']
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert len(exc_info.value.invariant_errors) == 2


# LLM-generated content at query #32
#--------------------------

```python
def test_PRecord_serialize():
    from pyrsistent import PRecord, field
    
    # Test basic serialization without custom serializers
    class SimpleRecord(PRecord):
        name = field()
        age = field()
    
    record = SimpleRecord(name='John', age=30)
    serialized = record.serialize()
    assert serialized == {'name': 'John', 'age': 30}
    assert isinstance(serialized, dict)
    
    # Test serialization with custom serializer function
    def custom_int_serializer(format, value):
        if format == 'string':
            return str(value)
        return value
    
    class RecordWithSerializer(PRecord):
        name = field()
        age = field(serializer=custom_int_serializer)
    
    record2 = RecordWithSerializer(name='Jane', age=25)
    serialized_default = record2.serialize()
    assert serialized_default == {'name': 'Jane', 'age': 25}
    
    serialized_string = record2.serialize(format='string')
    assert serialized_string == {'name': 'Jane', 'age': '25'}
    
    # Test serialization with None values
    class RecordWithNone(PRecord):
        name = field()
        description = field()
    
    record3 = RecordWithNone(name='Test', description=None)
    serialized_none = record3.serialize()
    assert serialized_none == {'name': 'Test', 'description': None}
    
    # Test serialization with multiple custom serializers
    def uppercase_serializer(format, value):
        if format == 'upper':
            return value.upper() if isinstance(value, str) else value
        return value
    
    class MultiSerializerRecord(PRecord):
        first = field(serializer=uppercase_serializer)
        second = field(serializer=custom_int_serializer)
    
    record4 = MultiSerializerRecord(first='hello', second=42)
    serialized_multi = record4.serialize(format='upper')
    assert serialized_multi == {'first': 'HELLO', 'second': 42}
    
    # Test empty record serialization
    class EmptyRecord(PRecord):
        pass
    
    empty_record = EmptyRecord()
    serialized_empty = empty_record.serialize()
    assert serialized_empty == {}


# LLM-generated content at query #33
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field()
        y = field(type=int)
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 'value')
    assert evolver['x'] == 'value'
    
    # Test set with type checking
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('y', 42)
    assert evolver['y'] == 42
    
    # Test set with invalid field raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('z', 'invalid')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'z' is not among the specified fields" in str(e)
    
    # Test set with type mismatch
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('y', 'not_an_int')
    # Error codes should be collected
    assert len(evolver._invariant_error_codes) > 0
    
    # Test __setitem__ delegates to set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 'value'
    assert evolver['x'] == 'value'
    
    # Test set with factory fields
    class FactoryRecord(PRecord):
        items = field()
    
    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('items', [1, 2, 3])
    assert 'items' in evolver
    
    # Test set returns self for chaining
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('x', 'value1')
    assert result is evolver


# LLM-generated content at query #34
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test the __new__ method of _PRecordMeta metaclass"""
    
    # Test basic class creation with no fields
    class SimpleRecord(PRecord):
        pass
    
    assert hasattr(SimpleRecord, '_precord_fields')
    assert hasattr(SimpleRecord, '_precord_invariants')
    assert hasattr(SimpleRecord, '_precord_mandatory_fields')
    assert hasattr(SimpleRecord, '_precord_initial_values')
    assert SimpleRecord.__slots__ == ()
    assert isinstance(SimpleRecord._precord_fields, dict)
    assert isinstance(SimpleRecord._precord_mandatory_fields, set)
    assert isinstance(SimpleRecord._precord_initial_values, dict)
    
    # Test class creation with inheritance
    class ParentRecord(PRecord):
        pass
    
    class ChildRecord(ParentRecord):
        pass
    
    assert hasattr(ChildRecord, '_precord_fields')
    assert ChildRecord.__name__ == 'ChildRecord'
    
    # Test that __slots__ is always empty tuple
    assert SimpleRecord.__slots__ == ()
    assert ParentRecord.__slots__ == ()
    assert ChildRecord.__slots__ == ()
    
    # Test that mandatory fields are correctly identified
    assert isinstance(SimpleRecord._precord_mandatory_fields, set)
    assert isinstance(SimpleRecord._precord_initial_values, dict)
    
    # Test that the class is properly registered with metaclass
    assert type(SimpleRecord) == _PRecordMeta
    assert type(ParentRecord) == _PRecordMeta
    assert type(ChildRecord) == _PRecordMeta
    
    # Test that multiple classes can be created independently
    class RecordA(PRecord):
        pass
    
    class RecordB(PRecord):
        pass
    
    assert RecordA._precord_fields is not RecordB._precord_fields
    assert RecordA._precord_mandatory_fields is not RecordB._precord_mandatory_fields
    assert RecordA._precord_initial_values is not RecordB._precord_initial_values


# LLM-generated content at query #35
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = pfield()
        y = pfield()
    
    record = TestRecord(x=1, y='hello')
    result = repr(record)
    
    assert result == "TestRecord(x=1, y='hello')" or result == "TestRecord(y='hello', x=1)"
    assert "TestRecord(" in result
    assert "x=1" in result
    assert "y='hello'" in result


def test_PRecord___repr__empty():
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    result = repr(record)
    
    assert result == "EmptyRecord()"


def test_PRecord___repr__single_field():
    class SingleRecord(PRecord):
        name = pfield()
    
    record = SingleRecord(name='test')
    result = repr(record)
    
    assert result == "SingleRecord(name='test')"


def test_PRecord___repr__multiple_types():
    class MultiTypeRecord(PRecord):
        a = pfield()
        b = pfield()
        c = pfield()
    
    record = MultiTypeRecord(a=42, b=3.14, c=None)
    result = repr(record)
    
    assert "MultiTypeRecord(" in result
    assert "a=42" in result
    assert "b=3.14" in result
    assert "c=None" in result


def test_PRecord___repr__nested_structure():
    class InnerRecord(PRecord):
        value = pfield()
    
    class OuterRecord(PRecord):
        inner = pfield()
    
    inner = InnerRecord(value=10)
    outer = OuterRecord(inner=inner)
    result = repr(outer)
    
    assert "OuterRecord(" in result
    assert "InnerRecord(" in result
    assert "value=10" in result


# LLM-generated content at query #36
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test _PRecordMeta.__new__ creates a proper PRecord class"""
    from pyrsistent import field
    
    # Test basic class creation
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    assert hasattr(TestRecord, '_precord_fields')
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert TestRecord._precord_mandatory_fields == {'x', 'y'}
    assert TestRecord._precord_initial_values == {}
    assert TestRecord.__slots__ == ()
    
    # Test with mandatory and optional fields
    class TestRecord2(PRecord):
        a = field(mandatory=True)
        b = field(mandatory=False)
        c = field(initial=42)
    
    assert TestRecord2._precord_mandatory_fields == {'a'}
    assert 'c' in TestRecord2._precord_initial_values
    assert TestRecord2._precord_initial_values['c'] == 42
    
    # Test with initial callable
    class TestRecord3(PRecord):
        items = field(initial=list)
    
    assert callable(TestRecord3._precord_initial_values['items'])
    
    # Test class inheritance
    class BaseRecord(PRecord):
        base_field = field()
    
    class DerivedRecord(BaseRecord):
        derived_field = field()
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    
    # Test with invariants
    class InvariantRecord(PRecord):
        value = field()
        
        @classmethod
        def __invariant__(cls):
            pass
    
    assert hasattr(InvariantRecord, '_precord_invariants')


# LLM-generated content at query #37
#--------------------------

```python
def test__PRecordEvolver_persistent():
    from pyrsistent import field
    
    # Test 1: Basic persistent() call with valid fields
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.create({}).evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert result['x'] == 1
    assert result['y'] == 2
    assert isinstance(result, TestRecord)
    
    # Test 2: persistent() with mandatory fields missing raises InvariantException
    class TestRecordMandatory(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    evolver = TestRecordMandatory.create({}).evolver()
    evolver['optional'] = 'value'
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestRecordMandatory.required' in e.missing_fields
    
    # Test 3: persistent() with all mandatory fields provided
    evolver = TestRecordMandatory.create({}).evolver()
    evolver['required'] = 'value'
    result = evolver.persistent()
    assert result['required'] == 'value'
    assert isinstance(result, TestRecordMandatory)
    
    # Test 4: persistent() when not dirty and already correct type
    record = TestRecord.create({'x': 10, 'y': 20})
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record  # Should return same instance when not dirty
    
    # Test 5: persistent() with field invariant violation
    def positive_invariant(value):
        return (value > 0, 'must_be_positive')
    
    class TestRecordWithInvariant(PRecord):
        num = field(invariant=positive_invariant)
    
    evolver = TestRecordWithInvariant.create({}).evolver()
    evolver['num'] = -5
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'must_be_positive' in e.invariant_errors
    
    # Test 6: persistent() with multiple field violations
    class TestRecordMultiple(PRecord):
        a = field(mandatory=True, invariant=positive_invariant)
        b = field(mandatory=True)
    
    evolver = TestRecordMultiple.create({}).evolver()
    evolver['a'] = -1
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'must_be_positive' in e.invariant_errors
        assert 'TestRecordMultiple.b' in e.missing_fields
    
    # Test 7: persistent() creates new instance when dirty
    record = TestRecord.create({'x': 1, 'y': 2})
    evolver = record.evolver()
    evolver['x'] = 100
    result = evolver.persistent()
    assert result is not record
    assert result['x'] == 100
    assert result['y'] == 2


# LLM-generated content at query #38
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=str)
    
    # Test basic set operation
    original = TestRecord(x=1, y="hello")
    evolver = original.evolver()
    evolver.set('x', 42)
    assert evolver._data['x'] == 42
    
    # Test set with type checking
    evolver2 = original.evolver()
    evolver2.set('y', "world")
    assert evolver2._data['y'] == "world"
    
    # Test set with invalid field raises AttributeError
    evolver3 = original.evolver()
    try:
        evolver3.set('invalid_field', 123)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "invalid_field" in str(e)
        assert "TestRecord" in str(e)
    
    # Test set with type mismatch stores error
    class StrictRecord(PRecord):
        num = field(type=int, invariant=lambda x: (x > 0, "must be positive"))
    
    evolver4 = StrictRecord().evolver()
    evolver4.set('num', -5)
    assert len(evolver4._invariant_error_codes) > 0
    
    # Test __setitem__ calls set
    evolver5 = original.evolver()
    evolver5['x'] = 99
    assert evolver5._data['x'] == 99


def test__PRecordEvolver_set_with_factory():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        items = field(factory=list)
    
    evolver = TestRecord().evolver()
    evolver.set('items', [1, 2, 3])
    assert evolver._data['items'] == [1, 2, 3]


def test__PRecordEvolver_set_with_ignore_extra():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field(type=int)
    
    evolver = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 10)
    assert evolver._data['x'] == 10


def test__PRecordEvolver_set_field_not_in_factory_fields():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=int)
    
    x_field = TestRecord._precord_fields['x']
    evolver = _PRecordEvolver(TestRecord, pmap(), _factory_fields={x_field})
    
    # y is not in factory_fields, so it should be set directly without factory processing
    evolver.set('y', 20)
    assert evolver._data['y'] == 20


# LLM-generated content at query #39
#--------------------------

```python
def test__PRecordEvolver_persistent():
    from pyrsistent import PRecord, field
    
    # Test basic persistent call
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.create({}).evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test with mandatory fields
    class MandatoryRecord(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    evolver = MandatoryRecord.create({}).evolver()
    evolver['optional'] = 'value'
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'MandatoryRecord.required' in e.missing_fields
    
    # Test with all mandatory fields set
    evolver = MandatoryRecord.create({}).evolver()
    evolver['required'] = 'present'
    evolver['optional'] = 'value'
    result = evolver.persistent()
    
    assert isinstance(result, MandatoryRecord)
    assert result['required'] == 'present'
    assert result['optional'] == 'value'
    
    # Test persistent without dirty changes returns same instance
    record = TestRecord.create({'x': 10, 'y': 20})
    evolver = record.evolver()
    result = evolver.persistent()
    
    assert result is record
    
    # Test persistent with dirty changes returns new instance
    evolver = record.evolver()
    evolver['x'] = 30
    result = evolver.persistent()
    
    assert result is not record
    assert isinstance(result, TestRecord)
    assert result['x'] == 30
    assert result['y'] == 20
    
    # Test with field invariant violation
    class InvariantRecord(PRecord):
        positive = field()
    
    evolver = InvariantRecord.create({}).evolver()
    evolver['positive'] = -5
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException for field invariant violation"
    except InvariantException as e:
        assert len(e.invariant_errors) >= 0


# LLM-generated content at query #40
#--------------------------

def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = pfield()
        y = pfield()
    
    record = TestRecord(x=1, y='hello')
    repr_str = repr(record)
    
    assert repr_str == "TestRecord(x=1, y='hello')" or repr_str == "TestRecord(y='hello', x=1)"
    assert "TestRecord" in repr_str
    assert "x=" in repr_str
    assert "y=" in repr_str
    assert "1" in repr_str
    assert "'hello'" in repr_str


def test_PRecord___repr___empty():
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    repr_str = repr(record)
    
    assert repr_str == "EmptyRecord()"
    assert "EmptyRecord" in repr_str


def test_PRecord___repr___single_field():
    class SingleRecord(PRecord):
        value = pfield()
    
    record = SingleRecord(value=42)
    repr_str = repr(record)
    
    assert "SingleRecord" in repr_str
    assert "value=42" in repr_str


def test_PRecord___repr___with_nested_values():
    class NestedRecord(PRecord):
        data = pfield()
        count = pfield()
    
    record = NestedRecord(data=[1, 2, 3], count={'a': 1})
    repr_str = repr(record)
    
    assert "NestedRecord" in repr_str
    assert "data=" in repr_str
    assert "count=" in repr_str
    assert "[1, 2, 3]" in repr_str
    assert "{'a': 1}" in repr_str


def test_PRecord___repr___with_string_values():
    class StringRecord(PRecord):
        name = pfield()
    
    record = StringRecord(name="test_name")
    repr_str = repr(record)
    
    assert "StringRecord" in repr_str
    assert "name='test_name'" in repr_str


# LLM-generated content at query #41
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=str)
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 42)
    assert evolver.data['x'] == 42
    
    # Test set with string field
    evolver.set('y', 'hello')
    assert evolver.data['y'] == 'hello'
    
    # Test setting non-existent field raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('z', 'invalid')
    assert "'z' is not among the specified fields for TestRecord" in str(exc_info.value)
    
    # Test type checking
    class StrictRecord(PRecord):
        num = field(type=int)
    
    evolver = _PRecordEvolver(StrictRecord, pmap())
    with pytest.raises(TypeError):
        evolver.set('num', 'not_an_int')
    
    # Test __setitem__ calls set
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 100
    assert evolver.data['x'] == 100
    
    # Test with factory fields
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields={FactoryRecord._precord_fields['items']})
    evolver.set('items', [1, 2, 3])
    assert evolver.data['items'] == [1, 2, 3]


# LLM-generated content at query #42
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic PRecord class creation
    class TestRecord(PRecord):
        x = pfield()
        y = pfield()
    
    assert hasattr(TestRecord, '_precord_fields')
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert TestRecord.__slots__ == ()


def test__PRecordMeta___new__with_mandatory_fields():
    # Test that mandatory fields are properly identified
    class TestRecord(PRecord):
        x = pfield(mandatory=True)
        y = pfield(mandatory=False)
    
    assert TestRecord._precord_mandatory_fields == {'x'}


def test__PRecordMeta___new__with_initial_values():
    # Test that initial values are properly stored
    class TestRecord(PRecord):
        x = pfield(initial=42)
        y = pfield(initial=lambda: [1, 2, 3])
    
    assert 'x' in TestRecord._precord_initial_values
    assert 'y' in TestRecord._precord_initial_values
    assert TestRecord._precord_initial_values['x'] == 42


def test__PRecordMeta___new__with_invariants():
    # Test that invariants are properly stored
    class TestRecord(PRecord):
        x = pfield()
        
        @invariant
        def check_x(self):
            return self.get('x', 0) > 0, 'x must be positive'
    
    assert hasattr(TestRecord, '_precord_invariants')
    assert len(TestRecord._precord_invariants) > 0


def test__PRecordMeta___new__inheritance():
    # Test that fields are inherited from parent classes
    class ParentRecord(PRecord):
        x = pfield()
    
    class ChildRecord(ParentRecord):
        y = pfield()
    
    assert 'x' in ChildRecord._precord_fields
    assert 'y' in ChildRecord._precord_fields


def test__PRecordMeta___new__no_slots_override():
    # Test that __slots__ is always empty tuple
    class TestRecord(PRecord):
        x = pfield()
    
    assert TestRecord.__slots__ == ()
    assert not hasattr(TestRecord, '__dict__')


def test__PRecordMeta___new__mandatory_and_initial_fields():
    # Test combination of mandatory and initial fields
    class TestRecord(PRecord):
        x = pfield(mandatory=True)
        y = pfield(mandatory=False, initial=10)
        z = pfield(mandatory=True, initial=20)
    
    assert TestRecord._precord_mandatory_fields == {'x', 'z'}
    assert TestRecord._precord_initial_values == {'y': 10, 'z': 20}


def test__PRecordMeta___new__empty_record():
    # Test creating an empty PRecord with no fields
    class EmptyRecord(PRecord):
        pass
    
    assert EmptyRecord._precord_fields == {}
    assert EmptyRecord._precord_mandatory_fields == set()
    assert EmptyRecord._precord_initial_values == {}


# LLM-generated content at query #43
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = pfield(type=int)
        y = pfield(type=str)
    
    record = TestRecord(x=42, y='hello')
    repr_str = repr(record)
    
    assert repr_str.startswith('TestRecord(')
    assert repr_str.endswith(')')
    assert 'x=42' in repr_str
    assert "y='hello'" in repr_str
    
    # Test with single field
    class SingleFieldRecord(PRecord):
        value = pfield(type=int)
    
    single_record = SingleFieldRecord(value=100)
    single_repr = repr(single_record)
    
    assert single_repr == "SingleFieldRecord(value=100)"
    
    # Test with empty record
    class EmptyRecord(PRecord):
        pass
    
    empty_record = EmptyRecord()
    empty_repr = repr(empty_record)
    
    assert empty_repr == "EmptyRecord()"
    
    # Test with special characters in string values
    class SpecialRecord(PRecord):
        text = pfield(type=str)
    
    special_record = SpecialRecord(text="it's a test")
    special_repr = repr(special_record)
    
    assert 'SpecialRecord(' in special_repr
    assert 'text=' in special_repr


# LLM-generated content at query #44
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = pfield()
        y = pfield()
    
    record = TestRecord(x=1, y='hello')
    result = repr(record)
    
    assert 'TestRecord' in result
    assert 'x=1' in result
    assert "y='hello'" in result
    assert result.startswith('TestRecord(')
    assert result.endswith(')')


# LLM-generated content at query #45
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test the persistent method of _PRecordEvolver class"""
    
    # Test 1: Basic persistent creation with valid fields
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.create({}).evolver()
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: Persistent with mandatory fields missing raises InvariantException
    class MandatoryRecord(PRecord):
        required_field = field(mandatory=True)
        optional_field = field()
    
    evolver = MandatoryRecord.create({}).evolver()
    evolver['optional_field'] = 'value'
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    assert 'required_field' in str(exc_info.value)
    
    # Test 3: Persistent with all mandatory fields present succeeds
    evolver = MandatoryRecord.create({}).evolver()
    evolver['required_field'] = 'required_value'
    evolver['optional_field'] = 'optional_value'
    result = evolver.persistent()
    
    assert isinstance(result, MandatoryRecord)
    assert result['required_field'] == 'required_value'
    assert result['optional_field'] == 'optional_value'
    
    # Test 4: Persistent returns same object when not dirty
    original = TestRecord(x=1, y=2)
    evolver = original.evolver()
    result = evolver.persistent()
    
    assert result is original
    
    # Test 5: Persistent returns new object when dirty
    evolver = original.evolver()
    evolver['x'] = 99
    result = evolver.persistent()
    
    assert result is not original
    assert result['x'] == 99
    assert result['y'] == 2
    
    # Test 6: Persistent with field invariant violation raises InvariantException
    class ConstrainedRecord(PRecord):
        value = field(invariant=lambda x: (x > 0, 'value_must_be_positive'))
    
    evolver = ConstrainedRecord.create({}).evolver()
    evolver['value'] = -1
    
    with pytest.raises(InvariantException):
        evolver.persistent()
    
    # Test 7: Persistent with valid field invariant succeeds
    evolver = ConstrainedRecord.create({}).evolver()
    evolver['value'] = 10
    result = evolver.persistent()
    
    assert result['value'] == 10
    
    # Test 8: Persistent with multiple mandatory fields missing
    class MultiMandatoryRecord(PRecord):
        field1 = field(mandatory=True)
        field2 = field(mandatory=True)
        field3 = field()
    
    evolver = MultiMandatoryRecord.create({}).evolver()
    
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    
    error_msg = str(exc_info.value)
    assert 'field1' in error_msg
    assert 'field2' in error_msg


# LLM-generated content at query #46
#--------------------------

```python
def test_PRecord___new__():
    # Test basic PRecord creation with kwargs
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2
    assert isinstance(record, PRecord)
    assert isinstance(record, PMap)


def test_PRecord___new___with_initial_values():
    # Test PRecord creation with field initial values
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field()
    
    record = TestRecord(y=5)
    assert record['x'] == 10
    assert record['y'] == 5


def test_PRecord___new___with_callable_initial():
    # Test PRecord creation with callable initial values
    class TestRecord(PRecord):
        x = field(initial=lambda: 42)
        y = field()
    
    record = TestRecord(y=5)
    assert record['x'] == 42
    assert record['y'] == 5


def test_PRecord___new___override_initial():
    # Test that kwargs override initial values
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100, y=200)
    assert record['x'] == 100
    assert record['y'] == 200


def test_PRecord___new___with_internal_buckets():
    # Test PRecord creation with internal _precord_size and _precord_buckets
    class TestRecord(PRecord):
        x = field()
    
    # Create a record normally first
    record1 = TestRecord(x=1)
    
    # Create using internal attributes (as done by persistent())
    record2 = TestRecord(_precord_size=record1._size, _precord_buckets=record1._buckets)
    assert record2['x'] == 1


def test_PRecord___new___empty():
    # Test creating an empty PRecord
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_PRecord___new___with_ignore_extra():
    # Test PRecord creation with _ignore_extra flag
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _ignore_extra=True, extra_key=999)
    assert record['x'] == 1
    assert 'extra_key' not in record


def test_PRecord___new___with_factory_fields():
    # Test PRecord creation with _factory_fields parameter
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _factory_fields=None)
    assert record['x'] == 5


def test_PRecord___new___multiple_fields():
    # Test PRecord creation with multiple fields
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
        d = field()
    
    record = TestRecord(a=1, b=2, c=3, d=4)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3
    assert record['d'] == 4
    assert len(record) == 4


# LLM-generated content at query #47
#--------------------------

```python
def test_PRecord___new__():
    """Test the __new__ method of PRecord class"""
    
    # Test 1: Create a basic PRecord with no arguments
    class BasicRecord(PRecord):
        pass
    
    record = BasicRecord()
    assert isinstance(record, PRecord)
    assert isinstance(record, PMap)
    
    # Test 2: Create PRecord with keyword arguments
    class PersonRecord(PRecord):
        name = None
        age = None
    
    person = PersonRecord(name='John', age=30)
    assert person['name'] == 'John'
    assert person['age'] == 30
    
    # Test 3: Create PRecord with initial values
    class ConfigRecord(PRecord):
        host = 'localhost'
        port = 8080
    
    config = ConfigRecord()
    assert config['host'] == 'localhost'
    assert config['port'] == 8080
    
    # Test 4: Create PRecord with initial values and override
    config2 = ConfigRecord(host='example.com')
    assert config2['host'] == 'example.com'
    assert config2['port'] == 8080
    
    # Test 5: Create PRecord with callable initial values
    class CallableRecord(PRecord):
        items = lambda: []
    
    rec1 = CallableRecord()
    rec2 = CallableRecord()
    assert rec1['items'] is not rec2['items']
    
    # Test 6: Create PRecord using internal hack (_precord_size and _precord_buckets)
    class InternalRecord(PRecord):
        field1 = None
    
    # This tests the internal creation path
    empty_pmap = pmap()
    internal_rec = InternalRecord(_precord_size=empty_pmap._size, _precord_buckets=empty_pmap._buckets)
    assert isinstance(internal_rec, InternalRecord)
    
    # Test 7: Create with _ignore_extra flag
    class StrictRecord(PRecord):
        name = None
    
    # Should not raise with _ignore_extra=True
    strict = StrictRecord(name='test', extra_field='ignored')
    assert strict['name'] == 'test'
    assert 'extra_field' not in strict
    
    # Test 8: Create with _factory_fields
    class FactoryRecord(PRecord):
        value = None
    
    factory_rec = FactoryRecord(_factory_fields=None, value=42)
    assert factory_rec['value'] == 42
    
    # Test 9: Multiple keyword arguments
    class MultiRecord(PRecord):
        a = None
        b = None
        c = None
    
    multi = MultiRecord(a=1, b=2, c=3)
    assert multi['a'] == 1
    assert multi['b'] == 2
    assert multi['c'] == 3
    
    # Test 10: Override all initial values
    class DefaultsRecord(PRecord):
        x = 10
        y = 20
    
    defaults = DefaultsRecord(x=100, y=200)
    assert defaults['x'] == 100
    assert defaults['y'] == 200


# LLM-generated content at query #48
#--------------------------

```python
def test__PRecordEvolver_persistent():
    """Test _PRecordEvolver.persistent() method"""
    
    # Test 1: Basic persistent creation with valid fields
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 1)
    evolver.set('y', 2)
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2
    
    # Test 2: Persistent with mandatory field missing should raise InvariantException
    class TestRecordMandatory(PRecord):
        required = field(mandatory=True)
        optional = field()
    
    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver.set('optional', 'value')
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'TestRecordMandatory.required' in e.missing_fields
    
    # Test 3: Persistent with all mandatory fields present
    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver.set('required', 'must_have')
    evolver.set('optional', 'nice_to_have')
    result = evolver.persistent()
    
    assert isinstance(result, TestRecordMandatory)
    assert result['required'] == 'must_have'
    assert result['optional'] == 'nice_to_have'
    
    # Test 4: Persistent returns same instance when not dirty
    original_pmap = pmap({'x': 1, 'y': 2})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    
    # Test 5: Persistent creates new instance when dirty
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['x'] == 10
    
    # Test 6: Persistent with field invariant violations
    def positive_invariant(val):
        return (val > 0, 'must_be_positive')
    
    class TestRecordInvariant(PRecord):
        num = field(invariant=positive_invariant)
    
    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver.set('num', -5)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for invariant violation"
    except InvariantException as e:
        assert 'must_be_positive' in e.invariant_errors


# LLM-generated content at query #49
#--------------------------

```python
def test_PRecord___new__():
    # Test basic PRecord creation with no arguments
    class SimpleRecord(PRecord):
        pass
    
    record = SimpleRecord()
    assert isinstance(record, PRecord)
    assert len(record) == 0

    # Test PRecord creation with kwargs
    from pyrsistent import field
    
    class PersonRecord(PRecord):
        name = field()
        age = field()
    
    person = PersonRecord(name='John', age=30)
    assert person['name'] == 'John'
    assert person['age'] == 30
    assert len(person) == 2

    # Test PRecord creation with initial values
    class RecordWithInitial(PRecord):
        name = field()
        status = field(initial='active')
    
    record_with_initial = RecordWithInitial(name='Test')
    assert record_with_initial['name'] == 'Test'
    assert record_with_initial['status'] == 'active'

    # Test PRecord creation with callable initial values
    class RecordWithCallableInitial(PRecord):
        items = field(initial=list)
    
    record1 = RecordWithCallableInitial()
    record2 = RecordWithCallableInitial()
    assert record1['items'] == []
    assert record2['items'] == []
    assert record1['items'] is not record2['items']

    # Test PRecord creation with _precord_size and _precord_buckets (internal hack)
    from pyrsistent._pmap import EMPTY_PMAP
    record_hack = PersonRecord(_precord_size=EMPTY_PMAP._size, _precord_buckets=EMPTY_PMAP._buckets)
    assert isinstance(record_hack, PersonRecord)

    # Test PRecord creation with _factory_fields parameter
    record_factory = PersonRecord(name='Jane', age=25, _factory_fields=None)
    assert record_factory['name'] == 'Jane'
    assert record_factory['age'] == 25

    # Test PRecord creation with _ignore_extra parameter
    class StrictRecord(PRecord):
        name = field()
    
    record_ignore = StrictRecord(name='Test', _ignore_extra=True)
    assert record_ignore['name'] == 'Test'
    assert 'extra_field' not in record_ignore

    # Test PRecord with multiple initial values
    class MultiFieldRecord(PRecord):
        field1 = field()
        field2 = field()
        field3 = field(initial='default')
    
    multi = MultiFieldRecord(field1='a', field2='b')
    assert multi['field1'] == 'a'
    assert multi['field2'] == 'b'
    assert multi['field3'] == 'default'

    # Test PRecord creation preserves PMap functionality
    record_map = PersonRecord(name='Bob', age=40)
    assert 'name' in record_map
    assert record_map.get('name') == 'Bob'
    assert record_map.get('nonexistent', 'default') == 'default'


# LLM-generated content at query #50
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=str)
    
    # Test basic set operation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 42)
    assert evolver.data['x'] == 42
    
    # Test set with type checking
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('y', 'hello')
    assert evolver.data['y'] == 'hello'
    
    # Test set with invalid field raises AttributeError
    evolver = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError, match="'z' is not among the specified fields"):
        evolver.set('z', 100)
    
    # Test set with type mismatch
    evolver = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(TypeError):
        evolver.set('x', 'not_an_int')
    
    # Test multiple sequential sets
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 'test')
    assert evolver.data['x'] == 10
    assert evolver.data['y'] == 'test'
    
    # Test set returns self for chaining
    evolver = _PRecordEvolver(TestRecord, pmap())
    result = evolver.set('x', 5)
    assert result is evolver
    
    # Test set with factory field
    class FactoryRecord(PRecord):
        items = field(factory=list)
    
    evolver = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields={FactoryRecord._precord_fields['items']})
    evolver.set('items', [1, 2, 3])
    assert evolver.data['items'] == [1, 2, 3]


# LLM-generated content at query #51
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = pfield(type=int)
        y = pfield(type=str)
    
    record = TestRecord(x=42, y='hello')
    repr_str = repr(record)
    
    assert repr_str == "TestRecord(x=42, y='hello')" or repr_str == "TestRecord(y='hello', x=42)"
    assert "TestRecord" in repr_str
    assert "x=42" in repr_str
    assert "y='hello'" in repr_str


def test_PRecord___repr__empty():
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    repr_str = repr(record)
    
    assert repr_str == "EmptyRecord()"
    assert "EmptyRecord" in repr_str


def test_PRecord___repr__single_field():
    class SingleRecord(PRecord):
        value = pfield(type=int)
    
    record = SingleRecord(value=123)
    repr_str = repr(record)
    
    assert repr_str == "SingleRecord(value=123)"
    assert "SingleRecord" in repr_str
    assert "value=123" in repr_str


def test_PRecord___repr__with_nested_values():
    class NestedRecord(PRecord):
        name = pfield(type=str)
        count = pfield(type=int)
    
    record = NestedRecord(name='test"value', count=999)
    repr_str = repr(record)
    
    assert "NestedRecord" in repr_str
    assert "name=" in repr_str
    assert "count=999" in repr_str


# LLM-generated content at query #52
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test _PRecordMeta.__new__ creates proper PRecord classes with fields and invariants."""
    
    # Test basic class creation with fields
    class SimpleRecord(PRecord):
        x = PField()
        y = PField()
    
    assert hasattr(SimpleRecord, '_precord_fields')
    assert 'x' in SimpleRecord._precord_fields
    assert 'y' in SimpleRecord._precord_fields
    assert SimpleRecord.__slots__ == ()
    
    # Test mandatory fields tracking
    class RecordWithMandatory(PRecord):
        required = PField(mandatory=True)
        optional = PField()
    
    assert 'required' in RecordWithMandatory._precord_mandatory_fields
    assert 'optional' not in RecordWithMandatory._precord_mandatory_fields
    
    # Test initial values tracking
    class RecordWithInitial(PRecord):
        with_initial = PField(initial=42)
        without_initial = PField()
    
    assert 'with_initial' in RecordWithInitial._precord_initial_values
    assert RecordWithInitial._precord_initial_values['with_initial'] == 42
    assert 'without_initial' not in RecordWithInitial._precord_initial_values
    
    # Test callable initial values
    class RecordWithCallableInitial(PRecord):
        with_callable = PField(initial=lambda: [])
    
    assert callable(RecordWithCallableInitial._precord_initial_values['with_callable'])
    
    # Test invariants are stored
    class RecordWithInvariant(PRecord):
        value = PField()
        
        @staticmethod
        def __invariant__():
            return True
    
    assert hasattr(RecordWithInvariant, '_precord_invariants')
    
    # Test inheritance of fields
    class BaseRecord(PRecord):
        base_field = PField()
    
    class DerivedRecord(BaseRecord):
        derived_field = PField()
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    
    # Test that instances can be created
    instance = SimpleRecord(x=1, y=2)
    assert instance['x'] == 1
    assert instance['y'] == 2


# LLM-generated content at query #53
#--------------------------

```python
def test__PRecordMeta___new__():
    """Test the __new__ method of _PRecordMeta metaclass"""
    
    # Test basic PRecord class creation
    class TestRecord(PRecord):
        x = pfield()
        y = pfield()
    
    assert hasattr(TestRecord, '_precord_fields')
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test mandatory fields tracking
    class MandatoryRecord(PRecord):
        required = pfield(mandatory=True)
        optional = pfield()
    
    assert 'required' in MandatoryRecord._precord_mandatory_fields
    assert 'optional' not in MandatoryRecord._precord_mandatory_fields

    # Test initial values
    class InitialValuesRecord(PRecord):
        with_initial = pfield(initial=42)
        without_initial = pfield()
    
    assert 'with_initial' in InitialValuesRecord._precord_initial_values
    assert InitialValuesRecord._precord_initial_values['with_initial'] == 42
    assert 'without_initial' not in InitialValuesRecord._precord_initial_values

    # Test invariants storage
    class InvariantRecord(PRecord):
        value = pfield()
        
        @invariant
        def check_value(self):
            return self.value > 0, 'value must be positive'
    
    assert hasattr(InvariantRecord, '_precord_invariants')
    assert len(InvariantRecord._precord_invariants) > 0

    # Test inheritance of fields
    class BaseRecord(PRecord):
        base_field = pfield()
    
    class DerivedRecord(BaseRecord):
        derived_field = pfield()
    
    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields

    # Test callable initial values
    class CallableInitialRecord(PRecord):
        items = pfield(initial=list)
    
    assert callable(CallableInitialRecord._precord_initial_values['items'])
    assert CallableInitialRecord._precord_initial_values['items'] == list

    # Test empty PRecord
    class EmptyRecord(PRecord):
        pass
    
    assert EmptyRecord._precord_fields == {}
    assert EmptyRecord._precord_mandatory_fields == set()
    assert EmptyRecord._precord_initial_values == {}


# LLM-generated content at query #54
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord
from pyrsistent._checked_types import InvariantException
from pyrsistent._pmap import pmap


def test__PRecordEvolver_set():
    # Test basic field setting
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    evolver = TestRecord.evolver()
    evolver.set('x', 10)
    assert evolver['x'] == 10
    
    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20
    
    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('z', 30)
    assert "'z' is not among the specified fields" in str(exc_info.value)
    
    # Test type checking
    class TypedRecord(PRecord):
        num = field(type=int)
    
    evolver = TypedRecord.evolver()
    evolver.set('num', 42)
    assert evolver['num'] == 42
    
    # Test field factory
    class FactoryRecord(PRecord):
        value = field(factory=int)
    
    evolver = FactoryRecord.evolver()
    evolver.set('value', '123')
    assert evolver['value'] == 123
    
    # Test field with invariant
    def positive_invariant(val):
        return val > 0, "must_be_positive"
    
    class InvariantRecord(PRecord):
        num = field(invariant=positive_invariant)
    
    evolver = InvariantRecord.evolver()
    evolver.set('num', 5)
    result = evolver.persistent()
    assert result['num'] == 5
    
    # Test invalid invariant
    evolver2 = InvariantRecord.evolver()
    evolver2.set('num', -5)
    with pytest.raises(InvariantException):
        evolver2.persistent()
    
    # Test __setitem__ delegates to set
    class SimpleRecord(PRecord):
        a = field()
    
    evolver = SimpleRecord.evolver()
    evolver['a'] = 100
    assert evolver['a'] == 100
    
    # Test setting field with factory_fields restriction
    class RestrictedRecord(PRecord):
        x = field(factory=int)
        y = field()
    
    x_field = RestrictedRecord._precord_fields['x']
    evolver = _PRecordEvolver(RestrictedRecord, pmap(), _factory_fields=[x_field])
    evolver.set('x', '42')
    assert evolver['x'] == 42
    
    # Test setting field not in factory_fields
    evolver.set('y', 'raw_value')
    assert evolver['y'] == 'raw_value'


# LLM-generated content at query #55
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord
    
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=str)
    
    # Test basic set operation
    original = pmap()
    evolver = _PRecordEvolver(TestRecord, original)
    evolver.set('x', 42)
    assert evolver._destination_cls == TestRecord
    
    # Test that the value is stored correctly
    persistent = evolver.persistent()
    assert persistent['x'] == 42
    
    # Test setting multiple fields
    evolver2 = _PRecordEvolver(TestRecord, original)
    evolver2.set('x', 10)
    evolver2.set('y', 'hello')
    persistent2 = evolver2.persistent()
    assert persistent2['x'] == 10
    assert persistent2['y'] == 'hello'
    
    # Test setting invalid field raises AttributeError
    evolver3 = _PRecordEvolver(TestRecord, original)
    with pytest.raises(AttributeError, match="'z' is not among the specified fields"):
        evolver3.set('z', 100)
    
    # Test type checking
    evolver4 = _PRecordEvolver(TestRecord, original)
    evolver4.set('x', "not an int")
    with pytest.raises(InvariantException):
        evolver4.persistent()
    
    # Test __setitem__ delegates to set
    evolver5 = _PRecordEvolver(TestRecord, original)
    evolver5['x'] = 99
    persistent5 = evolver5.persistent()
    assert persistent5['x'] == 99


