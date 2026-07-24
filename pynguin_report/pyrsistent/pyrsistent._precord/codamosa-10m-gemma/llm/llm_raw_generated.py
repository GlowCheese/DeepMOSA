####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, ignore_extra=True)

def test__PRecordEvolver_set():
    # Test case 1: Setting a valid existing field
    record = MockRecord(name="Alice", age=30)
    evolver = record.evolver()
    evolver.set('age', 31)
    new_record = evolver.persistent()
    assert new_record['age'] == 31
    assert new_record['name'] == "Alice"

    # Test case 2: Setting multiple fields via set (inherited from PMap/PRecord logic via evolver)
    evolver2 = record.evolver()
    evolver2.set('name', 'Bob')
    evolver2.set('age', 25)
    new_record2 = evolver2.persistent()
    assert new_record2['name'] == 'Bob'
    assert new_record2['age'] == 25

    # Test case 3: Setting a non-existent field should raise AttributeError
    evolver3 = record.evolver()
    with pytest.raises(AttributeError) as excinfo:
        evolver3.set('non_existent_field', True)
    assert "'non_existent_field' is not among the specified fields for MockRecord" in str(excinfo.value)

    # Test case 4: Setting a field with an invalid type (triggers check_type via field.factory/check_type)
    # Note: Depending on how check_type is implemented in the environment, 
    # this might raise an error during .set() or during .persistent()
    evolver4 = record.evolver()
    with pytest.raises(Exception):
        evolver4.set('age', "not_an_int")
    
    # Test case 5: Testing the 'factory_fields' logic
    # If a field is not in factory_fields, it should pass through the original value
    # We simulate this by creating an evolver with a restricted factory_fields list
    # Note: _PRecordEvolver is internal, but we can trigger its logic via PRecord constructor
    # if we had access to the internal mechanics, but we'll use the public API.
    
    # Case: Manual verification of the logic inside set() for factory_fields
    # We use the create method which allows passing _factory_fields
    evolver_restricted = MockRecord(name="Alice").evolver()
    # We can't easily inject _factory_fields into an existing evolver via public API,
    # but we can test that if a field is not part of the 'factory' logic, it behaves as expected.
    # Since we can't easily mutate the evolver's private _factory_fields without monkeypatching,
    # we test the standard behavior.
    
    # Test case 6: Invariant failure
    # Create a field with a custom invariant if possible, or use a field that fails type check
    # Since we can't easily add invariants to the class dynamically in this test scope without 
    # redefining, we rely on the type check which is part of the 'set' implementation.
    
    # Test case 7: Verify that setting a field does not mutate the original record
    original = MockRecord(name="Original", age=10)
    evolver_final = original.evolver()
    evolver_final.set('name', 'Changed')
    result = evolver_final.persistent()
    assert original['name'] == "Original"
    assert result['name'] == "Changed"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    
    def __invariant__(self, value):
        if self.age < 0:
            return False, "age_must_be_positive"
        return True, None

def test__PRecordEvolver_persistent():
    # Test 1: Successful persistence of a valid evolution
    e1 = TestRecord().evolver()
    e1['name'] = 'Alice'
    e1['age'] = 30
    result1 = e1.persistent()
    assert result1['name'] == 'Alice'
    assert result1['age'] == 30
    assert isinstance(result1, TestRecord)

    # Test 2: Persistence failure due to missing mandatory field
    e2 = TestRecord().evolver()
    e2['age'] = 25
    # 'name' is mandatory and not set in e2
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'TestRecord.name' in excinfo.value.missing_fields

    # Test 3: Persistence failure due to invariant violation
    e3 = TestRecord().evolver()
    e3['name'] = 'Bob'
    e3['age'] = -5  # Violates __invariant__
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'age_must_be_positive' in excinfo.value.invariant_errors

    # Test 4: Persistence returns existing object if not dirty and matches type
    original = TestRecord(name='Charlie', age=40)
    e4 = original.evolver()
    result4 = e4.persistent()
    assert result4 is original

    # Test 5: Persistence creates new object if dirty
    e5 = original.evolver()
    e5['age'] = 41
    result5 = e5.persistent()
    assert result5['age'] == 41
    assert result5 is not original
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    tags = field(type=list, initial=list)

class TestRecordWithCallable(PRecord):
    data = field(type=dict, initial=dict)

def test_PRecord___new__():
    # Test basic instantiation with mandatory and initial fields
    r1 = TestRecord(name="Alice", age=30)
    assert r1['name'] == "Alice"
    assert r1['age'] == 30
    assert r1['tags'] == []

    # Test instantiation with overriding initial values
    r2 = TestRecord(name="Bob", age=25, tags=["admin"])
    assert r2['name'] == "Bob"
    assert r2['age'] == 25
    assert r2['tags'] == ["admin"]

    # Test instantiation with callable initial values
    r3 = TestRecordWithCallable()
    assert r3['data'] == {}
    
    # Test that the __new__ logic handles the internal 'hack' for reconstruction
    # by simulating the behavior of the persistent() method of an evolver
    # which passes _precord_size and _precord_buckets
    # We use a dummy pmap-like structure for buckets
    from pyrsistent import pmap
    dummy_pmap = pmap({'name': 'Charlie', 'age': 40})
    # The internal structure of PMap/PRecord uses buckets. 
    # We can't easily mock the internal buckets without deep pyrsistent internals,
    # but we can verify that providing the specific magic kwargs avoids the Evolver.
    
    # Testing the error case: Missing mandatory field
    with pytest.raises(InvariantException) as excinfo:
        TestRecord(age=10)
    assert any("TestRecord.name" in err for err in excinfo.value.missing_fields)

    # Test that __new__ handles extra arguments via the evolver's logic
    # (though PRecord doesn't allow extra fields by default unless specified)
    with pytest.raises(AttributeError):
        TestRecord(name="Alice", unknown_field="error")

    # Test that __new__ correctly processes the factory_fields logic
    # via the create method which uses the __new__ underlying logic
    r4 = TestRecord.create({'name': 'Dave'}, ignore_extra=True)
    assert r4['name'] == 'Dave'
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class DummyField:
    def __init__(self, mandatory=False, initial=None, factory=None, invariant=None, serializer=None):
        self.mandatory = mandatory
        self.initial = initial
        self.factory = factory or (lambda x: x)
        self.invariant = invariant or (lambda x: (True, None))
        self.serializer = serializer

class TestPRecord(PRecord):
    field1 = DummyField()
    field2 = DummyField(mandatory=True)

def test__PRecordEvolver_set():
    # 1. Test setting a valid existing field
    initial_map = pmap({'field1': 'old_value', 'field2': 'val2'})
    evolver = _PRecordEvolver(TestPRecord, initial_map)
    evolver.set('field1', 'new_value')
    result = evolver.persistent()
    assert result['field1'] == 'new_value'
    assert result['field2'] == 'val2'

    # 2. Test setting an invalid field (AttributeError)
    with pytest.raises(AttributeError) as excinfo:
        evolver.set('non_existent_field', 123)
    assert "is not among the specified fields" in str(excinfo.value)

    # 3. Test invariant failure (returns evolver without updating, collects error)
    def failing_invariant(val):
        return False, "error_code_123"

    class InvariantField(DummyField):
        def __init__(self):
            super().__init__(invariant=failing_invariant)

    # We need a class with the failing field for this specific test
    class InvariantRecord(PRecord):
        bad_field = InvariantField()

    initial_map_inv = pmap({'bad_field': 'original'})
    evolver_inv = _PRecordEvolver(InvariantRecord, initial_map_inv)
    evolver_inv.set('bad_field', 'trigger_failure')
    
    with pytest.raises(InvariantException) as excinfo:
        evolver_inv.persistent()
    assert 'error_code_123' in excinfo.value.invariant_errors

    # 4. Test factory function usage
    class FactoryField(DummyField):
        def __init__(self):
            super().__init__(factory=lambda x: x.upper())

    class FactoryRecord(PRecord):
        f = FactoryField()

    e_factory = _PRecordEvolver(FactoryRecord, pmap({'f': 'start'}))
    e_factory.set('f', 'lowercase')
    assert e_factory.persistent()['f'] == 'LOWERCASE'

    # 5. Test _factory_fields filtering
    # If key is not in _factory_fields, it should use original value
    e_filtered = _PRecordEvolver(TestPRecord, initial_map, _factory_fields=['field1'])
    # field1 is in factory_fields, so factory(val) is called. 
    # field2 is NOT in factory_fields, so it takes the value as-is (though it's already in the map)
    # Let's test the logic: if field is NOT in _factory_fields, value = original_value
    e_filtered.set('field1', 'new') 
    # field1 is in _factory_fields, so it executes the factory (identity in DummyField)
    assert e_filtered.persistent()['field1'] == 'new'

    # 6. Test type checking (check_type)
    # Assuming check_type raises error on mismatch (standard pyrsistent behavior)
    # We mock check_type to trigger an error if it's not the default behavior in the env
    with pytest.raises(Exception):
        # Creating a field that expects int but passing str
        class IntField(DummyField):
            def __init__(self):
                super().__init__()
        
        class IntRecord(PRecord):
            num = IntField()
            
        e_type = _PRecordEvolver(IntRecord, pmap({'num': 1}))
        # We bypass the actual check_type implementation complexity by assuming 
        # it's part of the environment's validation logic
        # If check_type is working, this should fail.
        e_type.set('num', 'not_an_int')
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=int, mandatory=False)

def test__PRecordEvolver_set():
    # Setup: Create an initial record
    initial_record = TestRecord(name="Alice", age=30)
    
    # 1. Test setting an existing field with valid type
    e1 = initial_record.evolver()
    e1.set('name', 'Bob')
    res1 = e1.persistent()
    assert res1['name'] == 'Bob'
    assert res1['age'] == 30

    # 2. Test setting multiple fields using the set method (via update logic)
    e2 = initial_record.evolver()
    e2.set('age', 31)
    e2.set('extra', 100)
    res2 = e2.persistent()
    assert res2['age'] == 31
    assert res2['extra'] == 100

    # 3. Test setting a field with an invalid type (should raise error during set/check_type)
    # Note: check_type is called inside set()
    e3 = initial_record.evolver()
    with pytest.raises(Exception):
        e3.set('age', "not_an_int")

    # 4. Test setting a non-existent field (should raise AttributeError)
    e4 = initial_record.evolver()
    with pytest.raises(AttributeError) as excinfo:
        e4.set('non_existent', 123)
    assert "is not among the specified fields" in str(excinfo.value)

    # 5. Test the behavior with _factory_fields (filtering fields)
    # We define a field that should be ignored by the factory logic if not in factory_fields
    class FactoryRecord(PRecord):
        a = field(type=int)
        b = field(type=int)

    # Create evolver where only 'a' is allowed to be processed by the factory
    e5 = FactoryRecord(a=1, b=2).evolver()
    # We simulate the _factory_fields logic by passing it to the evolver via a custom setup
    # Since we cannot easily inject into the private _PRecordEvolver constructor from outside 
    # without knowing the exact internal state, we test the logic as it applies to the class.
    
    # 6. Test Invariant failure during persistent()
    class InvariantRecord(PRecord):
        value = field(type=int)
        @classmethod
        def __invariant__(cls, value):
            if value < 0:
                return False, "must_be_positive"
            return True, None

    e6 = InvariantRecord(value=10).evolver()
    e6.set('value', -5)
    with pytest.raises(InvariantException) as excinfo:
        e6.persistent()
    assert "must_be_positive" in excinfo.value.invariant_errors

    # 7. Test Mandatory field missing during persistent()
    e7 = TestRecord(name="Alice").evolver()
    # We use a trick: the evolver starts with a PMap that might be missing keys 
    # if we manually manipulate it, but in standard usage, PRecord constructor 
    # handles initial values. To trigger missing fields, we'd need to bypass the constructor.
    # However, we can test that if we don't set a mandatory field in an empty evolver:
    class MandatoryOnly(PRecord):
        req = field(type=int, mandatory=True)
    
    # This is difficult because the Evolver is initialized with the record.
    # But if the record itself was somehow missing a key (impossible via PRecord __new__),
    # the persistent() method checks it. 
    # We'll rely on the InvariantException test above for the logic coverage.
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

class TestPRecordEvolver:
    def test__PRecordEvolver_persistent_success(self):
        """Test that persistent() returns a valid PRecord when invariants are met."""
        e = MockRecord.evolver()
        e['name'] = 'Alice'
        e['age'] = 30
        record = e.persistent()
        
        assert isinstance(record, MockRecord)
        assert record['name'] == 'Alice'
        assert record['age'] == 30

    def test__PRecordEvolver_persistent_missing_mandatory_field(self):
        """Test that persistent() raises InvariantException when mandatory fields are missing."""
        e = MockRecord.evolver()
        # 'name' is mandatory and not set
        e['age'] = 25
        
        with pytest.raises(InvariantException) as excinfo:
            e.persistent()
        
        # Check that the error contains the missing field path
        assert any('MockRecord.name' in missing for missing in excinfo.value.missing_fields)

    def test__PKeyError_on_invalid_field(self):
        """Test that setting a non-existent field in evolver raises AttributeError."""
        e = MockRecord.evolver()
        with pytest.raises(AttributeError) as excinfo:
            e['non_existent'] = 'value'
        assert "is not among the specified fields" in str(excinfo.value)

    def test__PRecordEvolver_persistent_with_dirty_state(self):
        """Test that persistent() correctly handles the is_dirty logic for new object creation."""
        # Create initial record
        base = MockRecord(name='Bob', age=40)
        e = base.evolver()
        e['age'] = 41
        
        # is_dirty should be true because we modified a field
        assert e.is_dirty() is True
        
        record = e.persistent()
        assert record['age'] == 41
        assert record['name'] == 'Bob'
        assert isinstance(record, MockRecord)

    def test__PRecordEvolver_persistent_with_invariant_failure(self):
        """Test that persistent() raises InvariantException if a field invariant fails."""
        # Define a custom record with a custom invariant for testing
        class InvariantRecord(PRecord):
            value = field(type=int)
            
            @classmethod
            def __invariant__(cls, value):
                if value < 0:
                    return False, "value_must_be_positive"
                return True, None

        e = InvariantRecord.evolver()
        e['value'] = -10
        
        with pytest.raises(InvariantException) as excinfo:
            e.persistent()
        
        assert "value_must_be_positive" in excinfo.value.invariant_errors
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

def test__PRecordEvolver_persistent():
    # Test case 1: Successful persistence of a valid record
    e1 = MockRecord.evolver()
    e1['name'] = 'John'
    e1['age'] = 30
    record = e1.persistent()
    assert isinstance(record, MockRecord)
    assert record['name'] == 'John'
    assert record['age'] == 30

    # Test case 2: Persistence fails due to missing mandatory fields
    e2 = MockRecord.evolver()
    e2['age'] = 25
    # 'name' is mandatory and not provided
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'MockRecord.name' in excinfo.value.missing_fields

    # Test case 3: Persistence fails due to invariant violation
    # (Assuming a custom field with an invariant is defined)
    class InvariantRecord(PRecord):
        value = field(type=int)
        
        @classmethod
        def __invariant__(cls, value):
            if value < 0:
                return False, 'must_be_positive'
            return True, None

    e3 = InvariantRecord.evolver()
    e3['value'] = -10
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'must_be_positive' in excinfo.value.invariant_errors

    # Test case 4: Check that persistent() returns the same object if not dirty
    # (When using the internal mechanism for reconstruction)
    original_record = MockRecord(name='Alice')
    e4 = original_record.evolver()
    # No changes made to e4
    new_record = e4.persistent()
    assert new_record is original_record

    # Test case 5: Check that persistent() returns a new instance if dirty
    e5 = original_record.evolver()
    e5['age'] = 31
    new_record_dirty = e5.persistent()
    assert new_record_dirty is not original_record
    assert new_record_dirty['age'] == 31
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class TestRecord(PRecord):
    name = field(type=str)
    age = field(type=int, initial=0)
    active = field(type=bool, initial=True)

def test_PRecord___repr__():
    # Test empty/default record representation
    record_default = TestRecord()
    # Note: PMap/PRecord order depends on insertion/initialization order
    # We check if the string contains the expected key-value pairs
    repr_default = repr(record_default)
    assert "TestRecord" in repr_default
    assert "name" in repr_default or "age" in repr_default or "active" in repr_default
    
    # Test record with specific values
    record_custom = TestRecord(name="Alice", age=30)
    repr_custom = repr(record_custom)
    assert "TestRecord" in repr_custom
    assert "name='Alice'" in repr_custom
    assert "age=30" in repr_custom

    # Test record with different types
    record_bool = TestRecord(name="Bob", active=False)
    repr_bool = repr(record_bool)
    assert "active=False" in repr_bool

    # Test that the representation is a string
    assert isinstance(repr_custom, str)

    # Verify that all items in the record are present in the repr string
    for k, v in record_custom.items():
        expected_part = f"{k}={repr(v)}"
        assert expected_part in repr_custom
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from pyrsistent import field

class MockSerializer:
    def __call__(self, format, value):
        if format == 'upper':
            return str(value).upper()
        return value

class TestRecord(PRecord):
    name = field(type=str)
    age = field(type=int, serializer=MockSerializer())
    tags = field(type=list, serializer=lambda f, v: ",".join(v))

def test_PRecord_serialize():
    # Test default serialization (returns dict with original values)
    record = TestRecord(name="Alice", age=30, tags=["python", "unit"])
    serialized_default = record.serialize()
    assert serialized_default == {"name": "Alice", "age": 30, "tags": ["python", "unit"]}

    # Test serialization with custom format for 'age' (MockSerializer uses 'upper' logic)
    # Note: MockSerializer implementation in the test checks for 'upper'
    serialized_upper = record.serialize(format='upper')
    assert serialized_upper["name"] == "Alice"
    assert serialized_upper["age"] == "30"
    assert serialized_upper["tags"] == "python,unit"

    # Test serialization with different format for 'tags'
    # Since tags uses a lambda that ignores format, it should still work
    serialized_tags_only = record.serialize(format='something_else')
    assert serialized_tags_only["tags"] == "python,unit"

    # Test that it handles empty records/fields if they were optional
    class SimpleRecord(PRecord):
        val = field(type=int, serializer=lambda f, v: v * 2)
    
    simple_rec = SimpleRecord(val=10)
    assert simple_rec.serialize() == {"val": 20}
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, mandatory=False)

def test__PRecordEvolver_set():
    # Test successful set of existing field
    record = MockRecord(name="John", age=30)
    evolver = record.evolver()
    evolver.set("name", "Jane")
    evolver.set("age", 25)
    new_record = evolver.persistent()
    assert new_record["name"] == "Jane"
    assert new_record["age"] == 25
    assert new_record["extra"] is None

    # Test set using positional arguments (via PRecord.set logic if applicable)
    # Note: The requirement specifically asks for _PRecordEvolver.set
    evolver2 = record.evolver()
    evolver2.set("name", "Bob")
    assert evolver2.persistent()["name"] == "Bob"

    # Test setting a field that is not in the PRecord definition (should raise AttributeError)
    evolver3 = record.evroll_error = record.evolver()
    with pytest.raises(AttributeError) as excinfo:
        evolver3.set("non_existent_field", "value")
    assert "is not among the specified fields" in str(excinfo.value)

    # Test setting a field with an invalid type (should raise type error via check_type)
    evolver4 = record.evolver()
    with pytest.raises(TypeError):
        evolver4.set("age", "not_an_int")

    # Test setting a field that violates an invariant (if we had a custom invariant)
    # Since we can't easily inject an invariant into the class definition at runtime 
    # without modifying the class, we rely on the built-in check_type/type validation.
    
    # Test that the evolver maintains other fields
    evolver5 = record.evolver()
    evolver5.set("name", "Alice")
    result = evolver5.persistent()
    assert result["age"] == 30  # Original value preserved
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class SampleRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

def test__PRecordEvolver_persistent():
    # Case 1: Successful persistence of a valid record
    e1 = SampleRecord.evolver()
    e1['name'] = 'Alice'
    e1['age'] = 30
    record = e1.persistent()
    assert record['name'] == 'Alice'
    assert record['age'] == 30
    assert isinstance(record, SampleRecord)

    # Case 2: Persistence fails due to missing mandatory fields
    # 'name' is mandatory in SampleRecord
    e2 = SampleRecord.evolver()
    e2['age'] = 25
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'SampleRecord.name' in excinfo.value.missing_fields

    # Case 3: Persistence fails due to invariant violation
    # We define a custom field with an invariant for this test
    class InvariantRecord(PRecord):
        value = field(type=int)
        
        @classmethod
        def __invariant__(cls, value):
            if value < 0:
                return False, 'must_be_positive'
            return True, None

    e3 = InvariantRecord.evolver()
    e3['value'] = -5
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'must_be_positive' in excinfo.value.invariant_errors

    # Case 4: Persistence returns existing object if not dirty (optimization check)
    # Creating a record directly
    original = SampleRecord(name='Bob', age=40)
    e4 = original.evolver()
    # If we don't change anything, persistent() should return the same instance
    result = e4.persistent()
    assert result is original

    # Case 5: Persistence handles dirty state by creating new instance
    e5 = original.evolver()
    e5['age'] = 41
    result_dirty = e5.persistent()
    assert result_dirty is not original
    assert result_dirty['age'] == 41
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from pyrsistent import PRecord, pmap
from pyrsistent._checked_types import InvariantException, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

class TestPRecordEvolver:
    def test__PRecordEvolver_persistent(self):
        # Case 1: Successful persistence of a valid record
        e1 = _PRecordEvolver(MockRecord, pmap({'name': 'Alice', 'age': 30}))
        result1 = e1.persistent()
        assert isinstance(result1, MockRecord)
        assert result1['name'] == 'Alice'
        assert result1['age'] == 30

        # Case 2: Persistence fails due to missing mandatory fields
        # We manually bypass the constructor to create an incomplete state
        # because the constructor/evolver usually validates.
        # We simulate a state where 'name' (mandatory) is missing.
        e2 = _PRecordEvalver_incomplete_setup()
        with pytest.raises(InvariantException) as excinfo:
            e2.persistent()
        assert any('MockRecord.name' in err for err in excinfo.value.missing_fields)

        # Case 3: Persistence fails due to invariant violation
        # We use a field with a custom invariant
        class InvariantRecord(PRecord):
            value = field(type=int)
            def __invariant__(self, value):
                if value < 0:
                    return False, "must_be_positive"
                return True, None

        e3 = _PRecordEvolver(InvariantRecord, pmap({'value': 10}))
        e3.set('value', -5)  # This triggers the invariant check during set
        with pytest.raises(InvariantException) as excinfo:
            e3.persistent()
        assert "must_be_positive" in excinfo.value.invariant_errors

        # Case 4: Persistence returns existing object if not dirty
        # (Assuming the internal pmap is already an instance of MockRecord)
        existing_record = MockRecord(name='Bob', age=20)
        e4 = _PRecordEvolver(MockRecord, existing_record)
        result4 = e4.persistent()
        assert result4 is existing_record

def _PRecordEvolver_incomplete_setup():
    """
    Helper to create an evolver in a state that violates mandatory fields
    without triggering errors during the 'set' phase of the evolver.
    """
    # We use the low-level constructor to bypass the logic in __setitem__
    # that would normally enforce completeness.
    base_pmap = pmap({'age': 25})
    # Manually create an evolver that lacks the 'name' field
    e = _PRecordEvolver(MockRecord, base_pmap)
    return e
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

class InvariantRecord(PRecord):
    value = field(type=int, invariant=lambda x: (x > 0, "must_be_positive"))

def test__PRecordEvolver_persistent():
    # Test Case 1: Successful persistence of a standard PRecord
    e1 = MockRecord.evolver()
    e1['name'] = 'Alice'
    e1['age'] = 30
    res1 = e1.persistent()
    assert isinstance(res1, MockRecord)
    assert res1['name'] == 'Alice'
    assert res1['age'] == 30

    # Test Case 2: Persistence fails due to missing mandatory fields
    e2 = MockRecord.evolver()
    e2['age'] = 25
    # 'name' is mandatory and not set in evolver
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'MockRecord.name' in excinfo.value.missing_fields

    # Test Case 3: Persistence fails due to field invariant violation
    e3 = InvariantRecord.evolver()
    e3['value'] = -5
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'must_be_positive' in excinfo.value.invariant_errors

    # Test Case 4: Successful persistence when using factory/ignore_extra logic
    # (Testing the path where is_field_ignore_extra_complaint might be triggered)
    class ExtraFieldRecord(PRecord):
        a = field(type=int)

    e4 = ExtraFieldRecord.evolver()
    e4['a'] = 10
    res4 = e4.persistent()
    assert res4['a'] == 10

    # Test Case 5: Verification that it returns the same object if not dirty
    # We create a record and use its evolver without changes
    base = MockRecord(name='Bob', age=40)
    e5 = base.evolver()
    res5 = e5.persistent()
    assert res5 is base

    # Test Case 6: Verification that it returns a new object (new instance) if dirty
    e6 = base.evolver()
    e6['age'] = 41
    res6 = e6.persistent()
    assert res6 is not base
    assert res6['age'] == 41
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

class TestRecordWithInvariant(PRecord):
    value = field(type=int)

    def __invariant__(self, value):
        if value < 0:
            return False, "value_must_be_positive"
        return True, None

def test__PRecordEvolver_persistent():
    # 1. Test successful persistence of a valid record
    e1 = TestRecord().evolver()
    e1['name'] = 'Alice'
    e1['age'] = 30
    res1 = e1.persistent()
    assert isinstance(res1, TestRecord)
    assert res1['name'] == 'Alice'
    assert res1['age'] == 30

    # 2. Test persistence failure due to missing mandatory fields
    e2 = TestRecord().evolver()
    e2['age'] = 25
    # 'name' is mandatory and not set in evolver
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'TestRecord.name' in excinfo.value.missing_fields

    # 3. Test persistence failure due to invariant violation
    e3 = TestRecordWithInvariant().evolver()
    e3['value'] = -10
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'value_must_be_positive' in excinfo.value.invariant_errors

    # 4. Test persistence where the evolver is "dirty" (updates performed)
    # This triggers the internal re-instantiation logic
    e4 = TestRecord().evolver()
    e4['name'] = 'Bob'
    res4 = e4.persistent()
    assert res4['name'] == 'Bob'
    assert isinstance(res4, TestRecord)

    # 5. Test persistence with an extra field via factory_fields (if supported by implementation)
    # Given the provided code, we test if setting a non-existent field raises AttributeError
    e5 = TestRecord().evolver()
    with pytest.raises(AttributeError):
        e5['non_existent'] = 'error'

    # 6. Test that persistent() returns the same object if not dirty and matches class
    # We create a record and use its evolver without changing anything
    original = TestRecord(name='Charlie')
    e6 = original.evolver()
    res6 = e6.persistent()
    assert res6 is original
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, mandatory=False)

    def __invariant__(self, value):
        # Dummy invariant for testing
        return True, None

def test__PRecordEvolver_persistent():
    # 1. Test successful persistence of a valid record
    e = TestRecord.evolver()
    e['name'] = 'Alice'
    e['age'] = 30
    record = e.persistent()
    assert isinstance(record, TestRecord)
    assert record['name'] == 'Alice'
    assert record['age'] == 30

    # 2. Test persistence failure due to missing mandatory fields
    # 'name' is mandatory in TestRecord
    e_missing = TestRecord.evolver()
    e_missing['age'] = 25
    with pytest.raises(InvariantException) as excinfo:
        e_missing.persistent()
    assert 'TestRecord.name' in excinfo.value.missing_fields

    # 3. Test persistence failure due to invariant violation
    # We use a custom subclass to trigger a specific error code
    class InvariantRecord(PRecord):
        val = field(type=int)
        def __invariant__(self, value):
            if value < 0:
                return False, "negative_error"
            return True, None

    e_inv = InvariantRecord.evolver()
    e_inv['val'] = -10
    with pytest.raises(InvariantException) as excinfo:
        e_inv.persistent()
    assert "negative_error" in excinfo.value.invariant_errors

    # 4. Test that persistent() returns the same object if not dirty (optimization check)
    # Note: PMap's evolver returns the same object if no changes were made
    original = TestRecord(name='Bob', age=20)
    e_no_change = original.evolver()
    persisted_no_change = e_no_change.persistent()
    assert persisted_no_change is original

    # 5. Test that persistent() returns a new object if dirty
    e_dirty = original.evolver()
    e_dirty['age'] = 21
    persisted_dirty = e_dirty.persistent()
    assert persisted_dirty is not original
    assert persisted_dirty['age'] == 21
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

    def __invariant__(self, name, age):
        if age < 0:
            raise InvariantException(error_codes=('age_negative',), missing_fields=())

def test__PRecordEvolver_persistent():
    # Test Case 1: Successful persistence of a valid record
    e1 = MockRecord.evolver()
    e1['name'] = 'Alice'
    e1['age'] = 30
    record1 = e1.persistent()
    assert record1['name'] == 'Alice'
    assert record1['age'] == 30
    assert isinstance(record1, MockRecord)

    # Test Case 2: Persistence fails due to missing mandatory fields
    e2 = MockRecord.evolver()
    e2['age'] = 25
    # 'name' is mandatory and not set in evolver
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'MockRecord.name' in excinfo.value.missing_fields

    # Test Case 3: Persistence fails due to field invariant violation
    e3 = MockRecord.evolver()
    e3['name'] = 'Bob'
    e3['age'] = -5  # Violates __invariant__
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'age_negative' in excinfo.value.invariant_errors

    # Test Case 4: Persistence of a record without changes (is_dirty is False)
    # When is_dirty is false, it should return the original pmap if it's already the correct class
    original = MockRecord(name='Charlie', age=40)
    e4 = original.evolver()
    # No changes made to e4
    result4 = e4.persistent()
    assert result4 == original
    assert result4 is original

    # Test Case 5: Persistence when is_dirty is True but no errors
    e5 = MockRecord.evolver()
    e5['name'] = 'Dave'
    e5['age'] = 50
    result5 = e5.persistent()
    assert result5['name'] == 'Dave'
    assert result5 is not original # It's a new instance because it's dirty
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    tags = field(type=list, initial=list)

class TestRecordWithCallable(PRecord):
    timestamp = field(type=int, initial=lambda: 100)

class TestRecordWithExtra(PRecord):
    id = field(type=int)

def test_PRecord___new__():
    # Test basic instantiation with mandatory and optional fields
    r1 = TestRecord(name="Alice", age=30)
    assert r1['name'] == "Alice"
    assert r1['age'] == 30
    assert r1['tags'] == []

    # Test instantiation with initial values provided via kwargs
    r2 = TestRecord(name="Bob", tags=["admin"])
    assert r2['name'] == "Bob"
    assert r2['tags'] == ["admin"]
    assert r2['age'] == 0

    # Test instantiation with callable initial values
    r3 = TestRecordWithCallable(name="Charlie")
    assert r3['timestamp'] == 100

    # Test that the internal __new__ hack for restoration works (simulating unpickling)
    # We bypass the factory logic by providing the internal keys used by PMap
    # This mimics the behavior when _restore_pickle is called
    r4 = TestRecord.__new__(
        TestRecord, 
        _precord_size=2, 
        _precord_buckets={}, 
        name="Dave", 
        age=40
    )
    # Note: In a real environment, the __new__ signature for PMap restoration 
    # is more complex, but here we test the logic branch provided in the code.
    assert r4['name'] == "Dave"
    assert r4['age'] == 40

    # Test that attempting to set an undefined field raises AttributeError 
    # (This tests the logic inside the Evolver used by __new__)
    with pytest.raises(AttributeError) as excinfo:
        TestRecord(name="Eve", unknown_field="error")
    assert "is not among the specified fields" in str(excinfo.value)

    # Test mandatory field violation during the persistent() phase of __new__
    with pytest.raises(InvariantException) as excinfo:
        # Creating a record without 'name' which is mandatory
        TestRecord(age=25).persistent()
    assert any("TestRecord.name" in err for err in excinfo.value.missing_fields)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from pyrsistent import field

class MockSerializer:
    def __call__(self, format, value):
        if format == 'upper':
            return str(value).upper()
        return value

class TestRecord(PRecord):
    name = field(type=str)
    value = field(type=int, serializer=MockSerializer())
    metadata = field(type=dict, serializer=lambda f, v: f"fmt_{f}_{v}")

def test_PRecord_serialize():
    # Test basic serialization without custom serializers
    record1 = TestRecord(name="test", value=123, metadata={'a': 1})
    serialized1 = record1.serialize()
    assert serialized1['name'] == "test"
    assert serialized1['value'] == 123
    assert serialized1['metadata'] == {'a': 1}

    # Test serialization with 'upper' format for the value field
    # The MockSerializer handles the 'upper' format logic
    serialized2 = record1.serialize(format='upper')
    assert serialized2['name'] == "test"
    assert serialized2['value'] == "123"
    # The lambda for metadata uses the format in the string
    assert serialized2['metadata'] == "fmt_upper_{'a': 1}"

    # Test serialization with a different format for metadata
    serialized3 = record1.serialize(format='json')
    assert serialized3['metadata'] == "fmt_json_{'a': 1}"

    # Test that all keys from the record are present in the serialized dict
    assert set(serialized1.keys()) == set(record1.keys())
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockField:
    def __init__(self, mandatory=False, initial=None):
        self.mandatory = mandatory
        self.initial = initial

def test__PRecordMeta___new__():
    # Define a class that uses the metaclass
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int, initial=0)
        description = field(type=str, initial=lambda: "default")

    # Verify that __new__ correctly set up the class attributes
    
    # 1. Check if _precord_fields was set (via set_fields in __new__)
    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'description' in TestRecord._precord_fields

    # 2. Check if _precord_invariants was set (via store_invariants in __new__)
    assert hasattr(TestRecord, '_precord_invariants')

    # 3. Check if _precoid_mandatory_fields contains only mandatory fields
    assert TestRecord._precord_mandatory_fields == {'name'}

    # 4. Check if _precord_initial_values contains non-PFIELD_NO_INITIAL values
    # Note: 'description' is a callable, 'age' is a literal
    assert 'age' in TestRecord._precord_initial_values
    assert TestRecord._precord_initial_values['age'] == 0
    assert 'description' in TestRecord._precord_initial_values
    assert callable(TestRecord._precord_initial_values['description'])

    # 5. Check if __slots__ is set to empty tuple to prevent instance dict overhead
    assert TestRecord.__slots__ == ()

    # 6. Verify functionality of the resulting class
    instance = TestRecord(name="Alice")
    assert instance.name == "Alice"
    assert instance.age == 0
    assert instance.description == "default"

    # 7. Verify that the metaclass correctly handles inheritance
    class SubRecord(TestRecord):
        extra = field(type=bool)
    
    assert 'name' in SubRecord._precord_fields
    assert 'extra' in SubRecord._precord_fields
    assert 'name' in SubRecord._precord_mandatory_fields
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from pyrsistent import PRecord, PField

def test__PRecordMeta___new__():
    # Define a dummy class using the metaclass
    class MockRecord(PRecord):
        name = PField(type=str, mandatory=True)
        age = PField(type=int, initial=0)
        active = PField(type=bool, initial=lambda: True)

    # Test that the metaclass correctly populated the class attributes
    # 1. Check if _precord_fields was set by set_fields
    assert hasattr(MockRecord, '_precord_fields')
    assert 'name' in MockRecord._precort_fields
    assert 'age' in MockRecord._precort_fields
    assert 'active' in MockRecord._precort_fields

    # 2. Check if _precord_mandatory_fields was calculated correctly
    assert 'name' in MockRecord._precord_mandatory_fields
    assert 'age' not in MockRecord._precord_mandatory_fields
    assert 'active' not in MockRecord._precord_mandatory_fields

    # 3. Check if _precord_initial_values was calculated correctly
    # Note: 'active' is a callable (lambda), so it should be stored as the callable
    assert MockRecord._precord_initial_values['age'] == 0
    assert callable(MockRecord._precord_initial_values['active'])

    # 4. Check if invariants were stored
    assert hasattr(MockRecord, '_precord_invariants')

    # 5. Check if __slots__ is set to empty tuple (to prevent attribute injection)
    assert MockRecord.__slots__ == ()

    # 6. Test instantiation via __new__ logic (indirectly via class creation)
    # The __new__ logic in PRecord handles the reconstruction from buckets
    # We verify that the class behaves as a PRecord
    instance = MockRecord(name="Test")
    assert instance.name == "Test"
    assert instance.age == 0
    assert instance.active is True
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    tags = field(type=list, initial=list)

class CallableRecord(PRecord):
    data = field(type=list, initial=list)

def test_PRecord___new__():
    # Test basic instantiation with kwargs
    rec1 = MockRecord(name="Alice", age=30)
    assert rec1['name'] == "Alice"
    assert rec1['age'] == 30
    assert rec1['tags'] == []

    # Test instantiation with initial values being callables (factory)
    rec2 = CallableRecord()
    assert rec2['data'] == []
    assert isinstance(rec2['data'], list)

    # Test that mandatory fields must be provided (otherwise InvariantException on persistent/completion)
    # Note: The error is raised when the evolver tries to finalize the persistent object
    with pytest.raises(InvariantException) as excinfo:
        MockRecord(age=25)
    assert any("MockRecord.name" in err for err in excintfo.value.missing_fields)

    # Test the "hack" for internal reconstruction (using _precord_size and _precord_buckets)
    # We simulate the internal state of a PMap being restored
    from pyrsistent import pmap
    internal_map = pmap({'name': 'Bob', 'age': 40, 'tags': ['admin']})
    
    # This bypasses the standard __new__ logic to test the internal recovery path
    rec3 = MockRecord(
        _precord_size=internal_map._size,
        _precord_buckets=internal_map._buckets
    )
    assert rec3['name'] == 'Bob'
    assert rec3['age'] == 40
    assert rec3['tags'] == ['admin']

    # Test that extra fields are not allowed by default via __new__ 
    # (The evolver raises AttributeError if key not in _precord_fields)
    with pytest.raises(AttributeError) as excinfo:
        MockRecord(name="Charlie", unknown_field="error")
    assert "is not among the specified fields" in str(excinfo.value)

    # Test with _factory_fields and _ignore_extra
    # We use the create method which wraps the logic or call __new__ directly
    # testing the logic inside __new__ for _factory_fields
    rec4 = MockRecord.create({'name': 'Dave', 'age': 50, 'extra': 'ignored'}, ignore_extra=True)
    assert 'name' in rec4
    assert 'extra' not in rec4

    # Test that initial values are correctly merged with kwargs
    rec5 = MockRecord(name="Eve", age=10)
    assert rec5['age'] == 10
    assert rec5['name'] == "Eve"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, ignore_extra=True)

def test__PRecordEvolver_set():
    # Test 1: Setting a valid field
    e = _PRecordEvolver(MockRecord, pmap())
    e.set('name', 'Alice')
    assert e['name'] == 'Alice'

    # Test 2: Setting a field with an initial value (verifying it updates)
    e2 = _PRecordEvolver(MockRecord, pmap())
    e2.set('age', 25)
    assert e2['age'] == 25

    # Test 3: Setting an invalid type (should trigger check_type via field logic)
    # Note: Depending on pyrsistent version, this might raise AttributeError or InvariantException
    with pytest.raises(Exception):
        e2.set('age', 'not_an_int')

    # Test 4: Setting a non-existent field (should raise AttributeError)
    with pytest.raises(AttributeError) as excinfo:
        e.set('non_existent', 'value')
    assert "is not among the specified fields" in str(excinfo.value)

    # Test 5: Testing the 'factory_fields' filtering logic
    # If a field is not in factory_fields, the original value is used instead of being passed to factory
    # We use a custom field definition via a subclass for precise control
    class FactoryRecord(PRecord):
        val = field(type=int)

    # In the implementation: 
    # if self._factory_fields is None or field in self._factory_fields:
    #    value = field.factory(original_value, ...)
    # else:
    #    value = original_value
    
    e_factory = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields=['val'])
    e_factory.set('val', 10)
    assert e_factory['val'] == 10

    # Test 6: Testing the 'ignore_extra' logic for factory calls
    # We simulate a field that handles ignore_extra
    class ExtraRecord(PRecord):
        val = field(type=str, ignore_extra=True)

    e_ignore = _PRecordEvolver(ExtraRecord, pmap(), _ignore_extra=True)
    # If ignore_extra is True, the evolver calls field.factory(val, ignore_extra=True)
    e_ignore.set('val', 'test')
    assert e_ignore['val'] == 'test'

    # Test 7: Invariant failure during set
    class InvariantRecord(PRecord):
        score = field(type=int)
        def __invariant__(self, value):
            # This is a simplified mock of how invariants are checked in the evolver
            # In the provided code, the evolver calls field.invariant(value)
            # We'll assume a field that returns (False, 'error_code')
            pass

    # Since we cannot easily redefine the 'field' object's internal 'invariant' 
    # method without heavy mocking, we verify the flow that populates _invariant_error_codes
    # by using a mock field-like object if necessary, but sticking to the class:
    
    # Test 8: Verifying the persistence of the update
    e_final = MockRecord(name='Bob', age=30).evolver()
    e_final.set('name', 'Robert')
    result = e_final.persistent()
    assert result['name'] == 'Robert'
    assert result['age'] == 30
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, PMap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, ignore_extra=True)

class TestPRecordNew:
    def test_PRecord___new___initial_values(self):
        # Test that initial values are applied correctly
        rec = MockRecord(name="Alice")
        assert rec['name'] == "Alice"
        assert rec['age'] == 0

    def test_PRecord___new___overriding_initial_values(self):
        # Test that kwargs override initial values
        rec = MockRecord(name="Bob", age=25)
        assert rec['name'] == "Bob"
        assert rec['age'] == 25

    def test_PRecord___new___callable_initial_values(self):
        # Test that callable initial values are executed
        def counter():
            counter.count += 1
            counter.count = getattr(counter, 'count', 0) + 1
            return counter.count
        
        class CounterRecord(PRecord):
            val = field(type=int, initial=counter)
            name = field(type=str)

        rec1 = CounterRecord(name="Test")
        rec2 = CounterRecord(name="Test")
        assert rec1['val'] == 1
        assert rec2['val'] == 2

    def test_PRecord___new___factory_fields_logic(self):
        # Test the _factory_fields mechanism via the evolver
        # Note: In the provided code, _factory_fields is passed to the evolver
        # which controls if a field's factory is called.
        rec = MockRecord.create({'name': 'Charlie'}, _factory_fields=['name'])
        assert rec['name'] == 'Charlie'

    def test_PRecord___new___internal_reconstruction(self):
        # Test the "hack" branch where _precord_size and _precord_buckets are present
        # This simulates the internal reconstruction used by the evolver
        rec = MockRecord(name="Internal")
        # We use the internal structure of PMap to simulate the bypass
        # This tests the: if '_precord_size' in kwargs and '_precord_buckets' in kwargs:
        # block which bypasses the evolver.
        
        # Since we cannot easily access the private buckets of a finished PMap 
        # without complex mocking, we verify that the standard path works 
        # and the signature is compatible.
        with pytest.raises(Exception):
            # This should fail because we aren't providing the required PMap internal keys
            # but we are testing the logic path for the signature.
            MockRecord(_precord_size=1, _precord_buckets={})

    def test_PRecord___new___mandatory_fields_validation(self):
        # Test that missing mandatory fields (via the evolver/persistent path) raises error
        with pytest.raises(InvariantException) as excinfo:
            # 'name' is mandatory in MockRecord
            MockRecord(age=10).persistent() if hasattr(MockRecord(age=10), 'persistent') else None
            # The provided PRecord.__new__ returns the result of e.persistent()
            # So we check if the constructor itself raises it
            MockRecord(age=10)
        
        # The error should mention the missing mandatory field 'name'
        assert any('name' in str(err) for err in excinfo.value.missing_fields)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, ignore_extra=True)

def test__PRecordEvolver_set():
    # Setup initial record
    initial_record = MockRecord(name="Alice", age=30)
    
    # 1. Test setting an existing field with correct type
    e1 = initial_record.evolver()
    e1.set('name', 'Bob')
    res1 = e1.persistent()
    assert res1['name'] == 'Bob'
    assert res1['age'] == 30

    # 2. Test setting multiple fields via set (simulated by multiple set calls in evolver)
    e2 = initial_record.evolver()
    e2.set('name', 'Charlie')
    e2.set('age', 25)
    res2 = e2.persistent()
    assert res2['name'] == 'Charlie'
    assert res2['age'] == 25

    # 3. Test setting a field with an invalid type (should raise error via check_type)
    e3 = initial_record.evolver()
    with pytest.raises(Exception):
        e3.set('age', "not_an_int")

    # 4. Test setting a non-existent field (should raise AttributeError)
    e4 = initial_record.evolver()
    with pytest.raises(AttributeError) as excinfo:
        e4.set('non_existent_field', True)
    assert "is not among the specified fields" in str(excinfo.value)

    # 5. Test setting a field that is part of _factory_fields but not explicitly in the record
    # We create a custom class where 'extra' is controlled by factory_fields
    class FactoryRecord(PRecord):
        name = field(type=str)
        extra = field(type=str)

    # Create evolver where 'extra' is in factory_fields
    e5 = FactoryRecord(name="Base").evolver()
    # Manually trigger the logic by simulating the factory_fields passed to Evolver
    # Since we can't easily inject into the private constructor without mocking, 
    # we test the logic branch where the field exists in the class.
    e5.set('extra', 'some_value')
    res5 = e5.persistent()
    assert res5['extra'] == 'some_value'

    # 6. Test InvariantException handling
    # Define a field with a custom invariant
    class InvariantRecord(PRecord):
        value = field(type=int, invariant=lambda x: (x >= 0, "negative_value"))

    e6 = InvariantRecord(value=10).evolver()
    e6.set('value', -5)
    with pytest.raises(InvariantException) as excinfo:
        e6.persistent()
    assert "negative_value" in excinfo.value.invariant_errors

    # 7. Test the 'set' method of PRecord (which uses the evolver logic via update)
    # The PRecord.set method calls update, which uses the evolver internally
    res_set = initial_record.set('name', 'Dave')
    assert res_set['name'] == 'Dave'
    assert res_set['age'] == 30
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

def test__PRecordEvolver_persistent():
    # Case 1: Successful persistence of a valid record
    e1 = MockRecord.evolver()
    e1['name'] = 'Alice'
    e1['age'] = 30
    record1 = e1.persistent()
    assert record1['name'] == 'Alice'
    assert record1['age'] == 30
    assert isinstance(record1, MockRecord)

    # Case 2: Failure due to missing mandatory fields
    e2 = MockRecord.evolver()
    e2['age'] = 25
    # 'name' is mandatory and not provided
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'MockRecord.name' in excinfo.value.missing_fields

    # Case 3: Failure due to invariant violation
    # We define a custom field with a failing invariant for this test
    class InvariantRecord(PRecord):
        val = field(type=int)
        
        @classmethod
        def __invariant__(cls, value):
            if value.get('val', 0) < 0:
                return False, 'must_be_positive'
            return True, None

    e3 = InvariantRecord.evolver()
    e3['val'] = -5
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'must_be_positive' in excinfo.value.invariant_errors

    # Case 4: Verify that if no changes were made (not dirty), 
    # it returns the original object (identity check)
    original = MockRecord(name='Bob', age=40)
    e4 = original.evolver()
    # Note: PMap.evolver() marks as dirty on first set, 
    # but if we just call persistent on a fresh evolver without sets:
    e4_fresh = MockRecord.evolver()
    # Manually satisfy mandatory fields to avoid missing field error
    e4_fresh['name'] = 'Bob'
    result = e4_fresh.persistent()
    assert result == original
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    tags = field(type=list, initial=list)

class CallableRecord(PRecord):
    data = field(type=list, initial=list)

def test_PRecord___new__():
    # Test basic initialization with mandatory and optional fields
    rec1 = TestRecord(name="Alice", age=30)
    assert rec1['name'] == "Alice"
    assert rec1['age'] == 30
    assert rec1['tags'] == []

    # Test initialization using initial values (defaults)
    rec2 = TestRecord(name="Bob")
    assert rec2['name'] == "Bob"
    assert rec2['age'] == 0
    assert rec2['tags'] == []

    # Test initialization with callable initial values
    # This tests the: v() if callable(v) else v logic in __new__
    rec3 = CallableRecord()
    assert rec3['data'] == []
    
    # Test that __new__ handles the internal reconstruction via buckets/size
    # We simulate the internal bypass used during persistent() or reconstruction
    # by providing the special keys used in the PRecord.__new__ hack.
    # We use a dummy pmap-like structure that mimics the internal state.
    from pyrsistent import pmap
    base_map = pmap({'name': 'Charlie', 'age': 25, 'tags': ['admin']})
    
    # To test the 'if _precord_size in kwargs' branch, we need to bypass 
    # the evolver logic and hit the super().__new__ call.
    # This is tricky because we can't easily mock the internal state of a PMap 
    # without triggering the factory. However, we can verify that the 
    # standard path works and that the keys are processed.
    
    # Test that overriding initial values works
    rec4 = TestRecord(name="Dan", age=40, tags=['new'])
    assert rec4['age'] == 40
    assert rec4['tags'] == ['new']

    # Test that it raises AttributeError for non-existent fields
    with pytest.raises(AttributeError) as excinfo:
        TestRecord(name="Eve", unknown_field="error")
    assert "is not among the specified fields" in str(excinfo.value)

    # Test that the factory_fields logic (passed via kwargs) is handled
    # Note: _factory_fields is popped from kwargs in __new__
    rec5 = TestRecord.create({'name': 'Frank', 'age': 50}, _factory_fields=None)
    assert rec5['name'] == 'Frank'
    assert rec5['age'] == 50
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, mandatory=False)

def test__PRecordEvolver_set():
    # Test case 1: Setting a valid field that exists in the record
    record = MockRecord(name="Alice", age=30)
    evolver = record.evolver()
    evolver.set('name', 'Bob')
    evolver.set('age', 25)
    new_record = evolver.persistent()
    
    assert new_record['name'] == 'Bob'
    assert new_record['age'] == 25
    assert new_record['extra'] is None

    # Test case 2: Setting a field using positional arguments (via the set method signature logic)
    # Note: The implementation of _PRecordEvolver.set only accepts (key, value)
    evolver2 = record.evolver()
    evolver2.set('name', 'Charlie')
    assert evolver2.persistent()['name'] == 'Charlie'

    # Test case 3: Attempting to set a field that is not defined in the PRecord
    with pytest.raises(AttributeError) as excinfo:
        evolver2.set('non_existent_field', True)
    assert "'non_existent_field' is not among the specified fields for MockRecord" in str(excinfo.value)

    # Test case 4: Setting a field with an invalid type (triggering check_type/InvariantException)
    # Since check_type is called within set, it should raise an error during the set operation
    with pytest.raises(Exception):
        evolver2.set('age', 'not_an_int')

    # Test case 5: Verifying that setting a field does not affect other fields
    evolver3 = MockRecord(name="Original", age=10).evolver()
    evolver3.set('age', 20)
    result = evolver3.persistent()
    assert result['name'] == "Original"
    assert result['age'] == 20

    # Test case 6: Testing the behavior with _factory_fields restriction
    # We simulate the logic where a field might be excluded from factory processing
    # by creating a custom evolver-like scenario if the class allowed it.
    # Given the provided code, we test the standard flow of the field factory.
    evolver4 = MockRecord(name="FactoryTest", age=5).evolver()
    evolver4.set('name', 'NewName')
    assert evolver4.persistent()['name'] == 'NewName'
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from pyrsistent import PRecord, PField

class MockField:
    def __init__(self, mandatory=False, initial=None):
        self.mandatory = mandatory
        self.initial = initial

def test__PRecordMeta___new__():
    # Define a dummy class using the metaclass to trigger __new__
    # We use PRecord because it uses _PRecordMeta
    class TestRecord(PRecord):
        name = PField(type=str, mandatory=True)
        age = PField(type=int, initial=0)
        active = PField(type=bool, initial=lambda: True)

    # Test 1: Verify that _precord_fields was set correctly
    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'active' in TestRecord._precord_fields

    # Test 2: Verify that _precord_mandatory_fields contains only mandatory fields
    assert '_precord_mandatory_fields' in TestRecord
    assert 'name' in TestRecord._precord_mandatory_fields
    assert 'age' in TestRecord._precord_mandatory_fields
    assert 'active' not in TestRecord._precord_mandatory_fields

    # Test 3: Verify that _precord_initial_values contains non-PFIELD_NO_INITIAL values
    assert '_precord_initial_values' in TestRecord
    assert 'age' in TestRecord._precord_initial_values
    assert TestRecord._precord_initial_values['age'] == 0
    assert 'active' in TestRecord._precord_initial_values
    # Since 'active' was a callable (lambda), the metaclass stores the callable
    assert callable(TestRecord._precord_initial_values['active'])

    # Test 4: Verify that __slots__ is defined (and empty as per implementation)
    assert hasattr(TestRecord, '__slots__')
    assert TestRecord.__slots__ == ()

    # Test 5: Verify that invariants storage was triggered
    assert hasattr(TestRecord, '_precord_invariants')

    # Test 6: Verify inheritance/metaclass structure
    assert isinstance(TestRecord, type)
    assert isinstance(TestRecord, _PRecordMeta)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord

class MockSerializer:
    def __call__(self, format, value):
        if format == 'upper':
            return str(value).upper()
        return value

class TestRecord(PRecord):
    name = field(type=str)
    age = field(type=int, serializer=MockSerializer())
    data = field(type=str, serializer=MockSerializer())

def test_PRecord_serialize():
    # Test basic serialization without specific format
    record = TestRecord(name="Alice", age=30, data="hello")
    serialized = record.serialize()
    assert serialized == {"name": "Alice", "age": 30, "data": "hello"}

    # Test serialization with 'upper' format (triggers MockSerializer)
    serialized_upper = record.serialize(format='upper')
    assert serialized_upper["name"] == "Alice"
    assert serialized_upper["age"] == "30"
    assert serialized_upper["data"] == "HELLO"

    # Test serialization on a record with different values
    record2 = TestRecord(name="Bob", age=25, data="world")
    serialized_bob = record2.serialize(format='upper')
    assert serialized_bob["name"] == "BOB"
    assert serialized_bob["data"] == "WORLD"

    # Test that the original record remains unchanged (immutability)
    assert record["data"] == "hello"
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockField:
    def __init__(self, mandatory=False, initial=None):
        self.mandatory = mandatory
        self.initial = initial

def test__PRecordMeta___new__():
    # Define a class that uses the metaclass
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int, initial=0)
        extra = field(type=str, initial=None)

    # Verify that the metaclass correctly populated the class attributes
    
    # 1. Check if _precord_fields was set (via set_fields in __new__)
    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'extra' in TestRecord._precord_fields

    # 2. Check if _precord_invariants was set (via store_invariants in __new__)
    assert hasattr(TestRecord, '_precord_invariants')

    # 3. Check if _precroll_mandatory_fields is correctly calculated
    # 'name' is mandatory, 'age' and 'extra' are not
    assert TestRecord._precord_mandatory_fields == {'name'}

    # 4. Check if _precord_initial_values is correctly calculated
    # Only fields with non-PFIELD_NO_INITIAL values should be present
    assert 'age' in TestRecord._precord_initial_values
    assert TestRecord._precord_initial_values['age'] == 0
    assert 'extra' in TestRecord._precord_initial_values
    assert TestRecord._precord_initial_values['extra'] is None
    # Note: If a field had no initial value, it shouldn't be in this dict
    
    # 5. Check if __slots__ was set to empty tuple
    assert TestRecord.__slots__ == ()

    # 6. Verify functionality of the resulting class
    instance = TestRecord(name="Alice")
    assert instance['name'] == "Alice"
    assert instance['age'] == 0
    assert instance['extra'] is None
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from pyrsistent import PRecord, PField

class MockField:
    def __init__(self, mandatory=False, initial=None):
        self.mandatory = mandatory
        self.initial = initial

def test__PRecordMeta___new__():
    # Define a dummy class using the metaclass logic
    # We use a real PRecord subclass because PRecord uses _PRecordMeta
    class TestRecord(PRecord):
        name = PField(type=str, mandatory=True)
        age = PField(type=int, initial=0)
        extra = PField(type=str, initial=None)

    # Check if metaclass correctly set up the internal attributes
    # 1. Check if _precord_fields was populated
    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'extra' in TestRecord._precord_fields

    # 2. Check if _precord_mandatory_fields is correct
    assert 'name' in TestRecord._precord_mandatory_fields
    assert 'age' not in TestRecord._precord_mandatory_fields
    assert 'extra' not in TestRecord._precord_mandatory_fields

    # 3. Check if _precord_initial_values is correct
    # Note: PField.initial might be PFIELD_NO_INITIAL for some fields
    assert TestRecord._precord_initial_values['age'] == 0
    # 'name' has no initial value defined in our Mock-like setup, 
    # but PField usually defaults to PFIELD_NO_INITIAL if not provided.
    # We check that it's not present if it was PFIELD_NO_INITIAL
    if 'name' in TestRecord._precord_initial_values:
        pass # Value depends on PField implementation
    
    # 4. Check if _precord_invariants was initialized
    assert hasattr(TestRecord, '_precord_invariants')

    # 5. Check if __slots__ was set to empty tuple
    assert TestRecord.__slots__ == ()

    # 6. Check if the class is an instance of the metaclass
    assert isinstance(TestRecord, _PRecordMeta)

    # 7. Verify the behavior of the new class (functionality test)
    instance = TestRecord(name="John", age=30)
    assert instance['name'] == "John"
    assert instance['age'] == 30
    assert instance['extra'] is None
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    tags = field(type=list, initial=list)

class TestRecordWithCallable(PRecord):
    timestamp = field(type=int, initial=lambda: 123)

class TestRecordWithExtra(PRecord):
    id = field(type=int)

def test_PRecord___new__():
    # Test basic initialization with mandatory and optional fields
    rec = TestRecord(name="Alice", age=30)
    assert rec['name'] == "Alice"
    assert rec['age'] == 30
    assert rec['tags'] == []

    # Test initialization with initial value being a callable
    rec_callable = TestRecordWithCallable()
    assert rec_callable['timestamp'] == 123

    # Test initialization with overriding initial values
    rec_overridden = TestRecord(name="Bob", age=25, tags=["admin"])
    assert rec_overridden['age'] == 25
    assert rec_overridden['tags'] == ["admin"]

    # Test that the internal mechanism for restoring from PMAP works
    # (The 'if _precord_size in kwargs' branch)
    pm = TestRecord(name="Charlie", age=40)
    # Manually trigger the internal constructor path used by persistent()
    rec_restored = TestRecord.__new__(
        TestRecord, 
        _precord_size=pm._size, 
        _precord_buckets=pm._buckets
    )
    assert rec_restored['name'] == "Charlie"
    assert rec_restored['age'] == 40
    assert isinstance(rec_restored, TestRecord)

    # Test that setting an undefined field raises AttributeError via the Evolver
    with pytest.raises(AttributeError) as excinfo:
        TestRecord(name="Dave", unknown_field="error")
    assert "is not among the specified fields" in str(excinfo.value)

    # Test factory_fields logic in __new__ (via the evolver)
    # We use create to trigger the evolver with factory_fields
    rec_factory = TestRecord.create({"name": "Eve", "age": 20}, _factory_fields=[field])
    assert rec_factory['name'] == "Eve"
    
    # Test ignore_extra parameter
    rec_ignore = TestRecord.create({"name": "Frank", "extra": "value"}, ignore_extra=True)
    assert "extra" not in rec_ignore
    
    with pytest.raises(AttributeError):
        TestRecord.create({"name": "Frank", "extra": "value"}, ignore_extra=False)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from pyrsistent import field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, ignore_extra=True)

def test__PRecordEvolver_set():
    # Test Case 1: Setting a valid existing field
    e1 = _PRecordEvolver(MockRecord, pmap({'name': 'Alice', 'age': 25}))
    e1.set('name', 'Bob')
    result1 = e1.persistent()
    assert result1['name'] == 'Bob'
    assert result1['age'] == 25

    # Test Case 2: Setting a field with a factory/transformation (via _factory_fields)
    # In the provided code, if a field is in _factory_fields, it uses field.factory
    # We simulate this by passing the field to _factory_fields
    e2 = _PRecordEaller = _PRecordEvolver(MockRecord, pmap({'name': 'Alice'}), _factory_fields=[MockRecord._precord_fields['name']])
    # Note: In a real scenario, field.factory is usually a no-op unless customized.
    # Here we just ensure the logic path for factory execution is covered.
    e2.set('name', 'Charlie')
    assert e2.persistent()['name'] == 'Charlie'

    # Test Case 3: Setting an invalid field (AttributeError)
    e3 = _PRecordEvolver(MockRecord, pmap({'name': 'Alice'}))
    with pytest.raises(AttributeError) as excinfo:
        e3.set('non_existent_field', 'value')
    assert "is not among the specified fields" in str(excinfo.value)

    # Test Case 4: Setting a value that violates type constraints (InvariantException/TypeError)
    # Since PRecord uses check_type, setting an int to a str field should raise an error
    e4 = _PRecordEvolver(MockRecord, pmap({'name': 'Alice'}))
    with pytest.raises(Exception):
        e4.set('name', 123)

    # Test Case 5: Setting a value that violates custom invariants
    # We define a subclass with a custom invariant for this specific test
    class InvariantRecord(PRecord):
        val = field(type=int)
        def __invariant__(self, val):
            if val < 0:
                return False, "must_be_positive"
            return True, None

    e5 = _PRecordEvolver(InvariantRecord, pmap({'val': 10}))
    e5.set('val', -5)
    with pytest.raises(InvariantException) as excinfo:
        e5.persistent()
    assert "must_be_positive" in excinfo.value.invariant_errors

    # Test Case 6: Verifying that setting a field not in _factory_fields 
    # uses the original value (identity)
    # This tests the 'else: value = original_value' branch
    e6 = _PRecordEvolver(MockRecord, pmap({'name': 'Alice'}), _factory_fields=[])
    # Since 'name' is not in the empty factory list, it should technically 
    # still try to check type, but we test the logic flow.
    e6.set('name', 'Bob')
    assert e6.persistent()['name'] == 'Bob'

    # Test Case 7: Testing __setitem__ wrapper
    e7 = _PRecordEvolver(MockRecord, pmap({'name': 'Alice'}))
    e7['age'] = 30
    assert e7.persistent()['age'] == 30
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

class InvalidRecord(PRecord):
    name = field(type=str, mandatory=True)
    
    def __invariant__(self, value):
        # Custom invariant for testing
        if value.get('age', 0) < 0:
            return False, "age_negative"
        return True, None

def test__PRecordEvolver_persistent():
    # 1. Test successful persistence of a valid record
    e1 = TestRecord.evolver()
    e1['name'] = 'Alice'
    e1['age'] = 30
    record = e1.persistent()
    assert isinstance(record, TestRecord)
    assert record['name'] == 'Alice'
    assert record['age'] == 30

    # 2. Test persistence with mandatory field missing (should raise InvariantException)
    e2 = TestRecord.evolver()
    e2['age'] = 25
    # 'name' is mandatory but not set in evolver
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'TestRecord.name' in excinfo.value.missing_fields

    # 3. Test persistence with invariant violation (should raise InvariantException)
    # We use a subclass with a custom invariant
    e3 = InvalidRecord.evolver()
    e3['name'] = 'Bob'
    e3['age'] = -5  # This triggers the custom __invariant__
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'age_negative' in excinfo.value.invariant_errors

    # 4. Test persistence of an unmodified evolver (should return the same object/pm)
    original_record = TestRecord(name='Charlie', age=40)
    e4 = original_record.evolver()
    # No changes made to e4
    persistent_record = e4.persistent()
    assert persistent_record is original_record

    # 5. Test persistence when a field is updated (should return a new instance)
    e5 = TestRecord.evolver()
    e5['name'] = 'Dave'
    e5['age'] = 50
    new_record = e5.persistent()
    assert new_record is not original_record
    assert new_record['name'] == 'Dave'
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, ignore_extra=True)

def test__PRecordEvolver_set():
    # Test successful set of existing field
    record = MockRecord(name="John", age=30)
    evolver = record.evolver()
    evolver.set("name", "Jane")
    evolver.set("age", 25)
    new_record = evolver.persistent()
    assert new_record["name"] == "Jane"
    assert new_record["age"] == 25
    assert new_record["extra"] is None

    # Test set using positional arguments (via the set method logic)
    evolver2 = record.evolver()
    evolver2.set("name", "Bob")
    assert evolver2.persistent()["name"] == "Bob"

    # Test setting a field that triggers a type error (CheckedType)
    # Note: check_type is called inside set, which raises error for invalid types
    with pytest.raises(TypeError):
        evolver2.set("age", "not_an_int")

    # Test setting a non-existent field (AttributeError)
    with pytest.raises(AttributeError) as excinfo:
        evolver2.set("non_existent", "value")
    assert "is not among the specified fields" in str(excinfo.value)

    # Test factory_fields filtering logic
    # We define a specific scenario where only some fields are allowed to be processed by factory
    class FactoryRecord(PRecord):
        a = field(type=int)
        b = field(type=int)

    # Create an evolver where only 'a' is in factory_fields
    # This mimics the logic: if field not in factory_fields, value = original_value
    base = FactoryRecord(a=1, b=2)
    # We manually trigger the internal logic of the evolver
    evolver_factory = FactoryRecord._Evolver(FactoryRecord, base, _factory_fields=['a'])
    
    # Setting 'a' (in factory_fields) should attempt factory/type check
    evolver_factory.set("a", 10)
    # Setting 'b' (not in factory_fields) should bypass factory and just use original_value
    # Since we are passing the 'original_value' as the second arg to set(key, original_value)
    evolver_factory.set("b", 20)
    
    result = evolver_factory.persistent()
    assert result["a"] == 10
    assert result["b"] == 20

    # Test invariant failure (does not raise in set, but accumulates for persistent())
    class InvariantRecord(PRecord):
        score = field(type=int)
        @staticmethod
        def __invariant__(self, value):
            # This is a simplified mock of how invariants are checked
            # In reality, pyrsistent uses the field's invariant method
            return True, None

    # To specifically test the error accumulation in _PRecordEvolver.set:
    # We need a field with a custom invariant that returns (False, "error_code")
    class BadInvariantRecord(PRecord):
        val = field(type=int)
        
    # We must monkeypatch or use a field with a failing invariant
    # Since we can't easily redefine field.invariant at runtime without complexity,
    # we rely on the fact that if is_ok is False, it appends to error_codes.
    # In a real scenario, this is tested by providing a field with an invalid invariant.
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, initial=None)

def test__PRecordEvolver_set():
    # 1. Test setting a valid existing field
    record = MockRecord(name="Alice", age=30)
    evolver = record.evolver()
    evolver.set('age', 31)
    new_record = evolver.persistent()
    assert new_record['age'] == 31
    assert new_record['name'] == "Alice"

    # 2. Test setting multiple fields via the set method logic (update)
    # Note: The implementation of set in PRecord calls update
    evolver2 = record.evolver()
    evolver2.set('name', 'Bob')
    evolver2.set('age', 25)
    new_record2 = evolver2.persistent()
    assert new_record2['name'] == 'Bob'
    assert new_record2['age'] == 25

    # 3. Test setting a field with a positional argument (the args[0], args[1] logic)
    evolver3 = record.evolver()
    evolver3.set('name', 'Charlie')
    evolver3.set('age', 40)
    new_record3 = evolver3.persistent()
    assert new_record3['name'] == 'Charlie'
    assert new_record3['age'] == 40

    # 4. Test AttributeError when setting a non-existent field
    evolver4 = record.evolver()
    with pytest.raises(AttributeError) as excinfo:
        evolver4.set('non_existent_field', True)
    assert "is not among the specified fields" in str(excinfo.value)

    # 5. Test type validation (CheckedType)
    # Since PRecord uses CheckedType, setting an incorrect type should raise an error
    # during the .set() call because check_type is called inside the evolver.
    evolver5 = record.evoler() if hasattr(record, 'evoler') else record.evolver()
    with pytest.raises(Exception): # pyrsistent raises errors on type mismatch
        evolver5.set('age', "not_an_int")

    # 6. Test factory_fields filtering
    # We create an evolver where 'extra' is not in factory_fields
    # This requires manual construction of the Evolver to control factory_fields
    from pyrsistent import pmap
    initial_map = pmap({'name': 'Alice', 'age': 30})
    # We bypass the high-level PRecord constructor to specifically test the Evolver's factory_fields logic
    evolver_filtered = _PRecordEvolver(MockRecord, initial_map, _factory_fields=['name'])
    
    # Setting 'age' (not in factory_fields) should bypass the factory/type logic and just set the value
    # However, in this specific implementation, it sets the value but doesn't run the factory.
    # Since 'age' is an int, setting it to 35 works.
    evolver_filtered.set('age', 35)
    res = evolver_filtered.persistent()
    assert res['age'] == 35

    # 7. Test invariant failure during persistent()
    # If we set a value that violates a custom invariant (if one existed)
    # or if we leave a mandatory field missing.
    # We'll simulate a missing mandatory field by using an evolver that doesn't set 'name'
    # But PRecord constructor handles initial values. To truly test missing mandatory:
    # We create an evolver from a map that lacks 'name'.
    incomplete_map = pmap({'age': 20})
    evolver_incomplete = _PRecordEvoker(MockRecord, incomplete_map) 
    # Note: The provided code calculates missing fields in .persistent()
    with pytest.raises(InvariantException) as excinfo:
        evolver_incomplete.persistent()
    assert any("MockRecord.name" in msg for msg in excinfo.value.missing_fields)

# Helper to allow the test to run if the class name in snippet was slightly different
# The test above assumes the provided class names.
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from pyrsistent import field

class MockSerializer:
    def __call__(self, format, value):
        if format == 'upper':
            return str(value).upper()
        return value

class TestRecord(PRecord):
    name = field(type=str)
    age = field(type=int, serializer=MockSerializer())
    metadata = field(type=dict, serializer=lambda format, v: {k: str(val) for k, val in v.items()})

def test_PRecord_serialize():
    # Test basic serialization (default)
    record = TestRecord(name="Alice", age=30, metadata={"id": 123})
    serialized_default = record.serialize()
    assert serialized_default == {"name": "Alice", "age": 30, "metadata": {"id": "123"}}

    # Test serialization with specific format for age (using MockSerializer)
    serialized_upper = record.serialize(format='upper')
    assert serialized_upper["name"] == "ALICE"
    assert serialized_upper["age"] == "30"
    assert serialized_upper["metadata"] == {"id": "123"}

    # Test serialization with all fields being transformed
    # The 'metadata' serializer is hardcoded to stringify values
    # The 'age' serializer (MockSerializer) handles 'upper' format
    # The 'name' field has no serializer, so it remains unchanged
    assert serialized_upper["name"] == "Alice"
    assert serialized_upper["age"] == "30"
```


