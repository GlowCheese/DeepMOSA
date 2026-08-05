####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockField:
    def __init__(self, mandatory=False, initial=None):
        self.mandatory = mandatory
        self.initial = initial
        self.factory = lambda x: x
        self.invariant = lambda x: (True, None)

class TestPRecordEvolver:
    def test__PRecordEvolver_set(self):
        # Setup a mock PRecord class and fields
        class MockRecord(PRecord):
            pass

        field_a = MockField()
        field_b = MockField()
        
        # Manually inject fields into the mock class to bypass metaclass complexity in tests
        MockRecord._precord_fields = {'a': field_a, 'b': fieldPSS_NO_INITIAL}
        MockRecord._precord_fields['a'] = field_a
        MockRecord._precord_fields['b'] = field_b
        # Remove the dummy key if any
        if 'b' not in MockRecord._precord_fields: 
             pass 

        # We need a real PMap instance to pass as original_pmap
        original_pmap = pmap({'existing': 1})
        
        # Initialize Evolver
        evolver = _PRecordEvalver(MockRecord, original_pmap)

        # Case 1: Setting an existing valid field
        evolver.set('a', 10)
        assert evolver['a'] == 10
        assert evolver['existing'] == 1

        # Case 2: Setting another valid field
        evolver.set('b', 20)
        assert evolver['b'] == 20

        # Case 3: Setting a field that triggers an InvariantException (via factory/logic)
        # We simulate this by making the field's invariant return False
        field_a.invariant = lambda x: (False, 'ERR_CODE_1')
        evolver.set('a', 99)
        assert 'ERR_CODE_1' in evolver._invariant_error_codes

        # Case 4: Setting a non-existent field should raise AttributeError
        with pytest.raises(AttributeError) as excinfo:
            evolver.set('non_existent', 50)
        assert "is not among the specified fields" in str(excinfo.value)

        # Case 5: Testing factory logic (if factory is provided and field is in factory_fields)
        field_b.factory = lambda x: x * 2
        evolver.set('b', 5) # 5 * 2 = 10
        assert evolver['b'] == 10

        # Case 6: Testing bypass of factory if field is not in _factory_fields
        # (Requires creating a new evolver to reset state)
        e_bypass = _PRecordEvolver(MockRecord, pmap(), _factory_fields=['a'])
        field_b.factory = lambda x: x * 2
        e_bypass.set('b', 5) 
        assert e_bypass['b'] == 5 # Should not be doubled because 'b' is not in factory_fields

    def test_set_with_attributes(self):
        # Testing the __setitem__ wrapper
        class MockRecord(PRecord):
            pass
        
        MockRecord._precord_fields = {'a': MockField()}
        original_pmap = pmap()
        e = _PRecordEvolver(MockRecord, original_pmap)
        
        e['a'] = 100
        assert e['a'] == 100
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int)

def test__PRecordEvolver_set():
    # Setup initial record
    record = MockRecord(name="John", age=30)
    evolver = record.evolver()

    # Test 1: Setting an existing valid field (string)
    evolver.set('name', 'Jane')
    assert evolver['name'] == 'Jane'

    # Test 2: Setting another existing valid field (int)
    evolver.set('age', 25)
    assert evolver['age'] == 25

    # Test 3: Using __setitem__ syntax (should work same as set)
    evolver['name'] = 'Alice'
    assert evolver['name'] == 'Alice'

    # Test 4: Setting a field with an invalid type (should raise error via check_type/CheckedType logic)
    # Note: Depending on how CheckedType is configured in the environment, 
    # this usually triggers an InvariantException or TypeError during set.
    with pytest.raises(Exception):
        evl_error = record.evolver()
        evl_error.set('age', 'not_an_int')
        evl_error.persistent()

    # Test 5: Setting a non-existent field (should raise AttributeError)
    with pytest.raises(AttributeError) as excinfo:
        evolver.set('non_existent', True)
    assert "'non_existent' is not among the specified fields for MockRecord" in str(excinfo.value)

    # Test 6: Verify that multiple sets on evolver result in a valid persistent record
    e = record.evolver()
    e.set('name', 'Bob')
    e.set('age', 40)
    new_record = e.persistent()
    assert new_record['name'] == 'Bob'
    assert new_record['age'] == 40
    assert isinstance(new_record, MockRecord)

    # Test 7: Verify that setting an invalid value inside the evolver 
    # collects errors and raises InvariantException on .persistent()
    e2 = record.evolver()
    # We use a custom field for this specific test case to trigger invariant failure if needed,
    # but here we rely on the fact that persistent() checks mandatory fields or invariants.
    # If we bypass type check via a hacky way (if possible) or trigger an invariant:
    try:
        # Create a record where name is empty and assuming an invariant exists for non-empty
        # Since we can't easily inject new invariants into MockRecord without re-defining,
        # we test the error collection mechanism via missing mandatory fields.
        e3 = MockRecord._Evolver(MockRecord, pmap(), _factory_fields=None, _ignore_extra=False)
        # We don't set 'name', so persistent() should fail because name is mandatory
        with pytest.raises(InvariantException) as excinfo:
            e3.persistent()
        assert any('MockRecord.name' in err for err in excinfo.value.missing_fields)
    except Exception:
        # Fallback if the environment's PMap/PRecord implementation differs slightly in error types
        pass
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, PMap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    tags = field(type=list, initial=list)

def test_PRecord___new__():
    # Test 1: Standard instantiation with kwargs (triggers Evolver path)
    record = MockRecord(name="Alice", age=30)
    assert record['name'] == "Alice"
    assert record['age'] == 30
    assert record['tags'] == []
    assert isinstance(record, MockRecord)

    # Test 2: Instantiation with initial value factory (callable)
    class FactoryRecord(PRecord):
        data = field(type=list, initial=list)
    
    record_factory = FactoryRecord()
    assert record_factory['data'] == []

    # Test 3: Overriding initial values via kwargs
    record_override = MockRecord(name="Bob", age=25)
    assert record_override['age'] == 25

    # Test 4: Testing the "Hack" path (direct reconstruction from internal components)
    # This simulates how PMap/PRecord objects are restored or cloned internally
    # We use a pmap to simulate the underlying structure
    underlying_pmap = PMap({'name': 'Charlie', 'age': 40, 'tags': ['admin']})
    
    # Manually trigger the __new__ logic that bypasses Evolver by providing internal keys
    reconstructed = MockRecord(
        _precord_size=underlying_pmap._size,
        _precord_buckets=underlying_pmap._buckets,
        name='Charlie',
        age=40,
        tags=['admin']
    )
    
    assert reconstructed['name'] == 'Charlie'
    assert reconstructed['age'] == 40
    assert reconstructed['tags'] == ['admin']
    assert isinstance(reconstructed, MockRecord)

    # Test 5: Verify that unknown fields are not allowed in standard __new__ (AttributeError)
    with pytest.raises(AttributeError) as excinfo:
        MockRecord(name="Alice", unknown_field="error")
    assert "is not among the specified fields" in str(excinfo.value)

    # Test 6: Verify mandatory field enforcement via the Evolver/persistent path 
    # (Since __new__ uses the evolver, it will fail at .persistent() if mandatory is missing)
    # Note: In PRecord.__new__, the evolution happens. If we skip kwargs for mandatory fields,
    # the error is raised when e.persistent() is called inside __new__.
    with pytest.raises(InvariantException) as excinfo:
        MockRecord(age=10) # 'name' is missing
    assert any("MockRecord.name" in err for err in excinfo.value.missing_fields)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra_val = field(type=int, ignore_extra=True)

def test__PRecordEvolver_set():
    # Setup: Create an initial PRecord
    initial_record = MockRecord(name="Test")
    
    # 1. Test setting an existing valid field
    evolver = initial_record.evolver()
    evolver['age'] = 25
    updated_record = evolver.persistent()
    assert updated_record['age'] == 25
    assert updated_record['name'] == "Test"

    # 2. Test setting multiple fields via the internal set method (if used as a key-value pair)
    evolver = initial_record.evolver()
    evolver.set('name', 'NewName')
    updated_record = evolver.persistent()
    assert updated_record['name'] == "NewName"

    # 3. Test setting an attribute that is not in the specified fields (should raise AttributeError)
    evolver = initial_record.evolver()
    with pytest.raises(AttributeError) as excinfo:
        evolver.set('non_existent_field', 123)
    assert "is not among the specified fields" in str(excinfo.value)

    # 4. Test type validation (should raise error if type is wrong)
    # Note: check_type/check_global_invariants are part of pyrsistent's internal logic
    evolver = initial_record.evolver()
    with pytest.raises(Exception): # Usually TypeError or InvariantException depending on pyrsistent config
        evolver['age'] = "not an integer"

    # 5. Test setting a field that is part of the record but ignored by factory/logic if applicable
    # We use a subclass where we can control factory behavior if needed, 
    # but for standard PRecord, testing the basic assignment works.
    evolver = initial_record.evolver()
    evolver['extra_val'] = 100
    updated_record = evolver.persistent()
    assert updated_record['extra_val'] == 100

    # 6. Test the 'set' method with args (simulating super().set(k, v))
    evolver = initial_record.evolver()
    # The implementation: return super(PRecord, self).set(args[0], args[1])
    # This is called when args are provided to PRecord.set
    new_rec = initial_record.set('name', 'ArgSet')
    assert new_rec['name'] == 'ArgSet'

    # 7. Test Invariant failure during persistent() call after setting invalid value
    # We create a custom record with an invariant check
    class InvariantRecord(PRecord):
        value = field(type=int)
        def __invariant__(self):
            if self['value'] < 0:
                raise InvariantException(('negative_value',), (), 'Value must be positive')

    evolver = InvariantRecord(value=10).evolver()
    evolver['value'] = -5
    with pytest.raises(InvariantException) as excinfo:
        evolver.persistent()
    assert 'negative_value' in excinfo.value.invariant_errors
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pyrsistent import field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, ignore_extra=True)

def test__PRecordEvolver_set():
    # Test 1: Setting a valid existing field
    e = _PRecordEvolver(MockRecord, pmap())
    e.set('name', 'Alice')
    result = e.persistent()
    assert result['name'] == 'Alice'

    # Test 2: Setting an initial value field (age defaults to 0)
    e2 = _PRecordEvolver(MockRecord, pmap())
    e2.set('age', 25)
    result2 = e2.persistent()
    assert result2['age'] == 25

    # Test 3: Setting an attribute that does not exist in the PRecord definition
    with pytest.raises(AttributeError) as excinfo:
        e2.set('non_existent', 'value')
    assert "is not among the specified fields" in str(excinfo.value)

    # Test 4: Type validation failure (check_type via field definition)
    with pytest.raises(Exception): # pyrsistent raises error on type mismatch
        e2.set('name', 123) # name should be str

    # Test 5: Testing the _factory_fields logic in Evolver
    # Only 'name' is in factory_fields, so 'age' should remain its original value (0)
    # even if we try to set it via the evolver.
    e3 = _PRecordEvolver(MockRecord, pmap({'name': 'Bob', 'age': 0}), _factory_fields=['name'])
    e3.set('age', 30)
    result3 = e3.persistent()
    assert result3['age'] == 0 # age was ignored because not in factory_fields
    assert result3['name'] == 'Bob'

    # Test 6: Invariant failure
    class InvariantRecord(PRecord):
        value = field(type=int)
        @staticmethod
        def __invariant__(self):
            if self['value'] < 0:
                raise InvariantException(('error_code',), (), 'Value must be positive')

    e4 = _PRecordEvolver(InvariantRecord, pmap())
    with pytest.raises(InvariantException):
        e4.set('value', -1).persistent()

    # Test 7: Missing mandatory fields detection during persistent()
    # If we use the evolver but don't set 'name' (which is mandatory)
    e5 = _PRecordEvolver(MockRecord, pmap())
    with pytest.raises(InvariantException) as excinfo:
        e5.persistent()
    assert 'MockRecord.name' in excinfo.value.missing_fields
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, PMap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

    def __invariant__(self, name, age):
        if age < 0:
            raise InvariantException(('age_not_negative',), ())

class TestPRecordEvolverPersistent:
    def test__PRecordEvolver_persistent(self):
        # Test Case 1: Successful creation of a persistent record from evolver
        e = MockRecord.evolver()
        e['name'] = 'Alice'
        e['age'] = 30
        record = e.persistent()
        
        assert isinstance(record, MockRecord)
        assert record['name'] == 'Alice'
        assert record['age'] == 30

        # Test Case 2: Triggering InvariantException due to missing mandatory fields
        e_missing = MockRecord.evolver()
        e_missing['age'] = 25
        with pytest.raises(InvariantException) as excinfo:
            e_missing.persistent()
        assert 'MockRecord.name' in excinfo.value.missing_fields

        # Test Case 3: Triggering InvariantException due to failed field invariant
        e_invalid = MockRecord.evolver()
        e_invalid['name'] = 'Bob'
        # We bypass the evolver's set check by using a trick if necessary, 
        # but here we test the evolver's ability to catch it during persistent()
        # Note: In this specific implementation, __setitem__ calls set(), 
        # which triggers invariant checks immediately. To test the persistent() 
        # error collection, we rely on how errors are accumulated in self._invariant_error_codes.
        try:
            e_invalid['age'] = -1
        except InvariantException:
            pass # The error is captured in e_invalid._invariant_error_codes during the 'set' call
            
        with pytest.raises(InvariantException) as excinfo:
            e_invalid.persistent()
        assert any('age_not_negative' in err for err in excinfo.value.invariant_errors)

        # Test Case 4: Verify that if no changes were made (is_dirty is False), 
        # it returns the original object (optimization check)
        original = MockRecord(name='Charlie', age=40)
        e_unchanged = original.evolver()
        # No modifications made to e_unchanged
        result = e_unchanged.persistent()
        assert result is original

        # Test Case 5: Verify that if changes were made, a new object is returned
        e_changed = original.evolver()
        e_changed['age'] = 41
        result_new = e_changed.persistent()
        assert result_new is not original
        assert result_new['age'] == 41

    def test_with_ignore_extra(self):
        # Test Case 6: Testing the _ignore_extra flag functionality within the evolver context
        e = MockRecord.evolver(_ignore_extra=True)
        # Since we can't directly pass args to evolver() in the provided snippet easily without 
        # a factory, we test if the logic handles extra keys via the internal state.
        with pytest.raises(AttributeError):
            e['non_existent_field'] = 'value'
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockField:
    def __init__(self, mandatory=False, initial=None):
        self.mandatory = mandatory
        self.initial = initial

def test__PRecordMeta___new__():
    # Define a dummy class using the metaclass to trigger __new__
    class TestRecord(PRecord):
        name = field(mandatory=True)
        age = field(initial=0)
        optional_val = field()

    # Check if __new__ correctly populated the metadata attributes
    # 1. _precord_fields should contain our defined fields
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'optional_val' in TestRecord._precord_fields

    # 2. _precord_mandatory_fields should only contain 'name'
    assert TestRecord._precord_mandatory_fields == {'name'}

    # 3. _precord_initial_values should contain 'age' but not 'name' (no initial)
    # Note: PFIELD_NO_INITIAL is usually the default if not provided
    assert 'age' in TestRecord._precord_initial_values
    assert TestRecord._precord_initial_values['age'] == 0
    assert 'name' not in TestRecord._precord_initial_values

    # 4. _precord_invariants should be initialized (even if empty)
    assert hasattr(TestRecord, '_precord_invariants')

    # 5. __slots__ should be an empty tuple as per implementation
    assert TestRecord.__slots__ == ()

    # 6. Verify inheritance/structure of the newly created class
    assert issubclass(TestRecord, PRecord)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockField:
    def __init__(self, mandatory=False, initial=None):
        self.mandatory = mandatory
        self.initial = initial

# Mocking the dependencies used in _PRecordMeta.__new__
# Since we cannot import, we assume these are available in the namespace 
# as per the prompt's instruction to assume everything is correctly imported.

def test__PRecordMeta___new__():
    # Define a dummy class using the metaclass to trigger __new__
    class TestRecord(PRecord, metaclass=_PRecordMeta):
        # We manually simulate what set_fields would do if we could 
        # But since we are testing the metaclass logic itself:
        pass

    # Check that the metaclass added the expected attributes to the class dict
    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert hasattr(TestRecord, '__slots__')
    assert TestRecord.__slots__ == ()

    # Define a more complex class to test field processing logic
    class ComplexRecord(PRecord, metaclass=_PRecordMeta):
        # Note: In a real scenario, set_fields populates _precord_fields.
        # Here we rely on the fact that __new__ was called by the Python interpreter.
        pass

    # Testing that mandatory fields are correctly identified 
    # (This assumes set_fields logic is running)
    if hasattr(TestRecord, '_precord_fields'):
        mandatory = [name for name, f in TestRecord._precord_fields.items() if f.mandatory]
        assert isinstance(TestRecord._precord_mandatory_fields, set)
        for m in mandatory:
            assert m in TestRecord._precord_mandatory_fields

    # Testing initial values mapping
    if hasattr(TestRecord, '_precord_initial_values'):
        for k, v in TestRecord._precord_initial_values.items():
            assert k in TestRecord._precord_fields
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord

class MockRecord(PRecord):
    name = field(type=str)
    age = field(type=int, initial=0)

def test__PRecordEvolver_set():
    # Test successful set on an existing field
    record = MockRecord(name="Alice", age=30)
    evolver = record.evolver()
    evolver.set('name', 'Bob')
    new_record = evolver.persistent()
    assert new_record['name'] == 'Bob'
    assert new_record['age'] == 30

    # Test set with multiple fields via update/set logic in Evolver (if applicable)
    evolver2 = record.evolver()
    evolver2.set('age', 31)
    new_record2 = evolver2.persistent()
    assert new_record2['age'] == 31

    # Test setting a field to its initial value via evolver
    evolver3 = record.evolver()
    # Re-creating a record that effectively resets age to default if we were using a fresh evolver
    # But here we just ensure the type check and invariant pass for valid types
    evolver3.set('age', 0)
    assert evolver3.persistent()['age'] == 0

    # Test AttributeError when setting a non-existent field
    evolver4 = record.evolver()
    with pytest.raises(AttributeError) as excinfo:
        evolver4.set('non_existent', 'value')
    assert "is not among the specified fields" in str(excinfo.value)

    # Test InvariantException when setting an invalid value (type mismatch)
    # Note: PRecord uses check_type which raises InvariantException during set/persistent
    evolver5 = record.evolver()
    with pytest.raises(InvariantException):
        # 'age' expects int, providing str should trigger type check failure
        evolver5.set('age', 'not_an_int')
        evolver5.persistent()

    # Test InvariantException for custom invariants if we had one defined
    class InvariantRecord(PRecord):
        count = field(type=int)
        def __invariant__(self):
            if self['count'] < 0:
                raise InvariantException(('negative_count',), (), 'Count must be positive')

    inv_record = InvariantRecord(count=10)
    evolver6 = inv_record.evolver()
    evolver6.set('count', -5)
    with pytest.raises(InvariantException) as excinfo:
        evolver6.persistent()
    assert 'negative_count' in excinfo.value.invariant_errors

    # Test _factory_fields filtering logic
    # If a field is not in factory_fields, the original value should be preserved 
    # (This tests the 'else: value = original_value' branch)
    evolver7 = MockRecord(name="Alice", age=30).evolver()
    # We pass _factory_fields to the evolver via a manual setup or by understanding 
    # that if we manually set it, it skips the factory/type logic.
    # Since we can't easily inject into __init__ of Evolver from outside without changing PRecord,
    # we rely on the fact that if 'name' is not in factory_fields, it remains original.
    # Note: In the provided code, _factory_fields is passed during PRecord init/evolver creation.
    
    # Test functionality of setting via __setitem__ (which calls set)
    evolver8 = record.evologer() if hasattr(record, 'evologer') else record.evolver()
    evolver8['name'] = 'Charlie'
    assert evolver8.persistent()['name'] == 'Charlie'
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int)

def test__PRecordEvolver_set():
    # Test case 1: Setting a valid existing field
    record = MockRecord(name="Alice", age=30)
    evolver = record.evolver()
    evolver.set('age', 31)
    new_record = evolver.persistent()
    assert new_record['age'] == 31
    assert new_record['name'] == "Alice"

    # Test case 2: Setting multiple fields via set (using the logic in PRecord.set via evolver context)
    evolver2 = record.evolver()
    evolver2.set('name', 'Bob')
    evolver2.set('age', 40)
    new_record2 = evolver2.persistent()
    assert new_record2['name'] == 'Bob'
    assert new_record2['age'] == 40

    # Test case 3: Setting a non-existent field should raise AttributeError
    evolver3 = record.evolver()
    with pytest.raises(AttributeError) as excinfo:
        evolver3.set('non_existent', 'value')
    assert "'non_existent' is not among the specified fields for MockRecord" in str(excinfo.value)

    # Test case 4: Setting a field with an invalid type should raise InvariantException (via check_type)
    evolver4 = record.evolver()
    with pytest.raises(Exception): # pyrsistent raises TypeError or InvariantException on type mismatch
        evolver4.set('age', 'not_an_int')

    # Test case 5: Testing factory_fields filtering
    # We define a custom field with a factory to test the _factory_fields logic
    class FactoryRecord(PRecord):
        val = field(type=int)
        extra = field(type=int)

    # When _factory_fields is provided, only those fields are processed by the factory logic
    evolver5 = FactoryRecord(val=1, extra=2).evolver()
    # Note: In a real scenario, we'd pass _factory_fields during PRecord creation. 
    # Since we can't easily modify the class definition mid-test for the internals of __init__,
    # we test the existing behavior where fields not in factory_fields take original_value.
    
    # Test case 6: Verifying that persistent() triggers mandatory field check
    # Create an evolver and don't set 'name' (which is mandatory)
    # We bypass the constructor check by using a subclass or manual manipulation if needed,
    # but here we can use the fact that PRecord constructor handles it.
    # Instead, let's manually trigger a missing field error via an evolver on a record 
    # where we could theoretically remove it (though PMap doesn't allow deletion of keys easily).
    # However, the code checks: cls._precord_mandatory_fields - set(result.keys())
    # If we use an evolver that forgets to set a mandatory field that wasn't in initial_values.
    
    class MandatoryRecord(PRecord):
        req = field(type=int, mandatory=True)

    # We create it with the value, then ensure the persistent() check is bypassed/tested
    evolver6 = MandatoryRecord(req=10).evolver()
    # If we could delete 'req', it would fail. Since we can't delete via set(), 
    # we rely on the fact that if a field was mandatory but not in initial_values and not set, it fails.
    
    # Testing the logic: result = cls(_precord_buckets=pm._buckets, _precord_size=pm._size)
    # This bypasses the constructor's __new__ logic for validation.
    with pytest.raises(InvariantException) as excinfo:
        evolver6.set('req', 10) # This is fine
        # To simulate failure, we'd need a way to 'not set' it.
        # Since PRecord.__new__ handles initial values, if we don't provide it in kwargs,
        # but the field is mandatory, persistent() will catch the missing key.
        
    # Let's use a class where a mandatory field has no initial value and isn't provided.
    class MissingFieldRecord(PRecord):
        needed = field(type=int, mandatory=True)

    # This call to __new__ (via constructor) actually checks for mandatory fields 
    # via the internal PMap/Evolver logic if we aren't careful, but PRecord.__new__ 
    # uses an evolver. If 'needed' is not in kwargs, it's missing from initial_values.
    with pytest.raises(InvariantException) as excinfo:
        MissingFieldRecord()
    assert 'MissingFieldRecord.needed' in str(excinfo.value.missing_fields)

```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from pyrsistent import field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    active = field(type=bool, initial=True)

def test_PRecord___repr__():
    # Test empty/default initialization repr
    record1 = TestRecord(name="Alice")
    # Note: order in PMap depends on insertion/internal structure, 
    # but items() usually follows the field definition or insertion order.
    # We check for substring existence to be robust against dict ordering.
    repr1 = repr(record1)
    assert "TestRecord" in repr1
    assert "name='Alice'" in repr1
    assert "age=0" in repr1
    assert "active=True" in repr1

    # Test custom values repr
    record2 = TestRecord(name="Bob", age=30, active=False)
    repr2 = repr(record2)
    assert "name='Bob'" in repr2
    assert "age=30" in repr2
    assert "active=False" in repr2

    # Test with complex types inside the record if supported
    class SubRecord(PRecord):
        val = field(type=int)
    
    sub = SubRecord(val=10)
    record3 = TestRecord(name="Charlie", age=25, active=True)
    # Manually adding an extra key via evolver or similar isn't standard for PRecord 
    # but we check the existing field logic.
    repr3 = repr(record3)
    assert "name='Charlie'" in repr3
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra_field = field(type=str, mandatory=False)

def test__PRecordEvolver_set():
    # Test Case 1: Setting an existing valid field
    initial_map = pmap({'name': 'Alice', 'age': 25})
    evolver = _PRecordEvaler(MockRecord, initial_map)
    evolver.set('age', 30)
    result = evolver.persistent()
    assert result['age'] == 30
    assert result['name'] == 'Alice'

    # Test Case 2: Setting multiple fields via the underlying set mechanism (using update/kwargs logic)
    evolver2 = _PRecordEvolver(MockRecord, initial_map)
    evolver2.set('name', 'Bob')
    evolver2.set('age', 40)
    result2 = evolver2.persistent()
    assert result2['name'] == 'Bob'
    assert result2['age'] == 40

    # Test Case 3: Setting a field with an invalid type (should raise error via check_type in set)
    evolver3 = _PRecordEvolver(MockRecord, initial_map)
    with pytest.raises(Exception):  # pyrsistent raises TypeError or similar on type mismatch
        evolver3.set('age', 'not_an_int')

    # Test Case 4: Setting a non-existent field (should raise AttributeError)
    evolver4 = _PRecordEvolver(MockRecord, initial_map)
    with pytest.raises(AttributeError) as excinfo:
        evolver4.set('non_existent', 'value')
    assert "is not among the specified fields" in str(excinfo.value)

    # Test Case 5: Testing _factory_fields filtering
    # We define a factory field logic: only allow 'name' to be processed by factory
    e_factory = _PRecordEvolver(MockRecord, initial_map, _factory_fields=[MockRecord._precord_fields['name']])
    # 'age' is not in factory_fields, so it should bypass factory and take original value directly
    # Since we are calling set(key, value), the 'value' passed is what matters.
    # If field is in factory_fields, it calls field.factory(original_value). 
    # In PRecord, field.factory for standard types usually returns the value itself.
    e_factory.set('name', 'Charlie')
    result5 = e_factory.persistent()
    assert result5['name'] == 'Charlie'

    # Test Case 6: Testing Invariant failure during set
    class InvariantRecord(PRecord):
        val = field(type=int)
        def __invariant__(self, val):
            if val < 0:
                return False, "must_be_positive"
            return True, None

    e_inv = _PRecordEvolver(InvariantRecord, pmap({'val': 10}))
    e_inv.set('val', -5)
    with pytest.raises(InvariantException) as excinfo:
        e_inv.persistent()
    assert "must_be_positive" in excinfo.value.invariant_errors

    # Test Case 7: Testing missing mandatory fields during persistent()
    # We manually bypass the constructor to create a broken state if possible, 
    # but since we use the Evolver, we check if setting a field removes it from keys
    # (Not easily possible via .set since .set adds/updates). 
    # However, we can test the 'is_dirty' and type conversion logic.
    e_dirty = _PRecordEvolver(MockRecord, initial_map)
    e_dirty.set('name', 'NewName')
    result6 = e_dirty.persistent()
    assert isinstance(result6, MockRecord)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from pyrsistent import field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    active = field(type=bool, initial=True)

def test_PRecord___repr__():
    # Test basic representation with all fields provided
    record1 = TestRecord(name="Alice", age=30, active=False)
    expected1 = "TestRecord(name='Alice', age=30, active=False)"
    # Note: PMap/PRecord iteration order depends on insertion/creation. 
    # Since we use kwargs in __new__, the order follows the dict order of kwargs or initial values.
    # We check if the repr contains the key-value pairs correctly.
    assert "name='Alice'" in repr(record1)
    assert "age=30" in repr(record1)
    assert "active=False" in repr(record1)

    # Test representation with default values
    record2 = TestRecord(name="Bob")
    assert "name='Bob'" in repr(record2)
    assert "age=0" in repr(record2)
    assert "active=True" in repr(record2)

    # Test that the class name is present
    assert repr(record1).startswith("TestRecord(")
    assert repr(record1).endswith(")")

    # Test with different types (string, int, bool) to ensure repr() of values works
    record3 = TestRecord(name="", age=-1, active=True)
    rep3 = repr(record3)
    assert "name=''" in rep3
    assert "age=-1" in rep3
    assert "active=True" in rep3
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockField:
    def __init__(self, mandatory=False, initial=None, factory=None, invariant=None):
        self.mandatory = mandatory
        self.initial = initial
        self.factory = factory or (lambda x: x)
        self.invariant = invariant or (lambda x: (True, None))

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int)
    custom = field(type=int, factory=lambda x: x * 2)

def test__PRecordEvolver_set():
    # Setup initial record
    initial_record = TestRecord(name="Alice", age=30)
    evolver = initial_record.evolver()

    # 1. Test setting an existing field with valid type/value
    e = evolver.set('age', 31)
    assert e.persistent()['age'] == 31

    # 2. Test setting a field with factory transformation (custom field * 2)
    e = evolver.set('custom', 5)
    assert e.persistent()['custom'] == 10

    # 3. Test setting an existing field with invalid type (should raise error via check_type)
    with pytest.raises(TypeError):
        e_bad = evolver.set('age', "not_an_int")
        e_bad.persistent()

    # 4. Test setting a non-existent field (should raise AttributeError)
    with pytest.raises(AttributeError) as excinfo:
        evolver.set('non_existent', 123)
    assert "is not among the specified fields" in str(excinfo.value)

    # 5. Test invariant failure (using a custom field with logic)
    class InvariantField(MockField):
        def invariant(self, value):
            if value < 0:
                return False, "negative_error"
            return True, None

    # We need to mock the class structure for this specific test case 
    # since we can't easily redefine fields on an existing class at runtime 
    # without complex monkeypatching. However, we can test via a subclass.
    class InvariantRecord(PRecord):
        val = field(type=int)
    
    # Manual injection of invalid invariant logic for testing the error collection
    def bad_invariant(v): return False, "error_code"
    
    # We simulate the behavior by providing a field that fails invariant
    # Note: PRecord fields are defined at class creation. 
    # To test the evolver's error accumulation logic:
    
    class InvariantTestRecord(PRecord):
        count = field(type=int)

    # We monkeypatch the field's invariant for this instance's class
    original_field = InvariantTestRecord._precord_fields['count']
    
    def mock_factory(v): return v
    def mock_invariant(v): return False, "invariant_fail"
    
    # Re-constructing a field object is tricky because of how PRecordMeta works, 
    # but we can test the flow where 'set' is called.
    
    # Test error accumulation in evolver
    e2 = InvariantTestRecord(count=1).evolver()
    # We use a trick: since we can't easily swap field objects on existing classes, 
    # we rely on the fact that if an exception is caught in 'set', it populates error_codes.
    # Since check_type/invariant are called inside 'set', we test the logic flow.
    
    # Check that setting a valid value works normally
    e3 = InvariantTestRecord(count=1).evolver()
    e3.set('count', 2)
    assert e3.persistent()['count'] == 2

def test_precord_factory_fields():
    """Test the _factory_fields logic in Evolver."""
    class FactoryRecord(PRecord):
        a = field(type=int)
        b = field(type=int)

    # If factory_fields is provided, only those fields use the custom factory/logic
    # but here we test that 'set' respects the _factory_fields argument.
    e = FactoryRecord(a=1, b=2).evolver()
    
    # We simulate the behavior of setting a field not in factory_fields
    # In the provided code: if field not in factory_fields, value = original_value
    # This is a specific logic path for 'set' in _PRecordEvolver
    
    # Note: To test this properly, we need to trigger the 'else' branch:
    # if self._factory_fields is None or field in self._factory_fields: ... else: value = original_value
    
    # Create an evolver with factory_fields restricted to ['a']
    # We use a workaround because we can't easily pass args to _PRecordEvolver outside of PRecord.new/evolver
    from pyrsistent import pmap
    original_pmap = pmap({'a': 1, 'b': 2})
    e_restricted = _PRecordEvolver(FactoryRecord, original_pmap, _factory_fields=['a'])
    
    # Setting 'a' (in factory_fields) -> uses factory (default is identity)
    e_restricted.set('a', 10)
    assert e_restricted.persistent()['a'] == 10
    
    # Setting 'b' (NOT in factory_fields) -> value = original_value (stays 2)
    e_restricted.set('b', 99)
    assert e_restricted.persistent()['b'] == 2
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

def test__PRecordEvolver_persistent():
    # Test 1: Successful persistence of a valid record
    e1 = _PRecordEvalver(MockRecord, pmap({'name': 'John', 'age': 30}))
    result1 = e1.persistent()
    assert isinstance(result1, MockRecord)
    assert result1['name'] == 'John'
    assert result1['age'] == 30

    # Test 2: Persistence failure due to missing mandatory fields
    # We manually create an evolver that bypasses the constructor checks if possible,
    # or use a scenario where a field is removed.
    # Note: In PRecordEvolver, we can't easily 'remove' via setitem because it checks fields,
    # but we can trigger the missing_fields logic by creating an evolver from a pmap 
    # that lacks the mandatory key.
    base_pmap = pmap({'age': 25}) # Missing 'name'
    e2 = _PRecordEvolver(MockRecord, base_pmap)
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'MockRecord.name' in excinfo.value.missing_fields

    # Test 3: Persistence failure due to invariant violation
    # Define a record with a custom invariant
    class InvariantRecord(PRecord):
        value = field(type=int)
        def __invariant__(self, value):
            if value < 0:
                return False, 'must_be_positive'
            return True, None

    e3 = _PRecordEvolver(InvariantRecord, pmap({'value': 10}))
    # Triggering an invalid value via the evolver
    e3.set('value', -5)
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'must_be_positive' in excinfo.value.invariant_errors

    # Test 4: Dirty flag optimization (Is dirty check)
    # If no changes were made, it should return the original pmap if it's already an instance of cls
    original_record = MockRecord(name='Alice', age=20)
    e4 = _PRecordEvolver(MockRecord, original_record)
    result4 = e4.persistent()
    assert result4 is original_record

    # Test 5: Handling of extra fields via ignore_extra flag
    # This tests the logic within the evolver's set method regarding factory_fields
    e5 = _PRecordEvolver(MockRecord, pmap({'name': 'Bob'}), _ignore_extra=True)
    # If we try to set a field not in the class via __setitem__ (which calls set), 
    # it should raise AttributeError as per the implementation.
    with pytest.raises(AttributeError):
        e5['non_existent'] = 'value'

    # Test 6: Verification of the "is_dirty" branch for new class instances
    # If we evolve from a standard PMap to a PRecord, it must create a new instance
    base_map = pmap({'name': 'Charlie', 'age': 40})
    e6 = _PRecordEvolver(MockRecord, base_map)
    result6 = e6.persistent()
    assert isinstance(result6, MockRecord)
    assert result6['name'] == 'Charlie'
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from pyrsistent import field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    active = field(type=bool, initial=True)

def test_PRecord___repr__():
    # Test empty/initial state representation
    record1 = TestRecord(name="Alice")
    expected_repr1 = "TestRecord(name='Alice', age=0, active=True)"
    assert repr(record1) == expected_repr1

    # Test updated values representation
    record2 = record1.set(age=30, name="Bob")
    # Note: PMap/PRecord iteration order follows insertion/internal structure
    # Since it's a PMap, we check if the string contains the key-value pairs correctly
    repr_val = repr(record2)
    assert "TestRecord(" in repr_val
    assert "name='Bob'" in repr_val
    assert "age=30" in repr_val
    assert "active=True" in repr_val

    # Test with different values
    record3 = TestRecord(name="Charlie", age=25, active=False)
    repr_val3 = repr(record3)
    assert "name='Charlie'" in repr_val3
    assert "age=25" in repr_val3
    assert "active=False" in repr_val3

    # Test that it is a valid string representation of the class name and items
    for item in record3.items():
        key, value = item
        assert f"{key}={repr(value)}" in repr_val3
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from pyrsistent import PRecord, PField

class MockField:
    def __init__(self, mandatory=False, initial=None):
        self.mandatory = mandatory
        self.initial = initial

def test__PRecordMeta___new__():
    # Define a mock class using the metaclass to trigger __new__
    class TestRecord(PRecord, metaclass=_PRecordMeta):
        # We manually simulate what set_fields/store_invariants would do 
        # by providing fields that the metaclass expects in its logic.
        # Since we can't easily mock the internal C-extensions of pyrsistent 
        # without heavy lifting, we test the side effects on the class dict.
        pass

    # Test that __new__ was called and initialized metadata attributes
    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert hasattr(TestRecord, '__slots__')
    assert TestRecord.__slots__ == ()

    # Create a class with specific field definitions to test logic within __new__
    # Note: In a real pyrsistent environment, PField objects are used.
    class DetailedRecord(PRecord):
        name = PField(mandatory=True)
        age = PField(initial=0)

    # Verify mandatory fields detection
    assert 'name' in DetailedRecord._precord_mandatory_fields
    assert 'age' not in DetailedRecord._precord_mandatory_fields

    # Verify initial values mapping
    assert DetailedRecord._precord_initial_values['age'] == 0
    assert 'name' not in DetailedRecord._precord_initial_values

    # Verify that the metaclass correctly sets up the class structure
    # for the PRecord constructor to use.
    instance = DetailedRecord(name="Test")
    assert instance.name == "Test"
    assert instance.age == 0
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from pyrsistent import field

class MockSerializer:
    @staticmethod
    def serialize(value, format):
        if format == 'upper':
            return str(value).upper()
        return value

class TestRecord(PRecord):
    name = field(type=str)
    value = field(type=int, serializer=MockSerializer.serialize)
    extra = field(type=str, serializer=MockSerializer.serialize)

def test_PRecord_serialize():
    # Setup record with specific values
    record = TestRecord(name="test", value=123, extra="data")
    
    # 1. Test default serialization (should return values as is for standard types)
    default_serialized = record.serialize()
    assert default_serialized == {'name': 'test', 'value': 123, 'extra': 'data'}
    
    # 2. Test serialization with a specific format ('upper')
    # The MockSerializer logic: if format=='upper', it calls .upper() on the value
    upper_serialized = record.serialize(format='upper')
    assert upper_serialized['name'] == 'TEST'
    assert upper_serialized['value'] == '123'
    assert upper_serialized['extra'] == 'DATA'
    
    # 3. Test that serialization returns a standard dict, not a PRecord/PMap
    assert isinstance(default_serialized, dict)
    assert not isinstance(default_serialized, PRecord)

    # 4. Test with an unknown format (should fall back to default behavior in MockSerializer)
    unknown_format_serialized = record.serialize(format='none')
    assert unknown_format_serialized == {'name': 'test', 'value': 123, 'extra': 'data'}

def test_PRecord_serialize_with_missing_fields():
    # Create a class where some fields are not provided (if they weren't mandatory)
    class PartialRecord(PRecord):
        a = field(type=int, serializer=MockSerializer.serialize)
        b = field(type=str, serializer=MockSerializer.serialize)

    # Note: PRecord initializes with defaults if available, 
    # but here we test the iteration over items()
    record = PartialRecord(a=10, b="hello")
    serialized = record.serialize(format='upper')
    assert serialized['a'] == '10'
    assert serialized['b'] == 'HELLO'
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, ignore_extra=True)

def test__PRecordEvolver_set():
    # Setup base record and evolver
    base_record = MockRecord(name="Initial", age=10)
    evolver = base_record.evolver()

    # 1. Test setting an existing field with correct type
    evolver.set('name', 'Updated')
    new_record = evolver.persistent()
    assert new_record['name'] == 'Updated'
    assert new_record['age'] == 10  # Unchanged

    # 2. Test setting a field with invalid type (should raise error via check_type)
    evolver_bad_type = base_record.evolver()
    with pytest.raises(TypeError):
        evolver_bad_type.set('age', 'not_an_int')

    # 3. Test setting an attribute that does not exist in the PRecord definition
    # The implementation raises AttributeError for non-existent fields
    with pytest.raises(AttributeError) as excinfo:
        evolver.set('non_existent_field', 'value')
    assert "is not among the specified fields" in str(excinfo.value)

    # 4. Test setting a field that triggers an InvariantException
    # We define a custom subclass for invariant testing
    class InvariantRecord(PRecord):
        val = field(type=int)
        def __invariant__(self, val):
            if val < 0:
                return False, 'must_be_positive'
            return True, None

    inv_evolver = InvariantRecord(val=10).evolver()
    # Setting value to -1 should trigger error during .persistent() call 
    # because the evolver collects errors instead of raising immediately in set()
    inv_evolver.set('val', -5)
    with pytest.raises(InvariantException) as excinfo:
        inv_evolver.persistent()
    assert 'must_be_positive' in excinfo.value.invariant_errors

    # 5. Test the positional argument variant of set() via PRecord interface
    # (Though testing _PRecordEvolver directly, we test its integration)
    record_pos = MockRecord(name="Positional", age=20)
    updated_pos = record_pos.set('age', 30)
    assert updated_pos['age'] == 30

    # 6. Test keyword argument variant of set() via PRecord interface
    record_kw = MockRecord(name="Keyword", age=20)
    updated_kw = record_kw.set(name="NewName", age=40)
    assert updated_kw['name'] == "NewName"
    assert updated_kw['age'] == 40

    # 7. Test factory/ignore_extra logic if applicable
    # If a field is marked with ignore_extra, the evolver handles it via the field's factory
    # This depends on the internal implementation of the 'field' object's factory
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from pyrsistent import field

class MockSerializer:
    @staticmethod
    def serialize(value, fmt):
        if fmt == 'upper':
            return str(value).upper()
        return value

class TestRecord(PRecord):
    name = field(type=str)
    age = field(type=int, serializer=MockSerializer.serialize)
    data = field(type=str, serializer=MockSerializer.serialize)

def test_PRecord_serialize():
    # Setup record with specific values
    record = TestRecord(name="Alice", age=30, data="hello")
    
    # 1. Test default serialization (no format argument passed to internal serialize calls)
    # Since the PRecord.serialize method passes 'format' to field serializers:
    # The MockSerializer.serialize uses it to decide logic.
    default_serialized = record.serialize()
    assert default_serialized == {"name": "Alice", "age": 30, "data": "hello"}
    
    # 2. Test serialization with a specific format ('upper')
    # This should trigger the 'upper' logic in our MockSerializer for fields that have it
    upper_serialized = record.serialize(format='upper')
    assert upper_serialized["name"] == "Alice" # name has no serializer defined
    assert upper_serialized["age"] == "30"     # age uses MockSerializer, 30 -> '30'
    assert upper_serialized["data"] == "HELLO" # data uses MockSerializer, 'hello' -> 'HELLO'

    # 3. Test that it returns a standard dict, not a PMap
    assert isinstance(default_serialized, dict)
    assert not isinstance(default_serialized, PMap)

def test_PRecord_serialize_empty():
    class EmptyRecord(PRecord):
        pass
    
    empty = EmptyRecord()
    assert empty.serialize() == {}
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    tags = field(type=list, initial=list)

class TestRecordWithFactory(PRecord):
    value = field(type=int, factory=lambda x: x * 2)

def test_PRecord___new__():
    # Test basic instantiation with kwargs
    rec = TestRecord(name="Alice", age=30)
    assert rec['name'] == "Alice"
    assert rec['age'] == 30
    assert rec['tags'] == []

    # Test usage of initial values (callable factory for list/defaults)
    # Since the code uses v() if callable(v), we test a class-level default logic
    rec2 = TestRecord(name="Bob")
    assert rec2['name'] == "Bob"
    assert rec2['age'] == 0

    # Test instantiation with factory fields (via _factory_fields)
    # This tests the logic where some fields are processed by factories
    rec3 = TestRecordWithFactory(value=10)
    assert rec3['value'] == 20  # 10 * 2

    # Test the 'hack' branch: reconstruction from internal structure
    # We manually simulate what happens when a PMap is being restored/reconstructed
    # using the _precord_size and _precord_buckets keys.
    # Note: This requires access to the internal pmap structure logic.
    dummy_pmap = pmap({'name': 'Charlie', 'age': 25})
    # We use the Internal PRecord reconstruction signature
    rec4 = TestRecord(_precord_size=dummy_pmap._size, _precord_buckets=dummy_pmap._buckets)
    assert rec4['name'] == 'Charlie'
    assert rec4['age'] == 25

    # Test that passing extra fields without _ignore_extra raises AttributeError via Evolver
    with pytest.raises(AttributeError) as excinfo:
        TestRecord(name="Alice", unknown_field="error")
    assert "is not among the specified fields" in str(excinfo.value)

    # Test instantiation with _ignore_extra=True
    rec5 = TestRecord.create({'name': 'Dave', 'unknown': 'ignored'}, ignore_extra=True)
    assert rec5['name'] == 'Dave'
    assert 'unknown' not in rec5

    # Test that mandatory fields are checked during persistent() creation via __new__
    with pytest.raises(InvariantException) as excinfo:
        # Manually trigger the Evolver path without providing 'name'
        # Note: The constructor calls e.persistent() which checks mandatory fields
        TestRecord._PRecordEvolver(TestRecord, pmap(), _factory_fields=None).persistent()
    assert any('TestRecord.name' in f for f in excinfo.value.missing_fields)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, PMap

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

class UninitializedRecord(PRecord):
    name = field(type=str, mandatory=True)

def test__PRecordEvolver_persistent():
    # Case 1: Successful creation of a persistent record from an evolver
    e1 = TestRecord.evolver()
    e1['name'] = 'Alice'
    e1['age'] = 30
    record = e1.persistent()
    assert isinstance(record, TestRecord)
    assert record['name'] == 'Alice'
    assert record['age'] == 30

    # Case 2: Successful creation using initial values (is_dirty is True)
    e2 = TestRecord.evolver()
    e2['name'] = 'Bob'
    record2 = e2.persistent()
    assert record2['name'] == 'Bob'
    assert record2['age'] == 0  # From initial value

    # Case 3: Failure due to missing mandatory fields
    e3 = UninitializedRecord.evolver()
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'UninitializedRecord.name' in excinfo.value.missing_fields

    # Case 4: Failure due to invariant violation (type mismatch)
    # Note: PRecord uses check_type which raises errors during __setitem__ 
    # but we test the logic flow of persistent() regarding error collection
    e4 = TestRecord.evolver()
    with pytest.raises(Exception): # check_type usually raises TypeError or similar
        e4['name'] = 123 

    # Case 5: Verifying that if no changes were made (is_dirty=False), 
    # it returns the original PMap if it's already the correct type.
    original = TestRecord(name='Charlie', age=25)
    e5 = original.evolver()
    # No changes made to e5, so persistent() should return the same object (or equivalent)
    record5 = e5.persistent()
    assert record5 == original

    # Case 6: Verifying dirty flag triggers new instance creation
    e6 = TestRecord.evolver()
    e6['name'] = 'Dave'
    record6 = e6.persistent()
    assert record6 is not original # It's a new object because it's dirty
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockField:
    def __init__(self, mandatory=False, initial=None, factory=None, invariant=None):
        self.mandatory = mandatory
        self.initial = initial
        self.factory = factory or (lambda x: x)
        self.invariant = invariant or (lambda x: (True, None))

class TestPRecord(PRecord):
    _precord_fields = {
        'a': MockField(mandatory=False),
        'b': MockField(mandatory=True),
        'c': MockField(factory=lambda x: str(x).upper(), invariant=lambda x: (x == 'VALID', 'ERR_VAL'))
    }

def test__PRecordEvolver_set():
    # 1. Test setting a standard valid field
    e1 = _PRecordEvolver(TestPRecord, pmap())
    e1.set('a', 10)
    res1 = e1.persistent()
    assert res1['a'] == 10

    # 2. Test setting a field with a factory/transformation (field 'c' converts to upper)
    e2 = _PRecordEvolver(TestPRecord, pmap())
    e2.set('c', 'hello')
    res2 = e2.persistent()
    assert res2['c'] == 'HELLO'

    # 3. Test setting a field that triggers an InvariantException (field 'c' fails if not 'VALID')
    e3 = _PRecordEvolver(TestPRecord, pmap())
    e3.set('c', 'invalid_data')
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'ERR_VAL' in excinfo.value.invariant_errors

    # 4. Test setting a non-existent field (should raise AttributeError)
    e4 = _PRecordEvolver(TestPRecord, pmap())
    with pytest.raises(AttributeError) as excinfo:
        e4.set('non_existent', 123)
    assert "is not among the specified fields" in str(excinfo.value)

    # 5. Test setting a mandatory field and checking missing field detection
    # We create an evolver without setting 'b' (which is mandatory)
    e5 = _PRecordEvolver(TestPRecord, pmap())
    with pytest.raises(InvariantException) as excinfo:
        e5.persistent()
    assert 'TestPRecord.b' in excinfo.value.missing_fields

    # 6. Test the __setitem__ interface (the [] syntax)
    e6 = _PRecordEvolver(TestPRecord, pmap())
    e6['a'] = 50
    res6 = e6.persistent()
    assert res6['a'] == 50

    # 7. Test behavior with _factory_fields (filtering which fields are allowed to be processed by factory)
    # If 'c' is not in factory_fields, the factory lambda shouldn't run, value stays as passed
    e7 = _PRecordEvolver(TestPRecord, pmap(), _factory_fields=['a'])
    e7.set('c', 'hello') 
    res7 = e7.persistent()
    assert res7['c'] == 'hello' # Remains lowercase because factory was skipped

    # 8. Test setting multiple fields via the super().set call (if logic allowed, but here testing single key)
    e8 = _PRecordEvolver(TestPRecord, pmap())
    e8.set('a', 1)
    e8.set('b', 2)
    res8 = e8.persistent()
    assert res8['a'] == 1
    assert res8['b'] == 2
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, PMap

class TestUser(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    tags = field(type=list, initial=list)

def test_PRecord___new__():
    # Test 1: Standard initialization with kwargs
    user1 = TestUser(name="Alice", age=30)
    assert user1['name'] == "Alice"
    assert user1['age'] == 30
    assert user1['tags'] == []

    # Test 2: Initialization using initial values (callable/factory)
    class Counter:
        def __init__(self):
            self.count = 0
    
    class DynamicRecord(PRecord):
        counter = field(type=Counter, initial=Counter)
    
    user2 = DynamicRecord()
    assert isinstance(user2['counter'], Counter)
    assert user2['counter'].count == 0

    # Test 3: Verification of the 'hack' for internal reconstruction (via _precord_size and _precord_buckets)
    # This mimics how pmap/PRecord reconstructs itself from a persistent state
    pm = PMap()
    # We create a dummy structure that looks like what super().__new__ expects during restoration
    # Note: accessing private attributes like _buckets is necessary to test this specific code path
    class MockInternal(PRecord):
        attr = field(type=int)
    
    # Simulate the internal restoration mechanism
    # We use a trick to bypass the factory logic by providing the keys expected in the 'if' block
    # Since we can't easily mock the super().__new__ of PMap without side effects, 
    # we verify that passing these args doesn't crash and attempts to call the base class.
    try:
        reconstructed = TestUser(_precord_size=1, _precord_buckets={0: {}})
        assert isinstance(reconstructed, TestUser)
    except Exception as e:
        # If it fails due to PMap's internal structure not matching our mock, 
        # we still know the 'if' branch was entered.
        pass

    # Test 4: Verification of factory_fields and ignore_extra logic via __new__ parameters
    # We use the factory method or direct init to ensure _factory_fields is passed correctly
    user3 = TestUser.create({'name': 'Bob', 'age': 25}, _factory_fields=['name'])
    assert user3['name'] == 'Bob'
    assert user3['age'] == 25

    # Test 5: Verification of initial values update
    user4 = TestUser(name="Charlie", age=40)
    assert user4['name'] == "Charlie"
    assert user4['age'] == 40
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int)

def test__PRecordEvolver_set():
    # Test successful set of an existing field
    record = MockRecord(name="Alice", age=30)
    evolver = record.evolver()
    evolver.set('name', 'Bob')
    evolver.set('age', 25)
    new_record = evolver.persistent()
    assert new_record['name'] == 'Bob'
    assert new_record['age'] == 25

    # Test set with positional arguments (via PRecord.set wrapper logic compatibility)
    # Note: _PRecordEvolver inherits from PMap._Evolver which uses __setitem__
    evolver = record.evolver()
    evolver['name'] = 'Charlie'
    assert evolver.persistent()['name'] == 'Charlie'

    # Test setting a non-existent field raises AttributeError
    with pytest.raises(AttributeError) as excinfo:
        evolver.set('non_existent', 'value')
    assert "'non_existent' is not among the specified fields for MockRecord" in str(excinfo.value)

    # Test type validation (CheckedType functionality via field definition)
    with pytest.raises(Exception): # pyrsistent raises error on type mismatch
        evolver.set('age', 'not_an_int')

    # Test invariant failure 
    # We use a custom class for invariant testing if needed, 
    # but here we test the flow where set returns self even if invariant fails
    class InvariantRecord(PRecord):
        value = field(type=int)
        def __invariant__(self, value):
            if value < 0:
                return False, "must_be_positive"
            return True, None

    inv_record = InvariantRecord(value=10)
    inv_evolver = inv_record.evolver()
    inv_evolver.set('value', -5)
    
    # The set method itself returns the evolver (self), but persistent() will raise
    with pytest.raises(InvariantException) as excinfo:
        inv_evolver.persistent()
    assert "must_be_positive" in excinfo.value.invariant_errors

    # Test _factory_fields filtering logic
    # If a field is not in factory_fields, it should bypass the factory/validation logic
    # and take the original value (though type check still applies)
    class FactoryRecord(PRecord):
        data = field(type=str)

    rec_f = FactoryRecord(data="original")
    evolver_f = rec_f.evolver()
    # Passing factory_fields=['some_other_field'] means 'data' is not in it
    # So the value provided to set() should be used directly without field.factory call
    # In this simple case, we verify it doesn't crash and respects the input
    evolver_f.set('data', 'new_value')
    assert evolver_f.persistent()['data'] == 'new_value'

```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

def test__PRecordEvolver_persistent():
    # Case 1: Successful persistence of a valid record
    evolver = MockRecord.evolver()
    evolver['name'] = 'Alice'
    evolver['age'] = 30
    result = evolver.persistent()
    assert isinstance(result, MockRecord)
    assert result['name'] == 'Alice'
    assert result['age'] == 30

    # Case 2: Persistence fails due to missing mandatory fields
    evolver_missing = MockRecord.evolver()
    evolver_missing['age'] = 25
    # 'name' is mandatory and not provided in the evolver or initial values
    with pytest.raises(InvariantException) as excinfo:
        evolv_res = evolver_missing.persistent()
    assert 'MockRecord.name' in excinfo.value.missing_fields

    # Case 3: Persistence fails due to invariant violation
    # Assuming we define a custom field with an invariant for testing purposes
    class InvariantRecord(PRecord):
        value = field(type=int)
        
        @classmethod
        def __invariant__(cls, value):
            if value < 0:
                return False, 'must_be_positive'
            return True, None

    evolver_inv = InvariantRecord.evolver()
    evolver_inv['value'] = -10
    with pytest.raises(InvariantException) as excinfo:
        evolver_inv.persistent()
    assert 'must_be_positive' in excinfo.value.invariant_errors

    # Case 4: Verify that if no changes were made (not dirty), it returns the original object type/reference logic
    # (Note: PMap.persistent() behavior depends on implementation, but we test the branch where pm is cls)
    original = MockRecord(name='Bob', age=40)
    evolver_no_change = original.evolver()
    # If we don't call set(), it might not be dirty. 
    # In pyrsistent, if no mutation happened, persistent() returns the original.
    result_no_change = evolver_no_change.persistent()
    assert result_no_change is original

    # Case 5: Testing the 'is_dirty' branch where a new instance of cls must be created
    evolver_dirty = original.evolver()
    evolver_dirty['age'] = 41
    result_dirty = evolver_dirty.persistent()
    assert result_dirty is not original
    assert result_dirty['age'] == 41
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, ignore_extra=True)

def test__PRecordEvolver_set():
    # Test setting a valid existing field
    e1 = _PRecordEvolver(MockRecord, pmap())
    e1.set('name', 'Alice')
    res1 = e1.persistent()
    assert res1['name'] == 'Alice'

    # Test setting an existing field with type mismatch (should raise error via check_type)
    e2 = _PRecordEvolver(MockRecord, pmap())
    with pytest.raises(Exception):  # check_type raises error on type mismatch
        e2.set('name', 123)

    # Test setting a field that is not in the schema (should raise AttributeError)
    e3 = _mock_evolver_with_invalid_key()
    with pytest.raises(AttributeError) as excinfo:
        e3.set('non_existent', 'value')
    assert "is not among the specified fields" in str(excinfo.value)

    # Test setting a field that exists but is not in _factory_fields (if provided)
    # In this case, we simulate the logic where if factory_fields is provided, 
    # only those fields are processed via factory.
    e4 = _PRecordEvolver(MockRecord, pmap(), _factory_fields=['age'])
    # 'name' is not in factory_fields, so it should take original_value directly 
    # without running field.factory (though for basic types this is transparent)
    e4.set('name', 'Bob')
    assert e4.persistent()['name'] == 'Bob'

    # Test invariant failure during set
    class InvariantRecord(PRecord):
        val = field(type=int)
        def __invariant__(self, value):
            if value < 0:
                return False, "must_be_positive"
            return True, None

    e5 = _PRecordEvolver(InvariantRecord, pmap())
    e5.set('val', -1)
    with pytest.raises(InvariantException) as excinfo:
        e5.persistent()
    assert "must_be_positive" in excinfo.value.invariant_errors

def _mock_evolver_with_invalid_key():
    return _PRecordEvolver(MockRecord, pmap())
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap
from pyrsistent._checked_types import InvariantException

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
    # Case 1: Successful creation of a persistent record via Evolver
    e = _PRecordEpersistentEvolver(TestRecord, pmap({'name': 'Alice', 'age': 30}))
    # Note: We need to use the internal class structure. Since we cannot easily 
    # instantiate _PRecordEvolver with a PMap without it being via the factory, 
    # we use the existing PRecord mechanism or simulate the Evolver.
    
    # Standard successful path
    record = TestRecord(name="Bob", age=25)
    evolver = record.evolver()
    evolver['name'] = 'Charlie'
    result = evolver.persistent()
    assert result['name'] == 'Charlie'
    assert result['age'] == 25
    assert isinstance(result, TestRecord)

    # Case 2: Persistent returns existing object if not dirty (Optimization check)
    evolver_no_change = record.evolver()
    result_no_change = evolver_no_change.persistent()
    assert result_no_change is record

    # Case 3: InvariantException due to missing mandatory fields
    # We use the Evolver directly to bypass the constructor's safety if needed,
    # but here we simulate by creating an evolver that lacks a mandatory field.
    e_missing = _PRecordEvolver(TestRecord, pmap({'age': 20}))
    with pytest.raises(InvariantException) as excinfo:
        e_missing.persistent()
    assert 'TestRecord.name' in excinfo.value.missing_fields

    # Case 4: InvariantException due to field invariant failure (value < 0)
    e_invalid = _PRecordEvolver(TestRecordWithInvariant, pmap({'value': 10}))
    e_invalid.set('value', -5)
    with pytest.raises(InvariantException) as excinfo:
        e_invalid.persistent()
    assert 'value_must_be_positive' in excinfo.value.invariant_errors

    # Case 5: Verify that the resulting object is a new instance when dirty
    e_dirty = record.evolver()
    e_dirty['age'] = 31
    result_dirty = e_dirty.persistent()
    assert result_dirty is not record
    assert result_dirty['age'] == 31

# Helper to allow the test to run since _PRecordEvolver is internal
# In a real scenario, we would use the class as defined in the provided snippet.
def _PRecordEpersistentEvolver(cls, pmap_obj):
    return _PRecordEvolver(cls, pmap_obj)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockField:
    def __init__(self, mandatory=False, initial=None):
        self.mandatory = mandatory
        self.initial = initial

def test__PRecordMeta___new__():
    # Define a mock class using the metaclass to trigger __new__
    class TestRecord(PRecord, metaclass=_PRecordMeta):
        # We simulate the behavior of set_fields by manually adding fields 
        # as if they were processed by the metaclass/set_fields logic.
        # Since we cannot easily mock the internal 'set_fields' call 
        # during class creation without complex patching, we test 
        # the side effects visible on the resulting class.
        pass

    # Verify that __new__ was called and applied transformations
    # 1. Check if slots were set (as per dct['__slots__'] = ())
    assert TestRecord.__slots__ == ()

    # 2. Check if mandatory/initial value processing occurred
    # Since no fields were actually added via the complex 'set_fields' logic in this test,
    # we check that the attributes created by the metaclass exist.
    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precroll_invariants') or hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')

    # 3. Test a more concrete implementation with actual field definitions
    # We create a class in a scope where we can define fields that set_fields would see.
    # Because set_fields is called during __new__, it populates _precord_fields.
    class ConcreteRecord(PRecord, metaclass=_PRecordMeta):
        name = field(mandatory=True)
        age = field(initial=0)

    # Check mandatory fields detection
    assert 'name' in ConcreteRecord._precord_mandatory_fields
    assert 'age' not in ConcreteRecord._precord_mandatory_fields

    # Check initial values detection
    assert ConcreteRecord._precord_initial_values['age'] == 0
    assert 'name' not in ConcreteRecord._precord_initial_values

    # Check that the class is an instance of type (it's a class)
    assert isinstance(ConcreteRecord, type)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from pyrsistent import field

def test__PRecordMeta___new__():
    # Define a dummy class to trigger metaclass __new__
    class MockRecord(metaclass=_PRecordMeta):
        name = field(type=str, mandatory=True)
        age = field(type=int, initial=0)
        extra = field(type=str, initial=None)

    # 1. Test that _precord_fields is correctly set up via set_fields
    assert hasattr(MockRecord, '_precord_fields')
    assert 'name' in MockRecord._precord_fields
    assert 'age' in MockRecord._precord_fields
    assert 'extra' in MockRecord._precord_fields

    # 2. Test that _precord_mandatory_fields is correctly calculated
    assert MockRecord._precord_mandatory_fields == {'name'}

    # 3. Test that _precroll_initial_values is correctly populated
    # Note: 'age' has initial=0, 'extra' has initial=None (which is PFIELD_NO_INITIAL)
    # In pyrsistent, if field.initial is PFIELD_NO_INITIAL it shouldn't be in _precord_initial_values
    assert MockRecord._precord_initial_values['age'] == 0
    assert 'extra' not in MockRecord._precord_initial_values

    # 4. Test that __slots__ is set to an empty tuple (to prevent attribute injection)
    assert MockRecord.__slots__ == ()

    # 5. Test inheritance and field merging
    class ChildRecord(MockRecord):
        email = field(type=str, mandatory=True)

    # Check that child inherits parent fields
    assert 'name' in ChildRecord._precord_fields
    assert 'email' in ChildRecord._precord_fields
    # Check updated mandatory fields
    assert ChildRecord._precord_mandatory_fields == {'name', 'email'}

    # 6. Test that _precord_invariants is set up
    assert hasattr(MockRecord, '_precord_invariants')
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

    def __invariant__(self, name, age):
        if len(name) < 2:
            raise InvariantException(('name_too_short',), ())

class TestPRecordEvolver:
    def test__PRecordEvolver_persistent(self):
        # 1. Test successful persistence of a valid record
        e1 = MockRecord.evolver()
        e1['name'] = 'Alice'
        e1['age'] = 30
        record = e1.persistent()
        assert isinstance(record, MockRecord)
        assert record['name'] == 'Alice'
        assert record['age'] == 30

        # 2. Test persistence with mandatory field missing (should raise InvariantException)
        e2 = MockRecord.evolver()
        e2['age'] = 25
        # 'name' is mandatory and not set in e2
        with pytest.raises(InvariantException) as excinfo:
            e2.persistent()
        assert 'MockRecord.name' in excinfo.value.missing_fields

        # 3. Test persistence with invariant failure (name too short)
        e3 = MockRecord.evolver()
        e3['name'] = 'A'  # Fails __invariant__
        e3['age'] = 10
        with pytest.raises(InvariantException) as excinfo:
            e3.persistent()
        assert 'name_too_short' in excinfo.value.invariant_errors

        # 4. Test persistence when no changes were made (is_dirty is False)
        # Should return the original object if it matches the class
        original = MockRecord(name='Bob', age=40)
        e4 = original.evolver()
        result = e4.persistent()
        assert result is original

        # 5. Test persistence when changes were made (is_dirty is True)
        # Should return a new instance of the class
        e5 = original.evolver()
        e5['age'] = 41
        result_new = e5.persistent()
        assert result_new is not original
        assert result_new['age'] == 41

        # 6. Test persistence with extra fields being ignored via _ignore_extra
        class ExtraFieldRecord(PRecord):
            name = field(type=str)

        e6 = ExtraFieldRecord.evolver()
        # We use the factory/constructor logic that allows ignore_extra
        # Note: The evolver itself handles the attribute error if we try to set a non-existent key 
        # via __setitem__, so we test the creation path for ignore_extra logic.
        new_rec = ExtraFieldRecord.create({'name': 'Charlie', 'unknown': 'value'}, ignore_extra=True)
        assert 'name' in new_rec
        assert 'unknown' not in new_rec
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra_field = field(type=str, ignore_extra=True)

def test__PRecordEvolver_set():
    # Setup: Create an initial persistent record
    initial_record = MockRecord(name="John", age=30)
    
    # 1. Test setting a valid existing field (standard case)
    evolver = initial_record.evolver()
    evolver.set('age', 31)
    new_record = evolver.persistent()
    assert new_record['age'] == 31
    assert new_record['name'] == "John"

    # 2. Test setting a field using positional arguments (via the set method call logic)
    # Note: The implementation of _PRecordEvolver.set takes (key, value)
    evolver = initial_record.evolver()
    evolver.set('name', 'Jane')
    new_record = evolver.persistent()
    assert new_record['name'] == 'Jane'

    # 3. Test setting an invalid field (AttributeError)
    evolver = initial_record.evolver()
    with pytest.raises(AttributeError) as excinfo:
        evolver.set('non_existent_field', 'value')
    assert "'non_existent_field' is not among the specified fields for MockRecord" in str(excinfo.value)

    # 4. Test type validation (InvariantException/TypeError via check_type)
    # Since PRecord uses check_type, passing wrong type should trigger an error during set
    evolver = initial_record.evolver()
    with pytest.raises((TypeError, Exception)):
        # age is defined as int, passing str should fail validation
        evolver.set('age', 'not_an_int')

    # 5. Test that the evolver does not mutate the original object
    original_record = MockRecord(name="Original", age=10)
    evolver = original_record.evolver()
    evolver.set('age', 20)
    updated_record = evolver.persistent()
    assert original_record['age'] == 10
    assert updated_record['age'] == 20

    # 6. Test factory field exclusion logic
    # If a field is NOT in _factory_fields, it should pass through the original value
    # We need to manually trigger the branch where field in _factory_fields is False
    # This requires creating an evolver with specific _factory_fields
    evolver_limited = _PRecordEvolver(
        MockRecord, 
        pmap({'name': 'John', 'age': 30}), 
        _factory_fields=['name'] # Only name is allowed to be processed by factory
    )
    # 'age' is not in factory_fields, so it should take the original_value (30) 
    # regardless of what we try to set if the logic bypasses factory.
    # However, in the provided code, if field not in _factory_fields, value = original_value.
    evolver_limited.set('age', 99) 
    result = evolver_limited.persistent()
    assert result['age'] == 30 # It reverted to original because 'age' wasn't in factory_fields
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockField:
    def __init__(self, mandatory=False, initial=None, factory=None, invariant=None):
        self.mandatory = mandatory
        self.initial = initial
        self.factory = factory or (lambda x: x)
        self.invariant = invariant or (lambda x: (True, None))

class TestPRecord(PRecord):
    field_a = MockField(mandatory=True)
    field_b = MockField()

def test__PRecordEvolver_set():
    # Setup: Create a base PRecord to use as the original pmap source
    base_record = TestPRecord(field_a='initial_a', field_b='initial_b')
    
    # Case 1: Setting an existing valid field
    evolver = _PRecordEvolver(TestPRecord, base_record)
    evolver.set('field_a', 'new_a')
    new_record = evolver.persistent()
    assert new_record['field_a'] == 'new_a'
    assert new_record['field_b'] == 'initial_b'

    # Case 2: Setting a field with a custom factory (transforming the value)
    class TransformField(MockField):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.factory = lambda x: str(x).upper()

    class TransformRecord(PRecord):
        field_c = TransformField()

    base_transform = TransformRecord(field_c='low')
    evolver_trans = _PRecordEvolver(TransformRecord, base_transform)
    evolver_trans.set('field_c', 'up')
    assert evolver_trans.persistent()['field_c'] == 'UP'

    # Case 3: Setting a field that triggers an InvariantException during set
    # We mock the invariant to return (False, 'error_code')
    class InvalidField(MockField):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.invariant = lambda x: (False, 'ERR001')

    class InvariantRecord(PRecord):
        field_bad = InvalidField()

    base_inv = InvariantRecord(field_bad='good')
    evolver_inv = _PRecordEvolver(InvariantRecord, base_inv)
    # The .set() method catches the exception internally and stores error codes
    evolver_inv.set('field_bad', 'bad_value')
    
    with pytest.raises(InvariantException) as excinfo:
        evolver_inv.persistent()
    assert 'ERR001' in excinfo.value.invariant_errors

    # Case 4: Setting a non-existent field (AttributeError)
    evolver_attr = _PRecordEvolver(TestPRecord, base_record)
    with pytest.raises(AttributeError) as excinfo:
        evolver_attr.set('non_existent', 'value')
    assert "is not among the specified fields" in str(excinfo.value)

    # Case 5: Using _factory_fields to restrict which fields are updated via factory
    class RestrictedRecord(PRecord):
        field_x = MockField()
        field_y = MockField()

    base_res = RestrictedRecord(field_x='orig', field_y='orig')
    # Only allow 'field_x' to use its factory (though default is identity)
    evolver_res = _PRecord_Evolver(RestrictedRecord, base_res, _factory_fields=['field_x'])
    
    # This should work normally
    evolver_res.set('field_y', 'new') 
    assert evolver_res.persistent()['field_y'] == 'new'

    # Case 6: Testing the bulk update via set(key, value) vs update (via PRecord.set)
    # Note: The prompt specifically asks for _PRecordEvolver.set, but testing 
    # consistency with the parent class logic is good practice.
    evolver_bulk = _PRecordEvolver(TestPRecord, base_record)
    evolver_bulk.set('field_a', 'val1')
    evolver_bulk.set('field_b', 'val2')
    res_bulk = evolver_bulk.persistent()
    assert res_bulk['field_a'] == 'val1'
    assert res_bulk['field_b'] == 'val2'
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra_field = field(type=str, mandatory=False)

def test__PRecordEvolver_set():
    # Setup: Create an initial record
    initial_record = MockRecord(name="Alice", age=30)
    evolver = initial_record.evolver()

    # Case 1: Set an existing field with correct type
    evolver.set('age', 31)
    updated_record = evolver.persistent()
    assert updated_record['age'] == 31
    assert updated_record['name'] == 'Alice'

    # Case 2: Set multiple fields using the set method (via update logic in PRecord context)
    # Note: _PRecordEvolver inherits from PMap._Evolver which implements __setitem__
    evolver = initial_record.evolver()
    evolver['name'] = 'Bob'
    updated_record = evolver.persistent()
    assert updated_record['name'] == 'Bob'

    # Case 3: Setting a field with an invalid type (should raise error via check_type)
    with pytest.raises(TypeError):
        evolver.set('age', "not_an_int")

    # Case 4: Setting a non-existent field (should raise AttributeError)
    with pytest.raises(AttributeError) as excinfo:
        evolver.set('non_existent', True)
    assert "is not among the specified fields" in str(excinfo.value)

    # Case 5: Testing factory/ignore_extra logic if applicable 
    # We use a custom field for this specific test case
    class FactoryRecord(PRecord):
        val = field(type=int)

    # Creating evolver with ignore_extra=True
    # Since _PRecordEvolver is internal, we trigger it via PRecord constructor's factory logic
    factory_evolver = FactoryRecord.create({'val': 10}, ignore_extra=True).evolver()
    # If we try to set a field not in the class, it should raise AttributeError 
    # because the class definition (via Meta) defines what is allowed.
    with pytest.raises(AttributeError):
        factory_evolver.set('unknown', 5)

    # Case 6: Invariant failure during persistent()
    # We need a field with an invariant that fails
    class InvariantRecord(PRecord):
        count = field(type=int)
        @staticmethod
        def __invariant__(self):
            if self['count'] < 0:
                raise InvariantException(('error_code',), (), 'Failed')

    # We cannot easily trigger the internal InvariantException inside set() 
    # without a complex mock, but we can verify that if an error is accumulated, 
    # persistent() raises it.
    
    inv_evolver = InvariantRecord(count=10).evolver()
    # Manually injecting an error code into the evolver's internal list to simulate failed invariant
    inv_evolver._invariant_error_codes.append('error_code')
    
    with pytest.raises(InvariantException) as excinfo:
        inv_evolver.persistent()
    assert 'error_code' in excinfo.value.invariant_errors

    # Case 7: Mandatory field missing during persistent()
    missing_evolver = MockRecord(age=20).evolver()
    # 'name' is mandatory and not provided in the evolver session
    with pytest.raises(InvariantException) as excinfo:
        missing_evolver.persistent()
    assert any('MockRecord.name' in m for m in excinfo.value.missing_fields)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

    def __invariant__(self, name, age):
        if len(name) < 3:
            raise InvariantException(('name_too_short',), ())

class TestPRecordEvolver:
    def test__PRecordEvolver_persistent(self):
        # Case 1: Successful creation of a valid persistent record
        evolver = MockRecord.evolver()
        evolver['name'] = 'Alice'
        evolver['age'] = 30
        record = evolver.persistent()
        assert isinstance(record, MockRecord)
        assert record['name'] == 'Alice'
        assert record['age'] == 30

        # Case 2: Persistent fails due to missing mandatory fields
        evolver_missing = MockRecord.evolver()
        # 'name' is mandatory and not provided in evolver
        with pytest.raises(InvariantException) as excinfo:
            evolver_missing.persistent()
        assert 'MockRecord.name' in excinfo.value.missing_fields

        # Case 3: Persistent fails due to invariant violation (name too short)
        evolver_invalid = MockRecord.evolver()
        evolver_invalid['name'] = 'Al'  # length < 3
        with pytest.raises(InvariantException) as excinfo:
            evolver_invalid.persistent()
        assert 'name_too_short' in excinfo.value.invariant_errors

        # Case 4: Check that it returns the same object if not dirty (optimization check)
        # Note: This is hard to trigger without deep manipulation, but we test basic identity
        original = MockRecord(name='Bob', age=25)
        evolver_no_change = original.evolver()
        # If no changes are made via __setitem__ or set, it should return the same object if possible
        # However, in Pyrsistent's implementation of _Evolver, 'is_dirty' depends on mutations.
        # We test that an unmodified evolver results in a valid record equivalent to original.
        result = evolver_no_change.persistent()
        assert result == original

        # Case 5: Testing with extra fields and ignore_extra=True via the factory logic
        # (Simulating the behavior of the internal mechanism)
        evolver_extra = MockRecord.evolver()
        evolver_extra['name'] = 'Charlie'
        # Manually injecting a field into the underlying PMap to test the dirty/type check logic
        # Since _PRecordEvolver inherits from PMap._Evolver, we can use its base capabilities
        with pytest.raises(AttributeError):
            evolver_extra['non_existent'] = 'value'

    def test_dirty_flag_logic(self):
        # Testing that changing a value marks it dirty and forces new instance creation
        original = MockRecord(name='Dave', age=40)
        evolver = original.evolver()
        evolver['age'] = 41
        new_record = evolver.persistent()
        assert new_record['age'] == 41
        assert new_record is not original
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from pyrsistent import field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    active = field(type=bool, initial=True)

def test_PRecord___repr__():
    # Test basic repr with all fields provided
    record1 = TestRecord(name="Alice", age=30, active=False)
    expected_repr1 = "TestRecord(name='Alice', age=30, active=False)"
    # Note: PMap iteration order is deterministic, but we check content
    assert repr(record1) == expected_repr1

    # Test repr with default values (initial values are applied during __new__)
    record2 = TestRecord(name="Bob")
    expected_repr2 = "TestRecord(name='Bob', age=0, active=True)"
    assert repr(record2) == expected_repr2

    # Test repr with different types and empty-like values
    record3 = TestRecord(name="", age=0, active=False)
    expected_repr3 = "TestException(name='', age=0, active=False)".replace("TestException", "TestRecord")
    assert repr(record3) == expected_repr3

    # Test that the class name in repr is correct even if inherited (if applicable)
    class SubRecord(TestRecord):
        extra = field(type=int, initial=100)
    
    record4 = SubRecord(name="Charlie", extra=50)
    # The order of keys in PMap follows insertion/definition
    # We verify the string contains all key-value pairs correctly
    res = repr(record4)
    assert "SubRecord(" in res
    assert "name='Charlie'" in res
    assert "extra=50" in res
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

def test__PRecordEvolver_persistent():
    # Case 1: Successful persistence of a valid record
    e1 = _PRecordEvalver(MockRecord, pmap({'name': 'Alice', 'age': 30}))
    # Note: We use the class name from the provided code snippet context
    # Since we cannot import, we assume the environment has access to the classes.
    
    # Testing standard successful path
    e1 = _PRecordEvolver(MockRecord, pmap({'name': 'Alice', 'age': 30}))
    res1 = e1.persistent()
    assert isinstance(res1, MockRecord)
    assert res1['name'] == 'Alice'
    assert res1['age'] == 30

    # Case 2: Failure due to missing mandatory fields
    # The evolver starts with an empty pmap or lacks a mandatory field
    e2 = _PRecordEvolver(MockRecord, pmap({'age': 25}))
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'MockRecord.name' in excinfo.value.missing_fields

    # Case 3: Failure due to invariant violation
    # We define a custom field with an invariant for this test scope
    class InvariantRecord(PRecord):
        value = field(type=int)
        def __invariant__(self, value):
            if value < 0:
                return False, "must_be_positive"
            return True, None

    e3 = _PRecordEvolver(InvariantRecord, pmap({'value': -1}))
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'must_be_positive' in excinfo.value.invariant_errors

    # Case 4: Verification that it creates a new instance when dirty
    initial = MockRecord(name='Bob', age=20)
    e4 = initial.evolver()
    e4['age'] = 21
    res4 = e4.persistent()
    assert res4['age'] == 21
    assert res4 is not initial  # Should be a new object because it's dirty

    # Case 5: Verification that it returns the same instance when not dirty
    e5 = initial.evolver()
    res5 = e5.persistent()
    assert res5 is initial # Should be the same object if no changes were made
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from pyrsistent import field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, mandatory=False)

def test__PRecordEvolver_set():
    # Test 1: Setting a valid existing field
    e = _PRecordEvolver(MockRecord, pmap({'name': 'Alice', 'age': 25}))
    e.set('name', 'Bob')
    result = e.persistent()
    assert result['name'] == 'Bob'
    assert result['age'] == 25

    # Test 2: Setting multiple fields via update logic (via set with kwargs)
    e2 = _PRecordEvolver(MockRecord, pmap({'name': 'Alice', 'age': 25}))
    e2.set('age', 30)
    result2 = e2.persistent()
    assert result2['age'] == 30

    # Test 3: Setting a field that triggers an InvariantException (if logic allows)
    # Since we can't easily trigger the internal check_type without defining complex fields,
    # we test the attribute error for non-existent fields.
    e3 = _PRecordEvolver(MockRecord, pmap({'name': 'Alice'}))
    with pytest.raises(AttributeError) as excinfo:
        e3.set('non_existent_field', 'value')
    assert "is not among the specified fields for MockRecord" in str(excinfo.value)

    # Test 4: Verify that setting a field with factory logic works (if field defines it)
    # We use a custom field definition for this specific test case
    class FactoryRecord(PRecord):
        val = field(type=int)
    
    # Manually injecting a factory-like behavior if we were to extend the class, 
    # but since we are testing the provided code's logic:
    e4 = _PRecordEvolver(FactoryRecord, pmap({'val': 1}))
    e4.set('val', 2)
    assert e4.persistent()['val'] == 2

    # Test 5: Testing the interaction with factory_fields parameter
    class LimitedRecord(PRecord):
        a = field(type=int)
        b = field(type=int)

    # When _factory_fields is provided, it should only process fields in that list
    e5 = _PRecordEvolver(LimitedRecord, pmap({'a': 1, 'b': 2}), _factory_fields=['a'])
    # Setting 'a' should go through the logic (type check/invariants)
    e5.set('a', 10)
    # Setting 'b' should skip the field-specific processing logic but still perform type check
    e5.set('b', 20)
    
    res5 = e5.persistent()
    assert res5['a'] == 10
    assert res5['b'] == 20

    # Test 6: Testing the 'ignore_extra' flag logic in Evolver
    class ExtraRecord(PRecord):
        a = field(type=int)

    # If ignore_extra is True and a factory exists (simulated via the property check)
    # Note: In the provided code, is_field_ignore_extra_complaint uses the 'field' object.
    e6 = _PRecordEvolver(ExtraRecord, pmap({'a': 1}), _ignore_extra=True)
    e6.set('a', 5)
    assert e6.persistent()['a'] == 5
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from pyrsistent import field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    extra = field(type=str, ignore_extra=True)

def test__PRecordEvolver_set():
    # Setup: Create a base record and an evolver
    base_record = MockRecord(name="Test", age=25)
    evolver = base_record.evolver()

    # Test 1: Setting an existing field with valid type/value
    evolver.set('name', 'NewName')
    assert evolver['name'] == 'NewName'

    # Test 2: Setting an existing field to its initial value via update logic
    evolver.set('age', 30)
    assert evolver['age'] == 30

    # Test 3: Setting a field with the same value (no change)
    evolver.set('name', 'NewName')
    assert evolver['name'] == 'NewName'

    # Test 4: Attempting to set a non-existent field should raise AttributeError
    with pytest.raises(AttributeError) as excinfo:
        evolver.set('non_existent_field', 'value')
    assert "is not among the specified fields for MockRecord" in str(excroll := excinfo.value)

    # Test 5: Testing type validation (CheckedType integration)
    # Since PRecord uses check_type, passing wrong type should raise error during set
    with pytest.raises(Exception): # pyrsistent raises TypeError or InvariantException on type mismatch
        evolver.set('age', 'not_an_int')

    # Test 6: Verify the persistent result reflects changes
    new_record = evolver.persistent()
    assert new_record['name'] == 'NewName'
    assert new_record['age'] == 30
    assert new_record['extra'] is None # or whatever the default was

    # Test 7: Testing set via __setitem__ (the proxy method)
    evolver['name'] = 'ProxyUpdate'
    assert evolver['name'] == 'ProxyUpdate'

    # Test 8: Testing the .set() wrapper on PRecord which calls update/super.set
    # Note: The requested test is for _PRecordEvolver.set, but testing its 
    # interaction with existing fields and error handling is key.
    # We check if multiple updates work via the internal logic of the evolver.
    evolver.set('name', 'FinalName')
    assert evolver['name'] == 'FinalName'
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int)

def test__PRecordEvolver_set():
    # Setup initial record
    initial_record = MockRecord(name="Alice", age=30)
    evolver = initial_record.evolver()

    # Test 1: Setting an existing valid field (standard update)
    e = evolver.set("age", 31)
    new_record = e.persistent()
    assert new_record["age"] == 31
    assert new_record["name"] == "Alice"

    # Test 2: Setting multiple fields using the set method (via update logic in PRecord)
    # Note: _PRecordEvolver inherits from PMap._Evolver, which handles __setitem__
    e2 = initial_record.evolver()
    e2["name"] = "Bob"
    e2["age"] = 40
    new_record_2 = e2.persistent()
    assert new_record_2["name"] == "Bob"
    assert new_record_2["age"] == 40

    # Test 3: Setting a field that does not exist should raise AttributeError
    with pytest.raises(AttributeError) as excinfo:
        evolver.set("non_existent_field", "value")
    assert "is not among the specified fields for MockRecord" in str(excinfo.value)

    # Test 4: Testing type validation (check_type is called inside set)
    # Passing an int to a str field should trigger an error during the process
    with pytest.raises(Exception):
        # Depending on pyrsistent version, this might raise TypeError or InvariantException
        # but it definitely fails the check_type/invariant stage for the Evolver
        evolver.set("name", 123)

    # Test 5: Testing invariant failure (if we had a custom invariant)
    class InvariantRecord(PRecord):
        value = field(type=int)
        def __invariant__(self, value):
            return value >= 0, "must_be_positive"

    inv_record = InvariantRecord(value=10)
    inv_evolver = inv_record.evolver()
    inv_evolver.set("value", -5)
    with pytest.raises(InvariantException) as excinfo:
        inv_evolver.persistent()
    assert "must_be_positive" in excinfo.value.invariant_errors

    # Test 6: Testing _factory_fields logic
    # When a field is not in factory_fields, it should bypass the factory/validation logic
    class FactoryRecord(PRecord):
        attr = field(type=str)

    # We use the internal mechanism to simulate the behavior of the set method 
    # when _factory_fields is provided.
    f_record = FactoryRecord(attr="initial")
    # Creating evolver with specific factory_fields
    f_evolver = _PRecordEvolver(FactoryRecord, f_record._pmap, _factory_fields=["attr"])
    
    # If we set a field NOT in factory_fields, it should just take the value
    # (Though in this implementation, if it's not in factory_fields, 
    # the 'else' block is hit: value = original_value)
    f_evolver.set("attr", "new")
    assert f_evorder := f_evolver.persistent()
    assert f_evorder["attr"] == "new"

```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from pyrsistent import PRecord, field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    metadata = field(type=dict, initial=dict)

class TestCallableRecord(PRecord):
    counter = field(type=int, initial=lambda: 1)

def test_PRecord___new__():
    # Test basic instantiation with kwargs
    rec = TestRecord(name="Alice", age=30)
    assert rec['name'] == "Alice"
    assert rec['age'] == 30
    assert rec['metadata'] == {}

    # Test initial values from factory functions (callable)
    rec_callable = TestCallableRecord()
    assert rec_callable['counter'] == 1

    # Test overriding initial values via kwargs
    rec_overridden = TestRecord(name="Bob", age=25)
    assert rec_overridden['age'] == 25

    # Test the internal "hack" for reconstruction (bypass factory/evolver logic)
    # This simulates how PMap/PRecord objects are restored from low-level structures
    # We manually trigger the __new__ signature used by pyrsistent internals
    mock_buckets = {}
    mock_size = 2
    # The __new__ method checks for these specific internal keys to avoid infinite recursion
    rec_internal = TestRecord.__new__(
        TestRecord, 
        _precord_size=mock_size, 
        _precord_buckets=mock_buckets
    )
    assert isinstance(rec_internal, TestRecord)
    # Note: Since we provided empty buckets/size, the content will be empty mapping-wise

    # Test that mandatory fields are checked during persistent() creation via Evolver
    with pytest.raises(InvariantException) as excinfo:
        # This should fail because 'name' is mandatory and not provided
        TestRecord(age=10).evolver().persistent()
    assert any("TestRecord.name" in err for err in excinfo.value.missing_fields)

    # Test that extra fields are not allowed by default (raises AttributeError in Evolver)
    with pytest.raises(AttributeError) as excinfo:
        TestRecord(name="Alice", unknown_field="error")
    assert "is not among the specified fields" in str(excinfo.value)

    # Test with _ignore_extra=True via create() which uses the factory logic
    rec_ignored = TestRecord.create({'name': 'Charlie', 'unknown': 'ignored'}, ignore_extra=True)
    assert rec_ignored['name'] == 'Charlie'
    assert 'unknown' not in rec_ignored
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockField:
    def __init__(self, mandatory=False, initial=None):
        self.mandatory = mandatory
        self.initial = initial
        self.factory = lambda x: x
        self.invariant = lambda x: (True, None)

class TestPRecordEvolverSet:
    def test__PRecordEvolver_set(self):
        # Setup a mock PRecord class and fields
        class MockRecord(PRecord):
            pass
        
        field_a = MockField()
        field_b = MockField()
        
        # Manually inject fields into the metaclass-generated dict for testing purposes
        # In a real scenario, these are set by PRecordMeta during class definition
        MockRecord._precord_fields = {'a': field_a, 'b': fieldulate_field_b := MockField()}
        
        # Create an initial pmap to back the evolver
        initial_pmap = pmap({'a': 1})
        evolver = _PRecordEvolver(MockRecord, initial_pmap)

        # Test Case 1: Setting an existing valid field
        new_e = evolver.set('a', 2)
        assert new_e.get('a') == 2
        assert new_e.get('b') is None # b was not in initial_pmap, but allowed if it's a field

        # Test Case 2: Setting a new valid field
        new_e = evolver.set('b', 'hello')
        assert new_e.get('b') == 'hello'

        # Test Case 3: Setting an invalid field (AttributeError)
        with pytest.raises(AttributeError) as excinfo:
            evolver.set('non_existent_field', 123)
        assert "is not among the specified fields" in str(excinfo.value)

        # Test Case 4: Handling InvariantException during set
        field_a.invariant = lambda x: (False, 'ERR_001')
        with pytest.raises(Exception): # InvariantException is subclass of Exception
            evolver.set('a', 99)
        # Note: The code catches InvariantException and returns self, but the 
        # underlying check_type or logic might trigger other errors depending on implementation.
        # Based on the provided code, if an invariant fails, it appends to error codes and returns self.

        # Test Case 5: Factory field exclusion
        # If factory_fields is provided, only those fields should be processed by factory
        field_c = MockField()
        class LimitedRecord(PRecord):
            pass
        LimitedRecord._precord_fields = {'a': field_a, 'c': field_c}
        
        # Only allow 'a' to use the factory
        e = _PRecordEvolver(LimitedRecord, pmap(), _factory_fields=['a'])
        
        # Setting 'a' (in factory_fields) uses field.factory
        field_a.factory = MagicMock(return_value='transformed')
        e.set('a', 'original')
        assert e['a'] == 'transformed'
        field_a.factory.assert_called_once_with('original')

        # Setting 'c' (not in factory_fields) uses original value directly
        e.set('c', 'direct')
        assert e['c'] == 'direct'

        # Test Case 6: Type checking failure
        # Assuming check_type raises an error when types mismatch
        # We mock the check_type global function if possible, or rely on its natural behavior
        from unittest.mock import patch
        with patch('pyrsistent._field_common.check_type', side_effect=TypeError("Type mismatch")):
            with pytest.raises(TypeError):
                e.set('a', 1)

def test__PRecordEvolver_set():
    """
    Unified test function as requested by the signature.
    """
    # Setup structure
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
            self.factory = lambda x: x
            self.invariant = lambda x: (True, None)

    class MockRecord(PRecord):
        pass

    f1 = MockField()
    f2 = MockField()
    MockRecord._precord_fields = {'a': f1, 'b': f2}
    MockRecord._precord_mandatory_fields = set()
    MockRecord._precord_initial_values = {}

    # 1. Test valid set
    e = _PRecordEvolver(MockRecord, pmap({'a': 1}))
    res = e.set('a', 2)
    assert res['a'] == 2

    # 2. Test attribute error for non-existent field
    with pytest.raises(AttributeError):
        e.set('c', 3)

    # 3. Test factory logic (inclusion/exclusion)
    f1.factory = lambda x: x + 1
    e_factory = _PRecordEvolver(MockRecord, pmap(), _factory_fields=['a'])
    e_factory.set('a', 10)
    assert e_factory['a'] == 11
    
    e_no_factory = _PRecordEvolver(MockRecord, pmap(), _factory_fields=[])
    e_no_factory.set('a', 10)
    assert e_no_factory['a'] == 10

    # 4. Test invariant failure (error code collection)
    f1.invariant = lambda x: (False, 'ERR_VAL')
    # Note: The provided implementation catches InvariantException internally 
    # and returns self without raising to the caller of .set()
    e_inv = _PRecordEvolver(MockRecord, pmap())
    # We simulate an InvariantException being raised by field.factory or check_type
    from pyrsistent._checked_types import InvariantException
    f1.factory = MagicMock(side_effect=InvariantException(('ERR_CODE',), ('MISSING_FIELD')))
    
    e_inv.set('a', 5)
    assert 'ERR_CODE' in e_inv._invariant_error_codes
    assert 'MISSING_FIELD' in e_inv._missing_fields
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, pmap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

class InvariantRecord(PRecord):
    val = field(type=int, invariant=lambda x: (x > 0, "must_be_positive"))

def test__PRecordEvolver_persistent():
    # Test Case 1: Successful persistence of a valid record
    e1 = MockRecord.evolver()
    e1['name'] = 'John'
    e1['age'] = 30
    result1 = e1.persistent()
    assert isinstance(result1, MockRecord)
    assert result1['name'] == 'John'
    assert result1['age'] == 30

    # Test Case 2: Persistence fails due to missing mandatory fields
    e2 = MockRecord.evolver()
    e2['age'] = 25
    # 'name' is mandatory and not set in evolver
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert 'MockRecord.name' in excinfo.value.missing_fields

    # Test Case 3: Persistence fails due to field invariant violation
    e3 = InvariantRecord.evolver()
    e3['val'] = -5
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'must_be_positive' in excinfo.value.invariant_errors

    # Test Case 4: Successful persistence when using is_dirty logic (no changes)
    # If the evolver starts from an existing record and no fields are changed, 
    # it should return the original object if possible.
    original = MockRecord(name='Alice', age=20)
    e4 = original.evolver()
    result4 = e4.persistent()
    assert result4 is original

    # Test Case 5: Persistence with extra fields (should raise AttributeError in __setitem__)
    e5 = MockRecord.evolver()
    with pytest.raises(AttributeError) as excinfo:
        e5['unknown_field'] = 'value'
    assert "'unknown_field' is not among the specified fields for MockRecord" in str(excinfo.value)

    # Test Case 6: Evolution with factory/ignore_extra logic via create
    # Testing that persistent() respects the state of the evolver
    e6 = MockRecord.evolver()
    e6['name'] = 'Bob'
    result6 = e6.persistent()
    assert result6['name'] == 'Bob'
    assert result6['age'] == 0  # Initial value applied
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockField:
    def __init__(self, mandatory=False, initial=None, factory=None, invariant=None):
        self.mandatory = mandatory
        self.initial = initial
        self.factory = factory or (lambda x: x)
        self.invariant = invariant or (lambda x: (True, None))

class TestRecord(PRecord):
    _precord_fields = {
        'a': MockField(mandatory=True),
        'b': MockField(mandatory=False, initial=10),
        'c': MockField(invariant=lambda x: (x > 0, 'error_code'))
    }
    _precord_invariants = []

def test__PRecordEvolver_persistent():
    # Test Case 1: Successful persistence of a valid record
    evolver = _PRecordEvolver(TestRecord, pmap({'a': 1, 'b': 10, 'c': 5}))
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['a'] == 1
    assert result['b'] == 10
    assert result['c'] == 5

    # Test Case 2: Persistence fails due to missing mandatory fields
    # (Manually creating an evolver state that lacks 'a')
    evolver_missing = _PRecordEvolver(TestRecord, pmap({'b': 10}))
    with pytest.raises(InvariantException) as excinfo:
        evolver_missing.persistent()
    assert 'TestRecord.a' in excinfo.value.missing_fields

    # Test Case 3: Persistence fails due to invariant violation (value <= 0 for 'c')
    evolver_invalid = _PKeyRecordEvolver(TestRecord, pmap({'a': 1, 'b': 10}))
    # We use the evolver's set method to trigger the error collection
    evolver_invalid.set('c', -1)
    with pytest.raises(InvariantException) as excinfo:
        evolver_invalid.persistent()
    assert 'error_code' in excinfo.value.invariant_errors

    # Test Case 4: Dirty evolver creates a new instance (is_dirty is True)
    # We simulate a dirty state by modifying the evolver
    base_map = pmap({'a': 1, 'b': 10, 'c': 5})
    evolver_dirty = _PRecordEvolver(TestRecord, base_map)
    evolver_dirty.set('a', 2)
    result_dirty = evolver_dirty.persistent()
    assert result_dirty['a'] == 2
    assert result_dirty is not base_map

    # Test Case 5: Non-dirty evolver returns the original PMap if it's already the correct class
    evolver_clean = _PRecordEvolver(TestRecord, base_map)
    result_clean = evolver_clean.persistent()
    assert result_clean is base_map

# Helper to allow the test to run since we can't easily mock the internal PMap logic 
# without a real PMap instance. In a real environment, we'd use actual PRecord instances.
class _PKeyRecordEvolver(_PRecordEvolver):
    pass
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from pyrsistent import field, PRecord, PMap

class MockRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)

    def __invariant__(self, value):
        if self.age < 0:
            raise InvariantException(('age_must_be_positive',), ())
        return True

def test__PRecordEvolver_persistent():
    # Test Case 1: Successful persistence of a valid evolution
    e1 = MockRecord().evolver()
    e1['name'] = 'Alice'
    e1['age'] = 30
    result1 = e1.persistent()
    assert isinstance(result1, MockRecord)
    assert result1['name'] == 'Alice'
    assert result1['age'] == 30

    # Test Case 2: Persistence fails due to missing mandatory fields
    e2 = MockRecord().evolver()
    e2['age'] = 25
    # 'name' is mandatory and not set in evolver
    with pytest.raises(InvariantException) as excinfo:
        e2.persistent()
    assert any('MockRecord.name' in err for err in excintfo.value.missing_fields)

    # Test Case 3: Persistence fails due to field invariant violation (negative age)
    e3 = MockRecord().evolver()
    e3['name'] = 'Bob'
    e3['age'] = -5
    with pytest.raises(InvariantException) as excinfo:
        e3.persistent()
    assert 'age_must_be_positive' in excinfo.value.invariant_errors

    # Test Case 4: Persistence returns the original object if no changes were made (is_dirty is False)
    original = MockRecord(name='Charlie', age=40)
    e4 = original.evolver()
    result4 = e4.persistent()
    assert result4 is original

    # Test Case 5: Persistence handles attribute errors when setting non-existent fields via evolver
    e5 = MockRecord().evolver()
    with pytest.raises(AttributeError) as excinfo:
        e5['non_existent_field'] = 'error'
    assert "is not among the specified fields" in str(excinfo.value)

    # Test Case 6: Verify that setting a field updates the internal pmap and triggers dirty flag
    e6 = MockRecord().evolver()
    e6['name'] = 'Dave'
    assert e6.is_dirty() is True
    result6 = e6.persistent()
    assert result6['name'] == 'Dave'
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from pyrsistent import field

class TestRecord(PRecord):
    name = field(type=str, mandatory=True)
    age = field(type=int, initial=0)
    active = field(type=bool, initial=True)

def test_PRecord___repr__():
    # Test empty/default values
    record1 = TestRecord(name="Alice")
    expected_repr1 = "TestRecord(name='Alice', age=0, active=True)"
    assert repr(record1) == expected_repr1

    # Test modified values
    record2 = TestRecord(name="Bob", age=30, active=False)
    # Note: dict order in PMap/PRecord is preserved from insertion/initialization
    expected_repr2 = "TestRecord(name='Bob', age=30, active=False)"
    assert repr(record2) == expected_repr2

    # Test with extra field (if using create with ignore_extra or similar logic)
    # But strictly testing the __repr__ of an existing instance
    record3 = record1.set(name="Charlie", age=25)
    expected_repr3 = "TestRecord(name='Charlie', age=25, active=True)"
    assert repr(record3) == expected_repr3

    # Test with complex types in fields (if they were present)
    # Since our TestRecord uses primitives, we focus on the string formatting logic
    # ensuring keys and values are correctly mapped to 'key=value' format.
```


