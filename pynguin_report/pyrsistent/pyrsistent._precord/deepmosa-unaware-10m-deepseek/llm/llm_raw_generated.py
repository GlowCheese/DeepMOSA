####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PRecordEvolver_set():
    # Define a simple PRecord class for testing
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int)
        active = field(type=bool, initial=lambda: True)

    # Test 1: Set existing field with valid value
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    result = evolver.persistent()
    assert result['name'] == 'Alice'
    assert result['age'] is None
    assert result['active'] is True

    # Test 2: Set multiple fields sequentially
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    evolver.set('age', 30)
    evolver.set('active', False)
    result = evolver.persistent()
    assert result['name'] == 'Bob'
    assert result['age'] == 30
    assert result['active'] is False

    # Test 3: Set field with factory transformation
    evolver = TestRecord().evolver()
    evolver.set('age', '25')  # String should be converted to int by factory
    result = evolver.persistent()
    assert result['age'] == 25
    assert isinstance(result['age'], int)

    # Test 4: Set field that violates type constraint
    evolver = TestRecord().evolver()
    try:
        evolver.set('age', 'not_a_number')
        result = evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass  # Expected

    # Test 5: Set non-existent field should raise AttributeError
    evolver = TestRecord().evolver()
    try:
        evolver.set('nonexistent', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'nonexistent' is not among the specified fields" in str(e)

    # Test 6: Set field with invariant violation
    class RecordWithInvariant(PRecord):
        value = field(type=int, invariant=lambda x: (x > 0, 'value.positive'))

    evolver = RecordWithInvariant().evolver()
    evolver.set('value', -5)  # Violates invariant
    try:
        result = evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'value.positive' in e.invariant_errors

    # Test 7: Set field with missing mandatory field
    evolver = TestRecord().evolver()
    evolver.set('age', 25)  # But name is mandatory
    try:
        result = evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.name' in e.missing_fields

    # Test 8: Using __setitem__ syntax (delegates to set)
    evolver = TestRecord().evolver()
    evolver['name'] = 'Charlie'
    result = evolver.persistent()
    assert result['name'] == 'Charlie'

    # Test 9: Set field with ignore_extra flag
    class ComplexFieldRecord(PRecord):
        data = field(type=CheckedPMap[str, int], factory=CheckedPMap[str, int].create)

    evolver = ComplexFieldRecord().evolver()
    # This would normally work with ignore_extra behavior in factory
    evolver.set('data', {'a': 1, 'b': 2})
    result = evolver.persistent()
    assert result['data'] == pmap({'a': 1, 'b': 2})

    # Test 10: Override existing value
    record = TestRecord(name='Initial', age=40)
    evolver = record.evolver()
    evolver.set('name', 'Updated')
    result = evolver.persistent()
    assert result['name'] == 'Updated'
    assert result['age'] == 40


# LLM-generated content at query #2
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    from pyrsistent._checked_types import CheckedType

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        score = field(type=int, invariant=lambda x: (x >= 0, 'score.negative'))

    # Test normal field assignment
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    result = evolver.persistent()
    assert result['name'] == 'Alice'
    assert result['age'] == 25

    # Test type checking
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    try:
        evolver.set('age', 'not_an_int')
        evolver.persistent()
        assert False, "Should have raised type error"
    except TypeError:
        pass

    # Test field invariant
    evolver = TestRecord().evolver()
    evolver.set('name', 'Charlie')
    evolver.set('age', 30)
    evolver.set('score', -5)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'score.negative' in e.invariant_errors

    # Test non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('nonexistent', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'nonexistent' in str(e)

    # Test factory transformation
    class TestRecordWithFactory(PRecord):
        items = field(type=list, factory=lambda x: x if isinstance(x, list) else [x])

    evolver = TestRecordWithFactory().evolver()
    evolver.set('items', 'single')
    result = evolver.persistent()
    assert result['items'] == ['single']

    # Test with _factory_fields restriction
    class TestRecordFactoryFields(PRecord):
        a = field(type=int)
        b = field(type=int)

    evolver = TestRecordFactoryFields(_factory_fields=[TestRecordFactoryFields._precord_fields['a']]).evolver()
    evolver.set('a', 1)  # Should apply factory
    evolver.set('b', 2)  # Should not apply factory
    result = evolver.persistent()
    assert result['a'] == 1
    assert result['b'] == 2

    # Test mandatory field check
    evolver = TestRecord().evolver()
    evolver.set('name', 'David')
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'TestRecord.age' in e.missing_fields

    # Test multiple invariant failures accumulate
    class MultiInvariantRecord(PRecord):
        x = field(type=int, invariant=lambda v: (v > 0, 'x.positive'))
        y = field(type=int, invariant=lambda v: (v < 10, 'y.small'))

    evolver = MultiInvariantRecord().evolver()
    evolver.set('x', -1)
    evolver.set('y', 20)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 2
        assert 'x.positive' in e.invariant_errors
        assert 'y.small' in e.invariant_errors


# LLM-generated content at query #3
#--------------------------

```python
def test_PRecord___new__():
    # Test basic PRecord creation with fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        active = field(type=bool, initial=lambda: True)
    
    # Test 1: Create record with valid data
    record = TestRecord(name="Alice", age=30)
    assert record["name"] == "Alice"
    assert record["age"] == 30
    assert record["active"] == True
    
    # Test 2: Initial values should be applied when not provided
    record2 = TestRecord(name="Bob", age=25)
    assert record2["active"] == True
    
    # Test 3: Override initial value
    record3 = TestRecord(name="Charlie", age=35, active=False)
    assert record3["active"] == False
    
    # Test 4: Missing mandatory field should raise error
    try:
        TestRecord(name="David")
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "TestRecord.age" in e.missing_fields
    
    # Test 5: Create with _factory_fields parameter
    record5 = TestRecord(name="Eve", age=40, _factory_fields=None)
    assert record5["name"] == "Eve"
    assert record5["age"] == 40
    
    # Test 6: Create with _ignore_extra parameter
    record6 = TestRecord(name="Frank", age=45, _ignore_extra=False)
    assert record6["name"] == "Frank"
    assert record6["age"] == 45
    
    # Test 7: Direct creation with internal parameters
    record7 = TestRecord(_precord_size=2, _precord_buckets=pmap({"name": "Grace", "age": 50})._buckets)
    assert record7["name"] == "Grace"
    assert record7["age"] == 50
    
    # Test 8: Type checking should work
    try:
        TestRecord(name="Henry", age="not_an_int")
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass
    
    # Test 9: Create empty record with only initial values
    class SimpleRecord(PRecord):
        value = field(type=int, initial=42)
    
    record9 = SimpleRecord()
    assert record9["value"] == 42
    
    # Test 10: Callable initial should be called
    call_count = []
    def counter():
        call_count.append(1)
        return 100
    
    class CallableRecord(PRecord):
        count = field(type=int, initial=counter)
    
    record10 = CallableRecord()
    assert record10["count"] == 100
    assert len(call_count) == 1


# LLM-generated content at query #4
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    import pytest

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        score = field(type=float, invariant=lambda x: (x >= 0, 'score must be non-negative'))

    # Test basic field setting
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    evolver.set('score', 95.5)
    result = evolver.persistent()
    assert result['name'] == 'Alice'
    assert result['age'] == 25
    assert result['score'] == 95.5

    # Test type checking
    evolver = TestRecord().evolver()
    with pytest.raises(TypeError):
        evolver.set('age', 'not_an_int')

    # Test invariant violation
    evolver = TestRecord().evolver()
    evolver.set('age', 30)
    evolver.set('score', -5.0)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'score must be non-negative' in str(exc_info.value)

    # Test setting non-existent field
    evolver = TestRecord().evolver()
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('nonexistent', 'value')
    assert "'nonexistent' is not among the specified fields" in str(exc_info.value)

    # Test with factory fields
    class RecordWithFactory(PRecord):
        items = field(type=list, factory=lambda x: list(x) if x else [])

    evolver = RecordWithFactory().evolver()
    evolver.set('items', [1, 2, 3])
    result = evolver.persistent()
    assert result['items'] == [1, 2, 3]

    # Test missing mandatory field
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    evolver.set('score', 80.0)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'missing_fields' in str(exc_info.value)

    # Test multiple invariant violations accumulate
    class MultiInvariantRecord(PRecord):
        value1 = field(type=int, invariant=lambda x: (x > 0, 'value1 must be positive'))
        value2 = field(type=int, invariant=lambda x: (x < 100, 'value2 must be less than 100'))

    evolver = MultiInvariantRecord().evolver()
    evolver.set('value1', -5)
    evolver.set('value2', 200)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'value1 must be positive' in str(exc_info.value)
    assert 'value2 must be less than 100' in str(exc_info.value)

    # Test that __setitem__ also works
    evolver = TestRecord().evolver()
    evolver['name'] = 'Charlie'
    evolver['age'] = 40
    result = evolver.persistent()
    assert result['name'] == 'Charlie'
    assert result['age'] == 40


# LLM-generated content at query #5
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    from pyrsistent._checked_types import CheckedType

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        score = field(type=float, invariant=lambda x: (x >= 0, 'score_negative'))

    # Test basic field setting
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    evolver.set('score', 95.5)
    result = evolver.persistent()
    assert result['name'] == 'Alice'
    assert result['age'] == 25
    assert result['score'] == 95.5

    # Test type checking
    evolver = TestRecord().evolver()
    evolver.set('age', 30)  # Valid int
    try:
        evolver.set('age', 'thirty')  # Invalid type
        assert False, "Should have raised type error"
    except TypeError:
        pass

    # Test invariant violation
    evolver = TestRecord().evolver()
    evolver.set('age', 30)
    evolver.set('score', -5.0)  # Violates invariant
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'score_negative' in str(e)

    # Test setting non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('nonexistent', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'nonexistent' is not among the specified fields" in str(e)

    # Test mandatory field missing
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    evolver.set('score', 80.0)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'age' in str(e)

    # Test with factory fields
    class RecordWithFactory(PRecord):
        items = field(type=list, factory=lambda x: x if isinstance(x, list) else [x])

    evolver = RecordWithFactory().evolver()
    evolver.set('items', 42)  # Factory should convert to list
    result = evolver.persistent()
    assert result['items'] == [42]

    # Test multiple invariant violations accumulate
    class MultiInvariant(PRecord):
        value = field(type=int, invariant=lambda x: (x > 0, 'positive'))
        other = field(type=int, invariant=lambda x: (x < 10, 'less_than_ten'))

    evolver = MultiInvariant().evolver()
    evolver.set('value', -5)  # First violation
    evolver.set('other', 15)  # Second violation
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'positive' in str(e)
        assert 'less_than_ten' in str(e)

    # Test that set returns self for chaining
    evolver = TestRecord().evolver()
    returned = evolver.set('name', 'Charlie')
    assert returned is evolver


# LLM-generated content at query #6
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: (len(self.name) > 0, "name_non_empty")
        name = field(type=str, mandatory=True)
        age = field(type=int, initial=25)
        active = field(type=bool, initial=lambda: True)

    assert hasattr(TestRecord, '_precord_fields')
    assert isinstance(TestRecord._precord_fields, dict)
    assert set(TestRecord._precord_fields.keys()) == {'name', 'age', 'active'}
    
    assert TestRecord._precord_fields['name'].mandatory is True
    assert TestRecord._precord_fields['age'].mandatory is False
    assert TestRecord._precord_fields['active'].mandatory is False
    
    assert TestRecord._precord_fields['age'].initial == 25
    assert callable(TestRecord._precord_fields['active'].initial)
    assert TestRecord._precord_fields['active'].initial() is True
    
    assert TestRecord._precord_mandatory_fields == {'name'}
    assert TestRecord._precord_initial_values == {'age': 25, 'active': TestRecord._precord_fields['active'].initial}
    
    assert hasattr(TestRecord, '_precord_invariants')
    assert len(TestRecord._precord_invariants) == 1
    
    assert TestRecord.__slots__ == ()
    
    class EmptyRecord(PRecord):
        pass
    
    assert hasattr(EmptyRecord, '_precord_fields')
    assert EmptyRecord._precord_fields == {}
    assert EmptyRecord._precord_mandatory_fields == set()
    assert EmptyRecord._precord_initial_values == {}
    assert EmptyRecord._precord_invariants == []


# LLM-generated content at query #7
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic class creation with fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        active = field(type=bool, initial=True)
    
    # Verify _precord_fields is set
    assert hasattr(TestRecord, '_precord_fields')
    assert isinstance(TestRecord._precord_fields, dict)
    assert set(TestRecord._precord_fields.keys()) == {'name', 'age', 'active'}
    
    # Verify field types are correct
    name_field = TestRecord._precord_fields['name']
    age_field = TestRecord._precord_fields['age']
    active_field = TestRecord._precord_fields['active']
    
    assert name_field.type == str
    assert age_field.type == int
    assert active_field.type == bool
    
    # Verify mandatory fields
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert isinstance(TestRecord._precord_mandatory_fields, set)
    assert TestRecord._precord_mandatory_fields == {'age'}
    
    # Verify initial values
    assert hasattr(TestRecord, '_precord_initial_values')
    assert isinstance(TestRecord._precord_initial_values, dict)
    assert 'active' in TestRecord._precord_initial_values
    assert TestRecord._precord_initial_values['active'] is True
    
    # Verify invariants storage
    assert hasattr(TestRecord, '_precord_invariants')
    assert isinstance(TestRecord._precord_invariants, tuple)
    
    # Test class with invariants
    class RecordWithInvariant(PRecord):
        value = field(type=int)
        
        @invariant(lambda r: r.value >= 0)
        def value_non_negative(self):
            return self.value >= 0
    
    assert len(RecordWithInvariant._precord_invariants) == 1
    
    # Test inheritance - child class should have parent's fields
    class ParentRecord(PRecord):
        parent_field = field(type=str)
    
    class ChildRecord(ParentRecord):
        child_field = field(type=int)
    
    assert set(ChildRecord._precord_fields.keys()) == {'parent_field', 'child_field'}
    
    # Test that __slots__ is set to empty tuple
    assert TestRecord.__slots__ == ()
    assert RecordWithInvariant.__slots__ == ()
    assert ChildRecord.__slots__ == ()
    
    # Test with multiple inheritance
    class Mixin:
        pass
    
    class MultiRecord(PRecord, Mixin):
        field1 = field(type=str)
    
    assert hasattr(MultiRecord, '_precord_fields')
    assert 'field1' in MultiRecord._precord_fields
    
    # Test class without fields
    class EmptyRecord(PRecord):
        pass
    
    assert EmptyRecord._precord_fields == {}
    assert EmptyRecord._precord_mandatory_fields == set()
    assert EmptyRecord._precord_initial_values == {}


# LLM-generated content at query #8
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic class creation with fields
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int, initial=25)
        active = field(type=bool, initial=lambda: True)

    # Check that fields are properly set
    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'active' in TestRecord._precord_fields
    
    # Check field properties
    name_field = TestRecord._precord_fields['name']
    assert name_field.mandatory is True
    assert name_field.initial is PFIELD_NO_INITIAL
    
    age_field = TestRecord._precord_fields['age']
    assert age_field.mandatory is False
    assert age_field.initial == 25
    
    active_field = TestRecord._precord_fields['active']
    assert active_field.mandatory is False
    assert callable(active_field.initial)
    assert active_field.initial() is True
    
    # Check mandatory fields set
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert TestRecord._precord_mandatory_fields == {'name'}
    
    # Check initial values dict
    assert hasattr(TestRecord, '_precord_initial_values')
    assert 'age' in TestRecord._precord_initial_values
    assert 'active' in TestRecord._precord_initial_values
    assert TestRecord._precord_initial_values['age'] == 25
    assert callable(TestRecord._precord_initial_values['active'])
    
    # Check invariants storage
    assert hasattr(TestRecord, '_precord_invariants')
    assert isinstance(TestRecord._precord_invariants, tuple)
    
    # Test inheritance with new fields
    class ChildRecord(TestRecord):
        email = field(type=str)
    
    assert 'email' in ChildRecord._precord_fields
    assert 'name' in ChildRecord._precord_fields
    assert 'age' in ChildRecord._precord_fields
    assert 'active' in ChildRecord._precord_fields
    
    # Test that mandatory fields are inherited
    assert ChildRecord._precord_mandatory_fields == {'name'}
    
    # Test with invariant method
    class RecordWithInvariant(PRecord):
        value = field(type=int)
        
        @invariant(lambda x: x.value > 0)
        def value_positive(self):
            return self.value > 0
    
    assert len(RecordWithInvariant._precord_invariants) == 1
    
    # Test empty record
    class EmptyRecord(PRecord):
        pass
    
    assert EmptyRecord._precord_fields == {}
    assert EmptyRecord._precord_mandatory_fields == set()
    assert EmptyRecord._precord_initial_values == {}
    
    # Test slots are set
    assert TestRecord.__slots__ == ()
    assert ChildRecord.__slots__ == ()
    assert EmptyRecord.__slots__ == ()


# LLM-generated content at query #9
#--------------------------

```python
def test_PRecord_serialize():
    from pyrsistent import field
    
    class TestRecord(PRecord):
        name = field()
        age = field(type=int)
        tags = field(factory=list, serializer=lambda v, f: [t.upper() for t in v])
    
    # Test basic serialization without custom serializer
    record1 = TestRecord(name="Alice", age=30, tags=["a", "b"])
    result1 = record1.serialize()
    assert result1 == {"name": "Alice", "age": 30, "tags": ["A", "B"]}
    
    # Test serialization with format parameter
    record2 = TestRecord(name="Bob", age=25, tags=["x", "y"])
    result2 = record2.serialize(format="json")
    assert result2 == {"name": "Bob", "age": 25, "tags": ["X", "Y"]}
    
    # Test serialization with field that has no serializer
    class SimpleRecord(PRecord):
        value = field()
        count = field(type=int)
    
    record3 = SimpleRecord(value="test", count=5)
    result3 = record3.serialize()
    assert result3 == {"value": "test", "count": 5}
    
    # Test serialization with complex serializer
    class ComplexRecord(PRecord):
        data = field(serializer=lambda v, f: {"formatted": str(v)})
    
    record4 = ComplexRecord(data={"key": "value"})
    result4 = record4.serialize()
    assert result4 == {"data": {"formatted": "{'key': 'value'}"}}
    
    # Test serialization with multiple fields having serializers
    class MultiSerializerRecord(PRecord):
        items = field(factory=list, serializer=lambda v, f: len(v))
        price = field(type=float, serializer=lambda v, f: f"${v:.2f}")
    
    record5 = MultiSerializerRecord(items=["a", "b", "c"], price=19.99)
    result5 = record5.serialize()
    assert result5 == {"items": 3, "price": "$19.99"}
    
    # Test serialization on empty record
    class EmptyRecord(PRecord):
        pass
    
    record6 = EmptyRecord()
    result6 = record6.serialize()
    assert result6 == {}


# LLM-generated content at query #10
#--------------------------

```python
def test__PRecordEvolver_persistent():
    from pyrsistent import field, PRecord, InvariantException
    
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int, invariant=lambda x: (x >= 0, 'Age must be non-negative'))
        optional = field(type=str, initial='default')
    
    # Test 1: Basic persistence with valid data
    evolver = TestRecord(name='Alice', age=30).evolver()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['name'] == 'Alice'
    assert result['age'] == 30
    assert result['optional'] == 'default'
    
    # Test 2: Persistence after modification
    evolver = TestRecord(name='Bob', age=25).evolver()
    evolver['age'] = 26
    result = evolver.persistent()
    assert result['age'] == 26
    
    # Test 3: Missing mandatory field should raise InvariantException
    evolver = _PRecordEvolver(TestRecord, pmap({'age': 30}))
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.name' in e.missing_fields
    
    # Test 4: Field invariant violation should raise InvariantException
    evolver = TestRecord(name='Charlie', age=30).evolver()
    evolver['age'] = -5
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Age must be non-negative' in e.invariant_errors
    
    # Test 5: Multiple invariant violations accumulate
    class MultiInvariantRecord(PRecord):
        value1 = field(type=int, invariant=lambda x: (x > 0, 'value1 > 0'))
        value2 = field(type=int, invariant=lambda x: (x < 10, 'value2 < 10'))
    
    evolver = MultiInvariantRecord(value1=5, value2=5).evolver()
    evolver['value1'] = -1
    evolver['value2'] = 15
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 2
        assert 'value1 > 0' in e.invariant_errors
        assert 'value2 < 10' in e.invariant_errors
    
    # Test 6: Clean evolver returns same instance
    record = TestRecord(name='David', age=40)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record
    
    # Test 7: Dirty evolver returns new instance
    record = TestRecord(name='Eve', age=35)
    evolver = record.evolver()
    evolver['age'] = 36
    result = evolver.persistent()
    assert result is not record
    assert result['age'] == 36
    
    # Test 8: Global invariants are checked
    class GlobalInvariantRecord(PRecord):
        x = field(type=int)
        y = field(type=int)
        
        @staticmethod
        def __invariant__(rec):
            return rec['x'] <= rec['y'], 'x must be <= y'
    
    evolver = GlobalInvariantRecord(x=5, y=3).evolver()
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'x must be <= y' in e.invariant_errors
    
    # Test 9: Factory fields handling
    class FactoryRecord(PRecord):
        items = field(type=list, factory=lambda x: list(x))
    
    evolver = FactoryRecord(items=[1, 2, 3]).evolver()
    evolver['items'] = [4, 5, 6]
    result = evolver.persistent()
    assert result['items'] == [4, 5, 6]
    
    # Test 10: Empty record with all optional fields
    class OptionalRecord(PRecord):
        field1 = field(type=str, initial='default1')
        field2 = field(type=int, initial=42)
    
    evolver = _PRecordEvolver(OptionalRecord, pmap({}))
    result = evolver.persistent()
    assert result['field1'] == 'default1'
    assert result['field2'] == 42


# LLM-generated content at query #11
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    from pyrsistent._checked_types import CheckedType

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, invariant=lambda x: (x >= 0, 'age_negative'))
        optional = field(type=str, mandatory=False)

    # Test basic field setting
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 30)
    record = evolver.persistent()
    assert record['name'] == 'Alice'
    assert record['age'] == 30

    # Test type checking
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    try:
        evolver.set('age', 'not_an_int')
        assert False, "Should have raised type error"
    except TypeError:
        pass

    # Test invariant violation
    evolver = TestRecord().evolver()
    evolver.set('name', 'Charlie')
    evolver.set('age', -5)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'age_negative' in str(e)

    # Test setting non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('nonexistent', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'nonexistent' is not among the specified fields" in str(e)

    # Test multiple invariant violations accumulate
    evolver = TestRecord().evolver()
    evolver.set('age', -1)
    evolver.set('age', -2)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 2

    # Test with factory fields
    class TestRecord2(PRecord):
        items = field(type=list, factory=lambda x: list(x))

    evolver = TestRecord2().evolver()
    evolver.set('items', (1, 2, 3))
    record = evolver.persistent()
    assert record['items'] == [1, 2, 3]

    # Test ignore_extra parameter
    class TestRecord3(PRecord):
        data = field(type=CheckedType, factory=dict)

    evolver = TestRecord3(_ignore_extra=True).evolver()
    evolver.set('data', {'extra': 'ignored'})
    record = evolver.persistent()
    assert isinstance(record['data'], dict)

    # Test that setting same value doesn't cause issues
    evolver = TestRecord(name='Dave', age=25).evolver()
    evolver.set('name', 'Dave')
    evolver.set('age', 25)
    record = evolver.persistent()
    assert record['name'] == 'Dave'
    assert record['age'] == 25

    # Test __setitem__ alias works
    evolver = TestRecord().evolver()
    evolver['name'] = 'Eve'
    evolver['age'] = 28
    record = evolver.persistent()
    assert record['name'] == 'Eve'
    assert record['age'] == 28


# LLM-generated content at query #12
#--------------------------

```python
def test_PRecord_serialize():
    from pyrsistent import field
    
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        data = field(type=dict, serializer=lambda v, f: {"custom": v})
    
    # Test basic serialization without format
    record = TestRecord(name="Alice", age=30, data={"key": "value"})
    result = record.serialize()
    assert result == {
        "name": "Alice",
        "age": 30,
        "data": {"custom": {"key": "value"}}
    }
    
    # Test serialization with format parameter
    class FormatRecord(PRecord):
        value = field(type=str, serializer=lambda v, f: f"{f}:{v}")
    
    format_record = FormatRecord(value="test")
    result = format_record.serialize(format="prefix")
    assert result == {"value": "prefix:test"}
    
    # Test serialization with None format
    class NoneFormatRecord(PRecord):
        item = field(type=int, serializer=lambda v, f: v * 2)
    
    none_record = NoneFormatRecord(item=5)
    result = none_record.serialize()
    assert result == {"item": 10}
    
    # Test serialization with multiple fields having different serializers
    class MultiRecord(PRecord):
        a = field(type=int, serializer=lambda v, f: v + 1)
        b = field(type=str, serializer=lambda v, f: v.upper())
        c = field(type=list)  # No serializer
    
    multi = MultiRecord(a=10, b="hello", c=[1, 2, 3])
    result = multi.serialize()
    assert result == {"a": 11, "b": "HELLO", "c": [1, 2, 3]}
    
    # Test empty record serialization
    class EmptyRecord(PRecord):
        pass
    
    empty = EmptyRecord()
    result = empty.serialize()
    assert result == {}
    
    # Test that serializer receives None format when not provided
    class CheckFormatRecord(PRecord):
        test = field(type=str, serializer=lambda v, f: f is None)
    
    check = CheckFormatRecord(test="anything")
    result = check.serialize()
    assert result == {"test": True}


# LLM-generated content at query #13
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: (len(self.name) > 0, "name_non_empty")
        name = field(type=str, mandatory=True)
        age = field(type=int, initial=25)
        optional = field(type=str)

    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'optional' in TestRecord._precord_fields
    
    assert TestRecord._precord_fields['name'].mandatory is True
    assert TestRecord._precord_fields['age'].mandatory is False
    assert TestRecord._precord_fields['optional'].mandatory is False
    
    assert TestRecord._precord_mandatory_fields == {'name'}
    
    assert TestRecord._precord_initial_values['age'] == 25
    assert 'name' not in TestRecord._precord_initial_values
    assert 'optional' not in TestRecord._precord_initial_values
    
    assert hasattr(TestRecord, '_precord_invariants')
    assert len(TestRecord._precord_invariants) == 1
    
    assert TestRecord.__slots__ == ()
    
    class ChildRecord(TestRecord):
        score = field(type=float, initial=0.0)
    
    assert 'score' in ChildRecord._precord_fields
    assert ChildRecord._precord_fields['score'].type == float
    assert ChildRecord._precord_mandatory_fields == {'name'}
    assert ChildRecord._precord_initial_values['age'] == 25
    assert ChildRecord._precord_initial_values['score'] == 0.0
    assert len(ChildRecord._precord_invariants) == 1


# LLM-generated content at query #14
#--------------------------

```python
def test_PRecord___new__():
    # Test basic record creation with field values
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
    
    r = TestRecord(name="Alice", age=30)
    assert r["name"] == "Alice"
    assert r["age"] == 30
    assert isinstance(r, TestRecord)
    
    # Test with initial values
    class RecordWithInitial(PRecord):
        name = field(type=str, initial="Unknown")
        age = field(type=int, initial=0)
    
    r1 = RecordWithInitial()
    assert r1["name"] == "Unknown"
    assert r1["age"] == 0
    
    r2 = RecordWithInitial(name="Bob", age=25)
    assert r2["name"] == "Bob"
    assert r2["age"] == 25
    
    # Test with callable initial
    counter = 0
    def get_id():
        nonlocal counter
        counter += 1
        return counter
    
    class RecordWithCallableInitial(PRecord):
        id = field(type=int, initial=get_id)
        value = field(type=str)
    
    r3 = RecordWithCallableInitial(value="test")
    assert r3["id"] == 1
    assert r3["value"] == "test"
    
    r4 = RecordWithCallableInitial(value="test2")
    assert r4["id"] == 2
    assert r4["value"] == "test2"
    
    # Test with factory fields
    class RecordWithFactory(PRecord):
        items = field(type=PSet, factory=pset)
    
    r5 = RecordWithFactory(items=[1, 2, 3])
    assert isinstance(r5["items"], PSet)
    assert 1 in r5["items"]
    assert 2 in r5["items"]
    assert 3 in r5["items"]
    
    # Test ignore_extra parameter
    class SimpleRecord(PRecord):
        field1 = field(type=str)
    
    r6 = SimpleRecord(field1="value1", _ignore_extra=True)
    assert r6["field1"] == "value1"
    
    # Test with internal parameters for reconstruction
    buckets = pmap({"name": "Charlie", "age": 40})._buckets
    r7 = TestRecord(_precord_buckets=buckets, _precord_size=2)
    assert r7["name"] == "Charlie"
    assert r7["age"] == 40
    assert isinstance(r7, TestRecord)
    
    # Test mandatory fields
    class MandatoryRecord(PRecord):
        required = field(type=str, mandatory=True)
        optional = field(type=str)
    
    try:
        MandatoryRecord(optional="test")
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "MandatoryRecord.required" in e.missing_fields
    
    r8 = MandatoryRecord(required="present", optional="test")
    assert r8["required"] == "present"
    assert r8["optional"] == "test"
    
    # Test type checking
    class TypedRecord(PRecord):
        count = field(type=int)
    
    try:
        TypedRecord(count="not an int")
        assert False, "Should have raised type error"
    except TypeError:
        pass
    
    # Test with factory_fields parameter
    class ComplexRecord(PRecord):
        data = field(type=PMap, factory=pmap)
        value = field(type=int)
    
    factory_fields = {ComplexRecord._precord_fields["data"]}
    r9 = ComplexRecord(_factory_fields=factory_fields, data={"key": "value"}, value=42)
    assert isinstance(r9["data"], PMap)
    assert r9["data"]["key"] == "value"
    assert r9["value"] == 42


# LLM-generated content at query #15
#--------------------------

```python
def test__PRecordEvolver_persistent():
    from pyrsistent import field, PRecord, InvariantException
    
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int, invariant=lambda x: (x >= 0, 'age.negative'))
        optional = field(type=str, initial='default')
    
    # Test basic persistence with valid data
    evolver = TestRecord._PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['name'] == 'Alice'
    assert result['age'] == 25
    assert result['optional'] == 'default'
    
    # Test persistence with existing pmap
    existing = TestRecord(name='Bob', age=30)
    evolver = existing.evolver()
    evolver.set('age', 31)
    result = evolver.persistent()
    assert result['name'] == 'Bob'
    assert result['age'] == 31
    
    # Test invariant violation
    evolver = TestRecord._PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Charlie')
    evolver.set('age', -5)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'age.negative' in e.invariant_errors
    
    # Test missing mandatory field
    evolver = TestRecord._PRecordEvolver(TestRecord, pmap())
    evolver.set('age', 20)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.name' in e.missing_fields
    
    # Test multiple errors
    evolver = TestRecord._PRecordEvolver(TestRecord, pmap())
    evolver.set('age', -10)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'age.negative' in e.invariant_errors
        assert 'TestRecord.name' in e.missing_fields
    
    # Test global invariants
    class RecordWithGlobal(PRecord):
        x = field(type=int)
        y = field(type=int)
        
        @staticmethod
        def __invariant__(rec):
            return rec['x'] <= rec['y'], 'x_gt_y'
    
    evolver = RecordWithGlobal._PRecordEvolver(RecordWithGlobal, pmap())
    evolver.set('x', 10)
    evolver.set('y', 5)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'x_gt_y' in e.invariant_errors
    
    # Test clean persistence without changes
    record = TestRecord(name='Dave', age=40)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record
    
    # Test dirty persistence with factory fields
    class RecordWithFactory(PRecord):
        items = field(type=list, factory=lambda x: list(x))
    
    evolver = RecordWithFactory._PRecordEvolver(
        RecordWithFactory, 
        pmap(),
        _factory_fields={RecordWithFactory.items}
    )
    evolver.set('items', [1, 2, 3])
    result = evolver.persistent()
    assert result['items'] == [1, 2, 3]


# LLM-generated content at query #16
#--------------------------

```python
def test_PRecord___new__():
    # Test basic record creation with field values
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
    
    r = TestRecord(name="Alice", age=30)
    assert r["name"] == "Alice"
    assert r["age"] == 30
    assert isinstance(r, TestRecord)
    
    # Test creation with initial values
    class RecordWithInitial(PRecord):
        name = field(type=str, initial="Unknown")
        age = field(type=int, initial=0)
    
    r1 = RecordWithInitial()
    assert r1["name"] == "Unknown"
    assert r1["age"] == 0
    
    r2 = RecordWithInitial(name="Bob", age=25)
    assert r2["name"] == "Bob"
    assert r2["age"] == 25
    
    # Test creation with factory fields
    class RecordWithFactory(PRecord):
        items = field(type=PSet, factory=pset)
    
    r = RecordWithFactory(items=[1, 2, 3])
    assert isinstance(r["items"], PSet)
    assert 1 in r["items"]
    
    # Test creation with ignore_extra
    class SimpleRecord(PRecord):
        x = field(type=int)
        y = field(type=int)
    
    r = SimpleRecord.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert r["x"] == 1
    assert r["y"] == 2
    assert "z" not in r
    
    # Test creation from existing record
    original = SimpleRecord(x=1, y=2)
    created = SimpleRecord.create(original)
    assert created is original
    
    # Test creation with internal fields (_precord_size, _precord_buckets)
    buckets = ((0, ("key", "value")),)
    r = SimpleRecord(_precord_size=1, _precord_buckets=buckets)
    assert r["key"] == "value"
    
    # Test creation with missing mandatory fields raises error
    class MandatoryRecord(PRecord):
        required = field(type=str, mandatory=True)
        optional = field(type=int)
    
    try:
        MandatoryRecord(optional=5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "MandatoryRecord.required" in str(e.missing_fields)
    
    # Test creation with field type checking
    class TypedRecord(PRecord):
        count = field(type=int)
    
    try:
        TypedRecord(count="not_an_int")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test creation with field invariants
    class PositiveRecord(PRecord):
        value = field(type=int, invariant=lambda x: (x > 0, "value.positive"))
    
    r = PositiveRecord(value=10)
    assert r["value"] == 10
    
    try:
        PositiveRecord(value=-5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "value.positive" in str(e.invariant_errors)
    
    # Test creation with global invariants
    class BalancedRecord(PRecord):
        left = field(type=int)
        right = field(type=int)
        
        @invariant
        def balanced(self):
            return (abs(self["left"] - self["right"]) <= 1, "balanced")
    
    r = BalancedRecord(left=5, right=5)
    assert r["left"] == 5
    assert r["right"] == 5
    
    try:
        BalancedRecord(left=10, right=5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "balanced" in str(e.invariant_errors)
    
    # Test creation with callable initial values
    class CallableInitialRecord(PRecord):
        timestamp = field(type=float, initial=lambda: 100.0)
        counter = field(type=int, initial=0)
    
    r = CallableInitialRecord()
    assert r["timestamp"] == 100.0
    assert r["counter"] == 0
    
    # Test creation with factory_fields parameter
    class PartialRecord(PRecord):
        a = field(type=int, factory=lambda x: x * 2)
        b = field(type=int)
    
    r = PartialRecord.create({"a": 5, "b": 10}, _factory_fields={PartialRecord._precord_fields["a"]})
    assert r["a"] == 10  # Factory applied
    assert r["b"] == 10  # No factory applied
    
    # Test creation with ignore_extra parameter
    class StrictRecord(PRecord):
        field1 = field(type=str)
    
    try:
        StrictRecord(field1="test", extra="should_fail")
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "extra" in str(e)
    
    # Test that evolver is used for creation
    r = TestRecord(name="Test", age=42)
    assert isinstance(r, TestRecord)
    assert r._precord_size == 2


# LLM-generated content at query #17
#--------------------------

```python
def test__PRecordEvolver_persistent():
    from pyrsistent import field, PRecord, InvariantException
    
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int, invariant=lambda x: (x >= 0, 'Age must be non-negative'))
        optional = field(type=str, initial='default')
    
    # Test 1: Basic persistent creation
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 30)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['name'] == 'Alice'
    assert result['age'] == 30
    assert result['optional'] == 'default'
    
    # Test 2: Missing mandatory field should raise InvariantException
    evolver = TestRecord().evolver()
    evolver.set('age', 25)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.name' in e.missing_fields
    
    # Test 3: Field invariant violation should raise InvariantException
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    evolver.set('age', -5)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Age must be non-negative' in e.invariant_errors
    
    # Test 4: Multiple invariant violations accumulate
    class MultiInvariantRecord(PRecord):
        value1 = field(type=int, invariant=lambda x: (x > 0, 'value1 must be positive'))
        value2 = field(type=int, invariant=lambda x: (x < 10, 'value2 must be less than 10'))
    
    evolver = MultiInvariantRecord().evolver()
    evolver.set('value1', -1)
    evolver.set('value2', 20)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 2
        assert 'value1 must be positive' in e.invariant_errors
        assert 'value2 must be less than 10' in e.invariant_errors
    
    # Test 5: Clean evolver (no changes) should return original instance
    record = TestRecord(name='Charlie', age=40)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record
    
    # Test 6: Dirty evolver should return new instance
    record = TestRecord(name='David', age=35)
    evolver = record.evolver()
    evolver.set('age', 36)
    result = evolver.persistent()
    assert result is not record
    assert result['name'] == 'David'
    assert result['age'] == 36
    
    # Test 7: Global invariants are checked
    class GlobalInvariantRecord(PRecord):
        x = field(type=int)
        y = field(type=int)
        
        @staticmethod
        def __invariant__(x, y):
            return (x + y == 100, 'Sum must be 100')
    
    evolver = GlobalInvariantRecord().evolver()
    evolver.set('x', 30)
    evolver.set('y', 80)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Sum must be 100' in str(e)
    
    # Test 8: Factory fields handling
    class FactoryRecord(PRecord):
        items = field(type=list, factory=lambda x: list(x))
    
    evolver = FactoryRecord().evolver()
    evolver.set('items', [1, 2, 3])
    result = evolver.persistent()
    assert result['items'] == [1, 2, 3]
    
    # Test 9: Missing fields and invariant errors combined
    evolver = TestRecord().evolver()
    evolver.set('age', -10)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.name' in e.missing_fields
        assert 'Age must be non-negative' in e.invariant_errors


# LLM-generated content at query #18
#--------------------------

```python
def test_PRecord___repr__():
    from pyrsistent import PRecord, field

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        active = field(type=bool, initial=True)

    # Test basic repr with all fields
    record1 = TestRecord(name="Alice", age=30)
    assert repr(record1) == "TestRecord(name='Alice', age=30, active=True)"

    # Test repr with different field values
    record2 = TestRecord(name="Bob", age=25, active=False)
    assert repr(record2) == "TestRecord(name='Bob', age=25, active=False)"

    # Test repr with special characters in string
    record3 = TestRecord(name="O'Connor", age=40)
    assert repr(record3) == "TestRecord(name=\"O'Connor\", age=40, active=True)"

    # Test repr with empty string
    record4 = TestRecord(name="", age=0)
    assert repr(record4) == "TestRecord(name='', age=0, active=True)"

    # Test repr with None value
    class OptionalRecord(PRecord):
        value = field(type=str, mandatory=False)
        count = field(type=int, initial=0)

    record5 = OptionalRecord()
    assert repr(record5) == "OptionalRecord(count=0)"

    record6 = OptionalRecord(value="test")
    assert repr(record6) == "OptionalRecord(value='test', count=0)"

    # Test repr maintains field order as defined in class
    class MultiFieldRecord(PRecord):
        z = field(type=int)
        a = field(type=str)
        m = field(type=float)

    record7 = MultiFieldRecord(z=3, a="test", m=1.5)
    assert repr(record7) == "MultiFieldRecord(z=3, a='test', m=1.5)"

    # Test repr with numeric values
    class NumericRecord(PRecord):
        integer = field(type=int)
        floating = field(type=float)
        boolean = field(type=bool)

    record8 = NumericRecord(integer=42, floating=3.14, boolean=True)
    assert repr(record8) == "NumericRecord(integer=42, floating=3.14, boolean=True)"

    # Test repr with updated record
    record9 = TestRecord(name="Charlie", age=35)
    updated_record = record9.set(age=36)
    assert repr(updated_record) == "TestRecord(name='Charlie', age=36, active=True)"


# LLM-generated content at query #19
#--------------------------

```python
def test_PRecord___new__():
    # Test basic record creation with fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
    
    # Test creation with initial values
    record = TestRecord(name="Alice", age=30)
    assert record["name"] == "Alice"
    assert record["age"] == 30
    assert isinstance(record, TestRecord)
    
    # Test creation with missing mandatory field
    try:
        TestRecord(name="Bob")
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "TestRecord.age" in e.missing_fields
    
    # Test creation with field type checking
    try:
        TestRecord(name="Charlie", age="thirty")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test creation with internal fields for pickling
    record2 = TestRecord(_precord_size=2, _precord_buckets=record._buckets)
    assert record2 == record
    assert isinstance(record2, TestRecord)
    
    # Test creation with factory fields
    class ComplexRecord(PRecord):
        items = field(type=PSet, factory=pset)
    
    record3 = ComplexRecord(items=[1, 2, 3])
    assert isinstance(record3["items"], PSet)
    assert 1 in record3["items"]
    
    # Test creation with ignore_extra
    class SimpleRecord(PRecord):
        x = field(type=int)
    
    record4 = SimpleRecord(x=1, y=2, _ignore_extra=True)
    assert record4["x"] == 1
    assert "y" not in record4
    
    # Test creation without ignore_extra should raise AttributeError
    try:
        SimpleRecord(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'y' is not among the specified fields" in str(e)
    
    # Test creation with callable initial values
    class DefaultRecord(PRecord):
        timestamp = field(type=float, initial=lambda: 123.45)
        value = field(type=int, initial=42)
    
    record5 = DefaultRecord()
    assert record5["timestamp"] == 123.45
    assert record5["value"] == 42
    
    record6 = DefaultRecord(timestamp=999.99, value=100)
    assert record6["timestamp"] == 999.99
    assert record6["value"] == 100
    
    # Test creation with factory_fields parameter
    class FactoryRecord(PRecord):
        a = field(type=int, factory=lambda x: x * 2)
        b = field(type=int)
    
    record7 = FactoryRecord(a=5, b=10, _factory_fields=[FactoryRecord.a])
    assert record7["a"] == 10  # Factory applied
    assert record7["b"] == 10  # No factory applied
    
    # Test creation with empty record
    class EmptyRecord(PRecord):
        pass
    
    record8 = EmptyRecord()
    assert len(record8) == 0
    assert isinstance(record8, EmptyRecord)


# LLM-generated content at query #20
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    import pytest

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        score = field(type=float, invariant=lambda x: (x >= 0, 'score must be non-negative'))

    # Test basic field setting with type checking
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    evolver.set('score', 95.5)
    result = evolver.persistent()
    assert result['name'] == 'Alice'
    assert result['age'] == 25
    assert result['score'] == 95.5

    # Test type checking failure
    evolver = TestRecord().evolver()
    with pytest.raises(TypeError):
        evolver.set('age', 'not_an_int')

    # Test invariant failure
    evolver = TestRecord().evolver()
    evolver.set('age', 30)
    evolver.set('score', -5.0)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'score must be non-negative' in str(exc_info.value)

    # Test setting non-existent field
    evolver = TestRecord().evolver()
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('nonexistent', 'value')
    assert "'nonexistent' is not among the specified fields" in str(exc_info.value)

    # Test with factory fields
    class RecordWithFactory(PRecord):
        items = field(type=list, factory=lambda x: x if isinstance(x, list) else [x])

    evolver = RecordWithFactory().evolver()
    evolver.set('items', 42)
    result = evolver.persistent()
    assert result['items'] == [42]

    # Test multiple invariant failures accumulate
    class MultiInvariantRecord(PRecord):
        value1 = field(type=int, invariant=lambda x: (x > 0, 'value1 must be positive'))
        value2 = field(type=int, invariant=lambda x: (x < 10, 'value2 must be less than 10'))

    evolver = MultiInvariantRecord().evolver()
    evolver.set('value1', -5)
    evolver.set('value2', 15)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'value1 must be positive' in str(exc_info.value)
    assert 'value2 must be less than 10' in str(exc_info.value)

    # Test that missing mandatory fields are tracked
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    # age is mandatory but not set
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'missing_fields' in str(exc_info.value)
    assert 'TestRecord.age' in str(exc_info.value)

    # Test using __setitem__ syntax
    evolver = TestRecord().evolver()
    evolver['name'] = 'Charlie'
    evolver['age'] = 35
    result = evolver.persistent()
    assert result['name'] == 'Charlie'
    assert result['age'] == 35


# LLM-generated content at query #21
#--------------------------

```python
def test_PRecord___new__():
    # Test basic record creation with fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        active = field(type=bool, initial=lambda: True)
    
    # Test creation with initial values
    r1 = TestRecord(name="Alice", age=30)
    assert r1["name"] == "Alice"
    assert r1["age"] == 30
    assert r1["active"] == True
    assert isinstance(r1, TestRecord)
    
    # Test creation with callable initial
    r2 = TestRecord(name="Bob", age=25)
    assert r2["active"] == True
    
    # Test that mandatory fields are required
    try:
        TestRecord(name="Charlie")
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "TestRecord.age" in e.missing_fields
    
    # Test creation with factory fields
    r3 = TestRecord.create({"name": "David", "age": 35}, _factory_fields=None)
    assert r3["name"] == "David"
    assert r3["age"] == 35
    
    # Test creation with ignore_extra
    r4 = TestRecord.create({"name": "Eve", "age": 28, "extra": "ignored"}, ignore_extra=True)
    assert r4["name"] == "Eve"
    assert r4["age"] == 28
    assert "extra" not in r4
    
    # Test creation from existing record
    r5 = TestRecord.create(r4)
    assert r5 is r4
    
    # Test internal creation with _precord_size and _precord_buckets
    internal_record = TestRecord(_precord_size=2, _precord_buckets=[("name", "Frank"), ("age", 40)])
    assert internal_record["name"] == "Frank"
    assert internal_record["age"] == 40
    
    # Test type checking
    try:
        TestRecord(name="Grace", age="not_an_int")
        assert False, "Should have raised type error"
    except TypeError:
        pass
    
    # Test with no initial values for non-mandatory fields
    class SimpleRecord(PRecord):
        x = field(type=int, initial=10)
        y = field(type=str)
    
    r6 = SimpleRecord(y="test")
    assert r6["x"] == 10
    assert r6["y"] == "test"
    
    # Test that initial values can be overridden
    r7 = SimpleRecord(x=20, y="override")
    assert r7["x"] == 20
    assert r7["y"] == "override"


# LLM-generated content at query #22
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic class creation with fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        active = field(type=bool, initial=lambda: True)
    
    assert hasattr(TestRecord, '_precord_fields')
    assert isinstance(TestRecord._precord_fields, dict)
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'active' in TestRecord._precord_fields
    
    # Test mandatory fields extraction
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert isinstance(TestRecord._precord_mandatory_fields, set)
    assert 'age' in TestRecord._precord_mandatory_fields
    assert 'name' not in TestRecord._precord_mandatory_fields
    assert 'active' not in TestRecord._precord_mandatory_fields
    
    # Test initial values extraction
    assert hasattr(TestRecord, '_precord_initial_values')
    assert isinstance(TestRecord._precord_initial_values, dict)
    assert 'active' in TestRecord._precord_initial_values
    assert callable(TestRecord._precord_initial_values['active'])
    assert 'name' not in TestRecord._precord_initial_values
    assert 'age' not in TestRecord._precord_initial_values
    
    # Test invariants storage
    assert hasattr(TestRecord, '_precord_invariants')
    assert isinstance(TestRecord._precord_invariants, tuple)
    
    # Test slots
    assert hasattr(TestRecord, '__slots__')
    assert TestRecord.__slots__ == ()
    
    # Test inheritance with additional fields
    class ChildRecord(TestRecord):
        score = field(type=float)
    
    assert 'score' in ChildRecord._precord_fields
    assert 'name' in ChildRecord._precord_fields
    assert 'age' in ChildRecord._precord_fields
    assert 'active' in ChildRecord._precord_fields
    
    # Test that mandatory fields are inherited
    assert 'age' in ChildRecord._precord_mandatory_fields
    
    # Test class with invariant
    class InvariantRecord(PRecord):
        value = field(type=int)
        
        @invariant
        def value_must_be_positive(self):
            return self.value > 0, "Value must be positive"
    
    assert len(InvariantRecord._precord_invariants) == 1
    
    # Test class without fields
    class EmptyRecord(PRecord):
        pass
    
    assert EmptyRecord._precord_fields == {}
    assert EmptyRecord._precord_mandatory_fields == set()
    assert EmptyRecord._precord_initial_values == {}
    
    # Test that fields from parent are properly merged
    class ParentRecord(PRecord):
        parent_field = field(type=str)
    
    class MixedRecord(ParentRecord):
        child_field = field(type=int)
    
    assert 'parent_field' in MixedRecord._precord_fields
    assert 'child_field' in MixedRecord._precord_fields
    assert len(MixedRecord._precord_fields) == 2


# LLM-generated content at query #23
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        score = field(type=float, invariant=lambda x: (x >= 0, 'score_non_negative'))
    
    # Test basic field setting
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    evolver.set('score', 95.5)
    result = evolver.persistent()
    assert result['name'] == 'Alice'
    assert result['age'] == 25
    assert result['score'] == 95.5
    
    # Test type checking
    evolver = TestRecord().evolver()
    evolver.set('age', 30)  # Valid int
    result = evolver.persistent()
    assert result['age'] == 30
    
    # Test invariant violation
    evolver = TestRecord().evolver()
    evolver.set('score', -10.0)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'score_non_negative' in str(e)
    
    # Test setting non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('nonexistent', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'nonexistent' is not among the specified fields" in str(e)
    
    # Test with factory fields
    class TestRecordWithFactory(PRecord):
        items = field(type=list, factory=lambda x: list(x))
    
    evolver = TestRecordWithFactory().evolver()
    evolver.set('items', (1, 2, 3))
    result = evolver.persistent()
    assert result['items'] == [1, 2, 3]
    
    # Test missing mandatory field
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    # age is mandatory but not set
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'TestRecord.age' in str(e)
    
    # Test multiple invariant violations accumulate
    evolver = TestRecord().evolver()
    evolver.set('score', -5.0)
    evolver.set('score', -10.0)  # Second violation
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        # Should have both violations recorded
        assert e.invariant_errors.count('score_non_negative') == 2
    
    # Test using __setitem__ syntax
    evolver = TestRecord().evolver()
    evolver['name'] = 'Charlie'
    evolver['age'] = 35
    result = evolver.persistent()
    assert result['name'] == 'Charlie'
    assert result['age'] == 35


# LLM-generated content at query #24
#--------------------------

```python
def test_PRecord___repr__():
    # Test basic repr with simple fields
    class SimpleRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
    
    r1 = SimpleRecord(name="Alice", age=30)
    assert repr(r1) == "SimpleRecord(name='Alice', age=30)"
    
    # Test repr with empty record
    class EmptyRecord(PRecord):
        pass
    
    r2 = EmptyRecord()
    assert repr(r2) == "EmptyRecord()"
    
    # Test repr with multiple fields including special characters in values
    class ComplexRecord(PRecord):
        title = field(type=str)
        value = field(type=float)
        active = field(type=bool)
    
    r3 = ComplexRecord(title="Test\nLine", value=3.14159, active=True)
    assert repr(r3) == "ComplexRecord(title='Test\\nLine', value=3.14159, active=True)"
    
    # Test repr with None values
    class OptionalRecord(PRecord):
        data = field(type=str, optional=True)
        count = field(type=int)
    
    r4 = OptionalRecord(data=None, count=5)
    assert repr(r4) == "OptionalRecord(data=None, count=5)"
    
    # Test repr preserves field order as defined in class
    class OrderedRecord(PRecord):
        z = field(type=int)
        a = field(type=str)
        m = field(type=float)
    
    r5 = OrderedRecord(z=1, a="test", m=2.5)
    assert repr(r5) == "OrderedRecord(z=1, a='test', m=2.5)"
    
    # Test repr with nested structures
    class NestedRecord(PRecord):
        id = field(type=int)
        nested = field(type=dict)
    
    r6 = NestedRecord(id=1, nested={"key": "value"})
    assert repr(r6) == "NestedRecord(id=1, nested={'key': 'value'})"
    
    # Test repr after modification using set
    r7 = SimpleRecord(name="Bob", age=25)
    r7_updated = r7.set(name="Robert")
    assert repr(r7_updated) == "SimpleRecord(name='Robert', age=25)"
    
    # Test repr with boolean False value
    class BoolRecord(PRecord):
        flag = field(type=bool)
    
    r8 = BoolRecord(flag=False)
    assert repr(r8) == "BoolRecord(flag=False)"
    
    # Test repr with integer zero
    class NumberRecord(PRecord):
        count = field(type=int)
    
    r9 = NumberRecord(count=0)
    assert repr(r9) == "NumberRecord(count=0)"
    
    # Test repr with empty string
    class StringRecord(PRecord):
        text = field(type=str)
    
    r10 = StringRecord(text="")
    assert repr(r10) == "StringRecord(text='')"


# LLM-generated content at query #25
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: (len(self.name) > 0, "NAME_EMPTY")
        name = field(type=str, mandatory=True)
        age = field(type=int, initial=25)
        active = field(type=bool, initial=lambda: True)

    assert hasattr(TestRecord, '_precord_fields')
    assert set(TestRecord._precord_fields.keys()) == {'name', 'age', 'active'}
    
    assert TestRecord._precord_mandatory_fields == {'name'}
    
    assert TestRecord._precord_initial_values['age'] == 25
    assert TestRecord._precord_initial_values['active']() == True
    
    assert hasattr(TestRecord, '_precord_invariants')
    assert len(TestRecord._precord_invariants) == 1
    
    assert TestRecord.__slots__ == ()
    
    class ChildRecord(TestRecord):
        score = field(type=float, initial=0.0)
    
    assert set(ChildRecord._precord_fields.keys()) == {'name', 'age', 'active', 'score'}
    assert ChildRecord._precord_mandatory_fields == {'name'}
    assert ChildRecord._precord_initial_values['score'] == 0.0
    assert len(ChildRecord._precord_invariants) == 1


# LLM-generated content at query #26
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    import pytest

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        score = field(type=float, invariant=lambda x: (x >= 0, 'score must be non-negative'))

    # Test basic field setting
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    evolver.set('score', 95.5)
    record = evolver.persistent()
    assert record['name'] == 'Alice'
    assert record['age'] == 25
    assert record['score'] == 95.5

    # Test type checking
    evolver = TestRecord().evolver()
    with pytest.raises(TypeError):
        evolver.set('age', 'not_an_int')

    # Test invariant violation
    evolver = TestRecord().evolver()
    evolver.set('age', 30)
    evolver.set('score', -5.0)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'score must be non-negative' in str(exc_info.value)

    # Test missing mandatory field
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'age' in str(exc_info.value)

    # Test setting non-existent field
    evolver = TestRecord().evolver()
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('invalid_field', 'value')
    assert 'invalid_field' in str(exc_info.value)
    assert 'TestRecord' in str(exc_info.value)

    # Test factory field processing
    class NestedRecord(PRecord):
        data = field(type=dict, factory=lambda x: dict(x))

    class ContainerRecord(PRecord):
        nested = field(type=NestedRecord)

    evolver = ContainerRecord().evolver()
    evolver.set('nested', {'data': {'key': 'value'}})
    record = evolver.persistent()
    assert isinstance(record['nested'], NestedRecord)
    assert record['nested']['data'] == {'key': 'value'}

    # Test multiple invariant violations accumulate
    class MultiInvariantRecord(PRecord):
        value1 = field(type=int, invariant=lambda x: (x > 0, 'value1 must be positive'))
        value2 = field(type=int, invariant=lambda x: (x < 100, 'value2 must be less than 100'))

    evolver = MultiInvariantRecord().evolver()
    evolver.set('value1', -5)
    evolver.set('value2', 200)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'value1 must be positive' in str(exc_info.value)
    assert 'value2 must be less than 100' in str(exc_info.value)

    # Test that setting same field multiple times works
    evolver = TestRecord().evolver()
    evolver.set('age', 10)
    evolver.set('age', 20)
    evolver.set('age', 30)
    record = evolver.persistent()
    assert record['age'] == 30

    # Test with _factory_fields parameter
    evolver = TestRecord(_factory_fields={TestRecord._precord_fields['name']}).evolver()
    evolver.set('name', 'Charlie')
    evolver.set('age', 40)
    record = evolver.persistent()
    assert record['name'] == 'Charlie'
    assert record['age'] == 40


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PRecord___new__():
    # Test basic record creation with field values
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
    
    r = TestRecord(name="Alice", age=30)
    assert r["name"] == "Alice"
    assert r["age"] == 30
    assert isinstance(r, TestRecord)
    
    # Test creation with initial values
    class RecordWithInitial(PRecord):
        name = field(type=str, initial="Unknown")
        age = field(type=int, initial=0)
    
    r1 = RecordWithInitial()
    assert r1["name"] == "Unknown"
    assert r1["age"] == 0
    
    r2 = RecordWithInitial(name="Bob")
    assert r2["name"] == "Bob"
    assert r2["age"] == 0
    
    # Test creation with callable initial
    counter = 0
    def get_id():
        nonlocal counter
        counter += 1
        return counter
    
    class RecordWithCallableInitial(PRecord):
        id = field(type=int, initial=get_id)
        value = field(type=str)
    
    r3 = RecordWithCallableInitial(value="test1")
    r4 = RecordWithCallableInitial(value="test2")
    assert r3["id"] == 1
    assert r4["id"] == 2
    
    # Test creation with factory fields
    class RecordWithFactory(PRecord):
        items = field(type=PSet, factory=pset)
    
    r5 = RecordWithFactory(items=[1, 2, 3])
    assert isinstance(r5["items"], PSet)
    assert 1 in r5["items"]
    
    # Test creation with ignore_extra
    class SimpleRecord(PRecord):
        name = field(type=str)
    
    r6 = SimpleRecord.create({"name": "Alice", "extra": "value"}, ignore_extra=True)
    assert r6["name"] == "Alice"
    assert "extra" not in r6
    
    # Test creation from existing record
    r7 = TestRecord(name="Charlie", age=25)
    r8 = TestRecord.create(r7)
    assert r8 is r7
    
    # Test internal creation with _precord_size and _precord_buckets
    internal_data = {"_precord_size": 2, "_precord_buckets": pmap({"name": "Dave", "age": 40})._buckets}
    r9 = TestRecord(**internal_data)
    assert r9["name"] == "Dave"
    assert r9["age"] == 40
    
    # Test type checking
    class TypedRecord(PRecord):
        count = field(type=int)
    
    try:
        TypedRecord(count="not_an_int")
        assert False, "Should have raised type error"
    except TypeError:
        pass
    
    # Test mandatory fields
    class MandatoryRecord(PRecord):
        required = field(type=str, mandatory=True)
        optional = field(type=int)
    
    try:
        MandatoryRecord(optional=5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "required" in str(e)
    
    # Test field invariants
    class PositiveRecord(PRecord):
        value = field(type=int, invariant=lambda x: (x > 0, "value must be positive"))
    
    try:
        PositiveRecord(value=-5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "value must be positive" in str(e)
    
    r10 = PositiveRecord(value=10)
    assert r10["value"] == 10


# LLM-generated content at query #2
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, InvariantException
    from pyrsistent._precord import PRecord, _PRecordEvolver

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        score = field(type=int, invariant=lambda x: (x >= 0, 'score.negative'))

    # Test basic field setting
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    result = evolver.persistent()
    assert result['name'] == 'Alice'
    assert result['age'] == 25

    # Test type checking
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('age', 30)  # Valid int
    result = evolver.persistent()
    assert result['age'] == 30

    # Test invariant violation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('score', -5)  # Violates invariant
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'score.negative' in str(e)

    # Test setting non-existent field
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'invalid_field' is not among the specified fields" in str(e)

    # Test with factory fields
    class RecordWithFactory(PRecord):
        items = field(type=list, factory=lambda x: list(x))

    evolver = _PRecordEvolver(RecordWithFactory, pmap(), _factory_fields={RecordWithFactory._precord_fields['items']})
    evolver.set('items', (1, 2, 3))
    result = evolver.persistent()
    assert result['items'] == [1, 2, 3]

    # Test without factory fields (should use original value)
    evolver = _PRecordEvolver(RecordWithFactory, pmap(), _factory_fields=set())
    evolver.set('items', (1, 2, 3))
    result = evolver.persistent()
    assert result['items'] == (1, 2, 3)

    # Test multiple invariant violations accumulate
    class MultiInvariantRecord(PRecord):
        a = field(type=int, invariant=lambda x: (x > 0, 'a.positive'))
        b = field(type=int, invariant=lambda x: (x < 10, 'b.less_than_10'))

    evolver = _PRecordEvolver(MultiInvariantRecord, pmap())
    evolver.set('a', -1)  # First violation
    evolver.set('b', 20)  # Second violation
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'a.positive' in str(e)
        assert 'b.less_than_10' in str(e)

    # Test that valid values don't raise exceptions
    evolver = _PRecordEvolver(MultiInvariantRecord, pmap())
    evolver.set('a', 5)
    evolver.set('b', 8)
    result = evolver.persistent()
    assert result['a'] == 5
    assert result['b'] == 8

    # Test updating existing value
    record = TestRecord.create({'name': 'Bob', 'age': 30})
    evolver = record.evolver()
    evolver.set('name', 'Robert')
    result = evolver.persistent()
    assert result['name'] == 'Robert'
    assert result['age'] == 30

    # Test __setitem__ alias
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['name'] = 'Charlie'
    evolver['age'] = 35
    result = evolver.persistent()
    assert result['name'] == 'Charlie'
    assert result['age'] == 35


# LLM-generated content at query #3
#--------------------------

```python
def test__PRecordEvolver_persistent():
    from pyrsistent import field, InvariantException
    from pyrsistent._precord import PRecord, _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int, invariant=lambda x: (x >= 0, 'Age must be non-negative'))
        optional = field(type=str, initial='default')
    
    # Test 1: Basic persistence with valid data
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['name'] == 'Alice'
    assert result['age'] == 25
    assert result['optional'] == 'default'
    
    # Test 2: Persistence with existing valid PMap
    existing = TestRecord(name='Bob', age=30)
    evolver = _PRecordEvolver(TestRecord, existing)
    result = evolver.persistent()
    assert result is existing  # Should return same instance when not dirty
    
    # Test 3: Missing mandatory field should raise InvariantException
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('age', 25)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.name' in e.missing_fields
    
    # Test 4: Field invariant violation should raise InvariantException
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Charlie')
    evolver.set('age', -5)  # Violates invariant
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Age must be non-negative' in e.invariant_errors
    
    # Test 5: Multiple invariant violations accumulate
    class MultiInvariantRecord(PRecord):
        value = field(type=int, 
                     invariant=lambda x: (x > 0, 'Positive'),
                     mandatory=True)
        other = field(type=int,
                     invariant=lambda x: (x < 10, 'Less than 10'))
    
    evolver = _PRecordEvolver(MultiInvariantRecord, pmap())
    evolver.set('value', -1)  # Violates Positive
    evolver.set('other', 15)  # Violates Less than 10
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Positive' in e.invariant_errors
        assert 'Less than 10' in e.invariant_errors
    
    # Test 6: Check global invariants are called
    class GlobalInvariantRecord(PRecord):
        x = field(type=int)
        y = field(type=int)
        
        @staticmethod
        def __invariant__(obj):
            return obj['x'] <= obj['y'], 'x must be <= y'
    
    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 5)  # Violates global invariant
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException:
        # Global invariant check happens in persistent()
        pass
    
    # Test 7: Dirty evolver creates new instance
    existing = TestRecord(name='Dave', age=40)
    evolver = _PRecordEvolver(TestRecord, existing)
    evolver.set('age', 41)  # Make it dirty
    result = evolver.persistent()
    assert result is not existing
    assert result['age'] == 41
    assert result['name'] == 'Dave'
    
    # Test 8: Factory fields handling
    class FactoryRecord(PRecord):
        items = field(type=list, factory=lambda x: list(x))
    
    evolver = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields={FactoryRecord.items})
    evolver.set('items', [1, 2, 3])
    result = evolver.persistent()
    assert result['items'] == [1, 2, 3]
    
    # Test 9: No fields set on empty record
    class EmptyRecord(PRecord):
        pass
    
    evolver = _PRecordEvolver(EmptyRecord, pmap())
    result = evolver.persistent()
    assert isinstance(result, EmptyRecord)
    assert len(result) == 0


# LLM-generated content at query #4
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        score = field(type=float, invariant=lambda x: (x >= 0, 'score_nonnegative'))

    # Test basic field setting
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 30)
    evolver.set('score', 95.5)
    record = evolver.persistent()
    assert record['name'] == 'Alice'
    assert record['age'] == 30
    assert record['score'] == 95.5

    # Test type checking
    evolver = TestRecord().evolver()
    evolver.set('age', 25)  # Valid int
    record = evolver.persistent()
    assert record['age'] == 25

    # Test invariant violation
    evolver = TestRecord().evolver()
    evolver.set('score', -10.0)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'score_nonnegative' in str(e)

    # Test setting non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'invalid_field' is not among the specified fields" in str(e)

    # Test factory transformation
    class NestedRecord(PRecord):
        data = field(type=dict, factory=lambda x: dict(x))

    evolver = NestedRecord().evolver()
    evolver.set('data', {'key': 'value'})
    record = evolver.persistent()
    assert record['data'] == {'key': 'value'}
    assert isinstance(record['data'], dict)

    # Test with _factory_fields filtering
    class ComplexRecord(PRecord):
        a = field(type=int, factory=lambda x: x * 2)
        b = field(type=int)

    # When field not in _factory_fields, value should not be transformed
    evolver = ComplexRecord().evolver(_factory_fields=[ComplexRecord._precord_fields['a']])
    evolver.set('a', 5)  # Should apply factory: 5 * 2 = 10
    evolver.set('b', 7)  # Should not apply factory (no factory anyway)
    record = evolver.persistent()
    assert record['a'] == 10
    assert record['b'] == 7

    # Test multiple invariant errors accumulate
    class MultiInvariantRecord(PRecord):
        x = field(type=int, invariant=lambda v: (v > 0, 'positive'))
        y = field(type=int, invariant=lambda v: (v < 10, 'less_than_10'))

    evolver = MultiInvariantRecord().evolver()
    evolver.set('x', -5)  # Violates positive
    evolver.set('y', 15)  # Violates less_than_10
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'positive' in str(e)
        assert 'less_than_10' in str(e)

    # Test missing mandatory field detection
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    # Don't set mandatory 'age' field
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert 'age' in str(e)

    # Test that __setitem__ alias works
    evolver = TestRecord().evolver()
    evolver['name'] = 'Charlie'
    evolver['age'] = 40
    record = evolver.persistent()
    assert record['name'] == 'Charlie'
    assert record['age'] == 40


# LLM-generated content at query #5
#--------------------------

```python
def test__PRecordEvolver_persistent():
    from pyrsistent import field, PRecord, InvariantException
    
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int, invariant=lambda x: (x >= 0, 'Age must be non-negative'))
        optional = field(type=str, initial='default')
    
    # Test basic persistence with valid data
    evolver = TestRecord.evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 30)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['name'] == 'Alice'
    assert result['age'] == 30
    assert result['optional'] == 'default'
    
    # Test persistence without changes
    evolver2 = result.evolver()
    result2 = evolver2.persistent()
    assert result2 is result
    
    # Test persistence with changes
    evolver3 = result.evolver()
    evolver3.set('age', 25)
    result3 = evolver3.persistent()
    assert result3 is not result
    assert result3['age'] == 25
    assert result3['name'] == 'Alice'
    
    # Test missing mandatory field
    evolver4 = TestRecord.evolver()
    evolver4.set('age', 30)
    try:
        evolver4.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.name' in e.missing_fields
    
    # Test field invariant violation
    evolver5 = TestRecord.evolver()
    evolver5.set('name', 'Bob')
    evolver5.set('age', -5)
    try:
        evolver5.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Age must be non-negative' in e.invariant_errors
    
    # Test multiple errors
    evolver6 = TestRecord.evolver()
    evolver6.set('age', -10)
    try:
        evolver6.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.name' in e.missing_fields
        assert 'Age must be non-negative' in e.invariant_errors
    
    # Test with factory fields
    class ComplexRecord(PRecord):
        items = field(type=list, factory=lambda x: list(x) if x else [])
    
    evolver7 = ComplexRecord.evolver()
    evolver7.set('items', [1, 2, 3])
    result7 = evolver7.persistent()
    assert result7['items'] == [1, 2, 3]
    
    # Test that original pmap is returned when no changes
    evolver8 = result.evolver()
    assert not evolver8.is_dirty()
    result8 = evolver8.persistent()
    assert result8 is result


# LLM-generated content at query #6
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda x: (len(x.get('name', '')) > 0, 'ERR_NAME_EMPTY')
        name = field(type=str, mandatory=True)
        age = field(type=int, initial=25)
        active = field(type=bool, initial=lambda: True)

    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'active' in TestRecord._precord_fields
    
    assert TestRecord._precord_fields['name'].mandatory is True
    assert TestRecord._precord_fields['name'].type is str
    assert TestRecord._precord_fields['age'].type is int
    assert TestRecord._precord_fields['active'].type is bool
    
    assert TestRecord._precord_mandatory_fields == {'name'}
    
    assert TestRecord._precord_initial_values['age'] == 25
    assert TestRecord._precord_initial_values['active']() is True
    
    assert hasattr(TestRecord, '_precord_invariants')
    assert len(TestRecord._precord_invariants) == 1
    
    assert TestRecord.__slots__ == ()
    
    record = TestRecord(name='Alice')
    assert record['name'] == 'Alice'
    assert record['age'] == 25
    assert record['active'] is True
    
    class ChildRecord(TestRecord):
        score = field(type=float, initial=0.0)
    
    assert 'score' in ChildRecord._precord_fields
    assert ChildRecord._precord_fields['score'].type is float
    assert ChildRecord._precord_initial_values['score'] == 0.0
    assert ChildRecord._precord_mandatory_fields == {'name'}
    assert len(ChildRecord._precord_invariants) == 1


# LLM-generated content at query #7
#--------------------------

```python
def test__PRecordEvolver_persistent():
    from pyrsistent import field, InvariantException
    
    # Define a simple PRecord class for testing
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int, mandatory=False)
        score = field(type=int, initial=100)
        
        @staticmethod
        def __invariant__(record):
            if 'age' in record and record['age'] < 0:
                return (False, "age_negative")
            return (True, "")
    
    # Test 1: Basic persistent creation with valid data
    evolver1 = _PRecordEvolver(TestRecord, pmap())
    evolver1.set('name', 'Alice')
    evolver1.set('age', 25)
    evolver1.set('score', 95)
    result1 = evolver1.persistent()
    assert isinstance(result1, TestRecord)
    assert result1['name'] == 'Alice'
    assert result1['age'] == 25
    assert result1['score'] == 95
    
    # Test 2: Persistent with initial values
    evolver2 = _PRecordEvolver(TestRecord, pmap())
    evolver2.set('name', 'Bob')
    result2 = evolver2.persistent()
    assert result2['name'] == 'Bob'
    assert result2['score'] == 100  # Default initial value
    
    # Test 3: Missing mandatory field should raise InvariantException
    evolver3 = _PRecordEvolver(TestRecord, pmap())
    evolver3.set('age', 30)
    try:
        evolver3.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.name' in e.missing_fields
    
    # Test 4: Field invariant violation should raise InvariantException
    evolver4 = _PRecordEvolver(TestRecord, pmap())
    evolver4.set('name', 'Charlie')
    evolver4.set('age', -5)  # Negative age violates invariant
    try:
        evolver4.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'age_negative' in e.invariant_errors
    
    # Test 5: Global invariant violation should raise InvariantException
    class NegativeAgeRecord(PRecord):
        age = field(type=int)
        
        @staticmethod
        def __invariant__(record):
            if record['age'] < 0:
                return (False, "global_age_negative")
            return (True, "")
    
    evolver5 = _PRecordEvolver(NegativeAgeRecord, pmap())
    evolver5.set('age', -10)
    try:
        evolver5.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_age_negative' in e.invariant_errors
    
    # Test 6: Already persistent PMap that is instance of destination class
    existing_record = TestRecord(name='David', age=40)
    evolver6 = _PRecordEvolver(TestRecord, existing_record)
    result6 = evolver6.persistent()
    assert result6 is existing_record  # Should return same instance when not dirty
    
    # Test 7: Dirty evolver should create new instance
    existing_record = TestRecord(name='Eve', age=35)
    evolver7 = _PRecordEvolver(TestRecord, existing_record)
    evolver7.set('age', 36)  # Make it dirty
    result7 = evolver7.persistent()
    assert result7 is not existing_record
    assert result7['name'] == 'Eve'
    assert result7['age'] == 36
    
    # Test 8: Factory fields handling
    class FactoryRecord(PRecord):
        items = field(type=list, factory=lambda x: list(x))
    
    evolver8 = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields={FactoryRecord._precord_fields['items']})
    evolver8.set('items', [1, 2, 3])
    result8 = evolver8.persistent()
    assert result8['items'] == [1, 2, 3]
    
    # Test 9: Multiple missing fields
    class MultiMandatoryRecord(PRecord):
        field1 = field(mandatory=True)
        field2 = field(mandatory=True)
        field3 = field(mandatory=True)
    
    evolver9 = _PRecordEvolver(MultiMandatoryRecord, pmap())
    evolver9.set('field1', 'value1')
    try:
        evolver9.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 2
        assert 'MultiMandatoryRecord.field2' in e.missing_fields
        assert 'MultiMandatoryRecord.field3' in e.missing_fields
    
    # Test 10: Multiple invariant errors
    class MultiInvariantRecord(PRecord):
        value1 = field(type=int)
        value2 = field(type=int)
        
        @staticmethod
        def __invariant__(record):
            errors = []
            if 'value1' in record and record['value1'] < 0:
                errors.append(("value1_negative",))
            if 'value2' in record and record['value2'] < 0:
                errors.append(("value2_negative",))
            return (len(errors) == 0, errors[0][0] if errors else "")
    
    evolver10 = _PRecordEvolver(MultiInvariantRecord, pmap())
    evolver10.set('value1', -1)
    evolver10.set('value2', -2)
    try:
        evolver10.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        # Note: The current implementation only catches the first invariant error per field
        # but multiple fields can have errors
        assert len(e.invariant_errors) >= 1


# LLM-generated content at query #8
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: (len(self.name) > 0, "NAME_EMPTY")
        name = field(type=str, mandatory=True)
        age = field(type=int, initial=25)
        active = field(type=bool, initial=lambda: True)

    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'active' in TestRecord._precord_fields
    
    assert TestRecord._precord_fields['name'].mandatory is True
    assert TestRecord._precord_fields['name'].type is str
    assert TestRecord._precord_fields['age'].type is int
    assert TestRecord._precord_fields['age'].initial == 25
    assert TestRecord._precord_fields['active'].type is bool
    assert TestRecord._precord_fields['active'].initial() is True
    
    assert TestRecord._precord_mandatory_fields == {'name'}
    
    assert TestRecord._precord_initial_values['age'] == 25
    assert TestRecord._precord_initial_values['active']() is True
    
    assert hasattr(TestRecord, '_precord_invariants')
    assert len(TestRecord._precord_invariants) == 1
    
    assert TestRecord.__slots__ == ()
    
    record = TestRecord(name="John")
    assert record.name == "John"
    assert record.age == 25
    assert record.active is True
    
    with pytest.raises(AttributeError):
        record.nonexistent = "value"


# LLM-generated content at query #9
#--------------------------

```python
def test_PRecord___new__():
    # Test 1: Normal creation with keyword arguments
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        active = field(type=bool, initial=lambda: True)
    
    # Test basic creation with all required fields
    record = TestRecord(name="Alice", age=30)
    assert record["name"] == "Alice"
    assert record["age"] == 30
    assert record["active"] == True
    
    # Test 2: Creation with initial values
    record2 = TestRecord(name="Bob", age=25, active=False)
    assert record2["name"] == "Bob"
    assert record2["age"] == 25
    assert record2["active"] == False
    
    # Test 3: Missing mandatory field should raise InvariantException
    try:
        TestRecord(name="Charlie")
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "TestRecord.age" in e.missing_fields
    
    # Test 4: Creation with _factory_fields parameter
    record3 = TestRecord(name="David", age=35, _factory_fields=None)
    assert record3["name"] == "David"
    assert record3["age"] == 35
    
    # Test 5: Creation with _ignore_extra parameter
    record4 = TestRecord(name="Eve", age=40, _ignore_extra=True)
    assert record4["name"] == "Eve"
    assert record4["age"] == 40
    
    # Test 6: Direct creation with internal parameters
    record5 = TestRecord(_precord_size=2, _precord_buckets=pmap({"name": "Frank", "age": 45})._buckets)
    assert record5["name"] == "Frank"
    assert record5["age"] == 45
    
    # Test 7: Type checking should work
    try:
        TestRecord(name="Grace", age="not_an_int")
        assert False, "Should have raised type check error"
    except TypeError:
        pass
    
    # Test 8: Unknown field should raise AttributeError
    try:
        TestRecord(name="Henry", age=50, unknown_field="value")
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "is not among the specified fields" in str(e)
    
    # Test 9: Callable initial value
    class TestRecord2(PRecord):
        timestamp = field(type=float, initial=time.time)
        value = field(type=str, initial=lambda: "default")
    
    record6 = TestRecord2()
    assert isinstance(record6["timestamp"], float)
    assert record6["value"] == "default"
    
    # Test 10: Override initial value
    record7 = TestRecord2(value="custom")
    assert record7["value"] == "custom"
    
    # Test 11: Empty record creation
    class EmptyRecord(PRecord):
        pass
    
    record8 = EmptyRecord()
    assert len(record8) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    from pyrsistent._checked_types import CheckedType

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, invariant=lambda x: (x >= 0, 'age_negative'))
        optional = field(type=str, mandatory=False)

    # Test basic field setting with type checking
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    record = evolver.persistent()
    assert record['name'] == 'Alice'
    assert record['age'] == 25

    # Test type checking failure
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    evolver.set('age', 'not_an_int')
    record = evolver.persistent()
    assert record['age'] == 'not_an_int'

    # Test invariant failure
    evolver = TestRecord().evolver()
    evolver.set('name', 'Charlie')
    evolver.set('age', -5)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'age_negative' in e.invariant_errors

    # Test setting non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('nonexistent', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'nonexistent' is not among the specified fields" in str(e)

    # Test with factory fields
    class TestRecordWithFactory(PRecord):
        items = field(type=list, factory=lambda x: x if isinstance(x, list) else [x])

    evolver = TestRecordWithFactory().evolver()
    evolver.set('items', 'single')
    record = evolver.persistent()
    assert record['items'] == ['single']

    # Test with ignore_extra parameter
    class NestedRecord(PRecord):
        value = field(type=int)

    class TestRecordNested(PRecord):
        nested = field(type=NestedRecord)

    evolver = TestRecordNested().evolver()
    evolver.set('nested', {'value': 1, 'extra': 'should_be_ignored'})
    record = evolver.persistent()
    assert record['nested']['value'] == 1

    # Test multiple invariant failures accumulate
    evolver = TestRecord().evolver()
    evolver.set('age', -1)
    evolver.set('age', -2)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 2
        assert all(err == 'age_negative' for err in e.invariant_errors)

    # Test that missing fields are tracked
    evolver = TestRecord().evolver()
    evolver.set('optional', 'present')
    record = evolver.persistent()
    assert 'optional' in record

    # Test __setitem__ also works
    evolver = TestRecord().evolver()
    evolver['name'] = 'David'
    evolver['age'] = 30
    record = evolver.persistent()
    assert record['name'] == 'David'
    assert record['age'] == 30


# LLM-generated content at query #11
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException

    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int)
        score = field(type=float, invariant=lambda x: (x >= 0, 'score_non_negative'))

    # Test basic field setting
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 30)
    evolver.set('score', 95.5)
    record = evolver.persistent()
    assert record['name'] == 'Alice'
    assert record['age'] == 30
    assert record['score'] == 95.5

    # Test type checking
    evolver = TestRecord().evolver()
    evolver.set('age', 'not_an_int')
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Wrong type' in str(e)

    # Test field invariant
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    evolver.set('score', -10.0)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'score_non_negative' in str(e)

    # Test non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('nonexistent', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'nonexistent' is not among the specified fields" in str(e)

    # Test mandatory field missing
    evolver = TestRecord().evolver()
    evolver.set('age', 25)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.name' in str(e)

    # Test multiple invariants failing
    evolver = TestRecord().evolver()
    evolver.set('age', 'invalid')
    evolver.set('score', -5.0)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'Wrong type' in str(e)
        assert 'score_non_negative' in str(e)

    # Test using __setitem__ syntax
    evolver = TestRecord().evolver()
    evolver['name'] = 'Charlie'
    evolver['age'] = 40
    record = evolver.persistent()
    assert record['name'] == 'Charlie'
    assert record['age'] == 40

    # Test that setting same value doesn't break
    evolver = TestRecord(name='Dave', age=50).evolver()
    evolver.set('name', 'Dave')  # Same value
    record = evolver.persistent()
    assert record['name'] == 'Dave'
    assert record['age'] == 50


# LLM-generated content at query #12
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        pass

    # Test empty record
    record = TestRecord()
    assert repr(record) == "TestRecord()"

    # Test record with single field
    class TestRecord(PRecord):
        name = field(type=str)

    record = TestRecord(name="Alice")
    assert repr(record) == "TestRecord(name='Alice')"

    # Test record with multiple fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        active = field(type=bool)

    record = TestRecord(name="Bob", age=30, active=True)
    expected_repr = "TestRecord(name='Bob', age=30, active=True)"
    assert repr(record) == expected_repr

    # Test record with nested values
    class TestRecord(PRecord):
        data = field(type=dict)
        scores = field(type=list)

    record = TestRecord(data={"key": "value"}, scores=[1, 2, 3])
    repr_str = repr(record)
    assert repr_str.startswith("TestRecord(")
    assert "data={'key': 'value'}" in repr_str
    assert "scores=[1, 2, 3]" in repr_str

    # Test record with special characters in string values
    class TestRecord(PRecord):
        text = field(type=str)

    record = TestRecord(text="Line1\nLine2\tTab")
    assert repr(record) == "TestRecord(text='Line1\\nLine2\\tTab')"

    # Test record with None values
    class TestRecord(PRecord):
        name = field(type=str)
        value = field(type=type(None), optional=True)

    record = TestRecord(name="Charlie", value=None)
    assert repr(record) == "TestRecord(name='Charlie', value=None)"


# LLM-generated content at query #13
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic class creation with fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        active = field(type=bool, initial=lambda: True)
    
    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert 'active' in TestRecord._precord_fields
    
    # Test mandatory fields detection
    assert 'age' in TestRecord._precord_mandatory_fields
    assert 'name' not in TestRecord._precord_mandatory_fields
    assert 'active' not in TestRecord._precord_mandatory_fields
    
    # Test initial values collection
    assert 'active' in TestRecord._precord_initial_values
    assert callable(TestRecord._precord_initial_values['active'])
    assert 'name' not in TestRecord._precord_initial_values
    assert 'age' not in TestRecord._precord_initial_values
    
    # Test invariants storage
    assert hasattr(TestRecord, '_precord_invariants')
    
    # Test slots
    assert TestRecord.__slots__ == ()
    
    # Test inheritance with additional fields
    class ChildRecord(TestRecord):
        email = field(type=str)
    
    assert 'email' in ChildRecord._precord_fields
    assert 'name' in ChildRecord._precord_fields
    assert 'age' in ChildRecord._precord_fields
    assert 'active' in ChildRecord._precord_fields
    
    # Test that mandatory fields are inherited
    assert 'age' in ChildRecord._precord_mandatory_fields
    
    # Test that initial values are inherited
    assert 'active' in ChildRecord._precord_initial_values
    
    # Test class with invariant
    class InvariantRecord(PRecord):
        value = field(type=int)
        
        @invariant(lambda rec: rec.value > 0)
        def value_positive(self):
            return self.value > 0
    
    assert len(InvariantRecord._precord_invariants) == 1
    
    # Test class without fields
    class EmptyRecord(PRecord):
        pass
    
    assert EmptyRecord._precord_fields == {}
    assert EmptyRecord._precord_mandatory_fields == set()
    assert EmptyRecord._precord_initial_values == {}
    
    # Test that metaclass properly chains to parent
    assert isinstance(TestRecord, _PRecordMeta)
    assert isinstance(ChildRecord, _PRecordMeta)


# LLM-generated content at query #14
#--------------------------

```python
def test_PRecord_serialize():
    from pyrsistent import field
    
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        data = field(type=dict, serializer=lambda v, f: {"custom": v})
    
    # Test basic serialization without format
    record = TestRecord(name="Alice", age=30, data={"key": "value"})
    result = record.serialize()
    assert result == {
        "name": "Alice",
        "age": 30,
        "data": {"custom": {"key": "value"}}
    }
    
    # Test serialization with format parameter
    class RecordWithFormat(PRecord):
        value = field(type=str, serializer=lambda v, f: f"{f}:{v}")
    
    record2 = RecordWithFormat(value="test")
    result2 = record2.serialize("prefix")
    assert result2 == {"value": "prefix:test"}
    
    # Test serialization with None format
    result3 = record2.serialize(None)
    assert result3 == {"value": "None:test"}
    
    # Test serialization without serializer
    class RecordNoSerializer(PRecord):
        field1 = field(type=str)
        field2 = field(type=int)
    
    record3 = RecordNoSerializer(field1="hello", field2=42)
    result4 = record3.serialize()
    assert result4 == {"field1": "hello", "field2": 42}
    
    # Test serialization with mixed fields
    class MixedRecord(PRecord):
        plain = field(type=str)
        serialized = field(type=list, serializer=lambda v, f: len(v))
    
    record4 = MixedRecord(plain="text", serialized=[1, 2, 3, 4])
    result5 = record4.serialize()
    assert result5 == {"plain": "text", "serialized": 4}
    
    # Test empty record serialization
    class EmptyRecord(PRecord):
        pass
    
    record5 = EmptyRecord()
    result6 = record5.serialize()
    assert result6 == {}


# LLM-generated content at query #15
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    import pytest

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        score = field(type=float, invariant=lambda x: (x >= 0, 'score_negative'))

    # Test basic field setting with type checking
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    evolver.set('score', 95.5)
    record = evolver.persistent()
    assert record['name'] == 'Alice'
    assert record['age'] == 25
    assert record['score'] == 95.5

    # Test type checking failure
    evolver = TestRecord().evolver()
    with pytest.raises(TypeError):
        evolver.set('age', 'not_an_int')

    # Test invariant violation
    evolver = TestRecord().evolver()
    evolver.set('age', 30)
    evolver.set('score', -5.0)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'score_negative' in str(exc_info.value)

    # Test setting non-existent field
    evolver = TestRecord().evolver()
    with pytest.raises(AttributeError) as exc_info:
        evolver.set('nonexistent', 'value')
    assert "'nonexistent' is not among the specified fields" in str(exc_info.value)

    # Test factory field processing
    class NestedRecord(PRecord):
        data = field(type=dict, factory=lambda x: dict(x))

    class ContainerRecord(PRecord):
        nested = field(type=NestedRecord, factory=NestedRecord.create)

    evolver = ContainerRecord().evolver()
    evolver.set('nested', {'data': {'key': 'value'}})
    record = evolver.persistent()
    assert isinstance(record['nested'], NestedRecord)
    assert record['nested']['data'] == {'key': 'value'}

    # Test with _factory_fields parameter
    evolver = ContainerRecord().evolver(_factory_fields={ContainerRecord._precord_fields['nested']})
    evolver.set('nested', {'data': {'key': 'value'}})
    record = evolver.persistent()
    assert isinstance(record['nested'], NestedRecord)

    # Test without _factory_fields (should use raw value)
    evolver = ContainerRecord().evolver(_factory_fields=set())
    evolver.set('nested', {'data': {'key': 'value'}})
    record = evolver.persistent()
    assert record['nested'] == {'data': {'key': 'value'}}
    assert not isinstance(record['nested'], NestedRecord)

    # Test multiple invariant violations accumulate
    class MultiInvariantRecord(PRecord):
        a = field(type=int, invariant=lambda x: (x > 0, 'a_positive'))
        b = field(type=int, invariant=lambda x: (x < 10, 'b_small'))

    evolver = MultiInvariantRecord().evolver()
    evolver.set('a', -1)
    evolver.set('b', 20)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'a_positive' in str(exc_info.value)
    assert 'b_small' in str(exc_info.value)

    # Test that missing mandatory fields are tracked
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    # age is mandatory but not set
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'TestRecord.age' in str(exc_info.value)


# LLM-generated content at query #16
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    from pyrsistent._checked_types import CheckedType

    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int)
        score = field(type=float, invariant=lambda x: (x >= 0, 'score_non_negative'))

    # Test basic field setting
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 30)
    evolver.set('score', 95.5)
    record = evolver.persistent()
    assert record['name'] == 'Alice'
    assert record['age'] == 30
    assert record['score'] == 95.5

    # Test field type checking
    evolver = TestRecord().evolver()
    evolver.set('age', 'not_an_int')
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'type' in str(e).lower()

    # Test field invariant
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    evolver.set('score', -10.0)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'score_non_negative' in str(e)

    # Test non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('nonexistent', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'nonexistent' is not among the specified fields" in str(e)

    # Test mandatory field missing
    evolver = TestRecord().evolver()
    evolver.set('age', 25)
    evolver.set('score', 80.0)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'mandatory' in str(e).lower() or 'missing' in str(e).lower()

    # Test with factory fields
    class RecordWithFactory(PRecord):
        items = field(type=list, factory=lambda x: list(x) if x else [])

    evolver = RecordWithFactory().evolver()
    evolver.set('items', (1, 2, 3))
    record = evolver.persistent()
    assert record['items'] == [1, 2, 3]

    # Test multiple invariant failures accumulate
    class MultiInvariant(PRecord):
        value = field(type=int, invariant=lambda x: (x > 0, 'positive'), mandatory=True)
        other = field(type=int, invariant=lambda x: (x < 10, 'less_than_10'))

    evolver = MultiInvariant().evolver()
    evolver.set('value', -5)
    evolver.set('other', 15)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'positive' in str(e)
        assert 'less_than_10' in str(e)

    # Test that __setitem__ also works
    evolver = TestRecord().evolver()
    evolver['name'] = 'Charlie'
    evolver['age'] = 40
    record = evolver.persistent()
    assert record['name'] == 'Charlie'
    assert record['age'] == 40


# LLM-generated content at query #17
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda x: (len(x.get('name', '')) > 0, 'name_non_empty')
        field1 = field(type=str, mandatory=True)
        field2 = field(type=int, initial=42)
        field3 = field(type=list, initial_factory=list)

    assert hasattr(TestRecord, '_precord_fields')
    assert 'field1' in TestRecord._precord_fields
    assert 'field2' in TestRecord._precord_fields
    assert 'field3' in TestRecord._precord_fields
    
    assert TestRecord._precord_fields['field1'].mandatory is True
    assert TestRecord._precord_fields['field2'].mandatory is False
    assert TestRecord._precord_fields['field2'].initial == 42
    assert TestRecord._precord_fields['field3'].initial is PFIELD_NO_INITIAL
    
    assert hasattr(TestRecord, '_precord_invariants')
    assert len(TestRecord._precord_invariants) == 1
    
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert 'field1' in TestRecord._precord_mandatory_fields
    assert 'field2' not in TestRecord._precord_mandatory_fields
    
    assert hasattr(TestRecord, '_precord_initial_values')
    assert 'field2' in TestRecord._precord_initial_values
    assert TestRecord._precord_initial_values['field2'] == 42
    assert 'field3' not in TestRecord._precord_initial_values
    
    assert hasattr(TestRecord, '__slots__')
    assert TestRecord.__slots__ == ()


# LLM-generated content at query #18
#--------------------------

```python
def test_PRecord___new__():
    # Test basic record creation with fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        active = field(type=bool, initial=lambda: True)
    
    # Test creation with initial values
    r1 = TestRecord(name="Alice", age=30)
    assert r1["name"] == "Alice"
    assert r1["age"] == 30
    assert r1["active"] == True
    
    # Test that mandatory fields are required
    try:
        TestRecord(name="Bob")
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "TestRecord.age" in e.missing_fields
    
    # Test creation with factory fields
    r2 = TestRecord.create({"name": "Charlie", "age": 25}, _factory_fields=None)
    assert r2["name"] == "Charlie"
    assert r2["age"] == 25
    
    # Test creation with ignore_extra
    r3 = TestRecord.create({"name": "David", "age": 35, "extra": "ignored"}, ignore_extra=True)
    assert r3["name"] == "David"
    assert r3["age"] == 35
    assert "extra" not in r3
    
    # Test internal creation with _precord_size and _precord_buckets
    internal_data = {"_precord_size": 2, "_precord_buckets": pmap({"name": "Eve", "age": 40})._buckets}
    r4 = TestRecord(**internal_data)
    assert r4["name"] == "Eve"
    assert r4["age"] == 40
    
    # Test that initial callable works correctly
    r5 = TestRecord(name="Frank", age=45)
    assert r5["active"] == True
    
    # Test update through evolver
    r6 = r5.set(name="Franklin")
    assert r6["name"] == "Franklin"
    assert r6["age"] == 45
    assert r6["active"] == True
    
    # Test multiple field update
    r7 = r5.set(name="Frankie", age=46)
    assert r7["name"] == "Frankie"
    assert r7["age"] == 46
    
    # Test type checking
    try:
        TestRecord(name="Grace", age="not_an_int")
        assert False, "Should have raised type error"
    except TypeError:
        pass
    
    # Test field invariant
    class PositiveRecord(PRecord):
        value = field(type=int, invariant=lambda x: (x > 0, "value must be positive"))
    
    try:
        PositiveRecord(value=-5)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "value must be positive" in e.invariant_errors
    
    # Test successful creation with invariant
    r8 = PositiveRecord(value=10)
    assert r8["value"] == 10


# LLM-generated content at query #19
#--------------------------

```python
def test_PRecord_serialize():
    from pyrsistent import field
    
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, serializer=lambda age, _: f"{age} years")
        tags = field(type=list, serializer=lambda tags, format: tags if format == "raw" else ", ".join(tags))
    
    record = TestRecord(name="Alice", age=30, tags=["dev", "python"])
    
    # Test default serialization (format=None)
    result = record.serialize()
    assert result == {
        "name": "Alice",
        "age": "30 years",
        "tags": ["dev", "python"]
    }
    
    # Test serialization with custom format
    result = record.serialize(format="raw")
    assert result == {
        "name": "Alice",
        "age": "30 years",
        "tags": ["dev", "python"]
    }
    
    # Test serialization with different format that triggers different serializer behavior
    class FormatRecord(PRecord):
        data = field(type=list, serializer=lambda data, format: data if format == "raw" else f"count: {len(data)}")
    
    format_record = FormatRecord(data=[1, 2, 3])
    result = format_record.serialize(format="raw")
    assert result == {"data": [1, 2, 3]}
    
    result = format_record.serialize(format="summary")
    assert result == {"data": "count: 3"}
    
    # Test serialization with no serializer specified
    class NoSerializerRecord(PRecord):
        value = field(type=int)
    
    no_serializer_record = NoSerializerRecord(value=42)
    result = no_serializer_record.serialize()
    assert result == {"value": 42}
    
    # Test serialization with multiple fields having different serializers
    class MixedRecord(PRecord):
        id = field(type=int, serializer=lambda x, _: f"ID-{x}")
        active = field(type=bool, serializer=lambda x, _: "yes" if x else "no")
        score = field(type=float)
    
    mixed_record = MixedRecord(id=123, active=True, score=98.5)
    result = mixed_record.serialize()
    assert result == {
        "id": "ID-123",
        "active": "yes",
        "score": 98.5
    }
    
    # Test serialization on empty record
    class EmptyRecord(PRecord):
        pass
    
    empty_record = EmptyRecord()
    result = empty_record.serialize()
    assert result == {}
    
    # Test that serialize returns a new dict and doesn't modify the record
    original_record = TestRecord(name="Bob", age=25, tags=["test"])
    serialized = original_record.serialize()
    serialized["name"] = "Modified"
    assert original_record["name"] == "Bob"
    assert serialized["name"] == "Modified"


# LLM-generated content at query #20
#--------------------------

```python
def test_PRecord___new__():
    # Test basic record creation with fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        active = field(type=bool, initial=lambda: True)
    
    # Test creation with initial values
    r1 = TestRecord(name="Alice", age=30)
    assert r1["name"] == "Alice"
    assert r1["age"] == 30
    assert r1["active"] is True
    
    # Test that initial values can be overridden
    r2 = TestRecord(name="Bob", age=25, active=False)
    assert r2["name"] == "Bob"
    assert r2["age"] == 25
    assert r2["active"] is False
    
    # Test missing mandatory field raises error
    try:
        TestRecord(name="Charlie")
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "TestRecord.age" in e.missing_fields
    
    # Test creation with internal fields for pickling
    r3 = TestRecord(_precord_size=2, _precord_buckets=pmap({"name": "Dave", "age": 40})._buckets)
    assert r3["name"] == "Dave"
    assert r3["age"] == 40
    assert r3["active"] is True
    
    # Test creation with factory fields
    r4 = TestRecord.create({"name": "Eve", "age": 35}, _factory_fields=set())
    assert r4["name"] == "Eve"
    assert r4["age"] == 35
    
    # Test ignore_extra parameter
    r5 = TestRecord.create({"name": "Frank", "age": 45, "extra": "ignored"}, ignore_extra=True)
    assert r5["name"] == "Frank"
    assert r5["age"] == 45
    assert "extra" not in r5
    
    # Test that extra fields without ignore_extra raises error
    try:
        TestRecord(name="Grace", age=50, extra="not_allowed")
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'extra' is not among the specified fields" in str(e)
    
    # Test creation from existing record
    r6 = TestRecord.create(r1)
    assert r6 is r1
    
    # Test with callable initial value
    counter = 0
    def get_counter():
        nonlocal counter
        counter += 1
        return counter
    
    class CounterRecord(PRecord):
        count = field(type=int, initial=get_counter)
    
    r7 = CounterRecord()
    assert r7["count"] == 1
    r8 = CounterRecord()
    assert r8["count"] == 2


# LLM-generated content at query #21
#--------------------------

```python
def test__PRecordEvolver_set():
    from pyrsistent import field, PRecord, InvariantException
    from pyrsistent._checked_types import CheckedType

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, invariant=lambda x: (x >= 0, 'age_negative'))
        optional = field(type=str, mandatory=False)

    # Test basic field setting
    evolver = TestRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 25)
    record = evolver.persistent()
    assert record['name'] == 'Alice'
    assert record['age'] == 25

    # Test type checking
    evolver = TestRecord().evolver()
    evolver.set('age', 'not_an_int')
    try:
        evolver.persistent()
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test field invariant
    evolver = TestRecord().evolver()
    evolver.set('name', 'Bob')
    evolver.set('age', -5)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'age_negative' in str(e)

    # Test non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('nonexistent', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'nonexistent' in str(e)

    # Test factory transformation
    class TestRecord2(PRecord):
        items = field(type=list, factory=lambda x: x if isinstance(x, list) else [x])

    evolver = TestRecord2().evolver()
    evolver.set('items', 'single')
    record = evolver.persistent()
    assert record['items'] == ['single']

    # Test with _factory_fields restriction
    class TestRecord3(PRecord):
        a = field(type=int)
        b = field(type=int)

    evolver = TestRecord3(_factory_fields={TestRecord3._precord_fields['a']}).evolver()
    evolver.set('a', 1)  # Should use factory
    evolver.set('b', 2)  # Should not use factory (direct assignment)
    record = evolver.persistent()
    assert record['a'] == 1
    assert record['b'] == 2

    # Test multiple invariant errors accumulation
    class TestRecord4(PRecord):
        x = field(type=int, invariant=lambda v: (v > 0, 'positive'))
        y = field(type=int, invariant=lambda v: (v < 10, 'less_than_10'))

    evolver = TestRecord4().evolver()
    evolver.set('x', -1)
    evolver.set('y', 20)
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'positive' in str(e)
        assert 'less_than_10' in str(e)

    # Test that original_value is preserved when factory is not in _factory_fields
    class TestRecord5(PRecord):
        data = field(type=dict, factory=lambda x: {'processed': x})

    evolver = TestRecord5(_factory_fields=set()).evolver()
    evolver.set('data', {'raw': 'value'})
    record = evolver.persistent()
    assert record['data'] == {'raw': 'value'}


# LLM-generated content at query #22
#--------------------------

```python
def test_PRecord_serialize():
    from pyrsistent import field
    
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, serializer=lambda x, _: x * 2)
        tags = field(type=list, serializer=lambda x, fmt: f"{fmt}:{len(x)}")
    
    # Test 1: Basic serialization without format
    record1 = TestRecord(name="Alice", age=25, tags=["a", "b"])
    result1 = record1.serialize()
    assert result1 == {
        "name": "Alice",
        "age": 50,  # age serializer multiplies by 2
        "tags": "None:2"  # tags serializer returns format:length
    }
    
    # Test 2: Serialization with custom format
    result2 = record1.serialize(format="json")
    assert result2 == {
        "name": "Alice",
        "age": 50,
        "tags": "json:2"
    }
    
    # Test 3: Serialization with fields without serializers
    class SimpleRecord(PRecord):
        value = field(type=int)
        text = field(type=str)
    
    record3 = SimpleRecord(value=42, text="hello")
    result3 = record3.serialize()
    assert result3 == {"value": 42, "text": "hello"}
    
    # Test 4: Serialization with None format
    result4 = record1.serialize(format=None)
    assert result4["tags"] == "None:2"
    
    # Test 5: Serialization with empty record
    record5 = TestRecord(name="", age=0, tags=[])
    result5 = record5.serialize()
    assert result5 == {"name": "", "age": 0, "tags": "None:0"}
    
    # Test 6: Verify original record unchanged
    assert record1.name == "Alice"
    assert record1.age == 25
    assert record1.tags == ["a", "b"]


# LLM-generated content at query #23
#--------------------------

```python
def test_PRecord___new__():
    # Test basic record creation with fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        active = field(type=bool, initial=lambda: True)
    
    # Test creation with initial values
    r1 = TestRecord(name="Alice", age=30)
    assert r1["name"] == "Alice"
    assert r1["age"] == 30
    assert r1["active"] == True
    
    # Test that mandatory fields are required
    try:
        TestRecord(name="Bob")
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "TestRecord.age" in e.missing_fields
    
    # Test with factory fields parameter
    r2 = TestRecord.create({"name": "Charlie", "age": 25}, _factory_fields=[TestRecord._precord_fields["name"]])
    assert r2["name"] == "Charlie"
    assert r2["age"] == 25
    assert r2["active"] == True
    
    # Test ignore_extra parameter
    r3 = TestRecord.create({"name": "David", "age": 35, "extra": "ignored"}, ignore_extra=True)
    assert r3["name"] == "David"
    assert r3["age"] == 35
    assert "extra" not in r3
    
    # Test direct internal creation with _precord_size and _precord_buckets
    internal_map = pmap({"name": "Eve", "age": 40, "active": False})
    r4 = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert r4["name"] == "Eve"
    assert r4["age"] == 40
    assert r4["active"] == False
    
    # Test that callable initial values work
    r5 = TestRecord(name="Frank", age=45)
    assert r5["active"] == True  # From lambda initial
    
    # Test type checking
    try:
        TestRecord(name="Grace", age="not_an_int")
        assert False, "Should have raised type error"
    except TypeError:
        pass
    
    # Test with no arguments (should use initial values where available)
    try:
        TestRecord()
        assert False, "Should have raised InvariantException for missing mandatory field"
    except InvariantException as e:
        assert "TestRecord.age" in e.missing_fields


# LLM-generated content at query #24
#--------------------------

```python
def test_PRecord___new__():
    # Test basic record creation with fields
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int, mandatory=True)
        active = field(type=bool, initial=lambda: True)
    
    # Test creation with all mandatory fields provided
    record = TestRecord(name="Alice", age=30)
    assert record["name"] == "Alice"
    assert record["age"] == 30
    assert record["active"] == True
    
    # Test creation with initial values
    record2 = TestRecord(name="Bob", age=25, active=False)
    assert record2["name"] == "Bob"
    assert record2["age"] == 25
    assert record2["active"] == False
    
    # Test missing mandatory field raises error
    try:
        TestRecord(name="Charlie")
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "TestRecord.age" in e.missing_fields
    
    # Test creation with internal fields for pickling
    record3 = TestRecord(_precord_size=2, _precord_buckets=pmap({"name": "Dave", "age": 40})._buckets)
    assert record3["name"] == "Dave"
    assert record3["age"] == 40
    assert record3["active"] == True
    
    # Test creation with factory fields
    record4 = TestRecord(name="Eve", age=35, _factory_fields=set())
    assert record4["name"] == "Eve"
    assert record4["age"] == 35
    
    # Test creation with ignore_extra
    record5 = TestRecord(name="Frank", age=45, extra_field="should_be_ignored", _ignore_extra=True)
    assert record5["name"] == "Frank"
    assert record5["age"] == 45
    assert "extra_field" not in record5
    
    # Test that ignore_extra without flag raises AttributeError
    try:
        TestRecord(name="Grace", age=50, extra_field="error")
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "extra_field" in str(e)
    
    # Test callable initial values
    class TestRecord2(PRecord):
        timestamp = field(type=float, initial=time.time)
        counter = field(type=int, initial=lambda: 0)
    
    import time
    record6 = TestRecord2()
    assert isinstance(record6["timestamp"], float)
    assert record6["counter"] == 0
    
    # Test that initial values can be overridden
    record7 = TestRecord2(timestamp=100.0, counter=5)
    assert record7["timestamp"] == 100.0
    assert record7["counter"] == 5
    
    # Test empty record creation
    class EmptyRecord(PRecord):
        pass
    
    record8 = EmptyRecord()
    assert len(record8) == 0


# LLM-generated content at query #25
#--------------------------

```python
def test_PRecord___repr__():
    class MyRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
        active = field(type=bool, initial=True)

    # Test basic repr with all fields
    r1 = MyRecord(name="Alice", age=30)
    expected1 = "MyRecord(active=True, age=30, name='Alice')"
    assert repr(r1) == expected1
    
    # Test repr with different field values
    r2 = MyRecord(name="Bob", age=25, active=False)
    expected2 = "MyRecord(active=False, age=25, name='Bob')"
    assert repr(r2) == expected2
    
    # Test repr with empty record
    class EmptyRecord(PRecord):
        pass
    
    r3 = EmptyRecord()
    expected3 = "EmptyRecord()"
    assert repr(r3) == expected3
    
    # Test repr with nested values
    class NestedRecord(PRecord):
        data = field(type=dict)
        items = field(type=list)
    
    r4 = NestedRecord(data={"key": "value"}, items=[1, 2, 3])
    expected4 = "NestedRecord(data={'key': 'value'}, items=[1, 2, 3])"
    assert repr(r4) == expected4
    
    # Test repr with special string values
    r5 = MyRecord(name="Test\nLine", age=40)
    expected5 = "MyRecord(active=True, age=40, name='Test\\nLine')"
    assert repr(r5) == expected5
    
    # Test repr maintains field order (alphabetical)
    class MultiFieldRecord(PRecord):
        z_field = field(type=str)
        a_field = field(type=str)
        m_field = field(type=str)
    
    r6 = MultiFieldRecord(z_field="z", a_field="a", m_field="m")
    expected6 = "MultiFieldRecord(a_field='a', m_field='m', z_field='z')"
    assert repr(r6) == expected6
    
    # Test repr with None values
    class OptionalRecord(PRecord):
        value = field(type=str, optional=True)
        count = field(type=int)
    
    r7 = OptionalRecord(count=10)
    expected7 = "OptionalRecord(count=10, value=None)"
    assert repr(r7) == expected7


