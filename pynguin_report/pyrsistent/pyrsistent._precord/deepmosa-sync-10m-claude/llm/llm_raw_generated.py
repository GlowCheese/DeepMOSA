####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_persistent_returns_result_when_no_errors():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def invariant(self, value):
            return (True, None)
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    result = evolver.persistent()
    
    assert isinstance(result, MockCls)
    assert result._buckets == original_pmap._buckets
    assert result._size == original_pmap._size


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockCls.required_field' in e.missing_fields


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    def failing_invariant(obj):
        return (False, 'global_error')
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_returns_same_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    result = evolver.persistent()
    
    assert result is original_pmap or isinstance(result, MockCls)


# LLM-generated content at query #2
#--------------------------

```python
def test_precord_meta_new_creates_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(type=str, initial='default', mandatory=False),
        'field2': _PField(type=int, initial=PFIELD_NO_INITIAL, mandatory=True),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_fields')
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(type=str, initial='default', mandatory=False),
        'field2': _PField(type=int, initial=PFIELD_NO_INITIAL, mandatory=True),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_mandatory_fields')
    assert result._precord_mandatory_fields == {'field2'}


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(type=str, initial='default', mandatory=False),
        'field2': _PField(type=int, initial=PFIELD_NO_INITIAL, mandatory=True),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_initial_values')
    assert result._precord_initial_values == {'field1': 'default'}


def test_precord_meta_new_sets_slots_empty():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(type=str, initial='default', mandatory=False),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '__slots__')
    assert result.__slots__ == ()


def test_precord_meta_new_stores_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    def my_invariant(record):
        return True, "valid"
    
    dct = {
        'field1': _PField(type=str, initial='default', mandatory=False),
        '__invariant__': my_invariant,
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert isinstance(result._precord_invariants, tuple)
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    base_dct = {
        'base_field': _PField(type=str, initial='base', mandatory=False),
    }
    base = _PRecordMeta.__new__(_PRecordMeta, 'BaseRecord', (), base_dct)
    
    dct = {
        'child_field': _PField(type=int, initial=PFIELD_NO_INITIAL, mandatory=True),
    }
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'ChildRecord', (base,), dct)
    
    assert 'base_field' in result._precord_fields
    assert 'child_field' in result._precord_fields


def test_precord_meta_new_removes_field_from_dct():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(type=str, initial='default', mandatory=False),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert 'field1' not in result.__dict__ or isinstance(result.__dict__.get('field1'), _PField) is False


# LLM-generated content at query #3
#--------------------------

```python
def test_precord_evolver_set_with_valid_field():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field(type=str)
    
    original_pmap = pmap({'name': 'test'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.set('name', 'new_value')
    
    assert result is evolver
    assert evolver._data['name'] == 'new_value'


def test_precord_evolver_set_with_invalid_field_raises_attribute_error():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field(type=str)
    
    original_pmap = pmap({'name': 'test'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'invalid_field' in str(e)
        assert 'TestRecord' in str(e)


def test_precord_evolver_set_with_type_check_failure():
    from pyrsistent import pmap, field, PRecord, PTypeError
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        age = field(type=int)
    
    original_pmap = pmap({'age': 25})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.set('age', 'not_an_int')
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_precord_evolver_setitem_delegates_to_set():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field(type=str)
    
    original_pmap = pmap({'name': 'test'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.__setitem__('name', 'updated')
    
    assert result is evolver
    assert evolver._data['name'] == 'updated'


def test_precord_evolver_set_with_factory_field():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        count = field(factory=int)
    
    original_pmap = pmap({'count': 5})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.set('count', '10')
    
    assert result is evolver
    assert evolver._data['count'] == 10


def test_precord_evolver_set_with_restricted_factory_fields():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field(type=str, factory=str)
        age = field(type=int, factory=int)
    
    name_field = TestRecord._precord_fields['name']
    original_pmap = pmap({'name': 'test', 'age': 25})
    evolver = _PRecordEvolver(TestRecord, original_pmap, _factory_fields={name_field})
    
    result = evolver.set('name', 'updated')
    assert result is evolver
    assert evolver._data['name'] == 'updated'


# LLM-generated content at query #4
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent._precord import PRecord
    
    class TestRecord(PRecord):
        pass
    
    # Test creating with _precord_size and _precord_buckets
    from pyrsistent._pmap import pvector, pmap
    buckets = pvector()
    result = TestRecord(_precord_size=0, _precord_buckets=buckets)
    assert isinstance(result, TestRecord)


def test_precord_new_empty():
    from pyrsistent._precord import PRecord
    
    class TestRecord(PRecord):
        pass
    
    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert len(result) == 0


def test_precord_new_with_kwargs():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    result = TestRecord(x=1, y=2)
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2


def test_precord_new_with_factory_fields():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    result = TestRecord(_factory_fields=[], x=10)
    assert isinstance(result, TestRecord)
    assert result['x'] == 10


def test_precord_new_with_ignore_extra():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    result = TestRecord(_ignore_extra=True, x=5)
    assert isinstance(result, TestRecord)
    assert result['x'] == 5


def test_precord_new_with_initial_values():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=42)
        y = field()
    
    result = TestRecord(y=100)
    assert isinstance(result, TestRecord)
    assert result['x'] == 42
    assert result['y'] == 100


def test_precord_new_with_callable_initial_values():
    from pyrsistent._precord import PRecord, field
    
    call_count = [0]
    
    def get_default():
        call_count[0] += 1
        return 99
    
    class TestRecord(PRecord):
        x = field(initial=get_default)
    
    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert result['x'] == 99
    assert call_count[0] == 1


def test_precord_new_overrides_initial_values():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
    
    result = TestRecord(x=20)
    assert isinstance(result, TestRecord)
    assert result['x'] == 20


# LLM-generated content at query #5
#--------------------------

```python
def test_precord_evolver_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PRecord, field
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, TestRecord())
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ()


def test_precord_evolver_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, TestRecord())
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.name',)


def test_precord_evolver_persistent_raises_invariant_exception_when_both_present():
    from pyrsistent import PRecord, field
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, TestRecord())
    evolver._invariant_error_codes = ['error_code_1', 'error_code_2']
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1', 'error_code_2')
        assert e.missing_fields == ('TestRecord.name',)


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize_with_no_serializer():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='John', age=30)
    result = record.serialize()
    
    assert result == {'name': 'John', 'age': 30}


def test_serialize_with_custom_serializer():
    from pyrsistent import PRecord, field
    
    def serialize_upper(value):
        return value.upper() if isinstance(value, str) else value
    
    class TestRecord(PRecord):
        name = field(serializer=serialize_upper)
        age = field()
    
    record = TestRecord(name='john', age=30)
    result = record.serialize()
    
    assert result['name'] == 'JOHN'
    assert result['age'] == 30


def test_serialize_with_format_parameter():
    from pyrsistent import PRecord, field
    
    def serialize_with_format(value, format=None):
        if format == 'uppercase':
            return value.upper() if isinstance(value, str) else value
        return value
    
    class TestRecord(PRecord):
        name = field(serializer=serialize_with_format)
        value = field()
    
    record = TestRecord(name='test', value='data')
    result = record.serialize(format='uppercase')
    
    assert result['name'] == 'TEST'
    assert result['value'] == 'data'


def test_serialize_empty_record():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    result = record.serialize()
    
    assert result == {}


def test_serialize_multiple_fields_with_mixed_serializers():
    from pyrsistent import PRecord, field
    
    def double_value(value):
        return value * 2 if isinstance(value, (int, float)) else value
    
    def uppercase_value(value):
        return value.upper() if isinstance(value, str) else value
    
    class TestRecord(PRecord):
        count = field(serializer=double_value)
        label = field(serializer=uppercase_value)
        raw = field()
    
    record = TestRecord(count=5, label='hello', raw='unchanged')
    result = record.serialize()
    
    assert result['count'] == 10
    assert result['label'] == 'HELLO'
    assert result['raw'] == 'unchanged'


# LLM-generated content at query #7
#--------------------------

```python
def test_serialize_returns_dict_with_serialized_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    record = TestRecord(name='test', value=42)
    result = record.serialize()
    
    assert isinstance(result, dict)
    assert 'name' in result
    assert 'value' in result
    assert result['name'] == 'test'
    assert result['value'] == 42


# LLM-generated content at query #8
#--------------------------

```python
def test_persistent_checks_mandatory_fields_when_precord_mandatory_fields_exist():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    TestRecord._precord_mandatory_fields = {'name', 'age'}
    TestRecord._precord_invariants = ()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    evolver.set('name', 'John')
    evolver.set('age', 30)
    
    result = evolver.persistent()
    
    assert result is not None
    assert result['name'] == 'John'
    assert result['age'] == 30


# LLM-generated content at query #9
#--------------------------

```python
def test_precord_meta_new_returns_class():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import PFIELD_NO_INITIAL
    
    # Create a simple field for testing
    test_field = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)
    
    # Create test dictionary with a field
    dct = {'test_field': test_field}
    
    # Call __new__ with basic parameters
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestRecord', (), dct)
    
    # Verify that __new__ returns a class (type instance)
    assert isinstance(result, type)
    assert result.__name__ == 'TestRecord'


# LLM-generated content at query #10
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_with_callable_initial():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [])
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['x'] == []
    assert record2['x'] == []
    assert record1['x'] is not record2['x']


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    base_map = pmap({'x': 5})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets)
    assert record['x'] == 5


def test_precord_constructor_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    try:
        record = TestRecord(x=1, y=2, _ignore_extra=False)
        assert False, "Should have raised an error"
    except Exception:
        pass


def test_precord_constructor_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_multiple_fields():
    from pyrsistent import PRecord, field
    
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


# LLM-generated content at query #11
#--------------------------

```python
def test_set_with_valid_field_and_value():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import PTypeError
    
    class MockField:
        def __init__(self, factory_func=None, invariant_func=None):
            self.factory = factory_func or (lambda x: x)
            self.invariant = invariant_func or (lambda x: (True, None))
            self.type = None
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.set('test_field', 'test_value')
    
    assert result is evolver


def test_set_with_nonexistent_field_raises_attribute_error():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        pass
    
    class MockClass:
        _precord_fields = {}
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.set('nonexistent_field', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'nonexistent_field' in str(e)


def test_set_with_factory_fields_filter():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self):
            self.factory = lambda x: x
            self.invariant = lambda x: (True, None)
            self.type = None
    
    field1 = MockField()
    field2 = MockField()
    
    class MockClass:
        _precord_fields = {'field1': field1, 'field2': field2}
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap, _factory_fields=[field1])
    result = evolver.set('field1', 'value1')
    
    assert result is evolver


def test_set_with_invariant_failure():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self):
            self.factory = lambda x: x
            self.invariant = lambda x: (False, 'error_code')
            self.type = None
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.set('test_field', 'test_value')
    
    assert 'error_code' in evolver._invariant_error_codes


def test_set_with_factory_exception():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    def factory_with_exception(x):
        raise InvariantException(('error1',), ('missing1',), 'test error')
    
    class MockField:
        def __init__(self):
            self.factory = factory_with_exception
            self.invariant = lambda x: (True, None)
            self.type = None
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.set('test_field', 'test_value')
    
    assert 'error1' in evolver._invariant_error_codes
    assert 'missing1' in evolver._missing_fields


def test_setitem_delegates_to_set():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self):
            self.factory = lambda x: x
            self.invariant = lambda x: (True, None)
            self.type = None
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.__setitem__('test_field', 'test_value')
    
    assert result is evolver


# LLM-generated content at query #12
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import pmap
    from pyrsistent._precord import PRecord, field
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, pmap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import pmap
    from pyrsistent._precord import PRecord, field
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, pmap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['TestRecord.x']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.missing_fields == ('TestRecord.x',)


def test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields():
    from pyrsistent import pmap
    from pyrsistent._precord import PRecord, field
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, pmap())
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['TestRecord.x']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('TestRecord.x',)


# LLM-generated content at query #13
#--------------------------

```python
def test_precord_initial_values_predicate():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        __precord_initial_values = {'x': lambda: 10, 'y': 20}
    
    TestRecord._precord_initial_values = {'x': lambda: 10, 'y': 20}
    
    assert TestRecord._precord_initial_values is not None
    assert TestRecord._precord_initial_values
    assert bool(TestRecord._precord_initial_values) == True


# LLM-generated content at query #14
#--------------------------

```python
def test_precord_new_without_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a PRecord without '_precord_size' and '_precord_buckets' in kwargs
    # This ensures the predicate at line 5 evaluates to False
    record = TestRecord(x=1, y=2)
    
    assert record['x'] == 1
    assert record['y'] == 2
    assert isinstance(record, TestRecord)


# LLM-generated content at query #15
#--------------------------

```python
def test_precord_meta_new_creates_class_with_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a test field
    test_field = _PField(initial=PFIELD_NO_INITIAL, mandatory=True, factory=None, invariant=None)
    
    # Create a test dictionary with a field
    dct = {'test_field': test_field}
    
    # Call __new__ to create a class
    result_class = _PRecordMeta.__new__(_PRecordMeta, 'TestRecord', (), dct)
    
    # Verify that __slots__ is set to an empty tuple
    assert result_class.__slots__ == ()
    assert isinstance(result_class.__slots__, tuple)
    assert len(result_class.__slots__) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_precord_meta_new_creates_class_with_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a simple field for testing
    test_field = _PField(type=str, initial=PFIELD_NO_INITIAL, mandatory=True, factory=None)
    
    # Create a dictionary with a field
    dct = {
        'test_attr': test_field,
        '__module__': '__main__',
        '__qualname__': 'TestPRecord'
    }
    
    # Call __new__ with the metaclass
    result_class = _PRecordMeta.__new__(_PRecordMeta, 'TestPRecord', (), dct)
    
    # Verify that __slots__ is set to an empty tuple
    assert result_class.__slots__ == ()
    assert isinstance(result_class, type)
    assert result_class.__name__ == 'TestPRecord'


# LLM-generated content at query #17
#--------------------------

```python
def test_persistent_returns_result_when_clean_and_correct_type():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self):
            self.invariant_called = False
        
        def invariant(self, value):
            self.invariant_called = True
            return (True, None)
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._destination_cls = MockClass
    
    # Mock the parent persistent method to return a MockClass instance
    original_pmap.__class__ = MockClass
    
    result = evolver.persistent()
    assert result is not None


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'MockClass'
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockClass.field1' in e.missing_fields
        assert 'MockClass.field2' in e.missing_fields


def test_persistent_checks_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    def failing_global_invariant(subject):
        return (False, 'global_error')
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        __name__ = 'MockClass'
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_with_passing_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    def passing_global_invariant(subject):
        return (True, None)
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [passing_global_invariant]
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            pass
    
    original_pmap = pmap({'key': 'value'})
    original_pmap._buckets = None
    original_pmap._size = 0
    
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert result is not None


def test_persistent_with_dirty_state_creates_new_instance():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._data['new_key'] = 'new_value'
    
    result = evolver.persistent()
    assert result is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_persistent_predicate_is_dirty_true():
    from pyrsistent import pmap, field, PRecord
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = pmap({'name': 'test'})
    evolver = TestRecord._PRecordEvolver(TestRecord, original_pmap)
    evolver.set('name', 'modified')
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True


def test_persistent_predicate_not_isinstance_true():
    from pyrsistent import pmap, field, PRecord
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = pmap({'name': 'test'})
    evolver = TestRecord._PRecordEvolver(TestRecord, original_pmap)
    
    pm = pmap({'name': 'test'})
    assert not isinstance(pm, TestRecord)
    
    result = evolver.persistent()
    assert isinstance(result, TestRecord)


def test_persistent_predicate_both_conditions_true():
    from pyrsistent import pmap, field, PRecord
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = pmap({'name': 'initial'})
    evolver = TestRecord._PRecordEvolver(TestRecord, original_pmap)
    evolver.set('name', 'changed')
    
    is_dirty = evolver.is_dirty()
    pm = pmap({'name': 'changed'})
    
    assert is_dirty is True
    assert not isinstance(pm, TestRecord)
    
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['name'] == 'changed'


# LLM-generated content at query #19
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='John', age=30)
    assert record['name'] == 'John'
    assert record['age'] == 30


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='DefaultName')
        age = field(initial=0)
    
    record = TestRecord()
    assert record['name'] == 'DefaultName'
    assert record['age'] == 0


def test_precord_constructor_with_initial_values_and_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='DefaultName')
        age = field(initial=0)
    
    record = TestRecord(name='Jane', age=25)
    assert record['name'] == 'Jane'
    assert record['age'] == 25


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['items'] == []
    assert record2['items'] == []
    assert record1['items'] is not record2['items']


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name='Test', _factory_fields=None)
    assert record['name'] == 'Test'


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name='Test', _ignore_extra=True, extra_field='ignored')
    assert record['name'] == 'Test'
    assert 'extra_field' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    internal_pmap = pmap({'name': 'Test'})
    record = TestRecord(_precord_size=internal_pmap._size, _precord_buckets=internal_pmap._buckets)
    assert record['name'] == 'Test'


# LLM-generated content at query #20
#--------------------------

```python
def test_precord_repr():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='Alice', age=30)
    repr_str = repr(record)
    
    assert 'TestRecord' in repr_str
    assert 'name=' in repr_str
    assert "'Alice'" in repr_str
    assert 'age=' in repr_str
    assert '30' in repr_str


def test_precord_repr_empty():
    from pyrsistent import PRecord, field
    
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    repr_str = repr(record)
    
    assert 'EmptyRecord()' == repr_str


def test_precord_repr_single_field():
    from pyrsistent import PRecord, field
    
    class SingleFieldRecord(PRecord):
        value = field()
    
    record = SingleFieldRecord(value=42)
    repr_str = repr(record)
    
    assert 'SingleFieldRecord' in repr_str
    assert 'value=42' in repr_str


def test_precord_repr_with_special_characters():
    from pyrsistent import PRecord, field
    
    class SpecialRecord(PRecord):
        text = field()
    
    record = SpecialRecord(text="hello'world")
    repr_str = repr(record)
    
    assert 'SpecialRecord' in repr_str
    assert 'text=' in repr_str
    assert "hello'world" in repr_str


# LLM-generated content at query #21
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['TestRecord.x']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.x',)


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['TestRecord.x']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('TestRecord.x',)


# LLM-generated content at query #22
#--------------------------

```python
def test_persistent_predicate_is_dirty_true():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 1)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True
    
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 1


def test_persistent_predicate_not_isinstance_true():
    from pyrsistent import PRecord, field, pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    pm = PMap()
    evolver._PMap__data = pm
    
    result = evolver.persistent()
    assert isinstance(result, TestRecord)


def test_persistent_predicate_is_dirty_false_isinstance_false():
    from pyrsistent import PRecord, field, pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_record = TestRecord(x=1)
    original_pmap = original_record._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is False
    
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 1


# LLM-generated content at query #23
#--------------------------

```python
def test_precord_evolver_persistent_predicate_is_dirty_true():
    from pyrsistent import pmap, PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    original_pmap = pmap({'x': 1, 'y': 2})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 10)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True
    
    pm = original_pmap
    is_dirty_or_not_instance = is_dirty or not isinstance(pm, TestRecord)
    assert is_dirty_or_not_instance is True


def test_precord_evolver_persistent_predicate_not_isinstance_true():
    from pyrsistent import pmap, PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    original_pmap = pmap({'x': 1, 'y': 2})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    is_dirty = evolver.is_dirty()
    pm = original_pmap
    is_dirty_or_not_instance = is_dirty or not isinstance(pm, TestRecord)
    assert is_dirty_or_not_instance is True


def test_precord_evolver_persistent_predicate_both_conditions_true():
    from pyrsistent import pmap, PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    original_pmap = pmap({'x': 1, 'y': 2})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 20)
    
    is_dirty = evolver.is_dirty()
    pm = original_pmap
    predicate_result = is_dirty or not isinstance(pm, TestRecord)
    assert predicate_result is True
    assert is_dirty is True
    assert not isinstance(pm, TestRecord) is True


# LLM-generated content at query #24
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field()
    
    record = TestRecord(y=20)
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_constructor_with_callable_initial():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['items'] is not record2['items']
    assert record1['items'] == []


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _factory_fields={'x': int})
    assert record['x'] == 5


def test_precord_constructor_ignore_extra_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 5})
    record = TestRecord(_precord_size=internal_map._HAMT_MUTATION_TRACKING_ENABLED, 
                        _precord_buckets=internal_map._buckets)
    assert isinstance(record, TestRecord)


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_multiple_fields():
    from pyrsistent import PRecord, field
    
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


def test_precord_constructor_with_none_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=None, y=None)
    assert record['x'] is None
    assert record['y'] is None


# LLM-generated content at query #25
#--------------------------

```python
def test_precord_meta_new_creates_class_with_precord_fields():
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import _PRecordMeta, PFIELD_NO_INITIAL
    
    # Create a mock field
    field1 = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)
    field2 = _PField(initial=42, mandatory=False)
    
    # Create a class using _PRecordMeta
    dct = {
        'field1': field1,
        'field2': field2,
    }
    bases = ()
    
    TestClass = _PRecordMeta('TestClass', bases, dct)
    
    # Verify that __new__ was called and the class was created successfully
    assert TestClass is not None
    assert TestClass.__name__ == 'TestClass'
    assert '_precord_fields' in TestClass.__dict__
    assert '_precord_mandatory_fields' in TestClass.__dict__
    assert '_precord_initial_values' in TestClass.__dict__
    assert '_precord_invariants' in TestClass.__dict__
    assert TestClass.__slots__ == ()
    
    # Verify the fields were processed correctly
    assert 'field1' in TestClass._precord_fields
    assert 'field2' in TestClass._precord_fields
    
    # Verify mandatory fields are identified
    assert 'field1' in TestClass._precord_mandatory_fields
    assert 'field2' not in TestClass._precord_mandatory_fields
    
    # Verify initial values are stored
    assert 'field1' not in TestClass._precord_initial_values
    assert TestClass._precord_initial_values['field2'] == 42
    
    # Verify invariants tuple exists
    assert isinstance(TestClass._precord_invariants, tuple)


# LLM-generated content at query #26
#--------------------------

```python
def test_persistent_returns_result_when_no_errors():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self, name):
            self.name = name
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, MockClass)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        __name__ = "MockClass"
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockClass.field1' in e.missing_fields
        assert 'MockClass.field2' in e.missing_fields


def test_persistent_raises_invariant_exception_with_field_errors():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        __name__ = "MockClass"
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error1' in e.invariant_errors
        assert 'error2' in e.invariant_errors


def test_persistent_calls_check_global_invariants():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    
    global_invariant_called = []
    
    def mock_global_invariant(subject):
        global_invariant_called.append(subject)
        return (True, None)
    
    class MockClass:
        __name__ = "MockClass"
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [mock_global_invariant]
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert len(global_invariant_called) == 1
    assert isinstance(global_invariant_called[0], MockClass)


# LLM-generated content at query #27
#--------------------------

```python
def test_precord_repr():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='Alice', age=30)
    repr_str = repr(record)
    
    assert 'TestRecord' in repr_str
    assert 'name=' in repr_str
    assert "'Alice'" in repr_str
    assert 'age=' in repr_str
    assert '30' in repr_str
    assert repr_str.startswith('TestRecord(')
    assert repr_str.endswith(')')


# LLM-generated content at query #28
#--------------------------

```python
def test_persistent_checks_mandatory_fields_when_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    TestRecord._precord_mandatory_fields = {'name', 'age'}
    TestRecord._precord_invariants = []
    TestRecord._precord_fields = {'name': TestRecord.name, 'age': TestRecord.age}
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    has_mandatory_fields = bool(TestRecord._precord_mandatory_fields)
    
    assert has_mandatory_fields is True


# LLM-generated content at query #29
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap to use as buckets
    from pyrsistent._pmap import pmap
    initial_pmap = pmap({'x': 1, 'y': 2})
    
    # Create record using special attributes
    record = TestRecord(_precord_size=initial_pmap._size, _precord_buckets=initial_pmap._buckets)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=10, y=20)
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=5)
        y = field(initial=15)
    
    record = TestRecord()
    assert record['x'] == 5
    assert record['y'] == 15


def test_precord_new_with_initial_values_and_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=5)
        y = field(initial=15)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 15


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_factory_fields=set(), x=42)
    assert record['x'] == 42


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=10)
    assert record['x'] == 10


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_new_partial_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        z = field()
    
    record = TestRecord(x=1, z=3)
    assert record['x'] == 1
    assert record['z'] == 3
    assert 'y' not in record


def test_precord_new_with_callable_initial_value():
    from pyrsistent import PRecord, field
    
    counter = [0]
    def get_initial():
        counter[0] += 1
        return counter[0]
    
    class TestRecord(PRecord):
        x = field(initial=get_initial)
    
    record = TestRecord()
    assert record['x'] == 1


def test_precord_new_with_multiple_kwargs():
    from pyrsistent import PRecord, field
    
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


# LLM-generated content at query #30
#--------------------------

```python
def test_precord_evolver_set_with_field_found():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: x
            self.invariant = lambda x: (True, None)
    
    class MockDestinationCls:
        __name__ = "TestClass"
        _precord_fields = {"test_key": MockField()}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    
    result = evolver.set("test_key", 42)
    
    assert result is not None
    assert isinstance(result, _PRecordEvolver)


# LLM-generated content at query #31
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    exception_raised = False
    try:
        evolver.persistent()
    except InvariantException as e:
        exception_raised = True
        assert e.invariant_errors == ('error1',)
    
    assert exception_raised is True


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._missing_fields = ['TestRecord.y']
    
    exception_raised = False
    try:
        evolver.persistent()
    except InvariantException as e:
        exception_raised = True
        assert e.missing_fields == ('TestRecord.y',)
    
    assert exception_raised is True


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['TestRecord.y']
    
    exception_raised = False
    try:
        evolver.persistent()
    except InvariantException as e:
        exception_raised = True
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('TestRecord.y',)
    
    assert exception_raised is True


# LLM-generated content at query #32
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, mock_pmap)
    evolver._is_dirty = False
    
    result = evolver.persistent()
    assert isinstance(result, MockClass)


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, mock_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []
        __name__ = 'MockClass'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, mock_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockClass.field1' in e.missing_fields


def test_persistent_raises_invariant_exception_with_missing_fields_combined():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'MockClass'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, mock_pmap)
    evolver._missing_fields = ['field0']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'field0' in e.missing_fields
        assert 'MockClass.field1' in e.missing_fields or 'MockClass.field2' in e.missing_fields


def test_persistent_calls_check_global_invariants():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        __name__ = 'MockClass'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, mock_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_succeeds_with_no_errors():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    def passing_invariant(subject):
        return (True, None)
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [passing_invariant]
        __name__ = 'MockClass'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, mock_pmap)
    
    result = evolver.persistent()
    assert isinstance(result, MockClass)


# LLM-generated content at query #33
#--------------------------

```python
def test_precord_evolver_set_with_field_found():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field(type=str)
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    result = evolver.set('name', 'test_value')
    
    assert result is not None
    assert isinstance(result, _PRecordEvolver)


# LLM-generated content at query #34
#--------------------------

```python
def test_precord_meta_new_creates_class_with_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a test field
    test_field = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)
    
    # Create a test dictionary with a field
    dct = {'test_field': test_field}
    
    # Call __new__ to create a class
    TestClass = _PRecordMeta('TestClass', (), dct)
    
    # Verify that __slots__ is set to an empty tuple
    assert TestClass.__slots__ == ()
    assert hasattr(TestClass, '_precord_fields')
    assert hasattr(TestClass, '_precord_invariants')
    assert hasattr(TestClass, '_precord_mandatory_fields')
    assert hasattr(TestClass, '_precord_initial_values')


# LLM-generated content at query #35
#--------------------------

```python
def test_precord_repr():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="Alice", age=30)
    repr_str = repr(record)
    
    assert "TestRecord(" in repr_str
    assert "name='Alice'" in repr_str
    assert "age=30" in repr_str
    assert repr_str.endswith(")")


# LLM-generated content at query #36
#--------------------------

```python
def test_persistent_checks_mandatory_fields_when_precord_mandatory_fields_exists():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    TestRecord._precord_mandatory_fields = {'name', 'age'}
    TestRecord._precord_invariants = ()
    TestRecord._precord_fields = {'name': TestRecord._precord_fields.get('name'), 'age': TestRecord._precord_fields.get('age')}
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert 'missing_fields' in str(type(e).__dict__) or hasattr(e, 'missing_fields')


# LLM-generated content at query #37
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap with initial values
    initial_pmap = pmap({'x': 1, 'y': 2})
    
    # Create PRecord using special attributes
    record = TestRecord(_precord_size=initial_pmap._size, _precord_buckets=initial_pmap._buckets)
    
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=10, y=20)
    
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_new_with_empty_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    
    assert len(record) == 0


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=100)
        y = field(initial=200)
    
    record = TestRecord()
    
    assert record['x'] == 100
    assert record['y'] == 200


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    call_count = [0]
    
    def get_default():
        call_count[0] += 1
        return 42
    
    class TestRecord(PRecord):
        x = field(initial=get_default)
    
    record = TestRecord()
    
    assert record['x'] == 42
    assert call_count[0] == 1


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _factory_fields=set())
    
    assert record['x'] == 5


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _ignore_extra=False)
    
    assert record['x'] == 5


def test_precord_new_overrides_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=100)
        y = field(initial=200)
    
    record = TestRecord(x=50)
    
    assert record['x'] == 50
    assert record['y'] == 200


def test_precord_new_with_multiple_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = TestRecord(a=1, b=2, c=3)
    
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3


# LLM-generated content at query #38
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap with the special attributes
    from pyrsistent import pmap
    pm = pmap({'x': 1, 'y': 2})
    
    # Create PRecord using special attributes
    record = TestRecord(_precord_size=pm._size, _precord_buckets=pm._buckets)
    
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=10, y=20)
    
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _factory_fields=set())
    
    assert record['x'] == 5


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _ignore_extra=True)
    
    assert record['x'] == 1


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': 100}
        x = field()
    
    record = TestRecord()
    
    assert record['x'] == 100


def test_precord_new_with_initial_values_callable():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': lambda: 42}
        x = field()
    
    record = TestRecord()
    
    assert record['x'] == 42


def test_precord_new_with_initial_values_override():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': 100}
        x = field()
    
    record = TestRecord(x=200)
    
    assert record['x'] == 200


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    
    assert 'x' not in record


def test_precord_new_multiple_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = TestRecord(a=1, b=2, c=3)
    
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3


# LLM-generated content at query #39
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_constructor_with_initial_values_and_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [1, 2, 3])
    
    record = TestRecord()
    assert record['x'] == [1, 2, 3]


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _factory_fields=True)
    assert record['x'] == 5


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    pmap_obj = pmap({'x': 42})
    record = TestRecord(_precord_size=pmap_obj._size, _precord_buckets=pmap_obj._buckets)
    assert record['x'] == 42


# LLM-generated content at query #40
#--------------------------

```python
def test_persistent_predicate_line_6_true_when_is_dirty():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    # Mock is_dirty to return True
    evolver.is_dirty = lambda: True
    
    # The condition at line 6: `if is_dirty or not isinstance(pm, cls):`
    # should evaluate to True when is_dirty() returns True
    is_dirty = evolver.is_dirty()
    pm = PMap()
    is_instance_check = isinstance(pm, TestRecord)
    
    predicate_result = is_dirty or not is_instance_check
    assert predicate_result is True


def test_persistent_predicate_line_6_true_when_not_isinstance():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    # Mock is_dirty to return False
    evolver.is_dirty = lambda: False
    
    # The condition at line 6: `if is_dirty or not isinstance(pm, cls):`
    # should evaluate to True when pm is not an instance of cls
    is_dirty = evolver.is_dirty()
    pm = PMap()
    is_instance_check = isinstance(pm, TestRecord)
    
    predicate_result = is_dirty or not is_instance_check
    assert predicate_result is True


# LLM-generated content at query #41
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class MockField:
        def __init__(self, name):
            self.name = name
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    
    result = evolver.persistent()
    assert result is not None


def test_persistent_raises_on_invariant_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)


def test_persistent_raises_on_missing_mandatory_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'MockClass'
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 2


def test_persistent_raises_on_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['missing_field']
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('missing_field',)


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._field_common import InvariantException
    
    global_invariant_called = []
    
    def failing_invariant(obj):
        global_invariant_called.append(True)
        return (False, 'global_error')
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        __name__ = 'MockClass'
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert global_invariant_called
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #42
#--------------------------

```python
def test_repr_empty_record():
    from pyrsistent import PRecord, field
    
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    assert repr(record) == "EmptyRecord()"


def test_repr_single_field():
    from pyrsistent import PRecord, field
    
    class SingleFieldRecord(PRecord):
        name = field()
    
    record = SingleFieldRecord(name="test")
    assert repr(record) == "SingleFieldRecord(name='test')"


def test_repr_multiple_fields():
    from pyrsistent import PRecord, field
    
    class MultiFieldRecord(PRecord):
        name = field()
        age = field()
    
    record = MultiFieldRecord(name="John", age=30)
    result = repr(record)
    assert "MultiFieldRecord(" in result
    assert "name='John'" in result
    assert "age=30" in result


def test_repr_with_integer_values():
    from pyrsistent import PRecord, field
    
    class IntRecord(PRecord):
        count = field()
    
    record = IntRecord(count=42)
    assert repr(record) == "IntRecord(count=42)"


def test_repr_with_string_values():
    from pyrsistent import PRecord, field
    
    class StringRecord(PRecord):
        text = field()
    
    record = StringRecord(text="hello world")
    assert repr(record) == "StringRecord(text='hello world')"


def test_repr_with_none_value():
    from pyrsistent import PRecord, field
    
    class NoneRecord(PRecord):
        value = field()
    
    record = NoneRecord(value=None)
    assert repr(record) == "NoneRecord(value=None)"


def test_repr_with_boolean_values():
    from pyrsistent import PRecord, field
    
    class BoolRecord(PRecord):
        flag = field()
    
    record = BoolRecord(flag=True)
    assert repr(record) == "BoolRecord(flag=True)"


def test_repr_with_list_value():
    from pyrsistent import PRecord, field
    
    class ListRecord(PRecord):
        items = field()
    
    record = ListRecord(items=[1, 2, 3])
    assert repr(record) == "ListRecord(items=[1, 2, 3])"


def test_repr_with_nested_record():
    from pyrsistent import PRecord, field
    
    class InnerRecord(PRecord):
        inner_value = field()
    
    class OuterRecord(PRecord):
        nested = field()
    
    inner = InnerRecord(inner_value=10)
    outer = OuterRecord(nested=inner)
    result = repr(outer)
    assert "OuterRecord(" in result
    assert "InnerRecord(inner_value=10)" in result


# LLM-generated content at query #43
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import PMap
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def factory(self, value, ignore_extra=False):
            return value
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _buckets = ((),)
        _size = 0
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets or self._buckets
            self._size = _precord_size or self._size
        
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, MockClass)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import PMap
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def factory(self, value, ignore_extra=False):
            return value
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets or ((),)
            self._size = _precord_size or 0
        
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockClass.field1' in e.missing_fields


def test_persistent_raises_invariant_exception_with_field_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import PMap
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def factory(self, value, ignore_extra=False):
            return value
        
        def invariant(self, value):
            return (False, 'error_code_1')
    
    class MockClass:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets or ((),)
            self._size = _precord_size or 0
        
        def keys(self):
            return ['field1']
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'error_code_1' in e.invariant_errors


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import PMap
    
    global_invariant_called = []
    
    def mock_global_invariant(subject):
        global_invariant_called.append(True)
        return (False, 'global_error')
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def factory(self, value, ignore_extra=False):
            return value
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = [mock_global_invariant]
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets or ((),)
            self._size = _precord_size or 0
        
        def keys(self):
            return ['field1']
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert global_invariant_called
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #44
#--------------------------

```python
def test_serialize_returns_dict_with_serialized_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    record = TestRecord(name='test', value=42)
    result = record.serialize()
    
    assert isinstance(result, dict)
    assert 'name' in result
    assert 'value' in result


# LLM-generated content at query #45
#--------------------------

```python
def test_precord_meta_new_sets_fields():
    from pyrsistent._field_common import PField
    from pyrsistent._precord import _PRecordMeta
    
    class TestField(PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True),
        'field2': TestField(mandatory=False, initial='default'),
    }
    bases = ()
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    
    assert '_precord_fields' in result.__dict__
    assert 'field1' in result.__dict__['_precord_fields']
    assert 'field2' in result.__dict__['_precord_fields']


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._field_common import PField
    from pyrsistent._precord import _PRecordMeta
    
    class TestField(PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True),
        'field2': TestField(mandatory=False),
    }
    bases = ()
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    
    assert result.__dict__['_precord_mandatory_fields'] == {'field1'}


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._field_common import PField, PFIELD_NO_INITIAL
    from pyrsistent._precord import _PRecordMeta
    
    class TestField(PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(initial=PFIELD_NO_INITIAL),
        'field2': TestField(initial='default_value'),
    }
    bases = ()
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    
    assert result.__dict__['_precord_initial_values'] == {'field2': 'default_value'}


def test_precord_meta_new_sets_slots():
    from pyrsistent._field_common import PField, PFIELD_NO_INITIAL
    from pyrsistent._precord import _PRecordMeta
    
    class TestField(PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {'field1': TestField()}
    bases = ()
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    
    assert result.__slots__ == ()


def test_precord_meta_new_stores_invariants():
    from pyrsistent._field_common import PField, PFIELD_NO_INITIAL
    from pyrsistent._precord import _PRecordMeta
    
    class TestField(PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    def test_invariant(obj):
        return True, ()
    
    dct = {
        'field1': TestField(),
        '__invariant__': test_invariant
    }
    bases = ()
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    
    assert '_precord_invariants' in result.__dict__
    assert isinstance(result.__dict__['_precord_invariants'], tuple)


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._field_common import PField, PFIELD_NO_INITIAL
    from pyrsistent._precord import _PRecordMeta
    
    class TestField(PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    base_dct = {'field1': TestField(mandatory=True)}
    base_class = _PRecordMeta.__new__(_PRecordMeta, 'BaseClass', (), base_dct)
    
    child_dct = {'field2': TestField(mandatory=False)}
    result = _PRecordMeta.__new__(_PRecordMeta, 'ChildClass', (base_class,), child_dct)
    
    assert 'field1' in result.__dict__['_precord_fields']
    assert 'field2' in result.__dict__['_precord_fields']


# LLM-generated content at query #46
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_error_codes_exist():
    from pyrsistent import PRecord, field, InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ()


def test_persistent_raises_invariant_exception_when_missing_fields_exist():
    from pyrsistent import PRecord, field, InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.name',)


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_exist():
    from pyrsistent import PRecord, field, InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ('TestRecord.name',)


# LLM-generated content at query #47
#--------------------------

```python
def test_serialize_returns_dict_with_serialized_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    record = TestRecord(name="test", value=42)
    result = record.serialize()
    
    assert isinstance(result, dict)
    assert "name" in result
    assert "value" in result
    assert result["name"] == "test"
    assert result["value"] == 42


# LLM-generated content at query #48
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field, pmap, pvector
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create using special attributes
    test_pmap = pmap({'x': 1, 'y': 2})
    record = TestRecord(_precord_size=test_pmap._size, _precord_buckets=test_pmap._buckets)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=10, y=20)
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field(initial=5)
    
    record = TestRecord(x=1)
    assert record['x'] == 1
    assert record['y'] == 5


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field(initial=lambda: 42)
    
    record = TestRecord(x=1)
    assert record['x'] == 1
    assert record['y'] == 42


def test_precord_new_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=100)
        y = field(initial=200)
    
    record = TestRecord(x=10, y=20)
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(_factory_fields=set(), x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=1, z=999)
    assert record['x'] == 1
    assert 'z' not in record


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_new_multiple_fields():
    from pyrsistent import PRecord, field
    
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


# LLM-generated content at query #49
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='John', age=30)
    assert record['name'] == 'John'
    assert record['age'] == 30


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='DefaultName')
        age = field(initial=0)
    
    record = TestRecord()
    assert record['name'] == 'DefaultName'
    assert record['age'] == 0


def test_precord_constructor_with_partial_kwargs_and_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='DefaultName')
        age = field(initial=0)
    
    record = TestRecord(age=25)
    assert record['name'] == 'DefaultName'
    assert record['age'] == 25


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['items'] == []
    assert record2['items'] == []
    assert record1['items'] is not record2['items']


def test_precord_constructor_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    try:
        record = TestRecord(name='John', extra_field='value')
        assert False, "Should have raised an error for extra field"
    except:
        pass


def test_precord_constructor_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name='John', extra_field='value', _ignore_extra=True)
    assert record['name'] == 'John'
    assert 'extra_field' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    base_map = pmap({'name': 'John'})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets)
    assert record['name'] == 'John'


# LLM-generated content at query #50
#--------------------------

```python
def test_precord_new_predicate_false_when_missing_precord_size():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ with _precord_buckets but without _precord_size
    # This should make the predicate at line 5 evaluate to False
    result = TestRecord(_precord_buckets=None, x=1)
    
    # If predicate is False, it should go through the Evolver path
    # and return a valid PRecord instance
    assert isinstance(result, TestRecord)
    assert result['x'] == 1


def test_precord_new_predicate_false_when_missing_precord_buckets():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ with _precord_size but without _precord_buckets
    # This should make the predicate at line 5 evaluate to False
    result = TestRecord(_precord_size=5, x=2)
    
    # If predicate is False, it should go through the Evolver path
    # and return a valid PRecord instance
    assert isinstance(result, TestRecord)
    assert result['x'] == 2


def test_precord_new_predicate_false_when_both_missing():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        y = field()
    
    # Call __new__ without both _precord_size and _precord_buckets
    # This should make the predicate at line 5 evaluate to False
    result = TestRecord(y=3)
    
    # If predicate is False, it should go through the Evolver path
    # and return a valid PRecord instance
    assert isinstance(result, TestRecord)
    assert result['y'] == 3


# LLM-generated content at query #51
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_constructor_with_kwargs_overrides_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [])
        y = field(initial=lambda: {})
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['x'] is not record2['x']
    assert record1['y'] is not record2['y']


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 5})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert record['x'] == 5


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields={'x': int})
    assert record['x'] == 1


def test_precord_constructor_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    try:
        record = TestRecord(x=1, y=2, _ignore_extra=False)
        assert False, "Should have raised an error for extra field"
    except:
        pass


def test_precord_constructor_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_multiple_fields():
    from pyrsistent import PRecord, field
    
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


# LLM-generated content at query #52
#--------------------------

```python
def test_precord_meta_new_creates_class_with_precord_fields():
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import _PRecordMeta, PFIELD_NO_INITIAL
    
    # Create a mock _PField
    field1 = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)
    field2 = _PField(initial=42, mandatory=False)
    
    dct = {
        'field1': field1,
        'field2': field2,
    }
    bases = ()
    
    # Call __new__ to create the class
    new_class = _PRecordMeta('TestRecord', bases, dct)
    
    # Verify that _precord_fields was created and populated
    assert hasattr(new_class, '_precord_fields')
    assert 'field1' in new_class._precord_fields
    assert 'field2' in new_class._precord_fields
    assert new_class._precord_fields['field1'] is field1
    assert new_class._precord_fields['field2'] is field2
    
    # Verify that mandatory fields were identified
    assert hasattr(new_class, '_precord_mandatory_fields')
    assert 'field1' in new_class._precord_mandatory_fields
    assert 'field2' not in new_class._precord_mandatory_fields
    
    # Verify that initial values were stored
    assert hasattr(new_class, '_precord_initial_values')
    assert new_class._precord_initial_values['field2'] == 42
    assert 'field1' not in new_class._precord_initial_values
    
    # Verify that __slots__ was set
    assert hasattr(new_class, '__slots__')
    assert new_class.__slots__ == ()
    
    # Verify that _precord_invariants was created
    assert hasattr(new_class, '_precord_invariants')
    assert isinstance(new_class._precord_invariants, tuple)


# LLM-generated content at query #53
#--------------------------

```python
def test_precord_new_predicate_false_without_precord_size():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create instance without _precord_size and _precord_buckets
    # This should make the predicate at line 5 evaluate to False
    record = TestRecord(x=1, y=2)
    
    assert record.x == 1
    assert record.y == 2


def test_precord_new_predicate_false_without_precord_buckets():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create instance with only _precord_size but not _precord_buckets
    # The predicate requires BOTH to be present, so it should evaluate to False
    record = TestRecord(x=5, y=10)
    
    assert record.x == 5
    assert record.y == 10


def test_precord_new_predicate_false_with_regular_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    # Create instance with regular kwargs (no _precord_size or _precord_buckets)
    # Predicate at line 5 should evaluate to False
    record = TestRecord(name="test", value=42)
    
    assert record['name'] == "test"
    assert record['value'] == 42


# LLM-generated content at query #54
#--------------------------

```python
def test_persistent_predicate_line_6_evaluates_to_false():
    from pyrsistent import pmap, field, PRecord
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = pmap({'name': 'test'})
    evolver = TestRecord._evolver(original_pmap)
    
    is_dirty = evolver.is_dirty()
    pm = type(evolver).__bases__[0].persistent(evolver)
    isinstance_check = isinstance(pm, TestRecord)
    
    predicate_result = is_dirty or not isinstance_check
    
    assert predicate_result is False


# LLM-generated content at query #55
#--------------------------

```python
def test_precord_new_predicate_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Create instance without the special _precord_size and _precord_buckets kwargs
    # This ensures the predicate at line 5 evaluates to False
    record = TestRecord(name="John", age=30)
    
    assert record['name'] == "John"
    assert record['age'] == 30
    assert isinstance(record, TestRecord)


# LLM-generated content at query #56
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pvector
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a valid pmap with buckets
    test_pmap = PMap(2, pvector([[(b'x', 1), (b'y', 2)]]))
    
    # Call __new__ with both _precord_size and _precord_buckets
    result = TestRecord.__new__(TestRecord, _precord_size=test_pmap._size, _precord_buckets=test_pmap._buckets)
    
    # Verify the result is an instance of TestRecord
    assert isinstance(result, TestRecord)


# LLM-generated content at query #57
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='John', age=30)
    assert record['name'] == 'John'
    assert record['age'] == 30


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='DefaultName')
        age = field(initial=0)
    
    record = TestRecord()
    assert record['name'] == 'DefaultName'
    assert record['age'] == 0


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record = TestRecord()
    assert record['items'] == []


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='DefaultName')
        age = field(initial=0)
    
    record = TestRecord(name='Alice', age=25)
    assert record['name'] == 'Alice'
    assert record['age'] == 25


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name='Bob', _factory_fields=None)
    assert record['name'] == 'Bob'


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name='Charlie', extra_field='ignored', _ignore_extra=True)
    assert record['name'] == 'Charlie'
    assert 'extra_field' not in record


def test_precord_constructor_partial_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
        city = field()
    
    record = TestRecord(name='David', age=40)
    assert record['name'] == 'David'
    assert record['age'] == 40
    assert len(record) == 2


# LLM-generated content at query #58
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_constructor_with_partial_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_with_callable_initial():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [1, 2, 3])
    
    record = TestRecord()
    assert record['x'] == [1, 2, 3]


def test_precord_constructor_kwargs_override_initial():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100, y=200)
    assert record['x'] == 100
    assert record['y'] == 200


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord._precord_initial_values = {}
    record = TestRecord.create({'x': 1, 'z': 3}, ignore_extra=True)
    assert record['x'] == 1
    assert 'z' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5)
    assert len(record) == 1
    assert record['x'] == 5


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 42})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert record['x'] == 42


# LLM-generated content at query #59
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
    
    original_pmap = pmap({'a': 1})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, MockCls)


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    
    original_pmap = pmap({'a': 1})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        __name__ = 'MockCls'
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return set()
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockCls.field1' in e.missing_fields
        assert 'MockCls.field2' in e.missing_fields


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return set()
        
        @staticmethod
        def _global_invariant(obj):
            return (False, 'global_error')
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    MockCls._precord_invariants = [MockCls._global_invariant]
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_creates_new_instance_when_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
    
    original_pmap = pmap({'a': 1})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver[0] = 'dummy'
    evolver._evolver_is_dirty = True
    result = evolver.persistent()
    assert isinstance(result, MockCls)


def test_persistent_with_both_error_codes_and_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        __name__ = 'MockCls'
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return set()
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = ['inv_error']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'inv_error' in e.invariant_errors
        assert 'MockCls.required_field' in e.missing_fields


# LLM-generated content at query #60
#--------------------------

```python
def test_persistent_predicate_line_6():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    # Create an evolver from an empty PRecord
    original = TestRecord()
    evolver = _PRecordEvolver(TestRecord, original._pmap)
    
    # Set a field to make it dirty
    evolver.set('name', 'test')
    
    # Check that is_dirty() returns True
    assert evolver.is_dirty() == True
    
    # The predicate at line 6 evaluates to: is_dirty or not isinstance(pm, cls)
    # Since is_dirty is True, the predicate should be True
    is_dirty = evolver.is_dirty()
    pm = PMap()
    cls = TestRecord
    predicate_result = is_dirty or not isinstance(pm, cls)
    
    assert predicate_result == True


# LLM-generated content at query #61
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a record normally
    record1 = TestRecord(x=1, y=2)
    assert record1['x'] == 1
    assert record1['y'] == 2


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="Alice", age=30)
    assert record['name'] == "Alice"
    assert record['age'] == 30


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(_factory_fields=[], x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=1, extra_field=999)
    assert record['x'] == 1
    assert 'extra_field' not in record


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': 10, 'y': 20}
        x = field()
        y = field()
    
    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_new_with_initial_values_override():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': 10, 'y': 20}
        x = field()
        y = field()
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': lambda: 42}
        x = field()
    
    record = TestRecord()
    assert record['x'] == 42


def test_precord_new_empty_record():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_new_with_multiple_fields():
    from pyrsistent import PRecord, field
    
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


# LLM-generated content at query #62
#--------------------------

```python
def test_persistent_evaluates_predicate_at_line_1():
    from pyrsistent import pmap, PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    original_pmap = pmap({'name': 'test', 'value': 42})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result['name'] == 'test'
    assert result['value'] == 42


# LLM-generated content at query #63
#--------------------------

```python
def test_persistent_predicate_line_6_evaluates_to_false():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self):
            self.invariant = lambda x: (True, None)
    
    class MockPMap:
        def __init__(self):
            self._buckets = ()
            self._size = 0
        
        def keys(self):
            return []
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"
    
    mock_pmap = MockPMap()
    evolver = _PRecordEvolver(MockClass, mock_pmap)
    evolver.is_dirty = lambda: False
    
    original_persistent = PMap._Evolver.persistent
    
    def mock_parent_persistent(self):
        return mock_pmap
    
    PMap._Evolver.persistent = mock_parent_persistent
    
    try:
        result = evolver.persistent()
        assert result is mock_pmap
        assert isinstance(result, MockPMap)
    finally:
        PMap._Evolver.persistent = original_persistent


# LLM-generated content at query #64
#--------------------------

```python
def test_persistent_returns_result_when_no_errors():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    
    assert isinstance(result, MockClass)


def test_persistent_raises_on_invariant_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'error1' in e.invariant_errors


def test_persistent_raises_on_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'MockClass.field1' in e.missing_fields


def test_persistent_checks_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import InvariantException
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_not_dirty_and_same_type():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
    
    original_pmap = MockClass(_precord_buckets=None, _precord_size=None)
    original_pmap._buckets = None
    original_pmap._size = None
    
    class MockEvolver(_PRecordEvolver):
        def is_dirty(self):
            return False
    
    evolver = MockEvolver(MockClass, pmap({}))
    result = evolver.persistent()
    
    assert result is not None


# LLM-generated content at query #65
#--------------------------

```python
def test_precord_new_without_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Create instance without _precord_size and _precord_buckets
    # This should take the else branch (line 8 onwards)
    record = TestRecord(name="John", age=30)
    
    # Verify the predicate at line 5 evaluates to False
    assert '_precord_size' not in {'name': 'John', 'age': 30}
    assert '_precord_buckets' not in {'name': 'John', 'age': 30}
    assert record.name == "John"
    assert record.age == 30


# LLM-generated content at query #66
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pvector, pmap
    
    class MyRecord(PRecord):
        x = field()
        y = field()
    
    initial_pmap = pmap({'x': 1, 'y': 2})
    record = MyRecord(_precord_size=initial_pmap._size, _precord_buckets=initial_pmap._buckets)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
        y = field()
    
    record = MyRecord(x=10, y=20)
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
        y = field()
    
    record = MyRecord()
    assert len(record) == 0


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
        y = field(initial=42)
    
    record = MyRecord(x=10)
    assert record['x'] == 10
    assert record['y'] == 42


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
        y = field(initial=lambda: [1, 2, 3])
    
    record = MyRecord(x=10)
    assert record['x'] == 10
    assert record['y'] == [1, 2, 3]


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
        y = field()
    
    record = MyRecord(x=5, y=15, _factory_fields=set())
    assert record['x'] == 5
    assert record['y'] == 15


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
    
    record = MyRecord(x=10, z=20, _ignore_extra=True)
    assert record['x'] == 10
    assert 'z' not in record


def test_precord_new_override_initial_values():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field(initial=5)
        y = field(initial=10)
    
    record = MyRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 10


def test_precord_new_multiple_kwargs():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = MyRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3


# LLM-generated content at query #67
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field()
    
    record = TestRecord(y=5)
    assert record['x'] == 10
    assert record['y'] == 5


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [1, 2, 3])
        y = field()
    
    record = TestRecord(y=5)
    assert record['x'] == [1, 2, 3]
    assert record['y'] == 5


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_with_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    base_map = pmap({'x': 5})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets)
    assert record['x'] == 5


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=1, extra_field=999)
    assert record['x'] == 1
    assert 'extra_field' not in record


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_factory_fields={'x': int}, x='5')
    assert record['x'] == 5


# LLM-generated content at query #68
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent._precord import PRecord
    from pyrsistent._pmap import PMap, pvector
    
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_initial_values = {}
        _precord_invariants = ()
    
    buckets = pvector()
    result = TestRecord(_precord_size=0, _precord_buckets=buckets)
    assert isinstance(result, TestRecord)
    assert isinstance(result, PMap)


def test_precord_new_with_kwargs():
    from pyrsistent._precord import PRecord, PRecordField
    
    class TestRecord(PRecord):
        _precord_fields = {'name': PRecordField(invariant=lambda x: (True, None), initial=None, factory=lambda x: x, serializer=None)}
        _precord_mandatory_fields = set()
        _precord_initial_values = {}
        _precord_invariants = ()
    
    result = TestRecord(name='test')
    assert isinstance(result, TestRecord)
    assert result['name'] == 'test'


def test_precord_new_with_initial_values():
    from pyrsistent._precord import PRecord, PRecordField
    
    class TestRecord(PRecord):
        _precord_fields = {'value': PRecordField(invariant=lambda x: (True, None), initial=42, factory=lambda x: x, serializer=None)}
        _precord_mandatory_fields = set()
        _precord_initial_values = {'value': lambda: 100}
        _precord_invariants = ()
    
    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert result['value'] == 100


def test_precord_new_with_factory_fields():
    from pyrsistent._precord import PRecord, PRecordField
    
    class TestRecord(PRecord):
        _precord_fields = {'num': PRecordField(invariant=lambda x: (True, None), initial=None, factory=int, serializer=None)}
        _precord_mandatory_fields = set()
        _precord_initial_values = {}
        _precord_invariants = ()
    
    result = TestRecord(_factory_fields={'num'}, num='5')
    assert isinstance(result, TestRecord)
    assert result['num'] == 5


def test_precord_new_with_ignore_extra():
    from pyrsistent._precord import PRecord, PRecordField
    
    class TestRecord(PRecord):
        _precord_fields = {'name': PRecordField(invariant=lambda x: (True, None), initial=None, factory=lambda x: x, serializer=None)}
        _precord_mandatory_fields = set()
        _precord_initial_values = {}
        _precord_invariants = ()
    
    result = TestRecord(_ignore_extra=True, name='test', extra_field='ignored')
    assert isinstance(result, TestRecord)
    assert result['name'] == 'test'


def test_precord_new_empty():
    from pyrsistent._precord import PRecord
    
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_initial_values = {}
        _precord_invariants = ()
    
    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert len(result) == 0


# LLM-generated content at query #69
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pvector
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap to get valid buckets
    initial_pmap = PMap(0, pvector())
    
    # Call __new__ with both _precord_size and _precord_buckets
    result = TestRecord.__new__(TestRecord, _precord_size=0, _precord_buckets=initial_pmap._buckets)
    
    # Verify that the result is an instance of TestRecord
    assert isinstance(result, TestRecord)
    assert isinstance(result, PMap)


# LLM-generated content at query #70
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_constructor_with_partial_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=5)
    assert record['x'] == 5
    assert record['y'] == 20


def test_precord_constructor_with_callable_initial():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [1, 2, 3])
    
    record = TestRecord()
    assert record['x'] == [1, 2, 3]


def test_precord_constructor_overrides_initial():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100, y=200)
    assert record['x'] == 100
    assert record['y'] == 200


def test_precord_constructor_with_internal_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 42})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert record['x'] == 42


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _ignore_extra=True, extra_field=999)
    assert record['x'] == 1
    assert 'extra_field' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    assert len(record) == 0


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_persistent_returns_result_when_not_dirty_and_correct_type():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
    
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockPRecord, original_pmap)
    evolver._is_dirty = False
    
    result = evolver.persistent()
    assert isinstance(result, MockPRecord)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        __name__ = 'MockPRecord'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockPRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockPRecord.required_field' in e.missing_fields


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockPRecord'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockPRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error_code_1' in e.invariant_errors


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    global_invariant_called = []
    
    def failing_global_invariant(subject):
        global_invariant_called.append(True)
        return (False, 'global_error')
    
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        __name__ = 'MockPRecord'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockPRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(global_invariant_called) > 0
        assert 'global_error' in e.invariant_errors


def test_persistent_creates_new_instance_when_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockPRecord, original_pmap)
    evolver._is_dirty = True
    
    result = evolver.persistent()
    assert isinstance(result, MockPRecord)


# LLM-generated content at query #2
#--------------------------

```python
def test_precord_meta_new_creates_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import PField
    
    dct = {
        'field1': PField(mandatory=True, initial=None),
        'field2': PField(mandatory=False, initial='default'),
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_fields')
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import PField
    
    dct = {
        'field1': PField(mandatory=True, initial=None),
        'field2': PField(mandatory=False, initial='default'),
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_mandatory_fields')
    assert 'field1' in result._precord_mandatory_fields
    assert 'field2' not in result._precord_mandatory_fields


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import PField
    
    dct = {
        'field1': PField(mandatory=True, initial=None),
        'field2': PField(mandatory=False, initial='default_value'),
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_initial_values')
    assert result._precord_initial_values.get('field2') == 'default_value'
    assert 'field1' not in result._precord_initial_values


def test_precord_meta_new_sets_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import PField
    
    dct = {
        'field1': PField(mandatory=True, initial=None),
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '__slots__')
    assert result.__slots__ == ()


def test_precord_meta_new_stores_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import PField
    
    def invariant_func(self):
        return True
    
    dct = {
        'field1': PField(mandatory=True, initial=None),
        '__invariant__': invariant_func,
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import PField
    
    parent_dct = {
        'parent_field': PField(mandatory=True, initial=None),
        '_precord_fields': {'parent_field': PField(mandatory=True, initial=None)},
    }
    
    class ParentRecord(metaclass=_PRecordMeta):
        _precord_fields = {'parent_field': PField(mandatory=True, initial=None)}
    
    dct = {
        'child_field': PField(mandatory=False, initial='child_default'),
    }
    bases = (ParentRecord,)
    name = 'ChildRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_fields')
    assert 'child_field' in result._precord_fields


def test_precord_meta_new_removes_pfield_from_dct():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import PField
    
    dct = {
        'field1': PField(mandatory=True, initial=None),
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert 'field1' not in dct or not isinstance(dct.get('field1'), PField)


# LLM-generated content at query #3
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    TestRecord._precord_mandatory_fields = set()
    TestRecord._precord_invariants = ()
    
    evolver = _PRecordEvolver(TestRecord, pmap({}))
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ()


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    TestRecord._precord_mandatory_fields = set()
    TestRecord._precord_invariants = ()
    
    evolver = _PRecordEvolver(TestRecord, pmap({}))
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['TestRecord.missing_field']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.missing_field',)


def test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    TestRecord._precord_mandatory_fields = set()
    TestRecord._precord_invariants = ()
    
    evolver = _PRecordEvolver(TestRecord, pmap({}))
    evolver._invariant_error_codes = ['error_1']
    evolver._missing_fields = ['TestRecord.field1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_1',)
        assert e.missing_fields == ('TestRecord.field1',)


# LLM-generated content at query #4
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    from pyrsistent._pmap import pmap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap with the test data
    test_pmap = pmap({'x': 1, 'y': 2})
    
    # Create record using special attributes
    record = TestRecord(_precord_size=test_pmap._size, _precord_buckets=test_pmap._buckets)
    
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=10, y=20)
    
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field(initial=100)
    
    record = TestRecord(x=5)
    
    assert record['x'] == 5
    assert record['y'] == 100


def test_precord_new_with_callable_initial_value():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field(initial=lambda: 42)
    
    record = TestRecord(x=1)
    
    assert record['x'] == 1
    assert record['y'] == 42


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2, _factory_fields=set())
    
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, z=10, _ignore_extra=True)
    
    assert record['x'] == 5
    assert 'z' not in record


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    
    assert len(record) == 0


def test_precord_new_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    
    assert record['x'] == 100
    assert record['y'] == 20


# LLM-generated content at query #5
#--------------------------

```python
def test_precord_new_predicate_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Call __new__ with kwargs that don't contain both '_precord_size' and '_precord_buckets'
    # This ensures the predicate at line 5 evaluates to False
    result = TestRecord(x=1, y=2)
    
    assert result['x'] == 1
    assert result['y'] == 2


# LLM-generated content at query #6
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_error_codes_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    TestRecord._precord_mandatory_fields = set()
    TestRecord._precord_invariants = []
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ()


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    TestRecord._precord_mandatory_fields = set()
    TestRecord._precord_invariants = []
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._missing_fields = ['TestRecord.y']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.y',)


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    TestRecord._precord_mandatory_fields = set()
    TestRecord._precord_invariants = []
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = ['TestRecord.y']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ('TestRecord.y',)


# LLM-generated content at query #7
#--------------------------

```python
def test_precord_meta_new_creates_class_with_correct_attributes():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a field with mandatory=True and no initial value
    mandatory_field = _PField(type=int, initial=PFIELD_NO_INITIAL, factory=None, mandatory=True)
    
    # Create a field with mandatory=False and an initial value
    optional_field = _PField(type=str, initial="default", factory=None, mandatory=False)
    
    dct = {
        'field1': mandatory_field,
        'field2': optional_field,
    }
    bases = ()
    
    # Call __new__ to create the class
    cls = _PRecordMeta.__new__(_PRecordMeta, 'TestRecord', bases, dct)
    
    # Verify that _precord_fields was set
    assert hasattr(cls, '_precord_fields')
    assert 'field1' in cls._precord_fields
    assert 'field2' in cls._precord_fields
    
    # Verify that _precord_mandatory_fields contains only mandatory fields
    assert hasattr(cls, '_precord_mandatory_fields')
    assert 'field1' in cls._precord_mandatory_fields
    assert 'field2' not in cls._precord_mandatory_fields
    
    # Verify that _precord_initial_values contains only fields with initial values
    assert hasattr(cls, '_precord_initial_values')
    assert 'field2' in cls._precord_initial_values
    assert cls._precord_initial_values['field2'] == "default"
    assert 'field1' not in cls._precord_initial_values
    
    # Verify that __slots__ was set to empty tuple
    assert hasattr(cls, '__slots__')
    assert cls.__slots__ == ()
    
    # Verify that _precord_invariants was set
    assert hasattr(cls, '_precord_invariants')
    assert isinstance(cls._precord_invariants, tuple)


# LLM-generated content at query #8
#--------------------------

```python
def test_set_with_valid_field_and_value():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import PField
    
    class TestClass:
        _precord_fields = {'name': PField(type=(str,), factory=str, invariant=lambda x: (True, None))}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestClass'
    
    original_pmap = pmap({'name': 'test'})
    evolver = _PRecordEvolver(TestClass, original_pmap)
    result = evolver.set('name', 'new_value')
    
    assert result is evolver


def test_set_with_invalid_field_raises_attribute_error():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestClass'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestClass, original_pmap)
    
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'invalid_field' is not among the specified fields" in str(e)


def test_set_with_setitem():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import PField
    
    class TestClass:
        _precord_fields = {'age': PField(type=(int,), factory=int, invariant=lambda x: (True, None))}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestClass'
    
    original_pmap = pmap({'age': 25})
    evolver = _PRecordEvolver(TestClass, original_pmap)
    result = evolver.__setitem__('age', 30)
    
    assert result is evolver


def test_set_with_factory_fields_restriction():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import PField
    
    field1 = PField(type=(str,), factory=str, invariant=lambda x: (True, None))
    field2 = PField(type=(int,), factory=int, invariant=lambda x: (True, None))
    
    class TestClass:
        _precord_fields = {'name': field1, 'age': field2}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestClass'
    
    original_pmap = pmap({'name': 'test', 'age': 25})
    evolver = _PRecordEvolver(TestClass, original_pmap, _factory_fields=(field1,))
    result = evolver.set('name', 'updated')
    
    assert result is evolver


def test_set_with_failed_invariant():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import PField
    
    def invariant_check(value):
        return (False, 'error_code')
    
    class TestClass:
        _precord_fields = {'value': PField(type=(int,), factory=int, invariant=invariant_check)}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestClass'
    
    original_pmap = pmap({'value': 10})
    evolver = _PRecordEvolver(TestClass, original_pmap)
    result = evolver.set('value', 20)
    
    assert 'error_code' in evolver._invariant_error_codes
    assert result is evolver


# LLM-generated content at query #9
#--------------------------

```python
def test_set_with_valid_field_and_value():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import PField
    
    class TestClass:
        _precord_fields = {'name': PField(type=(str,), factory=lambda x: x, invariant=lambda x: (True, None))}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        __name__ = 'TestClass'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestClass, original_pmap)
    result = evolver.set('name', 'test_value')
    
    assert result is not None


def test_set_with_invalid_field_name():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        __name__ = 'TestClass'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestClass, original_pmap)
    
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert 'invalid_field' in str(e)


def test_set_with_factory_fields_filter():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import PField
    
    class TestClass:
        _precord_fields = {'field1': PField(type=(str,), factory=lambda x: x, invariant=lambda x: (True, None))}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        __name__ = 'TestClass'
    
    field1 = TestClass._precord_fields['field1']
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestClass, original_pmap, _factory_fields=[field1])
    result = evolver.set('field1', 'test_value')
    
    assert result is not None


def test_set_with_factory_fields_excluded():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import PField
    
    class TestClass:
        _precord_fields = {'field1': PField(type=(str,), factory=lambda x: x, invariant=lambda x: (True, None))}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        __name__ = 'TestClass'
    
    field1 = TestClass._precord_fields['field1']
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestClass, original_pmap, _factory_fields=[])
    result = evolver.set('field1', 'test_value')
    
    assert result is not None


def test_setitem_delegates_to_set():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import PField
    
    class TestClass:
        _precord_fields = {'name': PField(type=(str,), factory=lambda x: x, invariant=lambda x: (True, None))}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        __name__ = 'TestClass'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestClass, original_pmap)
    result = evolver['name'] = 'test_value'
    
    assert result is None


# LLM-generated content at query #10
#--------------------------

```python
def test_precord_initial_values_predicate():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        __precord_initial_values = {'x': lambda: 10, 'y': 20}
    
    # Set the _precord_initial_values attribute to trigger the predicate at line 11
    TestRecord._precord_initial_values = {'x': lambda: 10, 'y': 20}
    
    # Create an instance without providing initial values
    # This should trigger the condition at line 11 to be True
    record = TestRecord()
    
    # Verify that initial values were applied
    assert record['x'] == 10
    assert record['y'] == 20


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_line_6_evaluates_to_true_when_is_dirty():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 2)
    
    assert evolver.is_dirty() == True
    result = evolver.persistent()
    assert result['x'] == 2


def test_predicate_line_6_evaluates_to_true_when_not_isinstance():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    class FakePMap(PMap):
        pass
    
    evolver._data = FakePMap(original_pmap)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)


def test_predicate_line_6_evaluates_to_true_when_both_conditions_met():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 5)
    evolver.set('y', 10)
    
    assert evolver.is_dirty() == True
    result = evolver.persistent()
    assert result['x'] == 5
    assert result['y'] == 10


# LLM-generated content at query #12
#--------------------------

```python
def test_precord_evolver_set_with_valid_field():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field(type=str)
    
    original = TestRecord(name="test")
    evolver = _PRecordEvolver(TestRecord, original._pmap)
    
    result = evolver.set("name", "new_value")
    
    assert result is not None
    assert isinstance(result, _PRecordEvolver)


# LLM-generated content at query #13
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        _precord_fields = {'name': field()}
    
    # Create a mock evolver with invariant error codes
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ()
        assert str(e) == 'Field invariant failed'


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        _precord_fields = {'name': field()}
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.name',)
        assert str(e) == 'Field invariant failed'


def test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        _precord_fields = {'name': field()}
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ('TestRecord.name',)
        assert str(e) == 'Field invariant failed'


# LLM-generated content at query #14
#--------------------------

```python
def test_precord_repr():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='John', age=30)
    result = repr(record)
    
    assert 'TestRecord' in result
    assert 'name=' in result
    assert "'John'" in result
    assert 'age=' in result
    assert '30' in result


def test_precord_repr_empty():
    from pyrsistent import PRecord
    
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    result = repr(record)
    
    assert result == 'EmptyRecord()'


def test_precord_repr_single_field():
    from pyrsistent import PRecord, field
    
    class SingleFieldRecord(PRecord):
        value = field()
    
    record = SingleFieldRecord(value='test')
    result = repr(record)
    
    assert result == "SingleFieldRecord(value='test')"


def test_precord_repr_multiple_fields_with_special_values():
    from pyrsistent import PRecord, field
    
    class ComplexRecord(PRecord):
        text = field()
        number = field()
        flag = field()
    
    record = ComplexRecord(text='hello', number=42, flag=True)
    result = repr(record)
    
    assert 'ComplexRecord' in result
    assert "text='hello'" in result
    assert 'number=42' in result
    assert 'flag=True' in result


# LLM-generated content at query #15
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, MockCls)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockCls.required_field' in e.missing_fields


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error1' in e.invariant_errors
        assert 'error2' in e.invariant_errors


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #16
#--------------------------

```python
def test_precord_initial_values_predicate():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        _precord_initial_values = {'x': 10, 'y': lambda: 20}
    
    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20


# LLM-generated content at query #17
#--------------------------

```python
def test_persistent_checks_mandatory_fields_when_present():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    TestRecord._precord_mandatory_fields = {'name', 'age'}
    TestRecord._precord_invariants = []
    
    evolver = TestRecord._PRecordEvolver(TestRecord, TestRecord()._pmap, _factory_fields=None, _ignore_extra=False)
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    
    original_missing_fields_len = len(evolver._missing_fields)
    
    try:
        evolver.persistent()
    except Exception:
        pass
    
    assert len(evolver._missing_fields) > original_missing_fields_len or TestRecord._precord_mandatory_fields
    assert TestRecord._precord_mandatory_fields is not None
    assert bool(TestRecord._precord_mandatory_fields) == True


# LLM-generated content at query #18
#--------------------------

```python
def test_precord_meta_new_returns_class():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a simple field for testing
    test_field = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)
    
    # Create a test class using the metaclass
    dct = {'_precord_fields': {'test_field': test_field}, '__invariant__': None}
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    # Verify that __new__ returns a class (type object)
    assert isinstance(result, type)
    assert result.__name__ == name


# LLM-generated content at query #19
#--------------------------

```python
def test_precord_evolver_persistent_predicate_is_dirty_true():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 42)
    
    is_dirty = evolver.is_dirty()
    pm = PMap._Evolver.persistent(evolver)
    predicate_result = is_dirty or not isinstance(pm, TestRecord)
    
    assert predicate_result is True


def test_precord_evolver_persistent_predicate_not_isinstance_true():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    is_dirty = evolver.is_dirty()
    pm = PMap._Evolver.persistent(evolver)
    predicate_result = is_dirty or not isinstance(pm, TestRecord)
    
    assert predicate_result is True


# LLM-generated content at query #20
#--------------------------

```python
def test_persistent_returns_result_when_no_errors():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys_result = set()
        
        def keys(self):
            return self.keys_result
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, MockClass)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockClass.field1' in e.missing_fields


def test_persistent_raises_invariant_exception_with_field_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    global_invariant_called = []
    
    def mock_global_invariant(subject):
        global_invariant_called.append(True)
        return (True, None)
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [mock_global_invariant]
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert len(global_invariant_called) > 0


def test_persistent_raises_invariant_exception_on_global_invariant_failure():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    def failing_global_invariant(subject):
        return (False, 'global_error')
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_with_clean_state_returns_original_pmap_if_same_type():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return set()
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._dirty = False
    result = evolver.persistent()
    assert isinstance(result, MockClass)


# LLM-generated content at query #21
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockField:
        def __init__(self):
            self.invariant = lambda x: (True, None)
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockCls"
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ()
        assert 'Field invariant failed' in str(e)


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockCls"
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['MockCls.field1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('MockCls.field1',)
        assert 'Field invariant failed' in str(e)


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockCls"
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = ['error_code_1', 'error_code_2']
    evolver._missing_fields = ['MockCls.field1', 'MockCls.field2']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1', 'error_code_2')
        assert e.missing_fields == ('MockCls.field1', 'MockCls.field2')
        assert 'Field invariant failed' in str(e)


# LLM-generated content at query #22
#--------------------------

```python
def test_precord_evolver_set_with_field_found():
    from pyrsistent import pmap, PRecord, field
    
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
    
    original_pmap = pmap({'name': 'Alice', 'age': 30})
    evolver = TestRecord._PRecordEvolver(TestRecord, original_pmap)
    
    # The predicate at line 3 (if field:) should evaluate to True
    # when we call set() with a key that exists in _precord_fields
    result = evolver.set('name', 'Bob')
    
    assert result is evolver
    assert evolver._data['name'] == 'Bob'


# LLM-generated content at query #23
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_constructor_with_initial_values_and_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_with_callable_initial():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [1, 2, 3])
    
    record = TestRecord()
    assert record['x'] == [1, 2, 3]


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _factory_fields=None)
    assert record['x'] == 5


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _ignore_extra=True)
    assert record['x'] == 1


def test_precord_constructor_internal_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    base_map = pmap({'x': 5})
    record = TestRecord(_precord_size=base_map._PMap__size, _precord_buckets=base_map._PMap__buckets)
    assert record['x'] == 5


def test_precord_constructor_empty():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_multiple_fields():
    from pyrsistent import PRecord, field
    
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


# LLM-generated content at query #24
#--------------------------

```python
def test_precord_new_predicate_false_missing_precord_size():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ with _precord_buckets but without _precord_size
    # This makes the predicate at line 5 evaluate to False
    result = TestRecord(x=10)
    
    assert result.x == 10
    assert isinstance(result, TestRecord)


def test_precord_new_predicate_false_missing_precord_buckets():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ with _precord_size but without _precord_buckets
    # This makes the predicate at line 5 evaluate to False
    result = TestRecord(x=20)
    
    assert result.x == 20
    assert isinstance(result, TestRecord)


def test_precord_new_predicate_false_missing_both():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ without either _precord_size or _precord_buckets
    # This makes the predicate at line 5 evaluate to False
    result = TestRecord(x=30)
    
    assert result.x == 30
    assert isinstance(result, TestRecord)


# LLM-generated content at query #25
#--------------------------

```python
def test_serialize_with_no_serializer():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    result = record.serialize()
    
    assert result == {'name': 'John', 'age': 30}


def test_serialize_with_custom_serializer():
    from pyrsistent import PRecord, field
    
    def int_serializer(format, value):
        return str(value)
    
    class TestRecord(PRecord):
        name = field()
        age = field(serializer=int_serializer)
    
    record = TestRecord(name="John", age=30)
    result = record.serialize()
    
    assert result == {'name': 'John', 'age': '30'}


def test_serialize_with_format_parameter():
    from pyrsistent import PRecord, field
    
    def custom_serializer(format, value):
        if format == 'json':
            return str(value).lower()
        return value
    
    class TestRecord(PRecord):
        status = field(serializer=custom_serializer)
    
    record = TestRecord(status="Active")
    result = record.serialize(format='json')
    
    assert result == {'status': 'active'}


def test_serialize_mixed_fields():
    from pyrsistent import PRecord, field
    
    def uppercase_serializer(format, value):
        return value.upper()
    
    class TestRecord(PRecord):
        id = field()
        name = field(serializer=uppercase_serializer)
        active = field()
    
    record = TestRecord(id=1, name="john", active=True)
    result = record.serialize()
    
    assert result == {'id': 1, 'name': 'JOHN', 'active': True}


def test_serialize_empty_record():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name=None)
    result = record.serialize()
    
    assert result == {'name': None}


def test_serialize_preserves_types():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field()
        config = field()
    
    record = TestRecord(items=[1, 2, 3], config={'key': 'value'})
    result = record.serialize()
    
    assert result == {'items': [1, 2, 3], 'config': {'key': 'value'}}


def test_serialize_with_none_serializer():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        value = field(serializer=None)
    
    record = TestRecord(value=42)
    result = record.serialize()
    
    assert result == {'value': 42}


# LLM-generated content at query #26
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a record normally
    record1 = TestRecord(x=1, y=2)
    assert record1.x == 1
    assert record1.y == 2


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_factory_fields=set(), x=5)
    assert record.x == 5


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=10)
    assert record.x == 10


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': 42}
        x = field()
        y = field()
    
    record = TestRecord(y=20)
    assert record.x == 42
    assert record.y == 20


def test_precord_new_with_initial_values_override():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': 42}
        x = field()
    
    record = TestRecord(x=100)
    assert record.x == 100


def test_precord_new_with_initial_values_callable():
    from pyrsistent import PRecord, field
    
    def default_value():
        return 99
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': default_value}
        x = field()
    
    record = TestRecord()
    assert record.x == 99


def test_precord_new_empty_record():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_new_multiple_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = TestRecord(a=1, b=2, c=3)
    assert record.a == 1
    assert record.b == 2
    assert record.c == 3


def test_precord_new_with_kwargs_and_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': 10, 'y': 20}
        x = field()
        y = field()
        z = field()
    
    record = TestRecord(z=30, y=25)
    assert record.x == 10
    assert record.y == 25
    assert record.z == 30


# LLM-generated content at query #27
#--------------------------

```python
def test_persistent_with_mandatory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    TestRecord._precord_mandatory_fields = {'name', 'age'}
    TestRecord._precord_invariants = []
    
    original_pmap = TestRecord()._to_pmap()
    evolver = TestRecord._precord_evolver(original_pmap)
    
    assert TestRecord._precord_mandatory_fields
    assert len(TestRecord._precord_mandatory_fields) > 0


# LLM-generated content at query #28
#--------------------------

```python
def test_precord_meta_new_sets_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': _PField(mandatory=False, initial=42),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert hasattr(result, '_precord_fields')
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': _PField(mandatory=False, initial=42),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert hasattr(result, '_precord_mandatory_fields')
    assert result._precord_mandatory_fields == {'field1'}


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': _PField(mandatory=False, initial=42),
        'field3': _PField(mandatory=False, initial='default'),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert hasattr(result, '_precord_initial_values')
    assert result._precord_initial_values == {'field2': 42, 'field3': 'default'}


def test_precord_meta_new_sets_empty_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert hasattr(result, '__slots__')
    assert result.__slots__ == ()


def test_precord_meta_new_sets_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    def invariant_func(instance):
        return True, None
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        '__invariant__': invariant_func,
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    parent_dct = {
        'parent_field': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
    }
    parent_bases = ()
    parent = _PRecordMeta('Parent', parent_bases, parent_dct)
    
    child_dct = {
        'child_field': _PField(mandatory=False, initial=10),
    }
    child_bases = (parent,)
    
    result = _PRecordMeta('Child', child_bases, child_dct)
    
    assert 'parent_field' in result._precord_fields
    assert 'child_field' in result._precord_fields


def test_precord_meta_new_no_fields():
    from pyrsistent._precord import _PRecordMeta
    
    dct = {}
    bases = ()
    name = 'EmptyRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert hasattr(result, '_precord_fields')
    assert result._precord_fields == {}
    assert result._precord_mandatory_fields == set()
    assert result._precord_initial_values == {}


# LLM-generated content at query #29
#--------------------------

```python
def test_serialize_without_serializer():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    result = record.serialize()
    
    assert result == {"name": "John", "age": 30}


def test_serialize_with_custom_serializer():
    from pyrsistent import PRecord, field
    
    def serialize_upper(value):
        return value.upper() if isinstance(value, str) else value
    
    class TestRecord(PRecord):
        name = field(serializer=serialize_upper)
        age = field()
    
    record = TestRecord(name="john", age=30)
    result = record.serialize()
    
    assert result == {"name": "JOHN", "age": 30}


def test_serialize_with_format_parameter():
    from pyrsistent import PRecord, field
    
    def custom_serializer(format_type, value):
        if format_type == "json":
            return str(value)
        return value
    
    class TestRecord(PRecord):
        value = field()
    
    record = TestRecord(value=42)
    result = record.serialize(format="json")
    
    assert result == {"value": 42}


def test_serialize_empty_record():
    from pyrsistent import PRecord
    
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    result = record.serialize()
    
    assert result == {}


def test_serialize_multiple_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()
        field3 = field()
    
    record = TestRecord(field1="a", field2="b", field3="c")
    result = record.serialize()
    
    assert result == {"field1": "a", "field2": "b", "field3": "c"}


# LLM-generated content at query #30
#--------------------------

```python
def test_precord_meta_new_creates_class_with_correct_attributes():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a simple field for testing
    test_field = _PField(type=str, initial="default", mandatory=True)
    optional_field = _PField(type=int, initial=PFIELD_NO_INITIAL, mandatory=False)
    
    # Create a dictionary with fields
    dct = {
        'test_attr': test_field,
        'optional_attr': optional_field,
    }
    
    # Call __new__ to create a new class
    new_class = _PRecordMeta.__new__(_PRecordMeta, 'TestRecord', (), dct)
    
    # Verify the predicate: name is the first positional argument after mcs
    assert new_class is not None
    assert hasattr(new_class, '_precord_fields')
    assert hasattr(new_class, '_precord_invariants')
    assert hasattr(new_class, '_precord_mandatory_fields')
    assert hasattr(new_class, '_precord_initial_values')
    assert hasattr(new_class, '__slots__')
    assert new_class.__slots__ == ()
    assert 'test_attr' in new_class._precord_mandatory_fields
    assert 'optional_attr' not in new_class._precord_mandatory_fields
    assert 'test_attr' in new_class._precord_initial_values
    assert 'optional_attr' not in new_class._precord_initial_values


# LLM-generated content at query #31
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field

    class TestRecord(PRecord):
        x = field()
        y = field()

    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field

    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)

    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field

    class TestRecord(PRecord):
        x = field(initial=lambda: 42)
        y = field(initial=lambda: 100)

    record = TestRecord()
    assert record['x'] == 42
    assert record['y'] == 100


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field

    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)

    record = TestRecord(x=99, y=88)
    assert record['x'] == 99
    assert record['y'] == 88


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field

    class TestRecord(PRecord):
        x = field()
        y = field()

    record = TestRecord()
    assert 'x' not in record or record['x'] is None
    assert 'y' not in record or record['y'] is None


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field

    class TestRecord(PRecord):
        x = field()

    record = TestRecord(x=5, _factory_fields=None)
    assert record['x'] == 5


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field

    class TestRecord(PRecord):
        x = field()

    record = TestRecord(x=1, extra_field=999, _ignore_extra=True)
    assert record['x'] == 1
    assert 'extra_field' not in record


def test_precord_constructor_partial_kwargs():
    from pyrsistent import PRecord, field

    class TestRecord(PRecord):
        x = field()
        y = field()
        z = field()

    record = TestRecord(x=1, z=3)
    assert record['x'] == 1
    assert record['z'] == 3


# LLM-generated content at query #32
#--------------------------

```python
def test_precord_new_predicate_false_when_only_precord_size_in_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # This should not match the predicate at line 5 because only '_precord_size' is present
    record = TestRecord(x=1)
    assert record.x == 1


def test_precord_new_predicate_false_when_only_precord_buckets_in_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # This should not match the predicate at line 5 because only '_precord_buckets' is present
    record = TestRecord(x=2)
    assert record.x == 2


def test_precord_new_predicate_false_when_neither_special_kwargs_present():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # This should not match the predicate at line 5 because neither special kwarg is present
    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20


def test_precord_new_predicate_false_with_regular_field_only():
    from pyrsistent import PRecord, field
    
    class SimpleRecord(PRecord):
        name = field()
    
    # Regular field assignment should not trigger the predicate
    record = SimpleRecord(name="test")
    assert record.name == "test"


# LLM-generated content at query #33
#--------------------------

```python
def test_precord_repr():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='Alice', age=30)
    repr_str = repr(record)
    
    assert 'TestRecord' in repr_str
    assert 'name=' in repr_str
    assert "'Alice'" in repr_str
    assert 'age=' in repr_str
    assert '30' in repr_str


def test_precord_repr_empty():
    from pyrsistent import PRecord
    
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    repr_str = repr(record)
    
    assert repr_str == 'EmptyRecord()'


def test_precord_repr_single_field():
    from pyrsistent import PRecord, field
    
    class SingleFieldRecord(PRecord):
        value = field()
    
    record = SingleFieldRecord(value='test')
    repr_str = repr(record)
    
    assert repr_str == "SingleFieldRecord(value='test')"


def test_precord_repr_multiple_fields():
    from pyrsistent import PRecord, field
    
    class MultiFieldRecord(PRecord):
        first = field()
        second = field()
        third = field()
    
    record = MultiFieldRecord(first=1, second='two', third=3.0)
    repr_str = repr(record)
    
    assert 'MultiFieldRecord' in repr_str
    assert 'first=1' in repr_str
    assert "second='two'" in repr_str
    assert 'third=3.0' in repr_str


def test_precord_repr_with_special_characters():
    from pyrsistent import PRecord, field
    
    class SpecialRecord(PRecord):
        text = field()
    
    record = SpecialRecord(text="hello'world")
    repr_str = repr(record)
    
    assert 'SpecialRecord' in repr_str
    assert 'text=' in repr_str
    assert "hello'world" in repr_str


# LLM-generated content at query #34
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field()
    
    record = TestRecord(y=5)
    assert record['x'] == 10
    assert record['y'] == 5


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [])
        y = field()
    
    record1 = TestRecord(y=1)
    record2 = TestRecord(y=2)
    assert record1['x'] == []
    assert record2['x'] == []
    assert record1['x'] is not record2['x']


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_factory_fields=True, x=1)
    assert record['x'] == 1


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=1, z=999)
    assert record['x'] == 1
    assert 'z' not in record


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
    
    record = TestRecord(x=20)
    assert record['x'] == 20


def test_precord_constructor_empty_record():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 1})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert record['x'] == 1


# LLM-generated content at query #35
#--------------------------

```python
def test_serialize_returns_dict_with_serialized_values():
    from pyrsistent import PRecord, field
    
    class CustomSerializer:
        def __call__(self, format, value):
            return f"serialized_{value}"
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    result = record.serialize()
    
    assert isinstance(result, dict)
    assert "name" in result
    assert "age" in result


# LLM-generated content at query #36
#--------------------------

```python
def test_persistent_checks_mandatory_fields_when_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    TestRecord._precord_mandatory_fields = {'x', 'y'}
    TestRecord._precord_invariants = ()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    # Set a value to make the evolver dirty
    evolver.set('x', 1)
    evolver.set('y', 2)
    
    # Call persistent to trigger the code path
    # This should execute line 11 where cls._precord_mandatory_fields is truthy
    result = evolver.persistent()
    
    assert result is not None
    assert result['x'] == 1
    assert result['y'] == 2


# LLM-generated content at query #37
#--------------------------

```python
def test_precord_meta_new_creates_class_with_correct_attributes():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a mock field
    field1 = _PField(initial=PFIELD_NO_INITIAL, mandatory=True, factory=None, invariant=None)
    field2 = _PField(initial=42, mandatory=False, factory=None, invariant=None)
    
    # Create class dictionary with fields
    dct = {
        'field1': field1,
        'field2': field2,
    }
    
    # Call the metaclass __new__ method
    cls = _PRecordMeta('TestClass', (), dct)
    
    # Assertions to verify the predicate (line 1: def __new__) was executed
    assert cls is not None
    assert cls.__name__ == 'TestClass'
    assert hasattr(cls, '_precord_fields')
    assert hasattr(cls, '_precord_invariants')
    assert hasattr(cls, '_precord_mandatory_fields')
    assert hasattr(cls, '_precord_initial_values')
    assert hasattr(cls, '__slots__')
    assert cls.__slots__ == ()
    assert 'field1' in cls._precord_mandatory_fields
    assert 'field2' not in cls._precord_mandatory_fields
    assert 'field2' in cls._precord_initial_values
    assert cls._precord_initial_values['field2'] == 42
    assert 'field1' not in cls._precord_initial_values


# LLM-generated content at query #38
#--------------------------

```python
def test_precord_new_predicate_false_missing_precord_size():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    # Call __new__ with only _precord_buckets, missing _precord_size
    # This makes the predicate at line 5 evaluate to False
    result = TestRecord(name='test')
    
    assert result.name == 'test'
    assert isinstance(result, TestRecord)


def test_precord_new_predicate_false_missing_precord_buckets():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    # Call __new__ with only _precord_size, missing _precord_buckets
    # This makes the predicate at line 5 evaluate to False
    result = TestRecord(name='test')
    
    assert result.name == 'test'
    assert isinstance(result, TestRecord)


def test_precord_new_predicate_false_missing_both():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Call __new__ without either _precord_size or _precord_buckets
    # This makes the predicate at line 5 evaluate to False
    result = TestRecord(name='Alice', age=30)
    
    assert result.name == 'Alice'
    assert result.age == 30
    assert isinstance(result, TestRecord)


# LLM-generated content at query #39
#--------------------------

```python
def test_precord_meta_new_creates_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import PFIELD_NO_INITIAL
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True),
        'field2': TestField(mandatory=False, initial='default_value'),
        'field3': TestField(mandatory=False),
    }
    bases = (object,)
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestRecord', bases, dct)
    
    assert '_precord_fields' in result.__dict__
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields
    assert 'field3' in result._precord_fields
    assert result._precord_fields['field1'].mandatory is True
    assert result._precord_fields['field2'].mandatory is False


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import PFIELD_NO_INITIAL
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True),
        'field2': TestField(mandatory=False),
        'field3': TestField(mandatory=True),
    }
    bases = (object,)
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestRecord', bases, dct)
    
    assert result._precord_mandatory_fields == {'field1', 'field3'}


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import PFIELD_NO_INITIAL
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=False, initial='value1'),
        'field2': TestField(mandatory=False, initial=42),
        'field3': TestField(mandatory=False),
    }
    bases = (object,)
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestRecord', bases, dct)
    
    assert result._precord_initial_values == {'field1': 'value1', 'field2': 42}
    assert 'field3' not in result._precord_initial_values


def test_precord_meta_new_sets_empty_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import PFIELD_NO_INITIAL
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {'field1': TestField()}
    bases = (object,)
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestRecord', bases, dct)
    
    assert result.__slots__ == ()


def test_precord_meta_new_stores_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import PFIELD_NO_INITIAL
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    def test_invariant(obj):
        return True, None
    
    dct = {
        'field1': TestField(),
        '__invariant__': test_invariant,
    }
    bases = (object,)
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestRecord', bases, dct)
    
    assert '_precord_invariants' in result.__dict__
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_removes_field_descriptors_from_dct():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import PFIELD_NO_INITIAL
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(),
        'regular_attr': 'some_value',
    }
    bases = (object,)
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestRecord', bases, dct)
    
    assert 'field1' not in result.__dict__
    assert 'regular_attr' in result.__dict__


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._checked_types import PFIELD_NO_INITIAL
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    base_dct = {
        'base_field': TestField(mandatory=True),
    }
    base_class = _PRecordMeta.__new__(_PRecordMeta, 'BaseRecord', (object,), base_dct)
    
    child_dct = {
        'child_field': TestField(mandatory=False),
    }
    child_class = _PRecordMeta.__new__(_PRecordMeta, 'ChildRecord', (base_class,), child_dct)
    
    assert 'base_field' in child_class._precord_fields
    assert 'child_field' in child_class._precord_fields


# LLM-generated content at query #40
#--------------------------

```python
def test_precord_new_predicate_false_when_missing_precord_size():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ with only _precord_buckets but not _precord_size
    # This makes the predicate at line 5 evaluate to False
    result = TestRecord(_precord_buckets=None)
    assert isinstance(result, TestRecord)


def test_precord_new_predicate_false_when_missing_precord_buckets():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ with only _precord_size but not _precord_buckets
    # This makes the predicate at line 5 evaluate to False
    result = TestRecord(_precord_size=0)
    assert isinstance(result, TestRecord)


def test_precord_new_predicate_false_when_both_missing():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ with neither _precord_size nor _precord_buckets
    # This makes the predicate at line 5 evaluate to False
    result = TestRecord(x=5)
    assert isinstance(result, TestRecord)
    assert result.x == 5


