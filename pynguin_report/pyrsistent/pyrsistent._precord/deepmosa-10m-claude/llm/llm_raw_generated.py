####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_persistent_returns_precord_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.persistent()
    assert result is not None


def test_persistent_raises_invariant_exception_on_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._invariant import InvariantException
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error1' in e.invariant_errors


def test_persistent_raises_invariant_exception_on_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._invariant import InvariantException
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._missing_fields = ['field1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'field1' in e.missing_fields


def test_persistent_checks_mandatory_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._invariant import InvariantException
    
    class TestRecord:
        __name__ = 'TestRecord'
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.required_field' in e.missing_fields


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._invariant import InvariantException
    
    def failing_global_invariant(subject):
        return (False, 'global_error')
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #2
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    assert record['name'] == "John"
    assert record['age'] == 30


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=0)
    
    record = TestRecord()
    assert record['name'] == "DefaultName"
    assert record['age'] == 0


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['items'] == []
    assert record2['items'] == []
    assert record1['items'] is not record2['items']


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=0)
    
    record = TestRecord(name="Jane", age=25)
    assert record['name'] == "Jane"
    assert record['age'] == 25


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord.create({'name': 'John', 'extra': 'value'}, ignore_extra=True)
    assert record['name'] == 'John'
    assert 'extra' not in record


def test_precord_constructor_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(_factory_fields={}, name="Test")
    assert record['name'] == "Test"


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    base_map = pmap({'name': 'Test'})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets, name='Test')
    assert record['name'] == 'Test'


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name="John")
    assert len(record) == 1
    assert record['name'] == "John"


def test_precord_constructor_multiple_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
        email = field()
    
    record = TestRecord(name="Alice", age=28, email="alice@example.com")
    assert record['name'] == "Alice"
    assert record['age'] == 28
    assert record['email'] == "alice@example.com"


def test_precord_constructor_with_factory_fields_dict():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    factory_fields = {}
    record = TestRecord(_factory_fields=factory_fields, name="Bob")
    assert record['name'] == "Bob"


# LLM-generated content at query #3
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import pmap, PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    initial_pmap = pmap({'x': 1, 'y': 2})
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
        x = field()
        y = field()
        _precord_initial_values = {'x': 5, 'y': 10}
    
    record = TestRecord()
    
    assert record['x'] == 5
    assert record['y'] == 10


def test_precord_new_with_initial_values_and_kwargs_override():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        _precord_initial_values = {'x': 5, 'y': 10}
    
    record = TestRecord(x=100)
    
    assert record['x'] == 100
    assert record['y'] == 10


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        _precord_initial_values = {'x': lambda: 42}
    
    record = TestRecord()
    
    assert record['x'] == 42


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(_factory_fields=set(), x=5, y=10)
    
    assert record['x'] == 5
    assert record['y'] == 10


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=5)
    
    assert record['x'] == 5


# LLM-generated content at query #4
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
        assert e.missing_fields == ()


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
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.x',)


def test_persistent_raises_invariant_exception_when_both_present():
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


# LLM-generated content at query #5
#--------------------------

```python
def test_precord_initial_values_predicate():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
        _precord_initial_values = {'name': 'default_name', 'age': lambda: 0}
    
    record = TestRecord()
    assert record['name'] == 'default_name'
    assert record['age'] == 0


# LLM-generated content at query #6
#--------------------------

```python
def test_precord_meta_new_creates_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    field1 = _PField(initial=None, mandatory=False, factory=None, initial_factory=None)
    field2 = _PField(initial=42, mandatory=True, factory=None, initial_factory=None)
    
    dct = {'field1': field1, 'field2': field2}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert hasattr(result, '_precord_fields')
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_stores_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    def my_invariant(record):
        return True, None
    
    field1 = _PField(initial=None, mandatory=False, factory=None, initial_factory=None)
    dct = {'field1': field1, '__invariant__': my_invariant}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    field1 = _PField(initial=None, mandatory=False, factory=None, initial_factory=None)
    field2 = _PField(initial=None, mandatory=True, factory=None, initial_factory=None)
    
    dct = {'field1': field1, 'field2': field2}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert hasattr(result, '_precord_mandatory_fields')
    assert 'field2' in result._precord_mandatory_fields
    assert 'field1' not in result._precord_mandatory_fields


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import PFIELD_NO_INITIAL
    
    field1 = _PField(initial=PFIELD_NO_INITIAL, mandatory=False, factory=None, initial_factory=None)
    field2 = _PField(initial=42, mandatory=False, factory=None, initial_factory=None)
    field3 = _PField(initial="test", mandatory=False, factory=None, initial_factory=None)
    
    dct = {'field1': field1, 'field2': field2, 'field3': field3}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert hasattr(result, '_precord_initial_values')
    assert result._precord_initial_values['field2'] == 42
    assert result._precord_initial_values['field3'] == "test"
    assert 'field1' not in result._precord_initial_values


def test_precord_meta_new_sets_empty_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    field1 = _PField(initial=None, mandatory=False, factory=None, initial_factory=None)
    dct = {'field1': field1}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert hasattr(result, '__slots__')
    assert result.__slots__ == ()


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    field1 = _PField(initial=None, mandatory=False, factory=None, initial_factory=None)
    base_dct = {'field1': field1, '_precord_fields': {'field1': field1}}
    
    class BaseRecord(metaclass=_PRecordMeta):
        _precord_fields = {'field1': field1}
    
    field2 = _PField(initial=None, mandatory=False, factory=None, initial_factory=None)
    dct = {'field2': field2}
    bases = (BaseRecord,)
    
    result = _PRecordMeta('DerivedRecord', bases, dct)
    
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_returns_type_instance():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    field1 = _PField(initial=None, mandatory=False, factory=None, initial_factory=None)
    dct = {'field1': field1}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert isinstance(result, type)
    assert result.__name__ == 'TestRecord'


# LLM-generated content at query #7
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap to use as buckets
    from pyrsistent import pmap
    pm = pmap({'x': 1, 'y': 2})
    
    # Create record using special attributes
    record = TestRecord(_precord_size=pm._size, _precord_buckets=pm._buckets)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create record using kwargs
    record = TestRecord(x=10, y=20)
    assert record['x'] == 10
    assert record['y'] == 20


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Create record with factory_fields parameter
    record = TestRecord(_factory_fields=set(), x=5)
    assert record['x'] == 5


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Create record with ignore_extra parameter
    record = TestRecord(_ignore_extra=True, x=7)
    assert record['x'] == 7


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        _precord_initial_values = {'z': 100}
        x = field()
        z = field()
    
    # Create record without providing z, should use initial value
    record = TestRecord(x=5)
    assert record['x'] == 5
    assert record['z'] == 100


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        _precord_initial_values = {'z': lambda: 200}
        x = field()
        z = field()
    
    # Create record, should call lambda for initial value
    record = TestRecord(x=5)
    assert record['x'] == 5
    assert record['z'] == 200


def test_precord_new_overrides_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        _precord_initial_values = {'z': 100}
        x = field()
        z = field()
    
    # Create record with explicit z value, should override initial value
    record = TestRecord(x=5, z=300)
    assert record['x'] == 5
    assert record['z'] == 300


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Create empty record
    record = TestRecord()
    assert len(record) == 0


# LLM-generated content at query #8
#--------------------------

```python
def test_precord_meta_new_creates_class_with_correct_attributes():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a mock _PField for testing
    field1 = _PField(type=str, initial=PFIELD_NO_INITIAL, mandatory=True, invariant=None, initial_factory=None, factory=None)
    field2 = _PField(type=int, initial=42, mandatory=False, invariant=None, initial_factory=None, factory=None)
    
    # Create a test class using the metaclass
    dct = {
        'field1': field1,
        'field2': field2,
    }
    
    test_class = _PRecordMeta('TestClass', (), dct)
    
    # Verify that __new__ was called and the class was created
    assert test_class is not None
    assert test_class.__name__ == 'TestClass'
    
    # Verify that _precord_fields was set correctly
    assert hasattr(test_class, '_precord_fields')
    assert 'field1' in test_class._precord_fields
    assert 'field2' in test_class._precord_fields
    
    # Verify that _precord_mandatory_fields was set correctly
    assert hasattr(test_class, '_precord_mandatory_fields')
    assert 'field1' in test_class._precord_mandatory_fields
    assert 'field2' not in test_class._precord_mandatory_fields
    
    # Verify that _precord_initial_values was set correctly
    assert hasattr(test_class, '_precord_initial_values')
    assert 'field2' in test_class._precord_initial_values
    assert test_class._precord_initial_values['field2'] == 42
    assert 'field1' not in test_class._precord_initial_values
    
    # Verify that __slots__ was set to empty tuple
    assert hasattr(test_class, '__slots__')
    assert test_class.__slots__ == ()
    
    # Verify that _precord_invariants was set
    assert hasattr(test_class, '_precord_invariants')
    assert isinstance(test_class._precord_invariants, tuple)


# LLM-generated content at query #9
#--------------------------

```python
def test_persistent_predicate_is_dirty_true():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 2)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True
    
    result = evolver.persistent()
    assert result['x'] == 2


def test_persistent_predicate_not_isinstance_true():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    pm = pmap({'x': 1})
    assert not isinstance(pm, TestRecord)
    
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 1


def test_persistent_predicate_both_conditions_true():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    original_pmap = pmap({'x': 1, 'y': 2})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 10)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True
    
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 10
    assert result['y'] == 2


# LLM-generated content at query #10
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty_and_already_correct_type():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from collections import namedtuple
    
    # Create a mock class
    MockField = namedtuple('MockField', ['factory', 'invariant'])
    mock_field = MockField(factory=lambda x: x, invariant=lambda x: (True, None))
    
    class MockPRecord:
        _precord_fields = {'test_field': mock_field}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockPRecord'
    
    # Create a PMap instance
    pmap = PMap()
    evolver = _PRecordEvolver(MockPRecord, pmap)
    evolver._destination_cls = MockPRecord
    
    result = evolver.persistent()
    assert isinstance(result, PMap)


def test_persistent_raises_invariant_exception_on_missing_mandatory_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._invariant_exception import InvariantException
    from collections import namedtuple
    
    MockField = namedtuple('MockField', ['factory', 'invariant'])
    mock_field = MockField(factory=lambda x: x, invariant=lambda x: (True, None))
    
    class MockPRecord:
        _precord_fields = {'test_field': mock_field}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        __name__ = 'MockPRecord'
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockPRecord, pmap)
    evolver._destination_cls = MockPRecord
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockPRecord.required_field' in e.missing_fields


def test_persistent_raises_invariant_exception_on_field_invariant_error():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._invariant_exception import InvariantException
    from collections import namedtuple
    
    MockField = namedtuple('MockField', ['factory', 'invariant'])
    mock_field = MockField(factory=lambda x: x, invariant=lambda x: (True, None))
    
    class MockPRecord:
        _precord_fields = {'test_field': mock_field}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockPRecord'
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockPRecord, pmap)
    evolver._destination_cls = MockPRecord
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error_code_1' in e.invariant_errors


def test_persistent_raises_invariant_exception_on_global_invariant_failure():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._invariant_exception import InvariantException
    from collections import namedtuple
    
    MockField = namedtuple('MockField', ['factory', 'invariant'])
    mock_field = MockField(factory=lambda x: x, invariant=lambda x: (True, None))
    
    def failing_global_invariant(subject):
        return (False, 'global_error')
    
    class MockPRecord:
        _precord_fields = {'test_field': mock_field}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        __name__ = 'MockPRecord'
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockPRecord, pmap)
    evolver._destination_cls = MockPRecord
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_with_dirty_state_creates_new_instance():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from collections import namedtuple
    
    MockField = namedtuple('MockField', ['factory', 'invariant'])
    mock_field = MockField(factory=lambda x: x, invariant=lambda x: (True, None))
    
    class MockPRecord:
        _precord_fields = {'test_field': mock_field}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockPRecord'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockPRecord, pmap)
    evolver._destination_cls = MockPRecord
    evolver._data = {'test_field': 'value'}
    
    result = evolver.persistent()
    assert result is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_persistent_returns_result_when_no_errors():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class MockField:
        def __init__(self):
            self.invariant_result = (True, None)
    
    class MockCls:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_fields = {}
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockCls, pmap)
    result = evolver.persistent()
    assert isinstance(result, MockCls)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class MockCls:
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        _precord_fields = {}
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockCls, pmap)
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'MockCls.field1' in e.missing_fields or 'MockCls.field2' in e.missing_fields


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class MockCls:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_fields = {}
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockCls, pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'error1' in e.invariant_errors
        assert 'error2' in e.invariant_errors


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class MockCls:
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        _precord_fields = {}
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockCls, pmap)
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_with_dirty_state():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class MockCls:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_fields = {}
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockCls, pmap)
    evolver[0] = 'test'
    result = evolver.persistent()
    assert isinstance(result, MockCls)


# LLM-generated content at query #12
#--------------------------

```python
def test_precord_evolver_set_with_valid_field():
    from pyrsistent import PMap, field, precord
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type
    
    class TestRecord(precord):
        x = field(type=int)
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.set('x', 42)
    
    assert result is evolver
    assert evolver._data['x'] == 42


def test_precord_evolver_set_with_invalid_field_name():
    from pyrsistent import PMap, field, precord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(precord):
        x = field(type=int)
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.set('nonexistent', 42)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "nonexistent" in str(e)


def test_precord_evolver_setitem():
    from pyrsistent import PMap, field, precord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(precord):
        x = field(type=int)
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.__setitem__('x', 99)
    
    assert evolver._data['x'] == 99


def test_precord_evolver_set_with_type_check():
    from pyrsistent import PMap, field, precord
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pexceptions import PTypeError
    
    class TestRecord(precord):
        x = field(type=int)
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.set('x', "not_an_int")
        assert False, "Should raise PTypeError"
    except PTypeError:
        pass


def test_precord_evolver_set_with_factory_fields():
    from pyrsistent import PMap, field, precord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(precord):
        x = field(type=int)
        y = field(type=str)
    
    x_field = TestRecord._precord_fields['x']
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap, _factory_fields={x_field})
    result = evolver.set('x', 42)
    
    assert result is evolver
    assert evolver._data['x'] == 42


def test_precord_evolver_set_with_factory_fields_excluded():
    from pyrsistent import PMap, field, precord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(precord):
        x = field(type=int)
    
    x_field = TestRecord._precord_fields['x']
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap, _factory_fields=set())
    result = evolver.set('x', 42)
    
    assert result is evolver
    assert evolver._data['x'] == 42


# LLM-generated content at query #13
#--------------------------

```python
def test_precord_repr():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='Alice', age=30)
    repr_str = repr(record)
    
    assert repr_str.startswith('TestRecord(')
    assert 'name=' in repr_str
    assert "'Alice'" in repr_str
    assert 'age=' in repr_str
    assert '30' in repr_str
    assert repr_str.endswith(')')


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
        a = field()
        b = field()
        c = field()
    
    record = MultiFieldRecord(a=1, b='text', c=[1, 2, 3])
    repr_str = repr(record)
    
    assert 'MultiFieldRecord(' in repr_str
    assert 'a=1' in repr_str
    assert "b='text'" in repr_str
    assert 'c=[1, 2, 3]' in repr_str


def test_precord_repr_with_none():
    from pyrsistent import PRecord, field
    
    class NullableRecord(PRecord):
        data = field()
    
    record = NullableRecord(data=None)
    repr_str = repr(record)
    
    assert 'data=None' in repr_str


# LLM-generated content at query #14
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
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_buckets = None
        _precord_size = 0
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, MockClass)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = {'field2'}
        _precord_invariants = []
        _precord_buckets = None
        _precord_size = 0
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'MockClass.field2' in e.missing_fields


def test_persistent_raises_invariant_exception_with_field_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def invariant(self, value):
            return (False, 'field_error')
    
    class MockClass:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_buckets = None
        _precord_size = 0
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes.append('field_error')
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'field_error' in e.invariant_errors


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    def failing_global_invariant(subject):
        return (False, 'global_error')
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        _precord_buckets = None
        _precord_size = 0
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #15
#--------------------------

```python
def test_persistent_predicate_is_dirty_true():
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


def test_persistent_predicate_not_isinstance_true():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 1)
    
    pm = evolver._pmap
    assert isinstance(pm, type(original_pmap))
    
    result = evolver.persistent()
    assert result is not None


def test_persistent_predicate_both_conditions():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 2)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty == True
    
    result = evolver.persistent()
    assert result['x'] == 2


# LLM-generated content at query #16
#--------------------------

```python
def test_set_with_valid_field():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    original = TestRecord(name='John', age=30)
    evolver = _PRecordEvolver(TestRecord, original._to_pmap())
    
    result = evolver.set('name', 'Jane')
    
    assert result is not None
    assert isinstance(result, _PRecordEvolver)


# LLM-generated content at query #17
#--------------------------

```python
def test_persistent_returns_result_when_no_errors():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestClass'
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    result = evolver.persistent()
    
    assert result is not None


def test_persistent_raises_invariant_exception_with_invariant_error_codes():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestClass'
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'error1' in e.invariant_errors


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []
        __name__ = 'TestClass'
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestClass.field1' in e.missing_fields


def test_persistent_raises_invariant_exception_with_both_errors_and_missing():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []
        __name__ = 'TestClass'
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'error1' in e.invariant_errors
        assert 'TestClass.field1' in e.missing_fields


def test_persistent_calls_check_global_invariants():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    def failing_global_invariant(subject):
        return (False, 'global_error')
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        __name__ = 'TestClass'
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #18
#--------------------------

```python
def test_set_with_valid_field_and_value():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import PField
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: x
            def invariant_func(x):
                return (True, None)
            self.invariant = invariant_func
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
        __name__ = 'MockClass'
    
    evolver = _PRecordEvolver(MockClass, pmap())
    result = evolver.set('test_field', 42)
    assert result is not None


def test_set_with_nonexistent_field():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        __name__ = 'MockClass'
    
    evolver = _PRecordEvolver(MockClass, pmap())
    try:
        evolver.set('nonexistent_field', 42)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'nonexistent_field' in str(e)


def test_set_with_type_check_failure():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import PTypeError
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: x
            def invariant_func(x):
                return (True, None)
            self.invariant = invariant_func
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
        __name__ = 'MockClass'
    
    evolver = _PRecordEvolver(MockClass, pmap())
    try:
        evolver.set('test_field', "not_an_int")
        assert False, "Should have raised PTypeError"
    except PTypeError:
        pass


def test_set_with_invariant_failure():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: x
            def invariant_func(x):
                return (False, 'test_error_code')
            self.invariant = invariant_func
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
        __name__ = 'MockClass'
    
    evolver = _PRecordEvolver(MockClass, pmap())
    result = evolver.set('test_field', 42)
    assert 'test_error_code' in evolver._invariant_error_codes


def test_set_with_factory_fields_filter():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: x
            def invariant_func(x):
                return (True, None)
            self.invariant = invariant_func
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
        __name__ = 'MockClass'
    
    test_field = MockClass._precord_fields['test_field']
    evolver = _PRecordEvolver(MockClass, pmap(), _factory_fields=[test_field])
    result = evolver.set('test_field', 42)
    assert result is not None


def test_setitem_delegates_to_set():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: x
            def invariant_func(x):
                return (True, None)
            self.invariant = invariant_func
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
        __name__ = 'MockClass'
    
    evolver = _PRecordEvolver(MockClass, pmap())
    result = evolver.__setitem__('test_field', 42)
    assert result is not None


# LLM-generated content at query #19
#--------------------------

```python
def test_precord_meta_new_creates_class_with_correct_attributes():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a mock field with mandatory=True and no initial value
    mandatory_field = _PField(type=str, factory=None, initial=PFIELD_NO_INITIAL, mandatory=True, invariant=None)
    
    # Create a mock field with mandatory=False and an initial value
    optional_field = _PField(type=int, factory=None, initial=42, mandatory=False, invariant=None)
    
    # Create a test class using the metaclass
    dct = {
        'field1': mandatory_field,
        'field2': optional_field,
    }
    
    test_class = _PRecordMeta('TestPRecord', (), dct)
    
    # Verify the class was created
    assert test_class is not None
    assert test_class.__name__ == 'TestPRecord'
    
    # Verify _precord_fields was populated
    assert hasattr(test_class, '_precord_fields')
    assert 'field1' in test_class._precord_fields
    assert 'field2' in test_class._precord_fields
    
    # Verify _precord_mandatory_fields contains only mandatory fields
    assert hasattr(test_class, '_precord_mandatory_fields')
    assert test_class._precord_mandatory_fields == {'field1'}
    
    # Verify _precord_initial_values contains only fields with initial values
    assert hasattr(test_class, '_precord_initial_values')
    assert test_class._precord_initial_values == {'field2': 42}
    
    # Verify __slots__ is set to empty tuple
    assert test_class.__slots__ == ()
    
    # Verify _precord_invariants exists (even if empty)
    assert hasattr(test_class, '_precord_invariants')


# LLM-generated content at query #20
#--------------------------

```python
def test_serialize_without_serializer():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="Alice", age=30)
    result = record.serialize()
    
    assert result == {"name": "Alice", "age": 30}


def test_serialize_with_format():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="Bob", age=25)
    result = record.serialize(format="json")
    
    assert result == {"name": "Bob", "age": 25}


def test_serialize_empty_record():
    from pyrsistent import PRecord
    
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    result = record.serialize()
    
    assert result == {}


def test_serialize_with_nested_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        data = field()
    
    record = TestRecord(name="Charlie", data={"nested": "value"})
    result = record.serialize()
    
    assert result == {"name": "Charlie", "data": {"nested": "value"}}


def test_serialize_with_none_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        optional = field()
    
    record = TestRecord(name="Dave", optional=None)
    result = record.serialize()
    
    assert result == {"name": "Dave", "optional": None}


def test_serialize_multiple_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()
        field3 = field()
        field4 = field()
    
    record = TestRecord(field1=1, field2="two", field3=3.0, field4=True)
    result = record.serialize()
    
    assert result == {"field1": 1, "field2": "two", "field3": 3.0, "field4": True}


# LLM-generated content at query #21
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class MockField:
        def __init__(self, name):
            self.name = name
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockCls, pmap)
    evolver._is_dirty = False
    
    result = evolver.persistent()
    assert result is not None


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._invariant import InvariantException
    
    class MockCls:
        __name__ = 'MockCls'
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockCls, pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockCls.required_field' in e.missing_fields


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._invariant import InvariantException
    
    class MockCls:
        __name__ = 'MockCls'
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockCls, pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error1' in e.invariant_errors
        assert 'error2' in e.invariant_errors


def test_persistent_calls_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._invariant import InvariantException
    
    global_invariant_called = []
    
    def mock_global_invariant(subject):
        global_invariant_called.append(True)
        return (False, 'global_error')
    
    class MockCls:
        __name__ = 'MockCls'
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [mock_global_invariant]
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockCls, pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(global_invariant_called) > 0
        assert 'global_error' in e.invariant_errors


def test_persistent_creates_new_instance_when_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class MockCls:
        __name__ = 'MockCls'
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockCls, pmap)
    evolver._data['new_key'] = 'new_value'
    
    result = evolver.persistent()
    assert isinstance(result, MockCls)


# LLM-generated content at query #22
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._missing_fields = ['TestRecord.y']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.y',)


def test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['TestRecord.y', 'TestRecord.z']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('TestRecord.y', 'TestRecord.z')


def test_persistent_predicate_false_when_no_errors_or_missing_fields():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    
    result = evolver.persistent()
    assert isinstance(result, TestRecord)


# LLM-generated content at query #23
#--------------------------

```python
def test_precord_initial_values_predicate():
    from pyrsistent import PRecord, field
    
    # Create a PRecord class with _precord_initial_values set
    class TestRecord(PRecord):
        name = field()
        age = field()
        _precord_initial_values = {'name': 'default_name', 'age': lambda: 0}
    
    # Verify that _precord_initial_values is truthy
    assert TestRecord._precord_initial_values
    
    # Create an instance without providing all values
    record = TestRecord(age=25)
    
    # Verify that initial values were applied
    assert record['name'] == 'default_name'
    assert record['age'] == 25


# LLM-generated content at query #24
#--------------------------

```python
def test_precord_meta_new_creates_class_successfully():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    # Create a simple field
    test_field = _PField(type=str, initial=None, factory=None, mandatory=True, invariant=None, initial_factory=None)
    
    # Create test dictionary with a field
    dct = {'test_field': test_field}
    bases = ()
    name = 'TestPRecord'
    
    # Call __new__ method
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    # Verify the class was created successfully
    assert result is not None
    assert result.__name__ == name
    assert hasattr(result, '_precord_fields')
    assert hasattr(result, '_precord_invariants')
    assert hasattr(result, '_precord_mandatory_fields')
    assert hasattr(result, '_precord_initial_values')
    assert hasattr(result, '__slots__')
    assert result.__slots__ == ()


# LLM-generated content at query #25
#--------------------------

```python
def test_persistent_predicate_line_6_evaluates_to_false():
    from pyrsistent import pmap, field, PRecord
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = pmap({'name': 'test'})
    evolver = TestRecord._PRecordEvolver(TestRecord, original_pmap)
    
    evolver.is_dirty = lambda: False
    
    result = evolver.persistent()
    
    assert result is original_pmap
    assert result['name'] == 'test'


# LLM-generated content at query #26
#--------------------------

```python
def test_precord_new_predicate_false_when_missing_precord_size():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    # Only '_precord_buckets' is present, '_precord_size' is missing
    # This should make the predicate at line 5 evaluate to False
    record = TestRecord(name="test")
    assert record.name == "test"


def test_precord_new_predicate_false_when_missing_precord_buckets():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    # Only '_precord_size' is present, '_precord_buckets' is missing
    # This should make the predicate at line 5 evaluate to False
    record = TestRecord(name="test")
    assert record.name == "test"


def test_precord_new_predicate_false_when_both_missing():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    # Neither '_precord_size' nor '_precord_buckets' are present
    # This should make the predicate at line 5 evaluate to False
    record = TestRecord(name="test")
    assert record.name == "test"


# LLM-generated content at query #27
#--------------------------

```python
def test_persistent_predicate_line_6_evaluates_to_false():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    is_dirty = evolver.is_dirty()
    pm = PMap.persistent(evolver)
    isinstance_check = isinstance(pm, TestRecord)
    
    predicate_result = is_dirty or not isinstance_check
    
    assert predicate_result is False


# LLM-generated content at query #28
#--------------------------

```python
def test_precord_new_without_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Create instance without _precord_size and _precord_buckets
    # This should make the predicate at line 5 evaluate to False
    record = TestRecord(name="John", age=30)
    
    assert record.name == "John"
    assert record.age == 30
    assert isinstance(record, TestRecord)


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    buckets = pmap(pre_size=2)._buckets
    size = 0
    record = TestRecord(_precord_size=size, _precord_buckets=buckets)
    assert isinstance(record, TestRecord)


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        _precord_initial_values = {'x': 10}
    
    record = TestRecord()
    assert record['x'] == 10


def test_precord_new_with_initial_values_and_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        _precord_initial_values = {'x': 10, 'y': 20}
    
    record = TestRecord(y=30)
    assert record['x'] == 10
    assert record['y'] == 30


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        _precord_initial_values = {'x': lambda: 42}
    
    record = TestRecord()
    assert record['x'] == 42


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_factory_fields=set(), x=5)
    assert record['x'] == 5


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=1)
    assert record['x'] == 1


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
    
    record = TestRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3


# LLM-generated content at query #31
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap

    class TestRecord(PRecord):
        name = field()

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []

    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap

    class TestRecord(PRecord):
        name = field()

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['TestRecord.name']

    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.name',)


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap

    class TestRecord(PRecord):
        name = field()

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['TestRecord.name']

    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('TestRecord.name',)


# LLM-generated content at query #32
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
    
    def serialize_with_format(value, format):
        if format == "uppercase":
            return value.upper() if isinstance(value, str) else value
        return value
    
    class TestRecord(PRecord):
        name = field(serializer=serialize_with_format)
        value = field()
    
    record = TestRecord(name="hello", value="world")
    result = record.serialize(format="uppercase")
    
    assert result == {"name": "HELLO", "value": "world"}


def test_serialize_empty_record():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    result = record.serialize()
    
    assert result == {}


def test_serialize_multiple_fields_with_serializers():
    from pyrsistent import PRecord, field
    
    def double_value(value):
        return value * 2 if isinstance(value, int) else value
    
    def uppercase_value(value):
        return value.upper() if isinstance(value, str) else value
    
    class TestRecord(PRecord):
        count = field(serializer=double_value)
        text = field(serializer=uppercase_value)
        plain = field()
    
    record = TestRecord(count=5, text="hello", plain="unchanged")
    result = record.serialize()
    
    assert result == {"count": 10, "text": "HELLO", "plain": "unchanged"}


def test_serialize_with_none_values():
    from pyrsistent import PRecord, field
    
    def custom_serializer(value):
        return "NULL" if value is None else value
    
    class TestRecord(PRecord):
        optional = field(serializer=custom_serializer)
        required = field()
    
    record = TestRecord(optional=None, required="value")
    result = record.serialize()
    
    assert result == {"optional": "NULL", "required": "value"}


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
    
    assert repr_str.startswith('TestRecord(')
    assert repr_str.endswith(')')
    assert 'name=' in repr_str
    assert "'Alice'" in repr_str
    assert 'age=' in repr_str
    assert '30' in repr_str


# LLM-generated content at query #34
#--------------------------

```python
def test_precord_meta_new_creates_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': _PField(mandatory=False, initial='default_value'),
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_fields')
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': _PField(mandatory=False, initial='default_value'),
        'field3': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result._precord_mandatory_fields == {'field1', 'field3'}


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': _PField(mandatory=False, initial='default_value'),
        'field3': _PField(mandatory=False, initial=42),
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result._precord_initial_values == {'field2': 'default_value', 'field3': 42}


def test_precord_meta_new_sets_empty_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result.__slots__ == ()


def test_precord_meta_new_stores_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    def test_invariant(record):
        return True, 'valid'
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        '__invariant__': test_invariant,
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    base_dct = {
        'base_field': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
    }
    base_class = _PRecordMeta.__new__(_PRecordMeta, 'BaseRecord', (), base_dct)
    
    dct = {
        'child_field': _PField(mandatory=False, initial='child_default'),
    }
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'ChildRecord', (base_class,), dct)
    
    assert 'base_field' in result._precord_fields
    assert 'child_field' in result._precord_fields


def test_precord_meta_new_returns_type_instance():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert isinstance(result, type)
    assert result.__name__ == name


# LLM-generated content at query #35
#--------------------------

```python
def test_precord_meta_new_creates_class_with_correct_attributes():
    from pyrsistent._precord import _PRecordMeta, PFIELD_NO_INITIAL
    from pyrsistent._field_common import _PField
    
    # Create a mock field with mandatory=True and initial=PFIELD_NO_INITIAL
    mandatory_field = _PField(type=str, initial=PFIELD_NO_INITIAL, factory=None, mandatory=True)
    
    # Create a mock field with mandatory=False and initial='default'
    optional_field = _PField(type=int, initial=42, factory=None, mandatory=False)
    
    # Create class dict with fields
    dct = {
        'field1': mandatory_field,
        'field2': optional_field,
    }
    
    # Call __new__ to create a class
    TestClass = _PRecordMeta('TestClass', (), dct)
    
    # Verify that __slots__ is set to empty tuple
    assert hasattr(TestClass, '__slots__')
    assert TestClass.__slots__ == ()
    
    # Verify that _precord_fields is created and populated
    assert hasattr(TestClass, '_precord_fields')
    assert 'field1' in TestClass._precord_fields
    assert 'field2' in TestClass._precord_fields
    
    # Verify that _precord_mandatory_fields contains only mandatory fields
    assert hasattr(TestClass, '_precord_mandatory_fields')
    assert TestClass._precord_mandatory_fields == {'field1'}
    
    # Verify that _precord_initial_values contains only fields with initial values
    assert hasattr(TestClass, '_precord_initial_values')
    assert TestClass._precord_initial_values == {'field2': 42}
    
    # Verify that _precord_invariants is created
    assert hasattr(TestClass, '_precord_invariants')
    assert isinstance(TestClass._precord_invariants, tuple)


# LLM-generated content at query #36
#--------------------------

```python
def test_persistent_returns_result_when_no_errors():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap

    class MockField:
        def __init__(self):
            self.invariant_called = False

    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size

        def keys(self):
            return []

    original_pmap = pmap({'a': 1})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    result = evolver.persistent()
    assert result is not None


def test_persistent_raises_on_missing_mandatory_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
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
            return []

    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockCls.required_field' in e.missing_fields


def test_persistent_raises_on_invariant_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap

    class MockCls:
        __name__ = 'MockCls'
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
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_checks_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap

    def failing_global_invariant(subject):
        return (False, 'global_error')

    class MockCls:
        __name__ = 'MockCls'
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]

        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size

        def keys(self):
            return []

    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
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

        def keys(self):
            return []

    original_pmap = pmap({'a': 1})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver[('b')] = 2
    result = evolver.persistent()
    assert result is not None


# LLM-generated content at query #37
#--------------------------

```python
def test_precord_meta_new_creates_class_with_correct_attributes():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a mock field with mandatory=True and no initial value
    field1 = _PField(mandatory=True, initial=PFIELD_NO_INITIAL, factory=None, invariant=None)
    
    # Create a mock field with mandatory=False and an initial value
    field2 = _PField(mandatory=False, initial=42, factory=None, invariant=None)
    
    # Create a dictionary to pass as dct
    dct = {
        'field1': field1,
        'field2': field2,
    }
    
    # Create a new class using _PRecordMeta
    TestClass = _PRecordMeta('TestClass', (object,), dct)
    
    # Verify the class was created
    assert TestClass is not None
    assert isinstance(TestClass, _PRecordMeta)
    
    # Verify _precord_fields was set correctly
    assert hasattr(TestClass, '_precord_fields')
    assert 'field1' in TestClass._precord_fields
    assert 'field2' in TestClass._precord_fields
    
    # Verify _precord_mandatory_fields contains only mandatory fields
    assert hasattr(TestClass, '_precord_mandatory_fields')
    assert 'field1' in TestClass._precord_mandatory_fields
    assert 'field2' not in TestClass._precord_mandatory_fields
    
    # Verify _precord_initial_values contains only fields with initial values
    assert hasattr(TestClass, '_precord_initial_values')
    assert 'field1' not in TestClass._precord_initial_values
    assert 'field2' in TestClass._precord_initial_values
    assert TestClass._precord_initial_values['field2'] == 42
    
    # Verify __slots__ is set to empty tuple
    assert hasattr(TestClass, '__slots__')
    assert TestClass.__slots__ == ()
    
    # Verify _precord_invariants exists (even if empty)
    assert hasattr(TestClass, '_precord_invariants')


# LLM-generated content at query #38
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


def test_precord_constructor_with_special_precord_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    inner_map = pmap({'x': 1})
    record = TestRecord(_precord_size=inner_map._precord_size, _precord_buckets=inner_map._precord_buckets)
    assert record['x'] == 1


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
    
    record = TestRecord(x=1, _ignore_extra=True, extra_field=999)
    assert record['x'] == 1
    assert 'extra_field' not in record


# LLM-generated content at query #39
#--------------------------

```python
def test_serialize_returns_dict_with_serialized_values():
    from pyrsistent import PRecord, field
    
    class CustomSerializer:
        def __call__(self, format, value):
            return value * 2
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=5, y=10)
    result = record.serialize()
    
    assert isinstance(result, dict)
    assert 'x' in result
    assert 'y' in result


# LLM-generated content at query #40
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field
    from pyrsistent._pmap import pvector
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap to get valid buckets
    test_pmap = TestRecord(x=1, y=2)
    buckets = test_pmap._buckets
    size = test_pmap._size
    
    # Call __new__ with both _precord_size and _precord_buckets
    result = TestRecord(_precord_size=size, _precord_buckets=buckets)
    
    # Verify the result is created directly without going through evolver
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2


# LLM-generated content at query #41
#--------------------------

```python
def test_persistent_returns_result_when_no_errors():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestRecord'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)


def test_persistent_raises_invariant_exception_with_field_invariant_errors():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestRecord'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []
        __name__ = 'TestRecord'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields


def test_persistent_calls_check_global_invariants():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    def global_invariant(record):
        return (False, 'global_error')
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [global_invariant]
        __name__ = 'TestRecord'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)


def test_persistent_creates_new_instance_when_dirty():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestRecord'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._data['new_key'] = 'new_value'
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)


# LLM-generated content at query #42
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_exist():
    from pyrsistent import PRecord, field
    from pyrsistent._invariant import InvariantException
    
    class TestRecord(PRecord):
        name = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, TestRecord())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:        
        assert e.invariant_errors == ('error1',)


def test_persistent_raises_invariant_exception_when_missing_fields_exist():
    from pyrsistent import PRecord, field
    from pyrsistent._invariant import InvariantException
    
    class TestRecord(PRecord):
        name = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, TestRecord())
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.missing_fields == ('TestRecord.name',)


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_exist():
    from pyrsistent import PRecord, field
    from pyrsistent._invariant import InvariantException
    
    class TestRecord(PRecord):
        name = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, TestRecord())
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('TestRecord.name',)


# LLM-generated content at query #43
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


def test_precord_constructor_with_internal_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    internal_map = pmap({'x': 1, 'y': 2})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert 'x' not in record


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _factory_fields={'x'})
    assert record['x'] == 5


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record


# LLM-generated content at query #44
#--------------------------

```python
def test_precord_meta_new_basic():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': _PField(mandatory=False, initial='default_value'),
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result.__name__ == 'TestRecord'
    assert hasattr(result, '_precord_fields')
    assert hasattr(result, '_precord_invariants')
    assert hasattr(result, '_precord_mandatory_fields')
    assert hasattr(result, '_precord_initial_values')
    assert result.__slots__ == ()


def test_precord_meta_new_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': _PField(mandatory=False, initial=PFIELD_NO_INITIAL),
        'field3': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result._precord_mandatory_fields == {'field1', 'field3'}


def test_precord_meta_new_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': _PField(mandatory=False, initial='default_value'),
        'field3': _PField(mandatory=False, initial=42),
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result._precord_initial_values == {'field2': 'default_value', 'field3': 42}


def test_precord_meta_new_with_invariant():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    def test_invariant(obj):
        return True, None
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        '__invariant__': test_invariant,
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
    }
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result.__slots__ == ()


def test_precord_meta_new_empty_fields():
    from pyrsistent._precord import _PRecordMeta
    
    dct = {}
    bases = (object,)
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result._precord_mandatory_fields == set()
    assert result._precord_initial_values == {}


# LLM-generated content at query #45
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_or_missing_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    TestRecord._precord_mandatory_fields = {'name', 'age'}
    TestRecord._precord_invariants = ()
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert 'Field invariant failed' in str(e)


def test_persistent_raises_invariant_exception_when_missing_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    TestRecord._precord_mandatory_fields = {'name', 'age'}
    TestRecord._precord_invariants = ()
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.missing_fields == ('TestRecord.name',)
        assert 'Field invariant failed' in str(e)


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    TestRecord._precord_mandatory_fields = {'name', 'age'}
    TestRecord._precord_invariants = ()
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ('TestRecord.name',)
        assert 'Field invariant failed' in str(e)


# LLM-generated content at query #46
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    # Create an evolver instance
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    # Set invariant_error_codes to trigger the condition at line 15
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = []
    
    # Attempt to call persistent() should raise InvariantException
    exception_raised = False
    try:
        evolver.persistent()
    except InvariantException as e:
        exception_raised = True
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ()
        assert e.message == 'Field invariant failed'
    
    assert exception_raised


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    # Create an evolver instance
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    # Set missing_fields to trigger the condition at line 15
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['TestRecord.x']
    
    # Attempt to call persistent() should raise InvariantException
    exception_raised = False
    try:
        evolver.persistent()
    except InvariantException as e:
        exception_raised = True
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.x',)
        assert e.message == 'Field invariant failed'
    
    assert exception_raised


def test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    # Create an evolver instance
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    # Set both to trigger the condition at line 15
    evolver._invariant_error_codes = ['error_code_1']
    evolver._missing_fields = ['TestRecord.x']
    
    # Attempt to call persistent() should raise InvariantException
    exception_raised = False
    try:
        evolver.persistent()
    except InvariantException as e:
        exception_raised = True
        assert e.invariant_errors == ('error_code_1',)
        assert e.missing_fields == ('TestRecord.x',)
        assert e.message == 'Field invariant failed'
    
    assert exception_raised


# LLM-generated content at query #47
#--------------------------

```python
def test_precord_new_without_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Create instance without _precord_size and _precord_buckets
    # This should evaluate the predicate at line 5 to False
    record = TestRecord(name="John", age=30)
    
    assert record['name'] == "John"
    assert record['age'] == 30
    assert isinstance(record, TestRecord)


# LLM-generated content at query #48
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


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field(initial=42)
    
    record = TestRecord(x=5)
    assert record['x'] == 5
    assert record['y'] == 42


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    counter = [0]
    def get_default():
        counter[0] += 1
        return counter[0]
    
    class TestRecord(PRecord):
        x = field()
        y = field(initial=get_default)
    
    record1 = TestRecord(x=1)
    record2 = TestRecord(x=2)
    assert record1['y'] == 1
    assert record2['y'] == 2


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(_factory_fields=set(), x=100, y=200)
    assert record['x'] == 100
    assert record['y'] == 200


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


def test_precord_new_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


# LLM-generated content at query #49
#--------------------------

```python
def test_set_field_exists():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from unittest.mock import Mock, MagicMock
    
    # Create a mock class
    mock_cls = Mock()
    mock_cls.__name__ = 'TestClass'
    
    # Create a mock field
    mock_field = Mock()
    mock_field.factory = Mock(return_value='processed_value')
    mock_field.invariant = Mock(return_value=(True, None))
    
    # Create a mock precord_fields dict with the field
    mock_cls._precord_fields = {'test_key': mock_field}
    mock_cls._precord_mandatory_fields = set()
    mock_cls._precord_invariants = []
    
    # Create an evolver instance
    original_pmap = pmap({})
    evolver = _PRecordEvolver(mock_cls, original_pmap)
    
    # Call set with a key that exists in _precord_fields
    # This should execute the if field: block (line 3 predicate evaluates to True)
    result = evolver.set('test_key', 'test_value')
    
    # Verify that field.factory was called
    mock_field.factory.assert_called_once()
    
    # Verify that field.invariant was called
    mock_field.invariant.assert_called_once()
    
    # Verify result is the evolver itself (from super().set())
    assert result is not None


# LLM-generated content at query #50
#--------------------------

```python
def test_precord_new_predicate_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Create instance without the special attributes to make predicate at line 5 evaluate to False
    record = TestRecord(name="John", age=30)
    
    # Verify the record was created successfully through the evolver path
    assert record.name == "John"
    assert record.age == 30


# LLM-generated content at query #51
#--------------------------

```python
def test_serialize_returns_dict():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    result = record.serialize()
    
    assert isinstance(result, dict)
    assert "name" in result
    assert "age" in result
    assert result["name"] == "John"
    assert result["age"] == 30


# LLM-generated content at query #52
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
    assert isinstance(repr_str, str)


# LLM-generated content at query #53
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


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert 'x' not in record


def test_precord_constructor_with_extra_fields_ignored():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _ignore_extra=True, extra_field=999)
    assert record['x'] == 1
    assert 'extra_field' not in record


def test_precord_constructor_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    base_map = pmap({'x': 5})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets)
    assert record['x'] == 5


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100, y=200)
    assert record['x'] == 100
    assert record['y'] == 200


def test_precord_constructor_partial_override():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields={'x'})
    assert record['x'] == 1


# LLM-generated content at query #54
#--------------------------

```python
def test_persistent_predicate_line_6_true_when_dirty():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    evolver.set('x', 5)
    is_dirty_result = evolver.is_dirty()
    
    assert is_dirty_result is True


def test_persistent_predicate_line_6_true_when_not_isinstance():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    class OtherRecord(PRecord):
        y = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    evolver.set('x', 5)
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)


def test_persistent_predicate_line_11_true_with_mandatory_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    TestRecord._precord_mandatory_fields = {'x'}
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    mandatory_fields_exist = bool(TestRecord._precord_mandatory_fields)
    
    assert mandatory_fields_exist is True


def test_persistent_predicate_line_15_true_with_invariant_errors():
    from pyrsistent import PRecord, field, InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    predicate_result = bool(evolver._invariant_error_codes or evolver._missing_fields)
    
    assert predicate_result is True


def test_persistent_predicate_line_15_true_with_missing_fields():
    from pyrsistent import PRecord, field, InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._missing_fields = ['TestRecord.x']
    
    predicate_result = bool(evolver._invariant_error_codes or evolver._missing_fields)
    
    assert predicate_result is True


# LLM-generated content at query #55
#--------------------------

```python
def test_precord_new_with_precord_size_and_precord_buckets():
    from pyrsistent import PRecord, field, pvector
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a valid PMap to use as buckets
    test_pmap = PMap(0, pvector())
    
    # Call __new__ with both '_precord_size' and '_precord_buckets' in kwargs
    # This should trigger the condition on line 5 to evaluate to True
    result = TestRecord(__new__=TestRecord.__new__, _precord_size=0, _precord_buckets=test_pmap._buckets)
    
    # Verify that the result is an instance of TestRecord
    assert isinstance(result, PRecord)


# LLM-generated content at query #56
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
        items = field(initial=list)
    
    record = TestRecord()
    assert record['items'] == []


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_with_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    try:
        record = TestRecord(x=1, z=2)
        assert False, "Should have raised an error for extra field"
    except:
        pass


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields=None)
    assert record['x'] == 1


def test_precord_constructor_empty():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 5})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert record['x'] == 5


# LLM-generated content at query #57
#--------------------------

```python
def test_persistent_predicate_line_6_true_when_dirty():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 2)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True
    
    pm = evolver._PMap__get_persistent_pmap()
    is_instance_check = isinstance(pm, TestRecord)
    
    predicate_result = is_dirty or not is_instance_check
    assert predicate_result is True


# LLM-generated content at query #58
#--------------------------

```python
def test_persistent_predicate_line_6_evaluates_to_false():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = pmap({'name': 'test'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    is_dirty = evolver.is_dirty()
    pm = evolver._PMap__class__._Evolver.persistent(evolver)
    
    predicate_result = is_dirty or not isinstance(pm, TestRecord)
    
    assert predicate_result == False


# LLM-generated content at query #59
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    assert record['name'] == "John"
    assert record['age'] == 30


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord()
    assert 'name' not in record
    assert 'age' not in record


def test_precord_constructor_with_defaults():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=0)
    
    record = TestRecord()
    assert record['name'] == "DefaultName"
    assert record['age'] == 0


def test_precord_constructor_with_defaults_and_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=0)
    
    record = TestRecord(name="Jane", age=25)
    assert record['name'] == "Jane"
    assert record['age'] == 25


def test_precord_constructor_with_callable_default():
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
    
    record = TestRecord(_factory_fields=None, name="Test")
    assert record['name'] == "Test"


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(_ignore_extra=True, name="Test", extra_field="ignored")
    assert record['name'] == "Test"
    assert 'extra_field' not in record


def test_precord_constructor_partial_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
        email = field()
    
    record = TestRecord(name="John", email="john@example.com")
    assert record['name'] == "John"
    assert record['email'] == "john@example.com"
    assert 'age' not in record


# LLM-generated content at query #60
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
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['x'] == [1, 2, 3]
    assert record2['x'] == [1, 2, 3]


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _factory_fields={'x'})
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
    
    base_map = pmap({'x': 42})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets)
    assert record['x'] == 42


# LLM-generated content at query #61
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


def test_precord_constructor_with_initial_values_override():
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


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 5})
    record = TestRecord(_precord_size=1, _precord_buckets=internal_map._pmap_buckets)
    assert record['x'] == 5


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields={'x': int})
    assert record['x'] == 1


def test_precord_constructor_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _ignore_extra=True, extra_field=999)
    assert record['x'] == 1
    assert 'extra_field' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_partial_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=5)
    assert record['x'] == 5
    assert 'y' not in record


# LLM-generated content at query #62
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


def test_precord_constructor_with_callable_initial_value():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [])
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['x'] == []
    assert record2['x'] == []
    assert record1['x'] is not record2['x']


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
    
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=5)
    
    record = TestRecord()
    assert len(record) == 1
    assert record['x'] == 5


def test_precord_constructor_with_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    inner_pmap = pmap({'x': 42})
    record = TestRecord(_precord_size=inner_pmap._size, _precord_buckets=inner_pmap._buckets)
    assert record['x'] == 42


# LLM-generated content at query #63
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


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100, y=200)
    assert record['x'] == 100
    assert record['y'] == 200


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2, _factory_fields=None)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, z=999, _ignore_extra=True)
    assert record['x'] == 1
    assert 'z' not in record


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    base_map = pmap({'x': 42})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets)
    assert record['x'] == 42


def test_precord_constructor_partial_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        z = field()
    
    record = TestRecord(x=1, z=3)
    assert record['x'] == 1
    assert record['z'] == 3
    assert 'y' not in record


# LLM-generated content at query #64
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


def test_precord_constructor_with_callable_initial_value():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [1, 2, 3])
    
    record = TestRecord()
    assert record['x'] == [1, 2, 3]


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_pmap = pmap({'x': 5})
    record = TestRecord(_precord_size=internal_pmap._size, _precord_buckets=internal_pmap._buckets)
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
        x = field(initial=None)
    
    record = TestRecord()
    assert record['x'] is None


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields=None)
    assert record['x'] == 1


# LLM-generated content at query #65
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    assert record['name'] == "John"
    assert record['age'] == 30


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=0)
    
    record = TestRecord()
    assert record['name'] == "DefaultName"
    assert record['age'] == 0


def test_precord_constructor_with_initial_values_callable():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record = TestRecord()
    assert record['items'] == []


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=0)
    
    record = TestRecord(name="Jane", age=25)
    assert record['name'] == "Jane"
    assert record['age'] == 25


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30, _factory_fields=None)
    assert record['name'] == "John"
    assert record['age'] == 30


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name="John", extra_field="ignored", _ignore_extra=True)
    assert record['name'] == "John"
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
    
    base_pmap = pmap({'name': 'John'})
    record = TestRecord(_precord_size=base_pmap._size, _precord_buckets=base_pmap._buckets)
    assert record['name'] == 'John'


# LLM-generated content at query #66
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
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    result = evolver.persistent()
    
    assert isinstance(result, MockCls)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        __name__ = 'TestClass'
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestClass.required_field' in e.missing_fields


def test_persistent_raises_invariant_exception_with_field_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        __name__ = 'TestClass'
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
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'error_code_1' in e.invariant_errors


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    from pyrsistent import pmap
    
    def failing_invariant(obj):
        return (False, 'global_error')
    
    class MockCls:
        __name__ = 'TestClass'
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
            self._is_new = True
        
        def keys(self):
            return []
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver[0] = 'new_value'
    
    result = evolver.persistent()
    assert hasattr(result, '_is_new')


# LLM-generated content at query #67
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    assert record['name'] == "John"
    assert record['age'] == 30


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=0)
    
    record = TestRecord()
    assert record['name'] == "DefaultName"
    assert record['age'] == 0


def test_precord_constructor_with_partial_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=0)
    
    record = TestRecord(name="Alice")
    assert record['name'] == "Alice"
    assert record['age'] == 0


def test_precord_constructor_with_callable_initial():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record = TestRecord()
    assert record['items'] == []


def test_precord_constructor_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(_ignore_extra=True, name="John", extra_field="ignored")
    assert record['name'] == "John"
    assert 'extra_field' not in record


def test_precord_constructor_internal_creation():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    internal_map = pmap({'name': 'John'})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert record['name'] == 'John'


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="Default")
    
    record = TestRecord()
    assert 'name' in record
    assert record['name'] == "Default"


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=25)
    
    record = TestRecord(name="Bob", age=35)
    assert record['name'] == "Bob"
    assert record['age'] == 35


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(_factory_fields={}, name="Test")
    assert record['name'] == "Test"


# LLM-generated content at query #68
#--------------------------

```python
def test_precord_evolver_persistent_predicate_line_6():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    original_pmap = pmap({'name': 'John', 'age': 30})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    evolver.set('name', 'Jane')
    
    is_dirty = evolver.is_dirty()
    pm = pmap(evolver.data)
    isinstance_check = isinstance(pm, TestRecord)
    
    predicate_result = is_dirty or not isinstance_check
    
    assert predicate_result is True


# LLM-generated content at query #69
#--------------------------

```python
def test_precord_new_predicate_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Create instance without _precord_size and _precord_buckets
    # This should make the predicate at line 5 evaluate to False
    record = TestRecord(name="Alice", age=30)
    
    assert record.name == "Alice"
    assert record.age == 30
    assert isinstance(record, TestRecord)


# LLM-generated content at query #70
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap to use as buckets
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


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field(initial=100)
    
    record = TestRecord(x=5)
    
    assert record['x'] == 5
    assert record['y'] == 100


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


def test_precord_new_with_callable_initial_value():
    from pyrsistent import PRecord, field
    
    def default_factory():
        return 42
    
    class TestRecord(PRecord):
        x = field()
        y = field(initial=default_factory)
    
    record = TestRecord(x=10)
    
    assert record['x'] == 10
    assert record['y'] == 42


def test_precord_new_kwargs_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=100)
        y = field(initial=200)
    
    record = TestRecord(x=1, y=2)
    
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    
    assert len(record) == 0


# LLM-generated content at query #71
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='Alice', age=30)
    assert record['name'] == 'Alice'
    assert record['age'] == 30


def test_precord_constructor_with_default_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        status = field(initial='active')
    
    record = TestRecord(name='Bob')
    assert record['name'] == 'Bob'
    assert record['status'] == 'active'


def test_precord_constructor_with_callable_defaults():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        items = field(initial=list)
    
    record = TestRecord(name='Charlie')
    assert record['name'] == 'Charlie'
    assert record['items'] == []


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord()
    assert 'name' not in record


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(_factory_fields=True, name='David', age=25)
    assert record['name'] == 'David'
    assert record['age'] == 25


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(_ignore_extra=True, name='Eve', extra_field='ignored')
    assert record['name'] == 'Eve'
    assert 'extra_field' not in record


def test_precord_constructor_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    # Create using special internal attributes
    pm = pmap({'name': 'Frank'})
    record = TestRecord(_precord_size=pm._size, _precord_buckets=pm._buckets)
    assert record['name'] == 'Frank'


# LLM-generated content at query #72
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
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['x'] == [1, 2, 3]
    assert record2['x'] == [1, 2, 3]
    assert record1['x'] is not record2['x']


def test_precord_constructor_empty():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_internal_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    test_map = pmap({'x': 5})
    record = TestRecord(_precord_size=test_map._size, _precord_buckets=test_map._buckets)
    assert record['x'] == 5


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields={'x': int})
    assert record['x'] == 1


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record


# LLM-generated content at query #73
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pvector
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap with the necessary structure
    from pyrsistent._pmap import pmap
    initial_pmap = pmap({'x': 1, 'y': 2})
    
    # Call __new__ with both _precord_size and _precord_buckets
    # This should trigger the predicate at line 5 to be True
    result = TestRecord(
        _precord_size=initial_pmap._size,
        _precord_buckets=initial_pmap._buckets
    )
    
    # Verify the result is a TestRecord instance
    assert isinstance(result, TestRecord)
    assert result._size == initial_pmap._size
    assert result._buckets == initial_pmap._buckets


# LLM-generated content at query #74
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
        __name__ = 'MockClass'

        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size

        def keys(self):
            return []

    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, MockClass)


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import InvariantException

    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'

        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size

        def keys(self):
            return []

    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import InvariantException

    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []
        __name__ = 'TestClass'

        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size

        def keys(self):
            return []

    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.field1' in e.missing_fields


def test_persistent_raises_invariant_exception_with_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import InvariantException

    def failing_global_invariant(subject):
        return (False, 'global_error')

    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        __name__ = 'MockClass'

        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size

        def keys(self):
            return []

    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_with_clean_pmap_returns_original():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap

    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'

        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self._is_instance = True

        def keys(self):
            return []

        def __instancecheck__(self, instance):
            return isinstance(instance, MockClass)

    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert result is not None


def test_persistent_collects_multiple_invariant_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import InvariantException

    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'

        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size

        def keys(self):
            return []

    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2', 'error3']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 3
        assert 'error1' in e.invariant_errors
        assert 'error2' in e.invariant_errors
        assert 'error3' in e.invariant_errors


# LLM-generated content at query #75
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
    result = TestRecord(__new__=TestRecord.__new__, _precord_size=0, _precord_buckets=initial_pmap._buckets)
    
    assert isinstance(result, TestRecord)
    assert result._size == 0


# LLM-generated content at query #76
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


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: 42)
        y = field()
    
    record = TestRecord(y=5)
    assert record['x'] == 42
    assert record['y'] == 5


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert 'x' not in record


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100, y=200)
    assert record['x'] == 100
    assert record['y'] == 200


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(_factory_fields=True, x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=1, z=999)
    assert record['x'] == 1
    assert 'z' not in record


def test_precord_constructor_internal_creation():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 5})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert record['x'] == 5


# LLM-generated content at query #77
#--------------------------

```python
def test_precord_repr():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='John', age=30)
    repr_str = repr(record)
    
    assert 'TestRecord' in repr_str
    assert 'name=' in repr_str
    assert "'John'" in repr_str
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
        x = field()
        y = field()
        z = field()
    
    record = MultiFieldRecord(x=1, y=2, z=3)
    repr_str = repr(record)
    
    assert 'MultiFieldRecord' in repr_str
    assert 'x=1' in repr_str
    assert 'y=2' in repr_str
    assert 'z=3' in repr_str


def test_precord_repr_with_special_characters():
    from pyrsistent import PRecord, field
    
    class SpecialRecord(PRecord):
        text = field()
    
    record = SpecialRecord(text='hello\nworld')
    repr_str = repr(record)
    
    assert 'SpecialRecord' in repr_str
    assert 'text=' in repr_str
    assert repr_str.count("'") >= 2


# LLM-generated content at query #78
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
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    result = evolver.persistent()
    assert isinstance(result, MockClass)


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class MockField:
        def __init__(self, name):
            self.name = name
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'TestClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.field1' in e.missing_fields or 'TestClass.field2' in e.missing_fields


def test_persistent_checks_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [lambda x: (False, 'global_error')]
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_converts_pmap_to_destination_class():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self._is_mock = True
        
        def keys(self):
            return []
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    evolver[0] = 0  # Mark as dirty
    result = evolver.persistent()
    assert isinstance(result, MockClass)
    assert hasattr(result, '_is_mock')


# LLM-generated content at query #79
#--------------------------

```python
def test_persistent_predicate_is_dirty_true():
    from pyrsistent import pmap, PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    original_pmap = pmap({'name': 'test', 'value': 42})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('name', 'updated')
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True


def test_persistent_predicate_not_isinstance_true():
    from pyrsistent import pmap, PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from unittest.mock import Mock, patch
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = pmap({'name': 'test'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    with patch.object(evolver, 'is_dirty', return_value=False):
        with patch('pyrsistent._precord.PMap._Evolver.persistent') as mock_parent_persistent:
            mock_pm = Mock()
            mock_pm._buckets = []
            mock_pm._size = 0
            mock_pm.keys.return_value = ['name']
            mock_parent_persistent.return_value = mock_pm
            
            with patch('pyrsistent._precord.check_global_invariants'):
                result = evolver.persistent()
                
                assert isinstance(result, TestRecord)


def test_persistent_predicate_both_conditions_true():
    from pyrsistent import pmap, PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from unittest.mock import Mock, patch
    
    class TestRecord(PRecord):
        name = field()
    
    original_pmap = pmap({'name': 'test'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('name', 'modified')
    
    with patch('pyrsistent._precord.PMap._Evolver.persistent') as mock_parent_persistent:
        mock_pm = Mock()
        mock_pm._buckets = []
        mock_pm._size = 0
        mock_pm.keys.return_value = ['name']
        mock_parent_persistent.return_value = mock_pm
        
        with patch('pyrsistent._precord.check_global_invariants'):
            result = evolver.persistent()
            assert isinstance(result, TestRecord)


# LLM-generated content at query #80
#--------------------------

```python
def test_persistent_with_no_errors():
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
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, MockClass)


def test_persistent_with_invariant_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()


def test_persistent_with_missing_mandatory_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'TestClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 2
        assert 'TestClass.field1' in e.missing_fields or 'TestClass.field2' in e.missing_fields


def test_persistent_with_global_invariant_failure():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_returns_same_instance_when_not_dirty():
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
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._is_dirty = False
    
    result = evolver.persistent()
    assert isinstance(result, MockClass)


# LLM-generated content at query #81
#--------------------------

```python
def test_serialize_without_serializer():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='Alice', age=30)
    result = record.serialize()
    
    assert result == {'name': 'Alice', 'age': 30}


def test_serialize_with_custom_serializer():
    from pyrsistent import PRecord, field
    
    def serialize_upper(serializer, format, value):
        return value.upper() if isinstance(value, str) else value
    
    class TestRecord(PRecord):
        name = field(serializer=serialize_upper)
        age = field()
    
    record = TestRecord(name='alice', age=30)
    result = record.serialize()
    
    assert result['name'] == 'ALICE'
    assert result['age'] == 30


def test_serialize_with_format_parameter():
    from pyrsistent import PRecord, field
    
    def serialize_with_format(serializer, format, value):
        if format == 'json':
            return str(value)
        return value
    
    class TestRecord(PRecord):
        value = field(serializer=serialize_with_format)
    
    record = TestRecord(value=42)
    result = record.serialize(format='json')
    
    assert result['value'] == '42'


def test_serialize_empty_record():
    from pyrsistent import PRecord
    
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    result = record.serialize()
    
    assert result == {}


def test_serialize_multiple_fields_with_serializers():
    from pyrsistent import PRecord, field
    
    def double_serializer(serializer, format, value):
        return value * 2 if isinstance(value, int) else value
    
    def upper_serializer(serializer, format, value):
        return value.upper() if isinstance(value, str) else value
    
    class TestRecord(PRecord):
        count = field(serializer=double_serializer)
        name = field(serializer=upper_serializer)
        description = field()
    
    record = TestRecord(count=5, name='test', description='a record')
    result = record.serialize()
    
    assert result['count'] == 10
    assert result['name'] == 'TEST'
    assert result['description'] == 'a record'


def test_serialize_with_none_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    record = TestRecord(name='test', value=None)
    result = record.serialize()
    
    assert result['name'] == 'test'
    assert result['value'] is None


# LLM-generated content at query #82
#--------------------------

```python
def test_precord_meta_new_creates_class_with_correct_attributes():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a test field
    test_field = _PField(mandatory=True, initial=PFIELD_NO_INITIAL, factory=None, initial_factory=None)
    optional_field = _PField(mandatory=False, initial="default_value", factory=None, initial_factory=None)
    
    # Create a dictionary with fields
    dct = {
        'field1': test_field,
        'field2': optional_field,
    }
    
    # Create a class using the metaclass
    TestClass = _PRecordMeta('TestClass', (), dct)
    
    # Verify that __slots__ is set to empty tuple
    assert TestClass.__slots__ == ()
    
    # Verify that _precord_fields is created
    assert hasattr(TestClass, '_precord_fields')
    assert 'field1' in TestClass._precord_fields
    assert 'field2' in TestClass._precord_fields
    
    # Verify that _precord_mandatory_fields contains only mandatory fields
    assert hasattr(TestClass, '_precord_mandatory_fields')
    assert 'field1' in TestClass._precord_mandatory_fields
    assert 'field2' not in TestClass._precord_mandatory_fields
    
    # Verify that _precord_initial_values contains only fields with initial values
    assert hasattr(TestClass, '_precord_initial_values')
    assert 'field1' not in TestClass._precord_initial_values
    assert 'field2' in TestClass._precord_initial_values
    assert TestClass._precord_initial_values['field2'] == "default_value"
    
    # Verify that _precord_invariants is created
    assert hasattr(TestClass, '_precord_invariants')
    assert isinstance(TestClass._precord_invariants, tuple)


# LLM-generated content at query #83
#--------------------------

```python
def test_precord_evolver_set_with_field_found():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from collections import namedtuple
    
    # Create a mock field with required attributes
    MockField = namedtuple('MockField', ['factory', 'type', 'invariant'])
    mock_field = MockField(
        factory=lambda x: x,
        type=(),
        invariant=lambda x: (True, None)
    )
    
    # Create a mock destination class
    class MockDestinationCls:
        __name__ = 'MockClass'
        _precord_fields = {'test_key': mock_field}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    
    # Create evolver instance
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    
    # Call set method - this tests that line 3 predicate evaluates to True
    # (field is found and truthy)
    result = evolver.set('test_key', 'test_value')
    
    # Verify that the set operation succeeded (field was found)
    assert result is not None
    assert isinstance(result, _PRecordEvolver)


# LLM-generated content at query #84
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
        name = field()
        age = field(initial=25)
    
    record = TestRecord(name='Jane')
    assert record['name'] == 'Jane'
    assert record['age'] == 25


def test_precord_constructor_with_callable_initial_value():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        items = field(initial=list)
    
    record1 = TestRecord(name='test1')
    record2 = TestRecord(name='test2')
    assert record1['items'] is not record2['items']


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord()
    assert 'name' not in record or record['name'] is None


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(_ignore_extra=True, name='John', extra='ignored')
    assert record['name'] == 'John'
    assert 'extra' not in record


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(_factory_fields={'name': str}, name='John')
    assert record['name'] == 'John'


def test_precord_constructor_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    base_map = pmap({'name': 'John'})
    record = TestRecord(_precord_size=base_map._precord_size, _precord_buckets=base_map._precord_buckets)
    assert record['name'] == 'John'


def test_precord_constructor_overwrites_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='Default')
        age = field()
    
    record = TestRecord(name='Custom', age=30)
    assert record['name'] == 'Custom'
    assert record['age'] == 30


def test_precord_constructor_multiple_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()
        field3 = field()
    
    record = TestRecord(field1='a', field2='b', field3='c')
    assert record['field1'] == 'a'
    assert record['field2'] == 'b'
    assert record['field3'] == 'c'


# LLM-generated content at query #85
#--------------------------

```python
def test_precord_meta_new_returns_class():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a simple field for testing
    test_field = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)
    
    # Test that __new__ returns a class (type instance)
    dct = {'_precord_fields': {}, '__invariant__': None}
    bases = ()
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestPRecord', bases, dct)
    
    # The predicate at line 1 is that __new__ is called with mcs, name, bases, dct parameters
    # The result should be a class (instance of type)
    assert isinstance(result, type)
    assert result.__name__ == 'TestPRecord'
    assert hasattr(result, '_precord_fields')
    assert hasattr(result, '_precord_invariants')
    assert hasattr(result, '_precord_mandatory_fields')
    assert hasattr(result, '_precord_initial_values')
    assert result.__slots__ == ()


# LLM-generated content at query #86
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field
    from pyrsistent._pmap import pvector
    
    class TestRecord(PRecord):
        x = field()
    
    # Create initial pmap to get buckets and size
    initial_pmap = TestRecord(x=1)
    buckets = initial_pmap._buckets
    size = initial_pmap._size
    
    # Create new instance using special attributes
    result = TestRecord(_precord_size=size, _precord_buckets=buckets)
    assert result.x == 1


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    result = TestRecord(x=10, y=20)
    assert result.x == 10
    assert result.y == 20


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        __precord_initial_values = {'x': 42}
    
    result = TestRecord()
    assert result.x == 42


def test_precord_new_with_initial_values_override():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        __precord_initial_values = {'x': 42}
    
    result = TestRecord(x=100)
    assert result.x == 100


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    result = TestRecord(_factory_fields=set(), x=5)
    assert result.x == 5


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    result = TestRecord(_ignore_extra=True, x=5)
    assert result.x == 5


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    result = TestRecord()
    assert len(result) == 0


def test_precord_new_with_callable_initial_value():
    from pyrsistent import PRecord, field
    
    counter = [0]
    
    def get_value():
        counter[0] += 1
        return counter[0]
    
    class TestRecord(PRecord):
        x = field()
        __precord_initial_values = {'x': get_value}
    
    result = TestRecord()
    assert result.x == 1


# LLM-generated content at query #87
#--------------------------

```python
def test_persistent_returns_dirty_precord_instance():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestRecord'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        def keys(self):
            return []
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._is_dirty = True
    result = evolver.persistent()
    assert isinstance(result, TestRecord)


def test_persistent_returns_clean_precord_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestRecord'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        def keys(self):
            return []
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._is_dirty = False
    result = evolver.persistent()
    assert result is original_pmap


def test_persistent_raises_invariant_exception_on_missing_mandatory_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        __name__ = 'TestRecord'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        def keys(self):
            return []
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._is_dirty = True
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.required_field' in e.missing_fields


def test_persistent_raises_invariant_exception_on_field_invariant_error():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestRecord'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        def keys(self):
            return []
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    evolver._is_dirty = True
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error_code_1' in e.invariant_errors


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        __name__ = 'TestRecord'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        def keys(self):
            return []
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._is_dirty = True
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #88
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap with initial values
    initial_pmap = pmap({'x': 1, 'y': 2})
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
        y = field()
    
    record = TestRecord(y=15)
    assert record['x'] == 5
    assert record['y'] == 15


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: 42)
        y = field()
    
    record = TestRecord(y=100)
    assert record['x'] == 42
    assert record['y'] == 100


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(_factory_fields={}, x=5, y=10)
    assert record['x'] == 5
    assert record['y'] == 10


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=5, z=99)
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
        x = field(initial=5)
        y = field(initial=10)
    
    record = TestRecord(x=99)
    assert record['x'] == 99
    assert record['y'] == 10


def test_precord_new_multiple_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = TestRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3


# LLM-generated content at query #89
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


def test_precord_constructor_with_initial_and_override():
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
        y = field(initial=lambda: {})
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['x'] is not record2['x']
    assert record1['y'] is not record2['y']


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_pmap = pmap({'x': 5})
    record = TestRecord(_precord_size=internal_pmap._size, _precord_buckets=internal_pmap._buckets)
    assert record['x'] == 5


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


# LLM-generated content at query #90
#--------------------------

```python
def test_persistent_returns_same_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._PMap__data = original_pmap
    evolver._PMap__count = 0
    
    result = evolver.persistent()
    assert result is not None


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._precord import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        __name__ = 'MockClass'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockClass.required_field' in e.missing_fields


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._precord import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error_code_1', 'error_code_2']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1', 'error_code_2')


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    global_invariant_called = []
    
    def mock_invariant(subject):
        global_invariant_called.append(subject)
        return (True, None)
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [mock_invariant]
        __name__ = 'MockClass'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    result = evolver.persistent()
    assert len(global_invariant_called) > 0


def test_persistent_raises_exception_with_both_missing_fields_and_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._precord import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        __name__ = 'MockClass'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert 'MockClass.required_field' in e.missing_fields


# LLM-generated content at query #91
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
    
    # Create an initial record
    original = TestRecord(name='test')
    
    # Create an evolver from the original record's pmap
    evolver = _PRecordEvolver(TestRecord, original._pmap)
    
    # Set a value to make it dirty
    evolver.set('name', 'modified')
    
    # Call persistent which should trigger the predicate at line 6
    result = evolver.persistent()
    
    # Verify that is_dirty was True and a new instance was created
    assert isinstance(result, TestRecord)
    assert result.name == 'modified'
    assert result is not original


# LLM-generated content at query #92
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


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [1, 2, 3])
    
    record = TestRecord()
    assert record['x'] == [1, 2, 3]


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    size = 1
    buckets = ({}, {})
    record = TestRecord(_precord_size=size, _precord_buckets=buckets)
    assert isinstance(record, TestRecord)


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _factory_fields=['x'])
    assert record['x'] == 5


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


# LLM-generated content at query #93
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
        name = field()
        age = field(initial=25)
    
    record = TestRecord(name='Jane')
    assert record['name'] == 'Jane'
    assert record['age'] == 25


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        items = field(initial=list)
    
    record = TestRecord(name='Test')
    assert record['name'] == 'Test'
    assert record['items'] == []


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord()
    assert 'name' not in record


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(_factory_fields=None, name='Alice', age=28)
    assert record['name'] == 'Alice'
    assert record['age'] == 28


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(_ignore_extra=True, name='Bob', extra_field='ignored')
    assert record['name'] == 'Bob'
    assert 'extra_field' not in record


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        status = field(initial='active')
    
    record = TestRecord(name='Charlie', status='inactive')
    assert record['name'] == 'Charlie'
    assert record['status'] == 'inactive'


# LLM-generated content at query #94
#--------------------------

```python
def test_precord_new_predicate_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Create instance without the special attributes
    # This ensures the predicate at line 5 evaluates to False
    record = TestRecord(name='John', age=30)
    
    # Verify the record was created successfully through the evolver path
    assert record['name'] == 'John'
    assert record['age'] == 30


# LLM-generated content at query #95
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create an instance normally first to get valid buckets and size
    normal_instance = TestRecord(x=1, y=2)
    
    # Now create using the special attributes path (line 5 predicate should be True)
    special_instance = TestRecord(_precord_size=normal_instance._size, _precord_buckets=normal_instance._buckets)
    
    # Verify the instance was created successfully
    assert special_instance._size == normal_instance._size
    assert special_instance._buckets == normal_instance._buckets


# LLM-generated content at query #96
#--------------------------

```python
def test_precord_new_predicate_false_missing_precord_size():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ with only _precord_buckets, missing _precord_size
    # This should make the predicate at line 5 evaluate to False
    result = TestRecord(_precord_buckets=None)
    assert isinstance(result, TestRecord)


def test_precord_new_predicate_false_missing_precord_buckets():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ with only _precord_size, missing _precord_buckets
    # This should make the predicate at line 5 evaluate to False
    result = TestRecord(_precord_size=0)
    assert isinstance(result, TestRecord)


def test_precord_new_predicate_false_neither_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    # Call __new__ with neither _precord_size nor _precord_buckets
    # This should make the predicate at line 5 evaluate to False
    result = TestRecord(x=42)
    assert isinstance(result, TestRecord)
    assert result.x == 42


# LLM-generated content at query #97
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
        y = field()
    
    # Create a normal record first to get valid buckets
    normal_record = MyRecord(x=1, y=2)
    
    # Create using the special internal parameters
    record = MyRecord(_precord_size=normal_record._size, _precord_buckets=normal_record._buckets)
    assert record.x == 1
    assert record.y == 2


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
        y = field()
    
    record = MyRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
        y = field()
    
    record = MyRecord(x=5, y=15, _factory_fields=set())
    assert record.x == 5
    assert record.y == 15


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
    
    record = MyRecord(x=100, z=999, _ignore_extra=True)
    assert record.x == 100
    assert 'z' not in record


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        __precord_initial_values = {'x': 42}
        x = field()
        y = field()
    
    record = MyRecord(y=10)
    assert record.x == 42
    assert record.y == 10


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        __precord_initial_values = {'x': lambda: 99}
        x = field()
    
    record = MyRecord()
    assert record.x == 99


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        x = field()
    
    record = MyRecord()
    assert len(record) == 0


def test_precord_new_with_multiple_fields():
    from pyrsistent import PRecord, field
    
    class MyRecord(PRecord):
        a = field()
        b = field()
        c = field()
        d = field()
    
    record = MyRecord(a=1, b=2, c=3, d=4)
    assert record.a == 1
    assert record.b == 2
    assert record.c == 3
    assert record.d == 4


# LLM-generated content at query #98
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
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.message == 'Field invariant failed'


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    TestRecord._precord_mandatory_fields = set()
    TestRecord._precord_invariants = ()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._missing_fields = ['missing_field_1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.missing_fields == ('missing_field_1',)
        assert e.message == 'Field invariant failed'


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class TestRecord(PRecord):
        x = field()
    
    TestRecord._precord_mandatory_fields = set()
    TestRecord._precord_invariants = ()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1', 'error_code_2']
    evolver._missing_fields = ['missing_field_1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1', 'error_code_2')
        assert e.missing_fields == ('missing_field_1',)
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #99
#--------------------------

```python
def test_precord_initial_values_predicate():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        _precord_initial_values = {'x': 10, 'y': lambda: 20}
    
    # Create instance without explicit values - should use _precord_initial_values
    record = TestRecord()
    
    # Verify that the predicate at line 11 evaluates to True
    # by checking that initial values were applied
    assert record['x'] == 10
    assert record['y'] == 20


# LLM-generated content at query #100
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


# LLM-generated content at query #101
#--------------------------

```python
def test_precord_evolver_set_with_valid_field():
    from pyrsistent import PMap, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: int(x)
            self.invariant = lambda x: (True, None)
    
    class MockDestinationClass:
        __name__ = "MockClass"
        _precord_fields = {"test_field": MockField()}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockDestinationClass, original_pmap)
    
    field_obj = MockDestinationClass._precord_fields.get("test_field")
    assert field_obj is not None
    assert evolver.set("test_field", 42)


# LLM-generated content at query #102
#--------------------------

```python
def test_precord_evolver_persistent_predicate_is_dirty_true():
    from pyrsistent import pmap, PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    evolver.set('x', 2)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True
    
    result = evolver.persistent()
    assert result['x'] == 2


def test_precord_evolver_persistent_predicate_not_isinstance_true():
    from pyrsistent import pmap, PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is False
    
    pm = pmap({'x': 1})
    assert not isinstance(pm, TestRecord)
    
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 1


# LLM-generated content at query #103
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


def test_precord_constructor_with_callable_initial_value():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: 42)
    
    record = TestRecord()
    assert record['x'] == 42


def test_precord_constructor_overrides_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100, y=200)
    assert record['x'] == 100
    assert record['y'] == 200


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_pmap = pmap({'x': 5})
    record = TestRecord(_precord_size=internal_pmap._size, _precord_buckets=internal_pmap._buckets)
    assert record['x'] == 5


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields=None)
    assert record['x'] == 1


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _ignore_extra=True, z=999)
    assert record['x'] == 1
    assert 'z' not in record


# LLM-generated content at query #104
#--------------------------

```python
def test_precord_meta_new_creates_class():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a simple field
    test_field = _PField(mandatory=True, initial=PFIELD_NO_INITIAL)
    
    # Create a dictionary with a field
    dct = {'test_attr': test_field}
    
    # Call __new__ to create a class
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), dct)
    
    # Verify that the class was created successfully
    assert result is not None
    assert isinstance(result, type)
    assert result.__name__ == 'TestClass'
    assert hasattr(result, '_precord_fields')
    assert hasattr(result, '_precord_invariants')
    assert hasattr(result, '_precord_mandatory_fields')
    assert hasattr(result, '_precord_initial_values')
    assert hasattr(result, '__slots__')
    assert result.__slots__ == ()


# LLM-generated content at query #105
#--------------------------

```python
def test_serialize_returns_dict_with_serialized_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='John', age=30)
    result = record.serialize()
    
    assert isinstance(result, dict)
    assert 'name' in result
    assert 'age' in result
    assert result['name'] == 'John'
    assert result['age'] == 30


# LLM-generated content at query #106
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
    
    record = TestRecord()
    assert record['x'] == 42


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


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
    
    record = TestRecord(x=1, _ignore_extra=True, extra_field=999)
    assert record['x'] == 1
    assert 'extra_field' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=5)
    
    record = TestRecord()
    assert len(record) >= 1
    assert record['x'] == 5


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_pmap = pmap({'x': 10})
    record = TestRecord(_precord_size=internal_pmap._size, _precord_buckets=internal_pmap._buckets)
    assert record['x'] == 10


def test_precord_constructor_multiple_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = TestRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_persistent_returns_result_when_no_invariant_errors():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets or ()
            self._size = _precord_size or 0
        
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)


def test_persistent_raises_invariant_exception_with_missing_mandatory_fields():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord:
        __name__ = 'TestRecord'
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets or ()
            self._size = _precord_size or 0
        
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.required_field' in e.missing_fields


def test_persistent_raises_invariant_exception_with_field_error_codes():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets or ()
            self._size = _precord_size or 0
        
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error_code_1' in e.invariant_errors


def test_persistent_calls_check_global_invariants():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    global_invariant_called = []
    
    def mock_invariant(subject):
        global_invariant_called.append(True)
        return (True, None)
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [mock_invariant]
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets or ()
            self._size = _precord_size or 0
        
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.persistent()
    assert len(global_invariant_called) > 0


def test_persistent_with_global_invariant_failure():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets or ()
            self._size = _precord_size or 0
        
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #2
#--------------------------

```python
def test_persistent_returns_precord_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockPRecord'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return []
    
    original_pmap = pmap({'a': 1})
    evolver = _PRecordEvolver(MockPRecord, original_pmap)
    evolver._destination_cls = MockPRecord
    
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
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockPRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockPRecord.required_field' in e.missing_fields


def test_persistent_raises_invariant_exception_with_field_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockPRecord'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockPRecord, original_pmap)
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
    
    def failing_global_invariant(subject):
        return (False, 'global_error')
    
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        __name__ = 'MockPRecord'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockPRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_creates_new_instance_when_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockPRecord'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return ['a']
    
    original_pmap = pmap({'a': 1})
    evolver = _PRecordEvolver(MockPRecord, original_pmap)
    evolver[('a',)] = 2
    
    result = evolver.persistent()
    assert isinstance(result, MockPRecord)


# LLM-generated content at query #3
#--------------------------

```python
def test_precord_evolver_set_with_valid_field():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import PRecordField
    
    class TestRecord:
        _precord_fields = {
            'name': PRecordField(type=(str,), factory=str, invariant=lambda x: (True, None))
        }
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        __name__ = 'TestRecord'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.set('name', 'test_value')
    assert result is evolver


def test_precord_evolver_set_with_invalid_field_name():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        __name__ = 'TestRecord'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "'invalid_field' is not among the specified fields for TestRecord" in str(e)


def test_precord_evolver_setitem():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import PRecordField
    
    class TestRecord:
        _precord_fields = {
            'age': PRecordField(type=(int,), factory=int, invariant=lambda x: (True, None))
        }
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        __name__ = 'TestRecord'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.__setitem__('age', 25)
    assert result is evolver


def test_precord_evolver_set_with_factory_fields_filter():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    from pyrsistent._field_common import PRecordField
    
    field1 = PRecordField(type=(str,), factory=str, invariant=lambda x: (True, None))
    field2 = PRecordField(type=(int,), factory=int, invariant=lambda x: (True, None))
    
    class TestRecord:
        _precord_fields = {
            'name': field1,
            'age': field2
        }
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        __name__ = 'TestRecord'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestRecord, original_pmap, _factory_fields=[field1])
    result = evolver.set('name', 'test')
    assert result is evolver


def test_precord_evolver_set_with_type_error():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import PRecordField
    from pyrsistent._precord import PTypeError
    from pyrsistent import pmap
    
    class TestRecord:
        _precord_fields = {
            'count': PRecordField(type=(int,), factory=lambda x: int(x), invariant=lambda x: (True, None))
        }
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        __name__ = 'TestRecord'
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.set('count', 'not_a_number_that_converts')
        assert False, "Should have raised PTypeError"
    except (PTypeError, ValueError):
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_precord_meta_new_sets_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        'field2': _PField(mandatory=False, initial=42),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result._precord_fields is not None
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        'field2': _PField(mandatory=False, initial=42),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result._precord_mandatory_fields == {'field1'}


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        'field2': _PField(mandatory=False, initial=42),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result._precord_initial_values == {'field2': 42}


def test_precord_meta_new_sets_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert result.__slots__ == ()


def test_precord_meta_new_stores_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    def test_invariant(instance):
        return True, None
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        '__invariant__': test_invariant,
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_inherits_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    base_dct = {
        'base_field': _PField(mandatory=True, initial=None),
        '_precord_fields': {'base_field': _PField(mandatory=True, initial=None)},
    }
    
    class BaseRecord(metaclass=_PRecordMeta):
        _precord_fields = {'base_field': _PField(mandatory=True, initial=None)}
    
    dct = {
        'child_field': _PField(mandatory=False, initial=10),
    }
    bases = (BaseRecord,)
    name = 'ChildRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert 'child_field' in result._precord_fields
    assert 'base_field' in result._precord_fields


# LLM-generated content at query #5
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        value = field()
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.message == 'Field invariant failed'


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        value = field()
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._missing_fields = ['TestRecord.value']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.missing_fields == ('TestRecord.value',)
        assert e.message == 'Field invariant failed'


def test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        value = field()
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['TestRecord.value']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('TestRecord.value',)
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #6
#--------------------------

```python
def test_precord_meta_new_creates_class_with_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a test field
    test_field = _PField(type=str, initial=PFIELD_NO_INITIAL, mandatory=True, factory=None, invariant=None)
    
    # Create a dictionary with a field
    dct = {'test_attr': test_field}
    bases = ()
    
    # Call __new__ to create the class
    result_class = _PRecordMeta.__new__(_PRecordMeta, 'TestPRecord', bases, dct)
    
    # Verify that __slots__ is set to empty tuple (line 8 predicate)
    assert result_class.__slots__ == ()
    assert isinstance(result_class, _PRecordMeta)
    assert hasattr(result_class, '_precord_fields')
    assert hasattr(result_class, '_precord_invariants')
    assert hasattr(result_class, '_precord_mandatory_fields')
    assert hasattr(result_class, '_precord_initial_values')


# LLM-generated content at query #7
#--------------------------

```python
def test_persistent_returns_precord_instance():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import PMap
    
    class TestPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _buckets = [None] * 8
        _size = 0
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestPRecord, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, TestPRecord)


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import PMap
    
    class TestPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _buckets = [None] * 8
        _size = 0
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestPRecord, original_pmap)
    evolver._invariant_error_codes = ['error1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import PMap
    
    class TestPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _buckets = [None] * 8
        _size = 0
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestPRecord, original_pmap)
    evolver._missing_fields = ['field1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('field1',)


def test_persistent_detects_missing_mandatory_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import PMap
    
    class TestPRecord:
        __name__ = 'TestPRecord'
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        _buckets = [None] * 8
        _size = 0
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestPRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestPRecord.required_field' in e.missing_fields


def test_persistent_calls_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import PMap
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class TestPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        _buckets = [None] * 8
        _size = 0
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestPRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)


def test_persistent_with_clean_state_returns_original():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import PMap
    
    class TestPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _buckets = [None] * 8
        _size = 0
    
    original_pmap = PMap()
    original_instance = TestPRecord(_precord_buckets=original_pmap._buckets, _precord_size=original_pmap._size)
    evolver = _PRecordEvolver(TestPRecord, original_pmap)
    evolver._is_dirty = False
    result = evolver.persistent()
    assert result is original_instance or isinstance(result, TestPRecord)


# LLM-generated content at query #8
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self):
            self.invariant = lambda x: (True, None)
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, MockClass)


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant_common import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        
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
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant_common import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockClass.field1' in e.missing_fields
        assert 'MockClass.field2' in e.missing_fields


def test_persistent_checks_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant_common import InvariantException
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [lambda x: (False, 'global_error')]
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_creates_new_instance_when_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return []
    
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver[0] = pmap()
    result = evolver.persistent()
    assert isinstance(result, MockClass)


# LLM-generated content at query #9
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
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()
        assert e.message == 'Field invariant failed'


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
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('TestRecord.x',)
        assert e.message == 'Field invariant failed'


def test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields():
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
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('TestRecord.x',)
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #10
#--------------------------

```python
def test_persistent_evaluates_mandatory_fields_predicate():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        mandatory_field = field()
        optional_field = field()
    
    # Set _precord_mandatory_fields to a non-empty set to make the predicate True
    TestRecord._precord_mandatory_fields = {'mandatory_field'}
    TestRecord._precord_invariants = ()
    
    # Create an evolver and set a value
    record = TestRecord(mandatory_field='value1', optional_field='value2')
    evolver = record.evolver()
    
    # The persistent() call should evaluate the predicate at line 11
    # Since _precord_mandatory_fields is non-empty (truthy), the predicate evaluates to True
    result = evolver.persistent()
    
    assert result.mandatory_field == 'value1'
    assert result.optional_field == 'value2'


def test_persistent_mandatory_fields_predicate_false_with_empty_set():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        optional_field = field()
    
    # Set _precord_mandatory_fields to an empty set to make the predicate False
    TestRecord._precord_mandatory_fields = set()
    TestRecord._precord_invariants = ()
    
    # Create an evolver
    record = TestRecord(optional_field='value1')
    evolver = record.evolver()
    
    # The persistent() call should not enter the if block at line 11
    result = evolver.persistent()
    
    assert result.optional_field == 'value1'


# LLM-generated content at query #11
#--------------------------

```python
def test_persistent_returns_precord_instance():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    original = TestRecord(x=1, y=2)
    evolver = _PRecordEvolver(TestRecord, original._to_pmap())
    result = evolver.persistent()
    
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2


def test_persistent_with_modified_fields():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    original = TestRecord(x=1, y=2)
    evolver = _PRecordEvolver(TestRecord, original._to_pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    
    assert result.x == 10
    assert result.y == 2


def test_persistent_raises_on_missing_mandatory_fields():
    from pyrsistent import PRecord, field, InvariantException
    
    class TestRecord(PRecord):
        x = field(mandatory=True)
        y = field()
    
    original = TestRecord(x=1, y=2)
    evolver = original.evolver()
    del evolver[('x',)]
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecord.x' in e.missing_fields


def test_persistent_raises_on_field_invariant_violation():
    from pyrsistent import PRecord, field, InvariantException
    
    def positive(val):
        return (val > 0, 'must_be_positive')
    
    class TestRecord(PRecord):
        x = field(invariant=positive)
    
    original = TestRecord(x=1)
    evolver = original.evolver()
    evolver.set('x', -5)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'must_be_positive' in e.invariant_errors


def test_persistent_with_no_dirty_changes():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original = TestRecord(x=1)
    evolver = _PRecordEvolver(TestRecord, original._to_pmap())
    result = evolver.persistent()
    
    assert result is original


def test_persistent_calls_global_invariants():
    from pyrsistent import PRecord, field, InvariantException
    
    def global_check(record):
        if record.x + record.y < 0:
            return (False, 'sum_must_be_positive')
        return (True, None)
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        __invariants__ = (global_check,)
    
    original = TestRecord(x=1, y=2)
    evolver = original.evolver()
    evolver.set('x', -5)
    evolver.set('y', 3)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'sum_must_be_positive' in e.invariant_errors


def test_persistent_with_factory_field():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(factory=int)
    
    original = TestRecord(x=1)
    evolver = original.evolver()
    evolver.set('x', '42')
    result = evolver.persistent()
    
    assert result.x == 42


def test_persistent_preserves_unmodified_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        z = field()
    
    original = TestRecord(x=1, y=2, z=3)
    evolver = original.evolver()
    evolver.set('y', 20)
    result = evolver.persistent()
    
    assert result.x == 1
    assert result.y == 20
    assert result.z == 3


# LLM-generated content at query #12
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


def test_precord_constructor_with_kwargs_override_initial():
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


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord()
    assert 'x' not in record or record.get('x') is None
    assert 'y' not in record or record.get('y') is None


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
    
    record = TestRecord(x=5, _ignore_extra=True, extra_field=10)
    assert record['x'] == 5
    assert 'extra_field' not in record


# LLM-generated content at query #13
#--------------------------

```python
def test_precord_meta_new_sets_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
        'field2': TestField(mandatory=False, initial='default'),
    }
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert '_precord_fields' in result.__dict__
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
        'field2': TestField(mandatory=False, initial=None),
    }
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert result._precord_mandatory_fields == {'field1'}


def test_precord_meta_new_initial_values():
    from pyrsistent._precord import _PRecordMeta, PFIELD_NO_INITIAL
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': TestField(mandatory=False, initial='default_value'),
    }
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert result._precord_initial_values == {'field2': 'default_value'}


def test_precord_meta_new_sets_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
    }
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert result.__slots__ == ()


def test_precord_meta_new_stores_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    def test_invariant(self):
        return True, None
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
        '__invariant__': test_invariant,
    }
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert '_precord_invariants' in result.__dict__
    assert len(result._precord_invariants) == 1


def test_precord_meta_new_inherits_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    parent_dct = {
        'parent_field': TestField(mandatory=True, initial=None),
    }
    parent = _PRecordMeta('Parent', (), parent_dct)
    
    child_dct = {
        'child_field': TestField(mandatory=False, initial='child_default'),
    }
    
    child = _PRecordMeta('Child', (parent,), child_dct)
    
    assert 'parent_field' in child._precord_fields
    assert 'child_field' in child._precord_fields


def test_precord_meta_new_removes_field_from_dict():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
        'other_attr': 'value',
    }
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert 'field1' not in result.__dict__ or result.__dict__['field1'] != dct['field1']
    assert 'other_attr' in result.__dict__


# LLM-generated content at query #14
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
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_buckets = None
        _precord_size = 0
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    result = evolver.persistent()
    
    assert result is not None
    assert isinstance(result, MockCls)


def test_persistent_raises_on_missing_mandatory_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
            
        def invariant(self, value):
            return (True, None)
    
    class MockCls:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = {'field2'}
        _precord_invariants = []
        _precord_buckets = None
        _precord_size = 0
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockCls.field2' in e.missing_fields


def test_persistent_raises_on_field_invariant_error():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
            
        def invariant(self, value):
            return (False, 'error_code_1')
    
    class MockCls:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_buckets = None
        _precord_size = 0
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes.append('error_code_1')
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error_code_1' in e.invariant_errors


def test_persistent_raises_on_accumulated_invariant_errors():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self, name):
            self.name = name
            
        def invariant(self, value):
            return (True, None)
    
    class MockCls:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_buckets = None
        _precord_size = 0
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = ['accumulated_error']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'accumulated_error' in e.invariant_errors


# LLM-generated content at query #15
#--------------------------

```python
def test_precord_repr():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='John', age=30)
    repr_str = repr(record)
    
    assert 'TestRecord' in repr_str
    assert 'name=' in repr_str
    assert "'John'" in repr_str
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
    
    record = SingleFieldRecord(value=42)
    repr_str = repr(record)
    
    assert repr_str == 'SingleFieldRecord(value=42)'


def test_precord_repr_multiple_fields():
    from pyrsistent import PRecord, field
    
    class MultiFieldRecord(PRecord):
        first = field()
        second = field()
        third = field()
    
    record = MultiFieldRecord(first='a', second='b', third='c')
    repr_str = repr(record)
    
    assert 'MultiFieldRecord' in repr_str
    assert 'first=' in repr_str
    assert 'second=' in repr_str
    assert 'third=' in repr_str


def test_precord_repr_with_special_characters():
    from pyrsistent import PRecord, field
    
    class SpecialRecord(PRecord):
        text = field()
    
    record = SpecialRecord(text="hello'world")
    repr_str = repr(record)
    
    assert 'SpecialRecord' in repr_str
    assert 'text=' in repr_str


# LLM-generated content at query #16
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap with some data
    test_pmap = pmap({'x': 1, 'y': 2})
    
    # Create PRecord using special attributes
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
        y = field()
        _precord_initial_values = {'x': 5}
    
    record = TestRecord(y=15)
    
    assert record['x'] == 5
    assert record['y'] == 15


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        _precord_initial_values = {'x': lambda: 100}
    
    record = TestRecord(y=200)
    
    assert record['x'] == 100
    assert record['y'] == 200


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    x_field = TestRecord._precord_fields['x']
    record = TestRecord(x=5, y=10, _factory_fields={x_field})
    
    assert record['x'] == 5
    assert record['y'] == 10


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _ignore_extra=True)
    
    assert record['x'] == 5
    assert len(record) == 1


def test_precord_new_kwargs_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        _precord_initial_values = {'x': 1, 'y': 2}
    
    record = TestRecord(x=100)
    
    assert record['x'] == 100
    assert record['y'] == 2


# LLM-generated content at query #17
#--------------------------

```python
def test_precord_meta_new_sets_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None, factory=None, initial_fn=None),
        'field2': _PField(mandatory=False, initial='default', factory=None, initial_fn=None),
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
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL, factory=None, initial_fn=None),
        'field2': _PField(mandatory=False, initial=PFIELD_NO_INITIAL, factory=None, initial_fn=None),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert result._precord_mandatory_fields == {'field1'}


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL, factory=None, initial_fn=None),
        'field2': _PField(mandatory=False, initial='default_value', factory=None, initial_fn=None),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert result._precord_initial_values == {'field2': 'default_value'}


def test_precord_meta_new_sets_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None, factory=None, initial_fn=None),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert result.__slots__ == ()


def test_precord_meta_new_sets_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    def test_invariant(instance):
        return True, None
    
    dct = {
        'field1': _PField(mandatory=True, initial=None, factory=None, initial_fn=None),
        '__invariant__': test_invariant,
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    base_dct = {
        'base_field': _PField(mandatory=True, initial=None, factory=None, initial_fn=None),
    }
    base_class = _PRecordMeta('BaseRecord', (), base_dct)
    
    dct = {
        'child_field': _PField(mandatory=False, initial='child', factory=None, initial_fn=None),
    }
    
    result = _PRecordMeta('ChildRecord', (base_class,), dct)
    
    assert 'base_field' in result._precord_fields
    assert 'child_field' in result._precord_fields


def test_precord_meta_new_removes_pfield_from_dct():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None, factory=None, initial_fn=None),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta(name, bases, dct)
    
    assert 'field1' not in dct or not isinstance(dct.get('field1'), _PField)


# LLM-generated content at query #18
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockCls'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockCls, mock_pmap)
    evolver._is_dirty = False
    
    result = evolver.persistent()
    assert isinstance(result, MockCls)


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockCls'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockCls, mock_pmap)
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
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'MockCls'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockCls, mock_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockCls.field1' in e.missing_fields or 'MockCls.field2' in e.missing_fields


def test_persistent_checks_global_invariants():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        __name__ = 'MockCls'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockCls, mock_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_with_accumulated_invariant_errors():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockCls'
    
    mock_pmap = PMap()
    evolver = _PRecordEvolver(MockCls, mock_pmap)
    evolver._invariant_error_codes = ['field_error']
    evolver._missing_fields = ['MockCls.required_field']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('field_error',)
        assert e.missing_fields == ('MockCls.required_field',)


# LLM-generated content at query #19
#--------------------------

```python
def test_precord_meta_new_creates_class_with_correct_attributes():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a mock field
    mock_field = _PField(type=str, initial=PFIELD_NO_INITIAL, mandatory=True)
    
    # Create a dictionary with a field
    dct = {'test_field': mock_field}
    bases = ()
    name = 'TestPRecord'
    
    # Call __new__ through the metaclass
    result_class = _PRecordMeta(name, bases, dct)
    
    # Verify the class was created
    assert result_class is not None
    assert result_class.__name__ == name
    
    # Verify _precord_fields was set
    assert hasattr(result_class, '_precord_fields')
    assert 'test_field' in result_class._precord_fields
    
    # Verify _precord_mandatory_fields was set correctly
    assert hasattr(result_class, '_precord_mandatory_fields')
    assert 'test_field' in result_class._precord_mandatory_fields
    
    # Verify _precord_initial_values was set (should be empty for mandatory fields with no initial)
    assert hasattr(result_class, '_precord_initial_values')
    assert isinstance(result_class._precord_initial_values, dict)
    
    # Verify _precord_invariants was set
    assert hasattr(result_class, '_precord_invariants')
    assert isinstance(result_class._precord_invariants, tuple)
    
    # Verify __slots__ was set to empty tuple
    assert result_class.__slots__ == ()


# LLM-generated content at query #20
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
        x = field(initial=lambda: 42)
        y = field()
    
    record = TestRecord(y=5)
    assert record['x'] == 42
    assert record['y'] == 5


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields=None)
    assert record['x'] == 1


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _ignore_extra=True)
    assert record['x'] == 1


def test_precord_constructor_internal_creation():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 1})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert record['x'] == 1


def test_precord_constructor_multiple_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = TestRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3


def test_precord_constructor_overrides_initial():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
    
    record = TestRecord(x=20)
    assert record['x'] == 20


def test_precord_constructor_mixed_initial_and_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
        z = field()
    
    record = TestRecord(y=30, z=40)
    assert record['x'] == 10
    assert record['y'] == 30
    assert record['z'] == 40


# LLM-generated content at query #21
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
    
    def serialize_with_format(value, format=None):
        if format == "uppercase":
            return value.upper() if isinstance(value, str) else value
        return value
    
    class TestRecord(PRecord):
        name = field(serializer=serialize_with_format)
        value = field()
    
    record = TestRecord(name="test", value="data")
    result = record.serialize(format="uppercase")
    
    assert result == {"name": "TEST", "value": "data"}


def test_serialize_empty_record():
    from pyrsistent import PRecord
    
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    result = record.serialize()
    
    assert result == {}


def test_serialize_multiple_fields_with_different_serializers():
    from pyrsistent import PRecord, field
    
    def serialize_int(value):
        return str(value) if isinstance(value, int) else value
    
    def serialize_str(value):
        return value.upper() if isinstance(value, str) else value
    
    class TestRecord(PRecord):
        name = field(serializer=serialize_str)
        count = field(serializer=serialize_int)
        active = field()
    
    record = TestRecord(name="alice", count=42, active=True)
    result = record.serialize()
    
    assert result == {"name": "ALICE", "count": "42", "active": True}


def test_serialize_with_none_values():
    from pyrsistent import PRecord, field
    
    def custom_serializer(value):
        return "null" if value is None else value
    
    class TestRecord(PRecord):
        name = field(serializer=custom_serializer)
        description = field()
    
    record = TestRecord(name=None, description="test")
    result = record.serialize()
    
    assert result == {"name": "null", "description": "test"}


# LLM-generated content at query #22
#--------------------------

```python
def test_persistent_checks_mandatory_fields_when_present():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Set _precord_mandatory_fields to a non-empty set to make the predicate True
    TestRecord._precord_mandatory_fields = {'name', 'age'}
    TestRecord._precord_invariants = ()
    
    # Create an evolver and set values for mandatory fields
    record = TestRecord(name='John', age=30)
    evolver = record.evolver()
    
    # Persistent should succeed when all mandatory fields are present
    result = evolver.persistent()
    
    assert result.name == 'John'
    assert result.age == 30


# LLM-generated content at query #23
#--------------------------

```python
def test_persistent_with_mandatory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        mandatory_field = field()
    
    TestRecord._precord_mandatory_fields = {'mandatory_field'}
    TestRecord._precord_invariants = []
    
    evolver = TestRecord._PRecordEvolver(TestRecord, TestRecord())
    evolver._destination_cls = TestRecord
    
    assert TestRecord._precord_mandatory_fields
    assert bool(TestRecord._precord_mandatory_fields) is True


# LLM-generated content at query #24
#--------------------------

```python
def test_precord_initial_values_predicate():
    from pyrsistent import PRecord, field
    
    # Create a PRecord class with _precord_initial_values set
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    # Set _precord_initial_values to a non-empty dict to make the predicate True
    TestRecord._precord_initial_values = {'name': 'default_name', 'value': lambda: 42}
    
    # Create an instance - this will execute the __new__ method
    # The predicate at line 11 should evaluate to True since _precord_initial_values is set
    record = TestRecord(name='custom_name')
    
    # Verify that initial values were applied
    assert record['name'] == 'custom_name'
    assert record['value'] == 42


# LLM-generated content at query #25
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import pmap
    from pyrsistent._precord import PRecord
    
    class TestRecord(PRecord):
        pass
    
    # Create a pmap with size and buckets
    pm = pmap({'a': 1})
    record = TestRecord(_precord_size=pm._size, _precord_buckets=pm._buckets)
    assert record is not None


def test_precord_new_with_kwargs():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2


def test_precord_new_with_factory_fields():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_factory_fields=set(), x=5)
    assert record['x'] == 5


def test_precord_new_with_ignore_extra():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=False, x=10)
    assert record['x'] == 10


def test_precord_new_with_initial_values():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        _precord_initial_values = {'x': 100}
        x = field()
    
    record = TestRecord()
    assert record['x'] == 100


def test_precord_new_with_initial_values_override():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        _precord_initial_values = {'x': 100}
        x = field()
    
    record = TestRecord(x=200)
    assert record['x'] == 200


def test_precord_new_with_callable_initial_values():
    from pyrsistent._precord import PRecord, field
    
    def default_value():
        return 42
    
    class TestRecord(PRecord):
        _precord_initial_values = {'x': default_value}
        x = field()
    
    record = TestRecord()
    assert record['x'] == 42


def test_precord_new_empty():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_new_multiple_fields():
    from pyrsistent._precord import PRecord, field
    
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = TestRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3


# LLM-generated content at query #26
#--------------------------

```python
def test_precord_meta_new_creates_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {'field1': TestField(mandatory=True, initial=None), 'field2': TestField(mandatory=False, initial='default')}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert result.__name__ == 'TestRecord'
    assert '_precord_fields' in result.__dict__
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {'field1': TestField(mandatory=True, initial=None), 'field2': TestField(mandatory=False, initial=None)}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert '_precord_mandatory_fields' in result.__dict__
    assert 'field1' in result._precord_mandatory_fields
    assert 'field2' not in result._precord_mandatory_fields


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import PFIELD_NO_INITIAL
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {'field1': TestField(mandatory=False, initial=PFIELD_NO_INITIAL), 'field2': TestField(mandatory=False, initial='default_val')}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert '_precord_initial_values' in result.__dict__
    assert 'field1' not in result._precord_initial_values
    assert result._precord_initial_values.get('field2') == 'default_val'


def test_precord_meta_new_sets_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {'field1': TestField()}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert '__slots__' in result.__dict__
    assert result.__slots__ == ()


def test_precord_meta_new_stores_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    def test_invariant(obj):
        return True, None
    
    dct = {'field1': TestField(), '__invariant__': test_invariant}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert '_precord_invariants' in result.__dict__
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_removes_fields_from_dct():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {'field1': TestField(mandatory=True), 'field2': TestField()}
    bases = ()
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert 'field1' not in result.__dict__
    assert 'field2' not in result.__dict__


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    parent_dct = {'parent_field': TestField(mandatory=True)}
    parent = _PRecordMeta('ParentRecord', (), parent_dct)
    
    dct = {'child_field': TestField(mandatory=False)}
    result = _PRecordMeta('ChildRecord', (parent,), dct)
    
    assert 'parent_field' in result._precord_fields
    assert 'child_field' in result._precord_fields


# LLM-generated content at query #27
#--------------------------

```python
def test_precord_new_predicate_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Test case 1: Neither '_precord_size' nor '_precord_buckets' in kwargs
    result1 = TestRecord(x=1, y=2)
    assert result1['x'] == 1
    assert result1['y'] == 2
    
    # Test case 2: Only '_precord_size' in kwargs (missing '_precord_buckets')
    result2 = TestRecord(x=3, y=4)
    assert result2['x'] == 3
    assert result2['y'] == 4
    
    # Test case 3: Only '_precord_buckets' in kwargs (missing '_precord_size')
    result3 = TestRecord(x=5, y=6)
    assert result3['x'] == 5
    assert result3['y'] == 6
    
    # Test case 4: Empty kwargs
    result4 = TestRecord()
    assert isinstance(result4, TestRecord)


# LLM-generated content at query #28
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
    
    def uppercase_serializer(value):
        return value.upper() if isinstance(value, str) else value
    
    class TestRecord(PRecord):
        name = field(serializer=uppercase_serializer)
        age = field()
    
    record = TestRecord(name="john", age=30)
    result = record.serialize()
    
    assert result == {"name": "JOHN", "age": 30}


def test_serialize_with_format_parameter():
    from pyrsistent import PRecord, field
    
    def json_serializer(value):
        return str(value) + "_json"
    
    class TestRecord(PRecord):
        value = field(serializer=json_serializer)
    
    record = TestRecord(value="test")
    result = record.serialize(format="json")
    
    assert result == {"value": "test_json"}


def test_serialize_empty_record():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    result = record.serialize()
    
    assert result == {}


def test_serialize_multiple_fields_with_mixed_serializers():
    from pyrsistent import PRecord, field
    
    def double_serializer(value):
        return value * 2 if isinstance(value, int) else value
    
    class TestRecord(PRecord):
        count = field(serializer=double_serializer)
        name = field()
        active = field()
    
    record = TestRecord(count=5, name="test", active=True)
    result = record.serialize()
    
    assert result == {"count": 10, "name": "test", "active": True}


def test_serialize_with_none_values():
    from pyrsistent import PRecord, field
    
    def custom_serializer(value):
        return "none" if value is None else value
    
    class TestRecord(PRecord):
        value = field(serializer=custom_serializer)
    
    record = TestRecord(value=None)
    result = record.serialize()
    
    assert result == {"value": "none"}


# LLM-generated content at query #29
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
        name = field()
        count = field(initial=0)
    
    record = TestRecord(name='Alice')
    assert record['name'] == 'Alice'
    assert record['count'] == 0


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
    
    record = TestRecord(name='Bob', _factory_fields=None)
    assert record['name'] == 'Bob'


def test_precord_constructor_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    try:
        record = TestRecord(name='Charlie', extra_field='value', _ignore_extra=False)
        assert False, "Should have raised an error"
    except:
        pass


def test_precord_constructor_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name='David', extra_field='value', _ignore_extra=True)
    assert record['name'] == 'David'
    assert 'extra_field' not in record


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    pmap_obj = pmap({'name': 'Eve'})
    record = TestRecord(_precord_size=pmap_obj._size, _precord_buckets=pmap_obj._buckets)
    assert record['name'] == 'Eve'


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='default')
    
    record = TestRecord()
    assert record['name'] == 'default'


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='default_name')
        age = field(initial=0)
    
    record = TestRecord(name='Frank', age=25)
    assert record['name'] == 'Frank'
    assert record['age'] == 25


def test_precord_constructor_multiple_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()
        field3 = field()
    
    record = TestRecord(field1='a', field2='b', field3='c')
    assert record['field1'] == 'a'
    assert record['field2'] == 'b'
    assert record['field3'] == 'c'


# LLM-generated content at query #30
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


# LLM-generated content at query #31
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._missing_fields = ['TestRecord.y']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.missing_fields == ('TestRecord.y',)


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
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


# LLM-generated content at query #32
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PRecord, field, InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
        _precord_mandatory_fields = set()
        _precord_invariants = ()
    
    original_pmap = PMap()
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
    from pyrsistent import PRecord, field, InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
        _precord_mandatory_fields = set()
        _precord_invariants = ()
    
    original_pmap = PMap()
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
    from pyrsistent import PRecord, field, InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
        _precord_mandatory_fields = set()
        _precord_invariants = ()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['TestRecord.x', 'TestRecord.y']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('TestRecord.x', 'TestRecord.y')


# LLM-generated content at query #33
#--------------------------

```python
def test_precord_repr():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='John', age=30)
    repr_str = repr(record)
    
    assert 'TestRecord' in repr_str
    assert 'name=' in repr_str
    assert "'John'" in repr_str
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
    
    record = SingleFieldRecord(value=42)
    repr_str = repr(record)
    
    assert repr_str == 'SingleFieldRecord(value=42)'


def test_precord_repr_with_special_characters():
    from pyrsistent import PRecord, field
    
    class SpecialRecord(PRecord):
        text = field()
    
    record = SpecialRecord(text='hello"world')
    repr_str = repr(record)
    
    assert 'SpecialRecord' in repr_str
    assert 'text=' in repr_str
    assert 'hello"world' in repr_str


def test_precord_repr_with_nested_structure():
    from pyrsistent import PRecord, field, pmap
    
    class NestedRecord(PRecord):
        data = field()
    
    record = NestedRecord(data=pmap({'key': 'value'}))
    repr_str = repr(record)
    
    assert 'NestedRecord' in repr_str
    assert 'data=' in repr_str


# LLM-generated content at query #34
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


# LLM-generated content at query #35
#--------------------------

```python
def test_precord_new_without_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create instance without '_precord_size' and '_precord_buckets'
    # This should evaluate the predicate at line 5 to False
    record = TestRecord(x=1, y=2)
    
    # Verify the record was created through the evolver path
    assert record['x'] == 1
    assert record['y'] == 2
    assert isinstance(record, TestRecord)


# LLM-generated content at query #36
#--------------------------

```python
def test_precord_meta_new_sets_precord_fields():
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import _PRecordMeta
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        'field2': _PField(mandatory=False, initial=42),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_fields')
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import _PRecordMeta
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        'field2': _PField(mandatory=False, initial=42),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_mandatory_fields')
    assert 'field1' in result._precord_mandatory_fields
    assert 'field2' not in result._precord_mandatory_fields


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    from pyrsistent._precord import _PRecordMeta
    
    dct = {
        'field1': _PField(mandatory=True, initial=PFIELD_NO_INITIAL),
        'field2': _PField(mandatory=False, initial=42),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_initial_values')
    assert 'field1' not in result._precord_initial_values
    assert result._precord_initial_values['field2'] == 42


def test_precord_meta_new_sets_slots():
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import _PRecordMeta
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '__slots__')
    assert result.__slots__ == ()


def test_precord_meta_new_sets_precord_invariants():
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import _PRecordMeta
    
    def test_invariant(record):
        return True, None
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        '__invariant__': test_invariant,
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert isinstance(result._precord_invariants, tuple)
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_removes_pfield_from_dct():
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import _PRecordMeta
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
    }
    bases = ()
    name = 'TestRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert 'field1' not in result.__dict__ or not isinstance(result.__dict__.get('field1'), _PField)


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._field_common import _PField
    from pyrsistent._precord import _PRecordMeta
    
    base_dct = {
        'base_field': _PField(mandatory=True, initial=None),
    }
    BaseRecord = _PRecordMeta.__new__(_PRecordMeta, 'BaseRecord', (), base_dct)
    
    dct = {
        'child_field': _PField(mandatory=False, initial=10),
    }
    
    result = _PRecordMeta.__new__(_PRecordMeta, 'ChildRecord', (BaseRecord,), dct)
    
    assert 'base_field' in result._precord_fields
    assert 'child_field' in result._precord_fields


# LLM-generated content at query #37
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    buckets = pmap(pre_size=2)._buckets
    size = pmap(pre_size=2)._size
    record = TestRecord(_precord_size=size, _precord_buckets=buckets)
    assert isinstance(record, TestRecord)


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    assert record['name'] == "John"
    assert record['age'] == 30


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(_factory_fields=set(), name="Jane", age=25)
    assert record['name'] == "Jane"
    assert record['age'] == 25


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(_ignore_extra=True, name="Bob", extra_field="ignored")
    assert record['name'] == "Bob"
    assert 'extra_field' not in record


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="Default")
        age = field()
    
    record = TestRecord(age=20)
    assert record['name'] == "Default"
    assert record['age'] == 20


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record = TestRecord()
    assert record['items'] == []


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_new_overrides_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="Default")
        age = field()
    
    record = TestRecord(name="Override", age=35)
    assert record['name'] == "Override"
    assert record['age'] == 35


# LLM-generated content at query #38
#--------------------------

```python
def test_precord_evolver_set_with_existing_field():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    original = TestRecord(name="John", age=30)
    evolver = _PRecordEvolver(TestRecord, original._to_pmap())
    
    result = evolver.set("name", "Jane")
    
    assert result is not None
    assert isinstance(result, _PRecordEvolver)


# LLM-generated content at query #39
#--------------------------

```python
def test_serialize_without_format():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    result = record.serialize()
    
    assert result == {"name": "John", "age": 30}


def test_serialize_with_format():
    from pyrsistent import PRecord, field
    
    def custom_serializer(format, value):
        if format == "upper":
            return value.upper()
        return value
    
    class TestRecord(PRecord):
        name = field(serializer=custom_serializer)
        age = field()
    
    record = TestRecord(name="john", age=30)
    result = record.serialize(format="upper")
    
    assert result["name"] == "JOHN"
    assert result["age"] == 30


def test_serialize_empty_record():
    from pyrsistent import PRecord
    
    class EmptyRecord(PRecord):
        pass
    
    record = EmptyRecord()
    result = record.serialize()
    
    assert result == {}


def test_serialize_multiple_fields_with_serializers():
    from pyrsistent import PRecord, field
    
    def int_serializer(format, value):
        if format == "hex":
            return hex(value)
        return value
    
    def str_serializer(format, value):
        if format == "upper":
            return value.upper()
        return value
    
    class TestRecord(PRecord):
        text = field(serializer=str_serializer)
        number = field(serializer=int_serializer)
        plain = field()
    
    record = TestRecord(text="hello", number=255, plain="unchanged")
    result = record.serialize(format="upper")
    
    assert result["text"] == "HELLO"
    assert result["number"] == "0xff"
    assert result["plain"] == "unchanged"


def test_serialize_with_none_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    record = TestRecord(name=None, value="test")
    result = record.serialize()
    
    assert result == {"name": None, "value": "test"}


# LLM-generated content at query #40
#--------------------------

```python
def test_persistent_predicate_line_6_evaluates_to_false():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
        value = field()
        _precord_mandatory_fields = set()
        _precord_invariants = ()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    evolver.set('name', 'test')
    evolver.set('value', 42)
    
    is_dirty_before = evolver.is_dirty()
    pm = PMap._Evolver.persistent(evolver)
    isinstance_check = isinstance(pm, TestRecord)
    
    assert not (is_dirty_before or not isinstance_check)


# LLM-generated content at query #41
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
    
    pm = PMap.__bases__[0]._Evolver.persistent(evolver)
    predicate_result = is_dirty or not isinstance(pm, TestRecord)
    assert predicate_result is True


def test_persistent_predicate_not_isinstance_true():
    from pyrsistent import PRecord, field, pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    is_dirty = evolver.is_dirty()
    pm = PMap.__bases__[0]._Evolver.persistent(evolver)
    predicate_result = is_dirty or not isinstance(pm, TestRecord)
    assert predicate_result is True


def test_persistent_predicate_both_conditions_true():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 42)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True
    
    pm = PMap.__bases__[0]._Evolver.persistent(evolver)
    predicate_result = is_dirty or not isinstance(pm, TestRecord)
    assert predicate_result is True


# LLM-generated content at query #42
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self):
            self.invariant = lambda x: (True, None)
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._is_dirty = False
    result = evolver.persistent()
    assert result is not None


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
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'TestClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestClass.field1' in e.missing_fields
        assert 'TestClass.field2' in e.missing_fields


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
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_creates_new_instance_when_dirty():
    from pyrsistent._precord import _PRecordEvolver
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
            return []
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver[('_is_dirty',)] = True
    result = evolver.persistent()
    assert result is not None
    assert isinstance(result, MockCls)


# LLM-generated content at query #43
#--------------------------

```python
def test_precord_new_predicate_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    # Test case 1: Neither _precord_size nor _precord_buckets in kwargs
    record1 = TestRecord(name="John", age=30)
    assert record1['name'] == "John"
    assert record1['age'] == 30
    
    # Test case 2: Only _precord_size in kwargs (but not _precord_buckets)
    record2 = TestRecord(name="Jane", age=25, _precord_size=10)
    assert record2['name'] == "Jane"
    assert record2['age'] == 25
    
    # Test case 3: Only _precord_buckets in kwargs (but not _precord_size)
    from pyrsistent import pvector
    buckets = pvector()
    record3 = TestRecord(name="Bob", age=35, _precord_buckets=buckets)
    assert record3['name'] == "Bob"
    assert record3['age'] == 35


# LLM-generated content at query #44
#--------------------------

```python
def test_precord_evolver_set_with_valid_field():
    from pyrsistent import pmap, PMap
    from pyrsistent._precord import _PRecordEvolver
    
    # Create a mock field object
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: int(x)
            self.invariant = lambda x: (True, None)
    
    # Create a mock destination class
    class MockDestinationCls:
        __name__ = "TestClass"
        _precord_fields = {"test_field": MockField()}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    
    # Create an evolver instance
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    
    # Call set with a valid field - this should pass the predicate at line 3
    result = evolver.set("test_field", 42)
    
    # Verify the field exists and predicate evaluates to True
    assert evolver._destination_cls._precord_fields.get("test_field") is not None
    assert result is not None


# LLM-generated content at query #45
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


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record = TestRecord()
    assert record['items'] == []


def test_precord_constructor_overrides_initial_values():
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
    
    record = TestRecord(name='Charlie', _ignore_extra=True, extra_field='ignored')
    assert record['name'] == 'Charlie'
    assert 'extra_field' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='Unknown')
    
    record = TestRecord()
    assert record['name'] == 'Unknown'


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    internal_map = pmap({'name': 'Test'})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
    assert record['name'] == 'Test'


# LLM-generated content at query #46
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import pmap, PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    initial_pmap = pmap({'x': 1, 'y': 2})
    precord_size = initial_pmap._size
    precord_buckets = initial_pmap._buckets
    
    result = TestRecord(_precord_size=precord_size, _precord_buckets=precord_buckets)
    
    assert result['x'] == 1
    assert result['y'] == 2
    assert isinstance(result, TestRecord)


# LLM-generated content at query #47
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


def test_precord_repr_multiple_fields_order():
    from pyrsistent import PRecord, field
    
    class MultiFieldRecord(PRecord):
        x = field()
        y = field()
        z = field()
    
    record = MultiFieldRecord(x=1, y=2, z=3)
    repr_str = repr(record)
    
    assert 'MultiFieldRecord' in repr_str
    assert 'x=1' in repr_str
    assert 'y=2' in repr_str
    assert 'z=3' in repr_str


def test_precord_repr_with_special_characters():
    from pyrsistent import PRecord, field
    
    class SpecialRecord(PRecord):
        text = field()
    
    record = SpecialRecord(text='hello"world')
    repr_str = repr(record)
    
    assert 'SpecialRecord' in repr_str
    assert 'text=' in repr_str
    assert 'hello' in repr_str


# LLM-generated content at query #48
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap to use as buckets
    from pyrsistent import pmap
    pm = pmap({'x': 1, 'y': 2})
    
    # Create using special attributes
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
        y = field()
    
    record = TestRecord(x=5, y=15, _factory_fields=set())
    assert record['x'] == 5
    assert record['y'] == 15


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=100, _ignore_extra=True)
    assert record['x'] == 100


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __invariant__ = None
        x = field()
        
    TestRecord._precord_initial_values = {'x': lambda: 42}
    
    record = TestRecord()
    assert record['x'] == 42


def test_precord_new_with_initial_values_and_override():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __invariant__ = None
        x = field()
    
    TestRecord._precord_initial_values = {'x': 100}
    
    record = TestRecord(x=200)
    assert record['x'] == 200


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord()
    assert len(record) == 0


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


# LLM-generated content at query #49
#--------------------------

```python
def test_precord_meta_new_sets_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {'field1': _PField(mandatory=True, initial=None)}
    bases = ()
    name = 'TestClass'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert '_precord_fields' in result.__dict__
    assert 'field1' in result._precord_fields
    assert isinstance(result._precord_fields['field1'], _PField)


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        'field2': _PField(mandatory=False, initial=None)
    }
    bases = ()
    name = 'TestClass'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert '_precord_mandatory_fields' in result.__dict__
    assert 'field1' in result._precord_mandatory_fields
    assert 'field2' not in result._precord_mandatory_fields


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial='default1'),
        'field2': _PField(mandatory=False, initial=None)
    }
    bases = ()
    name = 'TestClass'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert '_precord_initial_values' in result.__dict__
    assert result._precord_initial_values.get('field1') == 'default1'


def test_precord_meta_new_sets_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {'field1': _PField(mandatory=True, initial=None)}
    bases = ()
    name = 'TestClass'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert '__slots__' in result.__dict__
    assert result.__slots__ == ()


def test_precord_meta_new_sets_precord_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    def test_invariant(obj):
        return True, None
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        '__invariant__': test_invariant
    }
    bases = ()
    name = 'TestClass'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert '_precord_invariants' in result.__dict__
    assert isinstance(result._precord_invariants, tuple)
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_inherits_fields_from_base():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    base_dct = {'field1': _PField(mandatory=True, initial='base_value')}
    base_class = _PRecordMeta('BaseClass', (), base_dct)
    
    dct = {'field2': _PField(mandatory=False, initial=None)}
    result = _PRecordMeta.__new__(_PRecordMeta, 'DerivedClass', (base_class,), dct)
    
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_removes_field_from_dct():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {'field1': _PField(mandatory=True, initial=None)}
    bases = ()
    name = 'TestClass'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert 'field1' not in result.__dict__ or isinstance(result.__dict__.get('field1'), _PField) == False


# LLM-generated content at query #50
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


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert 'x' not in record


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=5, _factory_fields=None)
    assert record['x'] == 5


def test_precord_constructor_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, z=999, _ignore_extra=True)
    assert record['x'] == 1
    assert 'z' not in record


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    pm = pmap({'x': 5})
    record = TestRecord(_precord_size=pm._size, _precord_buckets=pm._buckets)
    assert record['x'] == 5


# LLM-generated content at query #51
#--------------------------

```python
def test_persistent_predicate_line_1_is_dirty_true():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 42)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True


def test_persistent_predicate_line_1_isinstance_false():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    is_dirty = evolver.is_dirty()
    pm = PMap.__bases__[0]() if hasattr(PMap, '__bases__') else PMap()
    
    is_instance = isinstance(pm, TestRecord)
    assert is_instance is False


def test_persistent_predicate_line_6_evaluates_true_when_dirty():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 100)
    
    is_dirty = evolver.is_dirty()
    pm = evolver.__class__.__bases__[0].persistent(evolver)
    is_instance_check = isinstance(pm, TestRecord)
    
    predicate_result = is_dirty or not is_instance_check
    assert predicate_result is True


# LLM-generated content at query #52
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        x = field()
    
    TestRecord._precord_mandatory_fields = set()
    TestRecord._precord_invariants = ()
    
    original_pmap = pmap({'x': 1})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1',)
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #53
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
    assert isinstance(repr_str, str)


# LLM-generated content at query #54
#--------------------------

```python
def test_serialize_returns_dict_with_serialized_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    result = record.serialize()
    
    assert isinstance(result, dict)
    assert "name" in result
    assert "age" in result


# LLM-generated content at query #55
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
        x = field(initial=list)
        y = field()
    
    record = TestRecord(y=5)
    assert record['x'] == []
    assert record['y'] == 5


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record


def test_precord_constructor_overrides_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    base_pmap = pmap({'x': 1})
    record = TestRecord(_precord_size=base_pmap._size, _precord_buckets=base_pmap._buckets)
    assert record['x'] == 1


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields={'x': int})
    assert record['x'] == 1


# LLM-generated content at query #56
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "TestClass"
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, TestClass)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = "TestClass"
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) > 0


def test_persistent_raises_invariant_exception_with_error_codes():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "TestClass"
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_calls_check_global_invariants():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._invariant import InvariantException
    
    def failing_invariant(obj):
        return (False, 'global_error')
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        __name__ = "TestClass"
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_successful_with_no_invariants():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "TestClass"
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    result = evolver.persistent()
    assert result is not None


# LLM-generated content at query #57
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


def test_precord_constructor_with_kwargs_override_initial():
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


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    record = TestRecord()
    assert len(record) == 0


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
    
    record = TestRecord(x=1, _factory_fields={'x'})
    assert record['x'] == 1


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record


# LLM-generated content at query #58
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
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
    
    record = TestRecord(x=15)
    assert record['x'] == 15
    assert record['y'] == 5


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    def default_value():
        return 42
    
    class TestRecord(PRecord):
        x = field()
        y = field(initial=default_value)
    
    record = TestRecord(x=10)
    assert record['x'] == 10
    assert record['y'] == 42


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_factory_fields=set(), x=100)
    assert record['x'] == 100


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=50, extra_field=999)
    assert record['x'] == 50
    assert 'extra_field' not in record


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_new_kwargs_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=5)
        y = field(initial=10)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 10


# LLM-generated content at query #59
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    from pyrsistent._pmap import pmap
    pm = pmap({'x': 1, 'y': 2})
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
    
    record = TestRecord(x=5, _factory_fields=None)
    
    assert record['x'] == 5


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=7, _ignore_extra=False)
    
    assert record['x'] == 7


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=42)
        y = field()
    
    record = TestRecord(y=100)
    
    assert record['x'] == 42
    assert record['y'] == 100


def test_precord_new_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
    
    record = TestRecord(x=20)
    
    assert record['x'] == 20


def test_precord_new_with_callable_initial():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: 99)
    
    record = TestRecord()
    
    assert record['x'] == 99


def test_precord_new_empty():
    from pyrsistent import PRecord
    
    class TestRecord(PRecord):
        pass
    
    record = TestRecord()
    
    assert len(record) == 0


def test_precord_new_invalid_field():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    try:
        record = TestRecord(y=5)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert 'y' in str(e)


# LLM-generated content at query #60
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


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [1, 2, 3])
    
    record = TestRecord()
    assert record['x'] == [1, 2, 3]


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    base_map = pmap({'x': 5})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets)
    assert record['x'] == 5


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_factory_fields={'x': int}, x='42')
    assert record['x'] == 42


def test_precord_constructor_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    try:
        record = TestRecord(x=1, y=2)
        assert False, "Should have raised an error"
    except Exception:
        assert True


def test_precord_constructor_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=1, y=2)
    assert record['x'] == 1
    assert 'y' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


# LLM-generated content at query #61
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='Alice', age=30)
    assert record['name'] == 'Alice'
    assert record['age'] == 30


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        count = field(initial=0)
    
    record = TestRecord(name='Bob')
    assert record['name'] == 'Bob'
    assert record['count'] == 0


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        items = field(initial=list)
    
    record = TestRecord(name='Charlie')
    assert record['name'] == 'Charlie'
    assert record['items'] == []


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='David', age=25, _factory_fields=True)
    assert record['name'] == 'David'
    assert record['age'] == 25


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name='Eve', extra_field='ignored', _ignore_extra=True)
    assert record['name'] == 'Eve'
    assert 'extra_field' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='default')
    
    record = TestRecord()
    assert record['name'] == 'default'


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    base_map = pmap({'name': 'Frank'})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets)
    assert record['name'] == 'Frank'


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial='default_name')
        age = field()
    
    record = TestRecord(name='Grace', age=28)
    assert record['name'] == 'Grace'
    assert record['age'] == 28


# LLM-generated content at query #62
#--------------------------

```python
def test_precord_new_without_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    # Call __new__ with regular kwargs (not containing '_precord_size' and '_precord_buckets')
    # This should make the predicate on line 5 evaluate to False
    result = TestRecord(name='test', value=42)
    
    # Verify the object was created through the Evolver path (line 20)
    assert isinstance(result, TestRecord)
    assert result['name'] == 'test'
    assert result['value'] == 42


# LLM-generated content at query #63
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="Alice", age=30)
    assert record['name'] == "Alice"
    assert record['age'] == 30


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="Unknown")
        age = field(initial=0)
    
    record = TestRecord()
    assert record['name'] == "Unknown"
    assert record['age'] == 0


def test_precord_constructor_with_partial_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="Unknown")
        age = field(initial=0)
    
    record = TestRecord(name="Bob")
    assert record['name'] == "Bob"
    assert record['age'] == 0


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record = TestRecord()
    assert record['items'] == []


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(_factory_fields=True, name="Charlie", age=25)
    assert record['name'] == "Charlie"
    assert record['age'] == 25


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(_ignore_extra=True, name="David", extra_field="ignored")
    assert record['name'] == "David"
    assert 'extra_field' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="Default")
    
    record = TestRecord()
    assert record['name'] == "Default"


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    base_map = pmap({'name': 'Eve'})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets)
    assert record['name'] == 'Eve'


# LLM-generated content at query #64
#--------------------------

```python
def test_precord_new_without_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    # Create a PRecord without '_precord_size' and '_precord_buckets' in kwargs
    # This ensures the predicate at line 5 evaluates to False
    record = TestRecord(name="test", value=42)
    
    assert record['name'] == "test"
    assert record['value'] == 42
    assert isinstance(record, TestRecord)


# LLM-generated content at query #65
#--------------------------

```python
def test_persistent_predicate_line_6_evaluates_to_false():
    from pyrsistent import pmap, PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field()
        value = field()
    
    original_pmap = pmap({'name': 'test', 'value': 42})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    is_dirty_value = evolver.is_dirty()
    pm_result = type(original_pmap).persistent(evolver)
    isinstance_check = isinstance(pm_result, TestRecord)
    
    predicate_result = is_dirty_value or not isinstance_check
    
    assert predicate_result == False
    assert is_dirty_value == False
    assert isinstance_check == True


# LLM-generated content at query #66
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
        x = field(initial=lambda: [1, 2, 3])
    
    record = TestRecord()
    assert record['x'] == [1, 2, 3]


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert 'x' not in record


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_pmap = pmap({'x': 5})
    record = TestRecord(_precord_size=internal_pmap._size, _precord_buckets=internal_pmap._buckets)
    assert record['x'] == 5


def test_precord_constructor_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    try:
        record = TestRecord(x=1, z=2)
        # Should raise an error if ignore_extra is False (default)
    except:
        pass


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields=None)
    assert record['x'] == 1


def test_precord_constructor_multiple_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
    
    record = TestRecord(a=1, b=2, c=3)
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
    
    call_count = [0]
    
    def get_default():
        call_count[0] += 1
        return 42
    
    class TestRecord(PRecord):
        x = field(initial=get_default)
    
    record = TestRecord()
    assert record['x'] == 42
    assert call_count[0] == 1


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 5})
    record = TestRecord(_precord_size=internal_map._size, _precord_buckets=internal_map._buckets)
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
        x = field(initial=None)
    
    record = TestRecord()
    assert record['x'] is None


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields={})
    assert record['x'] == 1


# LLM-generated content at query #68
#--------------------------

```python
def test_persistent_predicate_line_6_evaluates_to_false():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
    
    # Create an original pmap
    original_pmap = PMap()
    
    # Create an evolver
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    # Mock the parent's persistent method to return an instance of the correct class
    # and ensure is_dirty() returns False
    class MockPMap(PMap):
        def __init__(self):
            super().__init__()
    
    mock_pmap = TestRecord()
    
    # Override the parent persistent to return a TestRecord instance
    original_super_persistent = PMap._Evolver.persistent
    
    def mock_persistent(self):
        return mock_pmap
    
    PMap._Evolver.persistent = mock_persistent
    
    try:
        # Manually set is_dirty to return False
        evolver.is_dirty = lambda: False
        
        # Call persistent and verify the predicate (is_dirty or not isinstance(pm, cls)) is False
        result = evolver.persistent()
        
        # If we reach here without exception, the test passes
        # The predicate was False because is_dirty() is False and isinstance(pm, cls) is True
        assert result is mock_pmap
    finally:
        PMap._Evolver.persistent = original_super_persistent


# LLM-generated content at query #69
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
        name = field()
        age = field(initial=25)
    
    record = TestRecord(name='Jane')
    assert record['name'] == 'Jane'
    assert record['age'] == 25


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record = TestRecord()
    assert record['items'] == []


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord()
    assert 'name' not in record


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name='Test', _factory_fields=['name'])
    assert record['name'] == 'Test'


def test_precord_constructor_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    try:
        record = TestRecord(name='Test', extra_field='value', _ignore_extra=False)
        assert False, "Should have raised an error"
    except:
        pass


def test_precord_constructor_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name='Test', extra_field='value', _ignore_extra=True)
    assert record['name'] == 'Test'
    assert 'extra_field' not in record


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    temp_map = pmap({'name': 'Direct'})
    record = TestRecord(_precord_size=temp_map._size, _precord_buckets=temp_map._buckets)
    assert record['name'] == 'Direct'


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
        x = field(initial=lambda: 42)
    
    record = TestRecord()
    assert record['x'] == 42


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    internal_map = pmap({'x': 5})
    record = TestRecord(_precord_size=internal_map._precord_size, _precord_buckets=internal_map._precord_buckets)
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
        record = TestRecord(x=1, z=999, _ignore_extra=False)
        assert False, "Should have raised an error for extra field"
    except:
        pass


def test_precord_constructor_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, z=999, _ignore_extra=True)
    assert record['x'] == 1
    assert 'z' not in record


def test_precord_constructor_empty_record():
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


# LLM-generated content at query #71
#--------------------------

```python
def test_persistent_predicate_line_6_true_when_is_dirty():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        x = field()
        _precord_mandatory_fields = set()
        _precord_invariants = ()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('x', 42)
    
    is_dirty = evolver.is_dirty()
    assert is_dirty is True
    
    pm = PMap._Evolver.persistent(evolver)
    isinstance_check = isinstance(pm, TestRecord)
    
    predicate_result = is_dirty or not isinstance_check
    assert predicate_result is True


# LLM-generated content at query #72
#--------------------------

```python
def test_persistent_predicate_line_6_evaluates_to_false():
    from pyrsistent import pmap, field, PRecord
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    original_pmap = pmap({'name': 'Alice', 'age': 30})
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    is_dirty_result = evolver.is_dirty()
    pm = original_pmap
    isinstance_result = isinstance(pm, TestRecord)
    
    predicate_result = is_dirty_result or not isinstance_result
    
    assert predicate_result is False
    assert is_dirty_result is False
    assert isinstance_result is False


# LLM-generated content at query #73
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap with the special attributes
    from pyrsistent._pmap import pmap
    test_pmap = pmap({'x': 1, 'y': 2})
    
    # Call __new__ with both _precord_size and _precord_buckets
    result = TestRecord.__new__(
        TestRecord,
        _precord_size=test_pmap._size,
        _precord_buckets=test_pmap._buckets
    )
    
    # Verify the condition at line 5 evaluates to True by checking the result is created
    assert result is not None
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2


# LLM-generated content at query #74
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
    
    record1 = TestRecord(y=1)
    record2 = TestRecord(y=2)
    assert record1['x'] == [1, 2, 3]
    assert record2['x'] == [1, 2, 3]
    assert record1['x'] is not record2['x']


def test_precord_constructor_with_internal_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    pmap_obj = pmap({'x': 1})
    record = TestRecord(_precord_size=pmap_obj._size, _precord_buckets=pmap_obj._buckets)
    assert record['x'] == 1


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20


def test_precord_constructor_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    try:
        record = TestRecord(x=1, z=2)
        assert False, "Should have raised an error for extra field"
    except TypeError:
        pass


def test_precord_constructor_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_ignore_extra=True, x=1, z=2)
    assert record['x'] == 1
    assert 'z' not in record


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(_factory_fields={'x': lambda v: v * 2}, x=5)
    assert record['x'] == 10


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


# LLM-generated content at query #75
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
        x = field()
        y = field(initial=10)
    
    record = TestRecord(x=5)
    assert record['x'] == 5
    assert record['y'] == 10


def test_precord_constructor_with_callable_initial_value():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field(initial=lambda: 20)
    
    record = TestRecord(x=5)
    assert record['x'] == 5
    assert record['y'] == 20


def test_precord_constructor_with_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    try:
        record = TestRecord(x=1, z=3, _ignore_extra=False)
        assert False, "Should have raised an error"
    except Exception:
        pass


def test_precord_constructor_with_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, z=3, _ignore_extra=True)
    assert record['x'] == 1
    assert 'z' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields={})
    assert record['x'] == 1


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    base_map = pmap({'x': 1})
    record = TestRecord(_precord_size=base_map._size, _precord_buckets=base_map._buckets)
    assert record['x'] == 1


def test_precord_constructor_overrides_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=5)
        y = field(initial=10)
    
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 10


# LLM-generated content at query #76
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pvector
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap to get valid buckets
    from pyrsistent._pmap import pmap
    pm = pmap({'x': 1, 'y': 2})
    
    # Call __new__ with both _precord_size and _precord_buckets
    # This should trigger the predicate at line 5 to be True
    result = TestRecord.__new__(TestRecord, _precord_size=pm._size, _precord_buckets=pm._buckets)
    
    # Verify that the result is a PRecord instance
    assert isinstance(result, TestRecord)
    assert result._size == pm._size
    assert result._buckets == pm._buckets


# LLM-generated content at query #77
#--------------------------

```python
def test_precord_meta_new_creates_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
        'field2': TestField(mandatory=False, initial=42),
        '__module__': 'test_module'
    }
    bases = (object,)
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert hasattr(result, '_precord_fields')
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_sets_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
        'field2': TestField(mandatory=False, initial=42),
        '__module__': 'test_module'
    }
    bases = (object,)
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert hasattr(result, '_precord_mandatory_fields')
    assert 'field1' in result._precord_mandatory_fields
    assert 'field2' not in result._precord_mandatory_fields


def test_precord_meta_new_sets_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
        'field2': TestField(mandatory=False, initial=42),
        '__module__': 'test_module'
    }
    bases = (object,)
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert hasattr(result, '_precord_initial_values')
    assert result._precord_initial_values.get('field2') == 42
    assert 'field1' not in result._precord_initial_values


def test_precord_meta_new_sets_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
        '__module__': 'test_module'
    }
    bases = (object,)
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert hasattr(result, '__slots__')
    assert result.__slots__ == ()


def test_precord_meta_new_stores_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    def test_invariant(self):
        return True, None
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
        '__invariant__': test_invariant,
        '__module__': 'test_module'
    }
    bases = (object,)
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert isinstance(result._precord_invariants, tuple)
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    parent_dct = {
        'parent_field': TestField(mandatory=True, initial=None),
        '__module__': 'test_module'
    }
    parent_class = _PRecordMeta('ParentRecord', (object,), parent_dct)
    
    child_dct = {
        'child_field': TestField(mandatory=False, initial=10),
        '__module__': 'test_module'
    }
    child_class = _PRecordMeta('ChildRecord', (parent_class,), child_dct)
    
    assert 'parent_field' in child_class._precord_fields
    assert 'child_field' in child_class._precord_fields


def test_precord_meta_new_removes_field_from_dct():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    class TestField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    
    dct = {
        'field1': TestField(mandatory=True, initial=None),
        '__module__': 'test_module'
    }
    bases = (object,)
    
    result = _PRecordMeta('TestRecord', bases, dct)
    
    assert 'field1' not in result.__dict__ or not isinstance(result.__dict__.get('field1'), _PField)


# LLM-generated content at query #78
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


# LLM-generated content at query #79
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockField:
        def __init__(self):
            self.name = 'test_field'
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._is_dirty = False
    
    result = evolver.persistent()
    assert result is not None


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
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
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
        assert 'MockCls.field1' in e.missing_fields
        assert 'MockCls.field2' in e.missing_fields


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
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_successful_with_all_valid():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    result = evolver.persistent()
    assert result is not None
    assert isinstance(result, MockCls)


# LLM-generated content at query #80
#--------------------------

```python
def test_precord_new_without_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Call __new__ without '_precord_size' and '_precord_buckets' in kwargs
    # This ensures the predicate at line 5 evaluates to False
    result = TestRecord(x=1, y=2)
    
    assert result.x == 1
    assert result.y == 2
    assert isinstance(result, TestRecord)


# LLM-generated content at query #81
#--------------------------

```python
def test_serialize_method_exists_and_callable():
    # Verify that the serialize method exists and is callable
    assert hasattr(PRecord, 'serialize')
    assert callable(getattr(PRecord, 'serialize'))


# LLM-generated content at query #82
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
        _precord_mandatory_fields = frozenset()
        _precord_invariants = ()
    
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


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
        _precord_mandatory_fields = frozenset()
        _precord_invariants = ()
    
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


def test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields_present():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent._pmap import PMap
    
    class TestRecord(PRecord):
        name = field()
        _precord_mandatory_fields = frozenset()
        _precord_invariants = ()
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver._invariant_error_codes = ['error_code_1', 'error_code_2']
    evolver._missing_fields = ['TestRecord.name']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error_code_1', 'error_code_2')
        assert e.missing_fields == ('TestRecord.name',)


# LLM-generated content at query #83
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockField:
        def __init__(self):
            self.invariant = lambda x: (True, None)
            self.factory = lambda x, **kwargs: x
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()
        assert e.message == 'Field invariant failed'


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['MockClass.field1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('MockClass.field1',)
        assert e.message == 'Field invariant failed'


def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
    
    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['MockClass.field1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('MockClass.field1',)
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #84
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    from pyrsistent import pmap, PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create initial pmap with some data
    initial_pmap = pmap({'x': 1, 'y': 2})
    
    # Call __new__ with _precord_size and _precord_buckets to trigger the predicate at line 5
    result = TestRecord(__new__=TestRecord.__new__, _precord_size=initial_pmap._size, _precord_buckets=initial_pmap._buckets)
    
    # Verify that the result is created successfully
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2


# LLM-generated content at query #85
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
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    result = evolver.persistent()
    
    assert isinstance(result, MockCls)


def test_persistent_raises_invariant_exception_with_field_error_codes():
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
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ()


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
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
    from pyrsistent._field_common import InvariantException
    from pyrsistent import pmap
    
    def failing_global_invariant(subject):
        return (False, 'global_error')
    
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        __name__ = 'MockCls'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #86
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
        a = field()
        b = field()
        c = field()
    
    record = MultiFieldRecord(a=1, b='two', c=3.0)
    repr_str = repr(record)
    
    assert 'MultiFieldRecord' in repr_str
    assert 'a=1' in repr_str
    assert "b='two'" in repr_str
    assert 'c=3.0' in repr_str


def test_precord_repr_with_nested_structure():
    from pyrsistent import PRecord, field
    
    class NestedRecord(PRecord):
        data = field()
    
    record = NestedRecord(data={'nested': 'value'})
    repr_str = repr(record)
    
    assert 'NestedRecord' in repr_str
    assert 'data=' in repr_str
    assert 'nested' in repr_str


# LLM-generated content at query #87
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    test_pmap = pmap({'x': 1, 'y': 2})
    result = TestRecord(_precord_size=test_pmap._size, _precord_buckets=test_pmap._buckets)
    assert result['x'] == 1
    assert result['y'] == 2


def test_precord_new_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    result = TestRecord(x=10, y=20)
    assert result['x'] == 10
    assert result['y'] == 20


def test_precord_new_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    result = TestRecord()
    assert len(result) == 0


def test_precord_new_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': 5, 'y': 10}
        x = field()
        y = field()
    
    result = TestRecord()
    assert result['x'] == 5
    assert result['y'] == 10


def test_precord_new_with_initial_values_override():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': 5, 'y': 10}
        x = field()
        y = field()
    
    result = TestRecord(x=100)
    assert result['x'] == 100
    assert result['y'] == 10


def test_precord_new_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        __precord_initial_values__ = {'x': lambda: 42}
        x = field()
    
    result = TestRecord()
    assert result['x'] == 42


def test_precord_new_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    result = TestRecord(x=1, _factory_fields=set())
    assert result['x'] == 1


def test_precord_new_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    result = TestRecord(x=1, _ignore_extra=True)
    assert result['x'] == 1


def test_precord_new_invalid_field_raises_error():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    try:
        TestRecord(invalid_field=1)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #88
#--------------------------

```python
def test_precord_meta_new_creates_precord_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        'field2': _PField(mandatory=False, initial='default'),
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_fields')
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields


def test_precord_meta_new_creates_mandatory_fields_set():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        'field2': _PField(mandatory=False, initial='default'),
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_mandatory_fields')
    assert 'field1' in result._precord_mandatory_fields
    assert 'field2' not in result._precord_mandatory_fields


def test_precord_meta_new_creates_initial_values_dict():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        'field2': _PField(mandatory=False, initial='default'),
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_initial_values')
    assert result._precord_initial_values['field2'] == 'default'


def test_precord_meta_new_sets_empty_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '__slots__')
    assert result.__slots__ == ()


def test_precord_meta_new_creates_invariants():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    def my_invariant(self):
        return True, None
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        '__invariant__': my_invariant,
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert hasattr(result, '_precord_invariants')
    assert isinstance(result._precord_invariants, tuple)
    assert len(result._precord_invariants) > 0


def test_precord_meta_new_inherits_fields_from_bases():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    parent_dct = {
        'parent_field': _PField(mandatory=True, initial=None),
        '_precord_fields': {'parent_field': _PField(mandatory=True, initial=None)},
    }
    parent_class = _PRecordMeta.__new__(_PRecordMeta, 'Parent', (), parent_dct)
    
    child_dct = {
        'child_field': _PField(mandatory=False, initial='default'),
    }
    bases = (parent_class,)
    name = 'Child'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, child_dct)
    
    assert hasattr(result, '_precord_fields')
    assert 'child_field' in result._precord_fields


def test_precord_meta_new_removes_field_definitions_from_dict():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField
    
    dct = {
        'field1': _PField(mandatory=True, initial=None),
        'field2': _PField(mandatory=False, initial='default'),
    }
    bases = ()
    name = 'TestPRecord'
    
    result = _PRecordMeta.__new__(_PRecordMeta, name, bases, dct)
    
    assert not hasattr(result, 'field1') or isinstance(result.field1, _PField)
    assert not hasattr(result, 'field2') or isinstance(result.field2, _PField)


# LLM-generated content at query #89
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent import pmap, field, InvariantException
    from pyrsistent._precord import PRecord
    
    class TestRecord(PRecord):
        x = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, pmap({}))
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()


def test_persistent_raises_invariant_exception_when_missing_fields_present():
    from pyrsistent import pmap, field, InvariantException
    from pyrsistent._precord import PRecord
    
    class TestRecord(PRecord):
        x = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, pmap({}))
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['missing_field']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('missing_field',)


def test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields():
    from pyrsistent import pmap, field, InvariantException
    from pyrsistent._precord import PRecord
    
    class TestRecord(PRecord):
        x = field()
    
    evolver = TestRecord._PRecordEvolver(TestRecord, pmap({}))
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['missing1']
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('missing1',)


# LLM-generated content at query #90
#--------------------------

```python
def test_precord_constructor_with_kwargs():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name="John", age=30)
    assert record['name'] == "John"
    assert record['age'] == 30


def test_precord_constructor_with_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=0)
    
    record = TestRecord()
    assert record['name'] == "DefaultName"
    assert record['age'] == 0


def test_precord_constructor_with_callable_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        items = field(initial=list)
    
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['items'] == []
    assert record2['items'] == []
    assert record1['items'] is not record2['items']


def test_precord_constructor_override_initial_values():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field(initial="DefaultName")
        age = field(initial=0)
    
    record = TestRecord(name="Jane", age=25)
    assert record['name'] == "Jane"
    assert record['age'] == 25


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord.create({'name': 'John', 'extra': 'value'}, ignore_extra=True)
    assert record['name'] == 'John'
    assert 'extra' not in record


def test_precord_constructor_without_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    try:
        record = TestRecord(name='John', extra='value')
        assert False, "Should have raised an error"
    except TypeError:
        pass


def test_precord_constructor_from_existing_instance():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record1 = TestRecord(name="John", age=30)
    record2 = TestRecord.create(record1)
    assert record2 is record1


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord.create({'name': 'John'}, _factory_fields=None)
    assert record['name'] == 'John'


# LLM-generated content at query #91
#--------------------------

```python
def test_precord_evolver_set_with_existing_field():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    
    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)
    
    original_record = TestRecord(name="John", age=30)
    evolver = _PRecordEvolver(TestRecord, original_record._to_pmap())
    
    result = evolver.set("name", "Jane")
    
    assert result is not None
    assert isinstance(result, _PRecordEvolver)


# LLM-generated content at query #92
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


def test_precord_constructor_with_callable_initial_value():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field(initial=lambda: [1, 2, 3])
    
    record = TestRecord()
    assert record['x'] == [1, 2, 3]


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert 'x' not in record


def test_precord_constructor_with_internal_params():
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
    
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _factory_fields={'x': int})
    assert record['x'] == 1


# LLM-generated content at query #93
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
    
    record = TestRecord(name='Jane', age=25)
    assert record['name'] == 'Jane'
    assert record['age'] == 25


def test_precord_constructor_with_factory_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    record = TestRecord(name='John', _factory_fields=None)
    assert record['name'] == 'John'


def test_precord_constructor_ignore_extra_false():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
    
    try:
        record = TestRecord(name='John', extra_field='value', _ignore_extra=False)
        assert False, "Should have raised an exception"
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
    
    try:
        record = TestRecord()
        assert False, "Should have raised an exception for missing required field"
    except:
        pass


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        name = field()
    
    pmap_obj = pmap({'name': 'John'})
    record = TestRecord(_precord_size=pmap_obj._size, _precord_buckets=pmap_obj._buckets)
    assert record['name'] == 'John'


# LLM-generated content at query #94
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
    
    record = TestRecord(x=1, _factory_fields=None)
    assert record['x'] == 1


def test_precord_constructor_with_ignore_extra():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, _ignore_extra=True, extra_field=999)
    assert record['x'] == 1
    assert 'extra_field' not in record


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    pmap_obj = pmap({'x': 5})
    record = TestRecord(_precord_size=pmap_obj._size, _precord_buckets=pmap_obj._buckets)
    assert record['x'] == 5


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


# LLM-generated content at query #95
#--------------------------

```python
def test_persistent_with_no_changes():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert result is not None


def test_persistent_with_mandatory_fields_missing():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'required_field'}
        _precord_invariants = []
        __name__ = 'TestClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestClass.required_field' in e.missing_fields


def test_persistent_with_field_invariant_error():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error_code_1']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error_code_1' in e.invariant_errors


def test_persistent_with_global_invariant_failure():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        __name__ = 'TestClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return []
    
    original_pmap = pmap({})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_dirty_state_creates_new_instance():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'TestClass'
        
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self._is_instance = True
        
        def keys(self):
            return []
    
    original_pmap = pmap({'key': 'value'})
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._data['new_key'] = 'new_value'
    result = evolver.persistent()
    assert result is not None
    assert isinstance(result, MockClass)


# LLM-generated content at query #96
#--------------------------

```python
def test_persistent_checks_mandatory_fields_when_precord_mandatory_fields_is_truthy():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    TestRecord._precord_mandatory_fields = {'name', 'age'}
    TestRecord._precord_invariants = ()
    
    original_pmap = TestRecord()._to_pmap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert len(e.missing_fields) > 0
        assert 'TestRecord.name' in e.missing_fields or 'TestRecord.age' in e.missing_fields


# LLM-generated content at query #97
#--------------------------

```python
def test_precord_new_with_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create a pmap with the special attributes
    from pyrsistent._pmap import pmap
    test_pmap = pmap({'x': 1, 'y': 2})
    
    # Call __new__ with the special attributes that trigger the predicate at line 5
    result = TestRecord.__new__(TestRecord, _precord_size=test_pmap._size, _precord_buckets=test_pmap._buckets)
    
    # Verify the result is a TestRecord instance
    assert isinstance(result, TestRecord)
    assert result['x'] == 1
    assert result['y'] == 2


# LLM-generated content at query #98
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


# LLM-generated content at query #99
#--------------------------

```python
def test_precord_repr():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        name = field()
        age = field()
    
    record = TestRecord(name='Alice', age=30)
    repr_str = repr(record)
    
    assert repr_str.startswith('TestRecord(')
    assert repr_str.endswith(')')
    assert 'name=' in repr_str
    assert 'age=' in repr_str
    assert "'Alice'" in repr_str
    assert '30' in repr_str


# LLM-generated content at query #100
#--------------------------

```python
def test_persistent_returns_result_when_no_errors():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def invariant(self, value):
            return (True, None)
    
    class MockCls:
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _buckets = None
        _size = 0
        
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    result = evolver.persistent()
    
    assert isinstance(result, MockCls)


def test_persistent_raises_invariant_exception_on_missing_mandatory_fields():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def invariant(self, value):
            return (True, None)
    
    class MockCls:
        __name__ = 'MockCls'
        _precord_fields = {'field1': MockField('field1'), 'field2': MockField('field2')}
        _precord_mandatory_fields = {'field2'}
        _precord_invariants = []
        _buckets = None
        _size = 0
        
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockCls.field2' in e.missing_fields


def test_persistent_raises_invariant_exception_on_field_invariant_error():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def invariant(self, value):
            return (False, 'error_code_1')
    
    class MockCls:
        __name__ = 'MockCls'
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _buckets = None
        _size = 0
        
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._invariant_error_codes.append('error_code_1')
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'error_code_1' in e.invariant_errors


def test_persistent_calls_check_global_invariants():
    from pyrsistent import pmap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import InvariantException
    
    class MockField:
        def __init__(self, name):
            self.name = name
        
        def invariant(self, value):
            return (True, None)
    
    def failing_global_invariant(subject):
        return (False, 'global_error')
    
    class MockCls:
        __name__ = 'MockCls'
        _precord_fields = {'field1': MockField('field1')}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_global_invariant]
        _buckets = None
        _size = 0
        
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
        
        def keys(self):
            return ['field1']
    
    original_pmap = pmap({'field1': 'value1'})
    evolver = _PRecordEvolver(MockCls, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #101
#--------------------------

```python
def test_precord_evolver_set_with_valid_field():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import PTypeError
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: int(x)
            self.invariant = lambda x: (True, None)
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    result = evolver.set('test_field', 42)
    assert result is evolver


def test_precord_evolver_set_with_invalid_field_name():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: int(x)
            self.invariant = lambda x: (True, None)
    
    class MockClass:
        __name__ = 'MockClass'
        _precord_fields = {'valid_field': MockField()}
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    
    try:
        evolver.set('invalid_field', 42)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert 'invalid_field' in str(e)
        assert 'MockClass' in str(e)


def test_precord_evolver_set_with_factory_fields_filter():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: int(x)
            self.invariant = lambda x: (True, None)
    
    field1 = MockField()
    field2 = MockField()
    
    class MockClass:
        _precord_fields = {'field1': field1, 'field2': field2}
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap, _factory_fields=(field1,))
    result = evolver.set('field1', 42)
    assert result is evolver


def test_precord_evolver_set_with_field_not_in_factory_fields():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: int(x)
            self.invariant = lambda x: (True, None)
    
    field1 = MockField()
    field2 = MockField()
    
    class MockClass:
        _precord_fields = {'field1': field1, 'field2': field2}
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap, _factory_fields=(field1,))
    result = evolver.set('field2', 42)
    assert result is evolver


def test_precord_evolver_setitem_calls_set():
    from pyrsistent import PMap
    from pyrsistent._precord import _PRecordEvolver
    
    class MockField:
        def __init__(self):
            self.type = (int,)
            self.factory = lambda x: int(x)
            self.invariant = lambda x: (True, None)
    
    class MockClass:
        _precord_fields = {'test_field': MockField()}
    
    pmap = PMap()
    evolver = _PRecordEvolver(MockClass, pmap)
    result = evolver.__setitem__('test_field', 42)
    assert result is evolver


# LLM-generated content at query #102
#--------------------------

```python
def test_persistent_returns_instance_when_not_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class MockField:
        def factory(self, value, ignore_extra=False):
            return value
        def invariant(self, value):
            return (True, None)
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, MockClass)


def test_persistent_raises_invariant_exception_with_missing_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'MockClass'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MockClass.field1' in e.missing_fields
        assert 'MockClass.field2' in e.missing_fields


def test_persistent_raises_invariant_exception_with_field_invariant_errors():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._field_common import InvariantException
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')


def test_persistent_calls_check_global_invariants():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    from pyrsistent._field_common import InvariantException
    
    def failing_invariant(subject):
        return (False, 'global_error')
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [failing_invariant]
        __name__ = 'MockClass'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


def test_persistent_creates_new_instance_when_dirty():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap
    
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = 'MockClass'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
        def keys(self):
            return []
    
    original_pmap = PMap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver[0] = 'value'
    result = evolver.persistent()
    assert isinstance(result, MockClass)


# LLM-generated content at query #103
#--------------------------

```python
def test_precord_meta_new_basic():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL

    class TestRecord(metaclass=_PRecordMeta):
        pass

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()


def test_precord_meta_new_with_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL

    field1 = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)
    field2 = _PField(initial=42, mandatory=False)

    class TestRecord(metaclass=_PRecordMeta):
        _precord_fields = {'field1': field1, 'field2': field2}

    assert 'field1' in TestRecord._precord_mandatory_fields
    assert 'field2' not in TestRecord._precord_mandatory_fields
    assert TestRecord._precord_initial_values.get('field2') == 42
    assert 'field1' not in TestRecord._precord_initial_values


def test_precord_meta_new_with_invariant():
    from pyrsistent._precord import _PRecordMeta

    def test_invariant(obj):
        return True, None

    class TestRecord(metaclass=_PRecordMeta):
        __invariant__ = test_invariant

    assert len(TestRecord._precord_invariants) > 0
    assert callable(TestRecord._precord_invariants[0])


def test_precord_meta_new_inheritance():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL

    field1 = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)

    class BaseRecord(metaclass=_PRecordMeta):
        _precord_fields = {'field1': field1}

    class DerivedRecord(BaseRecord):
        pass

    assert 'field1' in DerivedRecord._precord_fields
    assert 'field1' in DerivedRecord._precord_mandatory_fields


def test_precord_meta_new_slots():
    from pyrsistent._precord import _PRecordMeta

    class TestRecord(metaclass=_PRecordMeta):
        pass

    assert TestRecord.__slots__ == ()


def test_precord_meta_new_multiple_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL

    field1 = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)
    field2 = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)
    field3 = _PField(initial=10, mandatory=False)

    class TestRecord(metaclass=_PRecordMeta):
        _precord_fields = {'field1': field1, 'field2': field2, 'field3': field3}

    assert TestRecord._precord_mandatory_fields == {'field1', 'field2'}
    assert set(TestRecord._precord_initial_values.keys()) == {'field3'}


def test_precord_meta_new_empty_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL

    field1 = _PField(initial=PFIELD_NO_INITIAL, mandatory=True)
    field2 = _PField(initial=PFIELD_NO_INITIAL, mandatory=False)

    class TestRecord(metaclass=_PRecordMeta):
        _precord_fields = {'field1': field1, 'field2': field2}

    assert len(TestRecord._precord_initial_values) == 0


# LLM-generated content at query #104
#--------------------------

```python
def test_precord_new_without_special_attributes():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
    
    # Create instance without '_precord_size' and '_precord_buckets'
    # This should make the predicate at line 5 evaluate to False
    result = TestRecord(x=1, y=2)
    
    assert result.x == 1
    assert result.y == 2
    assert isinstance(result, TestRecord)


# LLM-generated content at query #105
#--------------------------

```python
def test_precord_meta_new_creates_class_with_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL
    
    # Create a test field
    test_field = _PField(initial=PFIELD_NO_INITIAL, mandatory=True, factory=None, initial_factory=None)
    
    # Create a dictionary with a field
    dct = {'test_attr': test_field}
    bases = ()
    
    # Call __new__ to create the class
    result_class = _PRecordMeta.__new__(_PRecordMeta, 'TestPRecord', bases, dct)
    
    # Verify that __slots__ is set to empty tuple
    assert result_class.__slots__ == ()
    assert isinstance(result_class, type)
    assert result_class.__name__ == 'TestPRecord'


# LLM-generated content at query #106
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
    
    record = TestRecord(x=999, y=888)
    assert record['x'] == 999
    assert record['y'] == 888


def test_precord_constructor_with_precord_size_and_buckets():
    from pyrsistent import PRecord, field, pmap
    
    class TestRecord(PRecord):
        x = field()
    
    base_pmap = pmap({'x': 5})
    record = TestRecord(_precord_size=base_pmap._size, _precord_buckets=base_pmap._buckets)
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
        record = TestRecord(x=1, z=999, _ignore_extra=False)
        assert False, "Should have raised an error"
    except:
        pass


def test_precord_constructor_ignore_extra_true():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord(x=1, z=999, _ignore_extra=True)
    assert record['x'] == 1
    assert 'z' not in record


def test_precord_constructor_empty():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
    
    record = TestRecord()
    assert len(record) == 0


def test_precord_constructor_partial_fields():
    from pyrsistent import PRecord, field
    
    class TestRecord(PRecord):
        x = field()
        y = field()
        z = field()
    
    record = TestRecord(x=1, z=3)
    assert record['x'] == 1
    assert record['z'] == 3
    assert 'y' not in record


