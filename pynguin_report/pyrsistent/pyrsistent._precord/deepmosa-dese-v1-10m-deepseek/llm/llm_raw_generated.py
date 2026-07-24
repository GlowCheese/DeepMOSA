####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_persistent_returns_instance_of_destination_class():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(TestRecord, PMap())
    result = evolver.persistent()
    assert isinstance(result, TestRecord)

def test_persistent_raises_invariant_exception_on_missing_mandatory_fields():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'mandatory_field'}
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(TestRecord, PMap())
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'mandatory_field' in e.missing_fields

def test_persistent_raises_invariant_exception_on_field_invariant_errors():
    class TestRecord:
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (False, 'error')})()}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: {'field'}
    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('field', 'value')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'error' in e.invariant_errors

def test_persistent_raises_invariant_exception_on_global_invariant_failure():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [lambda x: (False, 'global_error')]
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(TestRecord, PMap())
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors

def test_persistent_returns_pmap_when_not_dirty_and_already_instance():
    class TestRecord(PMap):
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    original_pmap = TestRecord()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    result = evolver.persistent()
    assert result is original_pmap

def test_persistent_creates_new_instance_when_dirty():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    original_pmap = PMap()
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('new_field', 'new_value')
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result._buckets is not original_pmap._buckets


# LLM-generated content at query #2
#--------------------------

def test_precord_new_with_special_attributes():
    class TestRecord(PRecord):
        pass
    record = TestRecord(_precord_size=0, _precord_buckets=pvector().extend([]))
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {'field1': 10, 'field2': 'default'}
    record = TestRecord()
    assert record['field1'] == 10
    assert record['field2'] == 'default'

def test_precord_new_with_kwargs_overrides_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {'field1': 10, 'field2': 'default'}
    record = TestRecord(field1=20)
    assert record['field1'] == 20
    assert record['field2'] == 'default'

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, factory=lambda x: x * 2)}
    record = TestRecord(_factory_fields={field(type=int, factory=lambda x: x * 2)}, field1=5)
    assert record['field1'] == 10

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
    record = TestRecord(_ignore_extra=True, field1=1, extra_field=2)
    assert record['field1'] == 1
    assert 'extra_field' not in record

def test_precord_new_without_ignore_extra_raises_attribute_error():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
    try:
        TestRecord(field1=1, extra_field=2)
        assert False
    except AttributeError as e:
        assert "'extra_field' is not among the specified fields for TestRecord" in str(e)

def test_precord_new_with_invariant_failure():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, invariant=lambda x: (x > 0, 'ERR'))}
    try:
        TestRecord(field1=-1)
        assert False
    except InvariantException as e:
        assert 'ERR' in e.invariant_errors

def test_precord_new_with_missing_mandatory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, mandatory=True), 'field2': field(type=str)}
        _precord_mandatory_fields = {'field1'}
    try:
        TestRecord(field2='test')
        assert False
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

def test_precord_new_with_valid_data():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
    record = TestRecord(field1=100, field2='hello')
    assert record['field1'] == 100
    assert record['field2'] == 'hello'

def test_precord_new_with_checked_type_factory():
    class InnerRecord(PRecord):
        _precord_fields = {'inner_field': field(type=int)}
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=InnerRecord)}
    inner = InnerRecord(inner_field=5)
    record = TestRecord(field1=inner)
    assert record['field1']['inner_field'] == 5

def test_precord_new_with_callable_initial_value():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
        _precord_initial_values = {'field1': lambda: 42}
    record = TestRecord()
    assert record['field1'] == 42


# LLM-generated content at query #3
#--------------------------

```python
def test_precord_initial_values_condition_true():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': lambda: 42}
    result = TestRecord()
    assert result['x'] == 42


# LLM-generated content at query #4
#--------------------------

def test_precord_repr_with_single_field():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'name': field(type=str)}
    record = TestRecord(name='Alice')
    result = repr(record)
    assert result == "TestRecord(name='Alice')"

def test_precord_repr_with_multiple_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(type=int), 'y': field(type=str)}
    record = TestRecord(x=10, y='test')
    result = repr(record)
    assert result == "TestRecord(x=10, y='test')"

def test_precord_repr_with_empty_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
    record = TestRecord()
    result = repr(record)
    assert result == "TestRecord()"

def test_precord_repr_with_nested_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'data': field(type=dict)}
    record = TestRecord(data={'key': 'value'})
    result = repr(record)
    assert result == "TestRecord(data={'key': 'value'})"

def test_precord_repr_with_special_characters_in_field_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'text': field(type=str)}
    record = TestRecord(text='line1\nline2')
    result = repr(record)
    assert result == "TestRecord(text='line1\\nline2')"

def test_precord_repr_with_integer_field_name_and_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'123': field(type=int)}
    record = TestRecord(**{'123': 456})
    result = repr(record)
    assert result == "TestRecord(123=456)"


# LLM-generated content at query #5
#--------------------------

def test_precord_repr_with_single_field():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'name': field(type=str)}
        _precord_initial_values = {}
    record = TestRecord(name='Alice')
    result = repr(record)
    expected = "TestRecord(name='Alice')"
    assert result == expected

def test_precord_repr_with_multiple_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(type=int), 'y': field(type=str)}
        _precord_initial_values = {}
    record = TestRecord(x=10, y='test')
    result = repr(record)
    expected = "TestRecord(x=10, y='test')"
    assert result == expected

def test_precord_repr_with_no_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
        _precord_initial_values = {}
    record = TestRecord()
    result = repr(record)
    expected = "TestRecord()"
    assert result == expected

def test_precord_repr_with_nested_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'data': field(type=dict), 'count': field(type=int)}
        _precord_initial_values = {}
    record = TestRecord(data={'key': 'value'}, count=5)
    result = repr(record)
    expected = "TestRecord(data={'key': 'value'}, count=5)"
    assert result == expected

def test_precord_repr_with_special_characters_in_field_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'text': field(type=str)}
        _precord_initial_values = {}
    record = TestRecord(text='line1\nline2')
    result = repr(record)
    expected = "TestRecord(text='line1\\nline2')"
    assert result == expected


# LLM-generated content at query #6
#--------------------------

```python
def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_initial_values = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    record = TestRecord()
    assert isinstance(record, TestRecord)
    assert len(record) == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_persistent_when_cls_has_mandatory_fields():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'mandatory_field'}
        _precord_invariants = ()
        __name__ = 'TestRecord'
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(TestRecord, None)
    evolver._missing_fields = []
    evolver._invariant_error_codes = []
    evolver._destination_cls = TestRecord
    evolver.is_dirty = lambda: False
    evolver._original_pmap = type('PMap', (), {'_buckets': None, '_size': 0})()
    evolver._buckets = evolver._original_pmap._buckets
    evolver._size = evolver._original_pmap._size
    evolver.persistent = lambda: evolver._original_pmap
    try:
        evolver.persistent()
    except InvariantException as e:
        assert 'TestRecord.mandatory_field' in e.missing_fields


# LLM-generated content at query #8
#--------------------------

def test_set_with_valid_field_and_value():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=lambda x: x, type=(int,), invariant=lambda x: (True, None))}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    assert evolver._original_pmap['key'] == 5

def test_set_with_field_factory_exception():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=lambda x: (_ for _ in ()).throw(InvariantException((), (), '')), type=(int,), invariant=lambda x: (True, None))}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    assert len(evolver._invariant_error_codes) == 0
    assert len(evolver._missing_fields) == 0

def test_set_with_field_invariant_failure():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=lambda x: x, type=(int,), invariant=lambda x: (False, 'error'))}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    assert evolver._invariant_error_codes == ['error']

def test_set_with_invalid_field():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockDestinationCls:
        _precord_fields = {}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    try:
        evolver.set('invalid', 5)
        assert False
    except AttributeError as e:
        assert "'invalid' is not among the specified fields" in str(e)

def test_set_with_factory_fields_skipped():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=lambda x: 'factory', type=(int,), invariant=lambda x: (True, None))}
    original_pmap = {}
    factory_fields = set()
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap, _factory_fields=factory_fields)
    evolver.set('key', 5)
    assert evolver._original_pmap['key'] == 5

def test_set_with_ignore_extra_complaint():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=lambda x, ignore_extra=False: x, type=(CheckedType,), invariant=lambda x: (True, None))}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap, _ignore_extra=True)
    evolver.set('key', 5)
    assert evolver._original_pmap['key'] == 5


# LLM-generated content at query #9
#--------------------------

def test___new___sets_fields_correctly():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class Base1:
        __invariant__ = lambda self: (True, ())
        _precord_fields = {'base1_field': _PField(False, PFIELD_NO_INITIAL)}
    class Base2:
        __invariant__ = lambda self: (True, ())
        _precord_fields = {'base2_field': _PField(True, 'initial2')}
    dct = {'new_field': _PField(False, 'new_initial')}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    dct['_precord_mandatory_fields'] = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
    dct['_precord_initial_values'] = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
    dct['__slots__'] = ()
    assert '_precord_fields' in dct
    assert 'new_field' in dct['_precord_fields']
    assert 'base1_field' in dct['_precord_fields']
    assert 'base2_field' in dct['_precord_fields']
    assert dct['_precord_fields']['new_field'].initial == 'new_initial'
    assert dct['_precord_fields']['base2_field'].initial == 'initial2'
    assert dct['_precord_fields']['base2_field'].mandatory is True
    assert dct['_precord_mandatory_fields'] == {'base2_field'}
    assert dct['_precord_initial_values'] == {'new_field': 'new_initial', 'base2_field': 'initial2'}
    assert dct['__slots__'] == ()
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 2
    assert all(callable(inv) for inv in dct['_precord_invariants'])

def test___new___handles_no_invariants():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class Base:
        _precord_fields = {'field': _PField(False, PFIELD_NO_INITIAL)}
    dct = {}
    bases = (Base,)
    set_fields(dct, bases, name='_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    dct['_precord_mandatory_fields'] = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
    dct['_precord_initial_values'] = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
    dct['__slots__'] = ()
    assert '_precord_fields' in dct
    assert 'field' in dct['_precord_fields']
    assert dct['_precord_mandatory_fields'] == set()
    assert dct['_precord_initial_values'] == {}
    assert dct['__slots__'] == ()
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 0

def test___new___raises_on_non_callable_invariant():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class Base:
        __invariant__ = "not callable"
        _precord_fields = {}
    dct = {}
    bases = (Base,)
    set_fields(dct, bases, name='_precord_fields')
    try:
        store_invariants(dct, bases, '_precord_invariants', '__invariant__')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test___new___merges_inherited_fields():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class GrandBase:
        _precord_fields = {'grand_field': _PField(True, PFIELD_NO_INITIAL)}
    class Parent(GrandBase):
        _precord_fields = {'parent_field': _PField(False, 'parent_initial')}
    dct = {'child_field': _PField(False, 'child_initial')}
    bases = (Parent,)
    set_fields(dct, bases, name='_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    dct['_precord_mandatory_fields'] = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
    dct['_precord_initial_values'] = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
    dct['__slots__'] = ()
    assert 'grand_field' in dct['_precord_fields']
    assert 'parent_field' in dct['_precord_fields']
    assert 'child_field' in dct['_precord_fields']
    assert dct['_precord_mandatory_fields'] == {'grand_field'}
    assert dct['_precord_initial_values'] == {'parent_field': 'parent_initial', 'child_field': 'child_initial'}
    assert dct['__slots__'] == ()


# LLM-generated content at query #10
#--------------------------

```python
def test_persistent_when_is_dirty_is_true():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_global_invariants
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, **kwargs):
            pass
    original_pmap = {}
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._destination_cls = MockClass
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    evolver._factory_fields = None
    evolver._ignore_extra = False
    evolver.is_dirty = lambda: True
    evolver._buckets = {}
    evolver._size = 0
    result = evolver.persistent()
    assert isinstance(result, MockClass)

def test_persistent_when_pm_is_not_instance_of_cls():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_global_invariants
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, **kwargs):
            pass
    class OtherClass:
        pass
    original_pmap = {}
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._destination_cls = MockClass
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    evolver._factory_fields = None
    evolver._ignore_extra = False
    evolver.is_dirty = lambda: False
    evolver._buckets = {}
    evolver._size = 0
    evolver.persistent = lambda: OtherClass()
    result = evolver.persistent()
    assert isinstance(result, MockClass)

def test_persistent_when_is_dirty_false_and_pm_is_instance_of_cls():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_global_invariants
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, **kwargs):
            pass
    original_pmap = {}
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._destination_cls = MockClass
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    evolver._factory_fields = None
    evolver._ignore_extra = False
    evolver.is_dirty = lambda: False
    pm_instance = MockClass()
    evolver.persistent = lambda: pm_instance
    result = evolver.persistent()
    assert result is pm_instance


# LLM-generated content at query #11
#--------------------------

def test_persistent_returns_instance_of_destination_class():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    result = evolver.persistent()
    assert isinstance(result, MockClass)

def test_persistent_raises_invariant_exception_on_missing_mandatory_fields():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'mandatory_field'}
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'mandatory_field' in e.missing_fields

def test_persistent_raises_invariant_exception_on_field_invariant_errors():
    class MockClass:
        _precord_fields = {'field': MockField(invariant=lambda v: (False, 'error'))}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    evolver.set('field', 'value')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'error' in e.invariant_errors

def test_persistent_raises_invariant_exception_on_global_invariant_failure():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [lambda s: (False, 'global_error')]
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors

def test_persistent_returns_same_instance_if_not_dirty_and_already_correct_class():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    original_pmap = MockClass()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    result = evolver.persistent()
    assert result is original_pmap

def test_persistent_creates_new_instance_if_dirty():
    class MockClass:
        _precord_fields = {'field': MockField(invariant=lambda v: (True, None))}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    original_pmap = {}
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver.set('field', 'value')
    result = evolver.persistent()
    assert isinstance(result, MockClass)
    assert result is not original_pmap


# LLM-generated content at query #12
#--------------------------

def test___new___sets_fields_correctly():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField:
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial
    _PField = MockPField
    class Base1:
        _precord_fields = {'base1_field': MockField(True, PFIELD_NO_INITIAL)}
    class Base2:
        _precord_fields = {'base2_field': MockField(False, 'default')}
    dct = {'custom_field': _PField(mandatory=True, initial=PFIELD_NO_INITIAL)}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    mandatory_fields = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
    initial_values = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
    assert '_precord_fields' in dct
    assert 'base1_field' in dct['_precord_fields']
    assert 'base2_field' in dct['_precord_fields']
    assert 'custom_field' in dct['_precord_fields']
    assert 'custom_field' not in dct
    assert mandatory_fields == {'base1_field', 'custom_field'}
    assert initial_values == {'base2_field': 'default'}

def test___new___handles_invariants():
    def invariant1(instance):
        return True, ()
    def invariant2(instance):
        return False, ('error',)
    class BaseWithInvariant:
        __invariant__ = invariant1
    dct = {'__invariant__': invariant2}
    bases = (BaseWithInvariant,)
    set_fields(dct, bases, name='_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    assert invariants[0](None) == (True, ())
    assert invariants[1](None) == (False, ('error',))

def test___new___raises_on_non_callable_invariant():
    class BaseWithNonCallableInvariant:
        __invariant__ = 'not callable'
    dct = {}
    bases = (BaseWithNonCallableInvariant,)
    set_fields(dct, bases, name='_precord_fields')
    try:
        store_invariants(dct, bases, '_precord_invariants', '__invariant__')
        assert False
    except TypeError:
        pass

def test___new___sets_slots():
    dct = {}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    dct['__slots__'] = ()
    assert '__slots__' in dct
    assert dct['__slots__'] == ()

def test___new___merges_invariant_results():
    def multi_invariant(instance):
        return [(True, ()), (False, ('err1',)), (False, ('err2',))]
    dct = {'__invariant__': multi_invariant}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    result = invariants[0](None)
    assert result == (False, ('err1', 'err2'))

def test___new___with_empty_bases():
    dct = {}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert '_precord_fields' in dct
    assert dct['_precord_fields'] == {}
    assert '_precord_invariants' in dct
    assert dct['_precord_invariants'] == ()
    assert '_precord_mandatory_fields' not in dct
    assert '_precord_initial_values' not in dct


# LLM-generated content at query #13
#--------------------------

def test_store_invariants_wraps_invariants():
    class MockInvariant1:
        __invariant__ = lambda *args, **kwargs: (True, ())
    class MockInvariant2:
        __invariant__ = lambda *args, **kwargs: (False, ("error",))
    bases = (MockInvariant1, MockInvariant2)
    dct = {}
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    result1 = invariants[0]()
    assert result1 == (True, ())
    result2 = invariants[1]()
    assert result2 == (False, ("error",))

def test_store_invariants_merges_multiple_results():
    def invariant_returns_list(*args, **kwargs):
        return [(True, ()), (False, ("err1",)), (False, ("err2",))]
    class MockInvariant:
        __invariant__ = invariant_returns_list
    bases = (MockInvariant,)
    dct = {}
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    assert len(invariants) == 1
    result = invariants[0]()
    assert result == (False, ("err1", "err2"))

def test_store_invariants_raises_on_non_callable():
    class MockInvariant:
        __invariant__ = "not callable"
    bases = (MockInvariant,)
    dct = {}
    try:
        store_invariants(dct, bases, '_precord_invariants', '__invariant__')
        assert False
    except TypeError:
        pass

def test_store_invariants_inherits_from_bases():
    class Base1:
        __invariant__ = lambda *args, **kwargs: (True, ())
    class Base2:
        __invariant__ = lambda *args, **kwargs: (False, ("base2",))
    class Child(Base1, Base2):
        pass
    bases = (Base1, Base2)
    dct = {}
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    result1 = invariants[0]()
    assert result1 == (True, ())
    result2 = invariants[1]()
    assert result2 == (False, ("base2",))

def test_store_invariants_includes_current_dict():
    def local_invariant(*args, **kwargs):
        return (True, ())
    bases = ()
    dct = {'__invariant__': local_invariant}
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    assert len(invariants) == 1
    result = invariants[0]()
    assert result == (True, ())

def test_store_invariants_handles_duplicate_inheritance():
    class Base:
        __invariant__ = lambda *args, **kwargs: (True, ())
    class Child1(Base):
        pass
    class Child2(Base):
        pass
    class GrandChild(Child1, Child2):
        pass
    bases = (Child1, Child2)
    dct = {}
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    assert len(invariants) == 1
    result = invariants[0]()
    assert result == (True, ())


# LLM-generated content at query #14
#--------------------------

def test_precord_new_creates_instance_with_special_attributes():
    class TestRecord(PRecord):
        pass
    instance = TestRecord(_precord_size=0, _precord_buckets=pvector().extend([]))
    assert isinstance(instance, TestRecord)
    assert instance._size == 0
    assert instance._buckets == pvector().extend([])

def test_precord_new_uses_evolver_for_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {}
    instance = TestRecord(field1=42, field2='test')
    assert instance['field1'] == 42
    assert instance['field2'] == 'test'

def test_precord_new_applies_initial_values_from_class():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {'field1': lambda: 100, 'field2': 'default'}
    instance = TestRecord()
    assert instance['field1'] == 100
    assert instance['field2'] == 'default'

def test_precord_new_overrides_initial_values_with_kwargs():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {'field1': lambda: 100, 'field2': 'default'}
    instance = TestRecord(field1=200)
    assert instance['field1'] == 200
    assert instance['field2'] == 'default'

def test_precord_new_handles_factory_fields_parameter():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, factory=lambda x: x * 2)}
    instance = TestRecord(_factory_fields={field(type=int, factory=lambda x: x * 2)}, field1=21)
    assert instance['field1'] == 42

def test_precord_new_handles_ignore_extra_parameter():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
    instance = TestRecord(_ignore_extra=True, field1=1, extra_field=2)
    assert instance['field1'] == 1
    assert 'extra_field' not in instance

def test_precord_new_raises_attribute_error_for_unknown_field():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
    try:
        TestRecord(unknown_field=1)
        assert False
    except AttributeError as e:
        assert "'unknown_field' is not among the specified fields for TestRecord" in str(e)

def test_precord_new_invokes_field_factory():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, factory=lambda x: x + 1)}
    instance = TestRecord(field1=5)
    assert instance['field1'] == 6

def test_precord_new_validates_field_type():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
    try:
        TestRecord(field1='not_an_int')
        assert False
    except TypeError:
        pass

def test_precord_new_enforces_field_invariants():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, invariant=lambda x: (x > 0, 'ERR_POSITIVE'))}
    try:
        TestRecord(field1=-1)
        assert False
    except InvariantException as e:
        assert 'ERR_POSITIVE' in e.invariant_errors

def test_precord_new_enforces_mandatory_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_mandatory_fields = {'field1'}
    try:
        TestRecord(field2='test')
        assert False
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

def test_precord_new_enforces_global_invariants():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=int)}
        _precord_invariants = [lambda r: (r['field1'] + r['field2'] == 10, 'ERR_SUM')]
    try:
        TestRecord(field1=3, field2=4)
        assert False
    except InvariantException as e:
        assert 'ERR_SUM' in e.invariant_errors

def test_precord_new_returns_same_instance_if_no_changes():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
    instance1 = TestRecord(field1=1)
    instance2 = TestRecord(_precord_size=instance1._size, _precord_buckets=instance1._buckets)
    assert instance1 is instance2

def test_precord_new_with_factory_fields_none_uses_original_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, factory=lambda x: x * 2)}
    instance = TestRecord(_factory_fields=None, field1=21)
    assert instance['field1'] == 21

def test_precord_new_with_ignore_extra_false_raises_for_extra_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
    try:
        TestRecord(_ignore_extra=False, field1=1, extra_field=2)
        assert False
    except AttributeError as e:
        assert "'extra_field' is not among the specified fields for TestRecord" in str(e)


# LLM-generated content at query #15
#--------------------------

def test_serialize_with_no_serializers():
    class TestRecord(PRecord):
        _precord_fields = {'field1': None, 'field2': None}
    record = TestRecord(field1='value1', field2=42)
    result = record.serialize()
    expected = {'field1': 'value1', 'field2': 42}
    assert result == expected

def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f'custom_{value}'
    class TestRecord(PRecord):
        _precord_fields = {'field1': type('Field', (), {'serializer': custom_serializer})(), 'field2': None}
    record = TestRecord(field1='value1', field2=42)
    result = record.serialize()
    expected = {'field1': 'custom_value1', 'field2': 42}
    assert result == expected

def test_serialize_with_format_parameter():
    def custom_serializer(format, value):
        return f'{format}_{value}'
    class TestRecord(PRecord):
        _precord_fields = {'field1': type('Field', (), {'serializer': custom_serializer})(), 'field2': None}
    record = TestRecord(field1='value1', field2=42)
    result = record.serialize(format='fmt')
    expected = {'field1': 'fmt_value1', 'field2': 42}
    assert result == expected

def test_serialize_with_multiple_fields_and_serializers():
    def serializer1(format, value):
        return value * 2
    def serializer2(format, value):
        return value + 10
    class TestRecord(PRecord):
        _precord_fields = {
            'field1': type('Field', (), {'serializer': serializer1})(),
            'field2': type('Field', (), {'serializer': serializer2})(),
            'field3': None
        }
    record = TestRecord(field1=5, field2=20, field3='test')
    result = record.serialize()
    expected = {'field1': 10, 'field2': 30, 'field3': 'test'}
    assert result == expected

def test_serialize_empty_record():
    class TestRecord(PRecord):
        _precord_fields = {}
    record = TestRecord()
    result = record.serialize()
    expected = {}
    assert result == expected


# LLM-generated content at query #16
#--------------------------

def test_precord_repr_with_single_field():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'name': field(type=str)}
        _precord_initial_values = {}
    record = TestRecord(name='Alice')
    result = repr(record)
    expected = "TestRecord(name='Alice')"
    assert result == expected

def test_precord_repr_with_multiple_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(type=int), 'y': field(type=str)}
        _precord_initial_values = {}
    record = TestRecord(x=10, y='test')
    result = repr(record)
    expected = "TestRecord(x=10, y='test')"
    assert result == expected

def test_precord_repr_with_no_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
        _precord_initial_values = {}
    record = TestRecord()
    result = repr(record)
    expected = "TestRecord()"
    assert result == expected

def test_precord_repr_with_nested_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'data': field(type=dict), 'count': field(type=int)}
        _precord_initial_values = {}
    record = TestRecord(data={'key': 'value'}, count=5)
    result = repr(record)
    expected = "TestRecord(data={'key': 'value'}, count=5)"
    assert result == expected

def test_precord_repr_with_special_characters_in_string():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'text': field(type=str)}
        _precord_initial_values = {}
    record = TestRecord(text='line1\nline2')
    result = repr(record)
    expected = "TestRecord(text='line1\\nline2')"
    assert result == expected


# LLM-generated content at query #17
#--------------------------

def test___new___creates_record_with_special_attributes():
    class TestRecord(PRecord):
        pass
    record = TestRecord(_precord_size=0, _precord_buckets=pvector().extend([]))
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test___new___creates_record_with_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
    record = TestRecord(x=10)
    assert record['x'] == 10

def test___new___uses_initial_values_from_class():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': 5}
    record = TestRecord()
    assert record['x'] == 5

def test___new___overrides_class_initial_values_with_kwargs():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': 5}
    record = TestRecord(x=10)
    assert record['x'] == 10

def test___new___handles_callable_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': lambda: 7}
    record = TestRecord()
    assert record['x'] == 7

def test___new___raises_attribute_error_for_unknown_field():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
    try:
        TestRecord(y=10)
        assert False
    except AttributeError as e:
        assert "'y' is not among the specified fields for TestRecord" in str(e)

def test___new___applies_factory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int, factory=int)}
    record = TestRecord(_factory_fields={'x'}, x='10')
    assert record['x'] == 10

def test___new___applies_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
    record = TestRecord(_ignore_extra=True, x=10, y=20)
    assert record['x'] == 10
    assert 'y' not in record

def test___new___raises_invariant_exception_for_missing_mandatory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(mandatory=True), 'y': field()}
        _precord_mandatory_fields = {'x'}
    try:
        TestRecord(y=10)
        assert False
    except InvariantException as e:
        assert 'TestRecord.x' in e.missing_fields

def test___new___raises_invariant_exception_for_field_invariant():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(invariant=lambda x: (x > 0, 'x positive'))}
    try:
        TestRecord(x=-5)
        assert False
    except InvariantException as e:
        assert 'x positive' in e.invariant_errors

def test___new___raises_invariant_exception_for_global_invariant():
    def global_inv(rec):
        return rec.get('x', 0) + rec.get('y', 0) > 0, 'sum positive'
    class TestRecord(PRecord):
        _precord_fields = {'x': field(), 'y': field()}
        _precord_invariants = [global_inv]
    try:
        TestRecord(x=-5, y=-5)
        assert False
    except InvariantException as e:
        assert 'sum positive' in e.invariant_errors


# LLM-generated content at query #18
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()
        assert str(e) == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"

def test_persistent_raises_invariant_exception_when_missing_fields_present():
    class MockClass:
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []

    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('MockClass.field1',)
        assert str(e) == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"

def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present():
    class MockClass:
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []

    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['MockClass.field3']
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('MockClass.field3', 'MockClass.field1', 'MockClass.field2')
        assert str(e) == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"

def test_persistent_does_not_raise_when_no_errors_or_missing_fields():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    result = evolver.persistent()
    assert isinstance(result, MockClass)


# LLM-generated content at query #19
#--------------------------

def test_precord_constructor_with_special_attributes():
    class TestRecord(PRecord):
        __slots__ = ()
    record = TestRecord(_precord_size=0, _precord_buckets=())
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=str), 'field2': field(type=int)}
    record = TestRecord(field1='test', field2=42)
    assert record['field1'] == 'test'
    assert record['field2'] == 42

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=str)}
    record = TestRecord(_factory_fields={'field1': 'factory_value'})
    assert record['field1'] == 'factory_value'

def test_precord_constructor_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=str)}
    record = TestRecord(field1='value1', extra_field='extra', _ignore_extra=True)
    assert record['field1'] == 'value1'
    assert 'extra_field' not in record

def test_precord_constructor_with_callable_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=str)}
        _precord_initial_values = {'field1': lambda: 'default'}
    record = TestRecord()
    assert record['field1'] == 'default'

def test_precord_constructor_overrides_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=str)}
        _precord_initial_values = {'field1': lambda: 'default'}
    record = TestRecord(field1='override')
    assert record['field1'] == 'override'

def test_precord_constructor_empty():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
    record = TestRecord()
    assert len(record) == 0


# LLM-generated content at query #20
#--------------------------

def test_set_with_valid_field_and_factory():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        factory = lambda x: x
        type = (int,)
        invariant = lambda self, value: (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    assert evolver.original_pmap['key'] == 5

def test_set_with_factory_exception():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def factory(self, value):
            raise InvariantException((), (), '')
        type = (int,)
        invariant = lambda self, value: (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    assert 'key' not in evolver.original_pmap

def test_set_with_type_check_failure():
    from pyrsistent import InvariantException, CheckedType, PTypeError
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        factory = lambda x: x
        type = (int,)
        invariant = lambda self, value: (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
        __name__ = 'MockDestinationCls'
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    try:
        evolver.set('key', 'string')
        assert False
    except PTypeError:
        assert True

def test_set_with_invariant_failure():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        factory = lambda x: x
        type = (int,)
        invariant = lambda self, value: (False, 'error') if value < 0 else (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', -1)
    assert evolver._invariant_error_codes == ['error']

def test_set_with_ignore_extra_compliant_factory():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    import inspect
    def factory_with_ignore_extra(value, ignore_extra=False):
        return value
    class MockField:
        factory = factory_with_ignore_extra
        type = (CheckedType,)
        invariant = lambda self, value: (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap, _ignore_extra=True)
    evolver.set('key', 5)
    assert evolver.original_pmap['key'] == 5

def test_set_with_non_existent_field():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockDestinationCls:
        _precord_fields = {}
        __name__ = 'MockDestinationCls'
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    try:
        evolver.set('nonexistent', 5)
        assert False
    except AttributeError:
        assert True

def test_set_with_factory_fields_skipped():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        factory = lambda x: x * 2
        type = (int,)
        invariant = lambda self, value: (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    original_pmap = {}
    factory_fields = set()
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap, _factory_fields=factory_fields)
    evolver.set('key', 5)
    assert evolver.original_pmap['key'] == 5


# LLM-generated content at query #21
#--------------------------

def test_precord_repr_returns_correct_format():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {}
    record = TestRecord(field1='value1', field2=123)
    result = repr(record)
    expected_start = "TestRecord("
    expected_end = ")"
    assert result.startswith(expected_start)
    assert result.endswith(expected_end)
    assert 'field1=' in result
    assert "'value1'" in result or '"value1"' in result
    assert 'field2=' in result
    assert '123' in result
    assert result.count(',') == 1


# LLM-generated content at query #22
#--------------------------

```python
def test_precord_initial_values_condition_true():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': 42}
    result = TestRecord()
    assert result['x'] == 42


# LLM-generated content at query #23
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import CheckedType
    import inspect

    class MockField:
        def __init__(self, type_set, factory_params):
            self.type = type_set
            self.factory = lambda **kwargs: None
            self.factory.__signature__ = inspect.signature(lambda **kwargs: None).replace(parameters=factory_params)

    field_type = {CheckedType}
    factory_params = [inspect.Parameter('ignore_extra', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    field = MockField(field_type, factory_params)
    ignore_extra = True
    result = is_field_ignore_extra_complaint(CheckedType, field, ignore_extra)
    assert result == True


# LLM-generated content at query #24
#--------------------------

def test_persistent_returns_instance_of_destination_class():
    class TestRecord(PRecord):
        field = field(type=int)
    evolver = TestRecord().evolver()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)

def test_persistent_raises_invariant_exception_on_field_invariant_failure():
    class TestRecord(PRecord):
        field = field(type=int, invariant=lambda x: (x > 0, 'ERR'))
    evolver = TestRecord().evolver()
    evolver.set('field', -1)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('ERR',)

def test_persistent_raises_invariant_exception_on_missing_mandatory_fields():
    class TestRecord(PRecord):
        field = field(type=int, mandatory=True)
    evolver = TestRecord().evolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.missing_fields == ('TestRecord.field',)

def test_persistent_calls_check_global_invariants():
    class TestRecord(PRecord):
        field = field(type=int)
        def _invariant(self):
            return (self.field > 0, 'GLOBAL_ERR')
    evolver = TestRecord().evolver()
    evolver.set('field', -1)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('GLOBAL_ERR',)

def test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type():
    class TestRecord(PRecord):
        field = field(type=int)
    original = TestRecord(field=1)
    evolver = original.evolver()
    result = evolver.persistent()
    assert result is original

def test_persistent_creates_new_instance_if_dirty():
    class TestRecord(PRecord):
        field = field(type=int)
    original = TestRecord(field=1)
    evolver = original.evolver()
    evolver.set('field', 2)
    result = evolver.persistent()
    assert result is not original
    assert result.field == 2

def test_persistent_aggregates_multiple_invariant_errors():
    class TestRecord(PRecord):
        field1 = field(type=int, invariant=lambda x: (x > 0, 'ERR1'))
        field2 = field(type=int, invariant=lambda x: (x < 0, 'ERR2'))
    evolver = TestRecord().evolver()
    evolver.set('field1', -1)
    evolver.set('field2', 1)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert set(e.invariant_errors) == {'ERR1', 'ERR2'}

def test_persistent_aggregates_missing_fields_and_invariant_errors():
    class TestRecord(PRecord):
        field1 = field(type=int, mandatory=True)
        field2 = field(type=int, invariant=lambda x: (x > 0, 'ERR'))
    evolver = TestRecord().evolver()
    evolver.set('field2', -1)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.missing_fields == ('TestRecord.field1',)
        assert e.invariant_errors == ('ERR',)


# LLM-generated content at query #25
#--------------------------

```python
def test_persistent_when_cls_has_mandatory_fields():
    class MockClass:
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'MockClass'
    
    evolver = _PRecordEvolver(MockClass, {})
    evolver._destination_cls = MockClass
    evolver._missing_fields = []
    evolver._invariant_error_codes = []
    result = {}
    result.keys = lambda: {'field1'}
    evolver.persistent = lambda: result
    evolver.is_dirty = lambda: False
    condition = MockClass._precord_mandatory_fields
    assert condition


# LLM-generated content at query #26
#--------------------------

def test_set_with_valid_field_and_factory():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=lambda x: x*2, type=(int,), invariant=lambda x: (True, None))}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    result = evolver.persistent()
    assert result['key'] == 10

def test_set_with_valid_field_and_no_factory():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=None, type=(int,), invariant=lambda x: (True, None))}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap, _factory_fields=set())
    evolver.set('key', 5)
    result = evolver.persistent()
    assert result['key'] == 5

def test_set_with_invalid_field():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockDestinationCls:
        _precord_fields = {}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    try:
        evolver.set('invalid_key', 5)
        assert False
    except AttributeError as e:
        assert "'invalid_key' is not among the specified fields" in str(e)

def test_set_with_factory_invariant_exception():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    def factory_raises(x):
        raise InvariantException(invariant_errors=('error',), missing_fields=('missing',))
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=factory_raises, type=(int,), invariant=lambda x: (True, None))}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'error' in e.invariant_errors
        assert 'missing' in e.missing_fields

def test_set_with_field_invariant_failure():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=lambda x: x, type=(int,), invariant=lambda x: (False, 'invariant_error'))}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'invariant_error' in e.invariant_errors

def test_set_with_type_check_failure():
    from pyrsistent import PTypeError, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=lambda x: x, type=(int,), invariant=lambda x: (True, None))}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    try:
        evolver.set('key', 'not_an_int')
        assert False
    except PTypeError as e:
        assert 'Invalid type for field' in str(e)

def test_set_with_ignore_extra_complaint():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    import inspect
    class MockField:
        def __init__(self, factory, type, invariant):
            self.factory = factory
            self.type = type
            self.invariant = invariant
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x*3
    factory_with_ignore_extra.__signature__ = inspect.signature(lambda x, ignore_extra=False: None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField(factory=factory_with_ignore_extra, type=(CheckedType,), invariant=lambda x: (True, None))}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap, _ignore_extra=True)
    evolver.set('key', 5)
    result = evolver.persistent()
    assert result['key'] == 15


# LLM-generated content at query #27
#--------------------------

```python
def test_new_without_special_attributes():
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_initial_values = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    record = TestRecord()
    assert isinstance(record, TestRecord)
    assert len(record) == 0


# LLM-generated content at query #28
#--------------------------

def test_serialize_with_custom_serializer():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        name = field(serializer=lambda fmt, v: v.upper())
        age = field()
    record = TestRecord(name="alice", age=30)
    result = record.serialize()
    expected = {"name": "ALICE", "age": 30}
    assert result == expected

def test_serialize_without_custom_serializer():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        name = field()
        age = field()
    record = TestRecord(name="alice", age=30)
    result = record.serialize()
    expected = {"name": "alice", "age": 30}
    assert result == expected

def test_serialize_with_format_argument():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        value = field(serializer=lambda fmt, v: f"{fmt}:{v}")
    record = TestRecord(value=42)
    result = record.serialize("fmt")
    expected = {"value": "fmt:42"}
    assert result == expected

def test_serialize_with_none_format():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        value = field(serializer=lambda fmt, v: v if fmt is None else fmt)
    record = TestRecord(value=42)
    result = record.serialize()
    expected = {"value": 42}
    assert result == expected

def test_serialize_empty_record():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        pass
    record = TestRecord()
    result = record.serialize()
    expected = {}
    assert result == expected

def test_serialize_with_multiple_fields_mixed_serializers():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        a = field(serializer=lambda fmt, v: v * 2)
        b = field()
        c = field(serializer=lambda fmt, v: str(v))
    record = TestRecord(a=10, b="test", c=3.14)
    result = record.serialize()
    expected = {"a": 20, "b": "test", "c": "3.14"}
    assert result == expected


# LLM-generated content at query #29
#--------------------------

def test___new___sets_fields_and_invariants():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    class MockPField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class Base1:
        _precord_fields = {'base1_field': MockField(False, PFIELD_NO_INITIAL)}
        __invariant__ = lambda self: (True, ())
    class Base2:
        _precord_fields = {'base2_field': MockField(True, PFIELD_NO_INITIAL)}
        __invariant__ = lambda self: (False, ('error',))
    dct = {'custom_field': MockPField(False, 'default')}
    bases = (Base1, Base2)
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_fields' in result.__dict__
    fields = result.__dict__['_precord_fields']
    assert 'base1_field' in fields
    assert 'base2_field' in fields
    assert 'custom_field' in fields
    assert fields['custom_field'].initial == 'default'
    assert '_precord_invariants' in result.__dict__
    invariants = result.__dict__['_precord_invariants']
    assert len(invariants) == 2
    assert invariants[0](None) == (True, ())
    assert invariants[1](None) == (False, ('error',))
    assert '_precord_mandatory_fields' in result.__dict__
    mandatory = result.__dict__['_precord_mandatory_fields']
    assert mandatory == {'base2_field'}
    assert '_precord_initial_values' in result.__dict__
    initials = result.__dict__['_precord_initial_values']
    assert initials == {'custom_field': 'default'}
    assert '__slots__' in result.__dict__
    assert result.__dict__['__slots__'] == ()


# LLM-generated content at query #30
#--------------------------

```python
def test_persistent_when_invariant_error_codes_present_raises_exception():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_fields = {}
        __name__ = "MockClass"
    
    class MockPMap:
        _buckets = {}
        _size = 0
        def keys(self):
            return []
    
    evolver = _PRecordEvolver(MockClass, {})
    evolver._invariant_error_codes = ["error1"]
    evolver._missing_fields = []
    evolver._original_pmap = MockPMap()
    evolver._destination_cls = MockClass
    evolver.is_dirty = lambda: False
    evolver._buckets = {}
    evolver._size = 0
    exception_raised = False
    try:
        evolver.persistent()
    except InvariantException as e:
        exception_raised = True
        assert e.invariant_errors == ("error1",)
        assert e.missing_fields == ()
        assert str(e) == "Field invariant failed"
    assert exception_raised


# LLM-generated content at query #31
#--------------------------

def test___new___sets_fields_and_invariants():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    field1 = MockField(mandatory=True, initial=10)
    field2 = MockField(mandatory=False, initial=None)
    dct = {'field1': field1, 'field2': field2}
    bases = ()
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_fields' in dct
    assert dct['_precord_fields']['field1'] == field1
    assert dct['_precord_fields']['field2'] == field2
    assert 'field1' not in dct
    assert 'field2' not in dct
    assert '_precord_invariants' in dct
    assert isinstance(dct['_precord_invariants'], tuple)
    assert '_precord_mandatory_fields' in dct
    assert dct['_precord_mandatory_fields'] == {'field1'}
    assert '_precord_initial_values' in dct
    assert dct['_precord_initial_values'] == {'field1': 10}
    assert '__slots__' in dct
    assert dct['__slots__'] == ()
    assert result is not None
def test___new___inherits_fields_and_invariants():
    class Base:
        _precord_fields = {'base_field': MockField(mandatory=False, initial=5)}
        __invariant__ = lambda self: (True, ())
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    dct = {'new_field': MockField(mandatory=True, initial=20)}
    bases = (Base,)
    result = _PRecordMeta.__new__(_PRecordMeta, 'DerivedClass', bases, dct)
    assert '_precord_fields' in dct
    assert 'base_field' in dct['_precord_fields']
    assert 'new_field' in dct['_precord_fields']
    assert dct['_precord_fields']['base_field'].initial == 5
    assert dct['_precord_fields']['new_field'].initial == 20
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 1
    assert '_precord_mandatory_fields' in dct
    assert dct['_precord_mandatory_fields'] == {'new_field'}
    assert '_precord_initial_values' in dct
    assert dct['_precord_initial_values'] == {'base_field': 5, 'new_field': 20}
    assert result is not None
def test___new___handles_no_fields():
    dct = {}
    bases = ()
    result = _PRecordMeta.__new__(_PRecordMeta, 'EmptyClass', bases, dct)
    assert '_precord_fields' in dct
    assert dct['_precord_fields'] == {}
    assert '_precord_invariants' in dct
    assert dct['_precord_invariants'] == ()
    assert '_precord_mandatory_fields' in dct
    assert dct['_precord_mandatory_fields'] == set()
    assert '_precord_initial_values' in dct
    assert dct['_precord_initial_values'] == {}
    assert result is not None
def test___new___wraps_invariants():
    def invariant1(self):
        return (False, 'error1')
    def invariant2(self):
        return [(True, ()), (False, 'error2')]
    class Base:
        __invariant__ = invariant1
    dct = {'__invariant__': invariant2}
    bases = (Base,)
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    wrapped1 = invariants[0]
    wrapped2 = invariants[1]
    assert wrapped1(None) == (False, 'error1')
    assert wrapped2(None) == (False, ('error2',))
    assert result is not None
def test___new___raises_on_non_callable_invariant():
    class Base:
        __invariant__ = 'not callable'
    dct = {}
    bases = (Base,)
    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
        assert False
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_persistent_returns_instance_of_destination_class():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    result = evolver.persistent()
    assert isinstance(result, MockClass)

def test_persistent_raises_invariant_exception_on_missing_mandatory_fields():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'mandatory_field'}
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'mandatory_field' in e.missing_fields[0]

def test_persistent_raises_invariant_exception_on_field_invariant_errors():
    class MockClass:
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (False, 'error')})()}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    evolver.set('field', 'value')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'error' in e.invariant_errors

def test_persistent_raises_invariant_exception_on_global_invariant_failure():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [lambda x: (False, 'global_error')]
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors

def test_persistent_returns_same_instance_if_not_dirty_and_already_correct_class():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    original = MockClass()
    evolver = _PRecordEvolver(MockClass, original)
    result = evolver.persistent()
    assert result is original

def test_persistent_creates_new_instance_if_dirty():
    class MockClass:
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (True, None)})()}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    evolver.set('field', 'value')
    result = evolver.persistent()
    assert isinstance(result, MockClass)
    assert result._buckets is not None


# LLM-generated content at query #2
#--------------------------

def test_set_with_valid_field_and_value():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    import inspect
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {}
    def mock_factory(x):
        return x
    def mock_invariant(value):
        return (True, None)
    field = MockField(type=(int,), factory=mock_factory, invariant=mock_invariant)
    MockDestinationCls._precord_fields['key'] = field
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    assert evolver._original_pmap['key'] == 5

def test_set_with_field_factory_ignore_extra():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    import inspect
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {}
    def mock_factory(x, ignore_extra=False):
        return x
    def mock_invariant(value):
        return (True, None)
    field = MockField(type=(CheckedType,), factory=mock_factory, invariant=mock_invariant)
    MockDestinationCls._precord_fields['key'] = field
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap, _ignore_extra=True)
    evolver.set('key', 5)
    assert evolver._original_pmap['key'] == 5

def test_set_invokes_check_type():
    from pyrsistent import PTypeError, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    import inspect
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {}
    def mock_factory(x):
        return x
    def mock_invariant(value):
        return (True, None)
    field = MockField(type=(int,), factory=mock_factory, invariant=mock_invariant)
    MockDestinationCls._precord_fields['key'] = field
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    try:
        evolver.set('key', 'not_an_int')
        assert False
    except PTypeError:
        assert True

def test_set_invariant_fails():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    import inspect
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {}
    def mock_factory(x):
        return x
    def mock_invariant(value):
        return (False, 'error_code')
    field = MockField(type=(int,), factory=mock_factory, invariant=mock_invariant)
    MockDestinationCls._precord_fields['key'] = field
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    assert 'error_code' in evolver._invariant_error_codes

def test_set_factory_raises_invariant_exception():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    import inspect
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {}
    def mock_factory(x):
        raise InvariantException(['invariant_error'], ['missing_field'], 'message')
    def mock_invariant(value):
        return (True, None)
    field = MockField(type=(int,), factory=mock_factory, invariant=mock_invariant)
    MockDestinationCls._precord_fields['key'] = field
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    evolver.set('key', 5)
    assert 'invariant_error' in evolver._invariant_error_codes
    assert 'missing_field' in evolver._missing_fields

def test_set_with_non_existent_field():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    import inspect
    class MockDestinationCls:
        _precord_fields = {}
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap)
    try:
        evolver.set('nonexistent', 5)
        assert False
    except AttributeError as e:
        assert "'nonexistent' is not among the specified fields" in str(e)

def test_set_with_factory_fields_skips_factory():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    import inspect
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {}
    factory_called = []
    def mock_factory(x):
        factory_called.append(True)
        return x
    def mock_invariant(value):
        return (True, None)
    field = MockField(type=(int,), factory=mock_factory, invariant=mock_invariant)
    MockDestinationCls._precord_fields['key'] = field
    original_pmap = {}
    evolver = _PRecordEvolver(MockDestinationCls, original_pmap, _factory_fields=set())
    evolver.set('key', 5)
    assert len(factory_called) == 0
    assert evolver._original_pmap['key'] == 5


# LLM-generated content at query #3
#--------------------------

def test___new___creates_record_with_special_attributes():
    class TestRecord(PRecord):
        pass
    record = TestRecord(_precord_size=0, _precord_buckets=pvector().extend([]))
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test___new___creates_record_with_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {'field1': 10, 'field2': 'default'}
    record = TestRecord()
    assert record['field1'] == 10
    assert record['field2'] == 'default'

def test___new___creates_record_with_overridden_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {'field1': 10, 'field2': 'default'}
    record = TestRecord(field1=20)
    assert record['field1'] == 20
    assert record['field2'] == 'default'

def test___new___creates_record_with_callable_initial_value():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
        _precord_initial_values = {'field1': lambda: 42}
    record = TestRecord()
    assert record['field1'] == 42

def test___new___creates_record_with_factory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, factory=lambda x: x * 2)}
    record = TestRecord(_factory_fields={field(type=int, factory=lambda x: x * 2)}, field1=5)
    assert record['field1'] == 10

def test___new___creates_record_with_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
    record = TestRecord(_ignore_extra=True, field1=1, extra_field=2)
    assert record['field1'] == 1
    assert 'extra_field' not in record

def test___new___raises_attribute_error_for_unknown_field():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
    try:
        TestRecord(field2=2)
        assert False
    except AttributeError as e:
        assert "'field2' is not among the specified fields for TestRecord" in str(e)

def test___new___raises_invariant_exception_for_invalid_field():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, invariant=lambda x: (x > 0, 'ERR'))}
    try:
        TestRecord(field1=-1)
        assert False
    except InvariantException as e:
        assert 'ERR' in e.invariant_errors

def test___new___raises_invariant_exception_for_missing_mandatory_field():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, mandatory=True)}
        _precord_mandatory_fields = {'field1'}
    try:
        TestRecord()
        assert False
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

def test___new___creates_record_with_global_invariant():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=int)}
        _precord_invariants = [lambda r: (r['field1'] + r['field2'] == 10, 'SUM_ERR')]
    try:
        TestRecord(field1=3, field2=8)
        assert False
    except InvariantException as e:
        assert 'SUM_ERR' in e.invariant_errors


# LLM-generated content at query #4
#--------------------------

def test___new___sets_fields_correctly():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    class MockPField(MockField):
        pass
    _PField = MockPField
    PFIELD_NO_INITIAL = object()
    class Base1:
        _precord_fields = {'base1_field': MockField()}
    class Base2:
        _precord_fields = {'base2_field': MockField()}
    dct = {'custom_field': MockField()}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    assert '_precord_fields' in dct
    assert 'base1_field' in dct['_precord_fields']
    assert 'base2_field' in dct['_precord_fields']
    assert 'custom_field' in dct['_precord_fields']

def test___new___moves_pfield_instances_to_fields():
    class MockPField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    _PField = MockPField
    PFIELD_NO_INITIAL = object()
    pfield_instance = MockPField()
    dct = {'pfield_key': pfield_instance}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    assert 'pfield_key' not in dct
    assert 'pfield_key' in dct['_precord_fields']
    assert dct['_precord_fields']['pfield_key'] is pfield_instance

def test___new___stores_invariants_correctly():
    def invariant1(x):
        return True, None
    def invariant2(x):
        return False, 'error'
    class BaseWithInvariant:
        __invariant__ = invariant1
    dct = {'__invariant__': invariant2}
    bases = (BaseWithInvariant,)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 2
    assert dct['_precord_invariants'][0](None) == (True, None)
    assert dct['_precord_invariants'][1](None) == (False, ('error',))

def test___new___wraps_invariants_that_return_multiple_results():
    def multi_invariant(x):
        return [(True, None), (False, 'err1'), (False, 'err2')]
    dct = {'__invariant__': multi_invariant}
    bases = ()
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    wrapped = dct['_precord_invariants'][0]
    result = wrapped(None)
    assert result == (False, ('err1', 'err2'))

def test___new___raises_type_error_for_non_callable_invariants():
    dct = {'__invariant__': 'not a callable'}
    bases = ()
    try:
        store_invariants(dct, bases, '_precord_invariants', '__invariant__')
        assert False
    except TypeError:
        pass

def test___new___sets_mandatory_fields():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    class MockPField(MockField):
        pass
    _PField = MockPField
    PFIELD_NO_INITIAL = object()
    mandatory_field = MockField(mandatory=True)
    non_mandatory_field = MockField(mandatory=False)
    dct = {'mandatory': mandatory_field, 'non_mandatory': non_mandatory_field}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    dct['_precord_mandatory_fields'] = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
    assert dct['_precord_mandatory_fields'] == {'mandatory'}

def test___new___sets_initial_values():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    class MockPField(MockField):
        pass
    _PField = MockPField
    PFIELD_NO_INITIAL = object()
    field_with_initial = MockField(initial='default')
    field_without_initial = MockField(initial=PFIELD_NO_INITIAL)
    dct = {'with_initial': field_with_initial, 'without_initial': field_without_initial}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    dct['_precord_initial_values'] = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
    assert dct['_precord_initial_values'] == {'with_initial': 'default'}

def test___new___sets_empty_slots():
    dct = {}
    bases = ()
    dct['__slots__'] = ()
    assert dct['__slots__'] == ()

def test___new___inherits_fields_and_invariants():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    class MockPField(MockField):
        pass
    _PField = MockPField
    PFIELD_NO_INITIAL = object()
    def base_invariant(x):
        return True, None
    class GrandBase:
        _precord_fields = {'grand_field': MockField()}
        __invariant__ = base_invariant
    class Parent(GrandBase):
        _precord_fields = {'parent_field': MockField()}
    dct = {'child_field': MockField()}
    bases = (Parent,)
    set_fields(dct, bases, name='_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert 'grand_field' in dct['_precord_fields']
    assert 'parent_field' in dct['_precord_fields']
    assert 'child_field' in dct['_precord_fields']
    assert len(dct['_precord_invariants']) == 1
    assert dct['_precord_invariants'][0](None) == (True, None)


# LLM-generated content at query #5
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_fields = {}
        __name__ = "MockClass"

    class MockPMap:
        _buckets = {}
        _size = 0
        def keys(self):
            return []

    class MockEvolver(_PRecordEvolver):
        def __init__(self):
            self._destination_cls = MockClass
            self._invariant_error_codes = ["error1"]
            self._missing_fields = []
            self._factory_fields = None
            self._ignore_extra = False
            self._original_pmap = MockPMap()
            self._buckets = {}
            self._size = 0

        def is_dirty(self):
            return False

        def persistent(self):
            return self._original_pmap

    evolver = MockEvolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ("error1",)
        assert e.missing_fields == ()
        assert str(e) == "Field invariant failed"


# LLM-generated content at query #6
#--------------------------

```python
def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        _precord_fields = {}

    record = TestRecord()
    assert '_precord_size' not in record._precord_buckets
    assert '_precord_buckets' not in record._precord_buckets


# LLM-generated content at query #7
#--------------------------

def test_precord_new_with_special_attributes():
    class TestRecord(PRecord):
        pass
    record = TestRecord(_precord_size=0, _precord_buckets=pvector().extend([]))
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        x = field()
        y = field()
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        x = field(factory=int)
    record = TestRecord(x="5", _factory_fields=None)
    assert record['x'] == 5

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        x = field()
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record

def test_precord_new_without_ignore_extra_raises():
    class TestRecord(PRecord):
        x = field()
    try:
        TestRecord(x=1, y=2)
        assert False
    except AttributeError:
        pass

def test_precord_new_with_initial_values_from_class():
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=lambda: 20)
    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20

def test_precord_new_overrides_initial_values():
    class TestRecord(PRecord):
        x = field(initial=10)
    record = TestRecord(x=30)
    assert record['x'] == 30

def test_precord_new_with_mandatory_fields_missing():
    class TestRecord(PRecord):
        x = field(mandatory=True)
    try:
        TestRecord()
        assert False
    except InvariantException as e:
        assert 'TestRecord.x' in e.missing_fields

def test_precord_new_with_invariant_failure():
    class TestRecord(PRecord):
        x = field(invariant=lambda x: (x > 0, 'x must be positive'))
    try:
        TestRecord(x=-5)
        assert False
    except InvariantException as e:
        assert 'x must be positive' in e.invariant_errors

def test_precord_new_with_global_invariant_failure():
    class TestRecord(PRecord):
        x = field()
        y = field()
        @invariant(lambda r: r['x'] < r['y'])
        def xy_invariant(self):
            return self['x'] < self['y']
    try:
        TestRecord(x=10, y=5)
        assert False
    except InvariantException:
        pass

def test_precord_new_with_factory_and_invariant():
    class TestRecord(PRecord):
        x = field(factory=int, invariant=lambda x: (x > 0, 'x positive'))
    record = TestRecord(x="5")
    assert record['x'] == 5

def test_precord_new_with_factory_exception():
    class TestRecord(PRecord):
        x = field(factory=lambda v: 1/0)
    try:
        TestRecord(x=5)
        assert False
    except InvariantException:
        pass

def test_precord_new_returns_same_instance_if_no_changes():
    class TestRecord(PRecord):
        x = field()
    record1 = TestRecord(x=1)
    record2 = TestRecord(_precord_size=record1._size, _precord_buckets=record1._buckets)
    assert record1 is record2

def test_precord_new_with_check_type_violation():
    class TestRecord(PRecord):
        x = field(type=int)
    try:
        TestRecord(x="string")
        assert False
    except PTypeError:
        pass

def test_precord_new_with_ignore_extra_and_factory():
    class TestRecord(PRecord):
        x = field(factory=int)
    record = TestRecord(x="10", y=20, _ignore_extra=True)
    assert record['x'] == 10
    assert 'y' not in record

def test_precord_new_with_factory_fields_list():
    class TestRecord(PRecord):
        x = field(factory=int)
        y = field()
    record = TestRecord(x="5", y=10, _factory_fields=[TestRecord._precord_fields['x']])
    assert record['x'] == 5
    assert record['y'] == 10

def test_precord_new_with_empty_initial():
    class TestRecord(PRecord):
        pass
    record = TestRecord()
    assert len(record) == 0

def test_precord_new_with_callable_initial():
    class TestRecord(PRecord):
        x = field(initial=lambda: 42)
    record = TestRecord()
    assert record['x'] == 42

def test_precord_new_with_multiple_fields_set():
    class TestRecord(PRecord):
        a = field()
        b = field()
        c = field()
    record = TestRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3


# LLM-generated content at query #8
#--------------------------

def test_new_without_precord_size_and_buckets():
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_initial_values = {}
    record = TestRecord()
    assert isinstance(record, TestRecord)
    assert record == {}

def test_new_with_factory_fields_and_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'a': field(type=int)}
        _precord_initial_values = {}
    record = TestRecord(a=1, _factory_fields={'a'}, _ignore_extra=True)
    assert record == {'a': 1}

def test_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'a': field(type=int), 'b': field(type=int)}
        _precord_initial_values = {'b': lambda: 2}
    record = TestRecord(a=1)
    assert record == {'a': 1, 'b': 2}

def test_new_with_kwargs_overrides_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'a': field(type=int), 'b': field(type=int)}
        _precord_initial_values = {'b': lambda: 2}
    record = TestRecord(a=1, b=3)
    assert record == {'a': 1, 'b': 3}

def test_new_with_no_special_attributes():
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_initial_values = {}
    record = TestRecord()
    assert '_precord_size' not in record.__dict__
    assert '_precord_buckets' not in record.__dict__

def test_new_with_regular_kwargs_only():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int)}
        _precord_initial_values = {}
    record = TestRecord(x=10)
    assert record == {'x': 10}

def test_new_with_empty_kwargs():
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_initial_values = {}
    record = TestRecord()
    assert record == {}

def test_new_with_multiple_fields():
    class TestRecord(PRecord):
        _precord_fields = {'a': field(type=str), 'b': field(type=int)}
        _precord_initial_values = {}
    record = TestRecord(a='test', b=42)
    assert record == {'a': 'test', 'b': 42}


# LLM-generated content at query #9
#--------------------------

def test_precord_constructor_without_special_attributes():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(), 'y': field()}
        _precord_initial_values = {}
    r = MyRecord(x=1, y=2)
    assert r['x'] == 1
    assert r['y'] == 2
    assert len(r) == 2

def test_precord_constructor_with_special_attributes():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(), 'y': field()}
        _precord_initial_values = {}
    size = 2
    buckets = [(hash('x'), 'x', 1), (hash('y'), 'y', 2)]
    r = MyRecord(_precord_size=size, _precord_buckets=buckets)
    assert r['x'] == 1
    assert r['y'] == 2
    assert len(r) == 2

def test_precord_constructor_with_initial_values():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(), 'y': field()}
        _precord_initial_values = {'x': lambda: 10, 'y': 20}
    r = MyRecord()
    assert r['x'] == 10
    assert r['y'] == 20

def test_precord_constructor_initial_values_overridden():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(), 'y': field()}
        _precord_initial_values = {'x': lambda: 10, 'y': 20}
    r = MyRecord(x=100)
    assert r['x'] == 100
    assert r['y'] == 20

def test_precord_constructor_with_factory_fields():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(), 'y': field()}
        _precord_initial_values = {}
    r = MyRecord(_factory_fields={'x': int}, x='5', y=2)
    assert r['x'] == 5
    assert r['y'] == 2

def test_precord_constructor_ignore_extra():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(), 'y': field()}
        _precord_initial_values = {}
    r = MyRecord(_ignore_extra=True, x=1, y=2, z=3)
    assert r['x'] == 1
    assert r['y'] == 2
    assert 'z' not in r

def test_precord_constructor_ignore_extra_false():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(), 'y': field()}
        _precord_initial_values = {}
    try:
        MyRecord(_ignore_extra=False, x=1, y=2, z=3)
        assert False
    except AttributeError:
        pass

def test_precord_constructor_empty_record():
    class EmptyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
        _precord_initial_values = {}
    r = EmptyRecord()
    assert len(r) == 0

def test_precord_constructor_with_callable_initial_value():
    call_count = [0]
    def initial_callable():
        call_count[0] += 1
        return call_count[0]
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': initial_callable}
    r1 = MyRecord()
    assert r1['x'] == 1
    r2 = MyRecord()
    assert r2['x'] == 2

def test_precord_constructor_initial_value_not_callable():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': 42}
    r = MyRecord()
    assert r['x'] == 42


# LLM-generated content at query #10
#--------------------------

def test_precord_repr_with_single_field():
    class TestRecord(PRecord):
        _precord_fields = {'name': field(type=str)}
    record = TestRecord(name="Alice")
    result = repr(record)
    expected = "TestRecord(name='Alice')"
    assert result == expected

def test_precord_repr_with_multiple_fields():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int), 'y': field(type=str)}
    record = TestRecord(x=10, y="test")
    result = repr(record)
    expected = "TestRecord(x=10, y='test')"
    assert result == expected

def test_precord_repr_with_no_fields():
    class TestRecord(PRecord):
        _precord_fields = {}
    record = TestRecord()
    result = repr(record)
    expected = "TestRecord()"
    assert result == expected

def test_precord_repr_with_nested_values():
    class TestRecord(PRecord):
        _precord_fields = {'data': field(type=dict)}
    record = TestRecord(data={'key': 'value'})
    result = repr(record)
    expected = "TestRecord(data={'key': 'value'})"
    assert result == expected

def test_precord_repr_with_integer_field_names():
    class TestRecord(PRecord):
        _precord_fields = {'1': field(type=int), '2': field(type=int)}
    record = TestRecord(**{'1': 100, '2': 200})
    result = repr(record)
    expected = "TestRecord(1=100, 2=200)"
    assert result == expected


# LLM-generated content at query #11
#--------------------------

def test___new___creates_precord_with_special_attributes():
    class TestRecord(PRecord):
        pass
    record = TestRecord(_precord_size=0, _precord_buckets=pvector().extend([]))
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test___new___uses_evolver_for_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
    record = TestRecord(field1=10, field2='test')
    assert record['field1'] == 10
    assert record['field2'] == 'test'

def test___new___applies_initial_values_from_class():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
        _precord_initial_values = {'field1': lambda: 5}
    record = TestRecord()
    assert record['field1'] == 5

def test___new___overrides_initial_values_with_kwargs():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
        _precord_initial_values = {'field1': lambda: 5}
    record = TestRecord(field1=10)
    assert record['field1'] == 10

def test___new___handles_factory_fields_parameter():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, factory=lambda x: x * 2)}
    record = TestRecord(field1=5, _factory_fields={'field1'})
    assert record['field1'] == 10

def test___new___handles_ignore_extra_parameter():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
    record = TestRecord(field1=5, extra_field=20, _ignore_extra=True)
    assert record['field1'] == 5
    assert 'extra_field' not in record

def test___new___raises_attribute_error_for_unknown_field():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
    try:
        TestRecord(field2=10)
        assert False
    except AttributeError as e:
        assert "'field2' is not among the specified fields for TestRecord" in str(e)

def test___new___raises_invariant_exception_for_invalid_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, invariant=lambda x: (x > 0, 'ERR1'))}
    try:
        TestRecord(field1=-5)
        assert False
    except InvariantException as e:
        assert 'ERR1' in e.invariant_errors

def test___new___raises_invariant_exception_for_missing_mandatory_field():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, mandatory=True)}
        _precord_mandatory_fields = {'field1'}
    try:
        TestRecord()
        assert False
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

def test___new___creates_empty_precord_without_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
    record = TestRecord()
    assert len(record) == 0


# LLM-generated content at query #12
#--------------------------

def test_precord_repr_with_single_field():
    class TestRecord(PRecord):
        _precord_fields = {'name': field(type=str)}
    record = TestRecord(name='Alice')
    result = repr(record)
    expected = "TestRecord(name='Alice')"
    assert result == expected

def test_precord_repr_with_multiple_fields():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int), 'y': field(type=str)}
    record = TestRecord(x=10, y='test')
    result = repr(record)
    expected = "TestRecord(x=10, y='test')"
    assert result == expected

def test_precord_repr_with_no_fields():
    class TestRecord(PRecord):
        _precord_fields = {}
    record = TestRecord()
    result = repr(record)
    expected = "TestRecord()"
    assert result == expected

def test_precord_repr_with_nested_values():
    class TestRecord(PRecord):
        _precord_fields = {'data': field(type=dict)}
    record = TestRecord(data={'key': 'value'})
    result = repr(record)
    expected = "TestRecord(data={'key': 'value'})"
    assert result == expected

def test_precord_repr_with_integer_field_names():
    class TestRecord(PRecord):
        _precord_fields = {'1': field(type=int), '2': field(type=int)}
    record = TestRecord(**{'1': 100, '2': 200})
    result = repr(record)
    expected = "TestRecord(1=100, 2=200)"
    assert result == expected


# LLM-generated content at query #13
#--------------------------

def test_precord_constructor_with_special_attributes():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field()}
    record = TestRecord(_precord_size=1, _precord_buckets=((('x', 10),),))
    assert record['x'] == 10
    assert len(record) == 1

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(), 'y': field()}
        _precord_initial_values = {'y': 20}
    record = TestRecord(x=10)
    assert record['x'] == 10
    assert record['y'] == 20

def test_precord_constructor_with_callable_initial_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': lambda: 100}
    record = TestRecord()
    assert record['x'] == 100

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field()}
    record = TestRecord(_factory_fields={'x': 5}, x=10)
    assert record['x'] == 10

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field()}
    record = TestRecord(_ignore_extra=True, x=10, y=20)
    assert record['x'] == 10
    assert 'y' not in record

def test_precord_constructor_without_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field()}
    try:
        TestRecord(x=10, y=20)
        assert False
    except AttributeError:
        pass

def test_precord_constructor_with_multiple_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'a': field(), 'b': field(), 'c': field()}
    record = TestRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3

def test_precord_constructor_with_no_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
    record = TestRecord()
    assert len(record) == 0

def test_precord_constructor_with_overriding_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(), 'y': field()}
        _precord_initial_values = {'x': 1, 'y': 2}
    record = TestRecord(x=10)
    assert record['x'] == 10
    assert record['y'] == 2

def test_precord_constructor_with_empty_kwargs():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': 5}
    record = TestRecord()
    assert record['x'] == 5


# LLM-generated content at query #14
#--------------------------

def test_precord_repr_returns_correct_format():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {}
    record = TestRecord(field1='value1', field2=123)
    result = repr(record)
    expected_start = "TestRecord("
    expected_end = ")"
    assert result.startswith(expected_start)
    assert result.endswith(expected_end)
    assert 'field1=' in result
    assert "'value1'" in result
    assert 'field2=' in result
    assert '123' in result
    parts = result[len(expected_start):-len(expected_end)].split(', ')
    assert len(parts) == 2
    assert "field1='value1'" in parts or "field2=123" in parts
    assert "field1='value1'" in result or 'field1="value1"' in result


# LLM-generated content at query #15
#--------------------------

```python
def test_persistent_creates_new_instance_when_not_dirty_and_pm_is_not_cls():
    from pyrsistent import PRecord, field
    from pyrsistent._precord import _PRecordEvolver

    class TestRecord(PRecord):
        x = field()

    original = TestRecord(x=1)
    evolver = _PRecordEvolver(TestRecord, original._map)
    evolver._destination_cls = TestRecord
    evolver._original_pmap = original._map
    evolver._buckets = original._map._buckets
    evolver._size = original._map._size
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    evolver._factory_fields = None
    evolver._ignore_extra = False

    pm = object()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result == original


# LLM-generated content at query #16
#--------------------------

def test___new___creates_fields_and_invariants():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class Base1:
        __invariant__ = lambda self: (True, ())
        _precord_fields = {'base_field': MockField()}
    class Base2:
        __invariant__ = lambda self: (False, ('error',))
    dct = {'field1': MockField(mandatory=True), 'field2': MockField(initial='default')}
    bases = (Base1, Base2)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_fields' in dct
    assert 'field1' in dct['_precord_fields']
    assert 'field2' in dct['_precord_fields']
    assert 'base_field' in dct['_precord_fields']
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 2
    assert '_precord_mandatory_fields' in dct
    assert dct['_precord_mandatory_fields'] == {'field1'}
    assert '_precord_initial_values' in dct
    assert dct['_precord_initial_values'] == {'field2': 'default'}
    assert '__slots__' in dct
    assert dct['__slots__'] == ()

def test___new___handles_no_invariants():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class Base:
        _precord_fields = {}
    dct = {}
    bases = (Base,)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_invariants' in dct
    assert dct['_precord_invariants'] == ()

def test___new___wraps_invariants():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants, wrap_invariant
    def invariant1(self):
        return (True, ())
    def invariant2(self):
        return [(False, 'err1'), (True, ()), (False, 'err2')]
    class Base:
        __invariant__ = invariant1
    dct = {'__invariant__': invariant2}
    bases = (Base,)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    result1 = invariants[0]('dummy')
    assert result1 == (True, ())
    result2 = invariants[1]('dummy')
    assert result2 == (False, ('err1', 'err2'))

def test___new___raises_on_non_callable_invariant():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class Base:
        __invariant__ = 'not callable'
    dct = {}
    bases = (Base,)
    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
        assert False
    except TypeError:
        pass

def test___new___handles_mandatory_and_initial_fields():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    dct = {
        'mandatory_field': MockField(mandatory=True),
        'initial_field': MockField(initial=42),
        'regular_field': MockField()
    }
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert dct['_precord_mandatory_fields'] == {'mandatory_field'}
    assert dct['_precord_initial_values'] == {'initial_field': 42}


# LLM-generated content at query #17
#--------------------------

def test___new___sets_fields_correctly():
    class MockPField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class Base:
        _precord_fields = {'base_field': MockPField()}
    dct = {'new_field': MockPField()}
    bases = (Base,)
    set_fields(dct, bases, name='_precord_fields')
    assert '_precord_fields' in dct
    assert 'base_field' in dct['_precord_fields']
    assert 'new_field' in dct['_precord_fields']
    assert 'new_field' not in dct

def test___new___handles_mandatory_fields():
    class MockPField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    dct = {'mandatory_field': MockPField(mandatory=True), 'optional_field': MockPField(mandatory=False)}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    mandatory_fields = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
    assert mandatory_fields == {'mandatory_field'}

def test___new___handles_initial_values():
    class MockPField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    dct = {'with_initial': MockPField(initial='default'), 'without_initial': MockPField(initial=PFIELD_NO_INITIAL)}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    initial_values = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
    assert initial_values == {'with_initial': 'default'}

def test___new___stores_invariants():
    def invariant1(instance):
        return True, ()
    def invariant2(instance):
        return False, ('error',)
    class Base:
        __invariant__ = invariant1
    dct = {'__invariant__': invariant2}
    bases = (Base,)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 2
    assert dct['_precord_invariants'][0](None) == (True, ())
    assert dct['_precord_invariants'][1](None) == (False, ('error',))

def test___new___raises_on_non_callable_invariant():
    class Base:
        pass
    dct = {'__invariant__': 'not a callable'}
    bases = (Base,)
    try:
        store_invariants(dct, bases, '_precord_invariants', '__invariant__')
        assert False
    except TypeError:
        pass

def test___new___sets_slots():
    class MockPField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    dct = {}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    dct['__slots__'] = ()
    assert dct['__slots__'] == ()

def test___new___inherits_fields_and_invariants():
    class MockPField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    def base_invariant(instance):
        return True, ()
    class GrandBase:
        _precord_fields = {'grand_field': MockPField()}
        __invariant__ = base_invariant
    class Base(GrandBase):
        _precord_fields = {'base_field': MockPField()}
    dct = {'child_field': MockPField()}
    bases = (Base,)
    set_fields(dct, bases, name='_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert 'grand_field' in dct['_precord_fields']
    assert 'base_field' in dct['_precord_fields']
    assert 'child_field' in dct['_precord_fields']
    assert len(dct['_precord_invariants']) == 1


# LLM-generated content at query #18
#--------------------------

def test_precord_new_creates_instance_with_special_attributes():
    class TestRecord(PRecord):
        pass
    instance = TestRecord(_precord_size=0, _precord_buckets=pvector().extend([]))
    assert isinstance(instance, TestRecord)
    assert instance._size == 0

def test_precord_new_uses_evolver_for_initial_values():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {}
    TestRecord._precord_initial_values = {}
    instance = TestRecord()
    assert isinstance(instance, TestRecord)

def test_precord_new_applies_initial_values():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'field1': object()}
    TestRecord._precord_initial_values = {'field1': lambda: 'default'}
    instance = TestRecord()
    assert instance['field1'] == 'default'

def test_precord_new_overrides_initial_values_with_kwargs():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'field1': object()}
    TestRecord._precord_initial_values = {'field1': lambda: 'default'}
    instance = TestRecord(field1='custom')
    assert instance['field1'] == 'custom'

def test_precord_new_passes_factory_fields_to_evolver():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {}
    TestRecord._precord_initial_values = {}
    instance = TestRecord(_factory_fields=set())
    assert isinstance(instance, TestRecord)

def test_precord_new_passes_ignore_extra_to_evolver():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {}
    TestRecord._precord_initial_values = {}
    instance = TestRecord(_ignore_extra=True)
    assert isinstance(instance, TestRecord)

def test_precord_new_handles_callable_initial_values():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'field1': object()}
    TestRecord._precord_initial_values = {'field1': lambda: 'callable_result'}
    instance = TestRecord()
    assert instance['field1'] == 'callable_result'

def test_precord_new_handles_non_callable_initial_values():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'field1': object()}
    TestRecord._precord_initial_values = {'field1': 'static_value'}
    instance = TestRecord()
    assert instance['field1'] == 'static_value'


# LLM-generated content at query #19
#--------------------------

def test_precord_new_with_special_attributes():
    class TestRecord(PRecord):
        pass
    record = TestRecord(_precord_size=0, _precord_buckets=pvector().extend([None] * 8))
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'a': field(), 'b': field()}
    record = TestRecord(a=1, b=2)
    assert record['a'] == 1
    assert record['b'] == 2

def test_precord_new_with_default_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'a': field(), 'b': field()}
        _precord_initial_values = {'a': lambda: 10, 'b': 20}
    record = TestRecord()
    assert record['a'] == 10
    assert record['b'] == 20

def test_precord_new_with_overridden_defaults():
    class TestRecord(PRecord):
        _precord_fields = {'a': field(), 'b': field()}
        _precord_initial_values = {'a': lambda: 10, 'b': 20}
    record = TestRecord(a=100)
    assert record['a'] == 100
    assert record['b'] == 20

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'a': field(type=int), 'b': field(type=str)}
    record = TestRecord(_factory_fields=[TestRecord._precord_fields['a']], a=5, b='hello')
    assert record['a'] == 5
    assert record['b'] == 'hello'

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'a': field()}
    record = TestRecord(_ignore_extra=True, a=1, b=2)
    assert 'a' in record
    assert 'b' not in record

def test_precord_new_with_invalid_field_raises_attribute_error():
    class TestRecord(PRecord):
        _precord_fields = {'a': field()}
    try:
        TestRecord(b=2)
        assert False
    except AttributeError as e:
        assert "'b' is not among the specified fields for TestRecord" in str(e)

def test_precord_new_with_field_invariant_failure():
    def invariant_check(value):
        return value > 0, 'value must be positive'
    class TestRecord(PRecord):
        _precord_fields = {'a': field(invariant=invariant_check)}
    try:
        TestRecord(a=-1)
        assert False
    except InvariantException as e:
        assert 'value must be positive' in str(e)

def test_precord_new_with_mandatory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'a': field(mandatory=True), 'b': field()}
        _precord_mandatory_fields = {'a'}
    try:
        TestRecord(b=2)
        assert False
    except InvariantException as e:
        assert 'TestRecord.a' in str(e)

def test_precord_new_with_global_invariant_failure():
    def global_invariant(record):
        return record.get('a', 0) + record.get('b', 0) > 0, 'sum must be positive'
    class TestRecord(PRecord):
        _precord_fields = {'a': field(), 'b': field()}
        _precord_invariants = [global_invariant]
    try:
        TestRecord(a=-1, b=-1)
        assert False
    except InvariantException as e:
        assert 'sum must be positive' in str(e)


# LLM-generated content at query #20
#--------------------------

def test_serialize_without_custom_serializer():
    class TestRecord(PRecord):
        _precord_fields = {'field1': None, 'field2': None}
    record = TestRecord(field1='value1', field2=123)
    result = record.serialize()
    expected = {'field1': 'value1', 'field2': 123}
    assert result == expected

def test_serialize_with_custom_serializer():
    def custom_serializer(format, value):
        return f'custom_{value}'
    class TestRecord(PRecord):
        _precord_fields = {'field1': type('Field', (), {'serializer': custom_serializer})(), 'field2': None}
    record = TestRecord(field1='value1', field2=123)
    result = record.serialize()
    expected = {'field1': 'custom_value1', 'field2': 123}
    assert result == expected

def test_serialize_with_format_parameter():
    def custom_serializer(format, value):
        return f'{format}_{value}'
    class TestRecord(PRecord):
        _precord_fields = {'field1': type('Field', (), {'serializer': custom_serializer})(), 'field2': None}
    record = TestRecord(field1='value1', field2=123)
    result = record.serialize('json')
    expected = {'field1': 'json_value1', 'field2': 123}
    assert result == expected

def test_serialize_with_multiple_fields_and_serializers():
    def serializer1(format, value):
        return value * 2
    def serializer2(format, value):
        return value + 10
    class TestRecord(PRecord):
        _precord_fields = {
            'field1': type('Field', (), {'serializer': serializer1})(),
            'field2': type('Field', (), {'serializer': serializer2})(),
            'field3': None
        }
    record = TestRecord(field1=5, field2=20, field3='test')
    result = record.serialize()
    expected = {'field1': 10, 'field2': 30, 'field3': 'test'}
    assert result == expected

def test_serialize_empty_record():
    class TestRecord(PRecord):
        _precord_fields = {}
    record = TestRecord()
    result = record.serialize()
    expected = {}
    assert result == expected


# LLM-generated content at query #21
#--------------------------

def test_precord_initial_values_used_when_present():
    class TestRecord(PRecord):
        _precord_fields = {'a': field()}
        _precord_initial_values = {'a': lambda: 42}
    result = TestRecord()
    assert result['a'] == 42

def test_precord_initial_values_ignored_when_overridden():
    class TestRecord(PRecord):
        _precord_fields = {'a': field()}
        _precord_initial_values = {'a': lambda: 42}
    result = TestRecord(a=100)
    assert result['a'] == 100

def test_precord_initial_values_with_callable():
    class TestRecord(PRecord):
        _precord_fields = {'a': field()}
        _precord_initial_values = {'a': lambda: 'default'}
    result = TestRecord()
    assert result['a'] == 'default'

def test_precord_initial_values_with_non_callable():
    class TestRecord(PRecord):
        _precord_fields = {'a': field()}
        _precord_initial_values = {'a': 'static_default'}
    result = TestRecord()
    assert result['a'] == 'static_default'

def test_precord_initial_values_empty_dict_no_effect():
    class TestRecord(PRecord):
        _precord_fields = {'a': field()}
        _precord_initial_values = {}
    result = TestRecord(a=5)
    assert result['a'] == 5

def test_precord_initial_values_combined_with_kwargs():
    class TestRecord(PRecord):
        _precord_fields = {'a': field(), 'b': field()}
        _precord_initial_values = {'a': lambda: 1, 'b': lambda: 2}
    result = TestRecord(b=20)
    assert result['a'] == 1
    assert result['b'] == 20


# LLM-generated content at query #22
#--------------------------

def test_precord_new_creates_instance_with_special_attributes():
    class TestRecord(PRecord):
        pass
    buckets = pvector().extend([None] * 8)
    instance = TestRecord(_precord_size=0, _precord_buckets=buckets)
    assert isinstance(instance, TestRecord)
    assert instance._size == 0
    assert instance._buckets == buckets

def test_precord_new_uses_evolver_for_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {}
    instance = TestRecord(field1=42, field2='test')
    assert instance['field1'] == 42
    assert instance['field2'] == 'test'

def test_precord_new_applies_initial_values_from_class():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {'field1': lambda: 100, 'field2': 'default'}
    instance = TestRecord()
    assert instance['field1'] == 100
    assert instance['field2'] == 'default'

def test_precord_new_overrides_initial_values_with_kwargs():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {'field1': lambda: 100, 'field2': 'default'}
    instance = TestRecord(field1=200)
    assert instance['field1'] == 200
    assert instance['field2'] == 'default'

def test_precord_new_handles_factory_fields_parameter():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, factory=lambda x: x * 2)}
    instance = TestRecord(_factory_fields=None, field1=21)
    assert instance['field1'] == 42

def test_precord_new_handles_ignore_extra_parameter():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
    instance = TestRecord(_ignore_extra=True, field1=10, extra_field=20)
    assert instance['field1'] == 10
    assert 'extra_field' not in instance

def test_precord_new_raises_attribute_error_for_unknown_field():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
    try:
        TestRecord(unknown_field=10)
        assert False
    except AttributeError as e:
        assert "'unknown_field' is not among the specified fields for TestRecord" in str(e)

def test_precord_new_invokes_field_factory_with_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=PMap, factory=lambda x, **kw: pmap(x), mandatory=True)}
    instance = TestRecord(_ignore_extra=True, field1={'a': 1})
    assert instance['field1'] == {'a': 1}

def test_precord_new_raises_invariant_exception_for_missing_mandatory_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, mandatory=True)}
        _precord_mandatory_fields = {'field1'}
    try:
        TestRecord()
        assert False
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

def test_precord_new_raises_invariant_exception_for_field_invariant_failure():
    def invariant_check(value):
        return value > 0, 'ERR1'
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, invariant=invariant_check)}
    try:
        TestRecord(field1=-5)
        assert False
    except InvariantException as e:
        assert 'ERR1' in e.invariant_errors

def test_precord_new_raises_invariant_exception_for_global_invariant_failure():
    def global_invariant(record):
        return record.get('field1', 0) + record.get('field2', 0) > 0, 'GLOBAL_ERR'
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=int)}
        _precord_invariants = [global_invariant]
    try:
        TestRecord(field1=-10, field2=5)
        assert False
    except InvariantException as e:
        assert 'GLOBAL_ERR' in e.invariant_errors


# LLM-generated content at query #23
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    class MockEvolver(_PRecordEvolver):
        def __init__(self):
            self._invariant_error_codes = ['error1']
            self._missing_fields = []
            self._destination_cls = type('MockClass', (), {'_precord_mandatory_fields': set(), '_precord_invariants': []})
        
        def is_dirty(self):
            return False
        
        def persistent(self):
            return super(MockEvolver, self).persistent()
    
    evolver = MockEvolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()
        assert str(e) == 'Field invariant failed'


# LLM-generated content at query #24
#--------------------------

def test___new___sets_fields_correctly():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    field1 = MockField(mandatory=True, initial=10)
    field2 = MockField(mandatory=False, initial=None)
    dct = {'field1': field1, 'field2': field2}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    assert '_precord_fields' in dct
    assert dct['_precord_fields']['field1'] == field1
    assert dct['_precord_fields']['field2'] == field2
    assert 'field1' not in dct
    assert 'field2' not in dct

def test___new___inherits_fields_from_bases():
    class Base:
        _precord_fields = {'base_field': 'base_value'}
    dct = {}
    bases = (Base,)
    set_fields(dct, bases, name='_precord_fields')
    assert dct['_precord_fields']['base_field'] == 'base_value'

def test___new___merges_fields_from_multiple_bases():
    class Base1:
        _precord_fields = {'field1': 'value1'}
    class Base2:
        _precord_fields = {'field2': 'value2'}
    dct = {}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    assert dct['_precord_fields']['field1'] == 'value1'
    assert dct['_precord_fields']['field2'] == 'value2'

def test___new___stores_invariants():
    def invariant1(x):
        return True, None
    def invariant2(x):
        return False, 'error'
    dct = {'__invariant__': invariant1}
    class Base:
        __invariant__ = invariant2
    bases = (Base,)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 2
    assert dct['_precord_invariants'][0](None) == (True, None)
    assert dct['_precord_invariants'][1](None) == (False, ('error',))

def test___new___wraps_invariants_correctly():
    def invariant_multiple(x):
        return [(True, None), (False, 'err1'), (False, 'err2')]
    dct = {'__invariant__': invariant_multiple}
    bases = ()
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    result = dct['_precord_invariants'][0](None)
    assert result == (False, ('err1', 'err2'))

def test___new___sets_mandatory_fields():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    field_mandatory = MockField(mandatory=True, initial=None)
    field_optional = MockField(mandatory=False, initial=None)
    dct = {'field_mandatory': field_mandatory, 'field_optional': field_optional}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    dct['_precord_mandatory_fields'] = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
    assert '_precord_mandatory_fields' in dct
    assert dct['_precord_mandatory_fields'] == {'field_mandatory'}

def test___new___sets_initial_values():
    PFIELD_NO_INITIAL = object()
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    field_with_initial = MockField(mandatory=False, initial=5)
    field_no_initial = MockField(mandatory=False, initial=PFIELD_NO_INITIAL)
    dct = {'field_with_initial': field_with_initial, 'field_no_initial': field_no_initial}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    dct['_precord_initial_values'] = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
    assert '_precord_initial_values' in dct
    assert dct['_precord_initial_values'] == {'field_with_initial': 5}

def test___new___sets_empty_slots():
    dct = {}
    bases = ()
    dct['__slots__'] = ()
    assert dct['__slots__'] == ()

def test___new___raises_on_non_callable_invariant():
    dct = {'__invariant__': 'not a callable'}
    bases = ()
    try:
        store_invariants(dct, bases, '_precord_invariants', '__invariant__')
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #25
#--------------------------

```python
def test_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    record = TestRecord()
    assert '_precord_size' not in record._precord_buckets
    assert '_precord_buckets' not in record._precord_buckets


# LLM-generated content at query #26
#--------------------------

def test_serialize_with_custom_serializer():
    from pyrsistent import PRecord, field
    from pyrsistent import serialize as ser_func
    class TestRecord(PRecord):
        name = field()
        value = field(serializer=lambda fmt, v: f"serialized_{v}")
    record = TestRecord(name="test", value=42)
    result = record.serialize()
    expected = {"name": "test", "value": "serialized_42"}
    assert result == expected

def test_serialize_without_custom_serializer():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        name = field()
        age = field()
    record = TestRecord(name="Alice", age=30)
    result = record.serialize()
    expected = {"name": "Alice", "age": 30}
    assert result == expected

def test_serialize_with_format_parameter():
    from pyrsistent import PRecord, field
    from pyrsistent import serialize as ser_func
    class TestRecord(PRecord):
        data = field(serializer=lambda fmt, v: f"{fmt}:{v}")
    record = TestRecord(data="example")
    result = record.serialize("json")
    expected = {"data": "json:example"}
    assert result == expected

def test_serialize_with_none_format():
    from pyrsistent import PRecord, field
    from pyrsistent import serialize as ser_func
    class TestRecord(PRecord):
        item = field(serializer=lambda fmt, v: v if fmt is None else f"{fmt}_{v}")
    record = TestRecord(item="test")
    result = record.serialize()
    expected = {"item": "test"}
    assert result == expected

def test_serialize_on_empty_record():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        pass
    record = TestRecord()
    result = record.serialize()
    expected = {}
    assert result == expected

def test_serialize_with_multiple_fields_mixed_serializers():
    from pyrsistent import PRecord, field
    from pyrsistent import serialize as ser_func
    class TestRecord(PRecord):
        a = field()
        b = field(serializer=lambda fmt, v: v * 2)
        c = field(serializer=lambda fmt, v: str(v))
    record = TestRecord(a=1, b=2, c=3)
    result = record.serialize()
    expected = {"a": 1, "b": 4, "c": "3"}
    assert result == expected


# LLM-generated content at query #27
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import InvariantException
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, **kwargs):
            pass
    original_pmap = {}
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()
        assert str(e) == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #28
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true_when_conditions_met():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import CheckedType
    import inspect
    
    class MockField:
        def __init__(self, field_type, factory_params):
            self.type = field_type
            self.factory = lambda **kwargs: None
            self.factory.__signature__ = inspect.signature(lambda **kwargs: None).replace(parameters=factory_params)
    
    class MockCheckedType(CheckedType):
        pass
    
    field_type = (MockCheckedType,)
    factory_params = [inspect.Parameter('ignore_extra', inspect.Parameter.KEYWORD_ONLY, default=False)]
    field = MockField(field_type, factory_params)
    ignore_extra = True
    result = is_field_ignore_extra_complaint(CheckedType, field, ignore_extra)
    assert result == True


