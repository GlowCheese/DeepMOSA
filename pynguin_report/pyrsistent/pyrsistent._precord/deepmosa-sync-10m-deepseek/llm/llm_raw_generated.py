####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_persistent_returns_instance_of_destination_class():
    class TestRecord(PRecord):
        field = field(type=str)
    evolver = TestRecord().evolver()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)

def test_persistent_raises_invariant_exception_on_missing_mandatory_fields():
    class TestRecord(PRecord):
        mandatory = mandatory_field(type=str)
    evolver = TestRecord().evolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.missing_fields

def test_persistent_raises_invariant_exception_on_field_invariant_failure():
    class TestRecord(PRecord):
        field = field(type=int, invariant=lambda x: (x > 0, 'ERR'))
    evolver = TestRecord().evolver()
    evolver.set('field', -1)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors

def test_persistent_raises_invariant_exception_on_global_invariant_failure():
    class TestRecord(PRecord):
        field = field(type=int)
        def __invariant__(self):
            return (self.field > 0, 'ERR')
    evolver = TestRecord().evolver()
    evolver.set('field', -1)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors

def test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type():
    original = PRecord()
    evolver = _PRecordEvolver(type(original), original._map)
    result = evolver.persistent()
    assert result is original

def test_persistent_creates_new_instance_if_dirty():
    class TestRecord(PRecord):
        field = field(type=str)
    original = TestRecord()
    evolver = original.evolver()
    evolver.set('field', 'new')
    result = evolver.persistent()
    assert result is not original
    assert result.field == 'new'

def test_persistent_includes_all_set_values():
    class TestRecord(PRecord):
        a = field(type=int)
        b = field(type=str)
    evolver = TestRecord().evolver()
    evolver.set('a', 1)
    evolver.set('b', 'test')
    result = evolver.persistent()
    assert result.a == 1
    assert result.b == 'test'


# LLM-generated content at query #2
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
    field1 = _PField(mandatory=True, initial=PFIELD_NO_INITIAL)
    field2 = _PField(mandatory=False, initial=10)
    dct = {'field1': field1, 'field2': field2}
    bases = ()
    result = set_fields(dct, bases, name='_precord_fields')
    assert dct['_precord_fields'] == {'field1': field1, 'field2': field2}
    assert 'field1' not in dct
    assert 'field2' not in dct

def test___new___inherits_fields_from_bases():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    base_field = _PField(mandatory=True, initial=PFIELD_NO_INITIAL)
    base_dct = {'base_field': base_field}
    class Base(metaclass=type):
        pass
    Base.__dict__ = base_dct
    dct = {'new_field': _PField(mandatory=False, initial=5)}
    bases = (Base,)
    result = set_fields(dct, bases, name='_precord_fields')
    assert dct['_precord_fields'] == {'base_field': base_field, 'new_field': dct['_precord_fields']['new_field']}
    assert 'new_field' not in dct

def test___new___stores_invariants():
    def invariant1(x):
        return True, ()
    def invariant2(x):
        return False, ('error',)
    dct = {'__invariant__': invariant1}
    class Base(metaclass=type):
        __invariant__ = invariant2
    bases = (Base,)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    assert invariants[0](None) == (True, ())
    assert invariants[1](None) == (False, ('error',))

def test___new___wraps_invariants():
    def invariant_multiple(x):
        return [(True, ()), (False, ('err1',)), (False, ('err2',))]
    dct = {'__invariant__': invariant_multiple}
    bases = ()
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    wrapped = dct['_precord_invariants'][0]
    result = wrapped(None)
    assert result == (False, ('err1', 'err2'))

def test___new___raises_on_non_callable_invariant():
    dct = {'__invariant__': 'not callable'}
    bases = ()
    try:
        store_invariants(dct, bases, '_precord_invariants', '__invariant__')
        assert False
    except TypeError:
        pass

def test___new___sets_mandatory_fields():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    field1 = _PField(mandatory=True, initial=PFIELD_NO_INITIAL)
    field2 = _PField(mandatory=False, initial=PFIELD_NO_INITIAL)
    dct = {'field1': field1, 'field2': field2}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    dct['_precord_mandatory_fields'] = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
    assert dct['_precord_mandatory_fields'] == {'field1'}

def test___new___sets_initial_values():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    field1 = _PField(mandatory=True, initial=PFIELD_NO_INITIAL)
    field2 = _PField(mandatory=False, initial=10)
    dct = {'field1': field1, 'field2': field2}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    dct['_precord_initial_values'] = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
    assert dct['_precord_initial_values'] == {'field2': 10}

def test___new___sets_slots():
    dct = {}
    bases = ()
    dct['__slots__'] = ()
    assert dct['__slots__'] == ()

def test___new___full_metaclass_creation():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    def invariant(x):
        return True, ()
    class Base(metaclass=type):
        __invariant__ = invariant
        base_field = _PField(mandatory=True, initial=PFIELD_NO_INITIAL)
    class TestClass(metaclass=_PRecordMeta):
        __invariant__ = invariant
        field1 = _PField(mandatory=True, initial=PFIELD_NO_INITIAL)
        field2 = _PField(mandatory=False, initial=20)
    assert hasattr(TestClass, '_precord_fields')
    assert 'field1' in TestClass._precord_fields
    assert 'field2' in TestClass._precord_fields
    assert 'base_field' in TestClass._precord_fields
    assert TestClass._precord_mandatory_fields == {'field1', 'base_field'}
    assert TestClass._precord_initial_values == {'field2': 20}
    assert TestClass.__slots__ == ()
    assert len(TestClass._precord_invariants) == 2


# LLM-generated content at query #3
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
        _precord_fields = {'name': field(type=str), 'age': field(type=int)}
    record = TestRecord(name='Bob', age=30)
    result = repr(record)
    expected = "TestRecord(name='Bob', age=30)"
    assert result == expected

def test_precord_repr_with_empty_fields():
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

def test_precord_repr_with_special_characters_in_field_name():
    class TestRecord(PRecord):
        _precord_fields = {'field_name': field(type=str)}
    record = TestRecord(field_name='test')
    result = repr(record)
    expected = "TestRecord(field_name='test')"
    assert result == expected


# LLM-generated content at query #4
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
    record = TestRecord(field1=42, field2='test')
    assert record['field1'] == 42
    assert record['field2'] == 'test'

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, factory=lambda x: x * 2)}
    record = TestRecord(field1=21, _factory_fields=set())
    assert record['field1'] == 21

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
    record = TestRecord(field1=1, extra_field=2, _ignore_extra=True)
    assert record['field1'] == 1
    assert 'extra_field' not in record

def test_precord_new_with_mandatory_fields_missing():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, mandatory=True), 'field2': field(type=str)}
    try:
        TestRecord(field2='test')
        assert False
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

def test_precord_new_with_invariant_failure():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, invariant=lambda x: (x > 0, 'ERR1'))}
    try:
        TestRecord(field1=-1)
        assert False
    except InvariantException as e:
        assert 'ERR1' in e.invariant_errors

def test_precord_new_with_initial_values_callable():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
        _precord_initial_values = {'field1': lambda: 100}
    record = TestRecord()
    assert record['field1'] == 100

def test_precord_new_with_initial_values_overridden():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
        _precord_initial_values = {'field1': lambda: 100}
    record = TestRecord(field1=200)
    assert record['field1'] == 200

def test_precord_new_with_factory_fields_and_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, factory=lambda x: x + 1)}
    record = TestRecord(field1=5, extra=10, _factory_fields={TestRecord._precord_fields['field1']}, _ignore_extra=True)
    assert record['field1'] == 6
    assert 'extra' not in record

def test_precord_new_with_no_initial_and_no_special_attributes():
    class TestRecord(PRecord):
        _precord_fields = {}
    record = TestRecord()
    assert isinstance(record, TestRecord)
    assert len(record) == 0


# LLM-generated content at query #5
#--------------------------

def test_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    r = TestRecord()
    assert isinstance(r, TestRecord)
    assert len(r) == 0

def test_new_with_regular_kwargs():
    class TestRecord(PRecord):
        pass
    r = TestRecord(a=1, b=2)
    assert r['a'] == 1
    assert r['b'] == 2

def test_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    r = TestRecord(_factory_fields=None)
    assert isinstance(r, TestRecord)

def test_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    r = TestRecord(_ignore_extra=False)
    assert isinstance(r, TestRecord)

def test_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'x': lambda: 10}
    r = TestRecord(y=20)
    assert r['x'] == 10
    assert r['y'] == 20

def test_new_with_overridden_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'x': lambda: 10}
    r = TestRecord(x=30, y=20)
    assert r['x'] == 30
    assert r['y'] == 20


# LLM-generated content at query #6
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()

    evolver = _PRecordEvolver(MockClass, pmap({}))
    evolver._invariant_error_codes = ["error1"]
    evolver._missing_fields = []
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ("error1",)
        assert e.missing_fields == ()
        assert str(e) == "Field invariant failed"

def test_persistent_raises_invariant_exception_when_missing_fields_present():
    class MockClass:
        _precord_mandatory_fields = {"field1"}
        _precord_invariants = []
        __name__ = "MockClass"
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()

    evolver = _PRecordEvolver(MockClass, pmap({}))
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ("MockClass.field1",)
        assert str(e) == "Field invariant failed"

def test_persistent_raises_invariant_exception_when_both_invariant_error_codes_and_missing_fields_present():
    class MockClass:
        _precord_mandatory_fields = {"field1", "field2"}
        _precord_invariants = []
        __name__ = "MockClass"
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: {"field1"}

    evolver = _PRecordEvolver(MockClass, pmap({}))
    evolver._invariant_error_codes = ["error1", "error2"]
    evolver._missing_fields = ["MockClass.field3"]
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ("error1", "error2")
        assert "MockClass.field2" in e.missing_fields
        assert "MockClass.field3" in e.missing_fields
        assert len(e.missing_fields) == 2
        assert str(e) == "Field invariant failed"

def test_persistent_does_not_raise_when_no_invariant_error_codes_and_no_missing_fields():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()

    evolver = _PRecordEvolver(MockClass, pmap({}))
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    result = evolver.persistent()
    assert isinstance(result, MockClass)

def test_persistent_handles_empty_mandatory_fields_correctly():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()

    evolver = _PRecordEvolver(MockClass, pmap({}))
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    result = evolver.persistent()
    assert isinstance(result, MockClass)


# LLM-generated content at query #7
#--------------------------

def test_precord_constructor_without_special_attributes():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {}

    instance = TestRecord(field1='value1', field2='value2')
    assert instance['field1'] == 'value1'
    assert instance['field2'] == 'value2'


def test_precord_constructor_with_special_attributes():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
        _precord_initial_values = {}

    size = 0
    buckets = []
    instance = TestRecord(_precord_size=size, _precord_buckets=buckets)
    assert isinstance(instance, TestRecord)


def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {'field1': lambda: 'default1', 'field2': 'default2'}

    instance = TestRecord()
    assert instance['field1'] == 'default1'
    assert instance['field2'] == 'default2'


def test_precord_constructor_with_initial_values_and_kwargs():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {'field1': lambda: 'default1', 'field2': 'default2'}

    instance = TestRecord(field1='custom1')
    assert instance['field1'] == 'custom1'
    assert instance['field2'] == 'default2'


def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {}

    instance = TestRecord(_factory_fields={'field1': 'factory_value'}, field2='value2')
    assert instance['field1'] == 'factory_value'
    assert instance['field2'] == 'value2'


def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field()}
        _precord_initial_values = {}

    instance = TestRecord(_ignore_extra=True, field1='value1', extra_field='extra')
    assert instance['field1'] == 'value1'
    assert 'extra_field' not in instance


def test_precord_constructor_without_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field()}
        _precord_initial_values = {}

    try:
        TestRecord(field1='value1', extra_field='extra')
        assert False
    except AttributeError:
        assert True


# LLM-generated content at query #8
#--------------------------

```python
def test_persistent_when_cls_has_precord_mandatory_fields():
    class MockClass:
        _precord_mandatory_fields = {"field1", "field2"}
        __name__ = "MockClass"

    evolver = _PRecordEvolver(MockClass, {})
    evolver._missing_fields = []
    result = {}
    evolver._destination_cls = MockClass
    condition = bool(MockClass._precord_mandatory_fields)
    assert condition is True


# LLM-generated content at query #9
#--------------------------

def test_precord_new_creates_instance_with_special_attributes():
    class TestRecord(PRecord):
        pass
    result = TestRecord(_precord_size=0, _precord_buckets=pvector().extend([]))
    assert isinstance(result, TestRecord)
    assert len(result) == 0

def test_precord_new_uses_evolver_for_normal_creation():
    class TestRecord(PRecord):
        pass
    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert len(result) == 0

def test_precord_new_applies_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'x': 5, 'y': lambda: 10}
    result = TestRecord()
    assert result['x'] == 5
    assert result['y'] == 10

def test_precord_new_overrides_initial_values_with_kwargs():
    class TestRecord(PRecord):
        _precord_initial_values = {'x': 5, 'y': 10}
    result = TestRecord(x=7)
    assert result['x'] == 7
    assert result['y'] == 10

def test_precord_new_passes_factory_fields_to_evolver():
    class TestRecord(PRecord):
        pass
    result = TestRecord(_factory_fields=set())
    assert isinstance(result, TestRecord)

def test_precord_new_passes_ignore_extra_to_evolver():
    class TestRecord(PRecord):
        pass
    result = TestRecord(_ignore_extra=True)
    assert isinstance(result, TestRecord)

def test_precord_new_handles_multiple_kwargs():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_raises_attribute_error_for_invalid_field():
    class TestRecord(PRecord):
        pass
    try:
        TestRecord(invalid_field=5)
        assert False
    except AttributeError as e:
        assert "'invalid_field' is not among the specified fields for TestRecord" in str(e)

def test_precord_new_invokes_field_factories():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'x': type('Field', (), {'factory': lambda v: v * 2, 'invariant': lambda v: (True, None)})()}
    result = TestRecord(x=3)
    assert result['x'] == 6

def test_precord_new_validates_field_types():
    class TestRecord(PRecord):
        pass
    def type_check(cls, field, key, value):
        if not isinstance(value, int):
            raise TypeError("Invalid type")
    original_check_type = check_type
    check_type = type_check
    try:
        TestRecord._precord_fields = {'x': type('Field', (), {'factory': lambda v: v, 'invariant': lambda v: (True, None)})()}
        try:
            TestRecord(x="not_an_int")
            assert False
        except TypeError as e:
            assert "Invalid type" in str(e)
    finally:
        check_type = original_check_type

def test_precord_new_enforces_invariants():
    class TestRecord(PRecord):
        pass
    def failing_invariant(value):
        return (False, "INVARIANT_FAILED")
    TestRecord._precord_fields = {'x': type('Field', (), {'factory': lambda v: v, 'invariant': failing_invariant})()}
    try:
        TestRecord(x=5)
        assert False
    except InvariantException as e:
        assert "INVARIANT_FAILED" in e.invariant_errors

def test_precord_new_checks_mandatory_fields():
    class TestRecord(PRecord):
        _precord_mandatory_fields = {'x'}
    try:
        TestRecord()
        assert False
    except InvariantException as e:
        assert "TestRecord.x" in e.missing_fields

def test_precord_new_checks_global_invariants():
    class TestRecord(PRecord):
        pass
    def failing_global_invariant(record):
        return False
    TestRecord._precord_invariants = [failing_global_invariant]
    try:
        TestRecord()
        assert False
    except InvariantException as e:
        pass


# LLM-generated content at query #10
#--------------------------

def test_set_with_valid_field_and_value():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str, invariant=lambda x: (len(x) > 0, 'ERR_EMPTY'))}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    original_pmap = {}
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('name', 'Alice')
    result = evolver.persistent()
    assert result['name'] == 'Alice'

def test_set_with_invalid_type_raises_ptype_error():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._exceptions import PTypeError
    class TestRecord(CheckedType):
        _precord_fields = {'age': field(type=int)}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    original_pmap = {}
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    try:
        evolver.set('age', 'not_an_int')
        assert False
    except PTypeError as e:
        assert e.__class__.__name__ == 'PTypeError'

def test_set_with_failed_invariant_adds_error_code():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._exceptions import InvariantException
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str, invariant=lambda x: (len(x) > 0, 'ERR_EMPTY'))}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    original_pmap = {}
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('name', '')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'ERR_EMPTY' in e.invariant_errors

def test_set_with_non_existent_field_raises_attribute_error():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    class TestRecord(CheckedType):
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    original_pmap = {}
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    try:
        evolver.set('unknown', 'value')
        assert False
    except AttributeError as e:
        assert "'unknown' is not among the specified fields" in str(e)

def test_set_with_factory_and_ignore_extra():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    def factory(value, ignore_extra=False):
        return value.upper()
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str, factory=factory)}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    original_pmap = {}
    evolver = _PRecordEvolver(TestRecord, original_pmap, _factory_fields=None, _ignore_extra=True)
    evolver.set('name', 'alice')
    result = evolver.persistent()
    assert result['name'] == 'ALICE'

def test_set_with_factory_invariant_exception_adds_errors():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._exceptions import InvariantException
    def factory(value):
        raise InvariantException(('FACTORY_ERR',), (), 'Factory failed')
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str, factory=factory)}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    original_pmap = {}
    evolver = _PRecordEvolver(TestRecord, original_pmap)
    evolver.set('name', 'alice')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'FACTORY_ERR' in e.invariant_errors

def test_set_with_factory_fields_skips_factory():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    def factory(value):
        return value.upper()
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str, factory=factory)}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    original_pmap = {}
    factory_fields = set()
    evolver = _PRecordEvolver(TestRecord, original_pmap, _factory_fields=factory_fields)
    evolver.set('name', 'alice')
    result = evolver.persistent()
    assert result['name'] == 'alice'


# LLM-generated content at query #11
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
    original = PRecord()
    evolver = _PRecordEvolver(type(original), original._map)
    result = evolver.persistent()
    assert result is original

def test_persistent_constructs_new_instance_if_dirty():
    class TestRecord(PRecord):
        field = field(type=int)
    original = TestRecord()
    evolver = original.evolver()
    evolver.set('field', 1)
    result = evolver.persistent()
    assert result is not original
    assert result.field == 1

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

def test_persistent_aggregates_missing_fields_from_set_and_mandatory():
    class TestRecord(PRecord):
        field1 = field(type=int, mandatory=True)
        field2 = field(type=int, mandatory=True)
    evolver = TestRecord().evolver()
    evolver._missing_fields = ['TestRecord.field1']
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert set(e.missing_fields) == {'TestRecord.field1', 'TestRecord.field2'}


# LLM-generated content at query #12
#--------------------------

```python
def test_persistent_when_not_dirty_and_pm_is_instance_of_cls():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        x = field()
    original = TestRecord(x=1)
    evolver = _PRecordEvolver(TestRecord, original._map)
    result = evolver.persistent()
    assert result is original

def test_persistent_when_dirty_and_pm_is_instance_of_cls():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        x = field()
    original = TestRecord(x=1)
    evolver = _PRecordEvolver(TestRecord, original._map)
    evolver.set('x', 2)
    result = evolver.persistent()
    assert result is not original
    assert result.x == 2

def test_persistent_when_not_dirty_and_pm_is_not_instance_of_cls():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import PMap
    class FakeClass:
        _precord_buckets = None
        _precord_size = None
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
    original_pmap = PMap({1: 2})
    evolver = _PRecordEvolver(FakeClass, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, FakeClass)
    assert result._buckets == original_pmap._buckets
    assert result._size == original_pmap._size

def test_persistent_when_dirty_and_pm_is_not_instance_of_cls():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import PMap
    class FakeClass:
        _precord_buckets = None
        _precord_size = None
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
    original_pmap = PMap({1: 2})
    evolver = _PRecordEvolver(FakeClass, original_pmap)
    evolver.set(1, 3)
    result = evolver.persistent()
    assert isinstance(result, FakeClass)
    assert result._buckets != original_pmap._buckets
    assert result._size == original_pmap._size


# LLM-generated content at query #13
#--------------------------

def test_set_with_valid_field_and_value():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str, invariant=lambda x: (len(x) > 0, 'ERR_EMPTY'))}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    evolver = _PRecordEvolver(TestRecord, {})
    evolver.set('name', 'Alice')
    result = evolver.persistent()
    assert result['name'] == 'Alice'

def test_set_with_invalid_type_raises_ptype_error():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType, PTypeError
    class TestRecord(CheckedType):
        _precord_fields = {'age': field(type=int)}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    evolver = _PRecordEvolver(TestRecord, {})
    try:
        evolver.set('age', 'not_an_int')
        assert False
    except PTypeError as e:
        assert e.destination_cls == TestRecord
        assert e.field_name == 'age'

def test_set_with_failed_invariant_adds_error_code():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType, InvariantException
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str, invariant=lambda x: (len(x) > 0, 'ERR_EMPTY'))}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    evolver = _PRecordEvolver(TestRecord, {})
    evolver.set('name', '')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'ERR_EMPTY' in e.invariant_errors

def test_set_with_non_existent_field_raises_attribute_error():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    class TestRecord(CheckedType):
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    evolver = _PRecordEvolver(TestRecord, {})
    try:
        evolver.set('unknown', 'value')
        assert False
    except AttributeError as e:
        assert 'unknown' in str(e)

def test_set_with_factory_field_and_ignore_extra():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    def factory_func(value, ignore_extra=False):
        return value.upper()
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str, factory=factory_func)}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    evolver = _PRecordEvolver(TestRecord, {}, _factory_fields={TestRecord._precord_fields['name']}, _ignore_extra=True)
    evolver.set('name', 'alice')
    result = evolver.persistent()
    assert result['name'] == 'ALICE'

def test_set_with_factory_field_without_ignore_extra():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    def factory_func(value):
        return value.upper()
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str, factory=factory_func)}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    evolver = _PRecordEvolver(TestRecord, {}, _factory_fields={TestRecord._precord_fields['name']})
    evolver.set('name', 'alice')
    result = evolver.persistent()
    assert result['name'] == 'ALICE'

def test_set_with_factory_field_invariant_exception_adds_errors():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType, InvariantException
    def factory_func(value):
        raise InvariantException(('ERR_FACTORY',), (), 'Factory failed')
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str, factory=factory_func)}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    evolver = _PRecordEvolver(TestRecord, {}, _factory_fields={TestRecord._precord_fields['name']})
    evolver.set('name', 'alice')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'ERR_FACTORY' in e.invariant_errors

def test_set_with_non_factory_field_uses_original_value():
    from pyrsistent import field
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._checked_types import CheckedType
    class TestRecord(CheckedType):
        _precord_fields = {'name': field(type=str)}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    evolver = _PRecordEvolver(TestRecord, {}, _factory_fields=set())
    evolver.set('name', 'Alice')
    result = evolver.persistent()
    assert result['name'] == 'Alice'


# LLM-generated content at query #14
#--------------------------

def test_precord_constructor_without_special_attributes():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {}
    result = MyRecord(field1='value1', field2='value2')
    assert result['field1'] == 'value1'
    assert result['field2'] == 'value2'

def test_precord_constructor_with_special_attributes():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
        _precord_initial_values = {}
    size = 0
    buckets = []
    result = MyRecord(_precord_size=size, _precord_buckets=buckets)
    assert isinstance(result, MyRecord)

def test_precord_constructor_with_initial_values():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {'field1': lambda: 'default1', 'field2': 'default2'}
    result = MyRecord()
    assert result['field1'] == 'default1'
    assert result['field2'] == 'default2'

def test_precord_constructor_overrides_initial_values():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {'field1': lambda: 'default1', 'field2': 'default2'}
    result = MyRecord(field1='custom1')
    assert result['field1'] == 'custom1'
    assert result['field2'] == 'default2'

def test_precord_constructor_with_factory_fields():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {}
    factory_fields = {'field1': lambda x: x.upper()}
    result = MyRecord(_factory_fields=factory_fields, field1='test', field2='value2')
    assert result['field1'] == 'TEST'
    assert result['field2'] == 'value2'

def test_precord_constructor_with_ignore_extra():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field()}
        _precord_initial_values = {}
    result = MyRecord(_ignore_extra=True, field1='value1', extra_field='extra')
    assert result['field1'] == 'value1'
    assert 'extra_field' not in result

def test_precord_constructor_without_ignore_extra():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field()}
        _precord_initial_values = {}
    try:
        MyRecord(field1='value1', extra_field='extra')
        assert False
    except AttributeError:
        assert True

def test_precord_constructor_empty():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
        _precord_initial_values = {}
    result = MyRecord()
    assert len(result) == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_persistent_when_mandatory_fields_present_and_no_invariants_violated():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        mandatory = field(mandatory=True)
        optional = field()
    instance = TestRecord(mandatory='value')
    evolver = instance.evolver()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.mandatory == 'value'
    assert 'optional' not in result


# LLM-generated content at query #16
#--------------------------

def test_precord_initial_values_condition_true():
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_initial_values = {'x': 1}
    result = TestRecord()
    assert result == {'x': 1}


# LLM-generated content at query #17
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
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (False, 'error_code')})()}
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
        assert 'error_code' in e.invariant_errors

def test_persistent_raises_invariant_exception_on_global_invariant_errors():
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

def test_persistent_returns_pmap_when_not_dirty_and_already_instance():
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

def test_persistent_creates_new_instance_when_dirty():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    evolver.set('new_field', 'new_value')
    result = evolver.persistent()
    assert isinstance(result, MockClass)
    assert result._buckets is not None


# LLM-generated content at query #18
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
    evolver = _PRecordEvolver(MockClass, {})
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()
        assert str(e) == 'Field invariant failed'


# LLM-generated content at query #19
#--------------------------

def test_precord_repr_with_single_field():
    class TestRecord(PRecord):
        _precord_fields = {'name': field(type=str)}
        _precord_initial_values = {}
    record = TestRecord(name='Alice')
    result = repr(record)
    expected = "TestRecord(name='Alice')"
    assert result == expected

def test_precord_repr_with_multiple_fields():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int), 'y': field(type=str)}
        _precord_initial_values = {}
    record = TestRecord(x=10, y='test')
    result = repr(record)
    expected = "TestRecord(x=10, y='test')"
    assert result == expected

def test_precord_repr_with_no_fields():
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_initial_values = {}
    record = TestRecord()
    result = repr(record)
    expected = "TestRecord()"
    assert result == expected

def test_precord_repr_with_nested_values():
    class TestRecord(PRecord):
        _precord_fields = {'data': field(type=dict)}
        _precord_initial_values = {}
    record = TestRecord(data={'key': 'value'})
    result = repr(record)
    expected = "TestRecord(data={'key': 'value'})"
    assert result == expected

def test_precord_repr_with_special_characters_in_field_value():
    class TestRecord(PRecord):
        _precord_fields = {'text': field(type=str)}
        _precord_initial_values = {}
    record = TestRecord(text='line1\nline2')
    result = repr(record)
    expected = "TestRecord(text='line1\\nline2')"
    assert result == expected


# LLM-generated content at query #20
#--------------------------

def test_persistent_returns_instance_of_destination_class():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(TestClass, {})
    result = evolver.persistent()
    assert isinstance(result, TestClass)

def test_persistent_raises_invariant_exception_on_missing_mandatory_fields():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = {'mandatory_field'}
        _precord_invariants = ()
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(TestClass, {})
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.missing_fields == ('TestClass.mandatory_field',)

def test_persistent_raises_invariant_exception_on_field_invariant_errors():
    class TestClass:
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (False, 'error_code')})()}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: {'field'}
    evolver = _PRecordEvolver(TestClass, {'field': 'value'})
    evolver.set('field', 'value')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('error_code',)

def test_persistent_raises_invariant_exception_on_global_invariant_failure():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [lambda x: (False, 'global_error')]
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(TestClass, {})
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)

def test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    original_pmap = TestClass()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    result = evolver.persistent()
    assert result is original_pmap

def test_persistent_creates_new_instance_if_dirty():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    original_pmap = TestClass()
    evolver = _PRecordEvolver(TestClass, original_pmap)
    evolver.set('new_field', 'new_value')
    result = evolver.persistent()
    assert result is not original_pmap
    assert isinstance(result, TestClass)


# LLM-generated content at query #21
#--------------------------

def test_precord_repr_returns_correct_format():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {}
    record = TestRecord(field1='value1', field2=42)
    result = repr(record)
    expected_start = "TestRecord("
    expected_end = ")"
    assert result.startswith(expected_start)
    assert result.endswith(expected_end)
    assert 'field1=' in result
    assert "'value1'" in result or '"value1"' in result
    assert 'field2=' in result
    assert '42' in result
    parts = result[len(expected_start):-len(expected_end)].split(', ')
    assert len(parts) == 2
    assert any('field1=' in part and ("'value1'" in part or '"value1"' in part) for part in parts)
    assert any('field2=' in part and '42' in part for part in parts)


# LLM-generated content at query #22
#--------------------------

```python
def test_persistent_when_is_dirty_true_and_pm_not_instance_of_cls():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: []
    original_pmap = type('MockPMap', (), {'_buckets': None, '_size': None})()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._destination_cls = MockCls
    evolver._original_pmap = original_pmap
    evolver._buckets = []
    evolver._extra_items = {}
    evolver.is_dirty = lambda: True
    pm = type('MockPMap', (), {'_buckets': 'buckets', '_size': 'size'})()
    super_persistent = lambda self: pm
    evolver.persistent = lambda: _PRecordEvolver.persistent.__get__(evolver, _PRecordEvolver)()
    type(evolver).persistent = lambda self: MockCls(_precord_buckets=pm._buckets, _precord_size=pm._size)
    result = evolver.persistent()
    assert isinstance(result, MockCls)
    assert result._buckets == pm._buckets
    assert result._size == pm._size

def test_persistent_when_is_dirty_true_and_pm_is_instance_of_cls():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: []
    original_pmap = type('MockPMap', (), {'_buckets': None, '_size': None})()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._destination_cls = MockCls
    evolver._original_pmap = original_pmap
    evolver._buckets = []
    evolver._extra_items = {}
    evolver.is_dirty = lambda: True
    pm = MockCls(_precord_buckets='buckets', _precord_size='size')
    type(evolver).persistent = lambda self: MockCls(_precord_buckets=pm._buckets, _precord_size=pm._size)
    result = evolver.persistent()
    assert isinstance(result, MockCls)
    assert result._buckets == pm._buckets
    assert result._size == pm._size

def test_persistent_when_is_dirty_false_and_pm_not_instance_of_cls():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: []
    original_pmap = type('MockPMap', (), {'_buckets': None, '_size': None})()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._destination_cls = MockCls
    evolver._original_pmap = original_pmap
    evolver._buckets = []
    evolver._extra_items = {}
    evolver.is_dirty = lambda: False
    pm = type('MockPMap', (), {'_buckets': 'buckets', '_size': 'size'})()
    type(evolver).persistent = lambda self: MockCls(_precord_buckets=pm._buckets, _precord_size=pm._size)
    result = evolver.persistent()
    assert isinstance(result, MockCls)
    assert result._buckets == pm._buckets
    assert result._size == pm._size

def test_persistent_when_is_dirty_false_and_pm_is_instance_of_cls():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: []
    original_pmap = type('MockPMap', (), {'_buckets': None, '_size': None})()
    evolver = _PRecordEvolver(MockCls, original_pmap)
    evolver._destination_cls = MockCls
    evolver._original_pmap = original_pmap
    evolver._buckets = []
    evolver._extra_items = {}
    evolver.is_dirty = lambda: False
    pm = MockCls(_precord_buckets='buckets', _precord_size='size')
    type(evolver).persistent = lambda self: pm
    result = evolver.persistent()
    assert result is pm
    assert result._buckets == pm._buckets
    assert result._size == pm._size


# LLM-generated content at query #23
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    evolver = _PRecordEvolver(cls=MockClass, original_pmap=pmap({}))
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()
        assert str(e) == 'Field invariant failed'

def test_persistent_raises_invariant_exception_when_missing_fields_present():
    evolver = _PRecordEvolver(cls=MockClass, original_pmap=pmap({}))
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['MockClass.field1']
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('MockClass.field1',)
        assert str(e) == 'Field invariant failed'

def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present():
    evolver = _PRecordEvolver(cls=MockClass, original_pmap=pmap({}))
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = ['MockClass.field1', 'MockClass.field2']
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ('MockClass.field1', 'MockClass.field2')
        assert str(e) == 'Field invariant failed'

def test_persistent_does_not_raise_when_no_errors_or_missing_fields():
    evolver = _PRecordEvolver(cls=MockClass, original_pmap=pmap({}))
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    evolver._destination_cls._precord_invariants = []
    result = evolver.persistent()
    assert isinstance(result, MockClass)


# LLM-generated content at query #24
#--------------------------

def test_precord_repr_returns_correct_format():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=str), 'field2': field(type=int)}
        _precord_initial_values = {}
    record = TestRecord(field1='value1', field2=42)
    result = repr(record)
    expected_start = "TestRecord("
    expected_end = ")"
    assert result.startswith(expected_start)
    assert result.endswith(expected_end)
    assert 'field1=' in result
    assert "'value1'" in result or '"value1"' in result
    assert 'field2=' in result
    assert '42' in result
    parts = result[len(expected_start):-len(expected_end)].split(', ')
    assert len(parts) == 2
    assert any('field1=' in part and ("'value1'" in part or '"value1"' in part) for part in parts)
    assert any('field2=' in part and '42' in part for part in parts)


# LLM-generated content at query #25
#--------------------------

```python
def test_persistent_creates_new_instance_when_is_dirty():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        x = field()
    evolver = TestRecord().evolver()
    evolver['x'] = 1
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 1

def test_persistent_creates_new_instance_when_pm_not_instance_of_cls():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        x = field()
    evolver = TestRecord().evolver()
    evolver._original_pmap = {}
    result = evolver.persistent()
    assert isinstance(result, TestRecord)

def test_persistent_returns_pm_when_not_dirty_and_pm_is_instance_of_cls():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        x = field()
    original = TestRecord(x=1)
    evolver = original.evolver()
    result = evolver.persistent()
    assert result is original

def test_persistent_handles_mandatory_fields():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        x = field(mandatory=True)
    evolver = TestRecord().evolver()
    try:
        evolver.persistent()
        assert False
    except Exception as e:
        assert 'Field invariant failed' in str(e)

def test_persistent_handles_invariant_errors():
    from pyrsistent import PRecord, field
    def invariant(value):
        return value > 0, 'INVARIANT'
    class TestRecord(PRecord):
        x = field(invariant=invariant)
    evolver = TestRecord(x=1).evolver()
    evolver['x'] = -1
    try:
        evolver.persistent()
        assert False
    except Exception as e:
        assert 'Field invariant failed' in str(e)

def test_persistent_checks_global_invariants():
    from pyrsistent import PRecord, field
    def global_invariant(record):
        return record.get('x', 0) > 0, 'GLOBAL_INVARIANT'
    class TestRecord(PRecord):
        x = field()
        _precord_invariants = (global_invariant,)
    evolver = TestRecord(x=1).evolver()
    evolver['x'] = -1
    try:
        evolver.persistent()
        assert False
    except Exception as e:
        assert 'Global invariant failed' in str(e)


# LLM-generated content at query #26
#--------------------------

```python
def test_persistent_when_cls_has_mandatory_fields():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import InvariantException

    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'mandatory_field'}
        _precord_invariants = ()
        __name__ = 'TestRecord'

        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size

        def keys(self):
            return []

    evolver = _PRecordEvolver(TestRecord, None)
    evolver._missing_fields = []
    evolver._invariant_error_codes = []
    evolver._destination_cls = TestRecord
    evolver.is_dirty = lambda: False
    evolver._original_pmap = type('PMap', (), {'_buckets': None, '_size': 0})()
    evolver._buckets = None
    evolver._size = 0

    try:
        evolver.persistent()
    except InvariantException as e:
        assert 'TestRecord.mandatory_field' in e.missing_fields


# LLM-generated content at query #27
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
        _precord_fields = {'field': type('Field', (), {'serializer': custom_serializer})()}
    record = TestRecord(field='data')
    result = record.serialize()
    expected = {'field': 'custom_data'}
    assert result == expected

def test_serialize_with_multiple_fields_and_mixed_serializers():
    def serializer1(format, value):
        return value * 2
    class TestRecord(PRecord):
        _precord_fields = {
            'field1': type('Field', (), {'serializer': serializer1})(),
            'field2': None,
            'field3': type('Field', (), {'serializer': lambda format, v: v.upper()})()
        }
    record = TestRecord(field1=3, field2='hello', field3='world')
    result = record.serialize()
    expected = {'field1': 6, 'field2': 'hello', 'field3': 'WORLD'}
    assert result == expected

def test_serialize_with_format_parameter():
    def format_serializer(format, value):
        return f'{format}:{value}'
    class TestRecord(PRecord):
        _precord_fields = {'field': type('Field', (), {'serializer': format_serializer})()}
    record = TestRecord(field='test')
    result = record.serialize('json')
    expected = {'field': 'json:test'}
    assert result == expected

def test_serialize_on_empty_record():
    class TestRecord(PRecord):
        _precord_fields = {}
    record = TestRecord()
    result = record.serialize()
    expected = {}
    assert result == expected


# LLM-generated content at query #28
#--------------------------

```python
def test_set_fields_inherits_fields_from_bases():
    from pyrsistent._field_common import set_fields
    from pyrsistent import PField, field

    class Base:
        _precord_fields = {'base_field': PField(type=str, mandatory=False, invariant=lambda x: (True, ''))}

    dct = {}
    bases = (Base,)
    set_fields(dct, bases, name='_precord_fields')
    assert '_precord_fields' in dct
    assert 'base_field' in dct['_precord_fields']
    assert dct['_precord_fields']['base_field'] is Base._precord_fields['base_field']

def test_set_fields_adds_new_fields_from_dct():
    from pyrsistent._field_common import set_fields
    from pyrsistent import PField, field

    class Base:
        _precord_fields = {'base_field': PField(type=str, mandatory=False, invariant=lambda x: (True, ''))}

    new_field = PField(type=int, mandatory=False, invariant=lambda x: (True, ''))
    dct = {'new_field': new_field}
    bases = (Base,)
    set_fields(dct, bases, name='_precord_fields')
    assert '_precord_fields' in dct
    assert 'base_field' in dct['_precord_fields']
    assert 'new_field' in dct['_precord_fields']
    assert dct['_precord_fields']['new_field'] is new_field
    assert 'new_field' not in dct

def test_set_fields_overrides_base_fields_with_dct():
    from pyrsistent._field_common import set_fields
    from pyrsistent import PField, field

    base_field = PField(type=str, mandatory=False, invariant=lambda x: (True, ''))
    class Base:
        _precord_fields = {'field': base_field}

    new_field = PField(type=int, mandatory=False, invariant=lambda x: (True, ''))
    dct = {'field': new_field}
    bases = (Base,)
    set_fields(dct, bases, name='_precord_fields')
    assert '_precord_fields' in dct
    assert 'field' in dct['_precord_fields']
    assert dct['_precord_fields']['field'] is new_field
    assert dct['_precord_fields']['field'] is not base_field
    assert 'field' not in dct

def test_set_fields_handles_empty_bases():
    from pyrsistent._field_common import set_fields
    from pyrsistent import PField, field

    dct = {}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    assert '_precord_fields' in dct
    assert dct['_precord_fields'] == {}

def test_set_fields_handles_multiple_bases():
    from pyrsistent._field_common import set_fields
    from pyrsistent import PField, field

    class Base1:
        _precord_fields = {'field1': PField(type=str, mandatory=False, invariant=lambda x: (True, ''))}

    class Base2:
        _precord_fields = {'field2': PField(type=int, mandatory=False, invariant=lambda x: (True, ''))}

    dct = {}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    assert '_precord_fields' in dct
    assert 'field1' in dct['_precord_fields']
    assert 'field2' in dct['_precord_fields']

def test_set_fields_handles_diamond_inheritance():
    from pyrsistent._field_common import set_fields
    from pyrsistent import PField, field

    class Base:
        _precord_fields = {'common': PField(type=str, mandatory=False, invariant=lambda x: (True, ''))}

    class Left(Base):
        _precord_fields = {'left': PField(type=int, mandatory=False, invariant=lambda x: (True, ''))}

    class Right(Base):
        _precord_fields = {'right': PField(type=float, mandatory=False, invariant=lambda x: (True, ''))}

    dct = {}
    bases = (Left, Right)
    set_fields(dct, bases, name='_precord_fields')
    assert '_precord_fields' in dct
    assert 'common' in dct['_precord_fields']
    assert 'left' in dct['_precord_fields']
    assert 'right' in dct['_precord_fields']

def test_store_invariants_inherits_invariants_from_bases():
    from pyrsistent._checked_types import store_invariants

    def base_invariant(obj):
        return True, ''

    class Base:
        __invariant__ = base_invariant

    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 1
    assert dct['_precord_invariants'][0].__wrapped__ is base_invariant

def test_store_invariants_adds_new_invariants_from_dct():
    from pyrsistent._checked_types import store_invariants

    def base_invariant(obj):
        return True, ''

    def new_invariant(obj):
        return True, ''

    class Base:
        __invariant__ = base_invariant

    dct = {'__invariant__': new_invariant}
    bases = (Base,)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 2
    assert dct['_precord_invariants'][0].__wrapped__ is new_invariant
    assert dct['_precord_invariants'][1].__wrapped__ is base_invariant

def test_store_invariants_handles_multiple_inheritance():
    from pyrsistent._checked_types import store_invariants

    def invariant1(obj):
        return True, ''

    def invariant2(obj):
        return True, ''

    class Base1:
        __invariant__ = invariant1

    class Base2:
        __invariant__ = invariant2

    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 2
    assert dct['_precord_invariants'][0].__wrapped__ is invariant1
    assert dct['_precord_invariants'][1].__wrapped__ is invariant2

def test_store_invariants_raises_type_error_for_non_callable():
    from pyrsistent._checked_types import store_invariants

    class Base:
        __invariant__ = 'not callable'

    dct = {}
    bases = (Base,)
    try:
        store_invariants(dct, bases, '_precord_invariants', '__invariant__')
        assert False
    except TypeError:
        pass

def test_store_invariants_wraps_invariants():
    from pyrsistent._checked_types import store_invariants

    def invariant(obj):
        return [(True, ''), (False, 'error')]

    dct = {'__invariant__': invariant}
    bases = ()
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 1
    wrapped = dct['_precord_invariants'][0]
    result = wrapped(None)
    assert result == (False, ('error',))

def test_precord_meta_creates_mandatory_fields():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent import PField, field, PFIELD_NO_INITIAL

    class TestRecord(metaclass=_PRecordMeta):
        _precord_fields = {
            'mandatory': PField(type=str, mandatory=True, invariant=lambda x: (True, '')),
            'optional': PField(type=int, mandatory=False, invariant=lambda x: (True, ''))
        }

    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert TestRecord._precord_mandatory_fields == {'mandatory'}

def test_precord_meta_creates_initial_values():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent import PField, field, PFIELD_NO_INITIAL

    class TestRecord(metaclass=_PRecordMeta):
        _precord_fields = {
            'with_initial': PField(type=str, mandatory=False, initial='default', invariant=lambda x: (True, '')),
            'without_initial': PField(type=int, mandatory=False, initial=PFIELD_NO_INITIAL, invariant=lambda x: (True, ''))
        }

    assert hasattr(TestRecord, '_precord_


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_persistent_returns_instance_of_destination_class():
    class TestRecord(PRecord):
        field = field(type=int)
    evolver = TestRecord().evolver()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)

def test_persistent_raises_invariant_exception_on_missing_mandatory_fields():
    class TestRecord(PRecord):
        field = field(type=int, mandatory=True)
    evolver = TestRecord().evolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.missing_fields == ('TestRecord.field',)

def test_persistent_raises_invariant_exception_on_field_invariant_failure():
    def invariant(value):
        return value > 0, 'INVARIANT'
    class TestRecord(PRecord):
        field = field(type=int, invariant=invariant)
    evolver = TestRecord(field=0).evolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('INVARIANT',)

def test_persistent_raises_invariant_exception_on_global_invariant_failure():
    def global_invariant(record):
        return record.get('field', 0) > 0, 'GLOBAL_INVARIANT'
    class TestRecord(PRecord):
        __invariants__ = [global_invariant]
    evolver = TestRecord().evolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('GLOBAL_INVARIANT',)

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
    evolver = TestRecord().evolver()
    evolver['field'] = 1
    result = evolver.persistent()
    assert result == TestRecord(field=1)
    assert result is not TestRecord()

def test_persistent_creates_new_instance_if_pmap_not_of_destination_class():
    class TestRecord(PRecord):
        field = field(type=int)
    evolver = TestRecord().evolver()
    evolver._original_pmap = pmap({})
    result = evolver.persistent()
    assert isinstance(result, TestRecord)

def test_persistent_aggregates_multiple_invariant_errors():
    def invariant1(value):
        return value > 0, 'INVARIANT1'
    def invariant2(value):
        return value < 10, 'INVARIANT2'
    class TestRecord(PRecord):
        field1 = field(type=int, invariant=invariant1)
        field2 = field(type=int, invariant=invariant2)
    evolver = TestRecord(field1=0, field2=20).evolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert set(e.invariant_errors) == {'INVARIANT1', 'INVARIANT2'}

def test_persistent_aggregates_missing_fields_and_invariant_errors():
    def invariant(value):
        return value > 0, 'INVARIANT'
    class TestRecord(PRecord):
        field1 = field(type=int, mandatory=True)
        field2 = field(type=int, invariant=invariant)
    evolver = TestRecord(field2=0).evolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.missing_fields == ('TestRecord.field1',)
        assert e.invariant_errors == ('INVARIANT',)


# LLM-generated content at query #2
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
    instance = TestRecord(a=1, b=2)
    assert isinstance(instance, TestRecord)
    assert instance.get('a') == 1
    assert instance.get('b') == 2

def test_precord_new_applies_initial_values_from_class():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': object(), 'b': object()}
    TestRecord._precord_initial_values = {'a': lambda: 10, 'b': 20}
    instance = TestRecord()
    assert instance.get('a') == 10
    assert instance.get('b') == 20

def test_precord_new_overrides_initial_values_with_kwargs():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': object(), 'b': object()}
    TestRecord._precord_initial_values = {'a': lambda: 10, 'b': 20}
    instance = TestRecord(a=30)
    assert instance.get('a') == 30
    assert instance.get('b') == 20

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

def test_precord_new_raises_attribute_error_for_invalid_field():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'valid': object()}
    TestRecord._precord_initial_values = {}
    try:
        TestRecord(invalid=5)
        assert False
    except AttributeError as e:
        assert "'invalid' is not among the specified fields for TestRecord" in str(e)

def test_precord_new_handles_invariant_exception():
    class TestRecord(PRecord):
        pass
    field = object()
    field.factory = lambda x: (_ for _ in ()).throw(InvariantException((), (), ''))
    field.invariant = lambda x: (True, None)
    TestRecord._precord_fields = {'a': field}
    TestRecord._precord_initial_values = {}
    try:
        TestRecord(a=1)
        assert False
    except InvariantException:
        pass

def test_precord_new_checks_mandatory_fields():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': object(), 'b': object()}
    TestRecord._precord_initial_values = {}
    TestRecord._precord_mandatory_fields = {'a', 'b'}
    try:
        TestRecord(a=1)
        assert False
    except InvariantException as e:
        assert any('TestRecord.b' in missing for missing in e.missing_fields)

def test_precord_new_validates_global_invariants():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {}
    TestRecord._precord_initial_values = {}
    TestRecord._precord_invariants = [lambda x: (_ for _ in ()).throw(InvariantException((), (), ''))]
    try:
        TestRecord()
        assert False
    except InvariantException:
        pass


# LLM-generated content at query #3
#--------------------------

def test_serialize_without_custom_serializer():
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
    result = record.serialize('json')
    expected = {'field1': 'json_value1', 'field2': 42}
    assert result == expected

def test_serialize_empty_record():
    class TestRecord(PRecord):
        _precord_fields = {}
    record = TestRecord()
    result = record.serialize()
    expected = {}
    assert result == expected

def test_serialize_with_none_values():
    class TestRecord(PRecord):
        _precord_fields = {'field1': None, 'field2': None}
    record = TestRecord(field1=None, field2=None)
    result = record.serialize()
    expected = {'field1': None, 'field2': None}
    assert result == expected


# LLM-generated content at query #4
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"

    class MockPMap:
        _buckets = {}
        _size = 0

    evolver = _PRecordEvolver(MockClass, MockPMap())
    evolver._invariant_error_codes = ["error1"]
    evolver._missing_fields = []
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ("error1",)
        assert e.missing_fields == ()
        assert e.message == "Field invariant failed"
    else:
        assert False

def test_persistent_raises_invariant_exception_when_missing_fields_present():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"

    class MockPMap:
        _buckets = {}
        _size = 0

    evolver = _PRecordEvolver(MockClass, MockPMap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = ["MockClass.field1"]
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ("MockClass.field1",)
        assert e.message == "Field invariant failed"
    else:
        assert False

def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"

    class MockPMap:
        _buckets = {}
        _size = 0

    evolver = _PRecordEvolver(MockClass, MockPMap())
    evolver._invariant_error_codes = ["error1", "error2"]
    evolver._missing_fields = ["MockClass.field1", "MockClass.field2"]
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ("error1", "error2")
        assert e.missing_fields == ("MockClass.field1", "MockClass.field2")
        assert e.message == "Field invariant failed"
    else:
        assert False


# LLM-generated content at query #5
#--------------------------

def test_set_with_valid_field_and_value():
    from pyrsistent import InvariantException, PTypeError, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import is_field_ignore_extra_complaint, check_type
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField((int,), lambda x: x, lambda x: (True, None))}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    evolver.set('key', 5)
    assert evolver._original_pmap['key'] == 5

def test_set_with_field_factory_exception():
    from pyrsistent import InvariantException, PTypeError, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import is_field_ignore_extra_complaint, check_type
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField((int,), lambda x: (_ for _ in ()).throw(InvariantException((), (), '')), lambda x: (True, None))}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    evolver.set('key', 5)
    assert len(evolver._invariant_error_codes) == 0
    assert len(evolver._missing_fields) == 0

def test_set_with_field_factory_invariant_exception():
    from pyrsistent import InvariantException, PTypeError, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import is_field_ignore_extra_complaint, check_type
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    def factory_raising_invariant(x):
        raise InvariantException(('error1',), ('missing1',), '')
    class MockDestinationCls:
        _precord_fields = {'key': MockField((int,), factory_raising_invariant, lambda x: (True, None))}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    evolver.set('key', 5)
    assert evolver._invariant_error_codes == ['error1']
    assert evolver._missing_fields == ['missing1']

def test_set_with_ignore_extra_complaint():
    from pyrsistent import InvariantException, PTypeError, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import is_field_ignore_extra_complaint, check_type
    import inspect
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    def factory_with_ignore_extra(x, ignore_extra=False):
        return x
    factory_with_ignore_extra.__signature__ = inspect.signature(factory_with_ignore_extra)
    class MockDestinationCls:
        _precord_fields = {'key': MockField((int,), factory_with_ignore_extra, lambda x: (True, None))}
    evolver = _PRecordEvolver(MockDestinationCls, {}, _ignore_extra=True)
    evolver.set('key', 5)
    assert evolver._original_pmap['key'] == 5

def test_set_with_type_check_failure():
    from pyrsistent import InvariantException, PTypeError, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import is_field_ignore_extra_complaint, check_type
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField((int,), lambda x: x, lambda x: (True, None))}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    try:
        evolver.set('key', 'not_an_int')
        assert False
    except PTypeError:
        assert True

def test_set_with_invariant_failure():
    from pyrsistent import InvariantException, PTypeError, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import is_field_ignore_extra_complaint, check_type
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    class MockDestinationCls:
        _precord_fields = {'key': MockField((int,), lambda x: x, lambda x: (False, 'invariant_error'))}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    evolver.set('key', 5)
    assert evolver._invariant_error_codes == ['invariant_error']

def test_set_with_nonexistent_field():
    from pyrsistent import InvariantException, PTypeError, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import is_field_ignore_extra_complaint, check_type
    class MockDestinationCls:
        _precord_fields = {}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    try:
        evolver.set('nonexistent', 5)
        assert False
    except AttributeError as e:
        assert "'nonexistent' is not among the specified fields" in str(e)

def test_set_with_factory_fields_skipping_factory():
    from pyrsistent import InvariantException, PTypeError, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import is_field_ignore_extra_complaint, check_type
    class MockField:
        def __init__(self, type, factory, invariant):
            self.type = type
            self.factory = factory
            self.invariant = invariant
    factory_called = []
    def factory(x):
        factory_called.append(x)
        return x
    field = MockField((int,), factory, lambda x: (True, None))
    class MockDestinationCls:
        _precord_fields = {'key': field}
    evolver = _PRecordEvolver(MockDestinationCls, {}, _factory_fields=set())
    evolver.set('key', 5)
    assert factory_called == []
    assert evolver._original_pmap['key'] == 5


# LLM-generated content at query #6
#--------------------------

def test_serialize_with_custom_serializer():
    from pyrsistent import PRecord, field
    from pyrsistent import serialize as ser_func
    class TestRecord(PRecord):
        name = field(type=str)
        value = field(type=int, serializer=lambda fmt, v: v * 2)
    rec = TestRecord(name="test", value=5)
    result = rec.serialize()
    expected = {"name": "test", "value": 10}
    assert result == expected

def test_serialize_without_custom_serializer():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        name = field(type=str)
        value = field(type=int)
    rec = TestRecord(name="test", value=5)
    result = rec.serialize()
    expected = {"name": "test", "value": 5}
    assert result == expected

def test_serialize_with_format_parameter():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        data = field(type=str, serializer=lambda fmt, v: f"{fmt}:{v}")
    rec = TestRecord(data="info")
    result = rec.serialize("json")
    expected = {"data": "json:info"}
    assert result == expected

def test_serialize_empty_record():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        pass
    rec = TestRecord()
    result = rec.serialize()
    expected = {}
    assert result == expected

def test_serialize_multiple_fields_mixed_serializers():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        a = field(type=int, serializer=lambda fmt, v: v + 1)
        b = field(type=str)
        c = field(type=float, serializer=lambda fmt, v: v * 2)
    rec = TestRecord(a=10, b="hello", c=3.5)
    result = rec.serialize()
    expected = {"a": 11, "b": "hello", "c": 7.0}
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_serialize_with_custom_serializer():
    from pyrsistent import PRecord, field
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
        value = field()
    record = TestRecord(name="test", value=42)
    result = record.serialize()
    expected = {"name": "test", "value": 42}
    assert result == expected

def test_serialize_with_format_parameter():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        value = field(serializer=lambda fmt, v: f"{fmt}_{v}")
    record = TestRecord(value=100)
    result = record.serialize("fmt")
    expected = {"value": "fmt_100"}
    assert result == expected

def test_serialize_with_none_format():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        value = field(serializer=lambda fmt, v: f"serialized_{v}")
    record = TestRecord(value=200)
    result = record.serialize(None)
    expected = {"value": "serialized_200"}
    assert result == expected

def test_serialize_empty_record():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        pass
    record = TestRecord()
    result = record.serialize()
    expected = {}
    assert result == expected

def test_serialize_multiple_fields_mixed_serializers():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        a = field()
        b = field(serializer=lambda fmt, v: v * 2)
        c = field()
    record = TestRecord(a=1, b=2, c=3)
    result = record.serialize()
    expected = {"a": 1, "b": 4, "c": 3}
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test_precord_initial_values_condition_true():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {'field1': lambda: 42, 'field2': 'default'}
    kwargs = {'field2': 'custom'}
    initial_values = dict((k, v() if callable(v) else v) for k, v in TestRecord._precord_initial_values.items())
    initial_values.update(kwargs)
    assert initial_values == {'field1': 42, 'field2': 'custom'}


# LLM-generated content at query #9
#--------------------------

```python
def test_persistent_creates_new_instance_when_is_dirty():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._precord import PRecord

    class TestRecord(PRecord):
        pass

    original = TestRecord()
    evolver = _PRecordEvolver(TestRecord, original._map)
    evolver.set('new_field', 'value')
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result == {'new_field': 'value'}

def test_persistent_creates_new_instance_when_pm_not_instance_of_cls():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._precord import PRecord
    from pyrsistent._pmap import pmap

    class TestRecord(PRecord):
        pass

    original = TestRecord()
    evolver = _PRecordEvolver(TestRecord, original._map)
    evolver._destination_cls = TestRecord
    evolver._original_pmap = pmap({'different': 'structure'})
    result = evolver.persistent()
    assert isinstance(result, TestRecord)

def test_persistent_returns_pm_when_not_dirty_and_pm_is_instance_of_cls():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._precord import PRecord

    class TestRecord(PRecord):
        pass

    original = TestRecord()
    evolver = _PRecordEvolver(TestRecord, original._map)
    result = evolver.persistent()
    assert result is original

def test_persistent_handles_mandatory_fields():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._precord import PRecord

    class TestRecord(PRecord):
        __mandatory_fields__ = {'required_field'}

    original = TestRecord()
    evolver = _PRecordEvolver(TestRecord, original._map)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'required_field' in str(e)

def test_persistent_handles_invariant_errors():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._precord import PRecord

    class TestRecord(PRecord):
        pass

    original = TestRecord()
    evolver = _PRecordEvolver(TestRecord, original._map)
    evolver._invariant_error_codes = ['error1']
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'error1' in e.invariant_errors

def test_persistent_checks_global_invariants():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._precord import PRecord

    def failing_invariant(record):
        return (False, 'global_error')

    class TestRecord(PRecord):
        __invariants__ = [failing_invariant]

    original = TestRecord()
    evolver = _PRecordEvolver(TestRecord, original._map)
    evolver.set('field', 'value')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'global_error' in str(e)


# LLM-generated content at query #10
#--------------------------

```python
def test_persistent_when_cls_has_mandatory_fields():
    class TestRecord:
        _precord_mandatory_fields = {"field1", "field2"}
        _precord_invariants = []
        __name__ = "TestRecord"

        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: []

    evolver = _PRecordEvolver(TestRecord, None)
    evolver._destination_cls = TestRecord
    evolver._missing_fields = []
    evolver._invariant_error_codes = []
    evolver.is_dirty = lambda: False
    evolver.persistent = lambda: TestRecord()
    result = evolver.persistent()
    assert TestRecord._precord_mandatory_fields


# LLM-generated content at query #11
#--------------------------

def test___new___sets_fields_correctly():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    def _all_dicts(bases, seen=None):
        if seen is None:
            seen = set()
        for cls in bases:
            if cls in seen:
                continue
            seen.add(cls)
            yield cls.__dict__
            for b in _all_dicts(cls.__bases__, seen):
                yield b
    def wrap_invariant(invariant):
        def f(*args, **kwargs):
            result = invariant(*args, **kwargs)
            if isinstance(result[0], bool):
                return result
            return _merge_invariant_results(result)
        return f
    def _merge_invariant_results(result):
        verdict = True
        data = []
        for verd, dat in result:
            if not verd:
                verdict = False
                data.append(dat)
        return verdict, tuple(data)
    def store_invariants(dct, bases, destination_name, source_name):
        invariants = []
        for ns in [dct] + list(_all_dicts(bases)):
            try:
                invariant = ns[source_name]
            except KeyError:
                continue
            invariants.append(invariant)
        if not all(callable(invariant) for invariant in invariants):
            raise TypeError('Invariants must be callable')
        dct[destination_name] = tuple(wrap_invariant(inv) for inv in invariants)
    class Base1(metaclass=type):
        pass
    Base1._precord_fields = {'base_field': MockField()}
    Base1.__invariant__ = lambda x: (True,)
    class Base2(metaclass=type):
        pass
    Base2._precord_fields = {'another_field': MockField(mandatory=True)}
    dct = {'custom_field': _PField()}
    dct['__invariant__'] = lambda x: (False, 'error')
    class _PRecordMeta(type):
        def __new__(mcs, name, bases, dct):
            set_fields(dct, bases, name='_precord_fields')
            store_invariants(dct, bases, '_precord_invariants', '__invariant__')
            dct['_precord_mandatory_fields'] = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
            dct['_precord_initial_values'] = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
            dct['__slots__'] = ()
            return super(_PRecordMeta, mcs).__new__(mcs, name, bases, dct)
    NewClass = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base1, Base2), dct)
    assert '_precord_fields' in NewClass.__dict__
    fields = NewClass._precord_fields
    assert 'base_field' in fields
    assert 'another_field' in fields
    assert 'custom_field' in fields
    assert isinstance(fields['custom_field'], _PField)
    assert '_precord_invariants' in NewClass.__dict__
    invariants = NewClass._precord_invariants
    assert len(invariants) == 3
    assert all(callable(inv) for inv in invariants)
    assert '_precord_mandatory_fields' in NewClass.__dict__
    mandatory = NewClass._precord_mandatory_fields
    assert 'another_field' in mandatory
    assert 'base_field' not in mandatory
    assert 'custom_field' not in mandatory
    assert '_precord_initial_values' in NewClass.__dict__
    initial_values = NewClass._precord_initial_values
    assert isinstance(initial_values, dict)
    assert '__slots__' in NewClass.__dict__
    assert NewClass.__slots__ == ()

def test___new___handles_no_invariants():
    class MockField:
        pass
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    def _all_dicts(bases, seen=None):
        if seen is None:
            seen = set()
        for cls in bases:
            if cls in seen:
                continue
            seen.add(cls)
            yield cls.__dict__
            for b in _all_dicts(cls.__bases__, seen):
                yield b
    def wrap_invariant(invariant):
        def f(*args, **kwargs):
            result = invariant(*args, **kwargs)
            if isinstance(result[0], bool):
                return result
            return _merge_invariant_results(result)
        return f
    def _merge_invariant_results(result):
        verdict = True
        data = []
        for verd, dat in result:
            if not verd:
                verdict = False
                data.append(dat)
        return verdict, tuple(data)
    def store_invariants(dct, bases, destination_name, source_name):
        invariants = []
        for ns in [dct] + list(_all_dicts(bases)):
            try:
                invariant = ns[source_name]
            except KeyError:
                continue
            invariants.append(invariant)
        if not all(callable(invariant) for invariant in invariants):
            raise TypeError('Invariants must be callable')
        dct[destination_name] = tuple(wrap_invariant(inv) for inv in invariants)
    class Base(metaclass=type):
        pass
    Base._precord_fields = {}
    dct = {}
    class _PRecordMeta(type):
        def __new__(mcs, name, bases, dct):
            set_fields(dct, bases, name='_precord_fields')
            store_invariants(dct, bases, '_precord_invariants', '__invariant__')
            dct['_precord_mandatory_fields'] = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
            dct['_precord_initial_values'] = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
            dct['__slots__'] = ()
            return super(_PRecordMeta, mcs).__new__(mcs, name, bases, dct)
    NewClass = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), dct)
    assert '_precord_fields' in NewClass.__dict__
    assert NewClass._precord_fields == {}
    assert '_precord_invariants' in NewClass.__dict__
    assert NewClass._precord_invariants == ()
    assert '_precord_mandatory_fields' in NewClass.__dict__
    assert NewClass._precord_mandatory_fields == set()
    assert '_precord_initial_values' in NewClass.__dict__
    assert NewClass._precord_initial_values == {}
    assert '__slots__' in NewClass.__dict__
    assert NewClass.__slots__ == ()

def test___new___raises_on_non_callable_invariant():
    class MockField:
        pass
    PFIELD_NO_INITIAL = object()
    class MockPField(MockField):
        pass
    _PField = MockPField
    def set_fields(dct, bases, name):
        dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
        for k, v in list(dct.items()):
            if isinstance(v, _PField):
                dct[name][k] = v
                del dct[k]
    def _all_dicts(bases, seen=None):
        if seen is None:
            seen = set()
        for cls in bases:
            if cls in seen:
                continue
            seen.add(cls)
            yield cls.__dict__
            for b in _all_dicts(cls.__bases__, seen):
                yield b
    def wrap_invariant(invariant):
        def f(*args, **kwargs):
            result = invariant(*args, **kwargs)
            if isinstance(result[0], bool):
                return result
            return _merge_invariant_results(result)
        return f
    def _merge_invariant_results(result):
        verdict = True
        data = []
        for verd, dat in result:
            if not verd:
                verdict =


# LLM-generated content at query #12
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
    evolver._invariant_error_codes = ["error1"]
    evolver._missing_fields = []
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ("error1",)
        assert e.missing_fields == ()
        assert str(e) == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #13
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import CheckedType
    import inspect

    class MockField:
        def __init__(self, field_type, factory_params):
            self.type = field_type
            self.factory = lambda **kwargs: None
            sig = inspect.signature(self.factory)
            params = dict(sig.parameters)
            for param in factory_params:
                params[param] = inspect.Parameter(param, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            self.factory.__signature__ = sig.replace(parameters=list(params.values()))

    class MockCheckedType(CheckedType):
        pass

    field_type = (MockCheckedType,)
    field = MockField(field_type, ['ignore_extra'])
    result = is_field_ignore_extra_complaint(CheckedType, field, True)
    assert result == True


# LLM-generated content at query #14
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

    field_type_set = {CheckedType}
    factory_params = [inspect.Parameter('ignore_extra', inspect.Parameter.KEYWORD_ONLY, default=True)]
    field = MockField(field_type_set, factory_params)
    result = is_field_ignore_extra_complaint(CheckedType, field, True)
    assert result == True


# LLM-generated content at query #15
#--------------------------

```python
def test_set_fields_creates_precord_fields():
    class MockPField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial

    PFIELD_NO_INITIAL = object()

    class Base1:
        _precord_fields = {'base_field1': MockPField()}

    class Base2:
        _precord_fields = {'base_field2': MockPField(mandatory=True)}

    dct = {'field1': MockPField(), 'field2': MockPField(initial='default')}
    bases = (Base1, Base2)

    set_fields(dct, bases, name='_precord_fields')

    assert '_precord_fields' in dct
    assert 'field1' in dct['_precord_fields']
    assert 'field2' in dct['_precord_fields']
    assert 'base_field1' in dct['_precord_fields']
    assert 'base_field2' in dct['_precord_fields']
    assert 'field1' not in dct
    assert 'field2' not in dct


# LLM-generated content at query #16
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
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()
        assert str(e) == 'Field invariant failed'

def test_persistent_raises_invariant_exception_when_missing_fields_present():
    class MockClass:
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []

    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ('MockClass.field1',)
        assert str(e) == 'Field invariant failed'

def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present():
    class MockClass:
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []

    evolver = _PRecordEvolver(MockClass, pmap({'field1': 'value1'}))
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = []
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert 'MockClass.field2' in e.missing_fields
        assert str(e) == 'Field invariant failed'


# LLM-generated content at query #17
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true():
    class MockField:
        type = (MockCheckedType,)
        factory = lambda self, value, ignore_extra=False: value

    class MockCheckedType:
        pass

    mock_field = MockField()
    result = is_field_ignore_extra_complaint(CheckedType, mock_field, True)
    assert result == True


# LLM-generated content at query #18
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
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (False, 'error_code')})()}
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
        assert 'error_code' in e.invariant_errors

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

def test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type():
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

def test_persistent_aggregates_multiple_invariant_errors():
    class MockClass:
        _precord_fields = {
            'field1': type('Field', (), {'invariant': lambda x: (False, 'error1')})(),
            'field2': type('Field', (), {'invariant': lambda x: (False, 'error2')})()
        }
        _precord_mandatory_fields = {'mandatory'}
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, {})
    evolver.set('field1', 'value1')
    evolver.set('field2', 'value2')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'error1' in e.invariant_errors
        assert 'error2' in e.invariant_errors
        assert 'mandatory' in e.missing_fields[0]


# LLM-generated content at query #19
#--------------------------

```python
def test_set_fields_creates_precord_fields_in_dct():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import _PField

    class Base1:
        _precord_fields = {'field1': _PField()}

    class Base2:
        _precord_fields = {'field2': _PField()}

    dct = {'field3': _PField()}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    result = '_precord_fields' in dct
    assert result

def test_set_fields_merges_fields_from_bases():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import _PField

    class Base1:
        _precord_fields = {'field1': _PField()}

    class Base2:
        _precord_fields = {'field2': _PField()}

    dct = {'field3': _PField()}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    result = set(dct['_precord_fields'].keys()) == {'field1', 'field2', 'field3'}
    assert result

def test_set_fields_moves_pfield_instances_from_dct():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import _PField

    field_instance = _PField()
    dct = {'my_field': field_instance}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    result = 'my_field' not in dct and dct['_precord_fields']['my_field'] is field_instance
    assert result

def test_store_invariants_creates_precord_invariants_in_dct():
    from pyrsistent._checked_types import store_invariants

    def invariant1(x):
        return True, ()

    class Base1:
        __invariant__ = invariant1

    dct = {}
    bases = (Base1,)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    result = '_precord_invariants' in dct
    assert result

def test_store_invariants_collects_invariants_from_bases():
    from pyrsistent._checked_types import store_invariants

    def invariant1(x):
        return True, ()

    def invariant2(x):
        return True, ()

    class Base1:
        __invariant__ = invariant1

    class Base2(Base1):
        __invariant__ = invariant2

    dct = {}
    bases = (Base2,)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    result = len(dct['_precord_invariants']) == 2
    assert result

def test_store_invariants_wraps_invariants():
    from pyrsistent._checked_types import store_invariants

    def invariant(x):
        return (True, ()), (False, 'error')

    class Base1:
        __invariant__ = invariant

    dct = {}
    bases = (Base1,)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    wrapped_invariant = dct['_precord_invariants'][0]
    result = wrapped_invariant(None) == (False, ('error',))
    assert result

def test_precord_meta_creates_mandatory_fields_set():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._checked_types import _PField

    class FieldWithMandatory(_PField):
        mandatory = True

    class FieldWithoutMandatory(_PField):
        mandatory = False

    dct = {
        'mandatory_field': FieldWithMandatory(),
        'optional_field': FieldWithoutMandatory()
    }
    bases = ()
    cls = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    result = cls._precord_mandatory_fields == {'mandatory_field'}
    assert result

def test_precord_meta_creates_initial_values_dict():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._checked_types import _PField, PFIELD_NO_INITIAL

    class FieldWithInitial(_PField):
        initial = 'default_value'

    class FieldWithoutInitial(_PField):
        initial = PFIELD_NO_INITIAL

    dct = {
        'field_with_default': FieldWithInitial(),
        'field_without_default': FieldWithoutInitial()
    }
    bases = ()
    cls = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    result = cls._precord_initial_values == {'field_with_default': 'default_value'}
    assert result

def test_precord_meta_sets_slots():
    from pyrsistent._precord import _PRecordMeta

    dct = {}
    bases = ()
    cls = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    result = cls.__slots__ == ()
    assert result


# LLM-generated content at query #20
#--------------------------

def test_serialize_with_custom_serializer():
    from pyrsistent import PRecord, field
    from pyrsistent import serialize as ser_func
    class TestRecord(PRecord):
        name = field()
        age = field(serializer=lambda fmt, x: f"Age: {x}")
    record = TestRecord(name="Alice", age=30)
    result = record.serialize()
    expected_name = ser_func(None, None, "Alice")
    expected_age = "Age: 30"
    assert result["name"] == expected_name
    assert result["age"] == expected_age

def test_serialize_without_custom_serializer():
    from pyrsistent import PRecord, field
    from pyrsistent import serialize as ser_func
    class TestRecord(PRecord):
        name = field()
        value = field()
    record = TestRecord(name="Bob", value=100)
    result = record.serialize()
    expected_name = ser_func(None, None, "Bob")
    expected_value = ser_func(None, None, 100)
    assert result["name"] == expected_name
    assert result["value"] == expected_value

def test_serialize_with_format_parameter():
    from pyrsistent import PRecord, field
    from pyrsistent import serialize as ser_func
    class TestRecord(PRecord):
        data = field(serializer=lambda fmt, x: f"{fmt}: {x}")
    record = TestRecord(data="test")
    result = record.serialize("json")
    expected = "json: test"
    assert result["data"] == expected

def test_serialize_empty_record():
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        pass
    record = TestRecord()
    result = record.serialize()
    assert result == {}

def test_serialize_multiple_fields_mixed_serializers():
    from pyrsistent import PRecord, field
    from pyrsistent import serialize as ser_func
    class TestRecord(PRecord):
        a = field()
        b = field(serializer=lambda fmt, x: x * 2)
        c = field()
    record = TestRecord(a=1, b=2, c=3)
    result = record.serialize()
    expected_a = ser_func(None, None, 1)
    expected_b = 4
    expected_c = ser_func(None, None, 3)
    assert result["a"] == expected_a
    assert result["b"] == expected_b
    assert result["c"] == expected_c


# LLM-generated content at query #21
#--------------------------

```python
def test_persistent_when_cls_has_mandatory_fields():
    class MockClass:
        _precord_mandatory_fields = {"field1", "field2"}
        _precord_invariants = []
        __name__ = "MockClass"

    evolver = _PRecordEvolver(MockClass, pmap({}))
    evolver._missing_fields = []
    evolver._invariant_error_codes = []
    result = pmap({"field1": "value1"})
    assert MockClass._precord_mandatory_fields


# LLM-generated content at query #22
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


# LLM-generated content at query #23
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

def test_precord_new_with_mandatory_fields_missing_raises_invariant_exception():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, mandatory=True), 'field2': field(type=str)}
        _precord_mandatory_fields = {'field1'}
    try:
        TestRecord(field2='test')
        assert False
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

def test_precord_new_with_field_invariant_failure_raises_invariant_exception():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, invariant=lambda x: (x > 0, 'ERR_POSITIVE'))}
    try:
        TestRecord(field1=-5)
        assert False
    except InvariantException as e:
        assert 'ERR_POSITIVE' in e.invariant_errors

def test_precord_new_with_global_invariant_failure_raises_invariant_exception():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=int)}
        _precord_invariants = [lambda r: (r['field1'] + r['field2'] > 0, 'ERR_SUM_POSITIVE')]
    try:
        TestRecord(field1=-10, field2=5)
        assert False
    except InvariantException as e:
        assert 'ERR_SUM_POSITIVE' in e.invariant_errors

def test_precord_new_with_valid_data_creates_record():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
    record = TestRecord(field1=42, field2='answer')
    assert record['field1'] == 42
    assert record['field2'] == 'answer'
    assert isinstance(record, TestRecord)


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

def test_precord_constructor_with_special_attributes():
    class TestRecord(PRecord):
        __slots__ = ()
    instance = TestRecord(_precord_size=0, _precord_buckets=[])
    assert isinstance(instance, TestRecord)
    assert len(instance) == 0

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=str), 'field2': field(type=int)}
        _precord_initial_values = {'field1': 'default', 'field2': 42}
    instance = TestRecord()
    assert instance['field1'] == 'default'
    assert instance['field2'] == 42

def test_precord_constructor_with_kwargs():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'name': field(type=str), 'value': field(type=int)}
    instance = TestRecord(name='test', value=10)
    assert instance['name'] == 'test'
    assert instance['value'] == 10

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'items': field(type=PVector)}
    instance = TestRecord(_factory_fields={'items': factory(PVector)}, items=[1, 2, 3])
    assert isinstance(instance['items'], PVector)
    assert list(instance['items']) == [1, 2, 3]

def test_precord_constructor_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'valid': field(type=str)}
    instance = TestRecord(_ignore_extra=True, valid='yes', extra='no')
    assert instance['valid'] == 'yes'
    assert 'extra' not in instance

def test_precord_constructor_with_callable_initial_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'counter': field(type=int)}
        _precord_initial_values = {'counter': lambda: 100}
    instance = TestRecord()
    assert instance['counter'] == 100

def test_precord_constructor_overrides_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'data': field(type=str)}
        _precord_initial_values = {'data': 'initial'}
    instance = TestRecord(data='overridden')
    assert instance['data'] == 'overridden'

def test_precord_constructor_empty_record():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
    instance = TestRecord()
    assert len(instance) == 0

def test_precord_constructor_with_nested_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'nested': field(type=PMap)}
        _precord_initial_values = {'nested': pmap({'key': 'value'})}
    instance = TestRecord()
    assert instance['nested'] == pmap({'key': 'value'})


# LLM-generated content at query #26
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
        _precord_initial_values = {'field1': 'default', 'field2': 42}
    record = TestRecord()
    assert record['field1'] == 'default'
    assert record['field2'] == 42

def test_precord_constructor_with_kwargs_overrides():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=str), 'field2': field(type=int)}
        _precord_initial_values = {'field1': 'default', 'field2': 42}
    record = TestRecord(field1='overridden')
    assert record['field1'] == 'overridden'
    assert record['field2'] == 42

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field': field(type=int, factory=lambda: 100)}
    record = TestRecord(_factory_fields=True)
    assert record['field'] == 100

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=str)}
    record = TestRecord(field1='value1', extra_field='extra', _ignore_extra=True)
    assert record['field1'] == 'value1'
    assert 'extra_field' not in record

def test_precord_constructor_without_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=str)}
    try:
        TestRecord(field1='value1', extra_field='extra', _ignore_extra=False)
        assert False
    except AttributeError:
        pass

def test_precord_constructor_with_callable_initial_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field': field(type=int)}
        _precord_initial_values = {'field': lambda: 999}
    record = TestRecord()
    assert record['field'] == 999

def test_precord_constructor_empty_record():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
    record = TestRecord()
    assert len(record) == 0

def test_precord_constructor_with_all_kwargs():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'a': field(type=str), 'b': field(type=int)}
    record = TestRecord(a='hello', b=123)
    assert record['a'] == 'hello'
    assert record['b'] == 123


# LLM-generated content at query #27
#--------------------------

```python
def test_set_fields_creates_precord_fields_in_dct():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import _PField

    class Base1:
        _precord_fields = {'field1': _PField(), 'field2': _PField()}

    class Base2:
        _precord_fields = {'field3': _PField()}

    dct = {'field4': _PField()}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    assert '_precord_fields' in dct
    assert 'field1' in dct['_precord_fields']
    assert 'field2' in dct['_precord_fields']
    assert 'field3' in dct['_precord_fields']
    assert 'field4' in dct['_precord_fields']
    assert 'field4' not in dct


# LLM-generated content at query #28
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

def test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type():
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

def test_persistent_creates_new_instance_if_pm_not_instance_of_cls():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    class OtherClass:
        pass
    original = OtherClass()
    original._buckets = None
    original._size = None
    original.keys = lambda: set()
    evolver = _PRecordEvolver(MockClass, original)
    result = evolver.persistent()
    assert isinstance(result, MockClass)


