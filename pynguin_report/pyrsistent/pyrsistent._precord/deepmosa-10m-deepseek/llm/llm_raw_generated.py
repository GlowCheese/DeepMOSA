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
        mandatory = mandatory_field(type=int)
    evolver = TestRecord().evolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'mandatory' in str(e)

def test_persistent_raises_invariant_exception_on_field_invariant_failure():
    class TestRecord(PRecord):
        value = field(type=int, invariant=lambda x: (x > 0, 'ERR'))
    evolver = TestRecord(value=1).evolver()
    evolver.set('value', -1)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'ERR' in str(e)

def test_persistent_raises_invariant_exception_on_global_invariant_failure():
    class TestRecord(PRecord):
        a = field(type=int)
        b = field(type=int)
        @invariant(lambda r: (r['a'] < r['b'], 'GLOBAL_ERR'))
        def check_order(self):
            pass
    evolver = TestRecord(a=5, b=3).evolver()
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'GLOBAL_ERR' in str(e)

def test_persistent_returns_unchanged_instance_when_not_dirty():
    original = TestRecord(field=10)
    evolver = original.evolver()
    result = evolver.persistent()
    assert result is original

def test_persistent_includes_all_set_values():
    class TestRecord(PRecord):
        x = field(type=int)
        y = field(type=int)
    evolver = TestRecord(x=1).evolver()
    evolver.set('y', 2)
    result = evolver.persistent()
    assert result['x'] == 1
    assert result['y'] == 2


# LLM-generated content at query #2
#--------------------------

def test___new___sets_fields():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial

    PFIELD_NO_INITIAL = object()
    class MockPField:
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial

    class Base1(metaclass=_PRecordMeta):
        _precord_fields = {}
        __invariant__ = lambda self: (True, ())
    class Base2(metaclass=_PRecordMeta):
        _precord_fields = {}
        __invariant__ = lambda self: (True, ())

    field_a = MockPField(mandatory=True, initial=PFIELD_NO_INITIAL)
    field_b = MockPField(mandatory=False, initial=10)
    dct = {'a': field_a, 'b': field_b, '__invariant__': lambda self: (False, ('error',))}
    bases = (Base1, Base2)
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_fields' in result.__dict__
    assert result._precord_fields['a'] is field_a
    assert result._precord_fields['b'] is field_b
    assert 'a' not in result.__dict__
    assert 'b' not in result.__dict__
    assert result._precord_mandatory_fields == {'a'}
    assert result._precord_initial_values == {'b': 10}
    assert len(result._precord_invariants) == 3
    assert result.__slots__ == ()

def test___new___inherits_fields():
    PFIELD_NO_INITIAL = object()
    class MockPField:
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial

    class Base(metaclass=_PRecordMeta):
        _precord_fields = {'base_field': MockPField(mandatory=True)}
        __invariant__ = lambda self: (True, ())

    field = MockPField()
    dct = {'field': field}
    bases = (Base,)
    result = _PRecordMeta.__new__(_PRecordMeta, 'ChildClass', bases, dct)
    assert 'base_field' in result._precord_fields
    assert 'field' in result._precord_fields
    assert result._precord_fields['base_field'].mandatory is True
    assert result._precord_fields['field'] is field

def test___new___merges_invariants():
    inv1 = lambda self: (True, ())
    inv2 = lambda self: (False, ('err',))
    class Base1(metaclass=_PRecordMeta):
        _precord_fields = {}
        __invariant__ = inv1
    class Base2(metaclass=_PRecordMeta):
        _precord_fields = {}
        __invariant__ = inv2

    dct = {'__invariant__': lambda self: (True, ())}
    bases = (Base1, Base2)
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert len(result._precord_invariants) == 3
    assert result._precord_invariants[0].__wrapped__ is inv1
    assert result._precord_invariants[1].__wrapped__ is inv2
    assert result._precord_invariants[2].__wrapped__ is dct['__invariant__']

def test___new___raises_on_non_callable_invariant():
    class Base(metaclass=_PRecordMeta):
        _precord_fields = {}
        __invariant__ = lambda self: (True, ())
    dct = {'__invariant__': 'not a callable'}
    bases = (Base,)
    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
        assert False
    except TypeError:
        pass

def test___new___handles_no_initial_values():
    PFIELD_NO_INITIAL = object()
    class MockPField:
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial

    field = MockPField()
    dct = {'field': field}
    bases = ()
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert result._precord_initial_values == {}

def test___new___handles_no_mandatory_fields():
    PFIELD_NO_INITIAL = object()
    class MockPField:
        def __init__(self, mandatory=False, initial=PFIELD_NO_INITIAL):
            self.mandatory = mandatory
            self.initial = initial

    field = MockPField(mandatory=False)
    dct = {'field': field}
    bases = ()
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert result._precord_mandatory_fields == set()

def test___new___wraps_invariants():
    def invariant_returning_tuple_of_tuples(self):
        return ((True, ()), (False, ('error1',)), (True, ()))
    dct = {'__invariant__': invariant_returning_tuple_of_tuples}
    bases = ()
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    wrapped_invariant = result._precord_invariants[0]
    verdict, data = wrapped_invariant(None)
    assert verdict is False
    assert data == ('error1',)


# LLM-generated content at query #3
#--------------------------

def test_precord_constructor_with_special_attributes():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
    record = TestRecord(_precord_size=0, _precord_buckets=[])
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None, 'field2': None}
        _precord_initial_values = {'field1': lambda: 'default1', 'field2': 'default2'}
    record = TestRecord()
    assert record['field1'] == 'default1'
    assert record['field2'] == 'default2'

def test_precord_constructor_with_kwargs_overrides():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None, 'field2': None}
        _precord_initial_values = {'field1': lambda: 'default1', 'field2': 'default2'}
    record = TestRecord(field1='custom1')
    assert record['field1'] == 'custom1'
    assert record['field2'] == 'default2'

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None}
    record = TestRecord(_factory_fields={'field1': 'factory_value'}, field1='provided_value')
    assert record['field1'] == 'provided_value'

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None}
    record = TestRecord(_ignore_extra=True, field1='value1', extra_field='extra')
    assert 'field1' in record
    assert record['field1'] == 'value1'
    assert 'extra_field' not in record

def test_precord_constructor_without_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None}
    try:
        TestRecord(field1='value1', extra_field='extra')
        assert False
    except AttributeError:
        pass

def test_precord_constructor_empty_record():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
    record = TestRecord()
    assert len(record) == 0

def test_precord_constructor_all_fields_provided():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'a': None, 'b': None, 'c': None}
    record = TestRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3

def test_precord_constructor_with_callable_initial_value():
    callable_counter = 0
    def counter_callable():
        nonlocal callable_counter
        callable_counter += 1
        return callable_counter
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field': None}
        _precord_initial_values = {'field': counter_callable}
    record1 = TestRecord()
    record2 = TestRecord()
    assert record1['field'] == 1
    assert record2['field'] == 2

def test_precord_constructor_with_non_callable_initial_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field': None}
        _precord_initial_values = {'field': 'static_default'}
    record = TestRecord()
    assert record['field'] == 'static_default'


# LLM-generated content at query #4
#--------------------------

```python
def test_set_fields_creates_precord_fields_in_dct():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import _PField

    class MockField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial

    class Base1:
        pass

    class Base2:
        pass

    Base1._precord_fields = {'field1': MockField()}
    Base2._precord_fields = {'field2': MockField()}

    dct = {}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    result = '_precord_fields' in dct
    assert result == True

def test_set_fields_moves_pfield_instances_from_dct_to_precord_fields():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import _PField

    class MockField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial

    class Base1:
        pass

    class Base2:
        pass

    Base1._precord_fields = {'field1': MockField()}
    Base2._precord_fields = {'field2': MockField()}

    field3 = MockField()
    dct = {'field3': field3}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    result = dct['_precord_fields']['field3'] is field3
    assert result == True

def test_set_fields_removes_pfield_instances_from_original_dct():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import _PField

    class MockField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial

    class Base1:
        pass

    class Base2:
        pass

    Base1._precord_fields = {'field1': MockField()}
    Base2._precord_fields = {'field2': MockField()}

    field3 = MockField()
    dct = {'field3': field3}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    result = 'field3' not in dct
    assert result == True

def test_store_invariants_creates_precord_invariants_in_dct():
    from pyrsistent._checked_types import store_invariants

    def invariant1(instance):
        return True, ()

    def invariant2(instance):
        return True, ()

    class Base1:
        __invariant__ = invariant1

    class Base2:
        __invariant__ = invariant2

    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    result = '_precord_invariants' in dct
    assert result == True

def test_store_invariants_wraps_invariants():
    from pyrsistent._checked_types import store_invariants

    def invariant1(instance):
        return True, ()

    def invariant2(instance):
        return (False, "error1"), (False, "error2")

    class Base1:
        __invariant__ = invariant1

    class Base2:
        __invariant__ = invariant2

    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    result = len(invariants) == 2
    assert result == True

def test_precord_meta_creates_all_required_attributes():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._checked_types import _PField

    class MockField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial

    class Base1:
        _precord_fields = {'field1': MockField(mandatory=True, initial='default1')}

    class Base2:
        _precord_fields = {'field2': MockField(mandatory=False, initial='default2')}

    dct = {'field3': MockField(mandatory=True, initial=None)}
    bases = (Base1, Base2)
    cls = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    result = hasattr(cls, '_precord_fields')
    assert result == True

def test_precord_meta_creates_mandatory_fields_set():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._checked_types import _PField

    class MockField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial

    class Base1:
        _precord_fields = {'field1': MockField(mandatory=True, initial='default1')}

    class Base2:
        _precord_fields = {'field2': MockField(mandatory=False, initial='default2')}

    dct = {'field3': MockField(mandatory=True, initial=None)}
    bases = (Base1, Base2)
    cls = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    result = 'field1' in cls._precord_mandatory_fields
    assert result == True

def test_precord_meta_creates_initial_values_dict():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._checked_types import _PField

    class MockField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial

    class Base1:
        _precord_fields = {'field1': MockField(mandatory=True, initial='default1')}

    class Base2:
        _precord_fields = {'field2': MockField(mandatory=False, initial='default2')}

    dct = {'field3': MockField(mandatory=True, initial=None)}
    bases = (Base1, Base2)
    cls = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    result = 'field1' in cls._precord_initial_values
    assert result == True

def test_precord_meta_sets_empty_slots():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._checked_types import _PField

    class MockField(_PField):
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial

    class Base1:
        _precord_fields = {'field1': MockField(mandatory=True, initial='default1')}

    dct = {}
    bases = (Base1,)
    cls = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    result = cls.__slots__ == ()
    assert result == True

def test_precord_meta_inherits_invariants_from_bases():
    from pyrsistent._precord import _PRecordMeta

    def invariant1(instance):
        return True, ()

    def invariant2(instance):
        return True, ()

    class Base1:
        __invariant__ = invariant1

    class Base2:
        __invariant__ = invariant2

    dct = {}
    bases = (Base1, Base2)
    cls = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    result = len(cls._precord_invariants) == 2
    assert result == True


# LLM-generated content at query #5
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
    evolver._invariant_error_codes = ['error']
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ('error',)

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
        assert e.invariant_errors == ('global_error',)

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
    evolver._original_pmap = type('MockPMap', (), {'_buckets': 'buckets', '_size': 'size'})()
    evolver.is_dirty = lambda: True
    result = evolver.persistent()
    assert result._buckets == 'buckets' and result._size == 'size'


# LLM-generated content at query #6
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

def test_precord_new_with_overridden_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
        _precord_initial_values = {'field1': 10, 'field2': 'default'}
    record = TestRecord(field1=20)
    assert record['field1'] == 20
    assert record['field2'] == 'default'

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, factory=lambda x: x * 2)}
    record = TestRecord(field1=5, _factory_fields=None)
    assert record['field1'] == 10

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
    record = TestRecord(field1=5, extra_field=10, _ignore_extra=True)
    assert record['field1'] == 5
    assert 'extra_field' not in record

def test_precord_new_without_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
    try:
        TestRecord(field1=5, extra_field=10, _ignore_extra=False)
        assert False
    except AttributeError:
        pass

def test_precord_new_with_mandatory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, mandatory=True), 'field2': field(type=str)}
        _precord_mandatory_fields = {'field1'}
    try:
        TestRecord(field2='test')
        assert False
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

def test_precord_new_with_invariant_failure():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int, invariant=lambda x: (x > 0, 'ERR1'))}
    try:
        TestRecord(field1=-5)
        assert False
    except InvariantException as e:
        assert 'ERR1' in e.invariant_errors

def test_precord_new_with_global_invariant():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=int)}
        _precord_invariants = [lambda r: (r['field1'] + r['field2'] == 10, 'ERR_SUM')]
    try:
        TestRecord(field1=3, field2=8)
        assert False
    except InvariantException as e:
        assert 'ERR_SUM' in e.invariant_errors

def test_precord_new_successful_creation():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int), 'field2': field(type=str)}
    record = TestRecord(field1=42, field2='hello')
    assert record['field1'] == 42
    assert record['field2'] == 'hello'

def test_precord_new_with_callable_initial_value():
    class TestRecord(PRecord):
        _precord_fields = {'field1': field(type=int)}
        _precord_initial_values = {'field1': lambda: 100}
    record = TestRecord()
    assert record['field1'] == 100

def test_precord_new_with_factory_and_ignore_extra():
    class Inner(CheckedType):
        pass
    class TestRecord(PRecord):
        _precord_fields = {'inner': field(type=Inner, factory=lambda x, ignore_extra: Inner.create(x, ignore_extra=ignore_extra))}
    record = TestRecord(inner={'key': 'value'}, _ignore_extra=True)
    assert isinstance(record['inner'], Inner)


# LLM-generated content at query #7
#--------------------------

def test___new___sets_fields_correctly():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    class MockPField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class Base:
        _precord_fields = {'base_field': MockPField(False, PFIELD_NO_INITIAL)}
    dct = {'field1': MockPField(True, 'default1'), 'field2': MockPField(False, PFIELD_NO_INITIAL)}
    bases = (Base,)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert dct['_precord_fields'] == {'base_field': Base._precord_fields['base_field'], 'field1': dct['field1'], 'field2': dct['field2']}
    assert 'field1' not in dct
    assert 'field2' not in dct

def test___new___sets_mandatory_fields():
    class MockPField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    dct = {'field1': MockPField(True, PFIELD_NO_INITIAL), 'field2': MockPField(False, PFIELD_NO_INITIAL)}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert dct['_precord_mandatory_fields'] == {'field1'}

def test___new___sets_initial_values():
    class MockPField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    dct = {'field1': MockPField(False, 'init1'), 'field2': MockPField(False, PFIELD_NO_INITIAL)}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert dct['_precord_initial_values'] == {'field1': 'init1'}

def test___new___stores_invariants():
    def invariant1(instance):
        return True, ()
    def invariant2(instance):
        return False, ('error',)
    class Base:
        __invariant__ = invariant1
    dct = {'__invariant__': invariant2}
    bases = (Base,)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    assert invariants[0](None) == (True, ())
    assert invariants[1](None) == (False, ('error',))

def test___new___raises_on_non_callable_invariant():
    class Base:
        __invariant__ = 'not callable'
    dct = {}
    bases = (Base,)
    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
        assert False
    except TypeError:
        pass

def test___new___sets_slots():
    dct = {}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert dct['__slots__'] == ()

def test___new___inherits_fields_from_multiple_bases():
    class MockPField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class Base1:
        _precord_fields = {'field1': MockPField(False, PFIELD_NO_INITIAL)}
    class Base2:
        _precord_fields = {'field2': MockPField(False, PFIELD_NO_INITIAL)}
    dct = {}
    bases = (Base1, Base2)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert 'field1' in dct['_precord_fields']
    assert 'field2' in dct['_precord_fields']

def test___new___merges_invariant_results():
    def multi_invariant(instance):
        return [(True, ()), (False, ('err1',)), (False, ('err2',))]
    dct = {'__invariant__': multi_invariant}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    invariants = dct['_precord_invariants']
    result = invariants[0](None)
    assert result == (False, ('err1', 'err2'))


# LLM-generated content at query #8
#--------------------------

def test_set_with_valid_field_and_value():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        type = (int,)
        factory = lambda x: x
        invariant = lambda self, v: (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    evolver.set('key', 5)
    assert evolver._original_pmap['key'] == 5

def test_set_with_factory_and_ignore_extra():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    def factory(value, ignore_extra=False):
        return value
    class MockField:
        type = (int,)
        factory = factory
        invariant = lambda self, v: (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    evolver = _PRecordEvolver(MockDestinationCls, {}, _ignore_extra=True)
    evolver.set('key', 5)
    assert evolver._original_pmap['key'] == 5

def test_set_with_factory_invariant_exception():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    def factory(value):
        raise InvariantException((), (), 'error')
    class MockField:
        type = (int,)
        factory = factory
        invariant = lambda self, v: (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    evolver.set('key', 5)
    assert len(evolver._invariant_error_codes) > 0

def test_set_with_invalid_type():
    from pyrsistent import InvariantException, CheckedType, PTypeError
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        type = (int,)
        factory = lambda x: x
        invariant = lambda self, v: (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    try:
        evolver.set('key', 'string')
        assert False
    except PTypeError:
        assert True

def test_set_with_failed_invariant():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        type = (int,)
        factory = lambda x: x
        invariant = lambda self, v: (False, 'error')
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    evolver.set('key', 5)
    assert evolver._invariant_error_codes == ['error']

def test_set_with_non_existent_field():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockDestinationCls:
        _precord_fields = {}
    evolver = _PRecordEvolver(MockDestinationCls, {})
    try:
        evolver.set('key', 5)
        assert False
    except AttributeError:
        assert True

def test_set_with_factory_fields_skipped():
    from pyrsistent import InvariantException, CheckedType
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._field_common import check_type, is_field_ignore_extra_complaint
    class MockField:
        type = (int,)
        factory = lambda x: x * 2
        invariant = lambda self, v: (True, None)
    class MockDestinationCls:
        _precord_fields = {'key': MockField()}
    evolver = _PRecordEvolver(MockDestinationCls, {}, _factory_fields=set())
    evolver.set('key', 5)
    assert evolver._original_pmap['key'] == 5


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_invariant_error_codes_present():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"

    evolver = _PRecordEvolver(MockClass, pmap())
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

    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ("MockClass.field1",)
        assert str(e) == "Field invariant failed"

def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present():
    class MockClass:
        _precord_mandatory_fields = {"field1", "field2"}
        _precord_invariants = []
        __name__ = "MockClass"

    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = ["error1", "error2"]
    evolver._missing_fields = ["MockClass.field3"]
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert set(e.invariant_errors) == {"error1", "error2"}
        assert "MockClass.field1" in e.missing_fields or "MockClass.field2" in e.missing_fields
        assert "MockClass.field3" in e.missing_fields
        assert str(e) == "Field invariant failed"

def test_persistent_does_not_raise_when_no_errors_and_no_mandatory_fields():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"

    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    result = evolver.persistent()
    assert isinstance(result, MockClass)

def test_persistent_does_not_raise_when_no_errors_and_all_mandatory_fields_present():
    class MockClass:
        _precord_mandatory_fields = {"field1"}
        _precord_invariants = []
        __name__ = "MockClass"

    evolver = _PRecordEvolver(MockClass, pmap({"field1": "value1"}))
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    result = evolver.persistent()
    assert isinstance(result, MockClass)
    assert result["field1"] == "value1"


# LLM-generated content at query #11
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
    r = MyRecord(_factory_fields={'x': lambda: 5}, x=1, y=2)
    assert r['x'] == 1
    assert r['y'] == 2

def test_precord_constructor_ignore_extra():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field()}
        _precord_initial_values = {}
    r = MyRecord(_ignore_extra=True, x=1, y=2)
    assert r['x'] == 1
    assert 'y' not in r

def test_precord_constructor_without_ignore_extra():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field()}
        _precord_initial_values = {}
    try:
        MyRecord(x=1, y=2)
        assert False
    except Exception:
        assert True


# LLM-generated content at query #12
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
    instance = TestRecord(field1=200, field2='override')
    assert instance['field1'] == 200
    assert instance['field2'] == 'override'

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

def test_precord_new_validates_field_invariants():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, invariant=lambda x: (x > 0, 'ERR_POSITIVE'))}
    try:
        TestRecord(field1=-1)
        assert False
    except InvariantException as e:
        assert 'ERR_POSITIVE' in e.invariant_errors

def test_precord_new_checks_mandatory_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int, mandatory=True), 'field2': field(type=str)}
        _precord_mandatory_fields = {'field1'}
    try:
        TestRecord(field2='test')
        assert False
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

def test_precord_new_checks_global_invariants():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int), 'field2': field(type=int)}
        _precord_invariants = [lambda r: (r['field1'] + r['field2'] == 10, 'ERR_SUM')]
    try:
        TestRecord(field1=3, field2=4)
        assert False
    except InvariantException as e:
        assert 'ERR_SUM' in e.invariant_errors

def test_precord_new_returns_same_instance_if_already_correct_type():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(type=int)}
    original = TestRecord(field1=5)
    same = TestRecord(_precord_size=original._size, _precord_buckets=original._buckets)
    assert same is original


# LLM-generated content at query #13
#--------------------------

def test___new___creates_precord_with_special_attributes():
    class TestRecord(PRecord):
        pass
    buckets = pvector().extend([None] * 8)
    record = TestRecord(_precord_size=0, _precord_buckets=buckets)
    assert isinstance(record, TestRecord)
    assert record._size == 0
    assert record._buckets == buckets

def test___new___creates_precord_via_evolver_with_initial_values():
    class TestRecord(PRecord):
        a = field()
        b = field()
    record = TestRecord(a=1, b=2)
    assert isinstance(record, TestRecord)
    assert record['a'] == 1
    assert record['b'] == 2

def test___new___applies_precord_initial_values():
    class TestRecord(PRecord):
        a = field(initial=10)
        b = field(initial=lambda: 20)
    record = TestRecord()
    assert record['a'] == 10
    assert record['b'] == 20

def test___new___overrides_precord_initial_values_with_kwargs():
    class TestRecord(PRecord):
        a = field(initial=10)
        b = field(initial=20)
    record = TestRecord(a=30)
    assert record['a'] == 30
    assert record['b'] == 20

def test___new___raises_attribute_error_for_unknown_field():
    class TestRecord(PRecord):
        a = field()
    try:
        TestRecord(b=2)
        assert False
    except AttributeError as e:
        assert "'b' is not among the specified fields for TestRecord" in str(e)

def test___new___handles_factory_fields_parameter():
    class TestRecord(PRecord):
        a = field(type=int)
    record = TestRecord(_factory_fields={TestRecord.a}, a='5')
    assert record['a'] == 5

def test___new___handles_ignore_extra_parameter():
    class TestRecord(PRecord):
        a = field()
    record = TestRecord(_ignore_extra=True, a=1, b=2)
    assert record['a'] == 1
    assert 'b' not in record

def test___new___raises_invariant_exception_on_invariant_failure():
    class TestRecord(PRecord):
        a = field(invariant=lambda x: (x > 0, 'a must be positive'))
    try:
        TestRecord(a=-1)
        assert False
    except InvariantException as e:
        assert 'a must be positive' in e.invariant_errors[0]

def test___new___raises_invariant_exception_on_missing_mandatory_fields():
    class TestRecord(PRecord):
        a = field(mandatory=True)
    try:
        TestRecord()
        assert False
    except InvariantException as e:
        assert 'TestRecord.a' in e.missing_fields[0]

def test___new___checks_global_invariants():
    class TestRecord(PRecord):
        a = field()
        b = field()
        @invariant(lambda r: (r['a'] <= r['b'], 'a must be <= b'))
        def check_order(self):
            pass
    try:
        TestRecord(a=10, b=5)
        assert False
    except InvariantException as e:
        assert 'a must be <= b' in e.invariant_errors[0]


# LLM-generated content at query #14
#--------------------------

```python
def test_persistent_when_cls_has_mandatory_fields():
    class MockClass:
        _precord_mandatory_fields = {"field1", "field2"}
        _precord_invariants = []
        __name__ = "MockClass"

    evolver = _PRecordEvolver(MockClass, {})
    evolver._missing_fields = []
    evolver._invariant_error_codes = []
    evolver._destination_cls = MockClass
    evolver.is_dirty = lambda: False
    evolver.persistent = lambda: type('pm', (), {'_buckets': None, '_size': 0, 'keys': lambda: set()})()
    result = evolver.persistent()
    assert MockClass._precord_mandatory_fields


# LLM-generated content at query #15
#--------------------------

```python
def test_persistent_when_not_dirty_and_pm_is_instance_of_cls():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import pmap
    class MockClass:
        _precord_mandatory_fields = None
        _precord_invariants = []
        def __init__(self, **kwargs):
            pass
    original_pmap = pmap()
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver._destination_cls = MockClass
    evolver._original_pmap = original_pmap
    evolver._buckets = original_pmap._buckets
    evolver._size = original_pmap._size
    result = evolver.persistent()
    assert result is original_pmap


# LLM-generated content at query #16
#--------------------------

```python
def test_persistent_creates_new_instance_when_not_dirty_but_pm_is_not_instance_of_cls():
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent import PRecord, field
    class TestRecord(PRecord):
        x = field()
    original = TestRecord(x=1)
    evolver = _PRecordEvolver(TestRecord, original._map)
    evolver._destination_cls = TestRecord
    evolver._original_pmap = original._map
    evolver._buckets = original._map._buckets
    evolver._size = original._map._size
    evolver.is_dirty = lambda: False
    pm = object()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result._map._buckets is original._map._buckets
    assert result._map._size == original._map._size


# LLM-generated content at query #17
#--------------------------

def test_precord_constructor_without_special_attributes():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {}
    instance = MyRecord(field1='value1', field2='value2')
    assert instance['field1'] == 'value1'
    assert instance['field2'] == 'value2'

def test_precord_constructor_with_special_attributes():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
        _precord_initial_values = {}
    size = 0
    buckets = []
    instance = MyRecord(_precord_size=size, _precord_buckets=buckets)
    assert isinstance(instance, MyRecord)

def test_precord_constructor_with_initial_values():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {'field1': lambda: 'default1', 'field2': 'default2'}
    instance = MyRecord()
    assert instance['field1'] == 'default1'
    assert instance['field2'] == 'default2'

def test_precord_constructor_with_initial_values_and_kwargs():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {'field1': lambda: 'default1', 'field2': 'default2'}
    instance = MyRecord(field1='new_value1')
    assert instance['field1'] == 'new_value1'
    assert instance['field2'] == 'default2'

def test_precord_constructor_with_factory_fields():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {}
    instance = MyRecord(_factory_fields={'field1': 'factory_value'}, field2='value2')
    assert instance['field1'] == 'factory_value'
    assert instance['field2'] == 'value2'

def test_precord_constructor_with_ignore_extra():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field()}
        _precord_initial_values = {}
    instance = MyRecord(_ignore_extra=True, field1='value1', extra_field='extra')
    assert instance['field1'] == 'value1'
    assert 'extra_field' not in instance

def test_precord_constructor_without_ignore_extra():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field()}
        _precord_initial_values = {}
    try:
        MyRecord(field1='value1', extra_field='extra')
        assert False
    except Exception:
        assert True


# LLM-generated content at query #18
#--------------------------

def test___new___sets_fields_correctly():
    class MockPField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class Base1:
        _precord_fields = {'base1_field': MockPField()}
    class Base2:
        _precord_fields = {'base2_field': MockPField()}
    dct = {'custom_field': MockPField()}
    bases = (Base1, Base2)
    set_fields(dct, bases, '_precord_fields')
    assert '_precord_fields' in dct
    assert 'base1_field' in dct['_precord_fields']
    assert 'base2_field' in dct['_precord_fields']
    assert 'custom_field' in dct['_precord_fields']
    assert 'custom_field' not in dct

def test___new___handles_mandatory_fields():
    class MockPField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    dct = {'mandatory_field': MockPField(mandatory=True), 'optional_field': MockPField(mandatory=False)}
    bases = ()
    set_fields(dct, bases, '_precord_fields')
    mandatory_fields = set(name for name, field in dct['_precord_fields'].items() if field.mandatory)
    assert mandatory_fields == {'mandatory_field'}

def test___new___handles_initial_values():
    class MockPField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    dct = {'with_initial': MockPField(initial=42), 'without_initial': MockPField(initial=PFIELD_NO_INITIAL)}
    bases = ()
    set_fields(dct, bases, '_precord_fields')
    initial_values = dict((k, field.initial) for k, field in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL)
    assert initial_values == {'with_initial': 42}

def test___new___stores_invariants_from_bases():
    def invariant1(instance):
        return True, ()
    def invariant2(instance):
        return False, ('error',)
    class Base1:
        __invariant__ = invariant1
    class Base2:
        __invariant__ = invariant2
    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert '_precord_invariants' in dct
    assert len(dct['_precord_invariants']) == 2
    assert dct['_precord_invariants'][0](None) == (True, ())
    assert dct['_precord_invariants'][1](None) == (False, ('error',))

def test___new___wraps_invariants_correctly():
    def multi_invariant(instance):
        return [(True, ()), (False, ('err1',)), (False, ('err2',))]
    dct = {'__invariant__': multi_invariant}
    bases = ()
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    wrapped = dct['_precord_invariants'][0]
    result = wrapped(None)
    assert result == (False, ('err1', 'err2'))

def test___new___raises_on_non_callable_invariant():
    class Base1:
        __invariant__ = 'not callable'
    dct = {}
    bases = (Base1,)
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
    dct = {'field1': MockPField()}
    bases = ()
    set_fields(dct, bases, '_precord_fields')
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
    class Parent(GrandBase):
        _precord_fields = {'parent_field': MockPField()}
    dct = {'child_field': MockPField()}
    bases = (Parent,)
    set_fields(dct, bases, '_precord_fields')
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    assert 'grand_field' in dct['_precord_fields']
    assert 'parent_field' in dct['_precord_fields']
    assert 'child_field' in dct['_precord_fields']
    assert len(dct['_precord_invariants']) == 1
    assert dct['_precord_invariants'][0](None) == (True, ())


# LLM-generated content at query #19
#--------------------------

```python
def test_new_without_precord_size_and_buckets():
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_initial_values = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert len(result) == 0


# LLM-generated content at query #20
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

def test_precord_repr_with_empty_fields():
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


# LLM-generated content at query #21
#--------------------------

def test_precord_new_with_special_attributes():
    class TestRecord(PRecord):
        pass
    record = TestRecord(_precord_size=0, _precord_buckets=pvector().extend([]))
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        a = field()
        b = field()
    record = TestRecord(a=1, b=2)
    assert record['a'] == 1
    assert record['b'] == 2

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        a = field(factory=int)
        b = field()
    record = TestRecord(a='5', b=2, _factory_fields={TestRecord.a})
    assert record['a'] == 5
    assert record['b'] == 2

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        a = field()
    record = TestRecord(a=1, b=2, _ignore_extra=True)
    assert record['a'] == 1
    assert 'b' not in record

def test_precord_new_without_ignore_extra_raises():
    class TestRecord(PRecord):
        a = field()
    try:
        TestRecord(a=1, b=2)
        assert False
    except AttributeError:
        pass

def test_precord_new_with_initial_values_from_class():
    class TestRecord(PRecord):
        a = field(initial=10)
        b = field(initial=lambda: 20)
    record = TestRecord()
    assert record['a'] == 10
    assert record['b'] == 20

def test_precord_new_overrides_initial_values():
    class TestRecord(PRecord):
        a = field(initial=10)
        b = field(initial=20)
    record = TestRecord(a=30)
    assert record['a'] == 30
    assert record['b'] == 20

def test_precord_new_with_invariant_failure():
    class TestRecord(PRecord):
        a = field(invariant=lambda x: (x > 0, 'a must be positive'))
    try:
        TestRecord(a=-1)
        assert False
    except InvariantException as e:
        assert 'a must be positive' in str(e)

def test_precord_new_with_missing_mandatory_field():
    class TestRecord(PRecord):
        a = field(mandatory=True)
    try:
        TestRecord()
        assert False
    except InvariantException as e:
        assert 'TestRecord.a' in str(e)

def test_precord_new_with_factory_and_invariant():
    class TestRecord(PRecord):
        a = field(factory=int, invariant=lambda x: (x > 0, 'a must be positive'))
    try:
        TestRecord(a='-5')
        assert False
    except InvariantException as e:
        assert 'a must be positive' in str(e)

def test_precord_new_with_factory_ignore_extra():
    class Inner(PRecord):
        x = field()
    class TestRecord(PRecord):
        a = field(factory=Inner.create)
    record = TestRecord(a={'x': 1, 'y': 2}, _ignore_extra=True)
    assert record['a']['x'] == 1
    assert 'y' not in record['a']


# LLM-generated content at query #22
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_error_codes_present():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"
    
    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = ["error1"]
    evolver._missing_fields = []
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ("error1",)
        assert e.missing_fields == ()
        assert str(e) == "Field invariant failed"
    else:
        assert False, "Expected InvariantException"

def test_persistent_raises_invariant_exception_when_missing_fields_present():
    class MockClass:
        _precord_mandatory_fields = {"field1"}
        _precord_invariants = []
        __name__ = "MockClass"
    
    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ()
        assert e.missing_fields == ("MockClass.field1",)
        assert str(e) == "Field invariant failed"
    else:
        assert False, "Expected InvariantException"

def test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present():
    class MockClass:
        _precord_mandatory_fields = {"field1", "field2"}
        _precord_invariants = []
        __name__ = "MockClass"
    
    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = ["error1", "error2"]
    evolver._missing_fields = ["MockClass.field3"]
    try:
        evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ("error1", "error2")
        assert set(e.missing_fields) == {"MockClass.field1", "MockClass.field2", "MockClass.field3"}
        assert str(e) == "Field invariant failed"
    else:
        assert False, "Expected InvariantException"

def test_persistent_does_not_raise_when_no_errors_or_missing_fields():
    class MockClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        __name__ = "MockClass"
    
    evolver = _PRecordEvolver(MockClass, pmap())
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    result = evolver.persistent()
    assert isinstance(result, MockClass)


# LLM-generated content at query #23
#--------------------------

def test_precord_initial_values_condition_true():
    class TestRecord(PRecord):
        _precord_fields = {'a': field()}
        _precord_initial_values = {'a': lambda: 10}
    instance = TestRecord()
    assert instance['a'] == 10


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
        _precord_initial_values = {}
    record = TestRecord(x=10)
    assert record['x'] == 10

def test___new___uses_initial_values_from_class():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': lambda: 5}
    record = TestRecord()
    assert record['x'] == 5

def test___new___overrides_initial_values_with_kwargs():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
        _precord_initial_values = {'x': lambda: 5}
    record = TestRecord(x=10)
    assert record['x'] == 10

def test___new___raises_attribute_error_for_unknown_field():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
        _precord_initial_values = {}
    try:
        TestRecord(y=10)
        assert False
    except AttributeError as e:
        assert "'y' is not among the specified fields for TestRecord" in str(e)

def test___new___handles_factory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int, factory=lambda v: v * 2)}
        _precord_initial_values = {}
    record = TestRecord(x=5, _factory_fields={field(type=int, factory=lambda v: v * 2)})
    assert record['x'] == 10

def test___new___handles_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'x': field()}
        _precord_initial_values = {}
    record = TestRecord(x=10, y=20, _ignore_extra=True)
    assert record['x'] == 10
    assert 'y' not in record

def test___new___propagates_invariant_exception():
    def invariant(v):
        return v > 0, 'INVARIANT'
    class TestRecord(PRecord):
        _precord_fields = {'x': field(invariant=invariant)}
        _precord_initial_values = {}
    try:
        TestRecord(x=0)
        assert False
    except InvariantException as e:
        assert 'INVARIANT' in e.invariant_errors

def test___new___checks_mandatory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(mandatory=True), 'y': field()}
        _precord_initial_values = {}
        _precord_mandatory_fields = {'x'}
    try:
        TestRecord(y=10)
        assert False
    except InvariantException as e:
        assert 'TestRecord.x' in e.missing_fields

def test___new___creates_record_without_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_initial_values = {}
    record = TestRecord()
    assert len(record) == 0


# LLM-generated content at query #2
#--------------------------

def test__PRecordMeta__new__sets_fields():
    class MockField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    class MockPField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPFieldNoInitial(MockPField):
        def __init__(self, mandatory):
            super().__init__(mandatory, PFIELD_NO_INITIAL)
    class Base:
        _precord_fields = {'base_field': MockPField(True, 10)}
    dct = {'field1': MockPField(False, 5), 'field2': MockPFieldNoInitial(True)}
    bases = (Base,)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_fields' in dct
    assert dct['_precord_fields']['base_field'].initial == 10
    assert dct['_precord_fields']['field1'].initial == 5
    assert dct['_precord_fields']['field2'].initial is PFIELD_NO_INITIAL
    assert 'field1' not in dct
    assert 'field2' not in dct

def test__PRecordMeta__new__stores_invariants():
    def invariant1(instance):
        return True, ()
    def invariant2(instance):
        return False, ('error',)
    class Base:
        __invariant__ = invariant1
    dct = {'__invariant__': invariant2}
    bases = (Base,)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_invariants' in dct
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    assert invariants[0](None) == (True, ())
    assert invariants[1](None) == (False, ('error',))

def test__PRecordMeta__new__sets_mandatory_fields():
    class MockPField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    dct = {'field1': MockPField(True, 1), 'field2': MockPField(False, 2)}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_mandatory_fields' in dct
    assert dct['_precord_mandatory_fields'] == {'field1'}

def test__PRecordMeta__new__sets_initial_values():
    class MockPField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    PFIELD_NO_INITIAL = object()
    class MockPFieldNoInitial(MockPField):
        def __init__(self, mandatory):
            super().__init__(mandatory, PFIELD_NO_INITIAL)
    dct = {'field1': MockPField(True, 100), 'field2': MockPFieldNoInitial(False)}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_initial_values' in dct
    assert dct['_precord_initial_values'] == {'field1': 100}

def test__PRecordMeta__new__sets_slots():
    dct = {}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '__slots__' in dct
    assert dct['__slots__'] == ()

def test__PRecordMeta__new__inherits_fields():
    class MockPField:
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial
    class Base:
        _precord_fields = {'inherited': MockPField(True, 99)}
    dct = {'own': MockPField(False, 50)}
    bases = (Base,)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert 'inherited' in dct['_precord_fields']
    assert 'own' in dct['_precord_fields']
    assert dct['_precord_fields']['inherited'].initial == 99
    assert dct['_precord_fields']['own'].initial == 50

def test__PRecordMeta__new__raises_on_non_callable_invariant():
    class Base:
        __invariant__ = 'not callable'
    dct = {}
    bases = (Base,)
    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
        assert False
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test__PRecordMeta__new__wraps_invariants():
    def multi_invariant(instance):
        return [(True, ()), (False, ('err1',)), (False, ('err2',))]
    dct = {'__invariant__': multi_invariant}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    invariants = dct['_precord_invariants']
    result = invariants[0](None)
    assert result == (False, ('err1', 'err2'))

def test__PRecordMeta__new__handles_empty():
    dct = {}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'EmptyClass', bases, dct)
    assert '_precord_fields' in dct
    assert dct['_precord_fields'] == {}
    assert '_precord_invariants' in dct
    assert dct['_precord_invariants'] == ()
    assert '_precord_mandatory_fields' in dct
    assert dct['_precord_mandatory_fields'] == set()
    assert '_precord_initial_values' in dct
    assert dct['_precord_initial_values'] == {}
    assert '__slots__' in dct
    assert dct['__slots__'] == ()


# LLM-generated content at query #3
#--------------------------

def test_precord_constructor_with_no_arguments():
    class TestRecord(PRecord):
        pass
    record = TestRecord()
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test_precord_constructor_with_field_assignments():
    class TestRecord(PRecord):
        x = field()
        y = field()
    record = TestRecord(x=1, y=2)
    assert record['x'] == 1
    assert record['y'] == 2

def test_precord_constructor_ignores_extra_fields_when_configured():
    class TestRecord(PRecord):
        x = field()
    record = TestRecord(x=1, y=2, _ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record

def test_precord_constructor_uses_initial_values_from_class():
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=lambda: 20)
    record = TestRecord()
    assert record['x'] == 10
    assert record['y'] == 20

def test_precord_constructor_overrides_initial_values():
    class TestRecord(PRecord):
        x = field(initial=10)
        y = field(initial=20)
    record = TestRecord(x=100)
    assert record['x'] == 100
    assert record['y'] == 20

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = field()
        y = field()
    record = TestRecord(x=1, y=2, _factory_fields={'x': True})
    assert record['x'] == 1
    assert record['y'] == 2

def test_precord_constructor_creates_same_instance_via_internal_attributes():
    class TestRecord(PRecord):
        x = field()
    internal_record = TestRecord(x=5)
    new_record = TestRecord(_precord_size=internal_record._precord_size, _precord_buckets=internal_record._precord_buckets)
    assert new_record == internal_record
    assert new_record['x'] == 5

def test_precord_constructor_raises_error_for_unknown_field_by_default():
    class TestRecord(PRecord):
        x = field()
    try:
        TestRecord(x=1, y=2)
        assert False
    except AttributeError:
        pass

def test_precord_constructor_accepts_mapping_in_create_method():
    class TestRecord(PRecord):
        x = field()
        y = field()
    record = TestRecord.create({'x': 1, 'y': 2})
    assert record['x'] == 1
    assert record['y'] == 2

def test_precord_constructor_create_ignores_extra_fields():
    class TestRecord(PRecord):
        x = field()
    record = TestRecord.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert record['x'] == 1
    assert 'y' not in record

def test_precord_constructor_create_returns_same_instance():
    class TestRecord(PRecord):
        x = field()
    original = TestRecord(x=1)
    created = TestRecord.create(original)
    assert created is original


# LLM-generated content at query #4
#--------------------------

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        _precord_fields = {}
        _precord_initial_values = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert result == {}

def test_precord_new_with_regular_kwargs():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int), 'y': field(type=str)}
        _precord_initial_values = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    result = TestRecord(x=10, y='hello')
    assert isinstance(result, TestRecord)
    assert result == {'x': 10, 'y': 'hello'}

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int, factory=lambda v: v * 2)}
        _precord_initial_values = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    result = TestRecord(x=5, _factory_fields={'x'})
    assert isinstance(result, TestRecord)
    assert result == {'x': 10}

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int)}
        _precord_initial_values = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    result = TestRecord(x=1, y=2, _ignore_extra=True)
    assert isinstance(result, TestRecord)
    assert result == {'x': 1}

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int), 'y': field(type=int)}
        _precord_initial_values = {'y': lambda: 100}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    result = TestRecord(x=5)
    assert isinstance(result, TestRecord)
    assert result == {'x': 5, 'y': 100}

def test_precord_new_with_initial_values_and_kwargs():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int), 'y': field(type=int)}
        _precord_initial_values = {'y': 200}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    result = TestRecord(x=5, y=300)
    assert isinstance(result, TestRecord)
    assert result == {'x': 5, 'y': 300}

def test_precord_new_with_mandatory_fields_missing():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int), 'y': field(type=int)}
        _precord_initial_values = {}
        _precord_mandatory_fields = {'x'}
        _precord_invariants = []
    try:
        TestRecord(y=5)
        assert False
    except InvariantException as e:
        assert 'TestRecord.x' in e.missing_fields

def test_precord_new_with_invariant_failure():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int, invariant=lambda v: (v > 0, 'ERR'))}
        _precord_initial_values = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    try:
        TestRecord(x=-1)
        assert False
    except InvariantException as e:
        assert 'ERR' in e.invariant_errors

def test_precord_new_with_global_invariant_failure():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int), 'y': field(type=int)}
        _precord_initial_values = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [lambda r: (r['x'] + r['y'] > 0, 'SUM_ERR')]
    try:
        TestRecord(x=-5, y=2)
        assert False
    except InvariantException as e:
        assert 'SUM_ERR' in e.invariant_errors

def test_precord_new_with_field_type_check_failure():
    class TestRecord(PRecord):
        _precord_fields = {'x': field(type=int)}
        _precord_initial_values = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    try:
        TestRecord(x='not_an_int')
        assert False
    except TypeError:
        pass


# LLM-generated content at query #5
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
        _precord_mandatory_fields = {'required_field'}
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
        assert 'required_field' in str(e.missing_fields)

def test_persistent_raises_invariant_exception_on_field_invariant_errors():
    class TestRecord:
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (False, 'error')})()}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
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

def test_persistent_returns_pmap_if_not_dirty_and_already_instance():
    class TestRecord(PMap):
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []
    original = TestRecord()
    evolver = _PRecordEvolver(TestRecord, original)
    result = evolver.persistent()
    assert result is original

def test_persistent_creates_new_instance_if_dirty():
    class TestRecord:
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (True, None), 'factory': lambda x: x})()}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=0):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: set()
    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('field', 'value')
    result = evolver.persistent()
    assert isinstance(result, TestRecord)


# LLM-generated content at query #6
#--------------------------

def test_serialize_with_no_serializers():
    class TestRecord(PRecord):
        _precord_fields = {'a': None, 'b': None}
    record = TestRecord(a=1, b='test')
    result = record.serialize()
    assert result == {'a': 1, 'b': 'test'}

def test_serialize_with_custom_serializer():
    class TestRecord(PRecord):
        _precord_fields = {'a': type('Field', (), {'serializer': lambda v, f: v * 2})}
    record = TestRecord(a=5)
    result = record.serialize()
    assert result == {'a': 10}

def test_serialize_with_format_parameter():
    class TestRecord(PRecord):
        _precord_fields = {'a': type('Field', (), {'serializer': lambda v, f: f"{f}:{v}"})}
    record = TestRecord(a=100)
    result = record.serialize(format='json')
    assert result == {'a': 'json:100'}

def test_serialize_with_multiple_fields_and_serializers():
    class TestRecord(PRecord):
        _precord_fields = {
            'x': type('Field', (), {'serializer': lambda v, f: v.upper()}),
            'y': type('Field', (), {'serializer': lambda v, f: v + v})
        }
    record = TestRecord(x='hello', y='ab')
    result = record.serialize()
    assert result == {'x': 'HELLO', 'y': 'abab'}

def test_serialize_mixed_fields_some_without_serializer():
    class TestRecord(PRecord):
        _precord_fields = {
            'with_serializer': type('Field', (), {'serializer': lambda v, f: v * 3}),
            'without_serializer': None
        }
    record = TestRecord(with_serializer=2, without_serializer='data')
    result = record.serialize()
    assert result == {'with_serializer': 6, 'without_serializer': 'data'}


# LLM-generated content at query #7
#--------------------------

def test___new___sets_fields_correctly():
    class MockField:
        def __init__(self, mandatory=False, initial=None):
            self.mandatory = mandatory
            self.initial = initial
    field1 = MockField(mandatory=True, initial=None)
    field2 = MockField(mandatory=False, initial=10)
    dct = {'field1': field1, 'field2': field2}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_fields' in dct
    assert dct['_precord_fields']['field1'] is field1
    assert dct['_precord_fields']['field2'] is field2
    assert 'field1' not in dct
    assert 'field2' not in dct

def test___new___inherits_fields_from_bases():
    class Base:
        _precord_fields = {'base_field': MockField()}
    class MockField:
        pass
    field = MockField()
    dct = {'new_field': field}
    bases = (Base,)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_fields' in dct
    assert 'base_field' in dct['_precord_fields']
    assert 'new_field' in dct['_precord_fields']
    assert dct['_precord_fields']['new_field'] is field

def test___new___sets_mandatory_fields():
    class MockField:
        def __init__(self, mandatory):
            self.mandatory = mandatory
            self.initial = None
    field1 = MockField(mandatory=True)
    field2 = MockField(mandatory=False)
    dct = {'field1': field1, 'field2': field2}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_mandatory_fields' in dct
    assert dct['_precord_mandatory_fields'] == {'field1'}

def test___new___sets_initial_values():
    class MockField:
        def __init__(self, initial):
            self.mandatory = False
            self.initial = initial
    field1 = MockField(initial=5)
    field2 = MockField(initial=None)
    dct = {'field1': field1, 'field2': field2}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_initial_values' in dct
    assert dct['_precord_initial_values'] == {'field1': 5}

def test___new___stores_invariants():
    def invariant1(obj):
        return True, ()
    def invariant2(obj):
        return False, ('error',)
    dct = {'__invariant__': invariant1}
    class Base:
        __invariant__ = invariant2
    bases = (Base,)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_invariants' in dct
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    assert invariants[0](None) == (True, ())
    assert invariants[1](None) == (False, ('error',))

def test___new___wraps_invariants():
    def invariant(obj):
        return [(True, ()), (False, ('err1',)), (False, ('err2',))]
    dct = {'__invariant__': invariant}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    wrapped_invariant = dct['_precord_invariants'][0]
    result = wrapped_invariant(None)
    assert result == (False, ('err1', 'err2'))

def test___new___raises_on_non_callable_invariant():
    dct = {'__invariant__': 'not a callable'}
    bases = ()
    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
        assert False
    except TypeError:
        pass

def test___new___sets_slots():
    dct = {}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '__slots__' in dct
    assert dct['__slots__'] == ()

def test___new___handles_empty_fields():
    dct = {}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    assert '_precord_fields' in dct
    assert dct['_precord_fields'] == {}
    assert '_precord_mandatory_fields' in dct
    assert dct['_precord_mandatory_fields'] == set()
    assert '_precord_initial_values' in dct
    assert dct['_precord_initial_values'] == {}

def test___new___inherits_invariants_from_multiple_bases():
    def inv1(obj):
        return True, ()
    def inv2(obj):
        return False, ('error',)
    class Base1:
        __invariant__ = inv1
    class Base2:
        __invariant__ = inv2
    dct = {}
    bases = (Base1, Base2)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    assert invariants[0](None) == (True, ())
    assert invariants[1](None) == (False, ('error',))


# LLM-generated content at query #8
#--------------------------

def test_precord_constructor_with_special_attributes():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
    record = TestRecord(_precord_size=0, _precord_buckets=[])
    assert isinstance(record, TestRecord)
    assert len(record) == 0

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None, 'field2': None}
    record = TestRecord(field1='value1', field2='value2')
    assert record['field1'] == 'value1'
    assert record['field2'] == 'value2'

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None}
    record = TestRecord(_factory_fields={'field1': 'factory_value'}, field1='value1')
    assert record['field1'] == 'value1'

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None}
    record = TestRecord(_ignore_extra=True, field1='value1', extra_field='extra')
    assert record['field1'] == 'value1'
    assert 'extra_field' not in record

def test_precord_constructor_with_class_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None, 'field2': None}
        _precord_initial_values = {'field1': lambda: 'default1', 'field2': 'default2'}
    record = TestRecord()
    assert record['field1'] == 'default1'
    assert record['field2'] == 'default2'

def test_precord_constructor_overrides_initial_values():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None, 'field2': None}
        _precord_initial_values = {'field1': lambda: 'default1', 'field2': 'default2'}
    record = TestRecord(field1='custom1')
    assert record['field1'] == 'custom1'
    assert record['field2'] == 'default2'

def test_precord_constructor_with_callable_initial_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None}
        _precord_initial_values = {'field1': lambda: 'callable_result'}
    record = TestRecord()
    assert record['field1'] == 'callable_result'

def test_precord_constructor_with_non_callable_initial_value():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': None}
        _precord_initial_values = {'field1': 'static_value'}
    record = TestRecord()
    assert record['field1'] == 'static_value'

def test_precord_constructor_creates_empty_record():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
    record = TestRecord()
    assert len(record) == 0

def test_precord_constructor_with_multiple_fields():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'a': None, 'b': None, 'c': None}
    record = TestRecord(a=1, b=2, c=3)
    assert record['a'] == 1
    assert record['b'] == 2
    assert record['c'] == 3


# LLM-generated content at query #9
#--------------------------

def test_set_with_valid_field_and_value():
    from pyrsistent import PRecord, field
    from pyrsistent._checked_types import CheckedType
    class TestRecord(PRecord):
        name = field(type=str)
    rec = TestRecord()
    evolver = rec.evolver()
    evolver.set('name', 'Alice')
    result = evolver.persistent()
    assert result.name == 'Alice'

def test_set_with_invalid_type_raises_ptype_error():
    from pyrsistent import PRecord, field
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import PTypeError
    class TestRecord(PRecord):
        age = field(type=int)
    rec = TestRecord()
    evolver = rec.evolver()
    try:
        evolver.set('age', 'not_an_int')
        assert False
    except PTypeError as e:
        assert e.destination_cls == TestRecord
        assert e.field_name == 'age'

def test_set_with_field_factory_and_ignore_extra():
    from pyrsistent import PRecord, field
    from pyrsistent._checked_types import CheckedType
    def factory_func(value, ignore_extra=False):
        return value.upper()
    class TestRecord(PRecord):
        data = field(type=str, factory=factory_func)
    rec = TestRecord()
    evolver = rec.evolver()
    evolver.set('data', 'hello')
    result = evolver.persistent()
    assert result.data == 'HELLO'

def test_set_with_invariant_failure():
    from pyrsistent import PRecord, field
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import InvariantException
    def invariant_func(value):
        return (value > 0, 'ERR_NEGATIVE')
    class TestRecord(PRecord):
        number = field(type=int, invariant=invariant_func)
    rec = TestRecord()
    evolver = rec.evolver()
    evolver.set('number', -5)
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'ERR_NEGATIVE' in e.invariant_errors

def test_set_with_nonexistent_field_raises_attribute_error():
    from pyrsistent import PRecord, field
    from pyrsistent._checked_types import CheckedType
    class TestRecord(PRecord):
        existing = field(type=str)
    rec = TestRecord()
    evolver = rec.evolver()
    try:
        evolver.set('nonexistent', 'value')
        assert False
    except AttributeError as e:
        assert 'nonexistent' in str(e)

def test_set_with_factory_fields_skipping_factory():
    from pyrsistent import PRecord, field
    from pyrsistent._checked_types import CheckedType
    def factory_func(value):
        return value * 2
    class TestRecord(PRecord):
        value = field(type=int, factory=factory_func)
    rec = TestRecord()
    evolver = rec.evolver()
    evolver._factory_fields = set()
    evolver.set('value', 3)
    result = evolver.persistent()
    assert result.value == 3

def test_set_with_factory_exception_adds_to_invariant_errors():
    from pyrsistent import PRecord, field
    from pyrsistent._checked_types import CheckedType
    from pyrsistent._field_common import InvariantException
    def factory_func(value):
        raise InvariantException(('ERR_FACTORY',), (), 'Factory failed')
    class TestRecord(PRecord):
        item = field(type=str, factory=factory_func)
    rec = TestRecord()
    evolver = rec.evolver()
    evolver.set('item', 'test')
    try:
        evolver.persistent()
        assert False
    except InvariantException as e:
        assert 'ERR_FACTORY' in e.invariant_errors


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

```python
def test_store_invariants_collects_and_wraps_invariants_from_all_bases():
    class Base1:
        __invariant__ = lambda self: (True, ())
    class Base2:
        __invariant__ = lambda self: (False, ("error",))
    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    assert len(invariants) == 2
    assert invariants[0](None) == (True, ())
    assert invariants[1](None) == (False, ("error",))


# LLM-generated content at query #12
#--------------------------

def test_precord_repr_returns_correct_format():
    class TestRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'x': field(type=int), 'y': field(type=str)}
        _precord_initial_values = {}
    record = TestRecord(x=10, y='hello')
    result = repr(record)
    expected = "TestRecord(x=10, y='hello')"
    assert result == expected


# LLM-generated content at query #13
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
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (False, 'error_code')})()}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: {'field'}
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
    original_pmap = {}
    evolver = _PRecordEvolver(MockClass, original_pmap)
    evolver.set('new_field', 'new_value')
    result = evolver.persistent()
    assert isinstance(result, MockClass)
    assert result._buckets is not None


# LLM-generated content at query #14
#--------------------------

def test_precord_constructor_without_special_attributes():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {}
    instance = MyRecord(field1='value1', field2='value2')
    assert instance['field1'] == 'value1'
    assert instance['field2'] == 'value2'

def test_precord_constructor_with_special_attributes():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {}
        _precord_initial_values = {}
    size = 0
    buckets = []
    instance = MyRecord(_precord_size=size, _precord_buckets=buckets)
    assert isinstance(instance, MyRecord)

def test_precord_constructor_with_factory_fields():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field()}
        _precord_initial_values = {}
    instance = MyRecord(_factory_fields={'field1': 'factory_value'}, field1='value1')
    assert instance['field1'] == 'value1'

def test_precord_constructor_with_ignore_extra():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field()}
        _precord_initial_values = {}
    instance = MyRecord(_ignore_extra=True, field1='value1', extra_field='extra')
    assert instance['field1'] == 'value1'
    assert 'extra_field' not in instance

def test_precord_constructor_with_initial_values():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {'field1': lambda: 'initial1', 'field2': 'initial2'}
    instance = MyRecord()
    assert instance['field1'] == 'initial1'
    assert instance['field2'] == 'initial2'

def test_precord_constructor_overrides_initial_values():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field(), 'field2': field()}
        _precord_initial_values = {'field1': lambda: 'initial1', 'field2': 'initial2'}
    instance = MyRecord(field1='override1')
    assert instance['field1'] == 'override1'
    assert instance['field2'] == 'initial2'

def test_precord_constructor_with_callable_initial_value():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field()}
        _precord_initial_values = {'field1': lambda: 'callable_result'}
    instance = MyRecord()
    assert instance['field1'] == 'callable_result'

def test_precord_constructor_with_non_callable_initial_value():
    class MyRecord(PRecord):
        __slots__ = ()
        _precord_fields = {'field1': field()}
        _precord_initial_values = {'field1': 'static_value'}
    instance = MyRecord()
    assert instance['field1'] == 'static_value'


# LLM-generated content at query #15
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

def test_persistent_returns_persistent_map_when_not_dirty_and_already_instance():
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
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (True, None), 'factory': lambda x: x})()}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: {'field'}
    evolver = _PRecordEvolver(MockClass, {})
    evolver.set('field', 'value')
    result = evolver.persistent()
    assert isinstance(result, MockClass)
    assert result._buckets is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_set_fields_creates_precord_fields_in_dct():
    from pyrsistent._field_common import set_fields
    from pyrsistent import PRecord, field

    class Base(PRecord):
        base_field = field()

    class Derived(PRecord):
        derived_field = field()

    dct = {}
    bases = (Base,)
    set_fields(dct, bases, name='_precord_fields')
    result = '_precord_fields' in dct
    assert result

def test_set_fields_moves_pfield_instances_to_precord_fields():
    from pyrsistent._field_common import set_fields
    from pyrsistent._field_common import _PField

    class MockPField(_PField):
        def __init__(self):
            self.mandatory = False
            self.initial = None

    dct = {'field1': MockPField(), 'field2': MockPField(), 'regular': 'value'}
    bases = ()
    set_fields(dct, bases, name='_precord_fields')
    result = 'field1' not in dct and 'field2' not in dct and 'regular' in dct
    assert result

def test_set_fields_includes_fields_from_bases():
    from pyrsistent._field_common import set_fields
    from pyrsistent._field_common import _PField

    class MockPField(_PField):
        def __init__(self, name):
            self.mandatory = False
            self.initial = None
            self.name = name

    class Base1:
        _precord_fields = {'base1_field': MockPField('base1')}

    class Base2:
        _precord_fields = {'base2_field': MockPField('base2')}

    dct = {'derived_field': MockPField('derived')}
    bases = (Base1, Base2)
    set_fields(dct, bases, name='_precord_fields')
    fields = dct['_precord_fields']
    result = 'base1_field' in fields and 'base2_field' in fields and 'derived_field' in fields
    assert result

def test_store_invariants_creates_precord_invariants_in_dct():
    from pyrsistent._checked_types import store_invariants

    def invariant1(instance):
        return True, ()

    dct = {'__invariant__': invariant1}
    bases = ()
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    result = '_precord_invariants' in dct
    assert result

def test_store_invariants_inherits_from_bases():
    from pyrsistent._checked_types import store_invariants

    def invariant1(instance):
        return True, ()

    def invariant2(instance):
        return True, ()

    class Base1:
        __invariant__ = invariant1

    class Base2:
        __invariant__ = invariant2

    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    invariants = dct['_precord_invariants']
    result = len(invariants) == 2
    assert result

def test_store_invariants_wraps_invariants():
    from pyrsistent._checked_types import store_invariants

    def invariant(instance):
        return [(True, ()), (False, 'error')]

    dct = {'__invariant__': invariant}
    bases = ()
    store_invariants(dct, bases, '_precord_invariants', '__invariant__')
    wrapped_invariant = dct['_precord_invariants'][0]
    verdict, data = wrapped_invariant(None)
    result = verdict == False and data == ('error',)
    assert result

def test_precord_meta_creates_mandatory_fields_set():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField

    class MockPField(_PField):
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial

    dct = {
        'mandatory_field': MockPField(True, None),
        'optional_field': MockPField(False, None)
    }
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    mandatory_fields = dct['_precord_mandatory_fields']
    result = 'mandatory_field' in mandatory_fields and 'optional_field' not in mandatory_fields
    assert result

def test_precord_meta_creates_initial_values_dict():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL

    class MockPField(_PField):
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial

    dct = {
        'with_initial': MockPField(False, 'default'),
        'without_initial': MockPField(False, PFIELD_NO_INITIAL)
    }
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    initial_values = dct['_precord_initial_values']
    result = 'with_initial' in initial_values and 'without_initial' not in initial_values
    assert result

def test_precord_meta_sets_slots():
    from pyrsistent._precord import _PRecordMeta

    dct = {}
    bases = ()
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    result = '__slots__' in dct and dct['__slots__'] == ()
    assert result

def test_precord_meta_integration():
    from pyrsistent._precord import _PRecordMeta
    from pyrsistent._field_common import _PField, PFIELD_NO_INITIAL

    class MockPField(_PField):
        def __init__(self, mandatory, initial):
            self.mandatory = mandatory
            self.initial = initial

    def invariant1(instance):
        return True, ()

    def invariant2(instance):
        return [(True, ()), (False, 'error')]

    class Base:
        __invariant__ = invariant1
        base_field = MockPField(True, 'base_default')

    dct = {
        '__invariant__': invariant2,
        'derived_field': MockPField(False, PFIELD_NO_INITIAL),
        'another_field': MockPField(True, 'derived_default')
    }
    bases = (Base,)
    _PRecordMeta.__new__(_PRecordMeta, 'TestClass', bases, dct)
    result = '_precord_fields' in dct and '_precord_invariants' in dct and '_precord_mandatory_fields' in dct and '_precord_initial_values' in dct and '__slots__' in dct
    assert result


# LLM-generated content at query #17
#--------------------------

```python
def test_persistent_when_not_dirty_and_pm_is_instance_of_cls():
    from pyrsistent import InvariantException
    from pyrsistent._precord import _PRecordEvolver
    from pyrsistent._pmap import PMap

    class MockPMap(PMap):
        _buckets = {}
        _size = 0

    class MockCls:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

        def __init__(self, _precord_buckets=None, _precord_size=None):
            pass

    class MockEvolver(_PRecordEvolver):
        def __init__(self):
            self._destination_cls = MockCls
            self._invariant_error_codes = []
            self._missing_fields = []
            self._factory_fields = None
            self._ignore_extra = False
            self._original_pmap = MockPMap()

        def is_dirty(self):
            return False

        def persistent(self):
            pm = MockPMap()
            return super(MockEvolver, self).persistent()

    evolver = MockEvolver()
    evolver._original_pmap = MockPMap()
    result = evolver.persistent()
    assert isinstance(result, MockPMap)


# LLM-generated content at query #18
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


# LLM-generated content at query #19
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
            self.keys = lambda: {'field'}
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

def test_persistent_returns_persistent_map_when_not_dirty_and_already_instance():
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
        _precord_fields = {'field': type('Field', (), {'invariant': lambda x: (True, None), 'factory': lambda x: x})()}
        _precord_mandatory_fields = set()
        _precord_invariants = []
        def __init__(self, _precord_buckets=None, _precord_size=None):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: {'field'}
    evolver = _PRecordEvolver(MockClass, {})
    evolver.set('field', 'value')
    result = evolver.persistent()
    assert isinstance(result, MockClass)
    assert result._buckets is not None


# LLM-generated content at query #20
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
            self.factory.__signature__ = sig.replace(parameters=factory_params)

    class MockCheckedType(CheckedType):
        pass

    field_type = (MockCheckedType,)
    factory_params = tuple(inspect.Parameter(name='ignore_extra', kind=inspect.Parameter.POSITIONAL_OR_KEYWORD))
    field = MockField(field_type, factory_params)
    result = is_field_ignore_extra_complaint(CheckedType, field, True)
    assert result == True


# LLM-generated content at query #21
#--------------------------

```python
def test_persistent_when_cls_has_precord_mandatory_fields():
    class MockClass:
        _precord_mandatory_fields = {"field1", "field2"}
        __name__ = "MockClass"
        _precord_buckets = None
        _precord_size = 0
        _precord_invariants = []
        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size
            self.keys = lambda: {"field1"}
    evolver = _PRecordEvolver(MockClass, None)
    evolver._missing_fields = []
    evolver._invariant_error_codes = []
    evolver.is_dirty = lambda: False
    evolver.persistent = lambda: MockClass(None, 0)
    result = evolver.persistent()
    assert MockClass._precord_mandatory_fields


# LLM-generated content at query #22
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true():
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import CheckedType
    import inspect

    class MockField:
        def __init__(self, type_set, factory_params):
            self.type = type_set
            self.factory = lambda x, **kwargs: x
            self.factory.__signature__ = inspect.signature(lambda x, **kwargs: x).replace(parameters=factory_params)

    field_type = {CheckedType}
    factory_params = [inspect.Parameter('ignore_extra', inspect.Parameter.KEYWORD_ONLY, default=False)]
    field = MockField(field_type, factory_params)
    ignore_extra = True
    result = is_field_ignore_extra_complaint(CheckedType, field, ignore_extra)
    assert result is True


# LLM-generated content at query #23
#--------------------------

```python
def test_persistent_when_cls_has_precord_mandatory_fields():
    class MockClass:
        _precord_mandatory_fields = {"field1", "field2"}
        __name__ = "MockClass"

    evolver = _PRecordEvolver(MockClass, {})
    evolver._missing_fields = []
    result = {"field1": "value1"}
    cls = MockClass
    missing_fields = cls._precord_mandatory_fields - set(result.keys())
    assert cls._precord_mandatory_fields


