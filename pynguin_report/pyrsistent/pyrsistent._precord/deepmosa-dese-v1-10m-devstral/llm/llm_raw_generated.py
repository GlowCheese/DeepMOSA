####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    cls = _PRecordEvolver._destination_cls
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'value1')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result._precord_buckets == evolver._buckets
    assert result._precord_size == evolver._size

def test_persistent_with_clean_and_cls_instance():
    cls = _PRecordEvolver._destination_cls
    pm = cls(_precord_buckets=PMap()._buckets, _precord_size=PMap()._size)
    evolver = _PRecordEvolver(cls, pm, _factory_fields=None, _ignore_extra=False)
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    cls = _PRecordEvolver._destination_cls
    cls._precord_mandatory_fields = {'field1', 'field2'}
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'value1')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'field2' in e.missing_fields

def test_persistent_with_invariant_errors():
    cls = _PRecordEvolver._destination_cls
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver._invariant_error_codes = ['error1', 'error2']
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')

def test_persistent_with_global_invariant_failure():
    cls = _PRecordEvolver._destination_cls
    def failing_invariant(subject):
        return (False, 'global_error')
    cls._precord_invariants = [failing_invariant]
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=None, _ignore_extra=False)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)


# LLM-generated content at query #2
#--------------------------

```python
def test_new_with_empty_bases_and_dct():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {})
    assert result._precord_fields == {}
    assert result._precord_invariants == ()
    assert result._precord_mandatory_fields == set()
    assert result._precord_initial_values == {}
    assert result.__slots__ == ()

def test_new_with_fields_in_dct():
    dct = {'field1': _PField(mandatory=True), 'field2': _PField(initial=42)}
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), dct)
    assert result._precord_fields == {'field1': _PField(mandatory=True), 'field2': _PField(initial=42)}
    assert result._precord_mandatory_fields == {'field1'}
    assert result._precord_initial_values == {'field2': 42}

def test_new_with_inherited_fields():
    class Parent:
        field1 = _PField(mandatory=True)

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Parent,), {})
    assert result._precord_fields == {'field1': _PField(mandatory=True)}
    assert result._precord_mandatory_fields == {'field1'}

def test_new_with_invariant_in_dct():
    def test_invariant(obj):
        return True, "Test"

    dct = {'__invariant__': test_invariant}
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), dct)
    assert len(result._precord_invariants) == 1
    assert callable(result._precord_invariants[0])

def test_new_with_inherited_invariant():
    def test_invariant(obj):
        return True, "Test"

    class Parent:
        __invariant__ = test_invariant

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Parent,), {})
    assert len(result._precord_invariants) == 1
    assert callable(result._precord_invariants[0])

def test_new_with_multiple_invariants():
    def test_invariant1(obj):
        return True, "Test1"

    def test_invariant2(obj):
        return True, "Test2"

    class Parent1:
        __invariant__ = test_invariant1

    class Parent2:
        __invariant__ = test_invariant2

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Parent1, Parent2), {})
    assert len(result._precord_invariants) == 2
    assert all(callable(inv) for inv in result._precord_invariants)

def test_new_with_non_callable_invariant_raises_type_error():
    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'__invariant__': "not callable"})
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'


# LLM-generated content at query #3
#--------------------------

```python
def test_persistent_returns_same_instance_when_not_dirty_and_correct_type():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    result = evolver.persistent()
    assert result is evolver._original_pmap

def test_persistent_creates_new_instance_when_dirty():
    class TestRecord:
        _precord_fields = {'field': None}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('field', 'value')
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result != evolver._original_pmap

def test_persistent_creates_new_instance_when_not_correct_type():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._original_pmap = PMap({'key': 'value'})
    result = evolver.persistent()
    assert isinstance(result, TestRecord)

def test_persistent_raises_invariant_exception_for_missing_mandatory_fields():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'missing_field'}
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'missing_field' in e.missing_fields

def test_persistent_raises_invariant_exception_for_field_invariants():
    class TestRecord:
        _precord_fields = {'field': None}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._invariant_error_codes = ['error_code']
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'error_code' in e.invariant_errors

def test_persistent_calls_check_global_invariants():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [lambda x: (False, 'global_error')]

    evolver = _PRecordEvolver(TestRecord, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'global_error' in e.invariant_errors


# LLM-generated content at query #4
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    cls = PRecord
    kwargs = {'_precord_size': 2, '_precord_buckets': [None, [(1, 2)]]}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result._buckets == [None, [(1, 2)]]

def test_precord_new_without_precord_size_and_buckets():
    cls = PRecord
    kwargs = {'a': 1, 'b': 2}
    result = cls.__new__(cls, **kwargs)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_factory_fields():
    cls = PRecord
    kwargs = {'a': 1, 'b': 2, '_factory_fields': ['a', 'b']}
    result = cls.__new__(cls, **kwargs)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_ignore_extra():
    cls = PRecord
    kwargs = {'a': 1, 'b': 2, 'c': 3, '_ignore_extra': True}
    result = cls.__new__(cls, **kwargs)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_initial_values():
    cls = PRecord
    cls._precord_initial_values = {'a': 1, 'b': 2}
    kwargs = {'b': 3}
    result = cls.__new__(cls, **kwargs)
    assert result == {'a': 1, 'b': 3}

def test_precord_new_with_callable_initial_values():
    cls = PRecord
    cls._precord_initial_values = {'a': lambda: 1, 'b': 2}
    kwargs = {'b': 3}
    result = cls.__new__(cls, **kwargs)
    assert result == {'a': 1, 'b': 3}


# LLM-generated content at query #5
#--------------------------

```python
def test_precord_mandatory_fields_contains_only_mandatory_fields():
    class TestRecord(metaclass=_PRecordMeta):
        mandatory_field = _PField(mandatory=True)
        optional_field = _PField(mandatory=False)

    assert 'mandatory_field' in TestRecord._precord_mandatory_fields
    assert 'optional_field' not in TestRecord._precord_mandatory_fields


# LLM-generated content at query #6
#--------------------------

```python
def test_precord_initial_values_present():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': lambda: 2}

    result = TestRecord()
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #7
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_default_values():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_callable_defaults():
    class TestRecord(PRecord):
        x = lambda: [1, 2, 3]
        y = 2

    record = TestRecord()
    assert record.x == [1, 2, 3]
    assert record.y == 2

def test_precord_constructor_with_extra_fields_ignored():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20, z=30, _ignore_extra=True)
    assert record.x == 10
    assert record.y == 20
    assert 'z' not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20, _factory_fields={'x': 5})
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_precord_size_and_buckets():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_precord_size=2, _precord_buckets=[('x', 1), ('y', 2)])
    assert record.x == 1
    assert record.y == 2


# LLM-generated content at query #8
#--------------------------

```python
def test_repr_empty():
    class EmptyRecord(PRecord):
        pass
    record = EmptyRecord()
    assert repr(record) == "EmptyRecord()"

def test_repr_single_field():
    class SingleFieldRecord(PRecord):
        field1 = Field()
    record = SingleFieldRecord(field1="value1")
    assert repr(record) == "SingleFieldRecord(field1='value1')"

def test_repr_multiple_fields():
    class MultiFieldRecord(PRecord):
        field1 = Field()
        field2 = Field()
    record = MultiFieldRecord(field1="value1", field2="value2")
    assert repr(record) == "MultiFieldRecord(field1='value1', field2='value2')"

def test_repr_with_complex_values():
    class ComplexRecord(PRecord):
        field1 = Field()
        field2 = Field()
    record = ComplexRecord(field1={"nested": "dict"}, field2=[1, 2, 3])
    assert repr(record) == "ComplexRecord(field1={'nested': 'dict'}, field2=[1, 2, 3])"


# LLM-generated content at query #9
#--------------------------

```python
def test_repr_format():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord()
    repr_str = repr(record)
    assert repr_str.startswith("TestRecord(") and repr_str.endswith(")")
    assert "x=1" in repr_str and "y=2" in repr_str


# LLM-generated content at query #10
#--------------------------

```python
def test_precord_mandatory_fields_predicate():
    class TestRecord(metaclass=_PRecordMeta):
        a = _PField(mandatory=True)
        b = _PField(mandatory=False)

    assert all(field.mandatory for field in TestRecord._precord_mandatory_fields)


# LLM-generated content at query #11
#--------------------------

```python
def test_persistent_returns_same_instance_when_not_dirty_and_correct_type():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    result = evolver.persistent()
    assert result is evolver._map

def test_persistent_creates_new_instance_when_dirty():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result != evolver._map

def test_persistent_creates_new_instance_when_wrong_type():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._map = PMap()  # Force a different type
    result = evolver.persistent()
    assert isinstance(result, TestRecord)

def test_persistent_raises_invariant_exception_for_missing_mandatory_fields():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    with pytest.raises(InvariantException) as excinfo:
        evolver.persistent()
    assert 'TestRecord.field1' in excinfo.value.missing_fields
    assert 'TestRecord.field2' in excinfo.value.missing_fields

def test_persistent_raises_invariant_exception_for_field_invariant_failures():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'invariant': lambda x: (False, 'ERROR')})()}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('field1', 'value')
    with pytest.raises(InvariantException) as excinfo:
        evolver.persistent()
    assert 'ERROR' in excinfo.value.invariant_errors

def test_persistent_raises_invariant_exception_for_global_invariant_failures():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [lambda x: (False, 'GLOBAL_ERROR')]

    evolver = _PRecordEvolver(TestRecord, PMap())
    with pytest.raises(InvariantException) as excinfo:
        evolver.persistent()
    assert 'GLOBAL_ERROR' in excinfo.value.invariant_errors


# LLM-generated content at query #12
#--------------------------

```python
def test_set_with_valid_field_and_factory():
    field = Mock()
    field.factory = Mock(return_value='processed_value')
    field.type = {int}
    field.invariant = Mock(return_value=(True, None))
    cls = Mock()
    cls._precord_fields = {'field_name': field}
    cls.__name__ = 'TestClass'
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields={field}, _ignore_extra=False)
    evolver.set('field_name', 42)
    assert evolver['field_name'] == 'processed_value'
    field.factory.assert_called_once_with(42)
    field.invariant.assert_called_once_with('processed_value')

def test_set_with_valid_field_no_factory():
    field = Mock()
    field.type = {int}
    field.invariant = Mock(return_value=(True, None))
    cls = Mock()
    cls._precord_fields = {'field_name': field}
    cls.__name__ = 'TestClass'
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field_name', 42)
    assert evolver['field_name'] == 42
    field.invariant.assert_called_once_with(42)

def test_set_with_invalid_type():
    field = Mock()
    field.type = {int}
    field.invariant = Mock(return_value=(True, None))
    cls = Mock()
    cls._precord_fields = {'field_name': field}
    cls.__name__ = 'TestClass'
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=None, _ignore_extra=False)
    with pytest.raises(PTypeError):
        evolver.set('field_name', 'not_an_int')

def test_set_with_invariant_failure():
    field = Mock()
    field.type = {int}
    field.invariant = Mock(return_value=(False, 'INVALID'))
    cls = Mock()
    cls._precord_fields = {'field_name': field}
    cls.__name__ = 'TestClass'
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field_name', 42)
    assert evolver._invariant_error_codes == ['INVALID']

def test_set_with_nonexistent_field():
    cls = Mock()
    cls._precord_fields = {}
    cls.__name__ = 'TestClass'
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=None, _ignore_extra=False)
    with pytest.raises(AttributeError):
        evolver.set('nonexistent_field', 42)

def test_set_with_ignore_extra_true():
    field = Mock()
    field.type = {CheckedType}
    field.factory = Mock(return_value='processed_value')
    field.invariant = Mock(return_value=(True, None))
    cls = Mock()
    cls._precord_fields = {'field_name': field}
    cls.__name__ = 'TestClass'
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields={field}, _ignore_extra=True)
    evolver.set('field_name', 42)
    assert evolver['field_name'] == 'processed_value'
    field.factory.assert_called_once_with(42, ignore_extra=True)

def test_set_with_ignore_extra_false():
    field = Mock()
    field.type = {CheckedType}
    field.factory = Mock(return_value='processed_value')
    field.invariant = Mock(return_value=(True, None))
    cls = Mock()
    cls._precord_fields = {'field_name': field}
    cls.__name__ = 'TestClass'
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields={field}, _ignore_extra=False)
    evolver.set('field_name', 42)
    assert evolver['field_name'] == 'processed_value'
    field.factory.assert_called_once_with(42)


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_5_evaluates_to_false():
    class TestRecord(PRecord):
        pass

    result = TestRecord.__new__(TestRecord)
    assert not ('_precord_size' in {} and '_precord_buckets' in {})


# LLM-generated content at query #14
#--------------------------

```python
def test_new_with_no_bases_and_no_fields():
    class TestRecord(metaclass=_PRecordMeta):
        pass
    assert TestRecord._precord_fields == {}
    assert TestRecord._precord_invariants == ()
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}
    assert TestRecord.__slots__ == ()

def test_new_with_fields_and_no_invariants():
    class TestRecord(metaclass=_PRecordMeta):
        field1 = _PField()
        field2 = _PField(mandatory=True)
        field3 = _PField(initial=42)
    assert TestRecord._precord_fields == {'field1': _PField(), 'field2': _PField(mandatory=True), 'field3': _PField(initial=42)}
    assert TestRecord._precord_invariants == ()
    assert TestRecord._precord_mandatory_fields == {'field2'}
    assert TestRecord._precord_initial_values == {'field3': 42}
    assert TestRecord.__slots__ == ()

def test_new_with_invariant():
    def test_invariant(instance):
        return True, "Test"
    class TestRecord(metaclass=_PRecordMeta):
        __invariant__ = test_invariant
    assert TestRecord._precord_fields == {}
    assert len(TestRecord._precord_invariants) == 1
    assert callable(TestRecord._precord_invariants[0])
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}
    assert TestRecord.__slots__ == ()

def test_new_with_inherited_fields():
    class BaseRecord(metaclass=_PRecordMeta):
        field1 = _PField()
    class TestRecord(BaseRecord):
        field2 = _PField(mandatory=True)
    assert TestRecord._precord_fields == {'field1': _PField(), 'field2': _PField(mandatory=True)}
    assert TestRecord._precord_invariants == ()
    assert TestRecord._precord_mandatory_fields == {'field2'}
    assert TestRecord._precord_initial_values == {}
    assert TestRecord.__slots__ == ()

def test_new_with_inherited_invariants():
    def base_invariant(instance):
        return True, "Base"
    class BaseRecord(metaclass=_PRecordMeta):
        __invariant__ = base_invariant
    class TestRecord(BaseRecord):
        pass
    assert TestRecord._precord_fields == {}
    assert len(TestRecord._precord_invariants) == 1
    assert callable(TestRecord._precord_invariants[0])
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}
    assert TestRecord.__slots__ == ()

def test_new_with_multiple_inherited_invariants():
    def base1_invariant(instance):
        return True, "Base1"
    def base2_invariant(instance):
        return True, "Base2"
    class Base1(metaclass=_PRecordMeta):
        __invariant__ = base1_invariant
    class Base2(metaclass=_PRecordMeta):
        __invariant__ = base2_invariant
    class TestRecord(Base1, Base2):
        pass
    assert TestRecord._precord_fields == {}
    assert len(TestRecord._precord_invariants) == 2
    assert all(callable(inv) for inv in TestRecord._precord_invariants)
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}
    assert TestRecord.__slots__ == ()

def test_new_with_non_callable_invariant_raises_type_error():
    try:
        class TestRecord(metaclass=_PRecordMeta):
            __invariant__ = "not callable"
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"

def test_new_with_invariant_returning_multiple_results():
    def multi_invariant(instance):
        return [(True, "Test1"), (False, "Test2")]
    class TestRecord(metaclass=_PRecordMeta):
        __invariant__ = multi_invariant
    assert TestRecord._precord_fields == {}
    assert len(TestRecord._precord_invariants) == 1
    result = TestRecord._precord_invariants[0](None)
    assert result == (False, ("Test2",))
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}
    assert TestRecord.__slots__ == ()


# LLM-generated content at query #15
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    evolver = _PRecordEvolver(TestPRecord, PMap({}), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'value1')
    assert evolver.is_dirty() is True
    result = evolver.persistent()
    assert isinstance(result, TestPRecord)
    assert result.field1 == 'value1'

def test_persistent_with_clean_and_cls_instance():
    original = TestPRecord(field1='value1')
    evolver = _PRecordEvolver(TestPRecord, original._as_pmap(), _factory_fields=None, _ignore_extra=False)
    assert evolver.is_dirty() is False
    result = evolver.persistent()
    assert result is original

def test_persistent_with_missing_mandatory_fields():
    evolver = _PRecordEvolver(TestPRecord, PMap({}), _factory_fields=None, _ignore_extra=False)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestPRecord.field1' in e.missing_fields

def test_persistent_with_field_invariant_failure():
    evolver = _PRecordEvolver(TestPRecord, PMap({}), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'invalid_value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0

def test_persistent_with_global_invariant_failure():
    evolver = _PRecordEvolver(TestPRecord, PMap({}), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'value1')
    evolver.set('field2', 'invalid_value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_persistent_with_dirty_and_non_instance():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    evolver.set('field1', 'value1')
    result = evolver.persistent()
    assert isinstance(result, TestPRecord)
    assert result['field1'] == 'value1'

def test_persistent_with_clean_and_instance():
    pm = TestPRecord(_precord_buckets=PMap()._buckets, _precord_size=PMap()._size)
    evolver = _PRecordEvolver(TestPRecord, pm)
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    evolver.set('field1', 'value1')
    try:
        evolver.persistent()
    except InvariantException as e:
        assert 'TestPRecord.mandatory_field' in e.missing_fields

def test_persistent_with_invariant_errors():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    evolver.set('field1', 'invalid_value')
    try:
        evolver.persistent()
    except InvariantException as e:
        assert 'INVALID_FIELD1' in e.invariant_errors

def test_persistent_with_global_invariant_failure():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    evolver.set('field1', 'value1')
    evolver.set('field2', 'value2')
    try:
        evolver.persistent()
    except InvariantException as e:
        assert 'GLOBAL_INVARIANT' in e.invariant_errors


# LLM-generated content at query #2
#--------------------------

```python
def test_set_with_valid_field_and_factory():
    class TestRecord(PRecord):
        field = field(type=CheckedType, factory=lambda x: x)

    evolver = TestRecord._evolver()
    result = evolver.set('field', 'value')
    assert result is evolver
    assert evolver['field'] == 'value'

def test_set_with_invalid_field():
    class TestRecord(PRecord):
        field = field(type=CheckedType, factory=lambda x: x)

    evolver = TestRecord._evolver()
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'invalid_field' is not among the specified fields for TestRecord"

def test_set_with_factory_ignore_extra():
    class TestRecord(PRecord):
        field = field(type=CheckedType, factory=lambda x, ignore_extra=False: x)

    evolver = TestRecord._evolver(_ignore_extra=True)
    result = evolver.set('field', 'value')
    assert result is evolver
    assert evolver['field'] == 'value'

def test_set_with_invariant_exception():
    class TestRecord(PRecord):
        field = field(type=CheckedType, factory=lambda x: x, invariant=lambda x: (False, 'error'))

    evolver = TestRecord._evolver()
    result = evolver.set('field', 'value')
    assert result is evolver
    assert evolver._invariant_error_codes == ['error']

def test_set_with_type_error():
    class TestRecord(PRecord):
        field = field(type=CheckedType, factory=lambda x: x)

    evolver = TestRecord._evolver()
    try:
        evolver.set('field', 123)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert str(e) == "Invalid type for field TestRecord.field, was int"


# LLM-generated content at query #3
#--------------------------

```python
def test__new__sets_precord_fields():
    class Parent:
        __metaclass__ = _PRecordMeta
        x = _PField()

    class Child(Parent):
        __metaclass__ = _PRecordMeta
        y = _PField()

    assert '_precord_fields' in Child.__dict__
    assert 'x' in Child._precord_fields
    assert 'y' in Child._precord_fields

def test__new__inherits_fields():
    class Parent:
        __metaclass__ = _PRecordMeta
        x = _PField()

    class Child(Parent):
        __metaclass__ = _PRecordMeta

    assert 'x' in Child._precord_fields

def test__new__sets_precord_invariants():
    def invariant(x):
        return True, ()

    class TestClass:
        __metaclass__ = _PRecordMeta
        __invariant__ = invariant

    assert '_precord_invariants' in TestClass.__dict__
    assert len(TestClass._precord_invariants) == 1

def test__new__inherits_invariants():
    def invariant(x):
        return True, ()

    class Parent:
        __metaclass__ = _PRecordMeta
        __invariant__ = invariant

    class Child(Parent):
        __metaclass__ = _PRecordMeta

    assert len(Child._precord_invariants) == 1

def test__new__sets_precord_mandatory_fields():
    class TestClass:
        __metaclass__ = _PRecordMeta
        x = _PField(mandatory=True)
        y = _PField()

    assert '_precord_mandatory_fields' in TestClass.__dict__
    assert 'x' in TestClass._precord_mandatory_fields
    assert 'y' not in TestClass._precord_mandatory_fields

def test__new__sets_precord_initial_values():
    class TestClass:
        __metaclass__ = _PRecordMeta
        x = _PField(initial=1)
        y = _PField()

    assert '_precord_initial_values' in TestClass.__dict__
    assert TestClass._precord_initial_values['x'] == 1
    assert 'y' not in TestClass._precord_initial_values

def test__new__sets_empty_slots():
    class TestClass:
        __metaclass__ = _PRecordMeta

    assert TestClass.__slots__ == ()


# LLM-generated content at query #4
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_error_codes_or_missing_fields():
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()

    evolver = _PRecordEvolver(MockPRecord, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['field1']

    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ('field1',)
        assert str(e) == 'Field invariant failed'


# LLM-generated content at query #5
#--------------------------

```python
def test_persistent_raises_exception_when_invariant_errors_or_missing_fields():
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()

    evolver = _PRecordEvolver(MockPRecord, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['missing1']

    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ('missing1',)
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #6
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class MockPMap:
        _buckets = None
        _size = 0

    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockPRecord, MockPMap())
    evolver._is_dirty = True
    pm = MockPMap()
    result = evolver.persistent()
    assert isinstance(result, MockPRecord)


# LLM-generated content at query #7
#--------------------------

```python
def test_set_fields_called_in_new():
    class TestMeta(_PRecordMeta):
        def __new__(mcs, name, bases, dct):
            original_fields = dct.get('_precord_fields', None)
            result = super().__new__(mcs, name, bases, dct)
            assert '_precord_fields' in dct
            assert original_fields is None or original_fields != dct['_precord_fields']
            return result

    class TestClass(metaclass=TestMeta):
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_set_with_valid_field_and_factory():
    class MockField:
        type = (int,)
        factory = lambda x: x * 2
        invariant = lambda x: (True, None)

    class MockPRecord:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockPRecord, PMap(), _factory_fields={MockField()})
    evolver.set('field1', 5)
    assert evolver._map['field1'] == 10

def test_set_with_invalid_field_type():
    class MockField:
        type = (int,)
        factory = lambda x: x
        invariant = lambda x: (True, None)

    class MockPRecord:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockPRecord, PMap(), _factory_fields={MockField()})
    with pytest.raises(PTypeError):
        evolver.set('field1', 'not_an_int')

def test_set_with_nonexistent_field():
    class MockPRecord:
        _precord_fields = {}

    evolver = _PRecordEvolver(MockPRecord, PMap())
    with pytest.raises(AttributeError):
        evolver.set('nonexistent_field', 123)

def test_set_with_invariant_failure():
    class MockField:
        type = (int,)
        factory = lambda x: x
        invariant = lambda x: (False, 'INVALID')

    class MockPRecord:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockPRecord, PMap(), _factory_fields={MockField()})
    evolver.set('field1', 5)
    assert evolver._invariant_error_codes == ['INVALID']

def test_set_with_ignore_extra_compliant_field():
    class MockField:
        type = {CheckedType}
        factory = lambda x, ignore_extra=False: x
        invariant = lambda x: (True, None)

    class MockPRecord:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockPRecord, PMap(), _factory_fields={MockField()}, _ignore_extra=True)
    evolver.set('field1', 'value')
    assert evolver._map['field1'] == 'value'

def test_set_with_ignore_extra_non_compliant_field():
    class MockField:
        type = (int,)
        factory = lambda x: x
        invariant = lambda x: (True, None)

    class MockPRecord:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockPRecord, PMap(), _factory_fields={MockField()}, _ignore_extra=True)
    evolver.set('field1', 5)
    assert evolver._map['field1'] == 5


# LLM-generated content at query #9
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    evolver = _PRecordEvolver(TestPRecord, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'value1')
    result = evolver.persistent()
    assert isinstance(result, TestPRecord)
    assert result.field1 == 'value1'

def test_persistent_with_clean_and_cls_instance():
    pm = TestPRecord(field1='value1')
    evolver = _PRecordEvolver(TestPRecord, pm, _factory_fields=None, _ignore_extra=False)
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    evolver = _PRecordEvolver(TestPRecord, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'value1')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestPRecord.mandatory_field' in e.missing_fields

def test_persistent_with_invariant_errors():
    evolver = _PRecordEvolver(TestPRecord, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'invalid_value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0

def test_persistent_with_global_invariant_failure():
    evolver = _PRecordEvolver(TestPRecord, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'value1')
    evolver.set('field2', 'value2')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'Global invariant failed' in str(e)

def test_persistent_with_valid_data():
    evolver = _PRecordEvolver(TestPRecord, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'valid_value')
    evolver.set('mandatory_field', 'required_value')
    result = evolver.persistent()
    assert isinstance(result, TestPRecord)
    assert result.field1 == 'valid_value'
    assert result.mandatory_field == 'required_value'


# LLM-generated content at query #10
#--------------------------

```python
def test_serialize_without_format():
    class TestRecord(PRecord):
        field1 = Field()
        field2 = Field(serializer=lambda x: x.upper())

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert result == {"field1": "value1", "field2": "VALUE2"}

def test_serialize_with_format():
    class TestRecord(PRecord):
        field1 = Field()
        field2 = Field(serializer=lambda x, fmt: x.upper() if fmt == "upper" else x.lower())

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize(format="upper")
    assert result == {"field1": "value1", "field2": "VALUE2"}

def test_serialize_with_no_serializer():
    class TestRecord(PRecord):
        field1 = Field()
        field2 = Field()

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert result == {"field1": "value1", "field2": "value2"}


# LLM-generated content at query #11
#--------------------------

```python
def test_missing_fields_added_when_mandatory_fields_exist():
    class TestRecord:
        _precord_mandatory_fields = {'field1', 'field2'}
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._missing_fields = []
    evolver.persistent()
    assert len(evolver._missing_fields) == 2
    assert 'TestRecord.field1' in evolver._missing_fields
    assert 'TestRecord.field2' in evolver._missing_fields


# LLM-generated content at query #12
#--------------------------

```python
def test_precord_new_with_special_attributes():
    kwargs = {'_precord_size': 2, '_precord_buckets': [(('a', 1),), (('b', 2),)]}
    result = PRecord.__new__(PRecord, **kwargs)
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    result = TestRecord.__new__(TestRecord, a=1, b=2)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    result = TestRecord.__new__(TestRecord, a=1, b=2, _factory_fields={'a'})
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    result = TestRecord.__new__(TestRecord, a=1, b=2, c=3, _ignore_extra=True)
    assert result['a'] == 1
    assert result['b'] == 2
    assert 'c' not in result

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    TestRecord._precord_initial_values = {'a': 10, 'b': 20}
    result = TestRecord.__new__(TestRecord, a=1, b=2)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_callable_initial_values():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    TestRecord._precord_initial_values = {'a': lambda: 10, 'b': lambda: 20}
    result = TestRecord.__new__(TestRecord, a=1, b=2)
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #13
#--------------------------

```python
def test_repr_with_empty_record():
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert repr(record) == "EmptyRecord()"

def test_repr_with_single_field():
    class SingleFieldRecord(PRecord):
        field1 = Field()

    record = SingleFieldRecord(field1="value1")
    assert repr(record) == "SingleFieldRecord(field1='value1')"

def test_repr_with_multiple_fields():
    class MultiFieldRecord(PRecord):
        field1 = Field()
        field2 = Field()
        field3 = Field()

    record = MultiFieldRecord(field1="value1", field2=2, field3=True)
    assert repr(record) == "MultiFieldRecord(field1='value1', field2=2, field3=True)"

def test_repr_with_nested_record():
    class InnerRecord(PRecord):
        inner_field = Field()

    class OuterRecord(PRecord):
        outer_field = Field()
        inner = Field(type=InnerRecord)

    inner = InnerRecord(inner_field="inner_value")
    outer = OuterRecord(outer_field="outer_value", inner=inner)
    assert repr(outer) == "OuterRecord(outer_field='outer_value', inner=InnerRecord(inner_field='inner_value'))"

def test_repr_with_special_characters():
    class SpecialRecord(PRecord):
        field = Field()

    record = SpecialRecord(field="value with spaces and 'quotes'")
    assert repr(record) == "SpecialRecord(field='value with spaces and \\'quotes\\'')"

def test_repr_with_none_value():
    class NoneRecord(PRecord):
        field = Field()

    record = NoneRecord(field=None)
    assert repr(record) == "NoneRecord(field=None)"


# LLM-generated content at query #14
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 2, '_precord_buckets': [('a', 1), ('b', 2)]}
    result = PRecord(**kwargs)
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2

def test_precord_constructor_with_factory_fields():
    kwargs = {'field1': 'value1', '_factory_fields': ['field1']}
    result = PRecord(**kwargs)
    assert result['field1'] == 'value1'

def test_precord_constructor_with_ignore_extra():
    kwargs = {'field1': 'value1', 'extra_field': 'extra_value', '_ignore_extra': True}
    result = PRecord(**kwargs)
    assert result['field1'] == 'value1'
    assert 'extra_field' not in result

def test_precord_constructor_with_initial_values():
    PRecord._precord_initial_values = {'field1': 'default1', 'field2': 'default2'}
    kwargs = {'field1': 'override1'}
    result = PRecord(**kwargs)
    assert result['field1'] == 'override1'
    assert result['field2'] == 'default2'

def test_precord_constructor_with_callable_initial_values():
    PRecord._precord_initial_values = {'field1': lambda: 'computed_value'}
    result = PRecord()
    assert result['field1'] == 'computed_value'

def test_precord_constructor_empty():
    result = PRecord()
    assert len(result) == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_new_with_no_bases_and_empty_dct():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {})
    assert result._precord_fields == {}
    assert result._precord_invariants == ()
    assert result._precord_mandatory_fields == set()
    assert result._precord_initial_values == {}
    assert result.__slots__ == ()

def test_new_with_single_base_and_empty_dct():
    class Base:
        pass
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {})
    assert result._precord_fields == {}
    assert result._precord_invariants == ()
    assert result._precord_mandatory_fields == set()
    assert result._precord_initial_values == {}
    assert result.__slots__ == ()

def test_new_with_fields_in_dct():
    from pyrsistent._field_common import _PField
    field1 = _PField(mandatory=True, initial=1)
    field2 = _PField(mandatory=False, initial=2)
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'field1': field1, 'field2': field2})
    assert result._precord_fields == {'field1': field1, 'field2': field2}
    assert result._precord_mandatory_fields == {'field1'}
    assert result._precord_initial_values == {'field1': 1, 'field2': 2}

def test_new_with_invariant_in_dct():
    def invariant_func():
        return True, "test"
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'__invariant__': invariant_func})
    assert len(result._precord_invariants) == 1
    assert callable(result._precord_invariants[0])

def test_new_with_inherited_fields():
    from pyrsistent._field_common import _PField
    class Base:
        __metaclass__ = _PRecordMeta
        field1 = _PField(mandatory=True, initial=1)
    field2 = _PField(mandatory=False, initial=2)
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {'field2': field2})
    assert 'field1' in result._precord_fields
    assert 'field2' in result._precord_fields
    assert result._precord_mandatory_fields == {'field1'}

def test_new_with_inherited_invariants():
    def base_invariant():
        return True, "base"
    class Base:
        __metaclass__ = _PRecordMeta
        __invariant__ = base_invariant
    def derived_invariant():
        return True, "derived"
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {'__invariant__': derived_invariant})
    assert len(result._precord_invariants) == 2

def test_new_with_non_callable_invariant():
    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'__invariant__': "not callable"})
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'


# LLM-generated content at query #16
#--------------------------

```python
def test_repr_empty_record():
    class EmptyRecord(PRecord):
        pass
    record = EmptyRecord()
    assert repr(record) == "EmptyRecord()"

def test_repr_single_field():
    class SingleFieldRecord(PRecord):
        field1 = field()
    record = SingleFieldRecord(field1="value1")
    assert repr(record) == "SingleFieldRecord(field1='value1')"

def test_repr_multiple_fields():
    class MultiFieldRecord(PRecord):
        field1 = field()
        field2 = field()
        field3 = field()
    record = MultiFieldRecord(field1="value1", field2=2, field3=True)
    assert repr(record) == "MultiFieldRecord(field1='value1', field2=2, field3=True)"

def test_repr_with_quoted_strings():
    class StringRecord(PRecord):
        name = field()
    record = StringRecord(name="John's record")
    assert repr(record) == "StringRecord(name=\"John's record\")"

def test_repr_with_nested_structure():
    class NestedRecord(PRecord):
        data = field()
    inner_dict = {"key": "value"}
    record = NestedRecord(data=inner_dict)
    assert repr(record) == "NestedRecord(data={'key': 'value'})"


# LLM-generated content at query #17
#--------------------------

```python
def test_serialize_returns_dict_with_serialized_fields():
    class TestRecord(PRecord):
        field1 = field(serializer=lambda x: f"serialized_{x}")
        field2 = field(serializer=lambda x: x * 2)

    record = TestRecord(field1="value1", field2=5)
    result = record.serialize()
    assert result == {"field1": "serialized_value1", "field2": 10}


# LLM-generated content at query #18
#--------------------------

```python
def test_set_fields_called_with_correct_arguments():
    dct = {}
    bases = (object,)
    name = '_precord_fields'
    set_fields(dct, bases, name=name)
    assert '_precord_fields' in dct


# LLM-generated content at query #19
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver.set('test_key', 'test_value')
    result = evolver.persistent()
    assert isinstance(result, TestClass)


# LLM-generated content at query #20
#--------------------------

```python
def test_precord_initial_values_used():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': lambda: 2}
        _precord_fields = {'a': None, 'b': None}

    result = TestRecord()
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #21
#--------------------------

```python
def test_persistent_predicate_false():
    evolver = _PRecordEvolver(cls=type('MockCls', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []}), original_pmap=PMap())
    evolver._is_dirty = False
    evolver._buckets = (None,)
    evolver._size = 0
    result = evolver.persistent()
    assert result == evolver


# LLM-generated content at query #22
#--------------------------

```python
def test_repr_format():
    class TestRecord(PRecord):
        pass

    record = TestRecord(a=1, b="test")
    result = repr(record)
    assert result.startswith("TestRecord(")
    assert "a=1" in result
    assert 'b="test"' in result
    assert result.endswith(")")


# LLM-generated content at query #23
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty():
    class TestPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestPRecord, PMap())
    evolver._set_dirty(True)
    result = evolver.persistent()
    assert isinstance(result, TestPRecord)


# LLM-generated content at query #24
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 5, '_precord_buckets': [1, 2, 3, 4, 5]}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result._precord_size == 5
    assert result._precord_buckets == [1, 2, 3, 4, 5]

def test_precord_constructor_with_factory_fields():
    kwargs = {'field1': 'value1', '_factory_fields': ['field1']}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'

def test_precord_constructor_with_ignore_extra():
    kwargs = {'field1': 'value1', 'extra_field': 'extra_value', '_ignore_extra': True}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'
    assert 'extra_field' not in result

def test_precord_constructor_with_initial_values():
    kwargs = {'field1': 'value1'}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'

def test_precord_constructor_with_callable_initial_values():
    kwargs = {'field1': lambda: 'value1'}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'


# LLM-generated content at query #25
#--------------------------

```python
def test_precord_metaclass_initialization():
    class TestRecord(metaclass=_PRecordMeta):
        pass

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert hasattr(TestRecord, '__slots__')


# LLM-generated content at query #26
#--------------------------

```python
def test_persistent_raises_when_invariant_error_codes_or_missing_fields():
    class MockPMap:
        _buckets = None
        _size = 0

    class MockField:
        invariant = lambda self, value: (True, None)

    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockClass, MockPMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['field1']

    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()

    assert exc_info.value.invariant_errors == ('error1',)
    assert exc_info.value.missing_fields == ('field1',)
    assert exc_info.value.msg == 'Field invariant failed'


# LLM-generated content at query #27
#--------------------------

```python
def test_precord_initial_values_are_used():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': lambda: 2}

    result = TestRecord()
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #28
#--------------------------

```python
def test_new_with_empty_bases_and_dct():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {})
    assert result._precord_fields == {}
    assert result._precord_invariants == ()
    assert result._precord_mandatory_fields == set()
    assert result._precord_initial_values == {}
    assert result.__slots__ == ()

def test_new_with_fields_in_dct():
    dct = {'field1': _PField(mandatory=True), 'field2': _PField(initial=42)}
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), dct)
    assert result._precord_fields == {'field1': _PField(mandatory=True), 'field2': _PField(initial=42)}
    assert result._precord_mandatory_fields == {'field1'}
    assert result._precord_initial_values == {'field2': 42}

def test_new_with_inherited_fields():
    class Base:
        field1 = _PField(mandatory=True)
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {})
    assert result._precord_fields == {'field1': _PField(mandatory=True)}
    assert result._precord_mandatory_fields == {'field1'}

def test_new_with_invariant():
    def test_inv():
        return True, "OK"
    dct = {'__invariant__': test_inv}
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), dct)
    assert len(result._precord_invariants) == 1
    assert callable(result._precord_invariants[0])

def test_new_with_inherited_invariant():
    class Base:
        __invariant__ = lambda: (True, "OK")
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {})
    assert len(result._precord_invariants) == 1
    assert callable(result._precord_invariants[0])

def test_new_with_non_callable_invariant_raises_type_error():
    dct = {'__invariant__': "not callable"}
    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), dct)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test_new_with_multiple_invariants():
    def inv1():
        return True, "OK1"
    def inv2():
        return True, "OK2"
    dct = {'__invariant__': [inv1, inv2]}
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), dct)
    assert len(result._precord_invariants) == 2
    assert all(callable(inv) for inv in result._precord_invariants)

def test_new_with_inherited_multiple_invariants():
    class Base1:
        __invariant__ = lambda: (True, "OK1")
    class Base2:
        __invariant__ = lambda: (True, "OK2")
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base1, Base2), {})
    assert len(result._precord_invariants) == 2
    assert all(callable(inv) for inv in result._precord_invariants)


# LLM-generated content at query #29
#--------------------------

```python
def test_missing_fields_added_when_mandatory_fields_exist():
    class TestRecord:
        _precord_mandatory_fields = {'field1', 'field2'}
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._missing_fields = []
    evolver.persistent()
    assert len(evolver._missing_fields) == 2
    assert 'TestRecord.field1' in evolver._missing_fields
    assert 'TestRecord.field2' in evolver._missing_fields


# LLM-generated content at query #30
#--------------------------

```python
def test_precord_new_with_special_attributes():
    cls = type('TestRecord', (PRecord,), {})
    result = cls(_precord_size=2, _precord_buckets=[[('a', 1)], [('b', 2)]])
    assert result == {'a': 1, 'b': 2}

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2, _factory_fields={'a', 'b'})
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2, ignore_extra=True)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': 2}
    result = TestRecord()
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_initial_values_and_kwargs():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': 2}
    result = TestRecord(a=3)
    assert result == {'a': 3, 'b': 2}


# LLM-generated content at query #31
#--------------------------

```python
def test_invariant_exception_raised_when_error_codes_or_missing_fields():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []

    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #32
#--------------------------

```python
def test_serialize_returns_dict():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=1, field2=2)
    result = record.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #33
#--------------------------

```python
def test_serialize_returns_dict():
    class TestRecord(PRecord):
        field1 = field(serializer=lambda x, fmt: x)
        field2 = field(serializer=lambda x, fmt: x)

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #34
#--------------------------

```python
def test_field_exists_in_precord_fields():
    class MockField:
        pass

    class MockDestinationCls:
        _precord_fields = {'key': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    field = evolver._destination_cls._precord_fields.get('key')
    assert field is not None


# LLM-generated content at query #35
#--------------------------

```python
def test_set_with_existing_field():
    class MockField:
        pass

    class MockDestinationCls:
        _precord_fields = {'existing_field': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    field = evolver._destination_cls._precord_fields.get('existing_field')

    assert field is not None


# LLM-generated content at query #36
#--------------------------

```python
def test_repr_returns_correct_string():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=10, field2="test")
    assert repr(record) == "TestRecord(field1=10, field2='test')"


# LLM-generated content at query #37
#--------------------------

```python
def test_persistent_returns_pmap_when_not_dirty_and_instance_of_cls():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    assert evolver.persistent() is original_pmap

def test_persistent_returns_new_instance_when_dirty():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result != original_pmap

def test_persistent_raises_invariant_exception_for_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'mandatory_field'}, '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'MockClass.mandatory_field' in exc_info.value.missing_fields

def test_persistent_raises_invariant_exception_for_field_invariant_failures():
    def failing_invariant(value):
        return (False, 'INVARIANT_FAILED')

    field = type('MockField', (), {'invariant': failing_invariant, 'factory': lambda x: x})()
    cls = type('MockClass', (), {'_precord_fields': {'field': field}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('field', 'value')
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'INVARIANT_FAILED' in exc_info.value.invariant_errors

def test_persistent_raises_invariant_exception_for_global_invariant_failures():
    def failing_invariant(subject):
        return (False, 'GLOBAL_INVARIANT_FAILED')

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'GLOBAL_INVARIANT_FAILED' in exc_info.value.invariant_errors


# LLM-generated content at query #38
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': None, '_size': 0})()
    evolver = _PRecordEvolver.__new__(_PRecordEvolver)
    evolver._destination_cls = cls
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    evolver._factory_fields = None
    evolver._ignore_extra = False
    evolver.__init__(cls, original_pmap)
    evolver._map = type('MockMap', (), {'_buckets': 'new_buckets', '_size': 1})()
    evolver.is_dirty = lambda: True
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result._precord_buckets == 'new_buckets'
    assert result._precord_size == 1

def test_persistent_with_clean_and_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': 'old_buckets', '_size': 0})()
    evolver = _PRecordEvolver.__new__(_PRecordEvolver)
    evolver._destination_cls = cls
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    evolver._factory_fields = None
    evolver._ignore_extra = False
    evolver.__init__(cls, original_pmap)
    evolver._map = original_pmap
    evolver.is_dirty = lambda: False
    result = evolver.persistent()
    assert result is original_pmap

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'field1', 'field2'}, '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': None, '_size': 0, 'keys': lambda: ['field1']})()
    evolver = _PRecordEvolver.__new__(_PRecordEvolver)
    evolver._destination_cls = cls
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    evolver._factory_fields = None
    evolver._ignore_extra = False
    evolver.__init__(cls, original_pmap)
    evolver._map = original_pmap
    evolver.is_dirty = lambda: True
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockClass.field2' in e.missing_fields

def test_persistent_with_invariant_errors():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': None, '_size': 0})()
    evolver = _PRecordEvolver.__new__(_PRecordEvolver)
    evolver._destination_cls = cls
    evolver._invariant_error_codes = ['error1', 'error2']
    evolver._missing_fields = []
    evolver._factory_fields = None
    evolver._ignore_extra = False
    evolver.__init__(cls, original_pmap)
    evolver._map = original_pmap
    evolver.is_dirty = lambda: True
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')

def test_persistent_with_global_invariant_failure():
    def failing_invariant(subject):
        return (False, 'global_error')

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    original_pmap = type('MockPMap', (), {'_buckets': None, '_size': 0})()
    evolver = _PRecordEvolver.__new__(_PRecordEvolver)
    evolver._destination_cls = cls
    evolver._invariant_error_codes = []
    evolver._missing_fields = []
    evolver._factory_fields = None
    evolver._ignore_extra = False
    evolver.__init__(cls, original_pmap)
    evolver._map = original_pmap
    evolver.is_dirty = lambda: True
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)


# LLM-generated content at query #39
#--------------------------

```python
def test_missing_mandatory_fields_are_added_to_missing_fields():
    class TestRecord:
        _precord_mandatory_fields = {'field1', 'field2'}
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._missing_fields = []
    result = PMap({'field1': 'value1'})
    evolver.persistent = lambda: result

    evolver.persistent()
    assert evolver._missing_fields == ('TestRecord.field2',)


# LLM-generated content at query #40
#--------------------------

```python
def test_persistent_with_dirty_state_and_non_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result['key'] == 'value'

def test_persistent_with_clean_state_and_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    pm = cls(_precord_buckets={}, _precord_size=0)
    evolver = _PRecordEvolver(cls, pm)
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'mandatory_field'}, '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'mandatory_field' in e.missing_fields

def test_persistent_with_invariant_errors():
    def failing_invariant(value):
        return False, 'INVARIANT_FAILED'

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('key', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVARIANT_FAILED' in e.invariant_errors

def test_persistent_with_global_invariant_failure():
    def global_invariant(subject):
        return False, 'GLOBAL_INVARIANT_FAILED'

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [global_invariant]})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'Global invariant failed' in str(e)


# LLM-generated content at query #41
#--------------------------

```python
def test_set_fields_called_before_store_invariants():
    class TestMeta(_PRecordMeta):
        def __new__(mcs, name, bases, dct):
            assert '_precord_fields' in dct
            return super().__new__(mcs, name, bases, dct)

    class TestRecord(metaclass=TestMeta):
        pass


# LLM-generated content at query #42
#--------------------------

```python
def test_set_fields_called_in_new():
    class TestMeta(_PRecordMeta):
        def __new__(mcs, name, bases, dct):
            original_fields = dct.get('_precord_fields', None)
            result = super().__new__(mcs, name, bases, dct)
            assert '_precord_fields' in dct
            assert original_fields is None or dct['_precord_fields'] != original_fields
            return result

    class TestClass(metaclass=TestMeta):
        pass


# LLM-generated content at query #43
#--------------------------

```python
def test_persistent_predicate_false():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    assert not (evolver.is_dirty() or not isinstance(evolver.persistent(), cls))


# LLM-generated content at query #44
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    cls = type('TestPRecord', (PRecord,), {})
    kwargs = {'_precord_size': 2, '_precord_buckets': pvector([[(1, 'a')], [(2, 'b')]])}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result._buckets == pvector([[(1, 'a')], [(2, 'b')]])

def test_precord_new_without_precord_size_and_buckets():
    cls = type('TestPRecord', (PRecord,), {})
    cls._precord_fields = {'a': None, 'b': None}
    cls._precord_initial_values = {}
    kwargs = {'a': 1, 'b': 2}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_factory_fields():
    cls = type('TestPRecord', (PRecord,), {})
    cls._precord_fields = {'a': None, 'b': None}
    cls._precord_initial_values = {}
    kwargs = {'a': 1, 'b': 2, '_factory_fields': {'a'}}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_ignore_extra():
    cls = type('TestPRecord', (PRecord,), {})
    cls._precord_fields = {'a': None, 'b': None}
    cls._precord_initial_values = {}
    kwargs = {'a': 1, 'b': 2, 'c': 3, '_ignore_extra': True}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2
    assert 'c' not in result

def test_precord_new_with_callable_initial_values():
    cls = type('TestPRecord', (PRecord,), {})
    cls._precord_fields = {'a': None, 'b': None}
    cls._precord_initial_values = {'a': lambda: 1, 'b': lambda: 2}
    kwargs = {}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_overriding_initial_values():
    cls = type('TestPRecord', (PRecord,), {})
    cls._precord_fields = {'a': None, 'b': None}
    cls._precord_initial_values = {'a': lambda: 1, 'b': lambda: 2}
    kwargs = {'a': 10}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result['a'] == 10
    assert result['b'] == 2


# LLM-generated content at query #45
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': 'mock_buckets', '_size': 'mock_size'})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: True
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result._precord_buckets == 'mock_buckets'
    assert result._precord_size == 'mock_size'

def test_persistent_with_clean_and_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': 'mock_buckets', '_size': 'mock_size'})()
    pm = type('MockPM', (cls,), {'_buckets': 'mock_buckets', '_size': 'mock_size'})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: False
    evolver.persistent = lambda: pm
    result = evolver.persistent()
    assert result == pm

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'field1'}, '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': 'mock_buckets', '_size': 'mock_size'})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: True
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('MockClass.field1',)

def test_persistent_with_invariant_errors():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': 'mock_buckets', '_size': 'mock_size'})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._invariant_error_codes = ['error1']
    evolver._is_dirty = lambda: True
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)

def test_persistent_with_global_invariant_failure():
    def failing_invariant(subject):
        return (False, 'global_error')

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    original_pmap = type('MockPMap', (), {'_buckets': 'mock_buckets', '_size': 'mock_size'})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: True
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)


# LLM-generated content at query #46
#--------------------------

```python
def test_persistent_with_mandatory_fields_missing():
    class TestClass:
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'TestClass'

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver.set('field1', 'value1')
    try:
        evolver.persistent()
    except InvariantException as e:
        assert 'TestClass.field2' in e.missing_fields


# LLM-generated content at query #47
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 1, '_precord_buckets': [('key', 'value')]}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['key'] == 'value'

def test_precord_constructor_with_factory_fields():
    kwargs = {'field1': 'value1', '_factory_fields': ['field1']}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'

def test_precord_constructor_with_ignore_extra():
    kwargs = {'field1': 'value1', 'extra_field': 'extra_value', '_ignore_extra': True}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'
    assert 'extra_field' not in result

def test_precord_constructor_with_initial_values():
    PRecord._precord_initial_values = {'field1': 'default_value'}
    kwargs = {'field1': 'new_value'}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'new_value'

def test_precord_constructor_with_callable_initial_values():
    PRecord._precord_initial_values = {'field1': lambda: 'default_value'}
    kwargs = {}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'default_value'


# LLM-generated content at query #48
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_default_values():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_callable_defaults():
    class TestRecord(PRecord):
        x = lambda: 1
        y = lambda: 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_extra_fields_ignored():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20, z=30, _ignore_extra=True)
    assert record.x == 10
    assert record.y == 20
    assert "z" not in record

def test_precord_constructor_with_extra_fields_not_ignored():
    class TestRecord(PRecord):
        x = 1
        y = 2

    try:
        record = TestRecord(x=10, y=20, z=30)
        assert False, "Expected an error for extra fields"
    except Exception:
        pass

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_factory_fields=["x", "y"], x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_internal_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_precord_size=2, _precord_buckets=[("x", 10), ("y", 20)])
    assert record.x == 10
    assert record.y == 20


# LLM-generated content at query #49
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=10, field2="test")
    assert record.field1 == 10
    assert record.field2 == "test"

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord.create({"field1": 10, "field2": "test", "extra": "ignored"}, ignore_extra=True)
    assert record.field1 == 10
    assert record.field2 == "test"
    assert "extra" not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord.create({"field1": 10, "field2": "test"}, _factory_fields=["field1"])
    assert record.field1 == 10
    assert record.field2 == "test"

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        field1 = field(initial=5)
        field2 = field()

    record = TestRecord(field2="test")
    assert record.field1 == 5
    assert record.field2 == "test"

def test_precord_constructor_with_callable_initial():
    class TestRecord(PRecord):
        field1 = field(initial=lambda: 10)
        field2 = field()

    record = TestRecord(field2="test")
    assert record.field1 == 10
    assert record.field2 == "test"

def test_precord_constructor_with_pickle_support():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=10, field2="test")
    restored = pickle.loads(pickle.dumps(record))
    assert restored.field1 == 10
    assert restored.field2 == "test"


# LLM-generated content at query #50
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord.create({'x': 10, 'y': 20, 'z': 30}, ignore_extra=True)
    assert record.x == 10
    assert record.y == 20
    assert 'z' not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord.create({'x': 10, 'y': 20}, _factory_fields=['x'])
    assert record.x == 10
    assert record.y == 2

def test_precord_constructor_with_callable_initial_values():
    class TestRecord(PRecord):
        x = lambda: 1
        y = 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_pickle_support():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    restored_record = pickle.loads(pickle.dumps(record))
    assert restored_record.x == 10
    assert restored_record.y == 20

def test_precord_constructor_with_precord_size_and_buckets():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_precord_size=2, _precord_buckets=[('x', 10), ('y', 20)])
    assert record.x == 10
    assert record.y == 20


# LLM-generated content at query #51
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 1, '_precord_buckets': [('field1', 'value1')]}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'

def test_precord_constructor_with_factory_fields():
    kwargs = {'field1': 'value1', '_factory_fields': ['field1']}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'

def test_precord_constructor_with_ignore_extra():
    kwargs = {'field1': 'value1', 'extra_field': 'extra_value', '_ignore_extra': True}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'
    assert 'extra_field' not in result

def test_precord_constructor_with_initial_values():
    PRecord._precord_initial_values = {'field1': 'initial_value'}
    kwargs = {'field1': 'updated_value'}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'updated_value'

def test_precord_constructor_with_callable_initial_values():
    PRecord._precord_initial_values = {'field1': lambda: 'computed_value'}
    result = PRecord()
    assert isinstance(result, PRecord)
    assert result['field1'] == 'computed_value'


# LLM-generated content at query #52
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class MockPMap:
        _buckets = None
        _size = 0

    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockPRecord, MockPMap())
    evolver._is_dirty = True
    pm = MockPMap()
    result = evolver.persistent()
    assert isinstance(result, MockPRecord)


# LLM-generated content at query #53
#--------------------------

```python
def test_predicate_line_1():
    assert '_precord_size' in {'_precord_size': 1, '_precord_buckets': []} and '_precord_buckets' in {'_precord_size': 1, '_precord_buckets': []}


# LLM-generated content at query #54
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    result = PRecord(_precord_size=5, _precord_buckets=[1, 2, 3, 4, 5])
    assert isinstance(result, PRecord)
    assert result._precord_size == 5
    assert result._precord_buckets == [1, 2, 3, 4, 5]

def test_precord_constructor_with_factory_fields():
    result = PRecord(_factory_fields=["field1", "field2"], field1="value1", field2="value2")
    assert isinstance(result, PRecord)
    assert result["field1"] == "value1"
    assert result["field2"] == "value2"

def test_precord_constructor_with_ignore_extra():
    result = PRecord(_ignore_extra=True, field1="value1", extra_field="extra_value")
    assert isinstance(result, PRecord)
    assert result["field1"] == "value1"
    assert "extra_field" not in result

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {"field1": None, "field2": None}
        _precord_initial_values = {"field1": "default1", "field2": "default2"}

    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert result["field1"] == "default1"
    assert result["field2"] == "default2"

def test_precord_constructor_with_initial_values_and_kwargs():
    class TestRecord(PRecord):
        _precord_fields = {"field1": None, "field2": None}
        _precord_initial_values = {"field1": "default1", "field2": "default2"}

    result = TestRecord(field1="new_value")
    assert isinstance(result, TestRecord)
    assert result["field1"] == "new_value"
    assert result["field2"] == "default2"

def test_precord_constructor_with_callable_initial_values():
    class TestRecord(PRecord):
        _precord_fields = {"field1": None, "field2": None}
        _precord_initial_values = {"field1": lambda: "computed1", "field2": "default2"}

    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert result["field1"] == "computed1"
    assert result["field2"] == "default2"


# LLM-generated content at query #55
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 1, '_precord_buckets': [('key', 'value')]}
    result = PRecord(**kwargs)
    assert result == PRecord(1, [('key', 'value')])

def test_precord_constructor_with_factory_fields():
    kwargs = {'field1': 'value1', '_factory_fields': ['field1']}
    result = PRecord(**kwargs)
    assert result.field1 == 'value1'

def test_precord_constructor_with_ignore_extra():
    kwargs = {'field1': 'value1', 'extra_field': 'extra_value', '_ignore_extra': True}
    result = PRecord(**kwargs)
    assert result.field1 == 'value1'
    assert 'extra_field' not in result

def test_precord_constructor_with_initial_values():
    PRecord._precord_initial_values = {'field1': 'initial_value'}
    kwargs = {'field1': 'new_value'}
    result = PRecord(**kwargs)
    assert result.field1 == 'new_value'

def test_precord_constructor_with_callable_initial_values():
    PRecord._precord_initial_values = {'field1': lambda: 'computed_value'}
    result = PRecord()
    assert result.field1 == 'computed_value'


# LLM-generated content at query #56
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    cls = PRecord
    kwargs = {'_precord_size': 2, '_precord_buckets': [None, [(1, 2)]]}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result._buckets == [None, [(1, 2)]]

def test_precord_new_without_precord_size_and_buckets():
    cls = PRecord
    kwargs = {'a': 1, 'b': 2}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_factory_fields():
    cls = PRecord
    kwargs = {'_factory_fields': ['a'], 'a': 1, 'b': 2}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_ignore_extra():
    cls = PRecord
    kwargs = {'_ignore_extra': True, 'a': 1, 'b': 2, 'c': 3}
    result = cls.__new__(cls, **kwargs)
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #57
#--------------------------

```python
def test_predicate_evaluates_to_false():
    class TestRecord(PRecord):
        pass

    kwargs = {'_precord_size': 10, '_precord_buckets': None}
    assert not ('_precord_size' in kwargs and '_precord_buckets' in kwargs)


# LLM-generated content at query #58
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_callable_defaults():
    class TestRecord(PRecord):
        x = lambda: 1
        y = 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_extra_fields_ignored():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20, z=30, _ignore_extra=True)
    assert record.x == 10
    assert record.y == 20
    assert 'z' not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, _factory_fields={'y': 20})
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_internal_params():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_precord_size=2, _precord_buckets=[('x', 10), ('y', 20)])
    assert record.x == 10
    assert record.y == 20


# LLM-generated content at query #59
#--------------------------

```python
def test_persistent_predicate_false():
    cls = type('MockClass', (), {'_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver._destination_cls = cls
    evolver.is_dirty = lambda: False
    evolver.persistent = lambda: cls()
    result = evolver.persistent()
    assert isinstance(result, cls)


# LLM-generated content at query #60
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty():
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockPRecord, PMap())
    evolver.set('test_key', 'test_value')
    pm = evolver.persistent()
    assert isinstance(pm, MockPRecord)


# LLM-generated content at query #61
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty():
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockPRecord, PMap())
    evolver.set('new_field', 'value')
    result = evolver.persistent()
    assert isinstance(result, MockPRecord)


# LLM-generated content at query #62
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        field1 = Field()
        field2 = Field()

    record = TestRecord(field1=10, field2="test")
    assert record.field1 == 10
    assert record.field2 == "test"

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        field1 = Field()

    record = TestRecord.create({"field1": 10, "extra_field": 20}, ignore_extra=True)
    assert record.field1 == 10
    assert "extra_field" not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        field1 = Field()
        field2 = Field()

    record = TestRecord.create({"field1": 10}, _factory_fields=["field2"])
    assert record.field1 == 10
    assert record.field2 is None

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        field1 = Field(initial=100)
        field2 = Field()

    record = TestRecord(field2="test")
    assert record.field1 == 100
    assert record.field2 == "test"

def test_precord_constructor_with_callable_initial():
    class TestRecord(PRecord):
        field1 = Field(initial=lambda: 42)
        field2 = Field()

    record = TestRecord(field2="test")
    assert record.field1 == 42
    assert record.field2 == "test"

def test_precord_constructor_with_internal_params():
    class TestRecord(PRecord):
        field1 = Field()

    record = TestRecord(_precord_size=1, _precord_buckets=[("field1", 10)])
    assert record.field1 == 10


# LLM-generated content at query #63
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_callable_defaults():
    class TestRecord(PRecord):
        x = lambda: 1
        y = 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_extra_fields_ignored():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20, z=30, _ignore_extra=True)
    assert record.x == 10
    assert record.y == 20
    assert 'z' not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, _factory_fields={'x': 5})
    assert record.x == 10

def test_precord_constructor_with_internal_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_precord_size=2, _precord_buckets=[('x', 10), ('y', 20)])
    assert record.x == 10
    assert record.y == 20


# LLM-generated content at query #64
#--------------------------

```python
def test_precord_new_with_special_attributes():
    result = PRecord(_precord_size=2, _precord_buckets=[[('a', 1)], [('b', 2)]])
    assert result == {'a': 1, 'b': 2}

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': 2}
    result = TestRecord()
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2, _factory_fields={'a', 'b'})
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2, _ignore_extra=True)
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #65
#--------------------------

```python
def test_precord_new_with_special_attributes():
    cls = PRecord
    size = 5
    buckets = [(('a', 1), ('b', 2)), None, None, None, None]
    result = cls.__new__(cls, _precord_size=size, _precord_buckets=buckets)
    assert result._size == size
    assert result._buckets == pvector().extend(buckets)

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    TestRecord._precord_initial_values = {'a': 1, 'b': 2}
    result = TestRecord.__new__(TestRecord, a=10, b=20)
    assert result['a'] == 10
    assert result['b'] == 20

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    result = TestRecord.__new__(TestRecord, _factory_fields={'a'}, a=10, b=20)
    assert result['a'] == 10
    assert result['b'] == 20

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None}
    result = TestRecord.__new__(TestRecord, _ignore_extra=True, a=10, b=20)
    assert result['a'] == 10
    assert 'b' not in result

def test_precord_new_with_callable_initial_values():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    TestRecord._precord_initial_values = {'a': lambda: 1, 'b': 2}
    result = TestRecord.__new__(TestRecord)
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #66
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 2, '_precord_buckets': [('a', 1), ('b', 2)]}
    result = PRecord(**kwargs)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_constructor_with_factory_fields():
    kwargs = {'field1': 'value1', '_factory_fields': ['field1']}
    result = PRecord(**kwargs)
    assert result['field1'] == 'value1'

def test_precord_constructor_with_ignore_extra():
    kwargs = {'field1': 'value1', 'extra_field': 'extra_value', '_ignore_extra': True}
    result = PRecord(**kwargs)
    assert result['field1'] == 'value1'
    assert 'extra_field' not in result

def test_precord_constructor_with_initial_values():
    PRecord._precord_initial_values = {'field1': 'default1', 'field2': 'default2'}
    kwargs = {'field1': 'new_value'}
    result = PRecord(**kwargs)
    assert result['field1'] == 'new_value'
    assert result['field2'] == 'default2'

def test_precord_constructor_with_callable_initial_values():
    PRecord._precord_initial_values = {'field1': lambda: 'computed_value'}
    result = PRecord()
    assert result['field1'] == 'computed_value'


# LLM-generated content at query #67
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 2, '_precord_buckets': [('a', 1), ('b', 2)]}
    result = PRecord(**kwargs)
    assert result == PRecord(a=1, b=2)

def test_precord_constructor_with_factory_fields():
    kwargs = {'a': 1, 'b': 2, '_factory_fields': ['a']}
    result = PRecord(**kwargs)
    assert result.a == 1
    assert result.b == 2

def test_precord_constructor_with_ignore_extra():
    kwargs = {'a': 1, 'b': 2, 'c': 3, '_ignore_extra': True}
    result = PRecord(**kwargs)
    assert result.a == 1
    assert result.b == 2
    assert 'c' not in result

def test_precord_constructor_with_initial_values():
    kwargs = {'a': 1, 'b': 2}
    result = PRecord(**kwargs)
    assert result.a == 1
    assert result.b == 2

def test_precord_constructor_with_callable_initial_values():
    kwargs = {'a': 1, 'b': 2}
    result = PRecord(**kwargs)
    assert result.a == 1
    assert result.b == 2

def test_precord_constructor_with_evolver():
    e = _PRecordEvolver(PRecord, pmap(pre_size=2), _factory_fields=None, _ignore_extra=False)
    e['a'] = 1
    e['b'] = 2
    result = e.persistent()
    assert result.a == 1
    assert result.b == 2


# LLM-generated content at query #68
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=10, field2="test")
    assert record.field1 == 10
    assert record.field2 == "test"

def test_precord_constructor_with_extra_fields_ignored():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord.create({"field1": 10, "field2": "test", "extra": "ignored"}, ignore_extra=True)
    assert record.field1 == 10
    assert record.field2 == "test"
    assert "extra" not in record

def test_precord_constructor_with_callable_initial_values():
    class TestRecord(PRecord):
        field1 = field(initial=10)
        field2 = field(initial=lambda: "default")

    record = TestRecord()
    assert record.field1 == 10
    assert record.field2 == "default"

def test_precord_constructor_with_existing_instance():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    original = TestRecord(field1=10, field2="test")
    record = TestRecord.create(original)
    assert record.field1 == 10
    assert record.field2 == "test"


# LLM-generated content at query #69
#--------------------------

```python
def test_precord_new_with_special_attributes():
    kwargs = {'_precord_size': 2, '_precord_buckets': [(('a', 1),), (('b', 2),)]}
    result = PRecord.__new__(PRecord, **kwargs)
    assert result == {'a': 1, 'b': 2}
    assert len(result) == 2

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    result = TestRecord.__new__(TestRecord, a=1, b=2)
    assert result == {'a': 1, 'b': 2}
    assert len(result) == 2

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    TestRecord._precord_initial_values = {'a': 10, 'b': lambda: 20}
    result = TestRecord.__new__(TestRecord)
    assert result == {'a': 10, 'b': 20}
    assert len(result) == 2

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    result = TestRecord.__new__(TestRecord, _factory_fields={'a'}, a=1, b=2)
    assert result == {'a': 1, 'b': 2}
    assert len(result) == 2

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None}
    result = TestRecord.__new__(TestRecord, _ignore_extra=True, a=1, b=2)
    assert result == {'a': 1}
    assert len(result) == 1


# LLM-generated content at query #70
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=10, field2="test")
    assert record.field1 == 10
    assert record.field2 == "test"

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord.create({"field1": 10, "field2": "test", "extra": "ignored"}, ignore_extra=True)
    assert record.field1 == 10
    assert record.field2 == "test"
    assert "extra" not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord.create({"field1": 10, "field2": "test"}, _factory_fields=["field1"])
    assert record.field1 == 10
    assert record.field2 == "test"

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        field1 = field(initial=5)
        field2 = field()

    record = TestRecord(field2="test")
    assert record.field1 == 5
    assert record.field2 == "test"

def test_precord_constructor_with_callable_initial():
    class TestRecord(PRecord):
        field1 = field(initial=lambda: 10)
        field2 = field()

    record = TestRecord(field2="test")
    assert record.field1 == 10
    assert record.field2 == "test"

def test_precord_constructor_with_existing_instance():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    original = TestRecord(field1=10, field2="test")
    record = TestRecord.create(original)
    assert record.field1 == 10
    assert record.field2 == "test"


# LLM-generated content at query #71
#--------------------------

```python
def test_persistent_with_clean_pmap_of_correct_type():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    pm = type('MockPMap', (), {
        '_buckets': None,
        '_size': 0,
        '__class__': cls
    })()
    evolver = _PRecordEvolver(cls, pm)
    assert evolver.persistent() is pm


# LLM-generated content at query #72
#--------------------------

```python
def test_pr_new_with_special_attributes():
    result = PRecord.__new__(PRecord, _precord_size=5, _precord_buckets=[1, 2, 3, 4, 5])
    assert isinstance(result, PRecord)

def test_pr_new_without_special_attributes():
    result = PRecord.__new__(PRecord, a=1, b=2)
    assert isinstance(result, PRecord)
    assert result['a'] == 1
    assert result['b'] == 2

def test_pr_new_with_factory_fields():
    result = PRecord.__new__(PRecord, a=1, b=2, _factory_fields=['a'])
    assert isinstance(result, PRecord)
    assert result['a'] == 1
    assert result['b'] == 2

def test_pr_new_with_ignore_extra():
    result = PRecord.__new__(PRecord, a=1, b=2, _ignore_extra=True)
    assert isinstance(result, PRecord)
    assert result['a'] == 1
    assert result['b'] == 2

def test_pr_new_with_initial_values():
    result = PRecord.__new__(PRecord, a=1, b=2)
    assert isinstance(result, PRecord)
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #73
#--------------------------

```python
def test_serialize_returns_dict():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #74
#--------------------------

```python
def test_persistent_no_changes():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    result = evolver.persistent()
    assert result == original_pmap

def test_persistent_with_changes():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert result != original_pmap
    assert result['key'] == 'value'

def test_persistent_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'mandatory_field'}, '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'mandatory_field' in e.missing_fields

def test_persistent_invariant_failure():
    def failing_invariant(value):
        return False, 'INVARIANT_FAILED'

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVARIANT_FAILED' in e.invariant_errors

def test_persistent_global_invariant_failure():
    def failing_global_invariant(subject):
        return False, 'GLOBAL_INVARIANT_FAILED'

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_global_invariant]})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'Global invariant failed' in str(e)

def test_persistent_field_invariant_failure():
    field = type('MockField', (), {'factory': lambda x: x, 'invariant': lambda x: (False, 'FIELD_INVARIANT_FAILED')})
    cls = type('MockClass', (), {'_precord_fields': {'field': field}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('field', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'FIELD_INVARIANT_FAILED' in e.invariant_errors


# LLM-generated content at query #75
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_error_codes_or_missing_fields():
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockPRecord, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['missing1']

    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ('missing1',)
        assert str(e) == 'Field invariant failed'


# LLM-generated content at query #76
#--------------------------

```python
def test_precord_initial_values_are_used():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': lambda: 2}

    result = TestRecord()
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #77
#--------------------------

```python
def test_repr_returns_correct_string():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=1, field2="test")
    assert repr(record) == "TestRecord(field1=1, field2='test')"


# LLM-generated content at query #78
#--------------------------

```python
def test_precord_constructor_with_valid_kwargs():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_callable_initial_values():
    class TestRecord(PRecord):
        x = lambda: 1
        y = lambda: 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord._factory_fields({'x': 10, 'y': 20})
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord._ignore_extra(True, x=10, y=20, z=30)
    assert record.x == 10
    assert record.y == 20
    assert 'z' not in record

def test_precord_constructor_with_precord_size_and_buckets():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_precord_size=2, _precord_buckets=((('x', 1), ('y', 2)),))
    assert record.x == 1
    assert record.y == 2


# LLM-generated content at query #79
#--------------------------

```python
def test_precord_metaclass_initialization():
    class TestRecord(metaclass=_PRecordMeta):
        __invariant__ = lambda self: True
        field1 = _PField(mandatory=True, initial=1)
        field2 = _PField(mandatory=False, initial=PFIELD_NO_INITIAL)

    assert '_precord_fields' in TestRecord.__dict__
    assert '_precord_invariants' in TestRecord.__dict__
    assert '_precord_mandatory_fields' in TestRecord.__dict__
    assert '_precord_initial_values' in TestRecord.__dict__
    assert '__slots__' in TestRecord.__dict__
    assert TestRecord.__slots__ == ()


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    buckets = pvector().extend([None] * 8)
    result = PRecord.__new__(PRecord, _precord_size=2, _precord_buckets=buckets)
    assert isinstance(result, PRecord)
    assert result._size == 2
    assert result._buckets == buckets

def test_precord_new_without_precord_size_and_buckets():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2)
    assert isinstance(result, TestRecord)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_factory_fields_and_ignore_extra():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2, _factory_fields={'a'}, _ignore_extra=True)
    assert isinstance(result, TestRecord)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 10, 'b': lambda: 20}
    result = TestRecord()
    assert result['a'] == 10
    assert result['b'] == 20

def test_precord_new_with_initial_values_and_kwargs():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 10, 'b': lambda: 20}
    result = TestRecord(a=1, b=2)
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_serialize_without_custom_serializer():
    class TestRecord(PRecord):
        field1 = Field()
        field2 = Field()

    record = TestRecord(field1="value1", field2="value2")
    assert record.serialize() == {"field1": "value1", "field2": "value2"}

def test_serialize_with_custom_serializer():
    class TestRecord(PRecord):
        field1 = Field(serializer=lambda x: x.upper())
        field2 = Field(serializer=lambda x: str(x).upper())

    record = TestRecord(field1="value1", field2=123)
    assert record.serialize() == {"field1": "VALUE1", "field2": "123"}

def test_serialize_with_format_parameter():
    class TestRecord(PRecord):
        field1 = Field(serializer=lambda x, fmt: f"{fmt}:{x}" if fmt else x)
        field2 = Field(serializer=lambda x, fmt: f"{fmt}:{x}" if fmt else x)

    record = TestRecord(field1="value1", field2="value2")
    assert record.serialize(format="custom") == {"field1": "custom:value1", "field2": "custom:value2"}

def test_serialize_empty_record():
    class TestRecord(PRecord):
        pass

    record = TestRecord()
    assert record.serialize() == {}


# LLM-generated content at query #3
#--------------------------

```python
def test_persistent_with_dirty_and_non_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('test_key', 'test_value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result.get('test_key') == 'test_value'

def test_persistent_with_clean_and_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    pm = cls(_precord_buckets={}, _precord_size=0)
    evolver = _PRecordEvolver(cls, pm)
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'mandatory_field'}, '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'mandatory_field' in e.missing_fields

def test_persistent_with_invariant_error_codes():
    def failing_invariant(value):
        return (False, 'INVARIANT_FAILED')

    field = type('MockField', (), {'invariant': failing_invariant, 'factory': lambda x: x})()
    cls = type('MockClass', (), {'_precord_fields': {'test_field': field}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('test_field', 'test_value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVARIANT_FAILED' in e.invariant_errors

def test_persistent_with_global_invariant_failure():
    def global_invariant(subject):
        return (False, 'GLOBAL_INVARIANT_FAILED')

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [global_invariant]})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'GLOBAL_INVARIANT_FAILED' in e.invariant_errors


# LLM-generated content at query #4
#--------------------------

```python
def test_repr_empty():
    class EmptyRecord(PRecord):
        pass
    assert repr(EmptyRecord()) == "EmptyRecord()"

def test_repr_single_field():
    class SingleFieldRecord(PRecord):
        field = field()
    record = SingleFieldRecord(field=42)
    assert repr(record) == "SingleFieldRecord(field=42)"

def test_repr_multiple_fields():
    class MultiFieldRecord(PRecord):
        field1 = field()
        field2 = field()
    record = MultiFieldRecord(field1=1, field2="test")
    assert repr(record) == "MultiFieldRecord(field1=1, field2='test')"

def test_repr_with_complex_values():
    class ComplexRecord(PRecord):
        list_field = field()
        dict_field = field()
    record = ComplexRecord(list_field=[1, 2, 3], dict_field={"a": 1})
    assert repr(record) == "ComplexRecord(list_field=[1, 2, 3], dict_field={'a': 1})"

def test_repr_with_nested_record():
    class InnerRecord(PRecord):
        value = field()
    class OuterRecord(PRecord):
        inner = field()
    inner = InnerRecord(value=10)
    outer = OuterRecord(inner=inner)
    assert repr(outer) == "OuterRecord(inner=InnerRecord(value=10))"


# LLM-generated content at query #5
#--------------------------

```python
def test_persistent_returns_pmap_when_not_dirty_and_instance_of_cls():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    result = evolver.persistent()
    assert result is original_pmap

def test_persistent_creates_new_instance_when_dirty():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result != original_pmap

def test_persistent_creates_new_instance_when_not_instance_of_cls():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, cls)

def test_persistent_raises_invariant_exception_for_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'mandatory_field'}, '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    with pytest.raises(InvariantException):
        evolver.persistent()

def test_persistent_raises_invariant_exception_for_invariant_errors():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._invariant_error_codes = ['error_code']
    with pytest.raises(InvariantException):
        evolver.persistent()

def test_persistent_raises_invariant_exception_for_missing_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._missing_fields = ['missing_field']
    with pytest.raises(InvariantException):
        evolver.persistent()

def test_persistent_checks_global_invariants():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [lambda x: (False, 'global_error')]})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #6
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver.set('test_key', 'test_value')

    assert evolver.is_dirty() is True
    result = evolver.persistent()
    assert isinstance(result, TestClass) is True


# LLM-generated content at query #7
#--------------------------

```python
def test__new__sets_fields_and_invariants():
    class Parent:
        __metaclass__ = _PRecordMeta
        x = _PField()
        __invariant__ = lambda self: True

    class Child(Parent):
        __metaclass__ = _PRecordMeta
        y = _PField()

    assert '_precord_fields' in Child.__dict__
    assert '_precord_invariants' in Child.__dict__
    assert 'x' in Child._precord_fields
    assert 'y' in Child._precord_fields
    assert len(Child._precord_invariants) == 1

def test__new__inherits_fields_and_invariants():
    class Parent:
        __metaclass__ = _PRecordMeta
        x = _PField()
        __invariant__ = lambda self: True

    class Child(Parent):
        __metaclass__ = _PRecordMeta
        y = _PField()

    assert 'x' in Child._precord_fields
    assert 'y' in Child._precord_fields
    assert len(Child._precord_invariants) == 1

def test__new__sets_mandatory_fields():
    class Test:
        __metaclass__ = _PRecordMeta
        x = _PField(mandatory=True)
        y = _PField(mandatory=False)

    assert Test._precord_mandatory_fields == {'x'}

def test__new__sets_initial_values():
    class Test:
        __metaclass__ = _PRecordMeta
        x = _PField(initial=1)
        y = _PField()

    assert Test._precord_initial_values == {'x': 1}

def test__new__sets_empty_slots():
    class Test:
        __metaclass__ = _PRecordMeta
        x = _PField()

    assert Test.__slots__ == ()

def test__new__raises_type_error_for_non_callable_invariant():
    try:
        class Test:
            __metaclass__ = _PRecordMeta
            __invariant__ = "not callable"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        assert False, "Expected TypeError"


# LLM-generated content at query #8
#--------------------------

```python
def test_set_fields_called_in_new():
    class TestClass(metaclass=_PRecordMeta):
        pass
    assert '_precord_fields' in TestClass.__dict__


# LLM-generated content at query #9
#--------------------------

```python
def test_new_with_no_bases_and_empty_dct():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {})
    assert result._precord_fields == {}
    assert result._precord_invariants == ()
    assert result._precord_mandatory_fields == set()
    assert result._precord_initial_values == {}
    assert result.__slots__ == ()

def test_new_with_single_base_and_inherited_fields():
    class Base:
        __metaclass__ = _PRecordMeta
        x = _PField(mandatory=True)
        y = _PField(initial=1)

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {})
    assert result._precord_fields == {'x': Base._precord_fields['x'], 'y': Base._precord_fields['y']}
    assert result._precord_invariants == Base._precord_invariants
    assert result._precord_mandatory_fields == {'x'}
    assert result._precord_initial_values == {'y': 1}
    assert result.__slots__ == ()

def test_new_with_multiple_bases_and_inherited_fields():
    class Base1:
        __metaclass__ = _PRecordMeta
        x = _PField(mandatory=True)

    class Base2:
        __metaclass__ = _PRecordMeta
        y = _PField(initial=1)

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base1, Base2), {})
    assert result._precord_fields == {'x': Base1._precord_fields['x'], 'y': Base2._precord_fields['y']}
    assert result._precord_invariants == Base1._precord_invariants + Base2._precord_invariants
    assert result._precord_mandatory_fields == {'x'}
    assert result._precord_initial_values == {'y': 1}
    assert result.__slots__ == ()

def test_new_with_invariant_function():
    def test_invariant(obj):
        return True, ()

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'__invariant__': test_invariant})
    assert result._precord_invariants == (wrap_invariant(test_invariant),)
    assert result._precord_fields == {}
    assert result._precord_mandatory_fields == set()
    assert result._precord_initial_values == {}
    assert result.__slots__ == ()

def test_new_with_invalid_invariant():
    with pytest.raises(TypeError):
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'__invariant__': 'not_callable'})

def test_new_with_field_in_dct():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'x': _PField(mandatory=True)})
    assert result._precord_fields == {'x': result._precord_fields['x']}
    assert 'x' not in result.__dict__
    assert result._precord_mandatory_fields == {'x'}
    assert result._precord_initial_values == {}
    assert result.__slots__ == ()

def test_new_with_field_and_initial_value():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'x': _PField(initial=42)})
    assert result._precord_fields == {'x': result._precord_fields['x']}
    assert result._precord_initial_values == {'x': 42}
    assert result._precord_mandatory_fields == set()
    assert result.__slots__ == ()

def test_new_with_no_initial_value_field():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'x': _PField()})
    assert result._precord_fields == {'x': result._precord_fields['x']}
    assert result._precord_initial_values == {}
    assert result._precord_mandatory_fields == set()
    assert result.__slots__ == ()


# LLM-generated content at query #10
#--------------------------

```python
def test__new__sets_fields_correctly():
    class Base:
        pass

    class TestRecord(metaclass=_PRecordMeta):
        field1 = _PField(mandatory=True)
        field2 = _PField(initial=42)

    assert '_precord_fields' in TestRecord.__dict__
    assert 'field1' in TestRecord._precord_fields
    assert 'field2' in TestRecord._precord_fields
    assert TestRecord._precord_fields['field1'].mandatory is True
    assert TestRecord._precord_fields['field2'].initial == 42

def test__new__inherits_fields():
    class Base(metaclass=_PRecordMeta):
        base_field = _PField()

    class Derived(Base):
        derived_field = _PField()

    assert 'base_field' in Derived._precord_fields
    assert 'derived_field' in Derived._precord_fields

def test__new__sets_mandatory_fields():
    class TestRecord(metaclass=_PRecordMeta):
        mandatory_field = _PField(mandatory=True)
        optional_field = _PField()

    assert TestRecord._precord_mandatory_fields == {'mandatory_field'}

def test__new__sets_initial_values():
    class TestRecord(metaclass=_PRecordMeta):
        field_with_initial = _PField(initial=10)
        field_without_initial = _PField()

    assert TestRecord._precord_initial_values == {'field_with_initial': 10}

def test__new__sets_empty_slots():
    class TestRecord(metaclass=_PRecordMeta):
        pass

    assert TestRecord.__slots__ == ()

def test__new__stores_invariants():
    def test_invariant(obj):
        return True, "OK"

    class TestRecord(metaclass=_PRecordMeta):
        __invariant__ = test_invariant

    assert hasattr(TestRecord, '_precord_invariants')
    assert len(TestRecord._precord_invariants) == 1
    assert callable(TestRecord._precord_invariants[0])

def test__new__inherits_invariants():
    def base_invariant(obj):
        return True, "Base OK"

    def derived_invariant(obj):
        return True, "Derived OK"

    class Base(metaclass=_PRecordMeta):
        __invariant__ = base_invariant

    class Derived(Base):
        __invariant__ = derived_invariant

    assert len(Derived._precord_invariants) == 2
    assert all(callable(inv) for inv in Derived._precord_invariants)

def test__new__raises_type_error_for_non_callable_invariant():
    try:
        class TestRecord(metaclass=_PRecordMeta):
            __invariant__ = "not callable"

        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'


# LLM-generated content at query #11
#--------------------------

```python
def test_persistent_with_dirty_and_not_instance():
    evolver = _PRecordEvolver(MockClass, PMap())
    evolver.set('field1', 'value1')
    result = evolver.persistent()
    assert isinstance(result, MockClass)
    assert result._precord_buckets == evolver._buckets
    assert result._precord_size == evolver._size

def test_persistent_without_dirty_and_is_instance():
    pm = MockClass(_precord_buckets={'field1': 'value1'}, _precord_size=1)
    evolver = _PRecordEvolver(MockClass, pm)
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    MockClass._precord_mandatory_fields = {'field1', 'field2'}
    evolver = _PRecordEvolver(MockClass, PMap())
    evolver.set('field1', 'value1')
    try:
        evolver.persistent()
    except InvariantException as e:
        assert 'MockClass.field2' in e.missing_fields

def test_persistent_with_invariant_errors():
    MockClass._precord_fields['field1'].invariant = lambda x: (False, 'error_code')
    evolver = _PRecordEvolver(MockClass, PMap())
    evolver.set('field1', 'invalid_value')
    try:
        evolver.persistent()
    except InvariantException as e:
        assert 'error_code' in e.invariant_errors

def test_persistent_with_global_invariant_failure():
    MockClass._precord_invariants = [lambda x: (False, 'global_error')]
    evolver = _PRecordEvolver(MockClass, PMap())
    evolver.set('field1', 'value1')
    try:
        evolver.persistent()
    except InvariantException as e:
        assert 'Global invariant failed' in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_set_with_valid_field_and_factory():
    field = type('Field', (), {'type': (int,), 'factory': lambda x: x, 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}})
    evolver = _PRecordEvolver(cls, PMap())
    result = evolver.set('field1', 10)
    assert result['field1'] == 10

def test_set_with_invalid_type():
    field = type('Field', (), {'type': (int,), 'factory': lambda x: x, 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.set('field1', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.message == "Invalid type for field TestClass.field1, was str"

def test_set_with_invariant_failure():
    field = type('Field', (), {'type': (int,), 'factory': lambda x: x, 'invariant': lambda x: (False, 'ERROR')})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}})
    evolver = _PRecordEvolver(cls, PMap())
    result = evolver.set('field1', 10)
    assert result._invariant_error_codes == ['ERROR']

def test_set_with_nonexistent_field():
    cls = type('TestClass', (), {'_precord_fields': {}})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.set('nonexistent', 10)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'nonexistent' is not among the specified fields for TestClass"

def test_set_with_ignore_extra_true_and_compliant_factory():
    field = type('Field', (), {'type': {CheckedType}, 'factory': lambda x, ignore_extra=False: x, 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}})
    evolver = _PRecordEvolver(cls, PMap(), _ignore_extra=True)
    result = evolver.set('field1', 10)
    assert result['field1'] == 10

def test_set_with_ignore_extra_false_and_compliant_factory():
    field = type('Field', (), {'type': {CheckedType}, 'factory': lambda x, ignore_extra=False: x, 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}})
    evolver = _PRecordEvolver(cls, PMap(), _ignore_extra=False)
    result = evolver.set('field1', 10)
    assert result['field1'] == 10

def test_set_with_factory_field_not_in_factory_fields():
    field = type('Field', (), {'type': (int,), 'factory': lambda x: x, 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}})
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=[])
    result = evolver.set('field1', 10)
    assert result['field1'] == 10

def test_set_with_factory_field_in_factory_fields():
    field = type('Field', (), {'type': (int,), 'factory': lambda x: x + 1, 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}})
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=[field])
    result = evolver.set('field1', 10)
    assert result['field1'] == 11


# LLM-generated content at query #13
#--------------------------

```python
def test_serialize_returns_dict():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_5_evaluates_to_false():
    class TestRecord(PRecord):
        pass

    result = TestRecord.__new__(TestRecord, a=1, b=2)
    assert result is not None


# LLM-generated content at query #15
#--------------------------

```python
def test__new_creates_class_with_correct_attributes():
    class TestClass(metaclass=_PRecordMeta):
        __invariant__ = lambda self: True
        field1 = _PField(mandatory=True, initial=1)
        field2 = _PField(mandatory=False, initial=PFIELD_NO_INITIAL)
        field3 = _PField(mandatory=True, initial=2)

    assert hasattr(TestClass, '_precord_fields')
    assert hasattr(TestClass, '_precord_invariants')
    assert hasattr(TestClass, '_precord_mandatory_fields')
    assert hasattr(TestClass, '_precord_initial_values')
    assert hasattr(TestClass, '__slots__')
    assert TestClass._precord_mandatory_fields == {'field1', 'field3'}
    assert TestClass._precord_initial_values == {'field1': 1, 'field3': 2}
    assert TestClass.__slots__ == ()


# LLM-generated content at query #16
#--------------------------

```python
def test_repr_format():
    class TestRecord(PRecord):
        pass

    record = TestRecord(a=1, b="test")
    result = repr(record)
    assert result.startswith("TestRecord(")
    assert "a=1" in result
    assert 'b="test"' in result
    assert result.endswith(")")


# LLM-generated content at query #17
#--------------------------

```python
def test__new__sets_precord_fields():
    class TestClass(metaclass=_PRecordMeta):
        field1 = _PField()
        field2 = _PField()

    assert '_precord_fields' in TestClass.__dict__
    assert 'field1' in TestClass._precord_fields
    assert 'field2' in TestClass._precord_fields

def test__new__sets_precord_mandatory_fields():
    class TestClass(metaclass=_PRecordMeta):
        mandatory_field = _PField(mandatory=True)
        optional_field = _PField(mandatory=False)

    assert '_precord_mandatory_fields' in TestClass.__dict__
    assert 'mandatory_field' in TestClass._precord_mandatory_fields
    assert 'optional_field' not in TestClass._precord_mandatory_fields

def test__new__sets_precord_initial_values():
    class TestClass(metaclass=_PRecordMeta):
        field_with_initial = _PField(initial=42)
        field_without_initial = _PField()

    assert '_precord_initial_values' in TestClass.__dict__
    assert TestClass._precord_initial_values['field_with_initial'] == 42
    assert 'field_without_initial' not in TestClass._precord_initial_values

def test__new__sets_empty_slots():
    class TestClass(metaclass=_PRecordMeta):
        pass

    assert TestClass.__slots__ == ()

def test__new__inherits_fields():
    class Parent(metaclass=_PRecordMeta):
        parent_field = _PField()

    class Child(Parent, metaclass=_PRecordMeta):
        child_field = _PField()

    assert 'parent_field' in Child._precord_fields
    assert 'child_field' in Child._precord_fields

def test__new__inherits_invariants():
    def parent_invariant(obj):
        return True, ()

    class Parent(metaclass=_PRecordMeta):
        __invariant__ = parent_invariant

    def child_invariant(obj):
        return True, ()

    class Child(Parent, metaclass=_PRecordMeta):
        __invariant__ = child_invariant

    assert len(Child._precord_invariants) == 2
    assert Child._precord_invariants[0](None) == (True, ())
    assert Child._precord_invariants[1](None) == (True, ())

def test__new__wraps_invariants():
    def test_invariant(obj):
        return [(True, ()), (True, ())]

    class TestClass(metaclass=_PRecordMeta):
        __invariant__ = test_invariant

    result = TestClass._precord_invariants[0](None)
    assert result == (True, ())

def test__new__raises_type_error_for_non_callable_invariant():
    try:
        class TestClass(metaclass=_PRecordMeta):
            __invariant__ = "not callable"

        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"


# LLM-generated content at query #18
#--------------------------

```python
def test_repr():
    class TestRecord(PRecord):
        field1 = 1
        field2 = "test"

    record = TestRecord(field1=2, field2="updated")
    assert repr(record) == "TestRecord(field1=2, field2='updated')"


# LLM-generated content at query #19
#--------------------------

```python
def test_precord_initial_values_are_used():
    class TestRecord(PRecord):
        _precord_fields = {'a': None, 'b': None}
        _precord_initial_values = {'a': 1, 'b': 2}

    result = TestRecord()
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_line_1():
    class TestClass(metaclass=_PRecordMeta):
        pass

    assert isinstance(TestClass, _PRecordMeta)


# LLM-generated content at query #21
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_error_codes_or_missing_fields_exist():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['field1']

    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()

    assert exc_info.value.invariant_errors == ('error1',)
    assert exc_info.value.missing_fields == ('field1',)
    assert exc_info.value.message == 'Field invariant failed'


# LLM-generated content at query #22
#--------------------------

```python
def test_repr_format():
    class TestRecord(PRecord):
        pass

    record = TestRecord(a=1, b="test")
    result = repr(record)
    assert result.startswith("TestRecord(") and result.endswith(")")
    assert "a=1" in result and "b='test'" in result


# LLM-generated content at query #23
#--------------------------

```python
def test_persistent_returns_same_instance_when_not_dirty_and_correct_type():
    cls = _PRecordEvolver._destination_cls
    evolver = _PRecordEvolver(cls, pmap(), _factory_fields=None, _ignore_extra=False)
    pm = evolver.persistent()
    assert pm is evolver
    assert isinstance(pm, cls)

def test_persistent_creates_new_instance_when_dirty():
    cls = _PRecordEvolver._destination_cls
    evolver = _PRecordEvolver(cls, pmap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field', 'value')
    pm = evolver.persistent()
    assert pm is not evolver
    assert isinstance(pm, cls)

def test_persistent_creates_new_instance_when_not_correct_type():
    cls = _PRecordEvolver._destination_cls
    evolver = _PRecordEvolver(cls, pmap(), _factory_fields=None, _ignore_extra=False)
    pm = evolver.persistent()
    assert pm is not evolver
    assert isinstance(pm, cls)

def test_persistent_raises_invariant_exception_when_mandatory_fields_missing():
    cls = _PRecordEvolver._destination_cls
    cls._precord_mandatory_fields = {'field1', 'field2'}
    evolver = _PRecordEvolver(cls, pmap(), _factory_fields=None, _ignore_extra=False)
    with pytest.raises(InvariantException):
        evolver.persistent()

def test_persistent_raises_invariant_exception_when_invariant_error_codes_exist():
    cls = _PRecordEvolver._destination_cls
    evolver = _PRecordEvolver(cls, pmap(), _factory_fields=None, _ignore_extra=False)
    evolver._invariant_error_codes = ['error1', 'error2']
    with pytest.raises(InvariantException):
        evolver.persistent()

def test_persistent_raises_invariant_exception_when_global_invariants_fail():
    cls = _PRecordEvolver._destination_cls
    cls._precord_invariants = [lambda x: (False, 'global_error')]
    evolver = _PRecordEvolver(cls, pmap(), _factory_fields=None, _ignore_extra=False)
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #24
#--------------------------

```python
def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass

    result = TestRecord.__new__(TestRecord)
    assert not ('_precord_size' in result and '_precord_buckets' in result)


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_false():
    class TestRecord(PRecord):
        pass

    result = TestRecord.__new__(TestRecord)
    assert result is not None


# LLM-generated content at query #26
#--------------------------

```python
def test_serialize_returns_dict():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #27
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockPRecord, PMap())
    evolver._is_dirty = True
    pm = evolver.persistent()

    assert isinstance(pm, MockPRecord)


# LLM-generated content at query #28
#--------------------------

```python
def test_serialize_returns_dict():
    class TestRecord(PRecord):
        pass

    record = TestRecord()
    result = record.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #29
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class TestClass:
        _precord_mandatory_fields = set()
        _precord_invariants = []
        _precord_fields = {}
        __name__ = 'TestClass'

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver.set('test_key', 'test_value')
    pm = evolver.persistent()
    assert isinstance(pm, TestClass)


# LLM-generated content at query #30
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    cls = PRecord
    size = 2
    buckets = [None, [(1, 'a')]]
    result = cls.__new__(cls, _precord_size=size, _precord_buckets=buckets)
    assert isinstance(result, PRecord)
    assert result._size == size
    assert result._buckets == buckets

def test_precord_new_without_precord_size_and_buckets():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    result = TestRecord.__new__(TestRecord, field1=1, field2=2)
    assert isinstance(result, TestRecord)
    assert result.field1 == 1
    assert result.field2 == 2

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    factory_fields = {TestRecord._precord_fields['field1']}
    result = TestRecord.__new__(TestRecord, field1=1, field2=2, _factory_fields=factory_fields)
    assert isinstance(result, TestRecord)
    assert result.field1 == 1
    assert result.field2 == 2

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    result = TestRecord.__new__(TestRecord, field1=1, field2=2, extra_field=3, _ignore_extra=True)
    assert isinstance(result, TestRecord)
    assert result.field1 == 1
    assert result.field2 == 2
    assert 'extra_field' not in result

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        field1 = field(initial=10)
        field2 = field(initial=20)

    result = TestRecord.__new__(TestRecord)
    assert isinstance(result, TestRecord)
    assert result.field1 == 10
    assert result.field2 == 20

def test_precord_new_with_initial_values_and_kwargs():
    class TestRecord(PRecord):
        field1 = field(initial=10)
        field2 = field(initial=20)

    result = TestRecord.__new__(TestRecord, field1=100)
    assert isinstance(result, TestRecord)
    assert result.field1 == 100
    assert result.field2 == 20


# LLM-generated content at query #31
#--------------------------

```python
def test_persistent_with_mandatory_fields_missing():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('field1', 'value1')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecord.field2' in e.missing_fields


# LLM-generated content at query #32
#--------------------------

```python
def test_set_with_valid_field_and_factory():
    class MockField:
        type = (int,)
        factory = lambda x: x * 2
        invariant = lambda x: (True, None)

    class MockDestinationCls:
        _precord_fields = {'test_field': MockField}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    result = evolver.set('test_field', 5)
    assert result['test_field'] == 10

def test_set_with_invalid_type():
    class MockField:
        type = (int,)
        factory = lambda x: x
        invariant = lambda x: (True, None)

    class MockDestinationCls:
        _precord_fields = {'test_field': MockField}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    try:
        evolver.set('test_field', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.message == "Invalid type for field MockDestinationCls.test_field, was str"

def test_set_with_invariant_failure():
    class MockField:
        type = (int,)
        factory = lambda x: x
        invariant = lambda x: (False, 'test_error') if x < 0 else (True, None)

    class MockDestinationCls:
        _precord_fields = {'test_field': MockField}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    result = evolver.set('test_field', -1)
    assert result._invariant_error_codes == ['test_error']

def test_set_with_nonexistent_field():
    class MockDestinationCls:
        _precord_fields = {}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    try:
        evolver.set('nonexistent_field', 123)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'nonexistent_field' is not among the specified fields for MockDestinationCls"

def test_set_with_factory_field_not_in_factory_fields():
    class MockField:
        type = (int,)
        factory = lambda x: x * 2
        invariant = lambda x: (True, None)

    class MockDestinationCls:
        _precord_fields = {'test_field': MockField}

    evolver = _PRecordEvolver(MockDestinationCls, PMap(), _factory_fields=[])
    result = evolver.set('test_field', 5)
    assert result['test_field'] == 5

def test_set_with_ignore_extra_and_compliant_factory():
    class MockField:
        type = {CheckedType}
        factory = lambda x, ignore_extra=False: x
        invariant = lambda x: (True, None)

    class MockDestinationCls:
        _precord_fields = {'test_field': MockField}

    evolver = _PRecordEvolver(MockDestinationCls, PMap(), _ignore_extra=True)
    result = evolver.set('test_field', {'key': 'value'})
    assert result['test_field'] == {'key': 'value'}

def test_set_with_ignore_extra_and_non_compliant_factory():
    class MockField:
        type = (int,)
        factory = lambda x: x
        invariant = lambda x: (True, None)

    class MockDestinationCls:
        _precord_fields = {'test_field': MockField}

    evolver = _PRecordEvolver(MockDestinationCls, PMap(), _ignore_extra=True)
    result = evolver.set('test_field', 5)
    assert result['test_field'] == 5


# LLM-generated content at query #33
#--------------------------

```python
def test_precord_new_with_special_attributes():
    result = PRecord(_precord_size=2, _precord_buckets=[[('a', 1)], [('b', 2)]])
    assert result == {'a': 1, 'b': 2}
    assert len(result) == 2

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass

    result = TestRecord(a=1, b=2)
    assert result == {'a': 1, 'b': 2}
    assert len(result) == 2

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': lambda: 2}

    result = TestRecord()
    assert result == {'a': 1, 'b': 2}
    assert len(result) == 2

def test_precord_new_with_initial_values_and_kwargs():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': 2}

    result = TestRecord(b=3, c=4)
    assert result == {'a': 1, 'b': 3, 'c': 4}
    assert len(result) == 3

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass

    result = TestRecord(a=1, b=2, _factory_fields={'a', 'b'})
    assert result == {'a': 1, 'b': 2}
    assert len(result) == 2

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass

    result = TestRecord(a=1, b=2, _ignore_extra=True)
    assert result == {'a': 1, 'b': 2}
    assert len(result) == 2


# LLM-generated content at query #34
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_error_codes_or_missing_fields():
    class MockClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockClass, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['field1']

    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ('field1',)
        assert str(e) == 'Field invariant failed'


# LLM-generated content at query #35
#--------------------------

```python
def test_persistent_with_dirty_and_non_instance():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    evolver.set('field1', 'value1')
    result = evolver.persistent()
    assert isinstance(result, TestPRecord)
    assert result['field1'] == 'value1'

def test_persistent_with_clean_and_instance():
    original = TestPRecord(field1='value1')
    evolver = _PRecordEvolver(TestPRecord, original._map)
    result = evolver.persistent()
    assert result is original

def test_persistent_with_missing_mandatory_fields():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'TestPRecord.mandatory_field' in exc_info.value.missing_fields

def test_persistent_with_invariant_errors():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    evolver.set('field_with_invariant', 'invalid_value')
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'INVALID_VALUE' in exc_info.value.invariant_errors

def test_persistent_with_global_invariant_failure():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    evolver.set('field1', 'value1')
    evolver.set('field2', 'value2')
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'GLOBAL_INVARIANT_FAILED' in exc_info.value.invariant_errors


# LLM-generated content at query #36
#--------------------------

```python
def test_precord_initial_values_are_used():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': lambda: 2}

    result = TestRecord()
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #37
#--------------------------

```python
def test_precord_initial_values_used():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': lambda: 2}

    result = TestRecord()
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #38
#--------------------------

```python
def test_invariant_exception_raised_when_error_codes_or_missing_fields():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['field1']

    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ('field1',)
        assert str(e) == 'Field invariant failed'


# LLM-generated content at query #39
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    result = TestRecord(x=10, y=20)
    assert result.x == 10
    assert result.y == 20

def test_precord_constructor_with_default_values():
    class TestRecord(PRecord):
        x = 1
        y = 2

    result = TestRecord()
    assert result.x == 1
    assert result.y == 2

def test_precord_constructor_with_callable_defaults():
    class TestRecord(PRecord):
        x = lambda: 1
        y = lambda: 2

    result = TestRecord()
    assert result.x == 1
    assert result.y == 2

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    result = TestRecord(_factory_fields={'x': 100}, y=20)
    assert result.x == 100
    assert result.y == 20

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        x = 1
        y = 2

    result = TestRecord(x=10, y=20, z=30, _ignore_extra=True)
    assert result.x == 10
    assert result.y == 20
    assert 'z' not in result

def test_precord_constructor_with_internal_params():
    class TestRecord(PRecord):
        x = 1
        y = 2

    result = TestRecord(_precord_size=2, _precord_buckets={'x': 10, 'y': 20})
    assert result.x == 10
    assert result.y == 20


# LLM-generated content at query #40
#--------------------------

```python
def test_persistent_with_mandatory_fields_missing():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._missing_fields = []

    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields
        assert 'TestRecord.field2' in e.missing_fields


# LLM-generated content at query #41
#--------------------------

```python
def test_persistent_with_dirty_and_valid_fields():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('field1', 'value1')
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['field1'] == 'value1'

def test_persistent_with_clean_and_valid_fields():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap({'field1': 'value1'}))
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['field1'] == 'value1'

def test_persistent_with_missing_mandatory_fields():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1'}
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'TestRecord.field1' in exc_info.value.missing_fields

def test_persistent_with_invariant_error():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'invariant': lambda x: (False, 'error_code')})()}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('field1', 'value1')
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'error_code' in exc_info.value.invariant_errors

def test_persistent_with_global_invariant_error():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = [lambda x: (False, 'global_error')]

    evolver = _PRecordEvolver(TestRecord, PMap())
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'global_error' in exc_info.value.invariant_errors


# LLM-generated content at query #42
#--------------------------

```python
def test_persistent_raises_when_invariant_errors_or_missing_fields():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['field1']

    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #43
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result._precord_buckets == evolver._buckets
    assert result._precord_size == evolver._size

def test_persistent_with_clean_and_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    pm = cls(_precord_buckets=PMap()._buckets, _precord_size=PMap()._size)
    evolver = _PRecordEvolver(cls, pm)
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'mandatory_field'}, '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('MockClass.mandatory_field',)

def test_persistent_with_invariant_errors():
    def failing_invariant(subject):
        return (False, 'INVARIANT_FAILED'),

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('INVARIANT_FAILED',)

def test_persistent_with_global_invariant_failure():
    def failing_global_invariant(subject):
        return False, 'GLOBAL_INVARIANT_FAILED'

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_global_invariant]})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.message == 'Global invariant failed'
        assert e.invariant_errors == ('GLOBAL_INVARIANT_FAILED',)


# LLM-generated content at query #44
#--------------------------

```python
def test_predicate_line_1_false():
    class TestRecord(PRecord):
        pass

    result = TestRecord.__new__(TestRecord)
    assert isinstance(result, PRecord)


# LLM-generated content at query #45
#--------------------------

```python
def test_field_exists_in_precord_fields():
    class MockField:
        pass

    class MockDestinationCls:
        _precord_fields = {'key': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    field = evolver._destination_cls._precord_fields.get('key')

    assert field is not None


# LLM-generated content at query #46
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        x = int
        y = str

    record = TestRecord(x=10, y="hello")
    assert record.x == 10
    assert record.y == "hello"
    assert isinstance(record, TestRecord)

def test_precord_constructor_with_default_values():
    class TestRecord(PRecord):
        x = int
        y = str

    record = TestRecord(x=10)
    assert record.x == 10
    assert record.y is None

def test_precord_constructor_with_callable_default():
    class TestRecord(PRecord):
        x = int
        y = lambda: "default"

    record = TestRecord(x=10)
    assert record.x == 10
    assert record.y == "default"

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        x = int
        y = str

    record = TestRecord(x=10, y="hello", z="extra", _ignore_extra=True)
    assert record.x == 10
    assert record.y == "hello"
    assert "z" not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = int
        y = str

    record = TestRecord(x=10, _factory_fields={"y": "default"})
    assert record.x == 10
    assert record.y == "default"

def test_precord_constructor_with_internal_fields():
    class TestRecord(PRecord):
        x = int
        y = str

    record = TestRecord(_precord_size=2, _precord_buckets={"x": 10, "y": "hello"})
    assert record.x == 10
    assert record.y == "hello"


# LLM-generated content at query #47
#--------------------------

```python
def test_set_fields_called_before_store_invariants():
    class Meta(_PRecordMeta):
        def __new__(mcs, name, bases, dct):
            original_set_fields = set_fields
            original_store_invariants = store_invariants
            set_fields_called = []
            store_invariants_called = []

            def mock_set_fields(*args, **kwargs):
                set_fields_called.append(True)
                return original_set_fields(*args, **kwargs)

            def mock_store_invariants(*args, **kwargs):
                store_invariants_called.append(True)
                return original_store_invariants(*args, **kwargs)

            import pyrsistent._field_common as field_common
            import pyrsistent._checked_types as checked_types
            field_common.set_fields = mock_set_fields
            checked_types.store_invariants = mock_store_invariants

            try:
                result = super(Meta, mcs).__new__(mcs, name, bases, dct)
                assert set_fields_called[0] is True
                assert store_invariants_called[0] is True
                assert len(set_fields_called) == 1
                assert len(store_invariants_called) == 1
                return result
            finally:
                field_common.set_fields = original_set_fields
                checked_types.store_invariants = original_store_invariants

    class TestClass(metaclass=Meta):
        pass


# LLM-generated content at query #48
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 2, '_precord_buckets': [('field1', 'value1'), ('field2', 'value2')]}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'
    assert result['field2'] == 'value2'

def test_precord_constructor_with_factory_fields():
    kwargs = {'field1': 'value1', 'field2': 'value2', '_factory_fields': ['field1']}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'
    assert result['field2'] == 'value2'

def test_precord_constructor_with_ignore_extra():
    kwargs = {'field1': 'value1', 'field2': 'value2', 'extra_field': 'extra_value', '_ignore_extra': True}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'
    assert result['field2'] == 'value2'
    assert 'extra_field' not in result

def test_precord_constructor_with_initial_values():
    PRecord._precord_initial_values = {'field1': 'initial_value1', 'field2': 'initial_value2'}
    kwargs = {'field1': 'updated_value1'}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'updated_value1'
    assert result['field2'] == 'initial_value2'

def test_precord_constructor_with_callable_initial_values():
    PRecord._precord_initial_values = {'field1': lambda: 'initial_value1', 'field2': lambda: 'initial_value2'}
    kwargs = {'field1': 'updated_value1'}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'updated_value1'
    assert result['field2'] == 'initial_value2'


# LLM-generated content at query #49
#--------------------------

```python
def test_persistent_raises_when_invariant_error_codes_or_missing_fields():
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()

    evolver = _PRecordEvolver(MockPRecord, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []

    with pytest.raises(InvariantException):
        evolver.persistent()

    evolver._invariant_error_codes = []
    evolver._missing_fields = ['field1']

    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #50
#--------------------------

```python
def test_field_exists_in_precord_fields():
    class MockField:
        pass

    class MockDestinationCls:
        _precord_fields = {'existing_field': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    field = evolver._destination_cls._precord_fields.get('existing_field')
    assert field is not None


# LLM-generated content at query #51
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_callable_initial_values():
    class TestRecord(PRecord):
        x = lambda: 1
        y = lambda: 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord._factory_fields(['x', 'y'], x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord._ignore_extra(True, x=10, y=20, z=30)
    assert record.x == 10
    assert record.y == 20
    assert 'z' not in record

def test_precord_constructor_with_precord_size_and_buckets():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_precord_size=2, _precord_buckets=[('x', 1), ('y', 2)])
    assert record.x == 1
    assert record.y == 2


# LLM-generated content at query #52
#--------------------------

```python
def test_persistent_with_dirty_and_non_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result.get('key') == 'value'

def test_persistent_with_clean_and_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    pm = cls(_precord_buckets={}, _precord_size=0)
    evolver = _PRecordEvolver(cls, pm)
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'mandatory_field'}, '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'mandatory_field' in e.missing_fields

def test_persistent_with_invariant_errors():
    def failing_invariant(value):
        return (False, 'INVARIANT_FAILED')

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVARIANT_FAILED' in e.invariant_errors

def test_persistent_with_global_invariant_failure():
    def global_invariant(subject):
        return (False, 'GLOBAL_INVARIANT_FAILED')

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [global_invariant]})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'GLOBAL_INVARIANT_FAILED' in e.invariant_errors


# LLM-generated content at query #53
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 2, '_precord_buckets': [('a', 1), ('b', 2)]}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_constructor_with_factory_fields():
    kwargs = {'field1': 'value1', '_factory_fields': ['field1']}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'

def test_precord_constructor_with_ignore_extra():
    kwargs = {'field1': 'value1', 'extra_field': 'extra_value', '_ignore_extra': True}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'
    assert 'extra_field' not in result

def test_precord_constructor_with_initial_values():
    PRecord._precord_initial_values = {'field1': 'default1', 'field2': 'default2'}
    kwargs = {'field1': 'new_value'}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'new_value'
    assert result['field2'] == 'default2'

def test_precord_constructor_with_callable_initial_values():
    PRecord._precord_initial_values = {'field1': lambda: 'computed_value'}
    result = PRecord()
    assert isinstance(result, PRecord)
    assert result['field1'] == 'computed_value'


# LLM-generated content at query #54
#--------------------------

```python
def test_precord_mandatory_fields_are_checked():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('field1', 'value1')
    result = evolver.persistent()

    assert 'TestRecord.field2' in evolver._missing_fields


# LLM-generated content at query #55
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 10, '_precord_buckets': [('key', 'value')]}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['key'] == 'value'

def test_precord_constructor_with_factory_fields():
    kwargs = {'field1': 'value1', '_factory_fields': ['field1']}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'

def test_precord_constructor_with_ignore_extra():
    kwargs = {'field1': 'value1', 'extra_field': 'extra_value', '_ignore_extra': True}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'
    assert 'extra_field' not in result

def test_precord_constructor_with_initial_values():
    PRecord._precord_initial_values = {'field1': 'initial_value'}
    kwargs = {'field1': 'new_value'}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'new_value'

def test_precord_constructor_with_callable_initial_values():
    PRecord._precord_initial_values = {'field1': lambda: 'computed_value'}
    result = PRecord()
    assert isinstance(result, PRecord)
    assert result['field1'] == 'computed_value'


# LLM-generated content at query #56
#--------------------------

```python
def test_predicate_line_1_false():
    class TestRecord(PRecord):
        pass

    result = TestRecord()
    assert isinstance(result, TestRecord)


# LLM-generated content at query #57
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver.set('test_key', 'test_value')

    result = evolver.persistent()

    assert isinstance(result, TestClass)
    assert result['test_key'] == 'test_value'


# LLM-generated content at query #58
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockPRecord, PMap())
    evolver._is_dirty = True
    pm = PMap()
    result = evolver.persistent()
    assert isinstance(result, MockPRecord)


# LLM-generated content at query #59
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_default_values():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_callable_defaults():
    class TestRecord(PRecord):
        x = lambda: 1
        y = lambda: 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2

def test_precord_constructor_with_extra_fields_ignored():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20, z=30, _ignore_extra=True)
    assert record.x == 10
    assert record.y == 20
    assert 'z' not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20, _factory_fields={'x', 'y'})
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_precord_size_and_buckets():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_precord_size=2, _precord_buckets=[('x', 1), ('y', 2)])
    assert record.x == 1
    assert record.y == 2


