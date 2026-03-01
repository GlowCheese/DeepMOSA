####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result['key'] == 'value'

def test_persistent_with_clean_and_cls_instance():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    pm = cls(_precord_buckets=PMap()._buckets, _precord_size=PMap()._size)
    evolver = _PRecordEvolver(cls, pm)
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': {'mandatory_field'},
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockClass.mandatory_field' in e.missing_fields

def test_persistent_with_invariant_errors():
    def failing_invariant(subject):
        return False, 'INVARIANT_FAILED'

    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [failing_invariant],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('key', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVARIANT_FAILED' in e.invariant_errors

def test_persistent_with_global_invariant_failure():
    def failing_global_invariant(subject):
        return False, 'GLOBAL_INVARIANT_FAILED'

    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [failing_global_invariant],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('key', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'Global invariant failed' in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test__new__with_no_bases_and_empty_dct():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {})
    assert result._precord_fields == {}
    assert result._precord_invariants == ()
    assert result._precord_mandatory_fields == set()
    assert result._precord_initial_values == {}
    assert result.__slots__ == ()

def test__new__with_single_base_and_field_in_dct():
    class Base:
        pass

    dct = {'field1': _PField(mandatory=True, initial=1)}
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), dct)
    assert 'field1' in result._precord_fields
    assert result._precord_fields['field1'].mandatory is True
    assert result._precord_fields['field1'].initial == 1
    assert result._precord_invariants == ()
    assert result._precord_mandatory_fields == {'field1'}
    assert result._precord_initial_values == {'field1': 1}
    assert result.__slots__ == ()

def test__new__with_inherited_fields():
    class Base:
        _precord_fields = {'field1': _PField(mandatory=True)}

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {})
    assert 'field1' in result._precord_fields
    assert result._precord_fields['field1'].mandatory is True
    assert result._precord_mandatory_fields == {'field1'}
    assert result._precord_initial_values == {}

def test__new__with_inherited_invariants():
    class Base:
        __invariant__ = lambda self: True

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {})
    assert len(result._precord_invariants) == 1
    assert callable(result._precord_invariants[0])

def test__new__with_multiple_inherited_invariants():
    class Base1:
        __invariant__ = lambda self: True

    class Base2:
        __invariant__ = lambda self: (True, "test")

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base1, Base2), {})
    assert len(result._precord_invariants) == 2
    assert all(callable(inv) for inv in result._precord_invariants)

def test__new__with_non_callable_invariant_raises_type_error():
    class Base:
        __invariant__ = "not callable"

    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {})
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test__new__with_field_initial_value():
    dct = {'field1': _PField(initial=42)}
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), dct)
    assert result._precord_initial_values == {'field1': 42}

def test__new__with_no_initial_value_field():
    dct = {'field1': _PField()}
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), dct)
    assert result._precord_initial_values == {}


# LLM-generated content at query #3
#--------------------------

```python
def test_set_with_valid_field_and_factory():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'factory': lambda x: x, 'type': (int,), 'invariant': lambda x: (True, None)})()}
        _precord_mandatory_fields = set()

    evolver = _PRecordEvolver(TestRecord, PMap())
    result = evolver.set('field1', 10)
    assert result['field1'] == 10

def test_set_with_invalid_type():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'factory': lambda x: x, 'type': (int,), 'invariant': lambda x: (True, None)})()}
        _precord_mandatory_fields = set()

    evolver = _PRecordEvolver(TestRecord, PMap())
    try:
        evolver.set('field1', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.message == "Invalid type for field TestRecord.field1, was str"

def test_set_with_invariant_failure():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'factory': lambda x: x, 'type': (int,), 'invariant': lambda x: (False, 'ERROR')})()}
        _precord_mandatory_fields = set()

    evolver = _PRecordEvolver(TestRecord, PMap())
    result = evolver.set('field1', 10)
    assert result._invariant_error_codes == ['ERROR']

def test_set_with_nonexistent_field():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()

    evolver = _PRecordEvolver(TestRecord, PMap())
    try:
        evolver.set('nonexistent', 10)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'nonexistent' is not among the specified fields for TestRecord"

def test_set_with_ignore_extra_compliant_field():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'factory': lambda x, ignore_extra=False: x, 'type': {CheckedType}, 'invariant': lambda x: (True, None)})()}
        _precord_mandatory_fields = set()

    evolver = _PRecordEvolver(TestRecord, PMap(), _ignore_extra=True)
    result = evolver.set('field1', 10)
    assert result['field1'] == 10

def test_set_with_factory_field_not_in_factory_fields():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'factory': lambda x: x, 'type': (int,), 'invariant': lambda x: (True, None)})()}
        _precord_mandatory_fields = set()

    evolver = _PRecordEvolver(TestRecord, PMap(), _factory_fields=set())
    result = evolver.set('field1', 10)
    assert result['field1'] == 10


# LLM-generated content at query #4
#--------------------------

```python
def test_precord_new_with_special_attributes():
    result = PRecord._PRecordMeta.__new__(PRecord, 'TestRecord', (PRecord,), {})()
    assert isinstance(result, PRecord)
    assert result._precord_size == 0
    assert result._buckets == pvector()

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert result._precord_size == 0
    assert result._buckets == pvector()

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': 2}
    result = TestRecord()
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_kwargs():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    result = TestRecord(_factory_fields={'a'}, a=1, b=2)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    result = TestRecord(_ignore_extra=True, a=1, b=2, c=3)
    assert result['a'] == 1
    assert result['b'] == 2
    assert 'c' not in result


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
        assert e.missing_fields == ('MockClass.mandatory_field',)

def test_persistent_with_invariant_errors():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver._invariant_error_codes = ['error1', 'error2']
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')

def test_persistent_with_global_invariant_failure():
    def failing_invariant(subject):
        return (False, 'global_error')

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)


# LLM-generated content at query #5
#--------------------------

```python
def test_persistent_with_dirty_state_and_non_instance():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: True
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result._precord_buckets == original_pmap._buckets
    assert result._precord_size == original_pmap._size

def test_persistent_with_clean_state_and_instance():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: False
    pm = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver._persistent = lambda: pm
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': {'field1'},
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: False
    pm = type('MockPMap', (), {'_buckets': {}, '_size': 0, 'keys': lambda: []})()
    evolver._persistent = lambda: pm
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ("MockClass.field1",)

def test_persistent_with_invariant_errors():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._invariant_error_codes = ['error1']
    evolver._is_dirty = lambda: False
    pm = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver._persistent = lambda: pm
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)

def test_persistent_with_global_invariant_failure():
    def failing_invariant(subject):
        return (False, 'global_error')

    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [failing_invariant],
        '__name__': 'MockClass'
    })
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: False
    pm = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver._persistent = lambda: pm
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize_with_no_custom_serializers():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=10, field2="test")
    serialized = record.serialize()
    assert serialized == {"field1": 10, "field2": "test"}

def test_serialize_with_custom_serializers():
    class TestRecord(PRecord):
        field1 = field(serializer=lambda x: str(x))
        field2 = field(serializer=lambda x: x.upper())

    record = TestRecord(field1=10, field2="test")
    serialized = record.serialize()
    assert serialized == {"field1": "10", "field2": "TEST"}

def test_serialize_with_format_parameter():
    class TestRecord(PRecord):
        field1 = field(serializer=lambda x, fmt: f"{x}:{fmt}")
        field2 = field(serializer=lambda x, fmt: f"{x.upper()}-{fmt}")

    record = TestRecord(field1=10, field2="test")
    serialized = record.serialize(format="json")
    assert serialized == {"field1": "10:json", "field2": "TEST-json"}

def test_serialize_empty_record():
    class TestRecord(PRecord):
        pass

    record = TestRecord()
    serialized = record.serialize()
    assert serialized == {}


# LLM-generated content at query #7
#--------------------------

```python
def test_repr_empty():
    class TestRecord(PRecord):
        pass
    record = TestRecord()
    assert repr(record) == "TestRecord()"

def test_repr_single_field():
    class TestRecord(PRecord):
        field1 = field()
    record = TestRecord(field1=42)
    assert repr(record) == "TestRecord(field1=42)"

def test_repr_multiple_fields():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()
    record = TestRecord(field1=42, field2="hello")
    assert repr(record) == "TestRecord(field1=42, field2='hello')"

def test_repr_with_complex_values():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()
    record = TestRecord(field1=[1, 2, 3], field2={"key": "value"})
    assert repr(record) == "TestRecord(field1=[1, 2, 3], field2={'key': 'value'})"


# LLM-generated content at query #8
#--------------------------

```python
def test_invariant_exception_raised_when_error_codes_or_missing_fields():
    evolver = _PRecordEvolver.__new__(_PRecordEvolver)
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []
    evolver._destination_cls = type('MockClass', (), {'_precord_invariants': []})

    with pytest.raises(InvariantException):
        evolver.persistent()

def test_invariant_exception_raised_when_missing_fields():
    evolver = _PRecordEvolver.__new__(_PRecordEvolver)
    evolver._invariant_error_codes = []
    evolver._missing_fields = ['field1']
    evolver._destination_cls = type('MockClass', (), {'_precord_invariants': []})

    with pytest.raises(InvariantException):
        evolver.persistent()

def test_invariant_exception_raised_when_both_error_codes_and_missing_fields():
    evolver = _PRecordEvolver.__new__(_PRecordEvolver)
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['field1']
    evolver._destination_cls = type('MockClass', (), {'_precord_invariants': []})

    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #9
#--------------------------

```python
def test_repr_empty_record():
    class EmptyRecord(PRecord):
        pass
    record = EmptyRecord()
    assert repr(record) == "EmptyRecord()"

def test_repr_simple_record():
    class SimpleRecord(PRecord):
        field1 = field()
        field2 = field()
    record = SimpleRecord(field1=1, field2="test")
    assert repr(record) == "SimpleRecord(field1=1, field2='test')"

def test_repr_nested_record():
    class NestedRecord(PRecord):
        inner = field()
        value = field()
    inner_record = SimpleRecord(field1=1, field2="test")
    record = NestedRecord(inner=inner_record, value=42)
    assert repr(record) == "NestedRecord(inner=SimpleRecord(field1=1, field2='test'), value=42)"

def test_repr_with_special_characters():
    class SpecialRecord(PRecord):
        name = field()
        path = field()
    record = SpecialRecord(name="test's record", path="/path/to/file")
    assert repr(record) == "SpecialRecord(name=\"test's record\", path='/path/to/file')"

def test_repr_with_none_values():
    class NullableRecord(PRecord):
        optional = field()
        required = field()
    record = NullableRecord(optional=None, required=100)
    assert repr(record) == "NullableRecord(optional=None, required=100)"


# LLM-generated content at query #2
#--------------------------

```python
def test_set_with_valid_field_and_factory():
    class MockField:
        type = (int,)
        invariant = lambda self, value: (True, None)
        factory = lambda value: value * 2

    class MockDestinationCls:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    result = evolver.set('field1', 5)
    assert result['field1'] == 10

def test_set_with_invalid_field_type():
    class MockField:
        type = (int,)
        invariant = lambda self, value: (True, None)
        factory = lambda value: value

    class MockDestinationCls:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    with pytest.raises(PTypeError):
        evolver.set('field1', 'not_an_int')

def test_set_with_invariant_failure():
    class MockField:
        type = (int,)
        invariant = lambda self, value: (False, 'INVALID') if value < 0 else (True, None)
        factory = lambda value: value

    class MockDestinationCls:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    result = evolver.set('field1', -5)
    assert result._invariant_error_codes == ['INVALID']

def test_set_with_nonexistent_field():
    class MockDestinationCls:
        _precord_fields = {}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    with pytest.raises(AttributeError):
        evolver.set('nonexistent_field', 10)

def test_set_with_ignore_extra_and_compliant_factory():
    class MockField:
        type = (CheckedType,)
        invariant = lambda self, value: (True, None)
        factory = lambda value, ignore_extra=False: value if ignore_extra else value * 2

    class MockDestinationCls:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap(), _ignore_extra=True)
    result = evolver.set('field1', 5)
    assert result['field1'] == 5

def test_set_with_factory_fields_restriction():
    class MockField:
        type = (int,)
        invariant = lambda self, value: (True, None)
        factory = lambda value: value * 2

    class MockDestinationCls:
        _precord_fields = {'field1': MockField(), 'field2': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap(), _factory_fields=[MockDestinationCls._precord_fields['field1']])
    result = evolver.set('field1', 5)
    assert result['field1'] == 10
    result = evolver.set('field2', 5)
    assert result['field2'] == 5


# LLM-generated content at query #3
#--------------------------

```python
def test__new__sets_fields_and_invariants():
    class Base:
        __invariant__ = lambda self: (True, "test")
        x = _PField()

    class TestRecord(metaclass=_PRecordMeta):
        y = _PField()

    assert '_precord_fields' in TestRecord.__dict__
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert TestRecord._precord_invariants == (wrap_invariant(Base.__invariant__),)


# LLM-generated content at query #10
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result.get('key') == 'value'

def test_persistent_with_clean_and_cls_instance():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    pm = cls(_precord_buckets=PMap()._buckets, _precord_size=PMap()._size)
    evolver = _PRecordEvolver(cls, pm)
    result = evolver.persistent()
    assert result is pm

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': {'mandatory_field'},
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('MockClass.mandatory_field',)

def test_persistent_with_invariant_errors():
    def failing_invariant(subject):
        return (False, 'INVARIANT_FAILED')

    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [failing_invariant],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('key', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('INVARIANT_FAILED',)

def test_persistent_with_global_invariant_failure():
    def failing_global_invariant(subject):
        return (False, 'GLOBAL_INVARIANT_FAILED')

    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [failing_global_invariant],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('GLOBAL_INVARIANT_FAILED',)


# LLM-generated content at query #11
#--------------------------

```python
def test_persistent_with_dirty_and_non_instance():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    original_pmap = type('MockPMap', (), {
        '_buckets': 'mock_buckets',
        '_size': 'mock_size'
    })()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: True
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result._precord_buckets == 'mock_buckets'
    assert result._precord_size == 'mock_size'

def test_persistent_with_clean_and_instance():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    original_pmap = type('MockPMap', (), {
        '_buckets': 'mock_buckets',
        '_size': 'mock_size'
    })()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: False
    pm = type('MockPMap', (), {
        '_buckets': 'mock_buckets',
        '_size': 'mock_size'
    })()
    evolver._persistent = lambda: pm
    result = evolver.persistent()
    assert result == pm

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': {'field1'},
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    original_pmap = type('MockPMap', (), {
        '_buckets': {},
        '_size': 0
    })()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._is_dirty = lambda: True
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('MockClass.field1',)

def test_persistent_with_invariant_error_codes():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    original_pmap = type('MockPMap', (), {
        '_buckets': {},
        '_size': 0
    })()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._invariant_error_codes = ['error1']
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)

def test_persistent_with_global_invariant_failure():
    def failing_invariant(subject):
        return (False, 'global_error')

    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [failing_invariant],
        '__name__': 'MockClass'
    })
    original_pmap = type('MockPMap', (), {
        '_buckets': {},
        '_size': 0
    })()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_false():
    class TestRecord(PRecord):
        pass

    result = TestRecord.__new__(TestRecord)
    assert result is not None


# LLM-generated content at query #4
#--------------------------

```python
def test_precord_new_with_special_attributes():
    kwargs = {'_precord_size': 2, '_precord_buckets': [(('a', 1),), (('b', 2),)]}
    result = PRecord.__new__(PRecord, **kwargs)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    TestRecord._precord_initial_values = {}
    result = TestRecord(a=1, b=2)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    TestRecord._precord_initial_values = {'a': 10, 'b': 20}
    result = TestRecord()
    assert result == {'a': 10, 'b': 20}

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    result = TestRecord(a=1, b=2, _factory_fields={'a'})
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'a': None, 'b': None}
    result = TestRecord(a=1, b=2, c=3, ignore_extra=True)
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #13
#--------------------------

```python
def test_precord_constructor_with_valid_kwargs():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20
    assert len(record) == 2

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2
    assert len(record) == 2

def test_precord_constructor_with_callable_initial_values():
    class TestRecord(PRecord):
        x = lambda: 1
        y = lambda: 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2
    assert len(record) == 2

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord._factory_fields({'x': 10, 'y': 20})
    assert record.x == 10
    assert record.y == 20
    assert len(record) == 2

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord._ignore_extra(True, x=10, y=20, z=30)
    assert record.x == 10
    assert record.y == 20
    assert len(record) == 2
    assert 'z' not in record

def test_precord_constructor_with_precord_size_and_buckets():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_precord_size=2, _precord_buckets=[('x', 1), ('y', 2)])
    assert record.x == 1
    assert record.y == 2
    assert len(record) == 2


# LLM-generated content at query #5
#--------------------------

```python
def test_precord_fields_initialized():
    class TestRecord(metaclass=_PRecordMeta):
        pass

    assert '_precord_fields' in TestRecord.__dict__
    assert isinstance(TestRecord._precord_fields, dict)


# LLM-generated content at query #14
#--------------------------

```python
def test_missing_fields_added_when_mandatory_fields_not_present():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = {'field1', 'field2'}
        _precord_invariants = []
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('field1', 'value1')
    result = evolver.persistent()
    assert 'TestRecord.field2' in evolver._missing_fields


# LLM-generated content at query #6
#--------------------------

```python
def test_repr_returns_correct_string():
    class TestRecord(PRecord):
        pass

    record = TestRecord(a=1, b="test")
    assert repr(record) == "TestRecord(a=1, b='test')"


# LLM-generated content at query #7
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    cls = PRecord
    kwargs = {'_precord_size': 2, '_precord_buckets': [None, [('a', 1)], None, [('b', 2)]]}
    result = cls.__new__(cls, **kwargs)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_without_precord_size_and_buckets():
    class TestRecord(PRecord):
        pass
    kwargs = {'a': 1, 'b': 2}
    result = TestRecord.__new__(TestRecord, **kwargs)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    kwargs = {'a': 1, 'b': 2, '_factory_fields': ['a']}
    result = TestRecord.__new__(TestRecord, **kwargs)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    kwargs = {'a': 1, 'b': 2, '_ignore_extra': True}
    result = TestRecord.__new__(TestRecord, **kwargs)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': 2}
    kwargs = {'b': 3}
    result = TestRecord.__new__(TestRecord, **kwargs)
    assert result == {'a': 1, 'b': 3}

def test_precord_new_with_callable_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': lambda: 1, 'b': 2}
    kwargs = {'b': 3}
    result = TestRecord.__new__(TestRecord, **kwargs)
    assert result == {'a': 1, 'b': 3}


# LLM-generated content at query #8
#--------------------------

```python
def test_serialize_without_custom_serializer():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert result == {"field1": "value1", "field2": "value2"}

def test_serialize_with_custom_serializer():
    class TestRecord(PRecord):
        field1 = field(serializer=lambda x: x.upper())
        field2 = field(serializer=lambda x: x * 2)

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert result == {"field1": "VALUE1", "field2": "value2value2"}

def test_serialize_with_format_parameter():
    class TestRecord(PRecord):
        field1 = field(serializer=lambda x, fmt=None: f"{x}_{fmt}" if fmt else x)
        field2 = field(serializer=lambda x, fmt=None: x if fmt else x.upper())

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize(format="json")
    assert result == {"field1": "value1_json", "field2": "value2"}


# LLM-generated content at query #9
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    evolver = _PRecordEvolver(TestPRecord, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'value1')
    result = evolver.persistent()
    assert isinstance(result, TestPRecord)
    assert result['field1'] == 'value1'

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

def test_persistent_with_invariant_error():
    evolver = _PRecordEvolver(TestPRecord, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'invalid_value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVALID_VALUE' in e.invariant_errors

def test_persistent_with_global_invariant_failure():
    evolver = _PRecordEvolver(TestPRecord, PMap(), _factory_fields=None, _ignore_extra=False)
    evolver.set('field1', 'value1')
    evolver.set('field2', 'value2')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'GLOBAL_INVARIANT_FAILED' in e.invariant_errors


# LLM-generated content at query #15
#--------------------------

```python
def test_precord_meta_new_with_no_bases_and_no_fields():
    class TestRecord(metaclass=_PRecordMeta):
        pass

    assert hasattr(TestRecord, '_precord_fields')
    assert TestRecord._precord_fields == {}
    assert hasattr(TestRecord, '_precord_invariants')
    assert TestRecord._precord_invariants == ()
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert TestRecord._precord_mandatory_fields == set()
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord._precord_initial_values == {}
    assert hasattr(TestRecord, '__slots__')
    assert TestRecord.__slots__ == ()

def test_precord_meta_new_with_fields():
    class TestRecord(metaclass=_PRecordMeta):
        field1 = _PField()
        field2 = _PField(mandatory=True)
        field3 = _PField(initial=42)

    assert 'field1' in TestRecord._precord_fields
    assert 'field2' in TestRecord._precord_fields
    assert 'field3' in TestRecord._precord_fields
    assert TestRecord._precord_mandatory_fields == {'field2'}
    assert TestRecord._precord_initial_values == {'field3': 42}

def test_precord_meta_new_with_inherited_fields():
    class BaseRecord(metaclass=_PRecordMeta):
        field1 = _PField()

    class TestRecord(BaseRecord):
        field2 = _PField()

    assert 'field1' in TestRecord._precord_fields
    assert 'field2' in TestRecord._precord_fields

def test_precord_meta_new_with_invariants():
    def test_invariant(obj):
        return True, "OK"

    class TestRecord(metaclass=_PRecordMeta):
        __invariant__ = test_invariant

    assert len(TestRecord._precord_invariants) == 1
    assert callable(TestRecord._precord_invariants[0])

def test_precord_meta_new_with_inherited_invariants():
    def base_invariant(obj):
        return True, "Base OK"

    class BaseRecord(metaclass=_PRecordMeta):
        __invariant__ = base_invariant

    def test_invariant(obj):
        return True, "Test OK"

    class TestRecord(BaseRecord):
        __invariant__ = test_invariant

    assert len(TestRecord._precord_invariants) == 2
    assert all(callable(inv) for inv in TestRecord._precord_invariants)

def test_precord_meta_new_with_non_callable_invariant():
    try:
        class TestRecord(metaclass=_PRecordMeta):
            __invariant__ = "not callable"

        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"


# LLM-generated content at query #16
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    cls = PRecord
    size = 2
    buckets = pvector([None, [(1, 'a')]])
    result = cls.__new__(cls, _precord_size=size, _precord_buckets=buckets)
    assert result._size == size
    assert result._buckets == buckets

def test_precord_new_without_precord_size_and_buckets():
    cls = PRecord
    kwargs = {'a': 1, 'b': 2}
    result = cls.__new__(cls, **kwargs)
    assert isinstance(result, cls)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_factory_fields():
    cls = PRecord
    kwargs = {'a': 1, 'b': 2}
    factory_fields = {'a'}
    result = cls.__new__(cls, _factory_fields=factory_fields, **kwargs)
    assert isinstance(result, cls)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_ignore_extra():
    cls = PRecord
    kwargs = {'a': 1, 'b': 2, 'c': 3}
    result = cls.__new__(cls, _ignore_extra=True, **kwargs)
    assert isinstance(result, cls)
    assert result.get('a') == 1
    assert result.get('b') == 2
    assert result.get('c') is None

def test_precord_new_with_initial_values():
    cls = PRecord
    cls._precord_initial_values = {'a': 1, 'b': lambda: 2}
    kwargs = {'b': 3, 'c': 4}
    result = cls.__new__(cls, **kwargs)
    assert isinstance(result, cls)
    assert result['a'] == 1
    assert result['b'] == 3
    assert result['c'] == 4


# LLM-generated content at query #10
#--------------------------

```python
def test_precord_repr_format():
    class TestRecord(PRecord):
        pass

    record = TestRecord(a=1, b=2)
    assert repr(record) == "TestRecord(a=1, b=2)"


# LLM-generated content at query #17
#--------------------------

```python
def test_field_exists():
    class MockField:
        pass

    class MockDestinationCls:
        _precord_fields = {'key': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    field = evolver._destination_cls._precord_fields.get('key')
    assert field is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_missing_fields_are_added_when_mandatory_fields_exist():
    class TestClass:
        _precord_mandatory_fields = {'field1', 'field2'}
        __name__ = 'TestClass'

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver._missing_fields = []
    result = PMap({'field1': 'value1'})
    evolver._destination_cls = TestClass

    evolver.persistent()

    assert evolver._missing_fields == ('TestClass.field2',)


# LLM-generated content at query #12
#--------------------------

```python
def test_precord_mandatory_fields_is_subset_of_precord_fields():
    class TestRecord(metaclass=_PRecordMeta):
        a = _PField(mandatory=True)
        b = _PField(mandatory=False)
        c = _PField(mandatory=True)

    assert TestRecord._precord_mandatory_fields.issubset(TestRecord._precord_fields.keys())


# LLM-generated content at query #13
#--------------------------

```python
def test_persistent_no_changes():
    cls = type('MockPRecord', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockPRecord'
    })
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result == original_pmap

def test_persistent_with_changes():
    cls = type('MockPRecord', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockPRecord'
    })
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result != original_pmap
    assert result['key'] == 'value'

def test_persistent_missing_mandatory_fields():
    cls = type('MockPRecord', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': {'mandatory_field'},
        '_precord_invariants': [],
        '__name__': 'MockPRecord'
    })
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockPRecord.mandatory_field' in e.missing_fields

def test_persistent_invariant_errors():
    def failing_invariant(value):
        return (False, 'INVARIANT_FAILED')

    cls = type('MockPRecord', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [lambda x: failing_invariant(x)],
        '__name__': 'MockPRecord'
    })
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('key', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVARIANT_FAILED' in e.invariant_errors

def test_persistent_global_invariant_failure():
    def global_invariant(subject):
        return (False, 'GLOBAL_INVARIANT_FAILED')

    cls = type('MockPRecord', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [global_invariant],
        '__name__': 'MockPRecord'
    })
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'GLOBAL_INVARIANT_FAILED' in e.invariant_errors


# LLM-generated content at query #14
#--------------------------

```python
def test_persistent_raises_when_invariant_error_codes_or_missing_fields():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = []

    with pytest.raises(InvariantException):
        evolver.persistent()

    evolver._invariant_error_codes = []
    evolver._missing_fields = ['field1']

    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #15
#--------------------------

```python
def test_set_with_valid_field_and_factory():
    class MockField:
        type = (int,)
        invariant = lambda self, value: (True, None)
        factory = lambda value: value * 2

    class MockDestinationCls:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    result = evolver.set('field1', 5)
    assert result['field1'] == 10

def test_set_with_valid_field_and_factory_ignore_extra():
    class MockField:
        type = (int,)
        invariant = lambda self, value: (True, None)
        factory = lambda value, ignore_extra=False: value * 2 if not ignore_extra else value * 3

    class MockDestinationCls:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap(), _ignore_extra=True)
    result = evolver.set('field1', 5)
    assert result['field1'] == 15

def test_set_with_invalid_field_type():
    class MockField:
        type = (int,)
        invariant = lambda self, value: (True, None)
        factory = lambda value: value

    class MockDestinationCls:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    try:
        evolver.set('field1', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.message == "Invalid type for field MockDestinationCls.field1, was str"

def test_set_with_invariant_failure():
    class MockField:
        type = (int,)
        invariant = lambda self, value: (False, 'INVALID') if value < 0 else (True, None)
        factory = lambda value: value

    class MockDestinationCls:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    result = evolver.set('field1', -5)
    assert result._invariant_error_codes == ['INVALID']

def test_set_with_missing_field():
    class MockDestinationCls:
        _precord_fields = {}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    try:
        evolver.set('nonexistent_field', 5)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'nonexistent_field' is not among the specified fields for MockDestinationCls"

def test_set_with_factory_field_not_in_factory_fields():
    class MockField:
        type = (int,)
        invariant = lambda self, value: (True, None)
        factory = lambda value: value * 2

    class MockDestinationCls:
        _precord_fields = {'field1': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap(), _factory_fields=[])
    result = evolver.set('field1', 5)
    assert result['field1'] == 5


# LLM-generated content at query #16
#--------------------------

```python
def test_repr_format():
    class TestRecord(PRecord):
        pass

    record = TestRecord(a=1, b="test")
    repr_str = repr(record)
    assert repr_str.startswith("TestRecord(") and repr_str.endswith(")")


# LLM-generated content at query #17
#--------------------------

```python
def test_missing_mandatory_fields_are_added_to_missing_fields():
    class TestClass:
        _precord_mandatory_fields = {'field1', 'field2'}
        __name__ = 'TestClass'

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver._missing_fields = []
    result = PMap({'field1': 'value1'})

    evolver.persistent()

    assert evolver._missing_fields == ('TestClass.field2',)


# LLM-generated content at query #18
#--------------------------

```python
def test_persistent_predicate_false():
    class MockPMap:
        _buckets = None
        _size = 0

    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockPRecord, MockPMap())
    evolver._is_dirty = lambda: False
    evolver._destination_cls = MockPRecord
    evolver._invariant_error_codes = []
    evolver._missing_fields = []

    pm = MockPMap()
    evolver._destination_cls = type(pm)

    result = evolver.persistent()

    assert result is pm


# LLM-generated content at query #19
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
    kwargs = {'field1': 'value1'}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'value1'


# LLM-generated content at query #20
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    buckets = [(('a', 1), ('b', 2)), None, None, None]
    result = PRecord.__new__(PRecord, _precord_size=2, _precord_buckets=buckets)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_without_precord_size_and_buckets():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    result = TestRecord(field1=10, field2=20)
    assert result.field1 == 10
    assert result.field2 == 20

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    result = TestRecord._factory_fields={'field1'}, field1=10, field2=20)
    assert result.field1 == 10
    assert result.field2 == 20

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    result = TestRecord._ignore_extra=True, field1=10, field2=20, extra=30)
    assert result.field1 == 10
    assert result.field2 == 20
    assert 'extra' not in result

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        field1 = field(initial=5)
        field2 = field()

    result = TestRecord(field2=20)
    assert result.field1 == 5
    assert result.field2 == 20


# LLM-generated content at query #21
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver.set('test_key', 'test_value')
    result = evolver.persistent()
    assert isinstance(result, TestClass)
    assert result['test_key'] == 'test_value'


# LLM-generated content at query #22
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
    kwargs = {'field1': 'updated_value'}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert result['field1'] == 'updated_value'

def test_precord_constructor_with_callable_initial_values():
    PRecord._precord_initial_values = {'field1': lambda: 'computed_value'}
    result = PRecord()
    assert isinstance(result, PRecord)
    assert result['field1'] == 'computed_value'


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
    original = TestPRecord(field1='value1')
    evolver = _PRecordEvolver(TestPRecord, original._map)
    result = evolver.persistent()
    assert result is original

def test_persistent_with_missing_mandatory_fields():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestPRecord.mandatory_field' in e.missing_fields

def test_persistent_with_invariant_errors():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    evolver.set('field_with_invariant', 'invalid_value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('INVARIANT_ERROR',)

def test_persistent_with_global_invariant_failure():
    evolver = _PRecordEvolver(TestPRecord, PMap())
    evolver.set('field1', 'value1')
    evolver.set('field2', 'value2')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.message == 'Global invariant failed'


# LLM-generated content at query #2
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
        assert e.missing_fields == ("MockClass.mandatory_field",)

def test_persistent_with_invariant_error_codes():
    def failing_invariant(subject):
        return (False, 'INVARIANT_FAILED')

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('INVARIANT_FAILED',)

def test_persistent_with_global_invariant_failure():
    def failing_global_invariant(subject):
        return (False, 'GLOBAL_INVARIANT_FAILED')

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_global_invariant]})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('GLOBAL_INVARIANT_FAILED',)


# LLM-generated content at query #3
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    size = 2
    buckets = [None] * size
    result = PRecord.__new__(PRecord, _precord_size=size, _precord_buckets=buckets)
    assert result._size == size
    assert result._buckets == buckets

def test_precord_new_without_precord_size_and_buckets():
    class TestRecord(PRecord):
        pass
    kwargs = {'a': 1, 'b': 2}
    result = TestRecord.__new__(TestRecord, **kwargs)
    assert isinstance(result, TestRecord)
    assert result == kwargs

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    kwargs = {'a': 1, 'b': 2}
    factory_fields = {'a'}
    result = TestRecord.__new__(TestRecord, _factory_fields=factory_fields, **kwargs)
    assert isinstance(result, TestRecord)
    assert result == kwargs

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    kwargs = {'a': 1, 'b': 2}
    result = TestRecord.__new__(TestRecord, _ignore_extra=True, **kwargs)
    assert isinstance(result, TestRecord)
    assert result == kwargs

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': 2}
    kwargs = {'b': 3}
    result = TestRecord.__new__(TestRecord, **kwargs)
    assert isinstance(result, TestRecord)
    assert result == {'a': 1, 'b': 3}


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_false():
    assert not ('_precord_size' in {'a': 1} and '_precord_buckets' in {'a': 1})


# LLM-generated content at query #5
#--------------------------

```python
def test_new_with_no_bases_and_empty_dct():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {})
    assert result._precord_fields == {}
    assert result._precord_invariants == ()
    assert result._precord_mandatory_fields == set()
    assert result._precord_initial_values == {}
    assert result.__slots__ == ()

def test_new_with_bases_and_fields():
    class Base:
        x = _PField(mandatory=True, initial=1)

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {})
    assert 'x' in result._precord_fields
    assert result._precord_fields['x'].mandatory is True
    assert result._precord_fields['x'].initial == 1
    assert result._precord_mandatory_fields == {'x'}
    assert result._precord_initial_values == {'x': 1}

def test_new_with_invariant():
    def test_inv():
        return True

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'__invariant__': test_inv})
    assert len(result._precord_invariants) == 1
    assert callable(result._precord_invariants[0])

def test_new_with_inherited_invariant():
    class Base:
        __invariant__ = lambda: True

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base,), {})
    assert len(result._precord_invariants) == 1
    assert callable(result._precord_invariants[0])

def test_new_with_non_callable_invariant_raises():
    try:
        _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'__invariant__': 'not callable'})
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test_new_with_multiple_invariants():
    def inv1():
        return True

    def inv2():
        return (True, 'data')

    class Base1:
        __invariant__ = inv1

    class Base2:
        __invariant__ = inv2

    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (Base1, Base2), {})
    assert len(result._precord_invariants) == 2
    assert all(callable(inv) for inv in result._precord_invariants)

def test_new_with_field_in_dct():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'x': _PField(mandatory=True)})
    assert 'x' in result._precord_fields
    assert result._precord_mandatory_fields == {'x'}
    assert 'x' not in result.__dict__

def test_new_with_initial_value_none():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'x': _PField(initial=None)})
    assert result._precord_initial_values == {'x': None}

def test_new_with_initial_value_no_initial():
    result = _PRecordMeta.__new__(_PRecordMeta, 'TestClass', (), {'x': _PField(initial=PFIELD_NO_INITIAL)})
    assert result._precord_initial_values == {}


# LLM-generated content at query #6
#--------------------------

```python
def test_set_existing_field_with_valid_value():
    field = type('Field', (), {'type': (int,), 'factory': lambda x: x, 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}, '__name__': 'TestClass'})
    evolver = _PRecordEvolver(cls, PMap())
    result = evolver.set('field1', 10)
    assert result['field1'] == 10
    assert evolver._invariant_error_codes == []
    assert evolver._missing_fields == []

def test_set_existing_field_with_invalid_type():
    field = type('Field', (), {'type': (int,), 'factory': lambda x: x, 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}, '__name__': 'TestClass'})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.set('field1', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.message == "Invalid type for field TestClass.field1, was str"

def test_set_existing_field_with_invariant_failure():
    field = type('Field', (), {'type': (int,), 'factory': lambda x: x, 'invariant': lambda x: (False, 'INVALID')})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}, '__name__': 'TestClass'})
    evolver = _PRecordEvolver(cls, PMap())
    result = evolver.set('field1', 10)
    assert result['field1'] == 10
    assert evolver._invariant_error_codes == ['INVALID']
    assert evolver._missing_fields == []

def test_set_existing_field_with_factory_and_ignore_extra():
    field = type('Field', (), {'type': {CheckedType}, 'factory': lambda x, ignore_extra=False: x, 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}, '__name__': 'TestClass'})
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields={field}, _ignore_extra=True)
    result = evolver.set('field1', 10)
    assert result['field1'] == 10
    assert evolver._invariant_error_codes == []
    assert evolver._missing_fields == []

def test_set_existing_field_with_factory_and_invariant_exception():
    field = type('Field', (), {'type': (int,), 'factory': lambda x: (1/0), 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}, '__name__': 'TestClass'})
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields={field})
    result = evolver.set('field1', 10)
    assert result is evolver
    assert evolver._invariant_error_codes == []
    assert evolver._missing_fields == []

def test_set_nonexistent_field():
    cls = type('TestClass', (), {'_precord_fields': {}, '__name__': 'TestClass'})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.set('nonexistent', 10)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'nonexistent' is not among the specified fields for TestClass"

def test_set_existing_field_not_in_factory_fields():
    field = type('Field', (), {'type': (int,), 'factory': lambda x: x, 'invariant': lambda x: (True, None)})
    cls = type('TestClass', (), {'_precord_fields': {'field1': field}, '__name__': 'TestClass'})
    evolver = _PRecordEvolver(cls, PMap(), _factory_fields=set())
    result = evolver.set('field1', 10)
    assert result['field1'] == 10
    assert evolver._invariant_error_codes == []
    assert evolver._missing_fields == []


# LLM-generated content at query #7
#--------------------------

```python
def test_missing_fields_added_when_mandatory_fields_not_present():
    class MockPRecord:
        _precord_mandatory_fields = {'field1', 'field2'}
        __name__ = 'MockPRecord'

    evolver = _PRecordEvolver(MockPRecord, PMap())
    evolver.set('field1', 'value1')
    result = evolver.persistent()

    assert 'MockPRecord.field2' in evolver._missing_fields


# LLM-generated content at query #8
#--------------------------

```python
def test_persistent_raises_invariant_exception_when_error_codes_or_missing_fields():
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
        assert str(e) == 'Field invariant failed'


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

```python
def test_missing_fields_added_when_mandatory_fields_exist():
    class TestClass:
        _precord_mandatory_fields = {'field1', 'field2'}
        __name__ = 'TestClass'

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver._missing_fields = []
    result = PMap({'field1': 'value1'})

    evolver.persistent()

    assert evolver._missing_fields == ('TestClass.field2',)


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_line_1():
    class TestClass(metaclass=_PRecordMeta):
        pass

    assert isinstance(TestClass, _PRecordMeta)


# LLM-generated content at query #12
#--------------------------

```python
def test_precord_initial_values_used():
    class TestRecord(PRecord):
        _precord_fields = {'a': None, 'b': None}
        _precord_initial_values = {'a': lambda: 1, 'b': 2}

    result = TestRecord()
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #13
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
    record = MultiFieldRecord(field1="value1", field2=42, field3=None)
    assert repr(record) == "MultiFieldRecord(field1='value1', field2=42, field3=None)"

def test_repr_with_complex_values():
    class ComplexRecord(PRecord):
        list_field = field()
        dict_field = field()
    record = ComplexRecord(list_field=[1, 2, 3], dict_field={"a": 1})
    assert repr(record) == "ComplexRecord(list_field=[1, 2, 3], dict_field={'a': 1})"

def test_repr_with_escaped_strings():
    class StringRecord(PRecord):
        text = field()
    record = StringRecord(text="Hello\nWorld")
    assert repr(record) == "StringRecord(text='Hello\\nWorld')"


