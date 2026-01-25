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
        assert e.missing_fields == ("MockClass.mandatory_field",)

def test_persistent_with_invariant_errors():
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


# LLM-generated content at query #2
#--------------------------

```python
def test_repr_returns_correct_string():
    class TestRecord(PRecord):
        field1 = 1
        field2 = "test"

    record = TestRecord(field1=10, field2="value")
    assert repr(record) == "TestRecord(field1=10, field2='value')"


# LLM-generated content at query #3
#--------------------------

```python
def test_precord_new_with_special_attributes():
    cls = PRecord
    size = 5
    buckets = pvector().extend([None] * 8)
    result = cls.__new__(cls, _precord_size=size, _precord_buckets=buckets)
    assert result._size == size
    assert result._buckets == buckets

def test_precord_new_without_special_attributes():
    cls = PRecord
    kwargs = {'a': 1, 'b': 2}
    result = cls.__new__(cls, **kwargs)
    assert isinstance(result, cls)
    assert result == pmap(kwargs)

def test_precord_new_with_factory_fields():
    cls = PRecord
    kwargs = {'a': 1, 'b': 2}
    factory_fields = {'a'}
    result = cls.__new__(cls, _factory_fields=factory_fields, **kwargs)
    assert isinstance(result, cls)
    assert result == pmap(kwargs)

def test_precord_new_with_ignore_extra():
    cls = PRecord
    kwargs = {'a': 1, 'b': 2}
    result = cls.__new__(cls, _ignore_extra=True, **kwargs)
    assert isinstance(result, cls)
    assert result == pmap(kwargs)

def test_precord_new_with_initial_values():
    cls = PRecord
    cls._precord_initial_values = {'a': 1, 'b': lambda: 2}
    kwargs = {'b': 3, 'c': 4}
    result = cls.__new__(cls, **kwargs)
    assert isinstance(result, cls)
    assert result == pmap({'a': 1, 'b': 3, 'c': 4})


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_false():
    class TestRecord(PRecord):
        pass

    result = TestRecord()
    assert isinstance(result, TestRecord)


# LLM-generated content at query #5
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver.set('test_key', 'test_value')

    assert evolver.is_dirty() is True
    result = evolver.persistent()
    assert isinstance(result, TestClass)


# LLM-generated content at query #6
#--------------------------

```python
def test_precord_repr_format():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1=10, field2="test")
    repr_str = repr(record)
    assert repr_str.startswith("TestRecord(")
    assert "field1=10" in repr_str
    assert "field2='test'" in repr_str
    assert repr_str.endswith(")")


# LLM-generated content at query #7
#--------------------------

```python
def test_precord_repr_format():
    class TestRecord(PRecord):
        pass

    record = TestRecord(x=1, y="hello")
    result = repr(record)
    assert result.startswith("TestRecord(")
    assert "x=1" in result
    assert "y='hello'" in result
    assert result.endswith(")")


# LLM-generated content at query #8
#--------------------------

```python
def test_set_with_valid_field_and_factory():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'factory': lambda x: x, 'type': (int,), 'invariant': lambda x: (True, None)})()}
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap())
    result = evolver.set('field1', 10)
    assert result['field1'] == 10

def test_set_with_invalid_field():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'factory': lambda x: x, 'type': (int,), 'invariant': lambda x: (True, None)})()}
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap())
    try:
        evolver.set('invalid_field', 10)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'invalid_field' is not among the specified fields for TestRecord"

def test_set_with_type_error():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'factory': lambda x: x, 'type': (int,), 'invariant': lambda x: (True, None)})()}
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap())
    try:
        evolver.set('field1', 'not_an_int')
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.message == "Invalid type for field TestRecord.field1, was str"

def test_set_with_invariant_error():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'factory': lambda x: x, 'type': (int,), 'invariant': lambda x: (False, 'INVALID')})()}
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap())
    result = evolver.set('field1', 10)
    assert result._invariant_error_codes == ['INVALID']

def test_set_with_ignore_extra():
    class TestRecord:
        _precord_fields = {'field1': type('Field', (), {'factory': lambda x, ignore_extra=False: x, 'type': {CheckedType}, 'invariant': lambda x: (True, None)})()}
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap(), _ignore_extra=True)
    result = evolver.set('field1', 10)
    assert result['field1'] == 10


# LLM-generated content at query #9
#--------------------------

```python
def test__new__with_no_bases_and_no_fields():
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

def test__new__with_fields():
    class TestRecord(metaclass=_PRecordMeta):
        x = field()
        y = field(mandatory=True)
        z = field(initial=1)
    assert TestRecord._precord_fields == {'x': field(), 'y': field(mandatory=True), 'z': field(initial=1)}
    assert TestRecord._precord_mandatory_fields == {'y'}
    assert TestRecord._precord_initial_values == {'z': 1}

def test__new__with_invariant():
    def test_invariant(instance):
        return True, "test"
    class TestRecord(metaclass=_PRecordMeta):
        __invariant__ = test_invariant
    assert len(TestRecord._precord_invariants) == 1
    assert callable(TestRecord._precord_invariants[0])

def test__new__with_inherited_fields():
    class BaseRecord(metaclass=_PRecordMeta):
        x = field()
    class TestRecord(BaseRecord):
        y = field()
    assert TestRecord._precord_fields == {'x': field(), 'y': field()}

def test__new__with_inherited_invariant():
    def base_invariant(instance):
        return True, "base"
    class BaseRecord(metaclass=_PRecordMeta):
        __invariant__ = base_invariant
    class TestRecord(BaseRecord):
        pass
    assert len(TestRecord._precord_invariants) == 1
    assert callable(TestRecord._precord_invariants[0])

def test__new__with_multiple_inherited_invariants():
    def base_invariant1(instance):
        return True, "base1"
    def base_invariant2(instance):
        return True, "base2"
    class BaseRecord1(metaclass=_PRecordMeta):
        __invariant__ = base_invariant1
    class BaseRecord2(metaclass=_PRecordMeta):
        __invariant__ = base_invariant2
    class TestRecord(BaseRecord1, BaseRecord2):
        pass
    assert len(TestRecord._precord_invariants) == 2
    assert all(callable(inv) for inv in TestRecord._precord_invariants)

def test__new__with_non_callable_invariant():
    try:
        class TestRecord(metaclass=_PRecordMeta):
            __invariant__ = "not callable"
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"


# LLM-generated content at query #10
#--------------------------

```python
def test_serialize_without_custom_serializers():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert result == {"field1": "value1", "field2": "value2"}

def test_serialize_with_custom_serializers():
    def custom_serializer(value):
        return f"serialized_{value}"

    class TestRecord(PRecord):
        field1 = field(serializer=custom_serializer)
        field2 = field()

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert result == {"field1": "serialized_value1", "field2": "value2"}

def test_serialize_with_format_parameter():
    def custom_serializer(value, format=None):
        if format == "upper":
            return value.upper()
        return value

    class TestRecord(PRecord):
        field1 = field(serializer=custom_serializer)
        field2 = field()

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize(format="upper")
    assert result == {"field1": "VALUE1", "field2": "value2"}


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test_field_exists_in_precord_fields():
    class MockField:
        pass

    class MockDestinationCls:
        _precord_fields = {'test_key': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap())
    field = evolver._destination_cls._precord_fields.get('test_key')
    assert field is not None


# LLM-generated content at query #13
#--------------------------

```python
def test_persistent_raises_when_invariant_errors_or_missing_fields():
    class MockPMap:
        class _Evolver:
            def __init__(self, original_pmap):
                self._original_pmap = original_pmap

            def is_dirty(self):
                return False

            def persistent(self):
                return self._original_pmap

    class MockField:
        def __init__(self, name):
            self.name = name

    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(MockPRecord, MockPMap(), _factory_fields=None, _ignore_extra=False)
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['missing1']

    try:
        evolver.persistent()
        assert False, "Expected InvariantException to be raised"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ('missing1',)
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #14
#--------------------------

```python
def test_persistent_with_dirty_and_non_instance():
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

def test_persistent_with_clean_and_instance():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    original_pmap = PMap(key='value')
    evolver = _PRecordEvolver(cls, original_pmap)
    result = evolver.persistent()
    assert result is original_pmap

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
        return False, 'INVARIANT_ERROR'

    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [failing_invariant],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVARIANT_ERROR' in e.invariant_errors

def test_persistent_with_field_invariant_errors():
    field = type('MockField', (), {
        'invariant': lambda value: (False, 'FIELD_INVARIANT_ERROR'),
        'factory': lambda value: value
    })()
    cls = type('MockClass', (), {
        '_precord_fields': {'field': field},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('field', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'FIELD_INVARIANT_ERROR' in e.invariant_errors


# LLM-generated content at query #15
#--------------------------

```python
def test_repr_format():
    class TestRecord(PRecord):
        pass

    record = TestRecord(a=1, b="test")
    repr_str = repr(record)
    assert repr_str.startswith("TestRecord(") and repr_str.endswith(")")


# LLM-generated content at query #16
#--------------------------

```python
def test_set_fields_called_before_store_invariants():
    class TestClass(metaclass=_PRecordMeta):
        pass

    assert '_precord_fields' in TestClass.__dict__
    assert '_precord_invariants' in TestClass.__dict__


# LLM-generated content at query #17
#--------------------------

```python
def test_field_exists_in_precord_fields():
    class MockField:
        pass

    class MockDestinationCls:
        _precord_fields = {'test_key': MockField()}

    evolver = _PRecordEvolver(MockDestinationCls, PMap(), _factory_fields=None, _ignore_extra=False)
    field = evolver._destination_cls._precord_fields.get('test_key')
    assert field is not None


# LLM-generated content at query #18
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

    record = TestRecord(x=10, y=20, _factory_fields={'x': 100})
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_precord_size_and_buckets():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(_precord_size=2, _precord_buckets=[('x', 1), ('y', 2)])
    assert record.x == 1
    assert record.y == 2


# LLM-generated content at query #19
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    cls = PRecord
    size = 2
    buckets = [None, [("a", 1), ("b", 2)]]
    result = cls.__new__(cls, _precord_size=size, _precord_buckets=buckets)
    assert result._size == size
    assert result._buckets == buckets

def test_precord_new_without_precord_size_and_buckets():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {"a": None, "b": None}
    TestRecord._precord_initial_values = {}
    result = TestRecord.__new__(TestRecord, a=1, b=2)
    assert result["a"] == 1
    assert result["b"] == 2

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {"a": None, "b": None}
    TestRecord._precord_initial_values = {}
    factory_fields = {"a"}
    result = TestRecord.__new__(TestRecord, _factory_fields=factory_fields, a=1, b=2)
    assert result["a"] == 1
    assert result["b"] == 2

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {"a": None}
    TestRecord._precord_initial_values = {}
    result = TestRecord.__new__(TestRecord, _ignore_extra=True, a=1, b=2)
    assert result["a"] == 1
    assert "b" not in result

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {"a": None, "b": None}
    TestRecord._precord_initial_values = {"a": 10, "b": lambda: 20}
    result = TestRecord.__new__(TestRecord)
    assert result["a"] == 10
    assert result["b"] == 20

def test_precord_new_with_initial_values_and_kwargs():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {"a": None, "b": None}
    TestRecord._precord_initial_values = {"a": 10, "b": lambda: 20}
    result = TestRecord.__new__(TestRecord, a=1, b=2)
    assert result["a"] == 1
    assert result["b"] == 2


# LLM-generated content at query #20
#--------------------------

```python
def test_serialize_returns_dict():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #21
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    result = PRecord(_precord_size=10, _precord_buckets=[('a', 1), ('b', 2)])
    assert isinstance(result, PRecord)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_constructor_with_factory_fields():
    result = PRecord(_factory_fields=['x', 'y'], x=10, y=20)
    assert result['x'] == 10
    assert result['y'] == 20

def test_precord_constructor_with_ignore_extra():
    result = PRecord(_ignore_extra=True, x=10, y=20, z=30)
    assert result['x'] == 10
    assert result['y'] == 20
    assert 'z' not in result

def test_precord_constructor_with_initial_values():
    PRecord._precord_initial_values = {'a': 1, 'b': lambda: 2}
    result = PRecord()
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_constructor_with_initial_values_and_kwargs():
    PRecord._precord_initial_values = {'a': 1, 'b': 2}
    result = PRecord(c=3)
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #22
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    cls = type('MockPRecord', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('new_key', 'new_value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result.get('new_key') == 'new_value'

def test_persistent_with_clean_and_cls_instance():
    cls = type('MockPRecord', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = cls(_precord_buckets=PMap()._buckets, _precord_size=PMap()._size)
    evolver = _PRecordEvolver(cls, original_pmap)
    result = evolver.persistent()
    assert result is original_pmap

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockPRecord', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'mandatory_field'}, '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockPRecord.mandatory_field' in e.missing_fields

def test_persistent_with_invariant_errors():
    def failing_invariant(value):
        return (False, 'INVARIANT_ERROR')

    field = type('MockField', (), {'invariant': failing_invariant, 'factory': lambda x: x})()
    cls = type('MockPRecord', (), {
        '_precord_fields': {'field': field},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': []
    })
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('field', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVARIANT_ERROR' in e.invariant_errors

def test_persistent_with_global_invariant_failure():
    def global_invariant(subject):
        return (False, 'GLOBAL_INVARIANT_ERROR')

    cls = type('MockPRecord', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [global_invariant]
    })
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'GLOBAL_INVARIANT_ERROR' in e.invariant_errors


# LLM-generated content at query #23
#--------------------------

```python
def test__new__with_no_bases_and_empty_dct():
    class TestClass(metaclass=_PRecordMeta):
        pass
    assert hasattr(TestClass, '_precord_fields')
    assert TestClass._precord_fields == {}
    assert hasattr(TestClass, '_precord_invariants')
    assert TestClass._precord_invariants == ()
    assert hasattr(TestClass, '_precord_mandatory_fields')
    assert TestClass._precord_mandatory_fields == set()
    assert hasattr(TestClass, '_precord_initial_values')
    assert TestClass._precord_initial_values == {}
    assert hasattr(TestClass, '__slots__')
    assert TestClass.__slots__ == ()

def test__new__with_single_base_and_inherited_fields():
    class BaseClass(metaclass=_PRecordMeta):
        x = _PField()
        y = _PField(mandatory=True, initial=1)

    class TestClass(BaseClass):
        pass

    assert TestClass._precord_fields == {'x': BaseClass._precord_fields['x'], 'y': BaseClass._precord_fields['y']}
    assert TestClass._precord_invariants == BaseClass._precord_invariants
    assert TestClass._precord_mandatory_fields == {'y'}
    assert TestClass._precord_initial_values == {'y': 1}
    assert TestClass.__slots__ == ()

def test__new__with_multiple_bases_and_inherited_fields():
    class BaseClass1(metaclass=_PRecordMeta):
        x = _PField()
        y = _PField(mandatory=True, initial=1)

    class BaseClass2(metaclass=_PRecordMeta):
        z = _PField(mandatory=True)
        w = _PField(initial=2)

    class TestClass(BaseClass1, BaseClass2):
        pass

    expected_fields = {
        'x': BaseClass1._precord_fields['x'],
        'y': BaseClass1._precord_fields['y'],
        'z': BaseClass2._precord_fields['z'],
        'w': BaseClass2._precord_fields['w']
    }
    assert TestClass._precord_fields == expected_fields
    assert TestClass._precord_invariants == BaseClass1._precord_invariants + BaseClass2._precord_invariants
    assert TestClass._precord_mandatory_fields == {'y', 'z'}
    assert TestClass._precord_initial_values == {'y': 1, 'w': 2}
    assert TestClass.__slots__ == ()

def test__new__with_invariant_function():
    def test_invariant(obj):
        return True, "Test"

    class TestClass(metaclass=_PRecordMeta):
        __invariant__ = test_invariant

    assert len(TestClass._precord_invariants) == 1
    assert callable(TestClass._precord_invariants[0])

def test__new__with_non_callable_invariant_raises_type_error():
    try:
        class TestClass(metaclass=_PRecordMeta):
            __invariant__ = "not callable"
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"

def test__new__with_field_in_dct():
    class TestClass(metaclass=_PRecordMeta):
        x = _PField()
        y = _PField(mandatory=True, initial=1)

    assert TestClass._precord_fields == {'x': TestClass._precord_fields['x'], 'y': TestClass._precord_fields['y']}
    assert TestClass._precord_mandatory_fields == {'y'}
    assert TestClass._precord_initial_values == {'y': 1}
    assert not hasattr(TestClass, 'x')
    assert not hasattr(TestClass, 'y')

def test__new__with_mixed_fields_and_invariants_in_bases():
    def base_invariant(obj):
        return True, "Base"

    class BaseClass(metaclass=_PRecordMeta):
        __invariant__ = base_invariant
        x = _PField()

    def test_invariant(obj):
        return True, "Test"

    class TestClass(BaseClass):
        __invariant__ = test_invariant
        y = _PField(mandatory=True, initial=1)

    assert TestClass._precord_fields == {'x': BaseClass._precord_fields['x'], 'y': TestClass._precord_fields['y']}
    assert len(TestClass._precord_invariants) == 2
    assert TestClass._precord_mandatory_fields == {'y'}
    assert TestClass._precord_initial_values == {'y': 1}


# LLM-generated content at query #24
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=10, field2="test")
    assert record.field1 == 10
    assert record.field2 == "test"

def test_precord_constructor_with_default_values():
    class TestRecord(PRecord):
        field1 = field(initial=42)
        field2 = field(initial="default")

    record = TestRecord()
    assert record.field1 == 42
    assert record.field2 == "default"

def test_precord_constructor_with_callable_default():
    class TestRecord(PRecord):
        field1 = field(initial=lambda: [1, 2, 3])

    record = TestRecord()
    assert record.field1 == [1, 2, 3]

def test_precord_constructor_with_extra_fields_ignored():
    class TestRecord(PRecord):
        field1 = field()

    record = TestRecord(field1=10, extra_field=20, _ignore_extra=True)
    assert record.field1 == 10
    assert "extra_field" not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=10, _factory_fields=["field1"])
    assert record.field1 == 10
    assert record.field2 is None

def test_precord_constructor_with_internal_structure():
    class TestRecord(PRecord):
        field1 = field()

    record = TestRecord(_precord_size=1, _precord_buckets=[("field1", 10)])
    assert record.field1 == 10


# LLM-generated content at query #25
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_pmap():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result.get('key') == 'value'

def test_persistent_with_clean_and_cls_pmap():
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

    field = type('MockField', (), {'factory': lambda x: x, 'invariant': failing_invariant})()
    cls = type('MockClass', (), {'_precord_fields': {'field': field}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('field', 'value')
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
        assert 'Global invariant failed' in str(e)


# LLM-generated content at query #26
#--------------------------

```python
def test_serialize_returns_dict():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field(serializer=lambda x: x.upper())

    record = TestRecord(field1="value1", field2="value2")
    result = record.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #27
#--------------------------

```python
def test_missing_mandatory_fields_are_detected():
    class TestRecord:
        _precord_mandatory_fields = {'field1', 'field2'}
        __name__ = 'TestRecord'

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver.set('field1', 'value1')
    result = evolver.persistent()

    assert evolver._missing_fields == ('TestRecord.field2',)


# LLM-generated content at query #28
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        a = 1
        b = 2

    record = TestRecord(a=10, b=20)
    assert record.a == 10
    assert record.b == 20

def test_precord_constructor_with_default_values():
    class TestRecord(PRecord):
        a = 1
        b = 2

    record = TestRecord()
    assert record.a == 1
    assert record.b == 2

def test_precord_constructor_with_callable_defaults():
    class TestRecord(PRecord):
        a = lambda: 1
        b = lambda: 2

    record = TestRecord()
    assert record.a == 1
    assert record.b == 2

def test_precord_constructor_with_extra_fields_ignored():
    class TestRecord(PRecord):
        a = 1
        b = 2

    record = TestRecord(a=10, b=20, c=30, _ignore_extra=True)
    assert record.a == 10
    assert record.b == 20
    assert 'c' not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        a = 1
        b = 2

    record = TestRecord(a=10, _factory_fields={'b': 20})
    assert record.a == 10
    assert record.b == 20

def test_precord_constructor_with_internal_params():
    class TestRecord(PRecord):
        a = 1
        b = 2

    record = TestRecord(_precord_size=2, _precord_buckets=[('a', 1), ('b', 2)])
    assert record.a == 1
    assert record.b == 2


# LLM-generated content at query #29
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver.set('test_key', 'test_value')
    assert evolver.is_dirty()
    result = evolver.persistent()
    assert isinstance(result, TestClass)


# LLM-generated content at query #30
#--------------------------

```python
def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass

    result = TestRecord.__new__(TestRecord)
    assert not ('_precord_size' in result and '_precord_buckets' in result)


# LLM-generated content at query #31
#--------------------------

```python
def test_precord_meta_new_creates_slots():
    class TestClass(metaclass=_PRecordMeta):
        pass
    assert '__slots__' in TestClass.__dict__
    assert TestClass.__slots__ == ()


# LLM-generated content at query #32
#--------------------------

```python
def test_predicate_at_line_1():
    class TestClass(metaclass=_PRecordMeta):
        pass

    assert isinstance(TestClass, _PRecordMeta)


# LLM-generated content at query #33
#--------------------------

```python
def test_persistent_raises_when_invariant_errors_or_missing_fields():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['field1']

    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ('field1',)
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #34
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1=1, field2="test")
    assert record.field1 == 1
    assert record.field2 == "test"

def test_precord_constructor_with_extra_fields_ignored():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1=1, field2="test", extra_field="ignored", _ignore_extra=True)
    assert record.field1 == 1
    assert record.field2 == "test"
    assert "extra_field" not in record

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        field1 = 0
        field2 = "default"

    record = TestRecord()
    assert record.field1 == 0
    assert record.field2 == "default"

def test_precord_constructor_with_callable_initial_values():
    class TestRecord(PRecord):
        field1 = lambda: 42
        field2 = "default"

    record = TestRecord()
    assert record.field1 == 42
    assert record.field2 == "default"

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(_factory_fields=["field1"], field1=1, field2="test")
    assert record.field1 == 1
    assert record.field2 == "test"

def test_precord_constructor_with_internal_params():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(_precord_size=2, _precord_buckets=[("field1", 1), ("field2", "test")])
    assert record.field1 == 1
    assert record.field2 == "test"


# LLM-generated content at query #35
#--------------------------

```python
def test_precord_constructor_with_valid_fields():
    class TestRecord(PRecord):
        field1 = 0
        field2 = "default"

    record = TestRecord(field1=1, field2="value")
    assert record.field1 == 1
    assert record.field2 == "value"

def test_precord_constructor_with_default_values():
    class TestRecord(PRecord):
        field1 = 0
        field2 = "default"

    record = TestRecord()
    assert record.field1 == 0
    assert record.field2 == "default"

def test_precord_constructor_with_callable_defaults():
    class TestRecord(PRecord):
        field1 = lambda: 42
        field2 = "default"

    record = TestRecord()
    assert record.field1 == 42
    assert record.field2 == "default"

def test_precord_constructor_with_extra_fields_ignored():
    class TestRecord(PRecord):
        field1 = 0

    record = TestRecord(field1=1, extra_field="ignored", _ignore_extra=True)
    assert record.field1 == 1
    assert "extra_field" not in record

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        field1 = 0
        field2 = "default"

    record = TestRecord(field1=1, _factory_fields=["field1"])
    assert record.field1 == 1
    assert record.field2 == "default"

def test_precord_constructor_with_internal_params():
    record = PRecord(_precord_size=1, _precord_buckets=[("key", "value")])
    assert record["key"] == "value"


# LLM-generated content at query #36
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver.set('test_key', 'test_value')
    pm = evolver.persistent()
    assert isinstance(pm, TestClass)


# LLM-generated content at query #37
#--------------------------

```python
def test_persistent_predicate_false():
    class TestClass:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver._is_dirty = False
    evolver._root = TestClass(_precord_buckets=[], _precord_size=0)
    result = evolver.persistent()
    assert isinstance(result, TestClass)


# LLM-generated content at query #38
#--------------------------

```python
def test_precord_new_with_precord_size_and_buckets():
    buckets = [(('a', 1), ('b', 2)), None, None, None]
    result = PRecord.__new__(PRecord, _precord_size=2, _precord_buckets=buckets)
    assert result._size == 2
    assert result._buckets == pvector().extend(buckets)

def test_precord_new_without_precord_size_and_buckets():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2)
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2, _factory_fields=['a'])
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2, c=3, _ignore_extra=True)
    assert result['a'] == 1
    assert result['b'] == 2
    assert 'c' not in result

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': 2}
    result = TestRecord()
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_initial_values_and_kwargs():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': 2}
    result = TestRecord(a=3, c=4)
    assert result['a'] == 3
    assert result['b'] == 2
    assert result['c'] == 4


# LLM-generated content at query #39
#--------------------------

```python
def test_persistent_raises_when_invariant_errors_or_missing_fields():
    class TestRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

    evolver = _PRecordEvolver(TestRecord, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['field1']

    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #40
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


# LLM-generated content at query #41
#--------------------------

```python
def test_persistent_returns_same_instance_when_not_dirty_and_same_type():
    cls = type('MockPRecord', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    result = evolver.persistent()
    assert result is original_pmap

def test_persistent_creates_new_instance_when_dirty():
    cls = type('MockPRecord', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('key', 'value')
    result = evolver.persistent()
    assert result is not original_pmap
    assert isinstance(result, cls)

def test_persistent_creates_new_instance_when_not_same_type():
    cls = type('MockPRecord', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap({'key': 'value'})
    evolver = _PRecordEvolver(cls, original_pmap)
    result = evolver.persistent()
    assert result is not original_pmap
    assert isinstance(result, cls)

def test_persistent_raises_invariant_exception_for_missing_mandatory_fields():
    cls = type('MockPRecord', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'mandatory_field'}, '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('MockPRecord.mandatory_field',)

def test_persistent_raises_invariant_exception_for_field_invariant_failure():
    def failing_invariant(value):
        return (False, 'INVARIANT_FAILED')

    field = type('MockField', (), {'invariant': failing_invariant})
    cls = type('MockPRecord', (), {'_precord_fields': {'field': field}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver.set('field', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('INVARIANT_FAILED',)

def test_persistent_raises_invariant_exception_for_global_invariant_failure():
    def failing_global_invariant(subject):
        return (False, 'GLOBAL_INVARIANT_FAILED')

    cls = type('MockPRecord', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_global_invariant]})
    original_pmap = PMap()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('GLOBAL_INVARIANT_FAILED',)


# LLM-generated content at query #42
#--------------------------

```python
def test_precord_metaclass_creates_mandatory_fields_set():
    class TestRecord(metaclass=_PRecordMeta):
        __invariant__ = lambda self: True
        a = _PField(mandatory=True)
        b = _PField(mandatory=False)

    assert isinstance(TestRecord._precord_mandatory_fields, set)
    assert 'a' in TestRecord._precord_mandatory_fields
    assert 'b' not in TestRecord._precord_mandatory_fields


# LLM-generated content at query #43
#--------------------------

```python
def test_set_fields_and_store_invariants_are_called():
    class TestMeta(_PRecordMeta):
        def __new__(mcs, name, bases, dct):
            set_fields_called = False
            store_invariants_called = False

            def mock_set_fields(*args, **kwargs):
                nonlocal set_fields_called
                set_fields_called = True
                return dct

            def mock_store_invariants(*args, **kwargs):
                nonlocal store_invariants_called
                store_invariants_called = True

            import pyrsistent._field_common as field_common
            import pyrsistent._checked_types as checked_types
            original_set_fields = field_common.set_fields
            original_store_invariants = checked_types.store_invariants
            field_common.set_fields = mock_set_fields
            checked_types.store_invariants = mock_store_invariants

            try:
                result = super().__new__(mcs, name, bases, dct)
            finally:
                field_common.set_fields = original_set_fields
                checked_types.store_invariants = original_store_invariants

            assert set_fields_called
            assert store_invariants_called
            return result

    class TestClass(metaclass=TestMeta):
        pass


# LLM-generated content at query #44
#--------------------------

```python
def test_serialize_without_custom_serializer():
    class TestRecord(PRecord):
        field1 = Field()
        field2 = Field()

    record = TestRecord(field1=10, field2="test")
    result = record.serialize()
    assert result == {"field1": 10, "field2": "test"}

def test_serialize_with_custom_serializer():
    class TestRecord(PRecord):
        field1 = Field(serializer=lambda x: str(x))
        field2 = Field(serializer=lambda x: x.upper())

    record = TestRecord(field1=10, field2="test")
    result = record.serialize()
    assert result == {"field1": "10", "field2": "TEST"}

def test_serialize_with_format_parameter():
    class TestRecord(PRecord):
        field1 = Field(serializer=lambda x, fmt: f"{fmt}:{x}")
        field2 = Field(serializer=lambda x, fmt: f"{fmt}:{x.upper()}")

    record = TestRecord(field1=10, field2="test")
    result = record.serialize(format="custom")
    assert result == {"field1": "custom:10", "field2": "custom:TEST"}


# LLM-generated content at query #45
#--------------------------

```python
def test_new_with_no_bases_and_no_fields():
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
    assert TestRecord.__slots__ == ()

def test_new_with_inherited_fields():
    class BaseRecord(metaclass=_PRecordMeta):
        x = _PField(mandatory=True, initial=1)

    class DerivedRecord(BaseRecord):
        pass

    assert DerivedRecord._precord_fields == {'x': _PField(mandatory=True, initial=1)}
    assert DerivedRecord._precord_mandatory_fields == {'x'}
    assert DerivedRecord._precord_initial_values == {'x': 1}

def test_new_with_inherited_invariants():
    def test_invariant(obj):
        return True, "OK"

    class BaseRecord(metaclass=_PRecordMeta):
        __invariant__ = test_invariant

    class DerivedRecord(BaseRecord):
        pass

    assert len(DerivedRecord._precord_invariants) == 1
    assert callable(DerivedRecord._precord_invariants[0])
    assert DerivedRecord._precord_invariants[0](None) == (True, "OK")

def test_new_with_multiple_inherited_invariants():
    def test_invariant1(obj):
        return True, "OK1"

    def test_invariant2(obj):
        return True, "OK2"

    class BaseRecord1(metaclass=_PRecordMeta):
        __invariant__ = test_invariant1

    class BaseRecord2(metaclass=_PRecordMeta):
        __invariant__ = test_invariant2

    class DerivedRecord(BaseRecord1, BaseRecord2):
        pass

    assert len(DerivedRecord._precord_invariants) == 2
    assert DerivedRecord._precord_invariants[0](None) == (True, "OK1")
    assert DerivedRecord._precord_invariants[1](None) == (True, "OK2")

def test_new_with_non_callable_invariant_raises_type_error():
    with pytest.raises(TypeError, match="Invariants must be callable"):
        class TestRecord(metaclass=_PRecordMeta):
            __invariant__ = "not callable"


# LLM-generated content at query #46
#--------------------------

```python
def test_precord_new_with_special_attributes():
    result = PRecord(_precord_size=2, _precord_buckets=[[(1, 'a'), (2, 'b')], []])
    assert isinstance(result, PRecord)
    assert len(result) == 2
    assert result[1] == 'a'
    assert result[2] == 'b'

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2)
    assert isinstance(result, TestRecord)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_initial_values():
    class TestRecord(PRecord):
        _precord_initial_values = {'a': 1, 'b': lambda: 2}
    result = TestRecord()
    assert isinstance(result, TestRecord)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2, _factory_fields={'a', 'b'})
    assert isinstance(result, TestRecord)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    result = TestRecord(a=1, b=2, c=3, ignore_extra=True)
    assert isinstance(result, TestRecord)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #47
#--------------------------

```python
def test_precord_constructor_with_valid_kwargs():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_initial_values():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord()
    assert record.x == 0
    assert record.y == 1

def test_precord_constructor_with_callable_initial_values():
    class TestRecord(PRecord):
        x = lambda: 0
        y = lambda: 1

    record = TestRecord()
    assert record.x == 0
    assert record.y == 1

def test_precord_constructor_with_factory_fields():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord._factory_fields({'x': 10, 'y': 20})
    assert record.x == 10
    assert record.y == 20

def test_precord_constructor_with_ignore_extra():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord._ignore_extra(True, x=10, y=20, z=30)
    assert record.x == 10
    assert record.y == 20
    assert 'z' not in record

def test_precord_constructor_with_precord_size_and_buckets():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(_precord_size=2, _precord_buckets=[('x', 10), ('y', 20)])
    assert record.x == 10
    assert record.y == 20


# LLM-generated content at query #48
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
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'mandatory'}, '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MockClass.mandatory' in e.missing_fields

def test_persistent_with_invariant_errors():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver._invariant_error_codes = ['error1']
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)

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


# LLM-generated content at query #49
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._buckets = {'key': 'value'}
    evolver._size = 1
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result._precord_buckets == {'key': 'value'}
    assert result._precord_size == 1

def test_persistent_with_clean_and_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver = _PRecordEvolver(cls, original_pmap)
    result = evolver.persistent()
    assert result is original_pmap

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'field1'}, '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('MockClass.field1',)

def test_persistent_with_invariant_error_codes():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
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

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0})()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)


# LLM-generated content at query #50
#--------------------------

```python
def test_missing_fields_added_when_mandatory_fields_exist():
    class TestClass:
        _precord_mandatory_fields = {'field1', 'field2'}
        __name__ = 'TestClass'

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver._missing_fields = []
    evolver._invariant_error_codes = []
    evolver.persistent()
    assert evolver._missing_fields == ('TestClass.field1', 'TestClass.field2')


# LLM-generated content at query #51
#--------------------------

```python
def test_persistent_predicate_false():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    evolver = _PRecordEvolver(cls, PMap())
    evolver._is_dirty = lambda: False
    pm = cls()
    assert not (evolver.is_dirty() or not isinstance(pm, cls))


# LLM-generated content at query #52
#--------------------------

```python
def test_precord_new_with_special_attributes():
    result = PRecord(_precord_size=2, _precord_buckets=[[('a', 1)], [('b', 2)]])
    assert result == {'a': 1, 'b': 2}

def test_precord_new_without_special_attributes():
    result = PRecord(a=1, b=2)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_factory_fields():
    result = PRecord(a=1, b=2, _factory_fields={'a'})
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_ignore_extra():
    result = PRecord(a=1, b=2, ignore_extra=True)
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


# LLM-generated content at query #53
#--------------------------

```python
def test_precord_initial_values_are_used():
    class TestRecord(PRecord):
        _precord_fields = {'a': None, 'b': None}
        _precord_initial_values = {'a': 1, 'b': 2}

    result = TestRecord()
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #54
#--------------------------

```python
def test_missing_fields_added_when_mandatory_fields_exist():
    class TestClass:
        _precord_mandatory_fields = {'field1', 'field2'}
        __name__ = 'TestClass'

    evolver = _PRecordEvolver(TestClass, PMap())
    evolver._missing_fields = []
    evolver.persistent()
    assert evolver._missing_fields == ('TestClass.field1', 'TestClass.field2')


# LLM-generated content at query #55
#--------------------------

```python
def test_precord_constructor_with_special_attributes():
    kwargs = {'_precord_size': 2, '_precord_buckets': [('a', 1), ('b', 2)]}
    result = PRecord(**kwargs)
    assert isinstance(result, PRecord)
    assert len(result) == 2
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
def test_persistent_raises_when_invariant_errors_or_missing_fields():
    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = ()

    evolver = _PRecordEvolver(MockPRecord, PMap())
    evolver._invariant_error_codes = ['error1']
    evolver._missing_fields = ['field1']

    with pytest.raises(InvariantException):
        evolver.persistent()


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_persistent_with_dirty_and_non_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0, 'is_dirty': lambda: True})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._set_dirty(True)
    result = evolver.persistent()
    assert isinstance(result, cls)
    assert result._precord_buckets == original_pmap._buckets
    assert result._precord_size == original_pmap._size

def test_persistent_with_clean_and_cls_instance():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0, 'is_dirty': lambda: False})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._set_dirty(False)
    result = evolver.persistent()
    assert result is original_pmap

def test_persistent_with_missing_mandatory_fields():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': {'field1'}, '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0, 'is_dirty': lambda: False})()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('MockClass.field1',)
        assert e.invariant_errors == ()
        assert str(e) == 'Field invariant failed'

def test_persistent_with_invariant_errors():
    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': []})
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0, 'is_dirty': lambda: False})()
    evolver = _PRecordEvolver(cls, original_pmap)
    evolver._invariant_error_codes = ['error1', 'error2']
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('error1', 'error2')
        assert e.missing_fields == ()
        assert str(e) == 'Field invariant failed'

def test_persistent_with_global_invariant_failure():
    def failing_invariant(subject):
        return (False, 'global_error')

    cls = type('MockClass', (), {'_precord_fields': {}, '_precord_mandatory_fields': set(), '_precord_invariants': [failing_invariant]})
    original_pmap = type('MockPMap', (), {'_buckets': {}, '_size': 0, 'is_dirty': lambda: False})()
    evolver = _PRecordEvolver(cls, original_pmap)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('global_error',)
        assert e.missing_fields == ()
        assert str(e) == 'Global invariant failed'


# LLM-generated content at query #2
#--------------------------

```python
def test_invariant_exception_raised_when_error_codes_or_missing_fields():
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
        assert e.message == 'Field invariant failed'


# LLM-generated content at query #3
#--------------------------

```python
def test__new__stores_fields_and_invariants():
    class Parent:
        __metaclass__ = _PRecordMeta
        x = _PField(mandatory=True, initial=1)

    class Child(Parent):
        __metaclass__ = _PRecordMeta
        y = _PField(mandatory=False, initial=2)

    assert '_precord_fields' in Child.__dict__
    assert 'x' in Child._precord_fields
    assert 'y' in Child._precord_fields
    assert Child._precord_fields['x'].mandatory
    assert not Child._precord_fields['y'].mandatory
    assert Child._precord_fields['x'].initial == 1
    assert Child._precord_fields['y'].initial == 2
    assert 'x' in Child._precord_mandatory_fields
    assert 'y' not in Child._precord_mandatory_fields
    assert Child._precord_initial_values == {'x': 1, 'y': 2}
    assert Child.__slots__ == ()


# LLM-generated content at query #4
#--------------------------

```python
def test_precord_new_with_special_attributes():
    result = PRecord.__new__(PRecord, _precord_size=2, _precord_buckets=[[('a', 1)], [('b', 2)]])
    assert result._size == 2
    assert result._buckets == [[('a', 1)], [('b', 2)]]

def test_precord_new_without_special_attributes():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'x': 1, 'y': 2}
    TestRecord._precord_initial_values = {'x': 10, 'y': 20}
    result = TestRecord.__new__(TestRecord, x=100, y=200)
    assert result['x'] == 100
    assert result['y'] == 200

def test_precord_new_with_factory_fields():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'x': 1, 'y': 2}
    result = TestRecord.__new__(TestRecord, _factory_fields={'x'}, x=100, y=200)
    assert result['x'] == 100
    assert result['y'] == 200

def test_precord_new_with_ignore_extra():
    class TestRecord(PRecord):
        pass
    TestRecord._precord_fields = {'x': 1, 'y': 2}
    result = TestRecord.__new__(TestRecord, _ignore_extra=True, x=100, y=200, z=300)
    assert result['x'] == 100
    assert result['y'] == 200
    assert 'z' not in result


# LLM-generated content at query #5
#--------------------------

```python
def test_precord_new_with_special_attributes():
    cls = PRecord
    size = 5
    buckets = [(('a', 1),), (('b', 2),), None, None, None]
    result = cls.__new__(cls, _precord_size=size, _precord_buckets=buckets)
    assert isinstance(result, PRecord)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_without_special_attributes():
    cls = PRecord
    result = cls.__new__(cls, a=1, b=2)
    assert isinstance(result, PRecord)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_initial_values():
    cls = PRecord
    cls._precord_initial_values = {'a': 1, 'b': lambda: 2}
    result = cls.__new__(cls, b=3)
    assert isinstance(result, PRecord)
    assert result == {'a': 1, 'b': 3}

def test_precord_new_with_factory_fields():
    cls = PRecord
    factory_fields = {'a', 'b'}
    result = cls.__new__(cls, a=1, b=2, _factory_fields=factory_fields)
    assert isinstance(result, PRecord)
    assert result == {'a': 1, 'b': 2}

def test_precord_new_with_ignore_extra():
    cls = PRecord
    result = cls.__new__(cls, a=1, b=2, c=3, _ignore_extra=True)
    assert isinstance(result, PRecord)
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #6
#--------------------------

```python
def test_persistent_creates_new_instance_when_dirty_or_not_instance():
    class MockPMap:
        _buckets = (1, 2, 3)
        _size = 3

    class MockPRecord:
        _precord_fields = {}
        _precord_mandatory_fields = set()
        _precord_invariants = []

        def __init__(self, _precord_buckets, _precord_size):
            self._buckets = _precord_buckets
            self._size = _precord_size

    evolver = _PRecordEvolver(MockPRecord, MockPMap())
    evolver._is_dirty = True
    evolver._destination_cls = MockPRecord
    evolver._invariant_error_codes = []
    evolver._missing_fields = []

    result = evolver.persistent()

    assert isinstance(result, MockPRecord)
    assert result._buckets == (1, 2, 3)
    assert result._size == 3


# LLM-generated content at query #7
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

    record = MultiFieldRecord(field1="value1", field2=2, field3=3.0)
    assert repr(record) == "MultiFieldRecord(field1='value1', field2=2, field3=3.0)"

def test_repr_with_nested_structures():
    class NestedRecord(PRecord):
        field1 = Field()
        field2 = Field()

    inner_record = NestedRecord(field1="inner", field2=10)
    outer_record = NestedRecord(field1="outer", field2=inner_record)
    assert repr(outer_record) == "NestedRecord(field1='outer', field2=NestedRecord(field1='inner', field2=10))"

def test_repr_with_special_characters():
    class SpecialCharsRecord(PRecord):
        field1 = Field()

    record = SpecialCharsRecord(field1="value with spaces and 'quotes'")
    assert repr(record) == "SpecialCharsRecord(field1='value with spaces and \\'quotes\\'')"

def test_repr_with_none_value():
    class NoneValueRecord(PRecord):
        field1 = Field()

    record = NoneValueRecord(field1=None)
    assert repr(record) == "NoneValueRecord(field1=None)"


# LLM-generated content at query #8
#--------------------------

```python
def test_persistent_with_dirty_and_non_instance():
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
    assert 'key' in result

def test_persistent_with_clean_and_instance():
    cls = type('MockClass', (), {
        '_precord_fields': {},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    original = cls(_precord_buckets={}, _precord_size=0)
    evolver = _PRecordEvolver(cls, original)
    result = evolver.persistent()
    assert result is original

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
        assert 'mandatory_field' in e.missing_fields

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
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVARIANT_FAILED' in e.invariant_errors

def test_persistent_with_field_invariant_errors():
    field = type('MockField', (), {
        'invariant': lambda value: (False, 'FIELD_INVARIANT_FAILED')
    })()
    cls = type('MockClass', (), {
        '_precord_fields': {'field': field},
        '_precord_mandatory_fields': set(),
        '_precord_invariants': [],
        '__name__': 'MockClass'
    })
    evolver = _PRecordEvolver(cls, PMap())
    evolver.set('field', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'FIELD_INVARIANT_FAILED' in e.invariant_errors


