####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields via kwargs (though set doesn't directly support this, evolver should handle it)
    evolver.set('field2', 'value2')
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant failure
    class InvariantRecord(PRecord):
        field = None

        @invariant
        def check_field(self):
            return self.field != 'invalid', 'INVALID_VALUE'

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('field', 'invalid')
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field check
    class MandatoryRecord(PRecord):
        mandatory_field = field(mandatory=True)

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #2
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant failure
    class TestRecordWithInvariant(PRecord):
        field1 = None

        @invariant
        def check_field1(self, record):
            if record.field1 == 'invalid':
                return False, 'INVALID_FIELD1'
            return True, None

    evolver_with_invariant = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_invariant.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver_with_invariant.persistent()

    # Test mandatory field
    class TestRecordWithMandatory(PRecord):
        mandatory_field = None

    evolver_with_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_with_mandatory.persistent()

    # Test ignore_extra
    class TestRecordWithFactory(PRecord):
        field1 = None

    evolver_with_factory = _PRecordEvolver(TestRecordWithFactory, pmap(), _ignore_extra=True)
    evolver_with_factory.set('field1', 'value1')
    assert evolver_with_factory['field1'] == 'value1'


# LLM-generated content at query #3
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    r = EmptyRecord()
    assert isinstance(r, EmptyRecord)
    assert len(r) == 0

    # Test creation with initial values
    class SimpleRecord(PRecord):
        x = 1
        y = 2

    r = SimpleRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with custom initial values
    r = SimpleRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with factory_fields
    class FactoryRecord(PRecord):
        x = 1
        y = 2

    r = FactoryRecord.create({'x': 10, 'y': 20}, _factory_fields=[FactoryRecord._precord_fields['x']])
    assert r.x == 10
    assert r.y == 2

    # Test creation with ignore_extra
    r = FactoryRecord.create({'x': 10, 'y': 20, 'z': 30}, ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with callable initial values
    class CallableRecord(PRecord):
        x = lambda: 1
        y = 2

    r = CallableRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with existing PRecord
    r1 = SimpleRecord(x=10, y=20)
    r2 = SimpleRecord.create(r1)
    assert r2.x == 10
    assert r2.y == 20


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test valid field set
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test invalid field set
    with pytest.raises(AttributeError):
        evolver.set('z', 20)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0
        __invariant__ = lambda self: (self.x >= 0, "x must be non-negative")

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test factory field
    class FactoryRecord(PRecord):
        x = 0
        y = field(factory=lambda x: x * 2)

    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('y', 5)
    assert evolver['y'] == 10

    # Test mandatory field
    class MandatoryRecord(PRecord):
        x = 0
        y = field(mandatory=True)

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('x', 10)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test ignore_extra
    evolver = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 10)
    evolver.set('z', 20)  # Should not raise AttributeError
    assert evolver['x'] == 10


# LLM-generated content at query #2
#--------------------------

```python
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test basic persistence
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 10
    evolver['y'] = 20
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == 20

    # Test with mandatory fields
    class MandatoryRecord(PRecord):
        a = 0
        b = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['a'] = 100
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['x'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes (not dirty)
    record = TestRecord(x=5, y=15)
    evolver = _PRecordEvolver(TestRecord, record)
    result = evolver.persistent()
    assert result is record

    # Test with global invariants
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

        @global_invariant
        def check_sum(self):
            return self.x + self.y > 0, "sum must be positive"

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #3
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field(serializer=lambda x: x.upper())

    record = TestRecord(field1="value1", field2="value2")

    # Test default serialization (no format specified)
    serialized = record.serialize()
    assert serialized == {"field1": "value1", "field2": "VALUE2"}

    # Test with custom format
    serialized_with_format = record.serialize(format="custom")
    assert serialized_with_format == {"field1": "value1", "field2": "VALUE2"}

    # Test with no serializer
    class SimpleRecord(PRecord):
        field1 = field()
        field2 = field()

    simple_record = SimpleRecord(field1="value1", field2="value2")
    assert simple_record.serialize() == {"field1": "value1", "field2": "value2"}


# LLM-generated content at query #4
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant failure
    class TestRecordWithInvariant(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, 'x must be non-negative'
            return True, None

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('x', -1)
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field check
    class TestRecordWithMandatory(PRecord):
        x = 0
        y = PFIELD_NO_INITIAL

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    evolver_mandatory.set('x', 10)
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()

    # Test factory field
    class TestRecordWithFactory(PRecord):
        x = 0

        def __factory__(self, value):
            return value * 2

    evolver_factory = _PRecordEvolver(TestRecordWithFactory, pmap())
    evolver_factory.set('x', 5)
    assert evolver_factory['x'] == 10

    # Test ignore_extra
    evolver_ignore = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver_ignore.set('x', 10)
    evolver_ignore.set('invalid_field', 20)  # Should not raise
    assert evolver_ignore['x'] == 10
    assert 'invalid_field' not in evolver_ignore


# LLM-generated content at query #5
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: len(self.name) > 0
        name = None
        age = 0

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert '_precord_fields' in TestRecord.__dict__
    assert '_precord_invariants' in TestRecord.__dict__
    assert '_precord_mandatory_fields' in TestRecord.__dict__
    assert '_precord_initial_values' in TestRecord.__dict__
    assert '__slots__' in TestRecord.__dict__


# LLM-generated content at query #6
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 0
        y = 1

    r = TestRecord()
    assert r.x == 0
    assert r.y == 1

    # Test creation with custom values
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with initial values
    class TestRecord2(PRecord):
        x = 0
        y = 1
        z = 2

    r = TestRecord2(x=10)
    assert r.x == 10
    assert r.y == 1
    assert r.z == 2

    # Test creation with callable initial values
    class TestRecord3(PRecord):
        x = 0
        y = lambda: 1
        z = 2

    r = TestRecord3()
    assert r.x == 0
    assert r.y == 1
    assert r.z == 2

    # Test creation with factory fields
    class TestRecord4(PRecord):
        x = 0
        y = 1

    r = TestRecord4(_factory_fields=[TestRecord4._precord_fields['x']], x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with ignore_extra
    class TestRecord5(PRecord):
        x = 0
        y = 1

    r = TestRecord5(_ignore_extra=True, x=10, y=20, z=30)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with _precord_size and _precord_buckets
    class TestRecord6(PRecord):
        x = 0
        y = 1

    r = TestRecord6(_precord_size=2, _precord_buckets=[('x', 10), ('y', 20)])
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #7
#--------------------------

```python
def test_PRecord___new__():
    # Test 1: Basic creation with required fields
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test 2: Creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test 3: Creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 1
        y = lambda: 2
        z = 3

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test 4: Creation with factory fields
    class TestRecordWithFactory(PRecord):
        x = 1
        y = 2

    r = TestRecordWithFactory._factory_fields(['x', 'y'], x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test 5: Creation with ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 1
        y = 2

    r = TestRecordWithIgnoreExtra._ignore_extra(True, x=10, y=20, z=30)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test 6: Creation with internal parameters
    r = TestRecordWithIgnoreExtra(_precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20}))
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #8
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 0
        y = 1

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 5

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 1
    assert r.z == 5

    # Test with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 0
        y = lambda: 100

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 100

    # Test with factory fields
    class TestRecordWithFactory(PRecord):
        x = 0
        y = 1

    r = TestRecordWithFactory._PRecordEvolver(TestRecordWithFactory, pmap(), _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    r['x'] = 10
    r['y'] = 20
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test with ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 0
        y = 1

    r = TestRecordWithIgnoreExtra(x=10, y=20, z=30, _ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test with mandatory fields
    class TestRecordWithMandatory(PRecord):
        x = 0
        y = 1

    try:
        TestRecordWithMandatory(x=10)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'y' in str(e)

    # Test with pickle support
    class TestRecordWithPickle(PRecord):
        x = 0
        y = 1

    r = TestRecordWithPickle(x=10, y=20)
    restored = _restore_pickle(TestRecordWithPickle, dict(r))
    assert restored.x == 10
    assert restored.y == 20


# LLM-generated content at query #9
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant violation
    class InvariantRecord(PRecord):
        field1 = None

        def __invariant__(self):
            if self.field1 == 'invalid':
                return False, 'INVALID_VALUE'
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        mandatory_field = None

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #10
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 0

    e = _PRecordEvolver(TestRecord, pmap())
    e['x'] = 1
    e['y'] = 2
    result = e.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with mandatory fields missing
    class MandatoryRecord(PRecord):
        x = 0
        y = 0
        z = 0

    e = _PRecordEvolver(MandatoryRecord, pmap())
    e['x'] = 1
    e['y'] = 2
    with pytest.raises(InvariantException):
        e.persistent()

    # Test with invariant violation
    class InvariantRecord(PRecord):
        x = 0
        y = 0

        def __invariant__(self):
            if self.x + self.y != 10:
                return False, "x + y must equal 10"
            return True, None

    e = _PRecordEvolver(InvariantRecord, pmap())
    e['x'] = 1
    e['y'] = 2
    with pytest.raises(InvariantException):
        e.persistent()

    # Test with no changes (not dirty)
    record = TestRecord(x=1, y=2)
    e = _PRecordEvolver(TestRecord, record)
    result = e.persistent()
    assert result is record

    # Test with type check failure
    class TypedRecord(PRecord):
        x = 0
        y = 0

    e = _PRecordEvolver(TypedRecord, pmap())
    e['x'] = "not an int"
    with pytest.raises(AttributeError):
        e.persistent()

    # Test with global invariant
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

    def global_invariant(record):
        if record.x > record.y:
            return False, "x must be <= y"
        return True, None

    GlobalInvariantRecord._precord_invariants = (global_invariant,)

    e = _PRecordEvolver(GlobalInvariantRecord, pmap())
    e['x'] = 5
    e['y'] = 3
    with pytest.raises(InvariantException):
        e.persistent()


# LLM-generated content at query #11
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['x'] == 10
    assert evolver['y'] == 20

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class ValidatedRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x >= 0, "x must be non-negative"

    evolver = _PRecordEvolver(ValidatedRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field check
    class MandatoryRecord(PRecord):
        x = field(mandatory=True)

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test factory field
    class FactoryRecord(PRecord):
        x = field(factory=lambda v: v * 2)

    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('x', 5)
    assert evolver['x'] == 10

    # Test ignore_extra
    evolver = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 10)
    evolver.set('extra_field', 20)  # Should not raise AttributeError
    assert evolver['x'] == 10
    assert 'extra_field' not in evolver  # Should not be set


# LLM-generated content at query #12
#--------------------------

```python
def test_PRecord___new__():
    # Test normal creation with kwargs
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 1
        y = lambda: 2
        z = 3

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with _factory_fields
    class TestRecordWithFactoryFields(PRecord):
        x = 1
        y = 2

    r = TestRecordWithFactoryFields._PRecordEvolver(TestRecordWithFactoryFields, pmap(), _factory_fields=[TestRecordWithFactoryFields._precord_fields['x']])
    r['x'] = 10
    r = r.persistent()
    assert r.x == 10
    assert r.y == 2

    # Test creation with _ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 1
        y = 2

    r = TestRecordWithIgnoreExtra._PRecordEvolver(TestRecordWithIgnoreExtra, pmap(), _ignore_extra=True)
    r['x'] = 10
    r['z'] = 30  # This should be ignored
    r = r.persistent()
    assert r.x == 10
    assert r.y == 2
    assert 'z' not in r

    # Test creation with _precord_size and _precord_buckets
    class TestRecordWithBuckets(PRecord):
        x = 1
        y = 2

    r = TestRecordWithBuckets._PRecordEvolver(TestRecordWithBuckets, pmap(), _factory_fields=None, _ignore_extra=False)
    r['x'] = 10
    r['y'] = 20
    r = r.persistent()
    assert r.x == 10
    assert r.y == 20

    # Test creation with mandatory fields
    class TestRecordWithMandatoryFields(PRecord):
        x = 1
        y = 2

    r = TestRecordWithMandatoryFields._PRecordEvolver(TestRecordWithMandatoryFields, pmap(), _factory_fields=None, _ignore_extra=False)
    r['x'] = 10
    r = r.persistent()
    assert r.x == 10
    assert r.y == 2


# LLM-generated content at query #13
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record_empty = TestRecord()
    assert repr(record_empty) == "TestRecord(x=1, y=2)"


# LLM-generated content at query #14
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record_empty = TestRecord()
    assert repr(record_empty) == "TestRecord(x=1, y=2)"

    class AnotherRecord(PRecord):
        name = "default"
        value = 0

    record_with_strings = AnotherRecord(name="test", value=42)
    assert repr(record_with_strings) == "AnotherRecord(name='test', value=42)"


# LLM-generated content at query #15
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        x = field(mandatory=True)

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test factory field
    class FactoryRecord(PRecord):
        x = field(factory=lambda v: v * 2)

    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('x', 5)
    assert evolver['x'] == 10

    # Test ignore_extra
    evolver = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 10)
    assert evolver['x'] == 10


# LLM-generated content at query #16
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test that _PRecordMeta correctly sets up fields, invariants, and other attributes
    class TestRecord(PRecord):
        __invariant__ = lambda self: len(self.name) > 0
        name = None
        age = 0

    # Check that fields are set correctly
    assert hasattr(TestRecord, '_precord_fields')
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields

    # Check that invariants are stored
    assert hasattr(TestRecord, '_precord_invariants')
    assert len(TestRecord._precord_invariants) == 1

    # Check that mandatory fields are identified
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert isinstance(TestRecord._precord_mandatory_fields, set)

    # Check that initial values are set
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord._precord_initial_values == {'age': 0}

    # Check that __slots__ is set to empty tuple
    assert TestRecord.__slots__ == ()

    # Test with a record that has no fields
    class EmptyRecord(PRecord):
        pass

    assert EmptyRecord._precord_fields == {}
    assert EmptyRecord._precord_mandatory_fields == set()
    assert EmptyRecord._precord_initial_values == {}


# LLM-generated content at query #17
#--------------------------

```python
def test_PRecord___new__():
    # Test normal creation with kwargs
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 1
        y = lambda: 2
        z = 3

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with _factory_fields
    class TestRecordWithFactoryFields(PRecord):
        x = 1
        y = 2

    r = TestRecordWithFactoryFields._PRecordEvolver(TestRecordWithFactoryFields, pmap(), _factory_fields=[TestRecordWithFactoryFields._precord_fields['x']])
    r['x'] = 10
    r = r.persistent()
    assert r.x == 10
    assert r.y == 2

    # Test creation with _ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 1
        y = 2

    r = TestRecordWithIgnoreExtra._PRecordEvolver(TestRecordWithIgnoreExtra, pmap(), _ignore_extra=True)
    r['x'] = 10
    r['z'] = 30  # This should be ignored
    r = r.persistent()
    assert r.x == 10
    assert r.y == 2
    assert 'z' not in r

    # Test creation with _precord_size and _precord_buckets
    class TestRecordWithBuckets(PRecord):
        x = 1
        y = 2

    r = TestRecordWithBuckets(_precord_size=2, _precord_buckets=pmap(x=10, y=20)._buckets)
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #18
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass functionality
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field setup
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test mandatory fields
    assert TestRecord._precord_mandatory_fields == set()

    # Test initial values
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory field
    class TestRecordMandatory(PRecord):
        __invariant__ = lambda self: True
        x = 1, True  # (initial, mandatory)

    assert 'x' in TestRecordMandatory._precord_mandatory_fields

    # Test with initial values
    class TestRecordInitial(PRecord):
        __invariant__ = lambda self: True
        x = lambda: 1
        y = 2

    assert 'x' in TestRecordInitial._precord_initial_values
    assert 'y' in TestRecordInitial._precord_initial_values
    assert TestRecordInitial._precord_initial_values['x']() == 1
    assert TestRecordInitial._precord_initial_values['y'] == 2

    # Test inheritance
    class BaseRecord(PRecord):
        __invariant__ = lambda self: True
        base_field = 1

    class DerivedRecord(BaseRecord):
        derived_field = 2

    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields


# LLM-generated content at query #19
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with custom values
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with partial custom values
    r = TestRecord(x=10)
    assert r.x == 10
    assert r.y == 2

    # Test creation with factory_fields
    r = TestRecord(_factory_fields=[TestRecord._precord_fields['x']], x=100)
    assert r.x == 100
    assert r.y == 2

    # Test creation with ignore_extra
    r = TestRecord(_ignore_extra=True, x=10, z=30)
    assert r.x == 10
    assert r.y == 2
    assert 'z' not in r

    # Test creation with callable initial values
    class TestRecordCallable(PRecord):
        x = lambda: 1
        y = 2

    r = TestRecordCallable()
    assert r.x == 1
    assert r.y == 2

    # Test creation with existing _precord_size and _precord_buckets
    r1 = TestRecord(x=10, y=20)
    r2 = TestRecord(_precord_size=r1._size, _precord_buckets=r1._buckets)
    assert r2.x == 10
    assert r2.y == 20


# LLM-generated content at query #20
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}


# LLM-generated content at query #21
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation
    class TestRecord(PRecord):
        pass

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test with fields
    class TestRecordWithFields(PRecord):
        field1 = None
        field2 = None

    assert 'field1' in TestRecordWithFields._precord_fields
    assert 'field2' in TestRecordWithFields._precord_fields
    assert TestRecordWithFields._precord_mandatory_fields == set()

    # Test with mandatory fields
    class TestRecordWithMandatory(PRecord):
        field1 = None
        field2 = None

        class __invariant__:
            field1 = lambda x: (True, None)
            field2 = lambda x: (True, None)

    assert TestRecordWithMandatory._precord_mandatory_fields == set()

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        field1 = 1
        field2 = lambda: 2

    assert TestRecordWithInitial._precord_initial_values == {'field1': 1, 'field2': 2}

    # Test with invariants
    class TestRecordWithInvariant(PRecord):
        field1 = None

        class __invariant__:
            field1 = lambda x: (x > 0, "must be positive")

    assert 'field1' in TestRecordWithInvariant._precord_invariants


# LLM-generated content at query #22
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field(serializer=lambda x: x.upper())

    record = TestRecord(field1="value1", field2="value2")
    serialized = record.serialize()

    assert isinstance(serialized, dict)
    assert serialized["field1"] == "value1"
    assert serialized["field2"] == "value2".upper()

    record_without_serializer = TestRecord(field1="value1", field2=None)
    serialized_without = record_without_serializer.serialize()
    assert serialized_without["field1"] == "value1"
    assert serialized_without["field2"] is None


# LLM-generated content at query #23
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert isinstance(record, EmptyRecord)
    assert len(record) == 0

    # Test creation with fields
    class Person(PRecord):
        name = None
        age = 0

    person = Person(name="Alice", age=30)
    assert person.name == "Alice"
    assert person.age == 30
    assert len(person) == 2

    # Test creation with initial values
    class InitRecord(PRecord):
        x = 10
        y = 20

    record = InitRecord()
    assert record.x == 10
    assert record.y == 20

    # Test creation with factory fields
    class FactoryRecord(PRecord):
        data = None

    record = FactoryRecord(_factory_fields=[FactoryRecord._precord_fields['data']], data="test")
    assert record.data == "test"

    # Test creation with ignore_extra
    class IgnoreRecord(PRecord):
        a = None

    record = IgnoreRecord(a=1, b=2, _ignore_extra=True)
    assert record.a == 1
    assert len(record) == 1

    # Test creation with mandatory fields missing
    class MandatoryRecord(PRecord):
        required = None

    try:
        MandatoryRecord()
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test creation with invariant violation
    class InvariantRecord(PRecord):
        value = None

        @__invariant__
        def check_value(self, record):
            if record.value < 0:
                return False, "value must be non-negative"
            return True, None

    try:
        InvariantRecord(value=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test creation with correct invariant
    record = InvariantRecord(value=1)
    assert record.value == 1

    # Test creation from existing record
    original = Person(name="Bob", age=25)
    new_record = Person.create(original)
    assert new_record.name == "Bob"
    assert new_record.age == 25


# LLM-generated content at query #24
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 0

    # Test with dirty evolver
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with clean evolver (should return original)
    original = TestRecord(x=1, y=2)
    evolver = _PRecordEvolver(TestRecord, original)
    result = evolver.persistent()
    assert result is original

    # Test mandatory fields check
    class MandatoryRecord(PRecord):
        mandatory_field = 0
        optional_field = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['optional_field'] = 1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MandatoryRecord.mandatory_field' in e.missing_fields

    # Test invariant check
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            if self.x < 0:
                return False, "x_must_be_non_negative"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['x'] = -1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "x_must_be_non_negative" in e.invariant_errors

    # Test global invariants
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

        @__invariant__
        def check_sum(self):
            if self.x + self.y > 10:
                return False, "sum_too_large"
            return True, None

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 5
    evolver['y'] = 6
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "sum_too_large" in e.invariant_errors


# LLM-generated content at query #25
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test normal creation with kwargs
    r1 = TestRecord(x=10, y=20)
    assert r1.x == 10
    assert r1.y == 20

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 5

    r2 = TestRecordWithInitial(x=10)
    assert r2.x == 10
    assert r2.y == 1
    assert r2.z == 5

    # Test creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 0
        y = lambda: 100

    r3 = TestRecordWithCallableInitial(x=5)
    assert r3.x == 5
    assert r3.y == 100

    # Test creation with _factory_fields
    class TestRecordWithFactory(PRecord):
        x = 0
        y = 1

    r4 = TestRecordWithFactory._PRecordEvolver(TestRecordWithFactory, pmap(), _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    r4['x'] = 15
    r4['y'] = 25
    r5 = r4.persistent()
    assert r5.x == 15
    assert r5.y == 25

    # Test creation with _ignore_extra
    r6 = TestRecord.create({'x': 10, 'y': 20, 'extra': 30}, ignore_extra=True)
    assert r6.x == 10
    assert r6.y == 20
    assert 'extra' not in r6

    # Test creation with _precord_size and _precord_buckets
    r7 = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 100, 'y': 200}))
    assert r7.x == 100
    assert r7.y == 200


# LLM-generated content at query #26
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord(x=0, y=0)
    assert repr(record) == "TestRecord(x=0, y=0)"

    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert repr(record) == "EmptyRecord()"


# LLM-generated content at query #27
#--------------------------

```python
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        x = field()
        y = field()

    # Test basic persistent functionality
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with mandatory fields
    class MandatoryRecord(PRecord):
        a = field(mandatory=True)
        b = field()

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['b'] = 3
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MandatoryRecord.a' in e.missing_fields

    # Test with field invariants
    class InvariantRecord(PRecord):
        z = field(invariant=lambda x: (x > 0, "positive"))

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['z'] = -1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "positive" in e.invariant_errors

    # Test with global invariants
    class GlobalInvariantRecord(PRecord):
        __invariant__ = lambda self: (self.x != self.y, "x != y")
        x = field()
        y = field()

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "x != y" in e.invariant_errors

    # Test with existing PRecord
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 100
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 100
    assert result.y == 20


# LLM-generated content at query #28
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord(x=10)
    assert repr(record) == "TestRecord(x=10, y=2)"

    record = TestRecord()
    assert repr(record) == "TestRecord(x=1, y=2)"


# LLM-generated content at query #29
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 0
        y = 0

    e = _PRecordEvolver(TestRecord, pmap())
    e['x'] = 1
    e['y'] = 2
    result = e.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with mandatory fields
    class TestRecordMandatory(PRecord):
        x = 0
        y = 0

    e = _PRecordEvolver(TestRecordMandatory, pmap())
    e['x'] = 1
    with pytest.raises(InvariantException):
        e.persistent()

    # Test with invariant violation
    class TestRecordInvariant(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    e = _PRecordEvolver(TestRecordInvariant, pmap())
    e['x'] = -1
    with pytest.raises(InvariantException):
        e.persistent()

    # Test with no changes
    class TestRecordNoChanges(PRecord):
        x = 0
        y = 0

    original = TestRecordNoChanges(x=1, y=2)
    e = _PRecordEvolver(TestRecordNoChanges, original)
    result = e.persistent()
    assert result is original

    # Test with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = 0
        y = 0

        @__invariant__
        def check_sum(self):
            return self.x + self.y > 0, "sum must be positive"

    e = _PRecordEvolver(TestRecordGlobalInvariant, pmap())
    e['x'] = -1
    e['y'] = -2
    with pytest.raises(InvariantException):
        e.persistent()


# LLM-generated content at query #30
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 100
    new_record = evolver.persistent()
    assert new_record.x == 100
    assert new_record.y == 20
    assert isinstance(new_record, TestRecord)

    # Test persistence with mandatory fields missing
    class MandatoryRecord(PRecord):
        mandatory_field = 0
        other_field = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with invariant violation
    class InvariantRecord(PRecord):
        positive = 0

        @invariant
        def check_positive(self):
            return self.positive >= 0, "positive must be >= 0"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['positive'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with no changes
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    new_record = evolver.persistent()
    assert new_record is record

    # Test persistence with global invariant
    class GlobalInvariantRecord(PRecord):
        a = 0
        b = 0

        @global_invariant
        def check_sum(self):
            return self.a + self.b >= 0, "sum must be >= 0"

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['a'] = -1
    evolver['b'] = -2
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with factory fields
    class FactoryRecord(PRecord):
        x = field()
        y = field()

    evolver = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields={FactoryRecord._precord_fields['x']})
    evolver['x'] = 10
    evolver['y'] = 20
    new_record = evolver.persistent()
    assert new_record.x == 10
    assert new_record.y == 20


# LLM-generated content at query #31
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    result = evolver.set('x', 10)
    assert result['x'] == 10
    assert 'x' in result

    # Test setting multiple fields
    result = evolver.set('x', 20).set('y', 30)
    assert result['x'] == 20
    assert result['y'] == 30

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 40)

    # Test invariant violation
    class InvalidRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    evolver = _PRecordEvolver(InvalidRecord, pmap())
    result = evolver.set('x', -1)
    with pytest.raises(InvariantException):
        result.persistent()

    # Test mandatory field check
    class MandatoryRecord(PRecord):
        x = 0
        y = 1

        @invariant
        def check_mandatory(self):
            return 'x' in self and 'y' in self, "both x and y must be present"

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    result = evolver.set('x', 10)
    with pytest.raises(InvariantException):
        result.persistent()


# LLM-generated content at query #32
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert isinstance(record, EmptyRecord)
    assert len(record) == 0

    # Test creation with fields
    class Person(PRecord):
        name = None
        age = 0

    person = Person(name="Alice", age=30)
    assert person.name == "Alice"
    assert person.age == 30
    assert len(person) == 2

    # Test creation with initial values
    class Config(PRecord):
        host = "localhost"
        port = 8080

    config = Config()
    assert config.host == "localhost"
    assert config.port == 8080

    # Test creation with callable initial values
    class Timestamped(PRecord):
        timestamp = lambda: 1234567890

    ts = Timestamped()
    assert ts.timestamp == 1234567890

    # Test creation with factory fields
    class FactoryRecord(PRecord):
        x = None
        y = None

    evolver = FactoryRecord._Evolver(FactoryRecord, pmap(), _factory_fields=[FactoryRecord._precord_fields['x']])
    evolver['x'] = 10
    evolver['y'] = 20
    record = evolver.persistent()
    assert record.x == 10
    assert record.y == 20

    # Test creation with ignore_extra
    class StrictRecord(PRecord):
        a = None

    # Should raise AttributeError for extra field
    try:
        StrictRecord(a=1, b=2)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Should ignore extra field when ignore_extra=True
    record = StrictRecord.create({"a": 1, "b": 2}, ignore_extra=True)
    assert record.a == 1
    assert len(record) == 1

    # Test creation from existing record
    original = Person(name="Bob", age=25)
    copy = Person.create(original)
    assert copy.name == "Bob"
    assert copy.age == 25
    assert copy is original

    # Test pickling support
    import pickle
    pickled = pickle.dumps(person)
    unpickled = pickle.loads(pickled)
    assert unpickled.name == "Alice"
    assert unpickled.age == 30


# LLM-generated content at query #33
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test normal creation
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 3

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 1
    assert r.z == 3

    # Test with factory fields
    r = TestRecord._evolver._factory_fields = ['x']
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test with ignore_extra
    r = TestRecord._evolver._ignore_extra = True
    r = TestRecord(x=10, y=20, z=30)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test with _precord_size and _precord_buckets
    r = TestRecord(_precord_size=2, _precord_buckets=[('x', 10), ('y', 20)])
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #34
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 0

    # Test persistence with mandatory fields
    class TestRecordMandatory(PRecord):
        x = 0
        y = 0
        z = None, mandatory=True

    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with invariant
    class TestRecordInvariant(PRecord):
        x = 0
        y = 0

        __invariant__ = lambda self: (self.x >= 0, "x must be non-negative")

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['x'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with no changes
    record = TestRecord(x=1, y=2)
    evolver = _PRecordEvolver(TestRecord, record)
    result = evolver.persistent()
    assert result is record

    # Test persistence with changes
    record = TestRecord(x=1, y=2)
    evolver = _PRecordEvolver(TestRecord, record)
    evolver['x'] = 3
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 3
    assert result.y == 2
    assert result is not record


# LLM-generated content at query #35
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord()
    assert repr(record) == "TestRecord(x=0, y=1)"


# LLM-generated content at query #36
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field(serializer=lambda x: x.upper())
        field3 = field(serializer=lambda x: str(x))

    record = TestRecord(field1="value1", field2="value2", field3=42)

    # Test with no format specified
    serialized = record.serialize()
    assert serialized["field1"] == "value1"
    assert serialized["field2"] == "VALUE2"
    assert serialized["field3"] == "42"

    # Test with format specified (assuming serializer supports it)
    serialized_with_format = record.serialize(format="custom_format")
    assert serialized_with_format["field1"] == "value1"
    assert serialized_with_format["field2"] == "VALUE2"
    assert serialized_with_format["field3"] == "42"


# LLM-generated content at query #37
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant failure
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test type checking
    class TypedRecord(PRecord):
        x = field(type=int)

    evolver = _PRecordEvolver(TypedRecord, pmap())
    with pytest.raises(TypeError):
        evolver.set('x', 'not an int')


# LLM-generated content at query #38
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert isinstance(record, EmptyRecord)
    assert len(record) == 0

    # Test creation with initial values
    class Person(PRecord):
        name = None
        age = 0

    person = Person(name="Alice", age=30)
    assert person.name == "Alice"
    assert person.age == 30
    assert len(person) == 2

    # Test creation with default initial values
    class DefaultRecord(PRecord):
        x = 10
        y = "default"

    record = DefaultRecord()
    assert record.x == 10
    assert record.y == "default"

    # Test creation with override of default initial values
    record = DefaultRecord(x=20, y="custom")
    assert record.x == 20
    assert record.y == "custom"

    # Test creation with callable initial values
    class CallableRecord(PRecord):
        timestamp = lambda: 1234567890

    record = CallableRecord()
    assert record.timestamp == 1234567890

    # Test creation with factory fields
    class FactoryRecord(PRecord):
        a = None
        b = None

    evolver = FactoryRecord._Evolver(FactoryRecord, pmap(), _factory_fields=(FactoryRecord._precord_fields['a'],))
    evolver['a'] = "test"
    record = evolver.persistent()
    assert record.a == "test"

    # Test creation with ignore_extra
    class IgnoreExtraRecord(PRecord):
        field1 = None

    record = IgnoreExtraRecord(field1="value1", extra_field="should_be_ignored", _ignore_extra=True)
    assert record.field1 == "value1"
    assert "extra_field" not in record

    # Test creation from existing record
    original = Person(name="Bob", age=25)
    new_person = Person.create(original)
    assert new_person.name == "Bob"
    assert new_person.age == 25


# LLM-generated content at query #39
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord(x=0, y=0)
    assert repr(record) == "TestRecord(x=0, y=0)"

    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert repr(record) == "EmptyRecord()"


# LLM-generated content at query #40
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test valid field assignment
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test invalid field assignment
    with pytest.raises(AttributeError):
        evolver.set('z', 20)

    # Test invariant violation
    class ValidatedRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x_must_be_non_negative"
            return True, None

    evolver = _PRecordEvolver(ValidatedRecord, pmap())
    evolver.set('x', -5)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field enforcement
    class MandatoryRecord(PRecord):
        x = 0
        y = None

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('x', 10)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test type checking
    class TypedRecord(PRecord):
        x = 0

    evolver = _PRecordEvolver(TypedRecord, pmap())
    with pytest.raises(TypeError):
        evolver.set('x', 'not_an_int')


# LLM-generated content at query #41
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass creation
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        field1 = None
        field2 = None

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field attributes
    assert 'field1' in TestRecord._precord_fields
    assert 'field2' in TestRecord._precord_fields

    # Test mandatory fields
    assert isinstance(TestRecord._precord_mandatory_fields, set)

    # Test initial values
    assert isinstance(TestRecord._precord_initial_values, dict)

    # Test invariants
    assert TestRecord._precord_invariants == (lambda self: True,)

    # Test with mandatory field
    class TestRecordMandatory(PRecord):
        field1 = None
        field2 = None
        mandatory_field = None

    assert 'mandatory_field' in TestRecordMandatory._precord_mandatory_fields

    # Test with initial values
    class TestRecordInitial(PRecord):
        field1 = 1
        field2 = None

    assert TestRecordInitial._precord_initial_values['field1'] == 1


# LLM-generated content at query #42
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = 0
        field2 = "default"

    record = TestRecord(field1=42, field2="test")

    # Test with no format
    serialized = record.serialize()
    assert serialized == {'field1': 42, 'field2': "test"}

    # Test with format (assuming format is passed to serializer)
    serialized_with_format = record.serialize(format="json")
    assert serialized_with_format == {'field1': 42, 'field2': "test"}

    # Test with custom serializer
    class CustomRecord(PRecord):
        field1 = 0
        field2 = "default"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._precord_fields['field1'].serializer = lambda x, fmt: str(x) + "_serialized"

    custom_record = CustomRecord(field1=42, field2="test")
    serialized_custom = custom_record.serialize()
    assert serialized_custom == {'field1': "42_serialized", 'field2': "test"}


# LLM-generated content at query #43
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent call
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 15
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 15
    assert result.y == 20

    # Test with mandatory fields missing
    class TestRecordMandatory(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['x'] = 10
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'TestRecordMandatory.y' in exc_info.value.missing_fields

    # Test with invariant violation
    class TestRecordInvariant(PRecord):
        x = 0

        __invariant__ = lambda self: (self.x > 0, "x must be positive")

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['x'] = -5
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert "x must be positive" in exc_info.value.invariant_errors

    # Test with no changes (not dirty)
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test with global invariants
    class TestRecordGlobal(PRecord):
        x = 0
        y = 0

    def global_invariant(record):
        return record.x + record.y == 10, "sum must be 10"

    TestRecordGlobal._precord_invariants = (global_invariant,)
    evolver = _PRecordEvolver(TestRecordGlobal, pmap())
    evolver['x'] = 5
    evolver['y'] = 4
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert "sum must be 10" in exc_info.value.invariant_errors


# LLM-generated content at query #44
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record_empty = TestRecord()
    assert repr(record_empty) == "TestRecord(x=1, y=2)"

    class AnotherRecord(PRecord):
        name = "default"
        value = 0

    another_record = AnotherRecord(name="test", value=42)
    assert repr(another_record) == "AnotherRecord(name='test', value=42)"


# LLM-generated content at query #45
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    # Test normal creation
    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    record = TestRecordWithInitial(x=10)
    assert record.x == 10
    assert record.y == 2
    assert record.z == 3

    # Test with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 1
        y = lambda: 2
        z = 3

    record = TestRecordWithCallableInitial(x=10)
    assert record.x == 10
    assert record.y == 2
    assert record.z == 3

    # Test with factory fields
    class TestRecordWithFactoryFields(PRecord):
        x = 1
        y = 2

    record = TestRecordWithFactoryFields._PRecordEvolver(TestRecordWithFactoryFields, pmap(), _factory_fields=[TestRecordWithFactoryFields._precord_fields['x']])
    record['x'] = 10
    record = record.persistent()
    assert record.x == 10
    assert record.y == 2

    # Test with ignore extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 1
        y = 2

    record = TestRecordWithIgnoreExtra._PRecordEvolver(TestRecordWithIgnoreExtra, pmap(), _ignore_extra=True)
    record['x'] = 10
    record['z'] = 30
    record = record.persistent()
    assert record.x == 10
    assert record.y == 2
    assert 'z' not in record

    # Test with _precord_size and _precord_buckets
    record = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20})._buckets)
    assert record.x == 10
    assert record.y == 20


# LLM-generated content at query #46
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result.x == 0
    assert result.y == 20

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 20

    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('z', 30)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z' is not among the specified fields for TestRecord" in str(e)

    class TestRecordWithInvariant(PRecord):
        x = 0
        y = 0

        __invariant__ = lambda self: (self.x >= 0, "x must be non-negative")

    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver.set('x', -1)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be non-negative" in str(e)

    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver.set('x', 1)
    result = evolver.persistent()
    assert result.x == 1


# LLM-generated content at query #47
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass functionality
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field setup
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test mandatory fields
    assert TestRecord._precord_mandatory_fields == set()

    # Test initial values
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory field
    class TestRecordMandatory(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = PFIELD_NO_INITIAL

    assert 'z' in TestRecordMandatory._precord_mandatory_fields

    # Test with initial values
    class TestRecordInitial(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = 3

    assert TestRecordInitial._precord_initial_values == {'x': 1, 'y': 2, 'z': 3}

    # Test with callable initial value
    class TestRecordCallable(PRecord):
        __invariant__ = lambda self: True
        x = lambda: 1
        y = 2

    assert TestRecordCallable._precord_initial_values['x']() == 1
    assert TestRecordCallable._precord_initial_values['y'] == 2


# LLM-generated content at query #48
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    # Test setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    assert result.x == 10

    # Test setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test setting an invalid field (should raise AttributeError)
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('z', 30)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "is not among the specified fields" in str(e)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x >= 0, "x must be non-negative"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be non-negative" in str(e)

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        x = field(mandatory=True)
        y = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('y', 10)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "MandatoryRecord.x" in str(e.missing_fields)


# LLM-generated content at query #49
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with mandatory fields missing
    class MandatoryRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['x'] = 1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MandatoryRecord.y' in e.missing_fields

    # Test with invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def x_positive(self):
            return self.x > 0

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['x'] = -1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'x_positive' in e.invariant_errors

    # Test with global invariant
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

        @__invariant__
        def x_lt_y(self):
            return self.x < self.y

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 2
    evolver['y'] = 1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with no changes (not dirty)
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record


# LLM-generated content at query #50
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: len(self.name) > 0
        name = None
        age = 0

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {'age': 0}


# LLM-generated content at query #51
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = field(type=int)
        field2 = field(type=str, serializer=lambda x: x.upper())

    record = TestRecord(field1=42, field2="hello")

    # Test default serialization
    serialized = record.serialize()
    assert serialized == {"field1": 42, "field2": "HELLO"}

    # Test custom format serialization
    def custom_serializer(value):
        return f"custom_{value}"

    class CustomFormatRecord(PRecord):
        field1 = field(type=int, serializer=custom_serializer)
        field2 = field(type=str)

    custom_record = CustomFormatRecord(field1=10, field2="world")
    serialized_custom = custom_record.serialize(format="custom")
    assert serialized_custom == {"field1": "custom_10", "field2": "world"}

    # Test with no serializer
    class NoSerializerRecord(PRecord):
        field1 = field(type=int)
        field2 = field(type=str)

    no_serializer_record = NoSerializerRecord(field1=7, field2="test")
    serialized_no_serializer = no_serializer_record.serialize()
    assert serialized_no_serializer == {"field1": 7, "field2": "test"}


# LLM-generated content at query #52
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 100
    result = evolver.persistent()
    assert result.x == 100
    assert result.y == 20
    assert isinstance(result, TestRecord)

    # Test persistence with mandatory fields
    class MandatoryRecord(PRecord):
        mandatory_field = 0
        optional_field = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['optional_field'] = 10
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'MandatoryRecord.mandatory_field' in e.missing_fields

    # Test persistence with invariant violations
    class InvariantRecord(PRecord):
        positive = 0

        @invariant
        def check_positive(self):
            return self.positive >= 0, "positive.must_be_positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['positive'] = -1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'positive.must_be_positive' in e.invariant_errors

    # Test persistence with no changes
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test persistence with global invariants
    class GlobalInvariantRecord(PRecord):
        a = 0
        b = 0

        @invariant
        def check_sum(self):
            return self.a + self.b == 10, "sum.must_be_10"

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['a'] = 5
    evolver['b'] = 5
    result = evolver.persistent()
    assert result.a == 5
    assert result.b == 5

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['a'] = 5
    evolver['b'] = 6
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'sum.must_be_10' in e.invariant_errors


# LLM-generated content at query #53
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'invalid_field' is not among the specified fields for TestRecord"

    # Test invariant violation
    class InvariantRecord(PRecord):
        field1 = None

        @invariant
        def check_field1(self, record):
            if record.field1 == 'invalid':
                return False, 'field1_cannot_be_invalid'
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('field1', 'invalid')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'field1_cannot_be_invalid' in e.invariant_errors

    # Test mandatory field check
    class MandatoryRecord(PRecord):
        mandatory_field = field(mandatory=True)

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MandatoryRecord.mandatory_field' in e.missing_fields


# LLM-generated content at query #54
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    assert evolver['x'] == 10

    evolver.set('y', 20)
    assert evolver['y'] == 20

    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    class TestRecordWithFactory(PRecord):
        x = 0

    evolver = _PRecordEvolver(TestRecordWithFactory, pmap(), _factory_fields={TestRecordWithFactory._precord_fields['x']})
    evolver.set('x', 10)
    assert evolver['x'] == 10

    class TestRecordWithInvariant(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #55
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    assert result.x == 10

    # Test setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test setting an invalid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('z', 30)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z' is not among the specified fields for TestRecord" in str(e)

    # Test invariant failure
    class TestRecordWithInvariant(PRecord):
        x = 0

        @invariant
        def check_x(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver.set('x', -1)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be non-negative" in str(e)

    # Test mandatory field missing
    class TestRecordWithMandatoryField(PRecord):
        x = 0
        y = mandatory(1)

    evolver = _PRecordEvolver(TestRecordWithMandatoryField, pmap())
    evolver.set('x', 10)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestRecordWithMandatoryField.y" in str(e)


# LLM-generated content at query #56
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test with no arguments
    record = TestRecord()
    assert record.x == 0
    assert record.y == 1

    # Test with some arguments
    record = TestRecord(x=10)
    assert record.x == 10
    assert record.y == 1

    # Test with all arguments
    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

    # Test with _factory_fields
    record = TestRecord(x=10, _factory_fields=[TestRecord._precord_fields['x']])
    assert record.x == 10
    assert record.y == 1

    # Test with _ignore_extra
    record = TestRecord(x=10, z=30, _ignore_extra=True)
    assert record.x == 10
    assert record.y == 1
    assert 'z' not in record

    # Test with _precord_size and _precord_buckets
    record = TestRecord(x=10, y=20)
    new_record = TestRecord(_precord_size=record._size, _precord_buckets=record._buckets)
    assert new_record.x == 10
    assert new_record.y == 20


# LLM-generated content at query #57
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant violation
    class InvariantRecord(PRecord):
        field = None

        def __invariant__(self):
            if self.field == 'invalid':
                return False, 'INVALID_FIELD'
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('field', 'invalid')
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        mandatory_field = None

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #58
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert isinstance(TestRecord._precord_fields, dict)
    assert isinstance(TestRecord._precord_invariants, dict)
    assert isinstance(TestRecord._precord_mandatory_fields, set)
    assert isinstance(TestRecord._precord_initial_values, dict)


# LLM-generated content at query #59
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    assert evolver['x'] == 10

    evolver.set('y', 20)
    assert evolver['y'] == 20

    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    class TestRecordWithInvariant(PRecord):
        x = 0
        y = 0

        def __invariant__(self):
            if self.x > self.y:
                return False, "x should be <= y"
            return True, None

    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver.set('x', 10)
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 20

    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver.set('x', 30)
    evolver.set('y', 20)
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #60
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 1)
    evolver.set('y', 2)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test persistence with no changes
    evolver = _PRecordEvolver(TestRecord, pmap(x=1, y=2))
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test persistence with mandatory fields missing
    class TestRecordWithMandatory(PRecord):
        x = 0
        y = 0
        z = 0

    evolver = _PRecordEvolver(TestRecordWithMandatory, pmap())
    evolver.set('x', 1)
    evolver.set('y', 2)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with invariant violation
    class TestRecordWithInvariant(PRecord):
        x = 0
        y = 0

        __invariant__ = lambda self: (self.x + self.y) > 0

    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver.set('x', -1)
    evolver.set('y', -2)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with global invariant
    class TestRecordWithGlobalInvariant(PRecord):
        x = 0
        y = 0

    def global_invariant(record):
        return (record.x + record.y) > 0

    TestRecordWithGlobalInvariant._precord_invariants = (global_invariant,)

    evolver = _PRecordEvolver(TestRecordWithGlobalInvariant, pmap())
    evolver.set('x', -1)
    evolver.set('y', -2)
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #61
#--------------------------

```python
def test_PRecord_serialize():
    # Test basic serialization
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")
    assert record.serialize() == {"field1": "value1", "field2": "value2"}

    # Test serialization with custom serializer
    class TestRecordWithSerializer(PRecord):
        field1 = None
        field2 = None

        def __invariant__(self):
            return True, None

    record = TestRecordWithSerializer(field1="value1", field2="value2")
    assert record.serialize() == {"field1": "value1", "field2": "value2"}

    # Test serialization with format parameter
    class TestRecordWithFormat(PRecord):
        field1 = None
        field2 = None

    record = TestRecordWithFormat(field1="value1", field2="value2")
    assert record.serialize(format="json") == {"field1": "value1", "field2": "value2"}

    # Test serialization with empty record
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert record.serialize() == {}

    # Test serialization with nested PRecord
    class NestedRecord(PRecord):
        field1 = None
        field2 = None

    class ParentRecord(PRecord):
        nested = None

    nested = NestedRecord(field1="nested_value1", field2="nested_value2")
    parent = ParentRecord(nested=nested)
    assert parent.serialize() == {"nested": {"field1": "nested_value1", "field2": "nested_value2"}}


# LLM-generated content at query #62
#--------------------------

```python
def test_PRecord___new__():
    # Test normal creation with kwargs
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with factory_fields
    class TestRecordWithFactory(PRecord):
        x = 1
        y = 2

    r = TestRecordWithFactory._PRecordEvolver(TestRecordWithFactory, pmap(), _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    r['x'] = 10
    r['y'] = 20
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test creation with ignore_extra
    class TestRecordIgnoreExtra(PRecord):
        x = 1
        y = 2

    r = TestRecordIgnoreExtra(x=10, y=20, z=30, _ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with _precord_size and _precord_buckets
    class TestRecordDirect(PRecord):
        x = 1
        y = 2

    pm = pmap(x=10, y=20)
    r = TestRecordDirect(_precord_size=pm._size, _precord_buckets=pm._buckets)
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #63
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}


# LLM-generated content at query #64
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")

    serialized = record.serialize()
    assert isinstance(serialized, dict)
    assert serialized["field1"] == "value1"
    assert serialized["field2"] == "value2"

    class CustomSerializerRecord(PRecord):
        field1 = None
        field2 = None

        def __init__(self, field1, field2):
            self.field1 = field1
            self.field2 = field2

        def serializer(self, value):
            return str(value).upper()

    custom_record = CustomSerializerRecord(field1="value1", field2="value2")
    serialized_custom = custom_record.serialize()
    assert serialized_custom["field1"] == "VALUE1"
    assert serialized_custom["field2"] == "VALUE2"


# LLM-generated content at query #65
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")
    assert record.serialize() == {"field1": "value1", "field2": "value2"}

    class SerializerRecord(PRecord):
        field1 = None
        field2 = None

        def __init__(self, field1, field2):
            self.field1 = field1
            self.field2 = field2

    record = SerializerRecord(field1="value1", field2="value2")
    assert record.serialize() == {"field1": "value1", "field2": "value2"}


# LLM-generated content at query #66
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass functionality
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field setup
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test mandatory fields
    assert TestRecord._precord_mandatory_fields == set()

    # Test initial values
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory field
    class TestRecordMandatory(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = None

    assert TestRecordMandatory._precord_mandatory_fields == {'z'}

    # Test with initial values
    class TestRecordInitial(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = None

    assert TestRecordInitial._precord_initial_values == {}

    # Test with callable initial value
    class TestRecordCallableInitial(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = lambda: 3

    assert TestRecordCallableInitial._precord_initial_values == {'z': lambda: 3}

    # Test inheritance
    class BaseRecord(PRecord):
        __invariant__ = lambda self: True
        x = 1

    class DerivedRecord(BaseRecord):
        y = 2

    assert 'x' in DerivedRecord._precord_fields
    assert 'y' in DerivedRecord._precord_fields


# LLM-generated content at query #67
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert isinstance(record, EmptyRecord)
    assert len(record) == 0

    # Test creation with fields
    class Person(PRecord):
        name = None
        age = None

    person = Person(name="Alice", age=30)
    assert person.name == "Alice"
    assert person.age == 30
    assert len(person) == 2

    # Test creation with default values
    class DefaultRecord(PRecord):
        x = 1
        y = 2

    record = DefaultRecord()
    assert record.x == 1
    assert record.y == 2

    # Test creation with mixed default and provided values
    record = DefaultRecord(x=10)
    assert record.x == 10
    assert record.y == 2

    # Test creation with factory fields
    class FactoryRecord(PRecord):
        a = None
        b = None

    evolver = FactoryRecord._Evolver(FactoryRecord, pmap(), _factory_fields=[FactoryRecord._precord_fields['a']])
    evolver['a'] = 1
    evolver['b'] = 2
    record = evolver.persistent()
    assert record.a == 1
    assert record.b == 2

    # Test creation with ignore_extra
    class IgnoreExtraRecord(PRecord):
        field1 = None

    record = IgnoreExtraRecord(field1=1, extra_field=2, _ignore_extra=True)
    assert record.field1 == 1
    assert len(record) == 1

    # Test creation with pickle support
    import pickle
    pickled = pickle.dumps(person)
    unpickled = pickle.loads(pickled)
    assert unpickled.name == "Alice"
    assert unpickled.age == 30


# LLM-generated content at query #68
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")
    serialized = record.serialize()
    assert isinstance(serialized, dict)
    assert serialized["field1"] == "value1"
    assert serialized["field2"] == "value2"

    class TestRecordWithSerializer(PRecord):
        field1 = None
        field2 = None

        @staticmethod
        def __serialize_field1__(value, format=None):
            return f"serialized_{value}"

    record_with_serializer = TestRecordWithSerializer(field1="value1", field2="value2")
    serialized_with_custom = record_with_serializer.serialize()
    assert serialized_with_custom["field1"] == "serialized_value1"
    assert serialized_with_custom["field2"] == "value2"


# LLM-generated content at query #69
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field
    try:
        evolver.set('invalid_field', 'value')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "is not among the specified fields" in str(e)

    # Test invariant failure
    class InvariantRecord(PRecord):
        field1 = None
        field2 = None

        @invariant
        def check_field(self):
            if self.field1 == 'invalid':
                return False, 'INVALID_FIELD'
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('field1', 'invalid')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVALID_FIELD' in e.invariant_errors

    # Test mandatory field check
    class MandatoryRecord(PRecord):
        mandatory_field = field(mandatory=True)
        optional_field = None

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('optional_field', 'value')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'mandatory_field' in e.missing_fields


# LLM-generated content at query #70
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")
    serialized = record.serialize()
    assert serialized == {"field1": "value1", "field2": "value2"}

    class TestRecordWithSerializer(PRecord):
        field1 = None
        field2 = None

        def __init__(self, field1, field2):
            self.field1 = field1
            self.field2 = field2

        def serialize_field1(self, value):
            return f"serialized_{value}"

    record_with_serializer = TestRecordWithSerializer(field1="value1", field2="value2")
    serialized_with_custom = record_with_serializer.serialize()
    assert serialized_with_custom == {"field1": "serialized_value1", "field2": "value2"}


# LLM-generated content at query #71
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()


# LLM-generated content at query #72
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with kwargs
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with partial kwargs
    r = TestRecord(x=10)
    assert r.x == 10
    assert r.y == 2

    # Test creation with callable initial values
    class TestRecord2(PRecord):
        x = lambda: 1
        y = 2

    r = TestRecord2()
    assert r.x == 1
    assert r.y == 2

    # Test creation with factory_fields
    r = TestRecord._evolver().set('x', 10).persistent()
    assert r.x == 10
    assert r.y == 2

    # Test creation with ignore_extra
    class TestRecord3(PRecord):
        x = 1

    r = TestRecord3(x=10, z=30, _ignore_extra=True)
    assert r.x == 10
    assert 'z' not in r

    # Test creation with _precord_size and _precord_buckets
    r = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 1, 'y': 2}))
    assert r.x == 1
    assert r.y == 2


# LLM-generated content at query #73
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class TestRecordWithInvariant(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x >= 0, "x must be non-negative"

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('x', -1)
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field check
    class TestRecordWithMandatory(PRecord):
        x = field(mandatory=True)
        y = 0

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()

    # Test successful persistent call
    evolver_mandatory.set('x', 10)
    result = evolver_mandatory.persistent()
    assert isinstance(result, TestRecordWithMandatory)
    assert result.x == 10


# LLM-generated content at query #74
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation
    class TestRecord(PRecord):
        pass

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test with fields
    class TestRecordWithFields(PRecord):
        x = 1
        y = 2

    assert 'x' in TestRecordWithFields._precord_fields
    assert 'y' in TestRecordWithFields._precord_fields
    assert TestRecordWithFields._precord_mandatory_fields == set()

    # Test with mandatory fields
    class TestRecordWithMandatory(PRecord):
        x = 1
        y = 2
        z = 3

    assert TestRecordWithMandatory._precord_mandatory_fields == {'x', 'y', 'z'}

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2

    assert TestRecordWithInitial._precord_initial_values == {'x': 1, 'y': 2}

    # Test with invariants
    class TestRecordWithInvariant(PRecord):
        __invariant__ = lambda self: True

    assert TestRecordWithInvariant._precord_invariants == [lambda self: True]


# LLM-generated content at query #75
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    # Test setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 1)
    result = evolver.persistent()
    assert result.x == 1

    # Test setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 1)
    evolver.set('y', 2)
    result = evolver.persistent()
    assert result.x == 1
    assert result.y == 2

    # Test setting an invalid field (should raise AttributeError)
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('z', 3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z' is not among the specified fields for TestRecord" in str(e)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def x_positive(self):
            return self.x > 0

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        x = field(mandatory=True)
        y = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('y', 2)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "MandatoryRecord.x" in str(e.missing_fields)


# LLM-generated content at query #76
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")
    serialized = record.serialize()

    assert isinstance(serialized, dict)
    assert serialized["field1"] == "value1"
    assert serialized["field2"] == "value2"

    class TestRecordWithSerializer(PRecord):
        field1 = None
        field2 = None

        @serialize
        def field1(self, value):
            return f"serialized_{value}"

    record_with_serializer = TestRecordWithSerializer(field1="value1", field2="value2")
    serialized_with_custom = record_with_serializer.serialize()

    assert isinstance(serialized_with_custom, dict)
    assert serialized_with_custom["field1"] == "serialized_value1"
    assert serialized_with_custom["field2"] == "value2"


# LLM-generated content at query #77
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test with fields
    class TestRecordWithFields(PRecord):
        a = 1
        b = 2
        c = 3

    assert len(TestRecordWithFields._precord_fields) == 3
    assert 'a' in TestRecordWithFields._precord_fields
    assert 'b' in TestRecordWithFields._precord_fields
    assert 'c' in TestRecordWithFields._precord_fields

    # Test with mandatory fields
    class TestRecordWithMandatory(PRecord):
        mandatory_field = 1
        optional_field = 2

    assert 'mandatory_field' in TestRecordWithMandatory._precord_mandatory_fields
    assert 'optional_field' not in TestRecordWithMandatory._precord_mandatory_fields

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    assert TestRecordWithInitial._precord_initial_values == {'x': 1, 'y': 2, 'z': 3}

    # Test with invariants
    class TestRecordWithInvariant(PRecord):
        x = 1

        __invariant__ = lambda self: (True, "error") if self.x > 0 else (False, "error")

    assert len(TestRecordWithInvariant._precord_invariants) == 1


# LLM-generated content at query #78
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: len(self) > 0
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}


# LLM-generated content at query #79
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}

    class TestRecordWithFields(PRecord):
        x = 1
        y = 2
        __invariant__ = lambda self: True

    assert TestRecordWithFields._precord_invariants == (lambda self: True,)


# LLM-generated content at query #80
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert '_precord_fields' in TestRecord.__dict__
    assert '_precord_invariants' in TestRecord.__dict__
    assert '_precord_mandatory_fields' in TestRecord.__dict__
    assert '_precord_initial_values' in TestRecord.__dict__


# LLM-generated content at query #81
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord()
    assert repr(record) == "TestRecord(x=1, y=2)"


# LLM-generated content at query #82
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field inheritance
    class BaseRecord(PRecord):
        a = 1

    class DerivedRecord(BaseRecord):
        b = 2

    assert 'a' in DerivedRecord._precord_fields
    assert 'b' in DerivedRecord._precord_fields

    # Test mandatory fields
    class MandatoryRecord(PRecord):
        mandatory_field = None

    assert 'mandatory_field' in MandatoryRecord._precord_mandatory_fields

    # Test initial values
    class InitialRecord(PRecord):
        initial_field = 42

    assert InitialRecord._precord_initial_values == {'initial_field': 42}

    # Test invariant storage
    class InvariantRecord(PRecord):
        __invariant__ = lambda self: True

    assert InvariantRecord._precord_invariants == [lambda self: True]


# LLM-generated content at query #83
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass functionality
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field setup
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test mandatory fields
    assert TestRecord._precord_mandatory_fields == set()

    # Test initial values
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory field
    class TestRecordWithMandatory(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = None

    assert TestRecordWithMandatory._precord_mandatory_fields == {'z'}

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = None

    assert TestRecordWithInitial._precord_initial_values == {'z': None}

    # Test inheritance
    class BaseRecord(PRecord):
        __invariant__ = lambda self: True
        base_field = 1

    class DerivedRecord(BaseRecord):
        derived_field = 2

    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields


# LLM-generated content at query #84
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")
    serialized = record.serialize()
    assert isinstance(serialized, dict)
    assert serialized["field1"] == "value1"
    assert serialized["field2"] == "value2"

    class TestRecordWithSerializer(PRecord):
        field1 = None
        field2 = None

        def __init__(self, field1, field2):
            self.field1 = field1
            self.field2 = field2

    record_with_serializer = TestRecordWithSerializer(field1="value1", field2="value2")
    serialized_with_serializer = record_with_serializer.serialize()
    assert isinstance(serialized_with_serializer, dict)
    assert serialized_with_serializer["field1"] == "value1"
    assert serialized_with_serializer["field2"] == "value2"


# LLM-generated content at query #85
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test normal creation
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 1
    assert r.z == 3

    # Test with factory fields
    r = TestRecord._evolver().set('x', 10).set('y', 20).persistent()
    assert r.x == 10
    assert r.y == 20

    # Test with _precord_size and _precord_buckets
    r = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20})._buckets)
    assert r.x == 10
    assert r.y == 20

    # Test with ignore_extra
    r = TestRecord.create({'x': 10, 'y': 20, 'z': 30}, ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r


# LLM-generated content at query #86
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class TestRecordWithInvariant(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('x', -1)
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test factory field
    class TestRecordWithFactory(PRecord):
        x = field(factory=lambda x: x * 2)

    evolver_with_factory = _PRecordEvolver(TestRecordWithFactory, pmap())
    evolver_with_factory.set('x', 5)
    assert evolver_with_factory['x'] == 10

    # Test ignore_extra
    evolver_ignore = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver_ignore.set('x', 10)
    evolver_ignore.set('extra_field', 30)  # Should not raise AttributeError
    assert evolver_ignore['x'] == 10


# LLM-generated content at query #87
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    # Test normal creation
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 1
        y = lambda: 2
        z = 3

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test with factory fields
    r = TestRecord._Evolver(TestRecord, pmap(), _factory_fields=[TestRecord._precord_fields['x']])
    r['x'] = 10
    r['y'] = 20
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test with ignore_extra
    r = TestRecord._Evolver(TestRecord, pmap(), _ignore_extra=True)
    r['x'] = 10
    r['y'] = 20
    r['z'] = 30  # This should be ignored
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20
    assert 'z' not in result

    # Test with _precord_size and _precord_buckets
    r = TestRecord(_precord_size=2, _precord_buckets=pmap(x=10, y=20)._buckets)
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #88
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        field1 = None
        field2 = None

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert isinstance(TestRecord._precord_fields, dict)
    assert isinstance(TestRecord._precord_invariants, list)
    assert isinstance(TestRecord._precord_mandatory_fields, set)
    assert isinstance(TestRecord._precord_initial_values, dict)


# LLM-generated content at query #89
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 0
        y = 1

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 5

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 1
    assert r.z == 5

    # Test with callable initial value
    class TestRecordWithCallableInitial(PRecord):
        x = 0
        y = lambda: 100

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 100

    # Test with factory_fields
    class TestRecordWithFactory(PRecord):
        x = 0
        y = 1

    r = TestRecordWithFactory._Evolver(TestRecordWithFactory, pmap(), _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    r['x'] = 10
    r['y'] = 20
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test with ignore_extra
    class TestRecordIgnoreExtra(PRecord):
        x = 0
        y = 1

    r = TestRecordIgnoreExtra(x=10, y=20, z=30, _ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test with _precord_size and _precord_buckets
    class TestRecordDirect(PRecord):
        x = 0
        y = 1

    r = TestRecordDirect(_precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20})._buckets)
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #90
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field setup
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test mandatory fields
    assert TestRecord._precord_mandatory_fields == set()

    # Test initial values
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory field
    class TestRecordMandatory(PRecord):
        x = 1
        y = 2
        z = 3

        class __invariant__:
            def __bool__(self):
                return True

    assert TestRecordMandatory._precord_mandatory_fields == set()

    # Test with initial values
    class TestRecordInitial(PRecord):
        x = 1
        y = 2
        z = 3

        class __invariant__:
            def __bool__(self):
                return True

    assert TestRecordInitial._precord_initial_values == {}

    # Test with callable initial value
    class TestRecordCallableInitial(PRecord):
        x = 1
        y = 2
        z = lambda: 3

        class __invariant__:
            def __bool__(self):
                return True

    assert 'z' in TestRecordCallableInitial._precord_initial_values
    assert callable(TestRecordCallableInitial._precord_initial_values['z'])

    # Test with invariant
    class TestRecordInvariant(PRecord):
        x = 1
        y = 2

        class __invariant__:
            def __bool__(self):
                return True

    assert hasattr(TestRecordInvariant, '_precord_invariants')
    assert len(TestRecordInvariant._precord_invariants) == 1

    # Test with multiple inheritance
    class BaseRecord(PRecord):
        x = 1

    class TestRecordMultipleInheritance(BaseRecord):
        y = 2

    assert 'x' in TestRecordMultipleInheritance._precord_fields
    assert 'y' in TestRecordMultipleInheritance._precord_fields


# LLM-generated content at query #91
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord(x=10)
    assert repr(record) == "TestRecord(x=10, y=2)"

    record = TestRecord()
    assert repr(record) == "TestRecord(x=1, y=2)"


# LLM-generated content at query #92
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    # Test setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['x'] == 10
    assert evolver['y'] == 20

    # Test setting an invalid field
    try:
        evolver.set('z', 30)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'z' is not among the specified fields for TestRecord"

    # Test invariant violation
    class TestRecordWithInvariant(PRecord):
        x = 0
        __invariant__ = lambda self: (self.x >= 0, "x must be non-negative")

    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver.set('x', -1)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be non-negative" in e.invariant_errors

    # Test mandatory field missing
    class TestRecordWithMandatory(PRecord):
        x = 0
        y = 0
        __mandatory__ = ['x', 'y']

    evolver = _PRecordEvolver(TestRecordWithMandatory, pmap())
    evolver.set('x', 10)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestRecordWithMandatory.y" in e.missing_fields


# LLM-generated content at query #93
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record_empty = TestRecord()
    assert repr(record_empty) == "TestRecord(x=1, y=2)"

    class AnotherRecord(PRecord):
        name = "default"
        value = 0

    record_with_strings = AnotherRecord(name="test", value=42)
    assert repr(record_with_strings) == "AnotherRecord(name='test', value=42)"


# LLM-generated content at query #94
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test normal creation
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 1
    assert r.z == 3

    # Test with factory fields
    r = TestRecord.create({'x': 10, 'y': 20}, _factory_fields=TestRecord._precord_fields)
    assert r.x == 10
    assert r.y == 20

    # Test with ignore_extra
    r = TestRecord.create({'x': 10, 'y': 20, 'extra': 30}, ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'extra' not in r

    # Test with direct bucket creation
    r = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20})._buckets)
    assert r.x == 10
    assert r.y == 20

    # Test with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 0
        y = lambda: 1
        z = 3

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 1
    assert r.z == 3


# LLM-generated content at query #95
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation
    class TestRecord(PRecord):
        pass

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test with fields
    class TestRecordWithFields(PRecord):
        field1 = None
        field2 = 0

    assert 'field1' in TestRecordWithFields._precord_fields
    assert 'field2' in TestRecordWithFields._precord_fields
    assert TestRecordWithFields._precord_mandatory_fields == set()

    # Test with mandatory fields
    class TestRecordWithMandatory(PRecord):
        mandatory_field = None
        optional_field = 0

    assert 'mandatory_field' in TestRecordWithMandatory._precord_fields
    assert 'optional_field' in TestRecordWithMandatory._precord_fields
    assert TestRecordWithMandatory._precord_mandatory_fields == {'mandatory_field'}

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        field_with_initial = 42
        field_without_initial = None

    assert TestRecordWithInitial._precord_initial_values == {'field_with_initial': 42}


# LLM-generated content at query #96
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    empty_record = TestRecord()
    assert repr(empty_record) == "TestRecord(x=1, y=2)"


# LLM-generated content at query #97
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with kwargs
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = PFIELD_NO_INITIAL

    r = TestRecordWithInitial(z=3)
    assert r.x == 1
    assert r.y == 2
    assert r.z == 3

    # Test creation with factory fields
    class TestRecordWithFactory(PRecord):
        x = 1
        y = 2

    r = TestRecordWithFactory._PRecordEvolver(TestRecordWithFactory, pmap(), _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    r['x'] = 10
    r = r.persistent()
    assert r.x == 10
    assert r.y == 2

    # Test creation with ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 1
        y = 2

    r = TestRecordWithIgnoreExtra._PRecordEvolver(TestRecordWithIgnoreExtra, pmap(), _ignore_extra=True)
    r['x'] = 10
    r['z'] = 30
    r = r.persistent()
    assert r.x == 10
    assert r.y == 2
    assert 'z' not in r

    # Test creation with _precord_size and _precord_buckets
    class TestRecordWithBuckets(PRecord):
        x = 1
        y = 2

    r = TestRecordWithBuckets(_precord_size=2, _precord_buckets=pmap(x=10, y=20)._buckets)
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #98
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass functionality
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field setup
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test mandatory fields
    assert TestRecord._precord_mandatory_fields == set()

    # Test initial values
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory field
    class TestRecordMandatory(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = 3

    assert TestRecordMandatory._precord_mandatory_fields == set()

    # Test with initial values
    class TestRecordInitial(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = 3

    assert TestRecordInitial._precord_initial_values == {}

    # Test with actual mandatory field
    class TestRecordActualMandatory(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = 3

    assert TestRecordActualMandatory._precord_mandatory_fields == set()

    # Test with actual initial values
    class TestRecordActualInitial(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2
        z = 3

    assert TestRecordActualInitial._precord_initial_values == {}


# LLM-generated content at query #99
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}


# LLM-generated content at query #100
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = None
        y = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['x'] == 10
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class ValidatedRecord(PRecord):
        x = None

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(ValidatedRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field check
    class MandatoryRecord(PRecord):
        x = None
        y = None

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('x', 10)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with factory fields
    class FactoryRecord(PRecord):
        x = None
        y = None

    evolver = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields=[FactoryRecord._precord_fields['x']])
    evolver.set('x', 10)
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 20


# LLM-generated content at query #101
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant violation
    class InvariantRecord(PRecord):
        field1 = None

        @invariant
        def check_field1(self):
            return self.field1 != 'invalid', 'field1_cannot_be_invalid'

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field
    class MandatoryRecord(PRecord):
        mandatory_field = None

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test ignore_extra
    class IgnoreExtraRecord(PRecord):
        field1 = None

    evolver = _PRecordEvolver(IgnoreExtraRecord, pmap(), _ignore_extra=True)
    evolver.set('field1', 'value1')
    evolver.set('extra_field', 'extra_value')  # Should be ignored
    result = evolver.persistent()
    assert result['field1'] == 'value1'
    assert 'extra_field' not in result


# LLM-generated content at query #102
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    # Test setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    assert result.x == 10

    # Test setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test setting an invalid field (should raise AttributeError)
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('z', 30)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z' is not among the specified fields for TestRecord" in str(e)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def x_positive(self):
            return self.x > 0, "x must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be positive" in str(e)

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        x = field(mandatory=True)
        y = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('y', 10)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "MandatoryRecord.x" in str(e.missing_fields)


# LLM-generated content at query #103
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field
    class MandatoryRecord(PRecord):
        x = 0
        y = 1

        def __invariant__(self):
            if 'y' not in self:
                return False, "y is mandatory"
            return True, None

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('x', 10)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test factory field
    class FactoryRecord(PRecord):
        x = 0

        def __factory__(self, value):
            return value * 2

    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('x', 5)
    assert evolver['x'] == 10

    # Test ignore_extra
    class IgnoreExtraRecord(PRecord):
        x = 0

    evolver = _PRecordEvolver(IgnoreExtraRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 10)
    evolver.set('y', 20)  # This should be ignored
    assert evolver['x'] == 10
    assert 'y' not in evolver


# LLM-generated content at query #104
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord(x=5, y=10)
    assert repr(record) == "TestRecord(x=5, y=10)"

    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert repr(record) == "EmptyRecord()"


# LLM-generated content at query #105
#--------------------------

```python
def test_PRecord___new__():
    # Test normal creation with kwargs
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 1
        y = lambda: 2
        z = 3

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test with _factory_fields
    class TestRecordWithFactoryFields(PRecord):
        x = 1
        y = 2

    r = TestRecordWithFactoryFields._PRecordEvolver(TestRecordWithFactoryFields, pmap(), _factory_fields=[TestRecordWithFactoryFields._precord_fields['x']])
    r['x'] = 10
    r['y'] = 20
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test with _ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 1
        y = 2

    r = TestRecordWithIgnoreExtra._PRecordEvolver(TestRecordWithIgnoreExtra, pmap(), _ignore_extra=True)
    r['x'] = 10
    r['y'] = 20
    r['z'] = 30  # This should be ignored
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20
    assert 'z' not in result

    # Test with _precord_size and _precord_buckets
    class TestRecordWithBuckets(PRecord):
        x = 1
        y = 2

    r = TestRecordWithBuckets(_precord_size=2, _precord_buckets=pmap(x=10, y=20)._buckets)
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #106
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver.set('x', 15)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 15
    assert result.y == 20

    # Test persistence with mandatory fields
    class MandatoryRecord(PRecord):
        mandatory_field = 0
        optional_field = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('optional_field', 10)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with invariant violation
    class InvariantRecord(PRecord):
        positive = 0

        @invariant
        def positive_invariant(self):
            return self.positive >= 0, "positive must be non-negative"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('positive', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with no changes
    record = TestRecord(x=5, y=10)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test persistence with global invariants
    class GlobalInvariantRecord(PRecord):
        a = 0
        b = 0

        @global_invariant
        def a_less_than_b(self):
            return self.a < self.b, "a must be less than b"

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap(a=5, b=3))
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #107
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")
    serialized = record.serialize()
    assert serialized == {"field1": "value1", "field2": "value2"}

    class TestRecordWithSerializer(PRecord):
        field1 = None
        field2 = None

        @staticmethod
        def __serializer_field1__(value, format=None):
            return f"serialized_{value}"

    record_with_serializer = TestRecordWithSerializer(field1="value1", field2="value2")
    serialized_with_custom = record_with_serializer.serialize()
    assert serialized_with_custom == {"field1": "serialized_value1", "field2": "value2"}

    class TestRecordWithFormatSerializer(PRecord):
        field1 = None
        field2 = None

        @staticmethod
        def __serializer_field1__(value, format=None):
            if format == "upper":
                return value.upper()
            return value

    record_with_format = TestRecordWithFormatSerializer(field1="value1", field2="value2")
    serialized_format = record_with_format.serialize(format="upper")
    assert serialized_format == {"field1": "VALUE1", "field2": "value2"}


# LLM-generated content at query #108
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field inheritance
    class BaseRecord(PRecord):
        a = 1

    class DerivedRecord(BaseRecord):
        b = 2

    assert 'a' in DerivedRecord._precord_fields
    assert 'b' in DerivedRecord._precord_fields

    # Test mandatory fields
    class MandatoryRecord(PRecord):
        mandatory_field = 1
        optional_field = 2

    assert 'mandatory_field' in MandatoryRecord._precord_mandatory_fields
    assert 'optional_field' not in MandatoryRecord._precord_mandatory_fields

    # Test initial values
    class InitialRecord(PRecord):
        x = 1
        y = 2

    assert InitialRecord._precord_initial_values == {'x': 1, 'y': 2}

    # Test invariant storage
    class InvariantRecord(PRecord):
        x = 1

        def __invariant__(self):
            return True

    assert hasattr(InvariantRecord, '_precord_invariants')


# LLM-generated content at query #109
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field(serializer=lambda x: x.upper())

    record = TestRecord(field1="value1", field2="value2")

    # Test default serialization
    assert record.serialize() == {"field1": "value1", "field2": "value2"}

    # Test custom serializer
    assert record.serialize()["field2"] == "VALUE2"

    # Test with format parameter (assuming format is passed to serializer)
    assert record.serialize(format="test") == {"field1": "value1", "field2": "VALUE2"}


# LLM-generated content at query #110
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record_empty = TestRecord()
    assert repr(record_empty) == "TestRecord(x=1, y=2)"

    class AnotherRecord(PRecord):
        name = "default"
        value = 0

    record_another = AnotherRecord(name="test", value=42)
    assert repr(record_another) == "AnotherRecord(name='test', value=42)"


# LLM-generated content at query #111
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test valid field setting
    result = evolver.set('x', 10)
    assert result['x'] == 10
    assert 'x' in result

    # Test invalid field setting
    with pytest.raises(AttributeError):
        evolver.set('z', 20)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    result = evolver.set('x', -1)
    with pytest.raises(InvariantException):
        result.persistent()

    # Test mandatory field
    class MandatoryRecord(PRecord):
        x = 0
        y = None

        def __invariant__(self):
            if self.y is None:
                return False, "y is mandatory"
            return True, None

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    result = evolver.set('x', 10)
    with pytest.raises(InvariantException):
        result.persistent()

    # Test factory field
    class FactoryRecord(PRecord):
        x = 0

        def __factory__(self, value):
            return value * 2

    evolver = _PRecordEvolver(FactoryRecord, pmap())
    result = evolver.set('x', 5)
    assert result['x'] == 10


# LLM-generated content at query #112
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test normal creation
    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = PFIELD_NO_INITIAL

    record = TestRecordWithInitial(x=10, y=20, z=30)
    assert record.x == 10
    assert record.y == 20
    assert record.z == 30

    # Test with factory fields
    record = TestRecord._factory_fields={'x', 'y'}, x=10, y=20)
    assert record.x == 10
    assert record.y == 20

    # Test with ignore_extra
    record = TestRecord._ignore_extra=True, x=10, y=20, extra=30)
    assert record.x == 10
    assert record.y == 20
    assert 'extra' not in record

    # Test with internal parameters
    record = TestRecord._precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20}))
    assert record.x == 10
    assert record.y == 20


# LLM-generated content at query #113
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test invalid field
    with pytest.raises(AttributeError):
        evolver.set('z', 20)

    # Test invariant failure
    class InvariantRecord(PRecord):
        x = 0
        __invariant__ = lambda self: ('x must be positive', self.x > 0)

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test factory field
    class FactoryRecord(PRecord):
        x = 0
        y = 0
        z = 0

    evolver = _PRecordEvolver(FactoryRecord, pmap(), _factory_fields=[FactoryRecord._precord_fields['x']])
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test ignore_extra
    evolver = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 10)
    assert evolver['x'] == 10


# LLM-generated content at query #114
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant violation
    class TestRecordWithInvariant(PRecord):
        field1 = None

        def __invariant__(self):
            if self.field1 == 'invalid':
                return False, 'INVALID_FIELD1'
            return True, None

    evolver_with_invariant = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_invariant.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver_with_invariant.persistent()

    # Test mandatory field check
    class TestRecordWithMandatory(PRecord):
        mandatory_field = None

    evolver_with_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_with_mandatory.persistent()


# LLM-generated content at query #115
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test valid field assignment
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test invalid field assignment
    with pytest.raises(AttributeError):
        evolver.set('z', 20)

    # Test invariant violation
    class TestRecordWithInvariant(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('x', -1)
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test factory field
    class TestRecordWithFactory(PRecord):
        x = 0

        def __factory__(self, value):
            return value * 2

    evolver_with_factory = _PRecordEvolver(TestRecordWithFactory, pmap())
    evolver_with_factory.set('x', 5)
    assert evolver_with_factory['x'] == 10

    # Test ignore_extra
    evolver_ignore_extra = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver_ignore_extra.set('x', 10)
    assert evolver_ignore_extra['x'] == 10


# LLM-generated content at query #116
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass creation
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field setup
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test mandatory fields
    assert TestRecord._precord_mandatory_fields == set()

    # Test initial values
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory field
    class TestRecordMandatory(PRecord):
        x = 1
        y = 2
        z = 3

    assert TestRecordMandatory._precord_mandatory_fields == set()

    # Test with initial values
    class TestRecordInitial(PRecord):
        x = 1
        y = 2
        z = 3

    assert TestRecordInitial._precord_initial_values == {}

    # Test with slots
    assert TestRecordInitial.__slots__ == ()


# LLM-generated content at query #117
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 10
    evolver['y'] = 20
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 10
    assert result['y'] == 20

    # Test with mandatory fields missing
    class MandatoryRecord(PRecord):
        mandatory_field = PFIELD_NO_INITIAL

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violation
    class InvariantRecord(PRecord):
        positive_field = 1

        @__invariant__
        def check_positive(self):
            return self.positive_field > 0, "must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['positive_field'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with global invariant
    class GlobalInvariantRecord(PRecord):
        a = 1
        b = 2

        @__invariant__
        def check_sum(self):
            return self.a + self.b > 0, "sum must be positive"

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['a'] = -3
    evolver['b'] = -4
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with existing PRecord instance
    original = TestRecord(x=5, y=15)
    evolver = original.evolver()
    evolver['x'] = 7
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 7
    assert result['y'] == 15


# LLM-generated content at query #118
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")
    serialized = record.serialize()
    assert serialized == {"field1": "value1", "field2": "value2"}

    class TestRecordWithSerializer(PRecord):
        field1 = None
        field2 = None

        def __init__(self, field1, field2):
            self.field1 = field1
            self.field2 = field2

        def serialize_field1(self, value):
            return f"serialized_{value}"

    record_with_serializer = TestRecordWithSerializer(field1="value1", field2="value2")
    serialized_with_custom = record_with_serializer.serialize()
    assert serialized_with_custom == {"field1": "serialized_value1", "field2": "value2"}

    class TestRecordWithFormat(PRecord):
        field1 = None
        field2 = None

        def __init__(self, field1, field2):
            self.field1 = field1
            self.field2 = field2

        def serialize_field1(self, value, format):
            if format == "upper":
                return value.upper()
            return value

    record_with_format = TestRecordWithFormat(field1="value1", field2="value2")
    serialized_upper = record_with_format.serialize(format="upper")
    assert serialized_upper == {"field1": "VALUE1", "field2": "value2"}


# LLM-generated content at query #119
#--------------------------

```python
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        x = 0
        y = 0

    # Test normal case
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with mandatory fields
    class TestRecordMandatory(PRecord):
        x = 0
        y = 0
        z = PFIELD_NO_INITIAL

    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant
    class TestRecordInvariant(PRecord):
        x = 0
        y = 0

        __invariant__ = lambda self: (self.x + self.y) > 0

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['x'] = -1
    evolver['y'] = -2
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes
    record = TestRecord(x=1, y=2)
    evolver = _PRecordEvolver(TestRecord, record)
    result = evolver.persistent()
    assert result is record

    # Test with changes
    record = TestRecord(x=1, y=2)
    evolver = _PRecordEvolver(TestRecord, record)
    evolver['x'] = 3
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 3
    assert result.y == 2


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PRecord___new__():
    # Test 1: Basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert isinstance(record, EmptyRecord)
    assert len(record) == 0

    # Test 2: Creation with initial values
    class PersonRecord(PRecord):
        name = None
        age = 0

    record = PersonRecord(name="Alice", age=30)
    assert record.name == "Alice"
    assert record.age == 30
    assert len(record) == 2

    # Test 3: Creation with default initial values
    class DefaultRecord(PRecord):
        x = 10
        y = 20

    record = DefaultRecord()
    assert record.x == 10
    assert record.y == 20

    # Test 4: Overriding initial values
    record = DefaultRecord(x=5)
    assert record.x == 5
    assert record.y == 20

    # Test 5: Creation with callable initial values
    class CallableRecord(PRecord):
        timestamp = lambda: 12345
        value = 100

    record = CallableRecord()
    assert record.timestamp == 12345
    assert record.value == 100

    # Test 6: Creation with factory fields
    class FactoryRecord(PRecord):
        a = None
        b = None

    record = FactoryRecord.create({"a": 1, "b": 2}, _factory_fields=[FactoryRecord._precord_fields["a"]])
    assert record.a == 1
    assert record.b == 2

    # Test 7: Creation with ignore_extra
    class IgnoreExtraRecord(PRecord):
        field1 = None
        field2 = None

    record = IgnoreExtraRecord.create({"field1": 1, "field2": 2, "extra": 3}, ignore_extra=True)
    assert record.field1 == 1
    assert record.field2 == 2
    assert "extra" not in record

    # Test 8: Creation from existing record
    original = PersonRecord(name="Bob", age=25)
    new_record = PersonRecord.create(original)
    assert new_record.name == "Bob"
    assert new_record.age == 25
    assert new_record is original

    # Test 9: Creation with _precord_size and _precord_buckets
    class DirectRecord(PRecord):
        x = None

    record = DirectRecord(_precord_size=1, _precord_buckets=[("x", 100)])
    assert record.x == 100


# LLM-generated content at query #2
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    result = evolver.set('x', 10)
    assert result['x'] == 10
    assert 'x' in result

    # Test setting multiple fields
    result = result.set('y', 20)
    assert result['x'] == 10
    assert result['y'] == 20

    # Test setting an invalid field
    try:
        evolver.set('z', 30)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "'z' is not among the specified fields for TestRecord"

    # Test invariant failure
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    result = evolver.set('x', -1)
    try:
        result.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("x must be non-negative",)


# LLM-generated content at query #3
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 0
        y = 0

    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    evolver['x'] = 3
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 3
    assert result.y == 2

    # Test with mandatory fields missing
    class TestRecordMandatory(PRecord):
        x = 0
        y = 0
        z = None

    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violation
    class TestRecordInvariant(PRecord):
        x = 0
        y = 0

        __invariant__ = lambda self: (self.x >= 0, "x must be non-negative")

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['x'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = 0
        y = 0

        def __invariant__(self):
            if self.x + self.y != 10:
                return False, "x + y must equal 10"
            return True, ""

    evolver = _PRecordEvolver(TestRecordGlobalInvariant, pmap())
    evolver['x'] = 5
    evolver['y'] = 4
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test with changes
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    evolver['x'] = 3
    result = evolver.persistent()
    assert result is not record
    assert result.x == 3
    assert result.y == 2


# LLM-generated content at query #4
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field check
    class MandatoryRecord(PRecord):
        x = 0
        y = 0

        def __invariant__(self):
            if 'x' not in self:
                return False, "x is mandatory"
            return True, None

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('y', 10)
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #5
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with mandatory fields missing
    class MandatoryRecord(PRecord):
        mandatory_field = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violation
    class InvariantRecord(PRecord):
        positive = 0

        def __invariant__(self):
            if self.positive < 0:
                return False, "positive_must_be_positive"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['positive'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes (not dirty)
    record = TestRecord(x=1, y=2)
    evolver = _PRecordEvolver(TestRecord, record)
    result = evolver.persistent()
    assert result is record

    # Test with global invariants
    class GlobalInvariantRecord(PRecord):
        a = 0
        b = 0

    def global_invariant(record):
        if record.a + record.b != 10:
            raise ValueError("Sum must be 10")

    GlobalInvariantRecord._precord_invariants = (global_invariant,)

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['a'] = 5
    evolver['b'] = 5
    result = evolver.persistent()
    assert result.a == 5
    assert result.b == 5

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['a'] = 3
    evolver['b'] = 4
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #6
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class TestRecordWithInvariant(PRecord):
        x = 0
        __invariant__ = lambda self: (self.x >= 0, "x must be non-negative")

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('x', -1)
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field check
    class TestRecordWithMandatory(PRecord):
        x = 0
        y = 1
        _precord_mandatory_fields = {'x', 'y'}

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()

    # Test type checking
    class TestRecordWithType(PRecord):
        x = 0
        _precord_fields = {'x': PFIELD_NO_INITIAL}

    evolver_type = _PRecordEvolver(TestRecordWithType, pmap())
    with pytest.raises(TypeError):
        evolver_type.set('x', 'not an int')


# LLM-generated content at query #7
#--------------------------

```python
def test_PRecord___new__():
    # Test normal creation with field values
    class TestRecord(PRecord):
        x = 0
        y = 1

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 5

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 1
    assert r.z == 5

    # Test creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 0
        y = lambda: 100

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 100

    # Test creation with factory fields
    class TestRecordWithFactory(PRecord):
        x = 0
        y = 1

    r = TestRecordWithFactory._Evolver(TestRecordWithFactory, pmap(), _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    r['x'] = 10
    r['y'] = 20
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test creation with ignore_extra
    class TestRecordIgnoreExtra(PRecord):
        x = 0
        y = 1

    r = TestRecordIgnoreExtra(x=10, y=20, z=30, _ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with direct bucket/size parameters
    class TestRecordDirect(PRecord):
        x = 0
        y = 1

    r = TestRecordDirect(_precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20}))
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #8
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant violation
    class TestRecordWithInvariant(PRecord):
        field1 = None

        @invariant
        def check_field1(self, field1):
            return field1 != 'invalid', 'INVALID_FIELD1'

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field missing
    class TestRecordWithMandatory(PRecord):
        mandatory_field = None

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()


# LLM-generated content at query #9
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}


# LLM-generated content at query #10
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant violation
    class InvariantRecord(PRecord):
        field1 = None

        @invariant
        def check_field1(self, field1):
            if field1 == 'invalid':
                return False, 'INVALID_FIELD1'
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field check
    class MandatoryRecord(PRecord):
        mandatory_field = field(mandatory=True)

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #11
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation
    class TestRecord(PRecord):
        pass

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test with fields
    class TestRecordWithFields(PRecord):
        x = 1
        y = 2

    assert 'x' in TestRecordWithFields._precord_fields
    assert 'y' in TestRecordWithFields._precord_fields
    assert TestRecordWithFields._precord_mandatory_fields == set()

    # Test with mandatory fields
    class TestRecordWithMandatory(PRecord):
        x = 1
        y = 2

    assert TestRecordWithMandatory._precord_mandatory_fields == set()

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2

    assert TestRecordWithInitial._precord_initial_values == {}

    # Test with invariants
    class TestRecordWithInvariant(PRecord):
        __invariant__ = lambda self: True

    assert TestRecordWithInvariant._precord_invariants == [lambda self: True]

    # Test inheritance
    class BaseRecord(PRecord):
        x = 1

    class DerivedRecord(BaseRecord):
        y = 2

    assert 'x' in DerivedRecord._precord_fields
    assert 'y' in DerivedRecord._precord_fields


# LLM-generated content at query #12
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1=10, field2="test")
    assert repr(record) == "TestRecord(field1=10, field2='test')"

    record_empty = TestRecord()
    assert repr(record_empty) == "TestRecord()"


# LLM-generated content at query #13
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 1
        y = 2

    evolver = _PRecordEvolver(TestRecord, pmap({'x': 1, 'y': 2}))
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result == TestRecord(x=1, y=2)

    # Test with changes
    evolver = _PRecordEvolver(TestRecord, pmap({'x': 1, 'y': 2}))
    evolver['x'] = 10
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result == TestRecord(x=10, y=2)

    # Test with missing mandatory fields
    class TestRecordMandatory(PRecord):
        x = 1
        y = 2
        z = None

    evolver = _PRecordEvolver(TestRecordMandatory, pmap({'x': 1, 'y': 2}))
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violations
    class TestRecordInvariant(PRecord):
        x = 1
        y = 2

        __invariant__ = lambda self: (self.x > 0, "x must be positive")

    evolver = _PRecordEvolver(TestRecordInvariant, pmap({'x': -1, 'y': 2}))
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with global invariants
    class TestRecordGlobalInvariant(PRecord):
        x = 1
        y = 2

        __invariant__ = lambda self: (self.x + self.y > 0, "sum must be positive")

    evolver = _PRecordEvolver(TestRecordGlobalInvariant, pmap({'x': -1, 'y': -2}))
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes
    evolver = _PRecordEvolver(TestRecord, pmap({'x': 1, 'y': 2}))
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result == TestRecord(x=1, y=2)


# LLM-generated content at query #14
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = field()
        y = field()

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting invalid field
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class ValidatedRecord(PRecord):
        x = field(invariant=lambda x: (x > 0, 'must be positive'))

    evolver = _PRecordEvolver(ValidatedRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test type checking
    class TypedRecord(PRecord):
        x = field(type=int)

    evolver = _PRecordEvolver(TypedRecord, pmap())
    with pytest.raises(TypeError):
        evolver.set('x', 'not an int')


# LLM-generated content at query #15
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with kwargs
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with initial values
    class TestRecord2(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecord2(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with factory fields
    class TestRecord3(PRecord):
        x = 1
        y = 2

    r = TestRecord3(_factory_fields=[TestRecord3._precord_fields['x']], x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with ignore extra
    class TestRecord4(PRecord):
        x = 1
        y = 2

    r = TestRecord4(_ignore_extra=True, x=10, y=20, z=30)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with callable initial values
    class TestRecord5(PRecord):
        x = lambda: 1
        y = 2

    r = TestRecord5()
    assert r.x == 1
    assert r.y == 2

    # Test creation with _precord_size and _precord_buckets
    class TestRecord6(PRecord):
        x = 1
        y = 2

    r = TestRecord6()
    r2 = TestRecord6(_precord_size=r._size, _precord_buckets=r._buckets)
    assert r2.x == 1
    assert r2.y == 2


# LLM-generated content at query #16
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['x'] == 10
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field check
    class MandatoryRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('x', 10)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'y' in str(exc_info.value)

    # Test type check
    class TypedRecord(PRecord):
        x = 0
        y = ''

    evolver = _PRecordEvolver(TypedRecord, pmap())
    with pytest.raises(TypeError):
        evolver.set('y', 123)

    # Test ignore_extra
    class IgnoreExtraRecord(PRecord):
        x = 0

    evolver = _PRecordEvolver(IgnoreExtraRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 10)
    evolver.set('y', 20)  # This should be ignored
    result = evolver.persistent()
    assert result.x == 10
    assert 'y' not in result


# LLM-generated content at query #17
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord(x=30)
    assert repr(record) == "TestRecord(x=30, y=2)"


# LLM-generated content at query #18
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert isinstance(record, EmptyRecord)
    assert len(record) == 0

    # Test creation with fields
    class PersonRecord(PRecord):
        name = None
        age = None

    record = PersonRecord(name="Alice", age=30)
    assert record.name == "Alice"
    assert record.age == 30
    assert len(record) == 2

    # Test creation with initial values
    class InitialRecord(PRecord):
        x = 1
        y = 2

    record = InitialRecord()
    assert record.x == 1
    assert record.y == 2

    # Test creation with factory fields
    class FactoryRecord(PRecord):
        value = None

    record = FactoryRecord._factory_fields={FactoryRecord._precord_fields['value']}, value=42)
    assert record.value == 42

    # Test creation with ignore_extra
    class IgnoreExtraRecord(PRecord):
        a = None

    record = IgnoreExtraRecord(a=1, b=2, _ignore_extra=True)
    assert record.a == 1
    assert len(record) == 1

    # Test creation with direct bucket/size parameters
    record = PersonRecord(_precord_size=2, _precord_buckets=pmap({'name': 'Bob', 'age': 25})._buckets)
    assert record.name == "Bob"
    assert record.age == 25


# LLM-generated content at query #19
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    # Test setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 0

    # Test setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test setting an invalid field (should raise AttributeError)
    evolver = _PRecordEvolver(TestRecord, pmap())
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x >= 0, "x must be non-negative"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        x = field(mandatory=True)
        y = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('y', 20)
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'MandatoryRecord.x' in exc_info.value.missing_fields

    # Test factory field
    class FactoryRecord(PRecord):
        x = field(factory=lambda v: v * 2)

    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('x', 5)
    result = evolver.persistent()
    assert result.x == 10

    # Test ignore_extra
    class IgnoreExtraRecord(PRecord):
        x = field(ignore_extra=True)
        y = 0

    evolver = _PRecordEvolver(IgnoreExtraRecord, pmap(), _ignore_extra=True)
    evolver.set('x', {'a': 1, 'b': 2})
    result = evolver.persistent()
    assert result.x == {'a': 1, 'b': 2}


# LLM-generated content at query #20
#--------------------------

```python
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        x = 0
        y = 0

    # Test basic persistence
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with mandatory fields missing
    class MandatoryRecord(PRecord):
        x = 0
        y = 0
        z = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['x'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test with global invariants
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

        def __invariant__(self):
            if self.x + self.y != 10:
                return False, "x + y must equal 10"
            return True, None

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 5
    evolver['y'] = 5
    result = evolver.persistent()
    assert result.x == 5
    assert result.y == 5

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 3
    evolver['y'] = 4
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #21
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert isinstance(TestRecord._precord_fields, dict)
    assert isinstance(TestRecord._precord_invariants, dict)
    assert isinstance(TestRecord._precord_mandatory_fields, set)
    assert isinstance(TestRecord._precord_initial_values, dict)


# LLM-generated content at query #22
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record_empty = TestRecord()
    assert repr(record_empty) == "TestRecord(x=1, y=2)"


# LLM-generated content at query #23
#--------------------------

```python
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field(mandatory=True)
        field3 = field(invariant=lambda x: (True, None))

    # Test basic persistence
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('field1', 'value1')
    evolver.set('field2', 'value2')
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.field1 == 'value1'
    assert result.field2 == 'value2'

    # Test with missing mandatory field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('field1', 'value1')
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violation
    class TestRecordWithInvariant(PRecord):
        field1 = field(invariant=lambda x: (x > 0, 'must be positive'))

    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver.set('field1', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with global invariant
    class TestRecordWithGlobalInvariant(PRecord):
        field1 = field()
        field2 = field()

        __invariant__ = lambda self: (self.field1 != self.field2, 'fields must differ')

    evolver = _PRecordEvolver(TestRecordWithGlobalInvariant, pmap())
    evolver.set('field1', 'same')
    evolver.set('field2', 'same')
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes (not dirty)
    record = TestRecord(field1='value1', field2='value2')
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record


# LLM-generated content at query #24
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation of a PRecord class
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        field1 = None
        field2 = None

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test that fields are properly set
    assert 'field1' in TestRecord._precord_fields
    assert 'field2' in TestRecord._precord_fields

    # Test that invariants are stored
    assert TestRecord._precord_invariants == (lambda self: True,)

    # Test that mandatory fields are identified
    assert TestRecord._precord_mandatory_fields == set()

    # Test that initial values are stored
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory fields
    class TestRecordWithMandatory(PRecord):
        field1 = None
        field2 = None, True  # mandatory field

    assert TestRecordWithMandatory._precord_mandatory_fields == {'field2'}

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        field1 = 42
        field2 = lambda: "initial"

    assert TestRecordWithInitial._precord_initial_values == {'field1': 42, 'field2': lambda: "initial"}

    # Test that __slots__ is empty
    assert TestRecordWithInitial.__slots__ == ()


# LLM-generated content at query #25
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    assert result.x == 10

    # Test setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test setting an invalid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('z', 30)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z' is not among the specified fields for TestRecord" in str(e)

    # Test invariant failure
    class TestRecordWithInvariant(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver.set('x', -1)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be non-negative" in str(e)

    # Test mandatory field missing
    class TestRecordWithMandatory(PRecord):
        x = 0
        y = 1

        def __invariant__(self):
            return True, None

    evolver = _PRecordEvolver(TestRecordWithMandatory, pmap())
    evolver.set('x', 10)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestRecordWithMandatory.y" in str(e)


# LLM-generated content at query #26
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert isinstance(record, EmptyRecord)
    assert len(record) == 0

    # Test creation with fields
    class Person(PRecord):
        name = None
        age = 0

    person = Person(name="Alice", age=30)
    assert person.name == "Alice"
    assert person.age == 30
    assert len(person) == 2

    # Test creation with default values
    class DefaultRecord(PRecord):
        x = 10
        y = 20

    record = DefaultRecord()
    assert record.x == 10
    assert record.y == 20

    # Test creation with override of default values
    record = DefaultRecord(x=5)
    assert record.x == 5
    assert record.y == 20

    # Test creation with factory fields
    class FactoryRecord(PRecord):
        a = None
        b = None

    fields = [FactoryRecord._precord_fields['a'], FactoryRecord._precord_fields['b']]
    record = FactoryRecord._factory_fields=fields, a=1, b=2)
    assert record.a == 1
    assert record.b == 2

    # Test creation with ignore_extra
    class IgnoreExtraRecord(PRecord):
        x = None

    record = IgnoreExtraRecord._ignore_extra=True, x=1, y=2)
    assert record.x == 1
    assert 'y' not in record

    # Test creation with internal parameters
    record = Person(_precord_size=2, _precord_buckets=pmap({'name': 'Bob', 'age': 25}))
    assert record.name == 'Bob'
    assert record.age == 25


# LLM-generated content at query #27
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test valid field set
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test invalid field set
    with pytest.raises(AttributeError):
        evolver.set('z', 20)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test factory field
    class FactoryRecord(PRecord):
        x = field(factory=lambda x: x * 2)

    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('x', 5)
    assert evolver['x'] == 10

    # Test ignore_extra
    class IgnoreRecord(PRecord):
        x = field(ignore_extra=True)

    evolver = _PRecordEvolver(IgnoreRecord, pmap(), _ignore_extra=True)
    evolver.set('x', {'a': 1, 'b': 2})
    assert evolver['x'] == {'a': 1, 'b': 2}


# LLM-generated content at query #28
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test normal creation with kwargs
    r1 = TestRecord(x=10, y=20)
    assert r1.x == 10
    assert r1.y == 20

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 5

    r2 = TestRecordWithInitial(x=10)
    assert r2.x == 10
    assert r2.y == 1
    assert r2.z == 5

    # Test creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 0
        y = lambda: 100

    r3 = TestRecordWithCallableInitial(x=10)
    assert r3.x == 10
    assert r3.y == 100

    # Test creation with factory fields
    r4 = TestRecord._factory_fields({'x', 'y'}, x=10, y=20)
    assert r4.x == 10
    assert r4.y == 20

    # Test creation with ignore_extra
    r5 = TestRecord._ignore_extra(True, x=10, y=20, z=30)
    assert r5.x == 10
    assert r5.y == 20
    assert 'z' not in r5

    # Test creation with internal parameters
    r6 = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20}))
    assert r6.x == 10
    assert r6.y == 20


# LLM-generated content at query #29
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class TestRecordWithInvariant(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('x', -1)
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field missing
    class TestRecordWithMandatory(PRecord):
        x = field(mandatory=True)

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()

    # Test factory field
    class TestRecordWithFactory(PRecord):
        x = field(factory=lambda v: v * 2)

    evolver_factory = _PRecordEvolver(TestRecordWithFactory, pmap())
    evolver_factory.set('x', 5)
    assert evolver_factory['x'] == 10

    # Test ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = field(ignore_extra=True)

    evolver_ignore = _PRecordEvolver(TestRecordWithIgnoreExtra, pmap(), _ignore_extra=True)
    evolver_ignore.set('x', {'a': 1, 'b': 2})
    assert evolver_ignore['x'] == {'a': 1, 'b': 2}


# LLM-generated content at query #30
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record_empty = TestRecord()
    assert repr(record_empty) == "TestRecord(x=1, y=2)"

    class AnotherRecord(PRecord):
        name = "default"
        value = None

    record_with_none = AnotherRecord(name="test", value=None)
    assert repr(record_with_none) == "AnotherRecord(name='test', value=None)"


# LLM-generated content at query #31
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}


# LLM-generated content at query #32
#--------------------------

```python
def test_PRecord___new__():
    # Test normal creation with kwargs
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 1
        y = lambda: 2
        z = 3

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with factory fields
    class TestRecordWithFactoryFields(PRecord):
        x = 1
        y = 2

    r = TestRecordWithFactoryFields._PRecordEvolver(TestRecordWithFactoryFields, pmap(), _factory_fields=[TestRecordWithFactoryFields._precord_fields['x']])
    r['x'] = 10
    r['y'] = 20
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test creation with ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 1
        y = 2

    r = TestRecordWithIgnoreExtra._PRecordEvolver(TestRecordWithIgnoreExtra, pmap(), _ignore_extra=True)
    r['x'] = 10
    r['y'] = 20
    r['z'] = 30  # This should be ignored
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20
    assert 'z' not in result

    # Test creation with _precord_size and _precord_buckets
    class TestRecordWithBuckets(PRecord):
        x = 1
        y = 2

    r = TestRecordWithBuckets(_precord_size=2, _precord_buckets=pmap(x=10, y=20)._buckets)
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #33
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    r = EmptyRecord()
    assert isinstance(r, EmptyRecord)
    assert len(r) == 0

    # Test creation with fields
    class Person(PRecord):
        name = None
        age = 0

    p = Person(name="Alice", age=30)
    assert p.name == "Alice"
    assert p.age == 30
    assert len(p) == 2

    # Test creation with default values
    class DefaultRecord(PRecord):
        x = 1
        y = 2

    dr = DefaultRecord()
    assert dr.x == 1
    assert dr.y == 2

    # Test creation with override of default values
    dr2 = DefaultRecord(x=10)
    assert dr2.x == 10
    assert dr2.y == 2

    # Test creation with factory_fields
    class FactoryRecord(PRecord):
        a = None
        b = None

    fr = FactoryRecord.create({"a": 1, "b": 2}, _factory_fields=[FactoryRecord._precord_fields["a"]])
    assert fr.a == 1
    assert fr.b == 2

    # Test creation with ignore_extra
    class IgnoreExtraRecord(PRecord):
        field1 = None

    ier = IgnoreExtraRecord.create({"field1": 1, "extra": 2}, ignore_extra=True)
    assert ier.field1 == 1
    assert "extra" not in ier

    # Test creation with internal parameters
    class InternalRecord(PRecord):
        pass

    ir = InternalRecord(_precord_size=0, _precord_buckets=())
    assert isinstance(ir, InternalRecord)
    assert len(ir) == 0


# LLM-generated content at query #34
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record_empty = TestRecord()
    assert repr(record_empty) == "TestRecord(x=1, y=2)"

    class TestRecordWithString(PRecord):
        name = "default"

    record_string = TestRecordWithString(name="test")
    assert repr(record_string) == "TestRecordWithString(name='test')"


# LLM-generated content at query #35
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0
        __invariant__ = lambda self: self.x >= 0

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        x = 0
        y = 0
        __mandatory__ = ['x', 'y']

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('x', 10)
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #36
#--------------------------

```python
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        x = int
        y = str

    # Test basic persistent functionality
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 10
    evolver['y'] = "test"
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == "test"

    # Test with mandatory fields missing
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 10
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecord.y' in e.missing_fields

    # Test with invariant violation
    class ValidatedRecord(PRecord):
        x = int
        y = str

        def __invariant__(self):
            if self.x < 0:
                return False, "x_must_be_positive"
            return True, None

    evolver = _PRecordEvolver(ValidatedRecord, pmap())
    evolver['x'] = -1
    evolver['y'] = "test"
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "x_must_be_positive" in e.invariant_errors

    # Test with no changes (is_dirty=False)
    record = TestRecord(x=5, y="original")
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test with changes (is_dirty=True)
    evolver = record.evolver()
    evolver['x'] = 10
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == "original"
    assert result is not record


# LLM-generated content at query #37
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        name = None
        age = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('name', 'Alice')
    assert evolver.persistent().name == 'Alice'

    evolver.set('age', 30)
    assert evolver.persistent().age == 30

    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    with pytest.raises(InvariantException):
        class InvalidRecord(PRecord):
            name = None
            age = None

            __invariant__ = lambda self: (self.age >= 0, "Age must be non-negative")

        evolver = _PRecordEvolver(InvalidRecord, pmap())
        evolver.set('age', -1)
        evolver.persistent()


# LLM-generated content at query #38
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = TestRecord.evolver()

    # Test valid field setting
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test invalid field setting (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant violation
    class InvariantRecord(PRecord):
        field1 = None

        @invariant
        def check_field1(self):
            return self.field1 != 'invalid', 'field1_cannot_be_invalid'

    evolver = InvariantRecord.evolver()
    evolver.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test type checking
    class TypedRecord(PRecord):
        field1 = field(type=int)

    evolver = TypedRecord.evolver()
    with pytest.raises(TypeError):
        evolver.set('field1', 'not_an_int')


# LLM-generated content at query #39
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant violation
    class TestRecordWithInvariant(PRecord):
        field1 = None

        def __invariant__(self):
            if self.field1 == 'invalid':
                return False, 'INVALID_VALUE'
            return True, None

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field check
    class TestRecordWithMandatory(PRecord):
        mandatory_field = None

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()


# LLM-generated content at query #40
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = TestRecord().evolver()
    evolver['x'] = 1
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 0

    # Test with mandatory fields
    class MandatoryRecord(PRecord):
        mandatory_field = 0

    evolver = MandatoryRecord().evolver()
    evolver['mandatory_field'] = 1
    result = evolver.persistent()
    assert result.mandatory_field == 1

    # Test missing mandatory field raises InvariantException
    evolver = MandatoryRecord().evolver()
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test invariant violation raises InvariantException
    class InvariantRecord(PRecord):
        value = 0

        @__invariant__
        def check_value(self):
            return self.value >= 0, "value must be non-negative"

    evolver = InvariantRecord().evolver()
    evolver['value'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test global invariant
    class GlobalInvariantRecord(PRecord):
        a = 0
        b = 0

        @__invariant__
        def check_sum(self):
            return self.a + self.b == 0, "sum must be zero"

    evolver = GlobalInvariantRecord().evolver()
    evolver['a'] = 1
    evolver['b'] = -1
    result = evolver.persistent()
    assert result.a == 1
    assert result.b == -1

    evolver = GlobalInvariantRecord().evolver()
    evolver['a'] = 1
    evolver['b'] = 1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test that unchanged evolver returns original instance
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test that changed evolver returns new instance
    evolver = record.evolver()
    evolver['x'] = 3
    result = evolver.persistent()
    assert result is not record
    assert result.x == 3
    assert result.y == 2


# LLM-generated content at query #41
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = PFIELD_NO_INITIAL
        field2 = PFIELD_NO_INITIAL

    record = TestRecord(field1="value1", field2="value2")
    assert record.serialize() == {"field1": "value1", "field2": "value2"}

    class TestRecordWithSerializer(PRecord):
        field1 = PFIELD_NO_INITIAL
        field2 = PFIELD_NO_INITIAL

    record_with_serializer = TestRecordWithSerializer(field1="value1", field2="value2")
    assert record_with_serializer.serialize(format="json") == {"field1": "value1", "field2": "value2"}


# LLM-generated content at query #42
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord(x=10)
    assert repr(record) == "TestRecord(x=10, y=2)"

    record = TestRecord()
    assert repr(record) == "TestRecord(x=1, y=2)"


# LLM-generated content at query #43
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation of a PRecord class
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test that fields are properly set
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test that mandatory fields are identified
    assert TestRecord._precord_mandatory_fields == set()

    # Test that initial values are stored
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory fields
    class TestRecordMandatory(PRecord):
        x = 1
        y = 2
        z = mandatory()

    assert 'z' in TestRecordMandatory._precord_mandatory_fields

    # Test with initial values
    class TestRecordInitial(PRecord):
        x = field(initial=1)
        y = field(initial=lambda: 2)

    assert TestRecordInitial._precord_initial_values == {'x': 1, 'y': 2}

    # Test with invariants
    class TestRecordInvariant(PRecord):
        x = 1
        __invariant__ = lambda self: self.x > 0

    assert len(TestRecordInvariant._precord_invariants) == 1


# LLM-generated content at query #44
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record_empty = TestRecord()
    assert repr(record_empty) == "TestRecord(x=1, y=2)"

    class AnotherRecord(PRecord):
        name = "default"
        value = 0

    another_record = AnotherRecord(name="test", value=42)
    assert repr(another_record) == "AnotherRecord(name='test', value=42)"


# LLM-generated content at query #45
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 100
    result = evolver.persistent()
    assert result.x == 100
    assert result.y == 20
    assert isinstance(result, TestRecord)

    # Test with dirty evolver
    evolver = record.evolver()
    evolver['x'] = 200
    evolver['y'] = 300
    result = evolver.persistent()
    assert result.x == 200
    assert result.y == 300
    assert isinstance(result, TestRecord)

    # Test with clean evolver
    evolver = record.evolver()
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 20
    assert isinstance(result, TestRecord)

    # Test mandatory fields
    class TestRecordMandatory(PRecord):
        x = 1
        y = 2
        z = PFIELD_NO_INITIAL

    record = TestRecordMandatory(x=10, y=20, z=30)
    evolver = record.evolver()
    evolver['z'] = PFIELD_NO_INITIAL
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'TestRecordMandatory.z' in e.missing_fields

    # Test invariant violation
    class TestRecordInvariant(PRecord):
        x = 1
        y = 2

        def __invariant__(self):
            if self.x > self.y:
                return False, "x_must_be_less_than_y"
            return True, None

    record = TestRecordInvariant(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 30
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "x_must_be_less_than_y" in e.invariant_errors

    # Test with factory fields
    class TestRecordFactory(PRecord):
        x = 1
        y = 2

    record = TestRecordFactory(x=10, y=20)
    evolver = record.evolver(_factory_fields=[TestRecordFactory._precord_fields['x']])
    evolver['x'] = 100
    result = evolver.persistent()
    assert result.x == 100
    assert result.y == 20

    # Test with ignore_extra
    class TestRecordIgnoreExtra(PRecord):
        x = 1
        y = 2

    record = TestRecordIgnoreExtra(x=10, y=20)
    evolver = record.evolver(_ignore_extra=True)
    evolver['x'] = 100
    result = evolver.persistent()
    assert result.x == 100
    assert result.y == 20


# LLM-generated content at query #46
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 0
        y = 1

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 5

    r = TestRecordWithInitial(x=10, y=20)
    assert r.x == 10
    assert r.y == 20
    assert r.z == 5

    # Test with callable initial value
    class TestRecordWithCallableInitial(PRecord):
        x = 0
        y = lambda: 100

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 100

    # Test with factory fields
    class TestRecordWithFactory(PRecord):
        x = 0
        y = 1

    r = TestRecordWithFactory._Evolver(TestRecordWithFactory, pmap(), _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    r['x'] = 10
    r['y'] = 20
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test with ignore_extra
    class TestRecordIgnoreExtra(PRecord):
        x = 0
        y = 1

    r = TestRecordIgnoreExtra(x=10, y=20, z=30, _ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test with internal fields
    r = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20})._buckets)
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #47
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass functionality
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field setup
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test mandatory fields
    class MandatoryRecord(PRecord):
        a = 1
        b = 2
        c = 3

    assert MandatoryRecord._precord_mandatory_fields == {'a', 'b', 'c'}

    # Test initial values
    class InitialRecord(PRecord):
        x = 1
        y = 2
        z = 3

    assert InitialRecord._precord_initial_values == {'x': 1, 'y': 2, 'z': 3}

    # Test inheritance
    class BaseRecord(PRecord):
        base_field = 1

    class DerivedRecord(BaseRecord):
        derived_field = 2

    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields


# LLM-generated content at query #48
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation of a PRecord class
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test that fields are properly set
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test that mandatory fields are identified
    assert TestRecord._precord_mandatory_fields == set()

    # Test that initial values are set
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory fields
    class TestRecordMandatory(PRecord):
        x = 1
        y = 2
        z = 3

        __invariant__ = lambda self: len(self) > 0

    assert TestRecordMandatory._precord_mandatory_fields == set()

    # Test with initial values
    class TestRecordInitial(PRecord):
        x = 1
        y = 2
        z = 3

        __invariant__ = lambda self: len(self) > 0

    assert TestRecordInitial._precord_initial_values == {}

    # Test that invariants are stored
    assert hasattr(TestRecordInitial, '_precord_invariants')


# LLM-generated content at query #49
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass functionality
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field setup
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test mandatory fields
    assert TestRecord._precord_mandatory_fields == set()

    # Test initial values
    assert TestRecord._precord_initial_values == {}

    # Test with mandatory field
    class TestRecordMandatory(PRecord):
        x = 1
        y = 2
        z = 3

    assert TestRecordMandatory._precord_mandatory_fields == set()

    # Test with initial values
    class TestRecordInitial(PRecord):
        x = 1
        y = 2
        z = 3

    assert TestRecordInitial._precord_initial_values == {}

    # Test with actual field definitions
    from pyrsistent._field_common import PRecordField
    class TestRecordWithFields(PRecord):
        x = PRecordField(type=int, mandatory=True)
        y = PRecordField(type=str, initial="default")
        z = PRecordField(type=float)

    assert 'x' in TestRecordWithFields._precord_fields
    assert 'y' in TestRecordWithFields._precord_fields
    assert 'z' in TestRecordWithFields._precord_fields

    assert TestRecordWithFields._precord_mandatory_fields == {'x'}
    assert TestRecordWithFields._precord_initial_values == {'y': "default"}

    # Test inheritance
    class BaseRecord(PRecord):
        a = PRecordField(type=int)

    class DerivedRecord(BaseRecord):
        b = PRecordField(type=str)

    assert 'a' in DerivedRecord._precord_fields
    assert 'b' in DerivedRecord._precord_fields
    assert DerivedRecord._precord_mandatory_fields == set()
    assert DerivedRecord._precord_initial_values == {}


# LLM-generated content at query #50
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with kwargs
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with factory_fields
    r = TestRecord(_factory_fields=[TestRecord._precord_fields['x']], x=100)
    assert r.x == 100
    assert r.y == 2

    # Test creation with ignore_extra
    r = TestRecord(_ignore_extra=True, x=10, y=20, z=30)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with initial values
    class TestRecord2(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecord2(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with callable initial values
    class TestRecord3(PRecord):
        x = 1
        y = lambda: 2
        z = 3

    r = TestRecord3()
    assert r.x == 1
    assert r.y == 2
    assert r.z == 3

    # Test creation with _precord_size and _precord_buckets
    r = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 1, 'y': 2}))
    assert r.x == 1
    assert r.y == 2


# LLM-generated content at query #51
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: len(self) > 0
        field1 = None
        field2 = 1

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert 'field1' in TestRecord._precord_fields
    assert 'field2' in TestRecord._precord_fields
    assert TestRecord._precord_initial_values == {'field2': 1}
    assert TestRecord._precord_mandatory_fields == set()


# LLM-generated content at query #52
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 1
        y = 2

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 10
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 10
    assert result['y'] == 2

    # Test persistence with dirty evolver
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 10
    evolver['y'] = 20
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 10
    assert result['y'] == 20

    # Test persistence with clean evolver
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['x'] == 10
    assert result['y'] == 20

    # Test persistence with mandatory fields missing
    class TestRecordMandatory(PRecord):
        x = 1
        y = 2
        z = 3

    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['x'] = 10
    evolver['y'] = 20
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with invariant violation
    class TestRecordInvariant(PRecord):
        x = 1
        y = 2

        __invariant__ = lambda self: (self.x > 0, 'x must be positive')

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['x'] = -1
    evolver['y'] = 20
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = 1
        y = 2

        def __invariant__(self):
            if self.x + self.y != 3:
                return False, 'sum must be 3'
            return True, ''

    evolver = _PRecordEvolver(TestRecordGlobalInvariant, pmap())
    evolver['x'] = 1
    evolver['y'] = 1
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #53
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 0
        y = 1

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field storage
    assert 'x' in TestRecord._precord_fields
    assert 'y' in TestRecord._precord_fields

    # Test mandatory fields
    class TestRecordMandatory(PRecord):
        a = 0
        b = 1
        c = 2

    assert TestRecordMandatory._precord_mandatory_fields == {'a', 'b', 'c'}

    # Test initial values
    class TestRecordInitial(PRecord):
        x = 0
        y = 1
        z = 2

    assert TestRecordInitial._precord_initial_values == {'x': 0, 'y': 1, 'z': 2}

    # Test inheritance
    class BaseRecord(PRecord):
        base_field = 0

    class DerivedRecord(BaseRecord):
        derived_field = 1

    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields
    assert DerivedRecord._precord_mandatory_fields == {'base_field', 'derived_field'}


# LLM-generated content at query #54
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field(serializer=lambda x: x.upper())

    record = TestRecord(field1="value1", field2="value2")
    serialized = record.serialize()

    assert serialized["field1"] == "value1"
    assert serialized["field2"] == "VALUE2"


# LLM-generated content at query #55
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant failure
    class TestRecordWithInvariant(PRecord):
        field1 = None

        @invariant
        def check_field1(self):
            if self.field1 == 'invalid':
                return False, 'INVALID_FIELD1'
            return True, None

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field check
    class TestRecordWithMandatory(PRecord):
        mandatory_field = None

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()

    # Test type checking
    class TestRecordWithType(PRecord):
        typed_field = None

    evolver_type = _PRecordEvolver(TestRecordWithType, pmap())
    with pytest.raises(TypeError):
        evolver_type.set('typed_field', 'invalid_type')


# LLM-generated content at query #56
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        field1 = None
        field2 = 0
        __invariant__ = lambda self: self.field1 is not None

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert 'field1' in TestRecord._precord_fields
    assert 'field2' in TestRecord._precord_fields
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {'field2': 0}
    assert TestRecord._precord_invariants == [TestRecord.__invariant__]


# LLM-generated content at query #57
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x >= 0, "x must be non-negative"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        x = field(mandatory=True)

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #58
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 15
    result = evolver.persistent()
    assert result.x == 15
    assert result.y == 20

    # Test mandatory fields
    class MandatoryRecord(PRecord):
        x = 0
        y = 1

    evolver = MandatoryRecord(x=10).evolver()
    evolver['x'] = 15
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        __invariant__ = lambda self: (self.x > 0, "x must be positive")

    evolver = InvariantRecord(x=10).evolver()
    evolver['x'] = -5
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test global invariants
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

        __invariant__ = lambda self: (self.x + self.y > 0, "sum must be positive")

    evolver = GlobalInvariantRecord(x=10, y=-5).evolver()
    evolver['x'] = -10
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test no changes
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test factory fields
    class FactoryRecord(PRecord):
        x = 0
        y = 1

    evolver = FactoryRecord(x=10, y=20).evolver()
    evolver['x'] = 15
    result = evolver.persistent()
    assert result.x == 15
    assert result.y == 20


# LLM-generated content at query #59
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 100
    result = evolver.persistent()
    assert result.x == 100
    assert result.y == 20
    assert isinstance(result, TestRecord)

    # Test persistence with no changes
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test persistence with mandatory fields missing
    class MandatoryRecord(PRecord):
        mandatory_field = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with invariant violations
    class InvariantRecord(PRecord):
        positive = 1

        @invariant
        def positive_invariant(self, record):
            if record.positive <= 0:
                return False, "positive.must_be_positive"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap(positive=-1))
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with global invariants
    class GlobalInvariantRecord(PRecord):
        a = 1
        b = 2

        @__invariant__
        def global_invariant(self, record):
            if record.a + record.b != 3:
                return False, "sum.must_be_three"
            return True, None

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap(a=1, b=1))
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #60
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test normal creation
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 5
        y = 10

    r2 = TestRecordWithInitial()
    assert r2.x == 5
    assert r2.y == 10

    # Test with override of initial values
    r3 = TestRecordWithInitial(x=15)
    assert r3.x == 15
    assert r3.y == 10

    # Test with callable initial values
    class TestRecordWithCallable(PRecord):
        x = lambda: 5
        y = 10

    r4 = TestRecordWithCallable()
    assert r4.x == 5
    assert r4.y == 10

    # Test with factory fields
    r5 = TestRecord._PRecordEvolver(TestRecord, pmap()).persistent()
    assert isinstance(r5, TestRecord)

    # Test with _precord_size and _precord_buckets
    r6 = TestRecord(_precord_size=2, _precord_buckets=pmap(x=10, y=20)._buckets)
    assert r6.x == 10
    assert r6.y == 20

    # Test with ignore_extra
    r7 = TestRecord.create({'x': 10, 'y': 20, 'z': 30}, ignore_extra=True)
    assert r7.x == 10
    assert r7.y == 20
    assert 'z' not in r7


# LLM-generated content at query #61
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with mandatory fields
    class TestRecordMandatory(PRecord):
        x = 0
        y = 0
        z = 0

    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violation
    class TestRecordInvariant(PRecord):
        x = 0
        y = 0

        __invariant__ = lambda self: (self.x + self.y) > 0

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['x'] = -1
    evolver['y'] = -2
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes
    record = TestRecord(x=1, y=2)
    evolver = _PRecordEvolver(TestRecord, record)
    result = evolver.persistent()
    assert result is record

    # Test with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = 0
        y = 0

    def global_inv(record):
        if record.x < 0 or record.y < 0:
            raise InvariantException(("negative_value",), (), "Negative values not allowed")

    TestRecordGlobalInvariant._precord_invariants = (global_inv,)

    evolver = _PRecordEvolver(TestRecordGlobalInvariant, pmap())
    evolver['x'] = -1
    evolver['y'] = 2
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #62
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with kwargs
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with factory_fields
    r = TestRecord(_factory_fields={TestRecord._precord_fields['x']}, x=100)
    assert r.x == 100
    assert r.y == 2

    # Test creation with ignore_extra
    r = TestRecord(_ignore_extra=True, x=10, z=30)
    assert r.x == 10
    assert r.y == 2
    assert 'z' not in r

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = PFIELD_NO_INITIAL

    r = TestRecordWithInitial()
    assert r.x == 1
    assert r.y == 2
    assert 'z' not in r

    # Test creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = lambda: 1
        y = 2

    r = TestRecordWithCallableInitial()
    assert r.x == 1
    assert r.y == 2

    # Test creation with existing PRecord
    r1 = TestRecord(x=10, y=20)
    r2 = TestRecord.create(r1)
    assert r2.x == 10
    assert r2.y == 20


# LLM-generated content at query #63
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with kwargs
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with factory_fields
    r = TestRecord(_factory_fields=[TestRecord._precord_fields['x']], x=100)
    assert r.x == 100
    assert r.y == 2

    # Test creation with ignore_extra
    r = TestRecord(_ignore_extra=True, x=10, z=30)
    assert r.x == 10
    assert r.y == 2
    assert 'z' not in r

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = PFIELD_NO_INITIAL

    r = TestRecordWithInitial(z=3)
    assert r.x == 1
    assert r.y == 2
    assert r.z == 3

    # Test creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = lambda: 1
        y = 2

    r = TestRecordWithCallableInitial()
    assert r.x == 1
    assert r.y == 2

    # Test creation with _precord_size and _precord_buckets
    r = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 1, 'y': 2}))
    assert r.x == 1
    assert r.y == 2


# LLM-generated content at query #64
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 0

    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    evolver['x'] = 3
    result = evolver.persistent()
    assert result.x == 3
    assert result.y == 2
    assert isinstance(result, TestRecord)

    # Test persistence with no changes
    evolver2 = record.evolver()
    result2 = evolver2.persistent()
    assert result2 is record

    # Test persistence with mandatory fields missing
    class TestRecordMandatory(PRecord):
        x = 0
        y = 0
        z = PFIELD_NO_INITIAL

    evolver3 = TestRecordMandatory(x=1, y=2).evolver()
    evolver3['z'] = None  # Remove mandatory field
    try:
        evolver3.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecordMandatory.z' in e.missing_fields

    # Test persistence with invariant violation
    class TestRecordInvariant(PRecord):
        x = 0

        @__invariant__
        def check_x(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver4 = TestRecordInvariant(x=1).evolver()
    evolver4['x'] = -1
    try:
        evolver4.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert ("x must be non-negative",) == e.invariant_errors

    # Test persistence with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = 0
        y = 0

        @__invariant__
        def check_sum(self):
            if self.x + self.y != 10:
                return False, "sum must be 10"
            return True, None

    evolver5 = TestRecordGlobalInvariant(x=5, y=5).evolver()
    evolver5['x'] = 6
    try:
        evolver5.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert ("sum must be 10",) == e.invariant_errors


# LLM-generated content at query #65
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields via kwargs
    evolver.set(x=20, y=30)
    assert evolver['x'] == 20
    assert evolver['y'] == 30

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 40)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        x = field(mandatory=True)

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test ignore_extra flag
    evolver = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 10)
    evolver.set('z', 20)  # This should be ignored
    assert evolver['x'] == 10
    assert 'z' not in evolver


# LLM-generated content at query #66
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=10, field2="test")

    # Test default serialization
    serialized = record.serialize()
    assert serialized == {"field1": 10, "field2": "test"}

    # Test with custom serializer
    class CustomRecord(PRecord):
        field1 = field(serializer=lambda x: x * 2)
        field2 = field()

    custom_record = CustomRecord(field1=5, field2="hello")
    serialized_custom = custom_record.serialize()
    assert serialized_custom == {"field1": 10, "field2": "hello"}

    # Test with format parameter
    class FormatRecord(PRecord):
        field1 = field(serializer=lambda x, fmt: str(x) if fmt == "str" else x)

    format_record = FormatRecord(field1=42)
    serialized_format = format_record.serialize(format="str")
    assert serialized_format == {"field1": "42"}


# LLM-generated content at query #67
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field inheritance
    class ParentRecord(PRecord):
        a = 1

    class ChildRecord(ParentRecord):
        b = 2

    assert 'a' in ChildRecord._precord_fields
    assert 'b' in ChildRecord._precord_fields

    # Test mandatory fields tracking
    from pyrsistent._field_common import PField
    class MandatoryRecord(PRecord):
        mandatory_field = PField(mandatory=True)
        optional_field = PField()

    assert 'mandatory_field' in MandatoryRecord._precord_mandatory_fields
    assert 'optional_field' not in MandatoryRecord._precord_mandatory_fields

    # Test initial values
    class InitialRecord(PRecord):
        with_initial = PField(initial=42)
        with_callable = PField(initial=lambda: "test")
        no_initial = PField()

    assert InitialRecord._precord_initial_values['with_initial'] == 42
    assert InitialRecord._precord_initial_values['with_callable']() == "test"
    assert 'no_initial' not in InitialRecord._precord_initial_values


# LLM-generated content at query #68
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = PFIELD_NO_INITIAL
        field2 = PFIELD_NO_INITIAL

    record = TestRecord(field1="value1", field2="value2")

    # Test basic serialization
    serialized = record.serialize()
    assert serialized == {"field1": "value1", "field2": "value2"}

    # Test serialization with custom serializer
    class CustomRecord(PRecord):
        field1 = PFIELD_NO_INITIAL
        field2 = PFIELD_NO_INITIAL

    # Mock serializer function
    def custom_serializer(value):
        return f"serialized_{value}"

    CustomRecord._precord_fields["field1"].serializer = custom_serializer
    custom_record = CustomRecord(field1="value1", field2="value2")

    serialized_custom = custom_record.serialize()
    assert serialized_custom == {"field1": "serialized_value1", "field2": "value2"}

    # Test serialization with format parameter
    def format_serializer(format, value):
        if format == "upper":
            return value.upper()
        return value

    CustomRecord._precord_fields["field2"].serializer = format_serializer
    serialized_format = custom_record.serialize(format="upper")
    assert serialized_format == {"field1": "serialized_value1", "field2": "VALUE2"}


# LLM-generated content at query #69
#--------------------------

```python
def test_PRecord___new__():
    # Test normal creation with kwargs
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with factory_fields
    class TestRecordWithFactory(PRecord):
        x = 1
        y = 2

    r = TestRecordWithFactory._PRecordEvolver(TestRecordWithFactory, pmap(), _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    r['x'] = 10
    r['y'] = 20
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test creation with ignore_extra
    class TestRecordIgnoreExtra(PRecord):
        x = 1
        y = 2

    r = TestRecordIgnoreExtra(x=10, y=20, z=30, _ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with _precord_size and _precord_buckets
    class TestRecordDirect(PRecord):
        x = 1
        y = 2

    r = TestRecordDirect(_precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20})._buckets)
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #70
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: len(self) > 0
        field1 = None
        field2 = 1

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord._precord_initial_values == {'field2': 1}
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord.__slots__ == ()


# LLM-generated content at query #71
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 15
    result = evolver.persistent()
    assert result.x == 15
    assert result.y == 20
    assert isinstance(result, TestRecord)

    # Test with mandatory fields missing
    class MandatoryRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['x'] = 10
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'y' in str(e)

    # Test with invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['x'] = -1
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be non-negative" in str(e)

    # Test with no changes (not dirty)
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test with global invariant
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

    def global_invariant(record):
        if record.x + record.y != 10:
            raise ValueError("Sum must be 10")

    GlobalInvariantRecord._precord_invariants = [global_invariant]

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 5
    evolver['y'] = 5
    result = evolver.persistent()
    assert result.x == 5
    assert result.y == 5

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 3
    evolver['y'] = 4
    try:
        evolver.persistent()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Sum must be 10" in str(e)


# LLM-generated content at query #72
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with kwargs
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with partial kwargs
    r = TestRecord(x=10)
    assert r.x == 10
    assert r.y == 2

    # Test creation with callable initial values
    class TestRecord2(PRecord):
        x = lambda: 1
        y = lambda: 2

    r = TestRecord2()
    assert r.x == 1
    assert r.y == 2

    # Test creation with factory_fields
    r = TestRecord._evolver().set('x', 10).set('y', 20).persistent()
    assert r.x == 10
    assert r.y == 20

    # Test creation with ignore_extra
    r = TestRecord._evolver(_ignore_extra=True).set('x', 10).set('z', 30).persistent()
    assert r.x == 10
    assert 'z' not in r

    # Test creation with _precord_size and _precord_buckets
    r = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 10, 'y': 20}))
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #73
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with kwargs
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with partial kwargs
    r = TestRecord(x=10)
    assert r.x == 10
    assert r.y == 2

    # Test creation with factory_fields
    r = TestRecord(_factory_fields=[TestRecord._precord_fields['x']], x=10)
    assert r.x == 10
    assert r.y == 2

    # Test creation with ignore_extra
    r = TestRecord(_ignore_extra=True, x=10, y=20, z=30)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with _precord_size and _precord_buckets
    r = TestRecord(x=10, y=20)
    r2 = TestRecord(_precord_size=r._size, _precord_buckets=r._buckets)
    assert r2.x == 10
    assert r2.y == 20

    # Test creation with callable initial values
    class TestRecordCallable(PRecord):
        x = lambda: 1
        y = 2

    r = TestRecordCallable()
    assert r.x == 1
    assert r.y == 2


# LLM-generated content at query #74
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord()
    assert repr(record) == "TestRecord(x=0, y=1)"


# LLM-generated content at query #75
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field(serializer=lambda x: x.upper())

    record = TestRecord(field1="value1", field2="value2")

    # Test with default format
    serialized = record.serialize()
    assert serialized == {"field1": "value1", "field2": "VALUE2"}

    # Test with custom format
    serialized_custom = record.serialize(format="custom")
    assert serialized_custom == {"field1": "value1", "field2": "VALUE2"}


# LLM-generated content at query #76
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    record = EmptyRecord()
    assert isinstance(record, EmptyRecord)
    assert len(record) == 0

    # Test creation with fields
    class Person(PRecord):
        name = None
        age = 0

    person = Person(name="Alice", age=30)
    assert person.name == "Alice"
    assert person.age == 30
    assert len(person) == 2

    # Test creation with default values
    class DefaultRecord(PRecord):
        x = 1
        y = 2

    record = DefaultRecord()
    assert record.x == 1
    assert record.y == 2

    # Test creation with override of default values
    record = DefaultRecord(x=10)
    assert record.x == 10
    assert record.y == 2

    # Test creation with factory fields
    class FactoryRecord(PRecord):
        a = None
        b = None

    evolver = FactoryRecord._Evolver(FactoryRecord, pmap(), _factory_fields={FactoryRecord._precord_fields['a']})
    evolver['a'] = "test"
    evolver['b'] = "ignored"
    record = evolver.persistent()
    assert record.a == "test"
    assert 'b' not in record

    # Test creation with ignore_extra
    class IgnoreExtraRecord(PRecord):
        field1 = None

    record = IgnoreExtraRecord(field1="value1", extra_field="ignored", _ignore_extra=True)
    assert record.field1 == "value1"
    assert 'extra_field' not in record

    # Test creation from existing record
    original = Person(name="Bob", age=25)
    new_person = Person.create(original)
    assert new_person.name == "Bob"
    assert new_person.age == 25

    # Test pickling support
    import pickle
    pickled = pickle.dumps(person)
    unpickled = pickle.loads(pickled)
    assert unpickled.name == "Alice"
    assert unpickled.age == 30


# LLM-generated content at query #77
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent with no changes
    class TestRecord(PRecord):
        x = 0
        y = 0

    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result == record
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test persistent with changes
    evolver['x'] = 10
    result = evolver.persistent()
    assert result != record
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == 2

    # Test persistent with mandatory field missing
    class TestRecordMandatory(PRecord):
        x = 0
        y = 0
        z = None, mandatory=True

    record = TestRecordMandatory(x=1, y=2, z=3)
    evolver = record.evolver()
    del evolver['z']
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistent with invariant violation
    class TestRecordInvariant(PRecord):
        x = 0
        y = 0

        __invariant__ = lambda self: (self.x + self.y) > 0

    record = TestRecordInvariant(x=1, y=2)
    evolver = record.evolver()
    evolver['x'] = -10
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistent with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = 0
        y = 0

    def global_invariant(record):
        return (record.x + record.y) > 0

    TestRecordGlobalInvariant._precord_invariants = (global_invariant,)

    record = TestRecordGlobalInvariant(x=1, y=2)
    evolver = record.evolver()
    evolver['x'] = -10
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistent with factory fields
    class TestRecordFactory(PRecord):
        x = 0
        y = 0

    record = TestRecordFactory(x=1, y=2)
    evolver = record.evolver(_factory_fields=(TestRecordFactory._precord_fields['x'],))
    evolver['x'] = 10
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 2


# LLM-generated content at query #78
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field(serializer=lambda x: x.upper())

    record = TestRecord(field1="value1", field2="value2")

    # Test default serialization
    assert record.serialize() == {"field1": "value1", "field2": "value2"}

    # Test custom serializer
    assert record.serialize()["field2"] == "VALUE2"

    # Test with format parameter
    assert record.serialize(format="custom") == {"field1": "value1", "field2": "VALUE2"}


# LLM-generated content at query #79
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 100
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 100
    assert result.y == 20

    # Test with mandatory fields missing
    class MandatoryRecord(PRecord):
        mandatory_field = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violation
    class InvariantRecord(PRecord):
        field = 0

        @invariant
        def check_field(self):
            return self.field > 0, "field must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['field'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes (not dirty)
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test with global invariant
    class GlobalInvariantRecord(PRecord):
        a = 0
        b = 0

        @global_invariant
        def check_sum(self):
            return self.a + self.b == 10, "sum must be 10"

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['a'] = 5
    evolver['b'] = 5
    result = evolver.persistent()
    assert result.a == 5
    assert result.b == 5

    evolver['a'] = 6
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #80
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = TestRecord().evolver()

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = InvariantRecord().evolver()
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        x = 0
        y = 0

        def __invariant__(self):
            if 'x' not in self:
                return False, "x is mandatory"
            return True, None

    evolver = MandatoryRecord().evolver()
    evolver.set('y', 10)
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #81
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass creation
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        field1 = None
        field2 = None

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field setup
    assert 'field1' in TestRecord._precord_fields
    assert 'field2' in TestRecord._precord_fields

    # Test mandatory fields
    class TestRecordMandatory(PRecord):
        __invariant__ = lambda self: True
        mandatory_field = None

    assert 'mandatory_field' in TestRecordMandatory._precord_mandatory_fields

    # Test initial values
    class TestRecordInitial(PRecord):
        __invariant__ = lambda self: True
        initial_field = 42

    assert TestRecordInitial._precord_initial_values['initial_field'] == 42

    # Test inheritance
    class BaseRecord(PRecord):
        __invariant__ = lambda self: True
        base_field = None

    class DerivedRecord(BaseRecord):
        __invariant__ = lambda self: True
        derived_field = None

    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields


# LLM-generated content at query #82
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test valid field assignment
    result = evolver.set('x', 10)
    assert result['x'] == 10
    assert 'x' in result

    # Test invalid field assignment
    with pytest.raises(AttributeError):
        evolver.set('z', 20)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    result = evolver.set('x', -5)
    with pytest.raises(InvariantException):
        result.persistent()

    # Test factory field
    class FactoryRecord(PRecord):
        x = 0

        def __factory__(self, value):
            return value * 2

    evolver = _PRecordEvolver(FactoryRecord, pmap())
    result = evolver.set('x', 5)
    assert result['x'] == 10

    # Test ignore_extra
    class IgnoreExtraRecord(PRecord):
        x = 0

    evolver = _PRecordEvolver(IgnoreExtraRecord, pmap(), _ignore_extra=True)
    result = evolver.set('x', 10)
    assert result['x'] == 10


# LLM-generated content at query #83
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord(x="hello", y="world")
    assert repr(record) == "TestRecord(x='hello', y='world')"

    record = TestRecord(x=None, y=0)
    assert repr(record) == "TestRecord(x=None, y=0)"


# LLM-generated content at query #84
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test normal creation
    r1 = TestRecord(x=10, y=20)
    assert r1.x == 10
    assert r1.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 5

    r2 = TestRecordWithInitial(x=10)
    assert r2.x == 10
    assert r2.y == 1
    assert r2.z == 5

    # Test with factory fields
    r3 = TestRecord._evolver().set('x', 10).set('y', 20).persistent()
    assert r3.x == 10
    assert r3.y == 20

    # Test with _precord_size and _precord_buckets
    r4 = TestRecord(_precord_size=2, _precord_buckets=((('x', 10), ('y', 20)),))
    assert r4.x == 10
    assert r4.y == 20

    # Test with ignore_extra
    r5 = TestRecord.create({'x': 10, 'y': 20, 'z': 30}, ignore_extra=True)
    assert r5.x == 10
    assert r5.y == 20
    assert 'z' not in r5


# LLM-generated content at query #85
#--------------------------

```python
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        x = 0
        y = 0

    # Test basic persistence
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 0

    # Test with dirty evolver
    evolver = TestRecord(x=1, y=2).evolver()
    evolver['x'] = 3
    result = evolver.persistent()
    assert result.x == 3
    assert result.y == 2

    # Test with mandatory fields missing
    class MandatoryRecord(PRecord):
        x = 0
        y = 0

        __invariant__ = {
            'x': lambda x: (x > 0, "x must be positive")
        }

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['x'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with global invariants
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

        __invariant__ = {
            'x': lambda x: (x > 0, "x must be positive"),
            'y': lambda y: (y > 0, "y must be positive")
        }

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record


# LLM-generated content at query #86
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        name = field()
        age = field()

    record = TestRecord(name="John", age=30)
    serialized = record.serialize()
    assert serialized == {"name": "John", "age": 30}

    class TestRecordWithSerializer(PRecord):
        name = field(serializer=lambda x: x.upper())
        age = field()

    record_with_serializer = TestRecordWithSerializer(name="John", age=30)
    serialized_with_serializer = record_with_serializer.serialize()
    assert serialized_with_serializer == {"name": "JOHN", "age": 30}


# LLM-generated content at query #87
#--------------------------

```python
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        x = 0
        y = 0

    # Test setting a valid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 0

    # Test setting multiple fields
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 20)
    result = evolver.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test setting an invalid field
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.set('z', 30)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z' is not among the specified fields for TestRecord" in str(e)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x >= 0, "x must be non-negative"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be non-negative" in str(e)

    # Test mandatory field missing
    class MandatoryRecord(PRecord):
        x = field(mandatory=True)
        y = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver.set('y', 20)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "MandatoryRecord.x" in str(e.missing_fields)


# LLM-generated content at query #88
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        field1 = None
        field2 = None

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert isinstance(TestRecord._precord_fields, dict)
    assert isinstance(TestRecord._precord_invariants, tuple)
    assert isinstance(TestRecord._precord_mandatory_fields, set)
    assert isinstance(TestRecord._precord_initial_values, dict)


# LLM-generated content at query #89
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")
    serialized = record.serialize()
    assert serialized == {"field1": "value1", "field2": "value2"}

    class TestRecordWithSerializer(PRecord):
        field1 = None
        field2 = None

        def __init__(self, field1, field2):
            super().__init__(field1=field1, field2=field2)

    record_with_serializer = TestRecordWithSerializer(field1="value1", field2="value2")
    serialized_with_serializer = record_with_serializer.serialize()
    assert serialized_with_serializer == {"field1": "value1", "field2": "value2"}

    class TestRecordWithCustomSerializer(PRecord):
        field1 = None
        field2 = None

        def __init__(self, field1, field2):
            super().__init__(field1=field1, field2=field2)

        def serialize(self, format=None):
            return {"custom_field1": self.field1, "custom_field2": self.field2}

    record_with_custom_serializer = TestRecordWithCustomSerializer(field1="value1", field2="value2")
    serialized_with_custom_serializer = record_with_custom_serializer.serialize()
    assert serialized_with_custom_serializer == {"custom_field1": "value1", "custom_field2": "value2"}


# LLM-generated content at query #90
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord()
    assert r.x == 1
    assert r.y == 2

    # Test creation with kwargs
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with partial kwargs
    r = TestRecord(x=10)
    assert r.x == 10
    assert r.y == 2

    # Test creation with factory_fields
    r = TestRecord(_factory_fields=[TestRecord._precord_fields['x']], x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with ignore_extra
    r = TestRecord(_ignore_extra=True, x=10, y=20, z=30)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 1
        y = lambda: 2
        z = 3

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with mandatory fields
    class TestRecordWithMandatory(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithMandatory(x=10, y=20, z=30)
    assert r.x == 10
    assert r.y == 20
    assert r.z == 30

    # Test creation with missing mandatory fields
    try:
        r = TestRecordWithMandatory(x=10, y=20)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecordWithMandatory.z' in e.missing_fields

    # Test creation with invariant failure
    class TestRecordWithInvariant(PRecord):
        x = 1
        y = 2

        __invariant__ = lambda self: (self.x > 0, "x must be positive")

    try:
        r = TestRecordWithInvariant(x=-1, y=20)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be positive" in e.invariant_errors

    # Test creation with global invariant
    class TestRecordWithGlobalInvariant(PRecord):
        x = 1
        y = 2

        __invariant__ = lambda self: (self.x + self.y > 0, "sum must be positive")

    try:
        r = TestRecordWithGlobalInvariant(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "sum must be positive" in e.invariant_errors

    # Test creation with _precord_size and _precord_buckets
    r = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 1, 'y': 2})._buckets)
    assert r.x == 1
    assert r.y == 2


# LLM-generated content at query #91
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")

    # Test default serialization
    serialized = record.serialize()
    assert serialized == {"field1": "value1", "field2": "value2"}

    # Test with custom serializer
    class CustomRecord(PRecord):
        field1 = None
        field2 = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._precord_fields["field1"].serializer = lambda x: x.upper()

    custom_record = CustomRecord(field1="value1", field2="value2")
    serialized_custom = custom_record.serialize()
    assert serialized_custom == {"field1": "VALUE1", "field2": "value2"}

    # Test with format parameter
    def format_serializer(format, value):
        if format == "upper":
            return value.upper()
        return value

    class FormatRecord(PRecord):
        field1 = None
        field2 = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._precord_fields["field1"].serializer = format_serializer

    format_record = FormatRecord(field1="value1", field2="value2")
    serialized_format = format_record.serialize(format="upper")
    assert serialized_format == {"field1": "VALUE1", "field2": "value2"}


# LLM-generated content at query #92
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()


# LLM-generated content at query #93
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    record = TestRecord(x=30, y=40)
    assert repr(record) == "TestRecord(x=30, y=40)"

    class EmptyRecord(PRecord):
        pass

    empty_record = EmptyRecord()
    assert repr(empty_record) == "EmptyRecord()"


# LLM-generated content at query #94
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: len(self.name) > 0
        name = None
        age = 0

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert 'name' in TestRecord._precord_fields
    assert 'age' in TestRecord._precord_fields
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {'age': 0}


# LLM-generated content at query #95
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 10
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == 1

    # Test with dirty evolver
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 20
    evolver['y'] = 30
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 20
    assert result.y == 30

    # Test with clean evolver (should return original)
    original = TestRecord(x=10, y=20)
    evolver = _PRecordEvolver(TestRecord, original)
    result = evolver.persistent()
    assert result is original

    # Test mandatory fields check
    class MandatoryRecord(PRecord):
        mandatory_field = 0
        optional_field = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['optional_field'] = 10
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test invariant check
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, 'x_must_be_positive'
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['x'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test global invariant check
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

    def global_inv(record):
        if record.x + record.y != 10:
            return False, 'sum_must_be_10'
        return True, None

    GlobalInvariantRecord.__invariant__ = global_inv

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 5
    evolver['y'] = 4
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #96
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = None
        y = None

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with mandatory fields missing
    class TestRecordMandatory(PRecord):
        x = None
        y = None

    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['x'] = 1
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecordMandatory.y' in e.missing_fields

    # Test with invariant violation
    class TestRecordInvariant(PRecord):
        x = None

        __invariant__ = lambda self: (self.x > 0, "x must be positive")

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['x'] = -1
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be positive" in e.invariant_errors

    # Test with no changes (not dirty)
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = None
        y = None

        def __invariant__(self):
            if self.x is not None and self.y is not None:
                return self.x + self.y > 0, "sum must be positive"
            return True, ""

    evolver = _PRecordEvolver(TestRecordGlobalInvariant, pmap())
    evolver['x'] = -1
    evolver['y'] = -2
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "sum must be positive" in str(e)


# LLM-generated content at query #97
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test normal creation
    r1 = TestRecord(x=10, y=20)
    assert r1.x == 10
    assert r1.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 3

    r2 = TestRecordWithInitial(x=10)
    assert r2.x == 10
    assert r2.y == 1
    assert r2.z == 3

    # Test with callable initial value
    class TestRecordWithCallableInitial(PRecord):
        x = 0
        y = lambda: 2

    r3 = TestRecordWithCallableInitial(x=10)
    assert r3.x == 10
    assert r3.y == 2

    # Test with factory fields
    r4 = TestRecord._factory_fields(['x', 'y'], x=10, y=20)
    assert r4.x == 10
    assert r4.y == 20

    # Test with ignore_extra
    r5 = TestRecord._ignore_extra(True, x=10, y=20, z=30)
    assert r5.x == 10
    assert r5.y == 20
    assert 'z' not in r5

    # Test with _precord_size and _precord_buckets
    r6 = TestRecord(x=10, y=20)
    r7 = TestRecord(_precord_size=r6._size, _precord_buckets=r6._buckets)
    assert r7.x == 10
    assert r7.y == 20


# LLM-generated content at query #98
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with existing PRecord
    existing = TestRecord(x=3, y=4)
    evolver = _PRecordEvolver(TestRecord, existing)
    assert evolver.persistent() is existing

    # Test with dirty evolver
    evolver['x'] = 5
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 5
    assert result.y == 4
    assert result is not existing

    # Test mandatory fields
    class MandatoryRecord(PRecord):
        mandatory_field = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    evolver['mandatory_field'] = 10
    result = evolver.persistent()
    assert result.mandatory_field == 10

    # Test invariant violations
    class InvariantRecord(PRecord):
        positive_field = 0

        def __invariant__(self):
            if self.positive_field < 0:
                return False, "POSITIVE"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['positive_field'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    evolver['positive_field'] = 1
    result = evolver.persistent()
    assert result.positive_field == 1

    # Test global invariants
    class GlobalInvariantRecord(PRecord):
        a = 0
        b = 0

        def __invariant__(self):
            if self.a + self.b != 10:
                return False, "SUM_TEN"
            return True, None

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['a'] = 5
    evolver['b'] = 5
    result = evolver.persistent()
    assert result.a == 5
    assert result.b == 5

    evolver['a'] = 3
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #99
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant check
    class TestRecordWithInvariant(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('x', -1)
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test type check
    class TestRecordWithType(PRecord):
        x = 0

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    evolver_with_type = _PRecordEvolver(TestRecordWithType, pmap())
    with pytest.raises(TypeError):
        evolver_with_type.set('x', 'not an int')


# LLM-generated content at query #100
#--------------------------

```python
def test_PRecord___new__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test with valid fields
    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

    # Test with initial values
    class TestRecordWithInitial(PRecord):
        x = 0
        y = 1
        z = 5

    record = TestRecordWithInitial(x=10)
    assert record.x == 10
    assert record.y == 1
    assert record.z == 5

    # Test with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 0
        y = lambda: 1
        z = 5

    record = TestRecordWithCallableInitial(x=10)
    assert record.x == 10
    assert record.y == 1
    assert record.z == 5

    # Test with ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 0
        y = 1

    record = TestRecordWithIgnoreExtra(x=10, y=20, z=30, _ignore_extra=True)
    assert record.x == 10
    assert record.y == 20
    assert 'z' not in record

    # Test with factory_fields
    class TestRecordWithFactoryFields(PRecord):
        x = 0
        y = 1

    record = TestRecordWithFactoryFields(x=10, y=20, _factory_fields=[TestRecordWithFactoryFields._precord_fields['x']])
    assert record.x == 10
    assert record.y == 20

    # Test with _precord_size and _precord_buckets
    class TestRecordWithBuckets(PRecord):
        x = 0
        y = 1

    record = TestRecordWithBuckets(x=10, y=20)
    new_record = TestRecordWithBuckets(_precord_size=record._size, _precord_buckets=record._buckets)
    assert new_record.x == 10
    assert new_record.y == 20


# LLM-generated content at query #101
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    record = TestRecord(field1="value1", field2="value2")
    serialized = record.serialize()
    assert isinstance(serialized, dict)
    assert serialized["field1"] == "value1"
    assert serialized["field2"] == "value2"

    class TestRecordWithSerializer(PRecord):
        field1 = None
        field2 = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._precord_fields["field1"].serializer = lambda x: x.upper()

    record_with_serializer = TestRecordWithSerializer(field1="value1", field2="value2")
    serialized_with_serializer = record_with_serializer.serialize()
    assert isinstance(serialized_with_serializer, dict)
    assert serialized_with_serializer["field1"] == "VALUE1"
    assert serialized_with_serializer["field2"] == "value2"


# LLM-generated content at query #102
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()
    assert isinstance(TestRecord._precord_fields, dict)
    assert isinstance(TestRecord._precord_invariants, dict)
    assert isinstance(TestRecord._precord_mandatory_fields, set)
    assert isinstance(TestRecord._precord_initial_values, dict)


# LLM-generated content at query #103
#--------------------------

```python
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        x = 0
        y = 0

    # Test basic persistent functionality
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 2

    # Test with dirty evolver
    evolver = TestRecord(x=1, y=2).evolver()
    evolver['x'] = 3
    result = evolver.persistent()
    assert result.x == 3
    assert result.y == 2

    # Test with clean evolver
    evolver = TestRecord(x=1, y=2).evolver()
    result = evolver.persistent()
    assert result.x == 1
    assert result.y == 2

    # Test missing mandatory fields
    class MandatoryRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['x'] = 1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'y' in str(e)

    # Test invariant failure
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['x'] = -1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "x must be positive" in str(e)

    # Test global invariant
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

        @__invariant__
        def check_sum(self):
            return self.x + self.y > 0, "sum must be positive"

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = -1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "sum must be positive" in str(e)


# LLM-generated content at query #104
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = TestRecord.evolver()

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields
    evolver.set('field2', 'value2')
    assert evolver['field1'] == 'value1'
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant failure
    class TestRecordWithInvariant(PRecord):
        field1 = None

        def __invariant__(self):
            if self.field1 == 'invalid':
                return False, 'INVALID_FIELD1'
            return True, None

    evolver_with_inv = TestRecordWithInvariant.evolver()
    evolver_with_inv.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field check
    class TestRecordWithMandatory(PRecord):
        mandatory_field = None

    evolver_mandatory = TestRecordWithMandatory.evolver()
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()


# LLM-generated content at query #105
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('field1', 'value1')
    assert evolver['field1'] == 'value1'

    # Test setting multiple fields via kwargs (though set doesn't directly support it, test single field)
    evolver.set('field2', 'value2')
    assert evolver['field2'] == 'value2'

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant failure
    class TestRecordWithInvariant(PRecord):
        field1 = None

        @invariant
        def check_field1(self, record):
            return record.field1 != 'invalid', 'INVALID_FIELD1'

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test type checking
    class TestRecordWithType(PRecord):
        field1 = field(type=int)

    evolver_with_type = _PRecordEvolver(TestRecordWithType, pmap())
    with pytest.raises(TypeError):
        evolver_with_type.set('field1', 'not_an_int')

    # Test factory field
    class TestRecordWithFactory(PRecord):
        field1 = field(factory=lambda x: x.upper())

    evolver_with_factory = _PRecordEvolver(TestRecordWithFactory, pmap())
    evolver_with_factory.set('field1', 'lowercase')
    assert evolver_with_factory['field1'] == 'LOWERCASE'

    # Test ignore_extra
    class TestRecordIgnoreExtra(PRecord):
        field1 = field(ignore_extra=True)

    evolver_ignore_extra = _PRecordEvolver(TestRecordIgnoreExtra, pmap(), _ignore_extra=True)
    evolver_ignore_extra.set('field1', {'extra': 'value', 'actual': 'data'})
    assert evolver_ignore_extra['field1'] == {'actual': 'data'}

    # Test mandatory field tracking
    class TestRecordMandatory(PRecord):
        field1 = field(mandatory=True)

    evolver_mandatory = _PRecordEvolver(TestRecordMandatory, pmap())
    with pytest.raises(InvariantException) as exc_info:
        evolver_mandatory.persistent()
    assert 'TestRecordMandatory.field1' in exc_info.value.missing_fields


# LLM-generated content at query #106
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 15
    result = evolver.persistent()
    assert result.x == 15
    assert result.y == 20
    assert isinstance(result, TestRecord)

    # Test persistence with mandatory fields missing
    class TestRecordMandatory(PRecord):
        x = 0
        y = 1
        z = 2

    record = TestRecordMandatory(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 15
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with invariant violation
    class TestRecordInvariant(PRecord):
        x = 0
        y = 1

        __invariant__ = lambda self: (self.x > 0, "x must be positive")

    record = TestRecordInvariant(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = -5
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with no changes
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test persistence with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = 0
        y = 1

        __invariant__ = lambda self: (self.x + self.y > 0, "sum must be positive")

    record = TestRecordGlobalInvariant(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = -25
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #107
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['x'] = 1
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 1
    assert result.y == 0

    # Test persistence with mandatory fields missing
    class MandatoryRecord(PRecord):
        x = 0
        y = 0
        z = None  # mandatory field

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert 'MandatoryRecord.z' in exc_info.value.missing_fields

    # Test persistence with invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['x'] = -1
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert "x must be non-negative" in exc_info.value.invariant_errors

    # Test persistence with no changes
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test persistence with global invariants
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

        def __invariant__(self):
            if self.x + self.y != 10:
                return False, "x + y must equal 10"
            return True, None

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 5
    evolver['y'] = 5
    result = evolver.persistent()
    assert result.x == 5
    assert result.y == 5

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 3
    evolver['y'] = 4
    with pytest.raises(InvariantException) as exc_info:
        evolver.persistent()
    assert "x + y must equal 10" in exc_info.value.invariant_errors


# LLM-generated content at query #108
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 15
    result = evolver.persistent()
    assert result.x == 15
    assert result.y == 20
    assert isinstance(result, TestRecord)

    # Test with mandatory fields
    class MandatoryRecord(PRecord):
        mandatory_field = 0
        optional_field = 1

    evolver = MandatoryRecord(mandatory_field=5).evolver()
    evolver['mandatory_field'] = 10
    result = evolver.persistent()
    assert result.mandatory_field == 10
    assert result.optional_field == 1

    # Test with missing mandatory field
    evolver = MandatoryRecord(mandatory_field=5).evolver()
    del evolver['mandatory_field']
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "MandatoryRecord.mandatory_field" in e.missing_fields

    # Test with invariant violation
    class InvariantRecord(PRecord):
        positive = 0

        @invariant
        def check_positive(self):
            return self.positive > 0, "positive_must_be_positive"

    evolver = InvariantRecord(positive=5).evolver()
    evolver['positive'] = -1
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "positive_must_be_positive" in e.invariant_errors

    # Test with global invariant
    class GlobalInvariantRecord(PRecord):
        a = 0
        b = 0

        @__invariant__
        def check_sum(self):
            return self.a + self.b > 0, "sum_must_be_positive"

    evolver = GlobalInvariantRecord(a=1, b=1).evolver()
    evolver['a'] = 0
    evolver['b'] = 0
    try:
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "sum_must_be_positive" in e.invariant_errors

    # Test with no changes
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record


# LLM-generated content at query #109
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = 0
        y = 0

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test setting a valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting multiple fields
    evolver.set('y', 20)
    assert evolver['x'] == 10
    assert evolver['y'] == 20

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant failure
    class TestRecordWithInvariant(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x >= 0, "x must be non-negative"

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('x', -1)
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test factory field
    class TestRecordWithFactory(PRecord):
        x = field(factory=lambda x: x * 2)

    evolver_with_factory = _PRecordEvolver(TestRecordWithFactory, pmap())
    evolver_with_factory.set('x', 5)
    assert evolver_with_factory['x'] == 10

    # Test ignore_extra
    evolver_ignore_extra = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver_ignore_extra.set('x', 10)
    assert evolver_ignore_extra['x'] == 10

    # Test mandatory field
    class TestRecordWithMandatory(PRecord):
        x = field(mandatory=True)

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()


# LLM-generated content at query #110
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 15
    result = evolver.persistent()
    assert result.x == 15
    assert result.y == 20
    assert isinstance(result, TestRecord)

    # Test persistence with mandatory fields
    class MandatoryRecord(PRecord):
        mandatory_field = 0

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['mandatory_field'] = 5
    result = evolver.persistent()
    assert result.mandatory_field == 5

    # Test persistence with missing mandatory fields
    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MandatoryRecord.mandatory_field' in e.missing_fields

    # Test persistence with invariant violation
    class InvariantRecord(PRecord):
        positive = 0

        @invariant
        def check_positive(self):
            return self.positive > 0, "positive_must_be_positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['positive'] = -1
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'positive_must_be_positive' in e.invariant_errors

    # Test persistence with no changes
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test persistence with type change
    evolver = _PRecordEvolver(TestRecord, pmap({'x': 10, 'y': 20}))
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == 20


# LLM-generated content at query #111
#--------------------------

```python
def test_PRecord___new__():
    # Test normal creation with kwargs
    class TestRecord(PRecord):
        x = 1
        y = 2

    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with initial values
    class TestRecordWithInitial(PRecord):
        x = 1
        y = 2
        z = 3

    r = TestRecordWithInitial(x=10)
    assert r.x == 10
    assert r.y == 2
    assert r.z == 3

    # Test creation with factory fields
    class TestRecordWithFactory(PRecord):
        x = 1
        y = 2

    r = TestRecordWithFactory(x=10, y=20, _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    assert r.x == 10
    assert r.y == 20

    # Test creation with ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 1
        y = 2

    r = TestRecordWithIgnoreExtra(x=10, y=20, z=30, _ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with _precord_size and _precord_buckets
    class TestRecordWithBuckets(PRecord):
        x = 1
        y = 2

    r = TestRecordWithBuckets(x=10, y=20)
    r2 = TestRecordWithBuckets(_precord_size=r._size, _precord_buckets=r._buckets)
    assert r2.x == 10
    assert r2.y == 20

    # Test creation with callable initial values
    class TestRecordWithCallableInitial(PRecord):
        x = 1
        y = lambda: 2

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 2


