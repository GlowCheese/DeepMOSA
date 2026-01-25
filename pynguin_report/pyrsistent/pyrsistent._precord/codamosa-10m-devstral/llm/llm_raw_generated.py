####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    result = evolver.set('x', 10).set('y', 20)
    assert result['x'] == 10
    assert result['y'] == 20

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, 'x_must_be_positive'
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    result = evolver.set('x', -1)
    with pytest.raises(InvariantException):
        result.persistent()

    # Test mandatory field check
    class MandatoryRecord(PRecord):
        x = 0
        y = 0

        def __invariant__(self):
            if 'x' not in self:
                return False, 'x_is_mandatory'
            return True, None

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    result = evolver.set('y', 20)
    with pytest.raises(InvariantException):
        result.persistent()

    # Test type check
    class TypedRecord(PRecord):
        x = 0

        def __invariant__(self):
            if not isinstance(self.x, int):
                return False, 'x_must_be_int'
            return True, None

    evolver = _PRecordEvolver(TypedRecord, pmap())
    result = evolver.set('x', 'not_an_int')
    with pytest.raises(InvariantException):
        result.persistent()


# LLM-generated content at query #2
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

    # Test persistence with mandatory fields
    class MandatoryRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['x'] = 10
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with invariant checks
    class InvariantRecord(PRecord):
        x = 0

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['x'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with global invariants
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

        @invariant
        def check_sum(self):
            return self.x + self.y > 0, "sum must be positive"

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 1
    evolver['y'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with no changes
    class NoChangeRecord(PRecord):
        x = 0

    record = NoChangeRecord(x=10)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test persistence with type checking
    class TypedRecord(PRecord):
        x = field(type=int)

    evolver = _PRecordEvolver(TypedRecord, pmap())
    evolver['x'] = "not an int"
    with pytest.raises(TypeError):
        evolver.persistent()


# LLM-generated content at query #3
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
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord()
    assert record.x == 1
    assert record.y == 2
    assert len(record) == 2

    # Test creation with custom initial values
    record = TestRecord(x=10, y=20)
    assert record.x == 10
    assert record.y == 20

    # Test creation with partial custom values
    record = TestRecord(x=10)
    assert record.x == 10
    assert record.y == 2

    # Test creation with factory_fields
    record = TestRecord(_factory_fields=[TestRecord._precord_fields['x']], x=100)
    assert record.x == 100
    assert record.y == 2

    # Test creation with ignore_extra
    record = TestRecord(_ignore_extra=True, x=10, z=30)
    assert record.x == 10
    assert record.y == 2
    assert 'z' not in record

    # Test creation with internal parameters
    record = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 1, 'y': 2}))
    assert record.x == 1
    assert record.y == 2

    # Test with callable initial values
    class CallableRecord(PRecord):
        x = lambda: 1
        y = 2

    record = CallableRecord()
    assert record.x == 1
    assert record.y == 2


# LLM-generated content at query #4
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
        def check_field1(self, record):
            return record.field1 != 'invalid', 'INVALID_FIELD1'

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field missing
    class TestRecordWithMandatory(PRecord):
        mandatory_field = field(mandatory=True)

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()


# LLM-generated content at query #5
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
    class InvariantRecord(PRecord):
        field1 = None

        @invariant
        def check_field1(self, record):
            if record.field1 == 'invalid':
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

    # Test type checking
    class TypedRecord(PRecord):
        int_field = field(type=int)

    evolver = _PRecordEvolver(TypedRecord, pmap())
    with pytest.raises(TypeError):
        evolver.set('int_field', 'not_an_int')


# LLM-generated content at query #6
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
    class ValidatedRecord(PRecord):
        x = 0

        @invariant
        def validate_x(self):
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
        x = field(factory=lambda x: x * 2)

    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('x', 5)
    assert evolver['x'] == 10

    # Test ignore_extra
    evolver = _PRecordEvolver(TestRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 10)
    assert evolver['x'] == 10


# LLM-generated content at query #7
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

    # Test persistence with mandatory fields missing
    class TestRecordMandatory(PRecord):
        x = 1
        y = 2
        z = 3

    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['x'] = 10
    evolver['y'] = 20
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecordMandatory.z' in e.missing_fields

    # Test persistence with invariant violation
    class TestRecordInvariant(PRecord):
        x = 1

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, None

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['x'] = -1
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be non-negative" in e.invariant_errors

    # Test persistence with no changes
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test persistence with type check failure
    class TestRecordType(PRecord):
        x = 1

    evolver = _PRecordEvolver(TestRecordType, pmap())
    try:
        evolver.set('x', 'not an int')
        evolver.persistent()
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test persistence with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = 1
        y = 2

    def global_inv(record):
        if record.x + record.y != 3:
            raise ValueError("x + y must equal 3")

    TestRecordGlobalInvariant._precord_invariants = [global_inv]

    evolver = _PRecordEvolver(TestRecordGlobalInvariant, pmap())
    evolver['x'] = 1
    evolver['y'] = 1
    try:
        evolver.persistent()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "x + y must equal 3"


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

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set('invalid_field', 'value')

    # Test invariant failure
    class TestRecordWithInvariant(PRecord):
        field1 = None

        @invariant
        def check_field1(self):
            return self.field1 != 'forbidden', 'FIELD1_FORBIDDEN'

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('field1', 'forbidden')
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field check
    class TestRecordWithMandatory(PRecord):
        mandatory_field = None

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()


# LLM-generated content at query #9
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
        mandatory_field = PFIELD_NO_INITIAL
        optional_field = 1

    assert 'mandatory_field' in MandatoryRecord._precord_mandatory_fields
    assert 'optional_field' not in MandatoryRecord._precord_mandatory_fields

    # Test initial values
    class InitialRecord(PRecord):
        initialized_field = 42
        no_initial_field = PFIELD_NO_INITIAL

    assert 'initialized_field' in InitialRecord._precord_initial_values
    assert 'no_initial_field' not in InitialRecord._precord_initial_values
    assert InitialRecord._precord_initial_values['initialized_field'] == 42


# LLM-generated content at query #10
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic creation
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
    class MandatoryRecord(PRecord):
        __invariant__ = lambda self: True
        mandatory_field = None

    assert 'mandatory_field' in MandatoryRecord._precord_mandatory_fields

    # Test initial values
    class InitialRecord(PRecord):
        __invariant__ = lambda self: True
        initial_field = 42

    assert 'initial_field' in InitialRecord._precord_initial_values
    assert InitialRecord._precord_initial_values['initial_field'] == 42

    # Test invariants
    class InvariantRecord(PRecord):
        __invariant__ = lambda self: self.field > 0
        field = None

    assert len(InvariantRecord._precord_invariants) == 1

    # Test inheritance
    class BaseRecord(PRecord):
        __invariant__ = lambda self: True
        base_field = None

    class DerivedRecord(BaseRecord):
        derived_field = None

    assert 'base_field' in DerivedRecord._precord_fields
    assert 'derived_field' in DerivedRecord._precord_fields


# LLM-generated content at query #11
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

    # Test initial values tracking
    class InitialRecord(PRecord):
        with_initial = PField(initial=42)
        no_initial = PField()

    assert 'with_initial' in InitialRecord._precord_initial_values
    assert InitialRecord._precord_initial_values['with_initial'] == 42
    assert 'no_initial' not in InitialRecord._precord_initial_values


# LLM-generated content at query #12
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
                return False, "x must be non-negative"
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

    # Test ignore_extra flag
    class TestRecordIgnoreExtra(PRecord):
        x = 0

    evolver_ignore = _PRecordEvolver(TestRecordIgnoreExtra, pmap(), _ignore_extra=True)
    evolver_ignore.set('x', 10)
    evolver_ignore.set('extra_field', 20)  # Should not raise an error
    result = evolver_ignore.persistent()
    assert result.x == 10
    assert 'extra_field' not in result


# LLM-generated content at query #13
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


# LLM-generated content at query #14
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
    assert result['x'] == 1
    assert result['y'] == 2

    # Test with mandatory fields missing
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

        __invariant__ = lambda self: self.x + self.y == 10

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
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
    assert isinstance(result, TestRecord)
    assert result['x'] == 3
    assert result['y'] == 2


# LLM-generated content at query #15
#--------------------------

```python
def test_PRecord___new__():
    # Test basic creation with no fields
    class EmptyRecord(PRecord):
        pass

    empty = EmptyRecord()
    assert isinstance(empty, EmptyRecord)
    assert len(empty) == 0

    # Test creation with fields
    class Person(PRecord):
        name = None
        age = None

    person = Person(name="Alice", age=30)
    assert isinstance(person, Person)
    assert person.name == "Alice"
    assert person.age == 30
    assert len(person) == 2

    # Test creation with default values
    class PersonWithDefaults(PRecord):
        name = None
        age = 0

    person_default = PersonWithDefaults(name="Bob")
    assert person_default.name == "Bob"
    assert person_default.age == 0

    # Test creation with callable default values
    class PersonWithCallableDefaults(PRecord):
        name = None
        age = lambda: 25

    person_callable = PersonWithCallableDefaults(name="Charlie")
    assert person_callable.name == "Charlie"
    assert person_callable.age == 25

    # Test creation with mandatory fields
    class PersonWithMandatory(PRecord):
        name = None
        age = None

    try:
        person_missing = PersonWithMandatory(name="Dave")
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "PersonWithMandatory.age" in e.missing_fields

    # Test creation with factory fields
    class PersonWithFactory(PRecord):
        name = None
        age = None

    factory_fields = [PersonWithFactory._precord_fields['name']]
    person_factory = PersonWithFactory(_factory_fields=factory_fields, name="Eve", age=35)
    assert person_factory.name == "Eve"
    assert person_factory.age == 35

    # Test creation with ignore_extra
    class PersonWithIgnoreExtra(PRecord):
        name = None
        age = None

    person_ignore = PersonWithIgnoreExtra(_ignore_extra=True, name="Frank", age=40, extra="ignored")
    assert person_ignore.name == "Frank"
    assert person_ignore.age == 40
    assert "extra" not in person_ignore

    # Test creation from existing PRecord
    person_copy = PersonWithIgnoreExtra.create(person_ignore)
    assert person_copy.name == "Frank"
    assert person_copy.age == 40


# LLM-generated content at query #16
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


# LLM-generated content at query #17
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        field1 = PFIELD_NO_INITIAL
        field2 = PFIELD_NO_INITIAL

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
        field1 = PFIELD_NO_INITIAL

        @invariant
        def check_field1(self, field1):
            if field1 == 'invalid':
                return False, 'field1_cannot_be_invalid'
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('field1', 'invalid')
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test type checking
    class TypedRecord(PRecord):
        field1 = PFIELD_NO_INITIAL

        @check_type
        def check_field1_type(self, field1):
            if not isinstance(field1, int):
                raise TypeError('field1 must be an int')

    evolver = _PRecordEvolver(TypedRecord, pmap())
    evolver.set('field1', 'not_an_int')
    with pytest.raises(TypeError):
        evolver.persistent()

    # Test mandatory field
    class MandatoryRecord(PRecord):
        mandatory_field = PFIELD_NO_INITIAL

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    evolver.set('mandatory_field', 'value')
    result = evolver.persistent()
    assert result['mandatory_field'] == 'value'


# LLM-generated content at query #18
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 1
        y = 2

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == 2

    # Test with dirty evolver
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 20)
    evolver.set('y', 30)
    result = evolver.persistent()
    assert result.x == 20
    assert result.y == 30

    # Test with mandatory fields
    class MandatoryRecord(PRecord):
        mandatory_field = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violation
    class InvariantRecord(PRecord):
        positive = 1

        @invariant
        def check_positive(self, positive):
            return positive > 0, "must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('positive', -1)
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
    evolver.set('a', -1)
    evolver.set('b', -2)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record


# LLM-generated content at query #19
#--------------------------

```python
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda self: True
        field1 = None
        field2 = 1

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert hasattr(TestRecord, '__slots__')
    assert TestRecord.__slots__ == ()
    assert 'field1' in TestRecord._precord_fields
    assert 'field2' in TestRecord._precord_fields
    assert TestRecord._precord_initial_values == {'field2': 1}
    assert TestRecord._precord_mandatory_fields == set()


# LLM-generated content at query #20
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=5, y=10)
    assert repr(record) == "TestRecord(x=5, y=10)"

    empty_record = TestRecord()
    assert repr(empty_record) == "TestRecord(x=0, y=1)"

    class AnotherRecord(PRecord):
        name = "default"
        value = None

    another_record = AnotherRecord(name="test", value=42)
    assert repr(another_record) == "AnotherRecord(name='test', value=42)"


# LLM-generated content at query #21
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    assert repr(record) == "TestRecord(x=10, y=20)"

    class EmptyRecord(PRecord):
        pass

    empty_record = EmptyRecord()
    assert repr(empty_record) == "EmptyRecord()"

    class SingleFieldRecord(PRecord):
        field = None

    single_record = SingleFieldRecord(field="value")
    assert repr(single_record) == "SingleFieldRecord(field='value')"


# LLM-generated content at query #22
#--------------------------

```python
def test_PRecord___new__():
    # Test normal creation with kwargs
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
        y = lambda: 1
        z = 5

    r = TestRecordWithCallableInitial(x=10)
    assert r.x == 10
    assert r.y == 1
    assert r.z == 5

    # Test creation with _factory_fields
    class TestRecordWithFactoryFields(PRecord):
        x = 0
        y = 1

    r = TestRecordWithFactoryFields._PRecordEvolver(TestRecordWithFactoryFields, pmap(), _factory_fields=[TestRecordWithFactoryFields._precord_fields['x']])
    r['x'] = 10
    r['y'] = 20
    result = r.persistent()
    assert result.x == 10
    assert result.y == 20

    # Test creation with _ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 0
        y = 1

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
        x = 0
        y = 1

    r = TestRecordWithBuckets(_precord_size=2, _precord_buckets=pmap(x=10, y=20)._buckets)
    assert r.x == 10
    assert r.y == 20


# LLM-generated content at query #23
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

        class __invariant__:
            z = lambda self: (True, None)

    assert 'z' in TestRecordMandatory._precord_mandatory_fields

    # Test with initial values
    class TestRecordInitial(PRecord):
        x = 1
        y = 2
        z = 3

        class __initial__:
            z = 10

    assert TestRecordInitial._precord_initial_values == {'z': 10}

    # Test with callable initial value
    class TestRecordCallableInitial(PRecord):
        x = 1
        y = 2
        z = 3

        class __initial__:
            z = lambda: 10

    assert TestRecordCallableInitial._precord_initial_values['z']() == 10


# LLM-generated content at query #24
#--------------------------

```python
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test basic persistence
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == 1

    # Test persistence with dirty state
    evolver = _PRecordEvolver(TestRecord, pmap(x=5))
    evolver.set('x', 10)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10

    # Test persistence with mandatory fields missing
    class MandatoryRecord(PRecord):
        x = 0
        y = 1
        z = field(mandatory=True)

    evolver = _PRecordEvolver(MandatoryRecord, pmap(x=5, y=10))
    with pytest.raises(InvariantException) as excinfo:
        evolver.persistent()
    assert 'z' in str(excinfo.value)

    # Test persistence with invariant violation
    def positive_invariant(value):
        return value > 0, "must be positive"

    class InvariantRecord(PRecord):
        x = field(invariant=positive_invariant)

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException) as excinfo:
        evolver.persistent()
    assert "must be positive" in str(excinfo.value)

    # Test persistence with global invariant
    def sum_invariant(record):
        return record.x + record.y > 0, "sum must be positive"

    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0
        __invariant__ = sum_invariant

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap(x=-1, y=-1))
    with pytest.raises(InvariantException) as excinfo:
        evolver.persistent()
    assert "sum must be positive" in str(excinfo.value)


# LLM-generated content at query #25
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


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    record2 = TestRecord(x=30)
    assert record2.x == 30
    assert record2.y == 2  # Default value

    # Test with _factory_fields
    record3 = TestRecord(x=40, y=50, _factory_fields=[TestRecord._precord_fields['x']])
    assert record3.x == 40
    assert record3.y == 50

    # Test with _ignore_extra
    record4 = TestRecord(x=60, y=70, z=80, _ignore_extra=True)
    assert record4.x == 60
    assert record4.y == 70
    assert 'z' not in record4

    # Test with both _factory_fields and _ignore_extra
    record5 = TestRecord(x=90, y=100, z=110, _factory_fields=[TestRecord._precord_fields['x']], _ignore_extra=True)
    assert record5.x == 90
    assert record5.y == 100
    assert 'z' not in record5

    # Test with _precord_size and _precord_buckets (internal creation)
    internal_record = TestRecord(_precord_size=2, _precord_buckets=((('x', 100), ('y', 200)),))
    assert internal_record.x == 100
    assert internal_record.y == 200


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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

    # Test persistence with mandatory fields
    class TestRecordMandatory(PRecord):
        x = 0
        y = 0

        class __invariant__:
            x = lambda x: (x > 0, "x must be positive")

    evolver = TestRecordMandatory().evolver()
    evolver['x'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with missing mandatory fields
    class TestRecordMandatoryFields(PRecord):
        x = 0
        y = 0

        class __invariant__:
            x = lambda x: (x > 0, "x must be positive")

    evolver = TestRecordMandatoryFields().evolver()
    evolver['y'] = 1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with no changes
    record = TestRecord(x=1, y=2)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test persistence with global invariants
    class TestRecordGlobalInvariant(PRecord):
        x = 0
        y = 0

        def __invariant__(self):
            if self.x + self.y != 10:
                return False, "x + y must be 10"
            return True, ""

    evolver = TestRecordGlobalInvariant().evolver()
    evolver['x'] = 5
    evolver['y'] = 4
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #4
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistence
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    evolver['x'] = 30
    result = evolver.persistent()
    assert result.x == 30
    assert result.y == 20
    assert isinstance(result, TestRecord)

    # Test with mandatory fields
    class MandatoryRecord(PRecord):
        a = 0
        b = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['a'] = 10
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with invariant violation
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            return self.x > 0, "x must be positive"

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver['x'] = -1
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test with no changes (not dirty)
    record = TestRecord(x=5, y=10)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test with global invariants
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 0

        def __invariant__(self):
            return self.x + self.y == 10, "sum must be 10"

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 5
    evolver['y'] = 5
    result = evolver.persistent()
    assert result.x == 5
    assert result.y == 5

    evolver['x'] = 3
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #5
#--------------------------

```python
def test__PRecordEvolver_set():
    # Setup
    class TestRecord(PRecord):
        x = field()
        y = field()

    evolver = _PRecordEvolver(TestRecord, pmap())

    # Test valid field
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test invalid field
    with pytest.raises(AttributeError):
        evolver.set('z', 20)

    # Test invariant failure
    class InvariantRecord(PRecord):
        x = field(invariant=lambda x: (x > 0, 'positive'))

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -5)
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test type check
    class TypedRecord(PRecord):
        x = field(type=int)

    evolver = _PRecordEvolver(TypedRecord, pmap())
    with pytest.raises(TypeError):
        evolver.set('x', 'not an int')

    # Test factory field
    class FactoryRecord(PRecord):
        x = field(factory=lambda v: v.upper())

    evolver = _PRecordEvolver(FactoryRecord, pmap())
    evolver.set('x', 'hello')
    assert evolver['x'] == 'HELLO'

    # Test ignore_extra
    evolver = _PRecordEvolver(FactoryRecord, pmap(), _ignore_extra=True)
    evolver.set('x', 'world')
    assert evolver['x'] == 'WORLD'


# LLM-generated content at query #6
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

    # Test persistence with no changes
    record = TestRecord(x=1, y=2)
    evolver = _PRecordEvolver(TestRecord, record)
    result = evolver.persistent()
    assert result is record

    # Test persistence with mandatory fields missing
    class TestRecordMandatory(PRecord):
        x = 0
        y = 0
        z = 0

    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['x'] = 1
    evolver['y'] = 2
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with invariant violation
    class TestRecordInvariant(PRecord):
        x = 0
        y = 0

        __invariant__ = lambda self: (self.x + self.y) > 0

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['x'] = -1
    evolver['y'] = -2
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = 0
        y = 0

    def global_invariant(record):
        if record.x + record.y < 0:
            raise InvariantException(('negative_sum',), (), 'Global invariant failed')

    TestRecordGlobalInvariant._precord_invariants = (global_invariant,)

    evolver = _PRecordEvolver(TestRecordGlobalInvariant, pmap())
    evolver['x'] = -1
    evolver['y'] = -2
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #7
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

    # Test setting an invalid field
    with pytest.raises(AttributeError):
        evolver.set('z', 30)

    # Test invariant failure
    class InvariantRecord(PRecord):
        x = 0

        def __invariant__(self):
            if self.x < 0:
                return False, "x must be non-negative"
            return True, ""

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('x', -1)
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #8
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('x', 10)
    evolver.set('y', 20)
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == 20

    # Test with mandatory fields missing
    class MandatoryRecord(PRecord):
        mandatory_field = PFIELD_NO_INITIAL

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MandatoryRecord.mandatory_field' in e.missing_fields

    # Test with invariant violation
    class InvariantRecord(PRecord):
        positive = 0

        def __invariant__(self):
            if self.positive < 0:
                return False, "positive_must_be_positive"
            return True, None

    evolver = _PRecordEvolver(InvariantRecord, pmap())
    evolver.set('positive', -1)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "positive_must_be_positive" in e.invariant_errors

    # Test with no changes (not dirty)
    record = TestRecord(x=5, y=10)
    evolver = _PRecordEvolver(TestRecord, record)
    result = evolver.persistent()
    assert result is record

    # Test with global invariants
    class GlobalInvariantRecord(PRecord):
        a = 0
        b = 0

        def __invariant__(self):
            if self.a + self.b != 10:
                return False, "sum_must_be_10"
            return True, None

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver.set('a', 3)
    evolver.set('b', 7)
    result = evolver.persistent()
    assert result.a == 3
    assert result.b == 7

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver.set('a', 3)
    evolver.set('b', 8)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "sum_must_be_10" in e.invariant_errors


# LLM-generated content at query #9
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
    assert isinstance(TestRecord._precord_mandatory_fields, set)
    assert isinstance(TestRecord._precord_initial_values, dict)


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
    evolver.set("field1", "value1")
    assert evolver["field1"] == "value1"

    # Test setting multiple fields
    evolver.set("field2", "value2")
    assert evolver["field1"] == "value1"
    assert evolver["field2"] == "value2"

    # Test setting an invalid field (should raise AttributeError)
    with pytest.raises(AttributeError):
        evolver.set("invalid_field", "value")

    # Test invariant failure
    class TestRecordWithInvariant(PRecord):
        field1 = None

        @invariant
        def check_field1(self, record):
            return record.field1 != "invalid", "INVALID_FIELD1"

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set("field1", "invalid")
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test mandatory field missing
    class TestRecordWithMandatory(PRecord):
        mandatory_field = None

    evolver_mandatory = _PRecordEvolver(TestRecordWithMandatory, pmap())
    with pytest.raises(InvariantException):
        evolver_mandatory.persistent()


# LLM-generated content at query #11
#--------------------------

```python
def test_PRecord_serialize():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field(serializer=lambda x: x.upper())

    record = TestRecord(field1="value1", field2="value2")

    assert record.serialize() == {"field1": "value1", "field2": "VALUE2"}
    assert record.serialize(format="custom") == {"field1": "value1", "field2": "VALUE2"}


# LLM-generated content at query #12
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

    # Test persistence with no changes
    record = TestRecord(x=5, y=6)
    evolver = record.evolver()
    result = evolver.persistent()
    assert result is record

    # Test persistence with mandatory fields missing
    class MandatoryRecord(PRecord):
        x = 0
        y = 1

    evolver = _PRecordEvolver(MandatoryRecord, pmap())
    evolver['x'] = 10
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test persistence with invariant violation
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

    # Test persistence with global invariant
    class GlobalInvariantRecord(PRecord):
        x = 0
        y = 1

        def __invariant__(self):
            if self.x + self.y != 1:
                return False, "x + y must be 1"
            return True, None

    evolver = _PRecordEvolver(GlobalInvariantRecord, pmap())
    evolver['x'] = 2
    evolver['y'] = 3
    with pytest.raises(InvariantException):
        evolver.persistent()


# LLM-generated content at query #13
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


# LLM-generated content at query #14
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

    # Test creation with custom values
    r = TestRecord(x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with partial custom values
    r = TestRecord(x=10)
    assert r.x == 10
    assert r.y == 2

    # Test creation with callable initial values
    class TestRecordCallable(PRecord):
        x = lambda: 1
        y = 2

    r = TestRecordCallable()
    assert r.x == 1
    assert r.y == 2

    # Test creation with factory fields
    class TestRecordFactory(PRecord):
        x = 1
        y = 2

    r = TestRecordFactory(_factory_fields=[TestRecordFactory._precord_fields['x']], x=10, y=20)
    assert r.x == 10
    assert r.y == 20

    # Test creation with ignore_extra
    class TestRecordIgnoreExtra(PRecord):
        x = 1
        y = 2

    r = TestRecordIgnoreExtra(_ignore_extra=True, x=10, y=20, z=30)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test creation with invalid field
    try:
        TestRecord(z=30)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creation with internal parameters
    r = TestRecord(_precord_size=2, _precord_buckets=pmap({'x': 1, 'y': 2}))
    assert r.x == 1
    assert r.y == 2


# LLM-generated content at query #16
#--------------------------

```python
def test_PRecord___new__():
    # Test normal creation with fields
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

    r = TestRecordWithFactory.create({'x': 10, 'y': 20}, _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    assert r.x == 10
    assert r.y == 20

    # Test with ignore_extra
    class TestRecordWithIgnoreExtra(PRecord):
        x = 0
        y = 1

    r = TestRecordWithIgnoreExtra.create({'x': 10, 'y': 20, 'z': 30}, ignore_extra=True)
    assert r.x == 10
    assert r.y == 20
    assert 'z' not in r

    # Test with _precord_size and _precord_buckets
    class TestRecordWithBuckets(PRecord):
        x = 0
        y = 1

    r = TestRecordWithBuckets(x=10, y=20)
    r2 = TestRecordWithBuckets(_precord_size=r._size, _precord_buckets=r._buckets)
    assert r2.x == 10
    assert r2.y == 20


# LLM-generated content at query #17
#--------------------------

```python
def test_PRecord___repr__():
    class TestRecord(PRecord):
        x = 0
        y = 1

    record = TestRecord(x=5, y=10)
    assert repr(record) == "TestRecord(x=5, y=10)"

    record = TestRecord(x=0, y=0)
    assert repr(record) == "TestRecord(x=0, y=0)"

    class EmptyRecord(PRecord):
        pass

    empty_record = EmptyRecord()
    assert repr(empty_record) == "EmptyRecord()"


# LLM-generated content at query #18
#--------------------------

```python
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        x = 0
        y = 1

    # Test basic persistent with no changes
    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == 20
    assert not evolver.is_dirty()

    # Test persistent with changes
    evolver = record.evolver()
    evolver['x'] = 30
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 30
    assert result.y == 20
    assert evolver.is_dirty()

    # Test persistent with mandatory field missing
    class TestRecordMandatory(PRecord):
        x = 0
        y = 1
        z = 2

        __invariant__ = lambda self: ('z' in self, "z is mandatory")

    record = TestRecordMandatory(x=10, y=20)
    evolver = record.evolver()
    evolver.pop('z')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "z is mandatory" in e.invariant_errors

    # Test persistent with field invariant failure
    class TestRecordInvariant(PRecord):
        x = 0

        __invariant__ = lambda self: (self.x > 0, "x must be positive")

    record = TestRecordInvariant(x=10)
    evolver = record.evolver()
    evolver['x'] = -5
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "x must be positive" in e.invariant_errors

    # Test persistent with global invariant
    class TestRecordGlobalInvariant(PRecord):
        x = 0
        y = 0

        __invariant__ = lambda self: (self.x + self.y == 10, "sum must be 10")

    record = TestRecordGlobalInvariant(x=5, y=5)
    evolver = record.evolver()
    evolver['x'] = 6
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "sum must be 10" in e.invariant_errors


# LLM-generated content at query #19
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


# LLM-generated content at query #20
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
    assert TestRecord._precord_mandatory_fields == set()
    assert TestRecord._precord_initial_values == {}


# LLM-generated content at query #22
#--------------------------

```python
def test__PRecordMeta___new__():
    # Test basic metaclass creation
    class TestRecord(PRecord):
        pass

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord.__slots__ == ()

    # Test field inheritance
    class ParentRecord(PRecord):
        x = 1
        y = 2

    class ChildRecord(ParentRecord):
        z = 3

    assert 'x' in ChildRecord._precord_fields
    assert 'y' in ChildRecord._precord_fields
    assert 'z' in ChildRecord._precord_fields

    # Test mandatory fields tracking
    from pyrsistent._field_common import PField
    class MandatoryRecord(PRecord):
        a = PField(mandatory=True)
        b = PField(mandatory=False)
        c = PField()

    assert MandatoryRecord._precord_mandatory_fields == {'a'}

    # Test initial values tracking
    class InitialRecord(PRecord):
        x = PField(initial=1)
        y = PField(initial=lambda: 2)
        z = PField()

    assert InitialRecord._precord_initial_values == {'x': 1, 'y': lambda: 2}

    # Test invariant storage
    def test_invariant(obj):
        return True, ""

    class InvariantRecord(PRecord):
        __invariant__ = test_invariant
        x = PField()

    assert InvariantRecord._precord_invariants == [test_invariant]


# LLM-generated content at query #23
#--------------------------

```python
def test_PRecord_serialize():
    # Define a simple PRecord class for testing
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    # Create an instance of TestRecord
    record = TestRecord(field1="value1", field2="value2")

    # Test serialize with no format
    serialized = record.serialize()
    assert isinstance(serialized, dict)
    assert serialized == {"field1": "value1", "field2": "value2"}

    # Test serialize with a custom serializer (if applicable)
    # Assuming a serializer function that converts values to uppercase
    class UppercaseRecord(PRecord):
        field1 = None
        field2 = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Override the fields to include a serializer
            self._precord_fields['field1'].serializer = lambda x: x.upper() if isinstance(x, str) else x
            self._precord_fields['field2'].serializer = lambda x: x.upper() if isinstance(x, str) else x

    uppercase_record = UppercaseRecord(field1="value1", field2="value2")
    serialized_uppercase = uppercase_record.serialize()
    assert serialized_uppercase == {"field1": "VALUE1", "field2": "VALUE2"}

    # Test serialize with a format parameter (if applicable)
    # Assuming a format that converts values to a specific string format
    class FormattedRecord(PRecord):
        field1 = None
        field2 = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Override the fields to include a serializer with format
            self._precord_fields['field1'].serializer = lambda fmt, x: f"{fmt}:{x}" if fmt else x
            self._precord_fields['field2'].serializer = lambda fmt, x: f"{fmt}:{x}" if fmt else x

    formatted_record = FormattedRecord(field1="value1", field2="value2")
    serialized_formatted = formatted_record.serialize(format="FMT")
    assert serialized_formatted == {"field1": "FMT:value1", "field2": "FMT:value2"}


# LLM-generated content at query #24
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


# LLM-generated content at query #25
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


# LLM-generated content at query #26
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test basic persistent functionality
    class TestRecord(PRecord):
        field1 = None
        field2 = None

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver['field1'] = 'value1'
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result['field1'] == 'value1'
    assert 'field2' not in result

    # Test with mandatory fields
    class TestRecordMandatory(PRecord):
        mandatory_field = None

    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    evolver['mandatory_field'] = 'value'
    result = evolver.persistent()
    assert isinstance(result, TestRecordMandatory)
    assert result['mandatory_field'] == 'value'

    # Test missing mandatory field raises InvariantException
    evolver = _PRecordEvolver(TestRecordMandatory, pmap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'mandatory_field' in str(e)

    # Test invariant violation
    class TestRecordInvariant(PRecord):
        field = None

        @invariant
        def check_field(self):
            return self.field != 'invalid', 'INVALID_VALUE'

    evolver = _PRecordEvolver(TestRecordInvariant, pmap())
    evolver['field'] = 'invalid'
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'INVALID_VALUE' in e.invariant_errors

    # Test with existing PRecord instance
    original = TestRecord(field1='original')
    evolver = _PRecordEvolver(TestRecord, original)
    evolver['field1'] = 'modified'
    result = evolver.persistent()
    assert result['field1'] == 'modified'
    assert result is not original

    # Test with no changes (not dirty)
    original = TestRecord(field1='value')
    evolver = _PRecordEvolver(TestRecord, original)
    result = evolver.persistent()
    assert result is original


# LLM-generated content at query #27
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

        @invariant
        def check_x(self):
            return self.x >= 0, "x must be non-negative"

    evolver_with_inv = _PRecordEvolver(TestRecordWithInvariant, pmap())
    evolver_with_inv.set('x', -1)
    with pytest.raises(InvariantException):
        evolver_with_inv.persistent()

    # Test type checking
    class TestRecordWithType(PRecord):
        x = field(type=int)

    evolver_with_type = _PRecordEvolver(TestRecordWithType, pmap())
    with pytest.raises(TypeError):
        evolver_with_type.set('x', 'not an int')

    # Test factory fields
    class TestRecordWithFactory(PRecord):
        x = field(factory=lambda v: v * 2)

    evolver_with_factory = _PRecordEvolver(TestRecordWithFactory, pmap(), _factory_fields=[TestRecordWithFactory._precord_fields['x']])
    evolver_with_factory.set('x', 5)
    assert evolver_with_factory['x'] == 10

    # Test ignore_extra
    class TestRecordIgnoreExtra(PRecord):
        x = field(ignore_extra=True)

    evolver_ignore_extra = _PRecordEvolver(TestRecordIgnoreExtra, pmap(), _ignore_extra=True)
    evolver_ignore_extra.set('x', {'a': 1, 'b': 2})
    assert evolver_ignore_extra['x'] == {'a': 1, 'b': 2}


# LLM-generated content at query #28
#--------------------------

```python
def test__PRecordEvolver_persistent():
    # Test 1: Basic persistent call with no changes
    class TestRecord(PRecord):
        x = 1
        y = 2

    record = TestRecord(x=10, y=20)
    evolver = record.evolver()
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 10
    assert result.y == 20

    # Test 2: Persistent call after modification
    evolver = record.evolver()
    evolver['x'] = 100
    result = evolver.persistent()
    assert isinstance(result, TestRecord)
    assert result.x == 100
    assert result.y == 20

    # Test 3: Persistent call with missing mandatory field
    class MandatoryRecord(PRecord):
        x = 1
        y = 2

    evolver = MandatoryRecord(x=10).evolver()
    evolver.__dict__['_missing_fields'] = ['MandatoryRecord.y']
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test 4: Persistent call with invariant violation
    class InvariantRecord(PRecord):
        x = 1

    evolver = InvariantRecord(x=10).evolver()
    evolver.__dict__['_invariant_error_codes'] = ['INVALID_X']
    with pytest.raises(InvariantException):
        evolver.persistent()

    # Test 5: Persistent call with global invariant check
    class GlobalInvariantRecord(PRecord):
        x = 1
        y = 2

        __invariant__ = lambda self: self.x != self.y

    evolver = GlobalInvariantRecord(x=10, y=10).evolver()
    with pytest.raises(InvariantException):
        evolver.persistent()


