####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestPRecord(PRecord):
        field1 = field(type=str)
        field2 = field(type=int)

    evolver = TestPRecord().evolver()
    evolver.set('field1', 'value1')
    evolver.set('field2', 2)
    precord = evolver.persistent()

    assert precord['field1'] == 'value1'
    assert precord['field2'] == 2

    try:
        evolver.set('field3', 'value3')
    except AttributeError as e:
        assert str(e) == "'field3' is not among the specified fields for TestPRecord"


# LLM-generated content at query #2
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        field1 = field(mandatory=True)
        field2 = field()

    evolver = TestRecord().evolver()
    evolver.set('field1', 'value1')
    evolver.set('field2', 'value2')
    record = evolver.persistent()

    assert isinstance(record, TestRecord)
    assert record['field1'] == 'value1'
    assert record['field2'] == 'value2'

    evolver = TestRecord().evolver()
    try:
        evolver.persistent()
        assert False, "InvariantException should have been raised"
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

    evolver = TestRecord().evolver()
    evolver.set('field1', 'value1')
    record = evolver.persistent()
    assert isinstance(record, TestRecord)
    assert record['field1'] == 'value1'

    evolver = TestRecord().evolver()
    evolver.set('field1', 'value1')
    try:
        evolver.set('field3', 'value3')
        evolver.persistent()
        assert False, "AttributeError should have been raised"
    except AttributeError as e:
        assert "'field3' is not among the specified fields for TestRecord" in str(e)


# LLM-generated content at query #3
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class MyRecord(PRecord):
        field1 = field(type=int, mandatory=True)
        field2 = field(type=str, initial="default")

    # Test creation with mandatory field
    record = MyRecord(field1=42)
    assert record.field1 == 42
    assert record.field2 == "default"

    # Test creation with all fields
    record = MyRecord(field1=10, field2="custom")
    assert record.field1 == 10
    assert record.field2 == "custom"

    # Test missing mandatory field
    try:
        MyRecord(field2="test")
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass

    # Test invalid type
    try:
        MyRecord(field1="not an int", field2="test")
        assert False, "Should raise TypeError"
    except TypeError:
        pass

    # Test ignore extra
    class MyRecordIgnoreExtra(PRecord):
        __ignore_extra__ = True
        field1 = field(type=int)

    record = MyRecordIgnoreExtra(field1=1, extra_field=2)
    assert record.field1 == 1
    assert "extra_field" not in record


# LLM-generated content at query #4
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int)

    evolver = _PRecordEvolver(TestRecord, pmap(), _factory_fields={'name'}, _ignore_extra=True)
    evolver.set('name', 'John')
    evolver.set('age', 30)
    record = evolver.persistent()

    assert record['name'] == 'John'
    assert record['age'] == 30


# LLM-generated content at query #5
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class ExampleRecord(PRecord):
        name = field(type=str)
        age = field(type=int)

    record = ExampleRecord(name='Alice', age=30)
    assert record == {'name': 'Alice', 'age': 30}

    record = ExampleRecord(name='Bob', age=25, _ignore_extra=True)
    assert record == {'name': 'Bob', 'age': 25}

    record = ExampleRecord(name='Charlie', age=40, _factory_fields=['name'])
    assert record == {'name': 'Charlie', 'age': 40}


# LLM-generated content at query #6
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        foo = field(type=int, mandatory=True)
        bar = field(type=str)

    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('foo', 42)
    evolver.set('bar', 'baz')
    record = evolver.persistent()

    assert record['foo'] == 42
    assert record['bar'] == 'baz'


# LLM-generated content at query #7
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        __invariant__ = lambda value: (len(value) > 0, 'EMPTY')
        field1 = field(type=str, mandatory=True)
        field2 = field(type=int, initial=42)

    # Test with mandatory field missing
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

    # Test with valid fields
    evolver = _PRecordEvolver(TestRecord, pmap({'field1': 'value'}))
    record = evolver.persistent()
    assert record['field1'] == 'value'
    assert record['field2'] == 42

    # Test with invariant violation
    evolver = _PRecordEvolver(TestRecord, pmap({'field1': ''}))
    evolver._invariant_error_codes.append('EMPTY')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'EMPTY' in e.invariant_errors


# LLM-generated content at query #8
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class MyRecord(PRecord):
        field1 = field()
        field2 = field()

    # Test creating a record with initial values
    record = MyRecord(field1=1, field2=2)
    assert record.field1 == 1
    assert record.field2 == 2

    # Test creating a record with missing fields (should raise InvariantException)
    try:
        record = MyRecord(field1=1)
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

    # Test creating a record with extra fields (should raise AttributeError)
    try:
        record = MyRecord(field1=1, field2=2, field3=3)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creating a record with ignore_extra=True
    record = MyRecord.create({'field1': 1, 'field2': 2, 'field3': 3}, ignore_extra=True)
    assert record.field1 == 1
    assert record.field2 == 2
    assert 'field3' not in record

    # Test creating a record with factory fields
    record = MyRecord.create({'field1': 1, 'field2': 2}, _factory_fields={'field1'})
    assert record.field1 == 1
    assert record.field2 == 2

    # Test creating a record with initial values from class
    class MyRecordWithInitial(PRecord):
        field1 = field(initial=1)
        field2 = field(initial=2)

    record = MyRecordWithInitial()
    assert record.field1 == 1
    assert record.field2 == 2

    # Test creating a record with initial values from class and kwargs
    record = MyRecordWithInitial(field1=3)
    assert record.field1 == 3
    assert record.field2 == 2

    # Test creating a record with initial values from class and kwargs (ignore_extra=True)
    record = MyRecordWithInitial.create({'field1': 3, 'field3': 4}, ignore_extra=True)
    assert record.field1 == 3
    assert record.field2 == 2
    assert 'field3' not in record

    # Test creating a record with initial values from class and kwargs (ignore_extra=False)
    try:
        record = MyRecordWithInitial.create({'field1': 3, 'field3': 4})
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creating a record with initial values from class and kwargs (factory fields)
    record = MyRecordWithInitial.create({'field1': 3, 'field2': 4}, _factory_fields={'field1'})
    assert record.field1 == 3
    assert record.field2 == 4

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=True)
    record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'}, ignore_extra=True)
    assert record.field1 == 3
    assert record.field2 == 4
    assert 'field3' not in record

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=False)
    try:
        record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'})
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=True)
    record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'}, ignore_extra=True)
    assert record.field1 == 3
    assert record.field2 == 4
    assert 'field3' not in record

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=False)
    try:
        record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'})
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=True)
    record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'}, ignore_extra=True)
    assert record.field1 == 3
    assert record.field2 == 4
    assert 'field3' not in record

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=False)
    try:
        record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'})
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=True)
    record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'}, ignore_extra=True)
    assert record.field1 == 3
    assert record.field2 == 4
    assert 'field3' not in record

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=False)
    try:
        record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'})
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=True)
    record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'}, ignore_extra=True)
    assert record.field1 == 3
    assert record.field2 == 4
    assert 'field3' not in record

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=False)
    try:
        record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'})
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=True)
    record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'}, ignore_extra=True)
    assert record.field1 == 3
    assert record.field2 == 4
    assert 'field3' not in record

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=False)
    try:
        record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'})
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=True)
    record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'}, ignore_extra=True)
    assert record.field1 == 3
    assert record.field2 == 4
    assert 'field3' not in record

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=False)
    try:
        record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'})
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=True)
    record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'}, ignore_extra=True)
    assert record.field1 == 3
    assert record.field2 == 4
    assert 'field3' not in record

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=False)
    try:
        record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'})
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test creating a record with initial values from class and kwargs (factory fields, ignore_extra=True)
    record = MyRecordWithInitial.create({'field1': 3, 'field2': 4, 'field3': 5}, _factory_fields={'field1'}, ignore_extra=True)
    assert record.field1 == 


# LLM-generated content at query #9
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        field1 = field(type=int)
        field2 = field(type=str)

    evolver = TestRecord().evolver()
    evolver['field1'] = 42
    evolver['field2'] = 'test'
    record = evolver.persistent()

    assert record['field1'] == 42
    assert record['field2'] == 'test'

    try:
        evolver['field3'] = 'invalid'
        evolver.persistent()
    except AttributeError:
        pass
    else:
        assert False, "Setting an invalid field should raise an AttributeError"

    try:
        evolver['field1'] = 'invalid'
        evolver.persistent()
    except InvariantException:
        pass
    else:
        assert False, "Setting a field with an invalid type should raise an InvariantException"


# LLM-generated content at query #10
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class R(PRecord):
        a = field()
        b = field()

    r = R(a=1, b=2)
    assert r['a'] == 1
    assert r['b'] == 2

    # Test with initial values
    class R2(PRecord):
        a = field(initial=1)
        b = field(initial=2)

    r2 = R2()
    assert r2['a'] == 1
    assert r2['b'] == 2

    # Test with factory fields
    class R3(PRecord):
        a = field(factory=lambda x: x * 2)

    r3 = R3(a=1)
    assert r3['a'] == 2

    # Test with ignore_extra
    class R4(PRecord):
        a = field()

    r4 = R4(a=1, _ignore_extra=True)
    assert r4['a'] == 1
    assert len(r4) == 1

    # Test with _factory_fields
    class R5(PRecord):
        a = field(factory=lambda x: x * 2)

    r5 = R5(a=1, _factory_fields={'a'})
    assert r5['a'] == 2

    # Test with _precord_size and _precord_buckets
    r6 = R(_precord_size=2, _precord_buckets=[('a', 1), ('b', 2)])
    assert r6['a'] == 1
    assert r6['b'] == 2


# LLM-generated content at query #11
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        __invariant__ = lambda value: (value['x'] >= 0, 'x_negative')
        x = field(type=int, mandatory=True)
        y = field(type=str)

    # Test setting a valid field
    evolver = TestRecord().evolver()
    evolver.set('x', 10)
    assert evolver['x'] == 10

    # Test setting an invalid field (non-existent)
    try:
        evolver.set('z', 5)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test setting a field with invalid type
    try:
        evolver.set('x', "not an int")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test setting a field that violates invariant
    evolver = TestRecord().evolver()
    try:
        evolver.set('x', -1)
        evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert 'x_negative' in e.invariant_errors

    # Test setting multiple fields
    evolver = TestRecord().evolver()
    evolver.set('x', 5)
    evolver.set('y', 'test')
    record = evolver.persistent()
    assert record['x'] == 5
    assert record['y'] == 'test'


# LLM-generated content at query #12
#--------------------------

# Unit test for method __repr__ of class PRecord
def test_PRecord___repr__():
    class MyRecord(PRecord):
        pass
    instance = MyRecord(a=1, b=2)
    assert repr(instance) == "MyRecord(a=1, b=2)"


# LLM-generated content at query #13
#--------------------------

# Unit test for method __new__ of class _PRecordMeta
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        field1 = field(type=int, initial=42)
        field2 = field(type=str, mandatory=True)

    assert TestRecord._precord_fields == {'field1': field(type=int, initial=42), 'field2': field(type=str, mandatory=True)}
    assert TestRecord._precord_mandatory_fields == {'field2'}
    assert TestRecord._precord_initial_values == {'field1': 42}


# LLM-generated content at query #14
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = field(type=int)
        field2 = field(type=str)

    evolver = TestRecord().evolver()
    evolver.set('field1', 42)
    evolver.set('field2', 'hello')

    record = evolver.persistent()
    assert record['field1'] == 42
    assert record['field2'] == 'hello'


# LLM-generated content at query #15
#--------------------------

# Unit test for method __new__ of class _PRecordMeta
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda value: (True, None)
        field1 = field(type=int, mandatory=True)
        field2 = field(type=str, initial="default")

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord._precord_mandatory_fields == {'field1'}
    assert TestRecord._precord_initial_values == {'field2': 'default'}


# LLM-generated content at query #16
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    from pyrsistent import field

    class TestRecord(PRecord):
        name = field(type=str)
        age = field(type=int)

    evolver = TestRecord().evolver()
    evolver.set('name', 'John')
    evolver.set('age', 30)
    record = evolver.persistent()

    assert record['name'] == 'John'
    assert record['age'] == 30

    try:
        evolver.set('invalid_key', 'value')
    except AttributeError as e:
        assert str(e) == "'invalid_key' is not among the specified fields for TestRecord"
    else:
        assert False, "Expected AttributeError"

    try:
        evolver.set('age', 'not_an_int')
    except InvariantException:
        pass
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #17
#--------------------------

# Unit test for method __repr__ of class PRecord
def test_PRecord___repr__():
    class MyRecord(PRecord):
        field1 = field()
        field2 = field()

    record = MyRecord(field1='value1', field2='value2')
    assert repr(record) == "MyRecord(field1='value1', field2='value2')"


# LLM-generated content at query #18
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class MyRecord(PRecord):
        field1 = field(type=int, mandatory=True)
        field2 = field(type=str, initial="default")

    # Test creating with mandatory field
    record = MyRecord(field1=42)
    assert record.field1 == 42
    assert record.field2 == "default"

    # Test creating with all fields
    record = MyRecord(field1=10, field2="custom")
    assert record.field1 == 10
    assert record.field2 == "custom"

    # Test missing mandatory field
    try:
        record = MyRecord(field2="test")
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "MyRecord.field1" in e.missing_fields

    # Test wrong type
    try:
        record = MyRecord(field1="not an int", field2="test")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test ignore extra
    record = MyRecord(field1=1, extra_field=2, _ignore_extra=True)
    assert record.field1 == 1
    assert not hasattr(record, 'extra_field')

    # Test factory fields
    record = MyRecord(field1=1, field2="test", _factory_fields=[MyRecord._precord_fields['field1']])
    assert record.field1 == 1
    assert record.field2 == "test"  # Should be passed through without factory processing

    print("All PRecord.__new__ tests passed")

test_PRecord___new__()


# LLM-generated content at query #19
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class Person(PRecord):
        name = field(type=str, mandatory=True)
        age = field(type=int, mandatory=True)
    
    evolver = Person().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 30)
    person = evolver.persistent()
    
    assert person['name'] == 'Alice'
    assert person['age'] == 30
    
    evolver = Person().evolver()
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'Person.name' in e.missing_fields
        assert 'Person.age' in e.missing_fields
    
    evolver = Person().evolver()
    evolver.set('name', 'Bob')
    evolver.set('age', -1)
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'age' in str(e)


# LLM-generated content at query #20
#--------------------------

# Unit test for method __new__ of class _PRecordMeta
def test__PRecordMeta___new__():
    class TestClass(metaclass=_PRecordMeta):
        __slots__ = ('field1', 'field2')
        field1 = 'value1'
        field2 = 'value2'

    assert hasattr(TestClass, '_precord_fields')
    assert hasattr(TestClass, '_precord_invariants')
    assert hasattr(TestClass, '_precord_mandatory_fields')
    assert hasattr(TestClass, '_precord_initial_values')
    assert TestClass.__slots__ == ()


# LLM-generated content at query #21
#--------------------------

# Unit test for method __new__ of class _PRecordMeta
def test__PRecordMeta___new__():
    class TestFields:
        mandatory = True
        initial = None

    class TestRecord(PRecord):
        field1 = TestFields()
        field2 = TestFields()

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord._precord_fields == {'field1': TestFields(), 'field2': TestFields()}
    assert TestRecord._precord_mandatory_fields == {'field1', 'field2'}
    assert TestRecord._precord_initial_values == {}
    assert TestRecord.__slots__ == ()


# LLM-generated content at query #22
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda value: (value['x'] > 0, 'x must be positive')
        x = field(type=int, mandatory=True)
        y = field(type=str, initial='default')

    # Test mandatory field
    try:
        TestRecord(y='test')
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecord.x' in e.missing_fields

    # Test type check
    try:
        TestRecord(x='not an int', y='test')
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test invariant
    try:
        TestRecord(x=-1, y='test')
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'x must be positive' in e.invariant_errors

    # Test initial value
    record = TestRecord(x=1)
    assert record.y == 'default'

    # Test successful creation
    record = TestRecord(x=1, y='test')
    assert record.x == 1
    assert record.y == 'test'


# LLM-generated content at query #23
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class TestRecord(PRecord):
        a = field(int)
        b = field(int)
    
    rec = TestRecord(a=1, b=2)
    assert isinstance(rec, TestRecord)
    assert rec['a'] == 1
    assert rec['b'] == 2


# LLM-generated content at query #24
#--------------------------

# Unit test for method __new__ of class _PRecordMeta
def test__PRecordMeta___new__():
    class TestPRecordMeta(PRecord):
        field1 = field(type=int)
        field2 = field(type=str, mandatory=True)
        field3 = field(type=float, initial=3.14)
        __invariant__ = lambda self: self.field1 > 0

    # Check if fields are set correctly
    assert hasattr(TestPRecordMeta, '_precord_fields')
    assert TestPRecordMeta._precord_fields['field1'].type == int
    assert TestPRecordMeta._precord_fields['field2'].type == str
    assert TestPRecordMeta._precord_fields['field3'].type == float

    # Check if mandatory fields are set correctly
    assert hasattr(TestPRecordMeta, '_precord_mandatory_fields')
    assert TestPRecordMeta._precord_mandatory_fields == {'field2'}

    # Check if initial values are set correctly
    assert hasattr(TestPRecordMeta, '_precord_initial_values')
    assert TestPRecordMeta._precord_initial_values['field3'] == 3.14

    # Check if invariants are stored correctly
    assert hasattr(TestPRecordMeta, '_precord_invariants')
    assert len(TestPRecordMeta._precord_invariants) == 1

    # Check if slots are set correctly
    assert TestPRecordMeta.__slots__ == ()


# LLM-generated content at query #25
#--------------------------

# Unit test for method __repr__ of class PRecord
def test_PRecord___repr__():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    r = TestRecord(field1='value1', field2='value2')
    assert repr(r) == "TestRecord(field1='value1', field2='value2')"


# LLM-generated content at query #26
#--------------------------

# Unit test for method __repr__ of class PRecord
def test_PRecord___repr__():
    class MyRecord(PRecord):
        __slots__ = ()
        field1 = field(type=str)
        field2 = field(type=int)

    record = MyRecord(field1="test", field2=123)
    assert repr(record) == "MyRecord(field1='test', field2=123)"


# LLM-generated content at query #27
#--------------------------

# Unit test for method __new__ of class _PRecordMeta
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        __invariant__ = lambda value: (len(value) > 0, 'EMPTY')
        field1 = field(type=int, mandatory=True)
        field2 = field(type=str, initial='default')

    assert hasattr(TestRecord, '_precord_fields')
    assert hasattr(TestRecord, '_precord_invariants')
    assert hasattr(TestRecord, '_precord_mandatory_fields')
    assert hasattr(TestRecord, '_precord_initial_values')
    assert TestRecord._precord_mandatory_fields == {'field1'}
    assert TestRecord._precord_initial_values == {'field2': 'default'}
    assert len(TestRecord._precord_invariants) == 1


# LLM-generated content at query #28
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        __slots__ = ()
        field1 = field(type=int, invariant=lambda x: (x > 0, 'field1 must be positive'))
        field2 = field(type=str)

    # Test setting a valid field
    record = TestRecord(field1=1, field2='test')
    evolver = record.evolver()
    evolver.set('field1', 2)
    updated_record = evolver.persistent()
    assert updated_record['field1'] == 2

    # Test setting an invalid field (violates invariant)
    evolver = record.evolver()
    try:
        evolver.set('field1', -1)
        updated_record = evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('field1 must be positive',)

    # Test setting a non-existent field
    evolver = record.evolver()
    try:
        evolver.set('field3', 'value')
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert str(e) == "'field3' is not among the specified fields for TestRecord"

    # Test setting a field with wrong type
    evolver = record.evolver()
    try:
        evolver.set('field1', 'not an int')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert str(e) == "Expected field 'field1' to be of type int, got str"


# LLM-generated content at query #29
#--------------------------

# Unit test for method serialize of class PRecord
def test_PRecord_serialize():
    class MyRecord(PRecord):
        __invariant__ = lambda value: (len(value['name']) > 0, 'name must not be empty')
        name = field(type=str, mandatory=True)
        age = field(type=int, initial=0)

    r = MyRecord(name="Alice")
    serialized = r.serialize()
    assert serialized == {'name': 'Alice', 'age': 0}

    class MyRecordWithSerializer(PRecord):
        name = field(type=str, serializer=lambda v, _: v.upper())
        age = field(type=int)

    r = MyRecordWithSerializer(name="Bob", age=30)
    serialized = r.serialize()
    assert serialized == {'name': 'BOB', 'age': 30}


# LLM-generated content at query #30
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        pass

    evolver = TestRecord().evolver()
    evolver.set('key', 'value')
    assert evolver['key'] == 'value'


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class MyRecord(PRecord):
        field = field(type=int, mandatory=True)

    # Test creation with mandatory field
    try:
        MyRecord()
    except InvariantException as e:
        assert e.missing_fields == ('MyRecord.field',)
    else:
        assert False, "Expected InvariantException"

    # Test creation with valid field
    record = MyRecord(field=1)
    assert record['field'] == 1

    # Test creation with invalid field type
    try:
        MyRecord(field='not an int')
    except InvariantException as e:
        assert e.invariant_errors == ('type',)
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #2
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = field(type=str, mandatory=True)
        field2 = field(type=int)

    evolver = TestRecord().evolver()
    evolver.set('field1', 'value1')
    evolver.set('field2', 42)
    record = evolver.persistent()
    assert record['field1'] == 'value1'
    assert record['field2'] == 42

    try:
        evolver.set('field3', 'value3')
        assert False, "Setting an undefined field should raise AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #3
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class MyRecord(PRecord):
        field1 = field(type=int)
        field2 = field(type=str)
    
    record = MyRecord(field1=10, field2="test")
    assert record.field1 == 10
    assert record.field2 == "test"


# LLM-generated content at query #4
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = field(type=str)

    evolver = TestRecord().evolver()
    evolver.set('field1', 'value1')
    assert evolver.persistent() == TestRecord(field1='value1')

    try:
        evolver.set('field2', 'value2')
    except AttributeError:
        pass
    else:
        assert False, "Expected AttributeError"



# LLM-generated content at query #5
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        field1 = field(type=str)
        field2 = field(type=int, mandatory=True)

    # Test case 1: All mandatory fields are present
    evolver = TestRecord().evolver()
    evolver.set('field1', 'value1')
    evolver.set('field2', 42)
    record = evolver.persistent()
    assert record['field1'] == 'value1'
    assert record['field2'] == 42

    # Test case 2: Mandatory field is missing
    evolver = TestRecord().evolver()
    evolver.set('field1', 'value1')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecord.field2' in e.missing_fields

    # Test case 3: Invariant fails
    evolver = TestRecord().evolver()
    evolver.set('field1', 'value1')
    evolver.set('field2', 'not an int')
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TypeError' in e.invariant_errors

    # Test case 4: No changes made
    evolver = TestRecord(field1='value1', field2=42).evolver()
    record = evolver.persistent()
    assert record['field1'] == 'value1'
    assert record['field2'] == 42


# LLM-generated content at query #6
#--------------------------

# Unit test for method __new__ of class _PRecordMeta
def test__PRecordMeta___new__():
    class TestRecord(PRecord):
        field1 = field(type=int)
        field2 = field(type=str, mandatory=True)
    
    assert TestRecord._precord_fields['field1'].type == int
    assert TestRecord._precord_fields['field2'].type == str
    assert TestRecord._precord_mandatory_fields == {'field2'}
    assert TestRecord._precord_initial_values == {}
    assert TestRecord.__slots__ == ()


# LLM-generated content at query #7
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        __invariant__ = lambda value: (value['x'] > 0, 'x must be positive')
        x = field(type=int, mandatory=True, invariant=lambda x: (x < 10, 'x must be less than 10'))

    # Test setting a valid value
    evolver = TestRecord().evolver()
    evolver.set('x', 5)
    record = evolver.persistent()
    assert record['x'] == 5

    # Test setting an invalid value (violates field invariant)
    evolver = TestRecord().evolver()
    try:
        evolver.set('x', 15)
        record = evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'x must be less than 10' in e.invariant_errors

    # Test setting an invalid value (violates global invariant)
    evolver = TestRecord().evolver()
    try:
        evolver.set('x', -5)
        record = evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'x must be positive' in e.invariant_errors

    # Test setting a non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('y', 10)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'y' is not among the specified fields for TestRecord" in str(e)


# LLM-generated content at query #8
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = field(type=str)
        field2 = field(type=int)

    record = TestRecord(field1='value1', field2=42)
    evolver = record.evolver()
    evolver.set('field1', 'new_value')
    updated_record = evolver.persistent()

    assert updated_record['field1'] == 'new_value'
    assert updated_record['field2'] == 42

    # Test setting a non-existent field
    try:
        evolver.set('nonexistent_field', 'value')
        assert False, "Setting a non-existent field should raise an AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #9
#--------------------------

# Unit test for method __repr__ of class PRecord
def test_PRecord___repr__():
    class MyRecord(PRecord):
        pass

    r = MyRecord(a=1, b=2)
    assert repr(r) == "MyRecord(a=1, b=2)"


# LLM-generated content at query #10
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class MyRecord(PRecord):
        field1 = field(int, mandatory=True)
        field2 = field(int, initial=42)

    record = MyRecord(field1=1)
    assert record['field1'] == 1
    assert record['field2'] == 42

    try:
        MyRecord()
    except InvariantException as e:
        assert e.missing_fields == ('MyRecord.field1',)

    record = MyRecord(_ignore_extra=True, field1=1, field3=3)
    assert record['field1'] == 1
    assert record['field2'] == 42
    assert 'field3' not in record

    record = MyRecord(_factory_fields=[MyRecord.field1], field1=1)
    assert record['field1'] == 1
    assert record['field2'] == 42

    record = MyRecord(_factory_fields=[MyRecord.field1], field1='1')
    assert record['field1'] == 1
    assert record['field2'] == 42

    try:
        MyRecord(_factory_fields=[MyRecord.field1], field1='a')
    except InvariantException as e:
        assert e.invariant_errors == ('value must be an integer',)


# LLM-generated content at query #11
#--------------------------

# Unit test for method __new__ of class _PRecordMeta
def test__PRecordMeta___new__():
    class TestPRecord(PRecord):
        field1 = field(type=int)
        field2 = field(type=str, mandatory=True)
        field3 = field(type=float, initial=lambda: 3.14)

    assert hasattr(TestPRecord, '_precord_fields')
    assert hasattr(TestPRecord, '_precord_invariants')
    assert hasattr(TestPRecord, '_precord_mandatory_fields')
    assert hasattr(TestPRecord, '_precord_initial_values')
    assert TestPRecord._precord_mandatory_fields == {'field2'}
    assert TestPRecord._precord_initial_values == {'field3': 3.14}


# LLM-generated content at query #12
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class TestRecord(PRecord):
        field1 = field()
        field2 = field()

    record = TestRecord(field1=1, field2=2)
    assert record.field1 == 1
    assert record.field2 == 2

    try:
        TestRecord(field1=1, field2=2, field3=3)
    except AttributeError:
        pass
    else:
        assert False, "Should raise AttributeError when initializing with extra fields"

    record_with_initial = TestRecord()
    assert hasattr(record_with_initial, 'field1')
    assert hasattr(record_with_initial, 'field2')

    record_with_factory = TestRecord(_factory_fields={'field1'}, field1=1, field2=2)
    assert record_with_factory.field1 == 1
    assert record_with_factory.field2 == 2

    record_ignore_extra = TestRecord(_ignore_extra=True, field1=1, field2=2, field3=3)
    assert record_ignore_extra.field1 == 1
    assert record_ignore_extra.field2 == 2
    assert not hasattr(record_ignore_extra, 'field3')


# LLM-generated content at query #13
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        __invariant__ = lambda value: (value['field1'] > 0, 'ERR1')
        field1 = field(type=int, mandatory=True, invariant=lambda x: (x > 0, 'ERR2'))

    # Test setting a valid value
    evolver = TestRecord().evolver()
    evolver.set('field1', 1)
    record = evolver.persistent()
    assert record['field1'] == 1

    # Test setting an invalid value (violates field invariant)
    evolver = TestRecord().evolver()
    evolver.set('field1', -1)
    try:
        record = evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'ERR2' in e.invariant_errors

    # Test setting an invalid value (violates global invariant)
    evolver = TestRecord().evolver()
    evolver.set('field1', 0)
    try:
        record = evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'ERR1' in e.invariant_errors

    # Test setting a non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('nonexistent', 1)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'nonexistent' is not among the specified fields for TestRecord" in str(e)


# LLM-generated content at query #14
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        field1 = field(type=str, mandatory=True)
        field2 = field(type=int)

    evolver = TestRecord().evolver()
    evolver.set('field1', 'value1')
    evolver.set('field2', 42)
    record = evolver.persistent()

    assert record['field1'] == 'value1'
    assert record['field2'] == 42

    try:
        evolver.set('field3', 'value3')
        record = evolver.persistent()
    except AttributeError as e:
        assert str(e) == "'field3' is not among the specified fields for TestRecord"

    evolver = TestRecord().evolver()
    try:
        record = evolver.persistent()
    except InvariantException as e:
        assert e.missing_fields == ("TestRecord.field1",)

    class TestRecordWithInvariant(PRecord):
        field1 = field(type=str, invariant=lambda x: (len(x) > 0, 'EMPTY'))

    evolver = TestRecordWithInvariant().evolver()
    evolver.set('field1', '')
    try:
        record = evolver.persistent()
    except InvariantException as e:
        assert e.invariant_errors == ('EMPTY',)


# LLM-generated content at query #15
#--------------------------

# Unit test for method __repr__ of class PRecord
def test_PRecord___repr__():
    class MyRecord(PRecord):
        pass

    record = MyRecord(a=1, b=2)
    assert repr(record) == "MyRecord(a=1, b=2)"


# LLM-generated content at query #16
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = field(type=int, mandatory=True)
        field2 = field(type=str)

    evolver = TestRecord().evolver()
    evolver.set('field1', 10)
    evolver.set('field2', 'value')
    record = evolver.persistent()

    assert record['field1'] == 10
    assert record['field2'] == 'value'


# LLM-generated content at query #17
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = field(type=int)
        field2 = field(type=str)

    evolver = TestRecord().evolver()
    evolver.set('field1', 42)
    evolver.set('field2', "hello")
    record = evolver.persistent()

    assert record.field1 == 42
    assert record.field2 == "hello"


# LLM-generated content at query #18
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = field(type=int)
        field2 = field(type=str)

    evolver = TestRecord().evolver()
    evolver.set('field1', 42)
    evolver.set('field2', 'hello')
    record = evolver.persistent()

    assert record['field1'] == 42
    assert record['field2'] == 'hello'


# LLM-generated content at query #19
#--------------------------

# Unit test for method __repr__ of class PRecord
def test_PRecord___repr__():
    class MyRecord(PRecord):
        __slots__ = ()
        x = field(int, mandatory=True)
        y = field(str, mandatory=True)

    r = MyRecord(x=1, y='hello')
    assert repr(r) == "MyRecord(x=1, y='hello')"
    r2 = MyRecord(x=42, y='world')
    assert repr(r2) == "MyRecord(x=42, y='world')"


# LLM-generated content at query #20
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        __invariant__ = lambda x: (len(x) > 0, 'EMPTY')
        field1 = field(type=int, mandatory=True)
        field2 = field(type=str, initial='default')

    # Test with mandatory field missing
    evolver = TestRecord().evolver()
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'TestRecord.field1' in e.missing_fields

    # Test with empty record (violates global invariant)
    evolver = TestRecord(field1=1).evolver()
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'EMPTY' in e.invariant_errors

    # Test successful persistence
    evolver = TestRecord(field1=1, field2='test').evolver()
    record = evolver.persistent()
    assert record.field1 == 1
    assert record.field2 == 'test'

    # Test with field invariant violation
    evolver = TestRecord(field1='invalid', field2='test').evolver()
    try:
        evolver.persistent()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'field_type' in str(e.invariant_errors[0])


# LLM-generated content at query #21
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        field1 = field(type=int, mandatory=True)
        field2 = field(type=str)

    record = TestRecord(field1=42, field2="test")
    evolver = record.evolver()
    evolver.set("field1", 100)
    assert evolver.persistent()["field1"] == 100
    evolver.set("field2", "new_value")
    assert evolver.persistent()["field2"] == "new_value"

    try:
        evolver.set("field3", "invalid")
        assert False, "Expected AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #22
#--------------------------

# Unit test for method serialize of class PRecord
def test_PRecord_serialize():
    class MyRecord(PRecord):
        field1 = field(type=str)
        field2 = field(type=int)

    record = MyRecord(field1="value1", field2=42)
    serialized = record.serialize()
    assert serialized == {'field1': 'value1', 'field2': 42}


# LLM-generated content at query #23
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class ExampleRecord(PRecord):
        name = field(type=str)
        age = field(type=int)

    evolver = ExampleRecord().evolver()
    evolver.set('name', 'Alice')
    evolver.set('age', 30)
    record = evolver.persistent()

    assert record['name'] == 'Alice'
    assert record['age'] == 30

    try:
        evolver.set('invalid_key', 'value')
    except AttributeError:
        pass
    else:
        assert False, "Expected AttributeError for invalid key"


# LLM-generated content at query #24
#--------------------------

# Unit test for method __new__ of class PRecord
def test_PRecord___new__():
    class MyRecord(PRecord):
        __invariant__ = lambda value: (value['x'] > 0, 'x must be positive')
        x = field(type=int, mandatory=True)
        y = field(type=str, initial='default')

    # Test with mandatory field missing
    try:
        MyRecord(y='test')
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'MyRecord.x' in e.missing_fields

    # Test with invalid type
    try:
        MyRecord(x='not an int', y='test')
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Invalid type for field' in str(e)

    # Test with invariant violation
    try:
        MyRecord(x=-1, y='test')
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert 'x must be positive' in e.invariant_errors

    # Test successful creation
    r = MyRecord(x=1)
    assert r.x == 1
    assert r.y == 'default'

    r2 = MyRecord(x=2, y='test')
    assert r2.x == 2
    assert r2.y == 'test'

    # Test ignore_extra
    r3 = MyRecord.create({'x': 3, 'z': 'extra'}, ignore_extra=True)
    assert r3.x == 3
    assert 'z' not in r3

    # Test factory fields
    r4 = MyRecord.create({'x': 4, 'y': 'test'}, _factory_fields=[MyRecord.y])
    assert r4.x == 4  # Should be set directly without factory
    assert r4.y == 'test'  # Should go through factory (though str factory is pass-through)

    print("All tests passed for PRecord.__new__")


# LLM-generated content at query #25
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        field1 = field(type=int, mandatory=True)
        field2 = field(type=str)

    evolver = TestRecord().evolver()
    evolver.set('field1', 42)
    evolver.set('field2', 'hello')
    record = evolver.persistent()
    assert record['field1'] == 42
    assert record['field2'] == 'hello'

    evolver = TestRecord().evolver()
    evolver.set('field1', 42)
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert 'TestRecord.field2' in e.missing_fields

    evolver = TestRecord().evolver()
    evolver.set('field1', 'not an int')
    try:
        evolver.persistent()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('type',)

    evolver = TestRecord().evolver()
    evolver.set('field1', 42)
    evolver.set('field2', 'hello')
    evolver.set('field3', 'extra')
    try:
        evolver.persistent()
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert str(e) == "'field3' is not among the specified fields for TestRecord"


# LLM-generated content at query #26
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        __invariant__ = lambda self: (len(self) > 0, 'EMPTY_RECORD')
        field1 = field(type=int, invariant=lambda x: (x > 0, 'NON_POSITIVE'))
        field2 = field(type=str, mandatory=True)

    # Test successful persistence
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('field1', 1)
    evolver.set('field2', 'test')
    record = evolver.persistent()
    assert record['field1'] == 1
    assert record['field2'] == 'test'

    # Test missing mandatory field
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('field1', 1)
    try:
        evolver.persistent()
        assert False, 'Should have raised InvariantException'
    except InvariantException as e:
        assert 'TestRecord.field2' in e.missing_fields

    # Test field invariant violation
    evolver = _PRecordEvolver(TestRecord, pmap())
    evolver.set('field1', -1)
    evolver.set('field2', 'test')
    try:
        evolver.persistent()
        assert False, 'Should have raised InvariantException'
    except InvariantException as e:
        assert 'NON_POSITIVE' in e.invariant_errors

    # Test global invariant violation
    evolver = _PRecordEvolver(TestRecord, pmap())
    try:
        evolver.persistent()
        assert False, 'Should have raised InvariantException'
    except InvariantException as e:
        assert 'EMPTY_RECORD' in e.invariant_errors


# LLM-generated content at query #27
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        field1 = field(type=str)
        field2 = field(type=int)

    evolver = TestRecord().evolver()
    evolver.set('field1', 'value1')
    evolver.set('field2', 123)
    record = evolver.persistent()

    assert record['field1'] == 'value1'
    assert record['field2'] == 123
    assert isinstance(record, TestRecord)


# LLM-generated content at query #28
#--------------------------

# Unit test for method set of class _PRecordEvolver
def test__PRecordEvolver_set():
    class TestRecord(PRecord):
        __invariant__ = lambda value: (value['x'] > 0, 'X must be greater than 0')
        x = field(type=int, mandatory=True, invariant=lambda x: (x < 10, 'X must be less than 10'))

    # Test setting a valid value
    evolver = TestRecord().evolver()
    evolver.set('x', 5)
    record = evolver.persistent()
    assert record['x'] == 5

    # Test setting an invalid value (fails field invariant)
    evolver = TestRecord().evolver()
    evolver.set('x', 15)
    try:
        record = evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('X must be less than 10',)

    # Test setting an invalid value (fails global invariant)
    evolver = TestRecord().evolver()
    evolver.set('x', -5)
    try:
        record = evolver.persistent()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ('X must be greater than 0',)

    # Test setting a non-existent field
    evolver = TestRecord().evolver()
    try:
        evolver.set('y', 10)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert str(e) == "'y' is not among the specified fields for TestRecord"


# LLM-generated content at query #29
#--------------------------

# Unit test for method __new__ of class _PRecordMeta
def test__PRecordMeta___new__():
    class TestRecordMeta(metaclass=_PRecordMeta):
        pass

    assert hasattr(TestRecordMeta, '_precord_fields')
    assert hasattr(TestRecordMeta, '_precord_invariants')
    assert hasattr(TestRecordMeta, '_precord_mandatory_fields')
    assert hasattr(TestRecordMeta, '_precord_initial_values')
    assert hasattr(TestRecordMeta, '__slots__')


# LLM-generated content at query #30
#--------------------------

# Unit test for method persistent of class _PRecordEvolver
def test__PRecordEvolver_persistent():
    class TestRecord(PRecord):
        field1 = field(type=int, mandatory=True)
        field2 = field(type=str, mandatory=False)

    evolver = TestRecord({'field1': 42}).evolver()
    record = evolver.persistent()
    assert record == {'field1': 42}, "Expected record to contain field1=42"

    evolver['field2'] = "value"
    record = evolver.persistent()
    assert record == {'field1': 42, 'field2': "value"}, "Expected record to contain field1=42 and field2='value'"

    try:
        evolver['field3'] = "extra"
        record = evolver.persistent()
        assert False, "Setting an undefined field should raise an AttributeError"
    except AttributeError:
        pass

    try:
        TestRecord(field2="value").evolver().persistent()
        assert False, "Missing mandatory field should raise an InvariantException"
    except InvariantException:
        pass


