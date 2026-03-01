####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PClassEvolver_remove():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    evolver = original.evolver()

    # Test successful removal
    result = evolver.remove('x')
    assert result is evolver
    assert 'x' not in evolver._pclass_evolver_data
    assert 'y' in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty
    assert 'x' not in evolver._factory_fields

    # Test removal of non-existent field
    with pytest.raises(AttributeError):
        evolver.remove('z')

    # Test persistent after removal
    persistent = evolver.persistent()
    assert not hasattr(persistent, 'x')
    assert persistent.y == 2

    # Test removal from persistent object
    evolver2 = persistent.evolver()
    result2 = evolver2.remove('y')
    assert result2 is evolver2
    assert 'y' not in evolver2._pclass_evolver_data
    assert evolver2._pclass_evolver_data_is_dirty

    persistent2 = evolver2.persistent()
    assert not hasattr(persistent2, 'x')
    assert not hasattr(persistent2, 'y')


# LLM-generated content at query #2
#--------------------------

```python
def test_PClass___hash__():
    # Test basic hash functionality
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=2, y=1)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)

    # Test hash with different types
    class MixedClass(PClass):
        a = field(type=int)
        b = field(type=str)
        c = field(type=float)

    obj4 = MixedClass(a=10, b="test", c=3.14)
    obj5 = MixedClass(a=10, b="test", c=3.14)
    obj6 = MixedClass(a=20, b="other", c=6.28)

    assert hash(obj4) == hash(obj5)
    assert hash(obj4) != hash(obj6)

    # Test hash with missing optional fields
    class OptionalClass(PClass):
        required = field()
        optional = field(initial=None)

    obj7 = OptionalClass(required=1)
    obj8 = OptionalClass(required=1)
    obj9 = OptionalClass(required=1, optional=2)

    assert hash(obj7) == hash(obj8)
    assert hash(obj7) != hash(obj9)

    # Test hash with nested PClass
    class NestedClass(PClass):
        inner = field(type=TestClass)
        value = field()

    obj10 = NestedClass(inner=TestClass(x=1, y=2), value=10)
    obj11 = NestedClass(inner=TestClass(x=1, y=2), value=10)
    obj12 = NestedClass(inner=TestClass(x=2, y=1), value=10)

    assert hash(obj10) == hash(obj11)
    assert hash(obj10) != hash(obj12)

    # Test hash consistency
    hash1 = hash(obj1)
    hash2 = hash(obj1)
    assert hash1 == hash2


# LLM-generated content at query #3
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestPClass(x=1, y=2)
    serialized = obj.serialize()
    assert serialized == {'x': 1, 'y': 4}

    class TestPClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x.upper() if fmt == 'upper' else x)

    obj = TestPClassWithFormat(x='hello')
    serialized = obj.serialize(format='upper')
    assert serialized == {'x': 'HELLO'}

    serialized = obj.serialize(format='lower')
    assert serialized == {'x': 'hello'}

    class TestPClassNoSerializer(PClass):
        x = field()

    obj = TestPClassNoSerializer(x=1)
    serialized = obj.serialize()
    assert serialized == {'x': 1}


# LLM-generated content at query #4
#--------------------------

```python
def test_PClassMeta___new__():
    class TestPClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestPClass, '_pclass_fields')
    assert hasattr(TestPClass, '_pclass_invariants')
    assert hasattr(TestPClass, '__slots__')
    assert '__weakref__' in TestPClass.__slots__
    assert '_pclass_frozen' in TestPClass.__slots__
    assert 'x' in TestPClass.__slots__
    assert 'y' in TestPClass.__slots__

    class TestPClass2(PClass):
        z = field()

    assert hasattr(TestPClass2, '_pclass_fields')
    assert hasattr(TestPClass2, '_pclass_invariants')
    assert hasattr(TestPClass2, '__slots__')
    assert '__weakref__' in TestPClass2.__slots__
    assert '_pclass_frozen' in TestPClass2.__slots__
    assert 'z' in TestPClass2.__slots__


# LLM-generated content at query #5
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    assert repr(instance) == "TestPClass(x=1, y=2)"

    instance_empty = TestPClass()
    assert repr(instance_empty) == "TestPClass()"

    class TestPClassWithDefault(PClass):
        x = field(initial=10)
        y = field()

    instance_default = TestPClassWithDefault(y=2)
    assert repr(instance_default) == "TestPClassWithDefault(x=10, y=2)"


# LLM-generated content at query #6
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y='test')
    assert repr(instance) == "TestClass(x=1, y='test')"

    instance2 = TestClass(x=None, y=0)
    assert repr(instance2) == "TestClass(x=None, y=0)"

    class EmptyClass(PClass):
        pass

    empty_instance = EmptyClass()
    assert repr(empty_instance) == "EmptyClass()"


# LLM-generated content at query #7
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__slots__' in TestClass.__dict__
    assert '_pclass_frozen' in TestClass.__slots__

    # Test with multiple fields
    class MultiFieldClass(PClass):
        x = field()
        y = field()
        z = field()

    assert len(MultiFieldClass._pclass_fields) == 3
    assert 'x' in MultiFieldClass._pclass_fields
    assert 'y' in MultiFieldClass._pclass_fields
    assert 'z' in MultiFieldClass._pclass_fields

    # Test with invariants
    class InvariantClass(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))
        __invariant__ = lambda self: (self.x < 100, "x must be less than 100")

    assert len(InvariantClass._pclass_invariants) == 1

    # Test __weakref__ slot
    class WeakRefClass(PClass):
        x = field()

    assert '__weakref__' in WeakRefClass.__slots__

    # Test inheritance
    class BaseClass(PClass):
        x = field()

    class DerivedClass(BaseClass):
        y = field()

    assert len(DerivedClass._pclass_fields) == 2
    assert 'x' in DerivedClass._pclass_fields
    assert 'y' in DerivedClass._pclass_fields


# LLM-generated content at query #8
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestPClass(x=1, y=2)
    serialized = instance.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestPClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x + fmt)

    instance_with_format = TestPClassWithFormat(x=1)
    serialized_with_format = instance_with_format.serialize(format='_test')

    assert serialized_with_format == {'x': '1_test'}

    class TestPClassNoSerializer(PClass):
        x = field()

    instance_no_serializer = TestPClassNoSerializer(x=1)
    serialized_no_serializer = instance_no_serializer.serialize()

    assert serialized_no_serializer == {'x': 1}


# LLM-generated content at query #9
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(type=int)

    # Test basic instantiation
    obj = TestClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2

    # Test with missing mandatory field
    try:
        TestClass(x=1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.y" in str(e)

    # Test with invalid type
    try:
        TestClass(x=1, y="not an int")
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

    # Test with extra fields
    try:
        TestClass(x=1, y=2, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

    # Test with default values
    class TestClassWithDefaults(PClass):
        a = field(initial=10)
        b = field(initial=lambda: "default")

    obj = TestClassWithDefaults()
    assert obj.a == 10
    assert obj.b == "default"

    # Test with factory fields
    class TestClassWithFactory(PClass):
        x = field()

    obj = TestClassWithFactory(x=1)
    new_obj = obj.set(x=2)
    assert new_obj.x == 2
    assert obj.x == 1  # Original unchanged

    # Test with ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()

    obj = TestClassIgnoreExtra.create({"x": 1, "y": 2}, ignore_extra=True)
    assert obj.x == 1
    assert not hasattr(obj, "y")

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)

    obj = TestClassWithInvariant(x=5)
    assert obj.x == 5

    try:
        TestClassWithInvariant(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Value must be positive" in str(e)


# LLM-generated content at query #10
#--------------------------

```python
def test__PClassEvolver_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    evolver = original.evolver()

    # Test setting a new value
    evolver.set('x', 10)
    assert evolver['x'] == 10
    assert evolver._pclass_evolver_data_is_dirty
    assert 'x' in evolver._factory_fields

    # Test setting the same value (should not mark as dirty)
    evolver.set('x', 10)
    assert not evolver._pclass_evolver_data_is_dirty

    # Test setting a different field
    evolver.set('y', 20)
    assert evolver['y'] == 20
    assert evolver._pclass_evolver_data_is_dirty
    assert 'y' in evolver._factory_fields

    # Test persistent() creates new instance with modified values
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert original.x == 1  # Original should be unchanged
    assert original.y == 2


# LLM-generated content at query #11
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    assert obj.serialize() == {'x': 1, 'y': 4}

    obj2 = TestClass(x='a', y='b')
    assert obj2.serialize() == {'x': 'a', 'y': 'bb'}

    obj3 = TestClass(x=None, y=0)
    assert obj3.serialize() == {'x': None, 'y': 0}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x.upper() if fmt == 'upper' else x)

    obj4 = TestClassWithFormat(x='hello')
    assert obj4.serialize() == {'x': 'hello'}
    assert obj4.serialize(format='upper') == {'x': 'HELLO'}

    class TestClassWithNoSerializer(PClass):
        x = field()

    obj5 = TestClassWithNoSerializer(x=123)
    assert obj5.serialize() == {'x': 123}

    class TestClassWithMissingField(PClass):
        x = field()
        y = field()

    obj6 = TestClassWithMissingField(x=1)
    assert obj6.serialize() == {'x': 1}


# LLM-generated content at query #12
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test setting with kwargs
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test setting with args
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2

    # Test with factory field
    class FactoryClass(PClass):
        e = field()
        f = field()

    obj_factory = FactoryClass(e=1, f=2)
    new_obj_factory = obj_factory.set(e=10)
    assert new_obj_factory.e == 10
    assert new_obj_factory.f == 2


# LLM-generated content at query #13
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with kwargs
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with args
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj3 = MandatoryClass(a=1, b=2)
    new_obj3 = obj3.set(b=20)
    assert new_obj3.a == 1
    assert new_obj3.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj4 = InitialClass(d=3)
    new_obj4 = obj4.set(c=10)
    assert new_obj4.c == 10
    assert new_obj4.d == 3

    # Test setting to same value
    obj5 = TestClass(x=5, y=6)
    new_obj5 = obj5.set(x=5)
    assert new_obj5 is obj5  # Should return same object if no change

    # Test with factory fields
    class FactoryClass(PClass):
        e = field(type=int)
        f = field()

    obj6 = FactoryClass(e=1, f=2)
    new_obj6 = obj6.set(e="10")  # Should use factory
    assert new_obj6.e == 10
    assert new_obj6.f == 2


# LLM-generated content at query #14
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    func, args = instance.__reduce__()

    assert func == _restore_pickle
    assert len(args) == 2
    assert args[0] == TestClass
    assert args[1] == {'x': 1, 'y': 2}

    restored = func(*args)
    assert restored == instance
    assert restored.x == 1
    assert restored.y == 2


# LLM-generated content at query #15
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y="test")
    assert repr(obj) == "TestClass(x=1, y='test')"

    obj2 = TestClass(x=None, y=0)
    assert repr(obj2) == "TestClass(x=None, y=0)"

    class EmptyClass(PClass):
        pass

    obj3 = EmptyClass()
    assert repr(obj3) == "EmptyClass()"


# LLM-generated content at query #16
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=10)
        y = field(mandatory=True)

    instance = TestClassWithDefaults(y=20)
    assert instance.x == 10
    assert instance.y == 20

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 42)
        y = field(mandatory=True)

    instance = TestClassWithCallableInitial(y=100)
    assert instance.x == 42
    assert instance.y == 100

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    instance = TestClassWithInvariant(x=1, y=2)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1, y=2)

    # Test with factory
    def factory_func(value):
        return value * 2

    class TestClassWithFactory(PClass):
        x = field(factory=factory_func)
        y = field()

    instance = TestClassWithFactory(x=5, y=10)
    assert instance.x == 10
    assert instance.y == 10

    # Test with ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassWithIgnoreExtra(x=1, y=2, z=3, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')


# LLM-generated content at query #17
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    assert instance.x == 1

    # Test with missing mandatory field
    class TestClass2(PClass):
        x = field(mandatory=True)

    try:
        TestClass2()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 1
        assert 'TestClass2.x' in e.missing_fields

    # Test with invalid field value
    class TestClass3(PClass):
        x = field(invariant=lambda x: (x > 0, "Value must be positive"))

    try:
        TestClass3(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1

    # Test with extra fields
    class TestClass4(PClass):
        x = field()

    try:
        TestClass4(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

    # Test with initial value
    class TestClass5(PClass):
        x = field(initial=5)

    instance = TestClass5()
    assert instance.x == 5

    # Test with callable initial
    class TestClass6(PClass):
        x = field(initial=lambda: 10)

    instance = TestClass6()
    assert instance.x == 10

    # Test with factory fields
    class TestClass7(PClass):
        x = field()

    instance = TestClass7._create({'x': 1}, _factory_fields={'x'})
    assert instance.x == 1

    # Test with ignore_extra
    class TestClass8(PClass):
        x = field()

    instance = TestClass8._create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


# LLM-generated content at query #18
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restore_func, args = obj.__reduce__()
    restored_obj = restore_func(*args)

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2


# LLM-generated content at query #19
#--------------------------

```python
def test_PClass___eq__():
    # Test equality with same class and same field values
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with same class but different field values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different class
    class AnotherClass(PClass):
        x = field()
        y = field()

    obj4 = AnotherClass(x=1, y=2)
    assert obj1 != obj4

    # Test with non-PClass object
    assert obj1 != "not a PClass"
    assert obj1 != 123
    assert obj1 != None

    # Test with missing fields
    obj5 = TestClass(x=1)
    obj6 = TestClass(x=1)
    assert obj5 == obj6

    # Test with different missing fields
    obj7 = TestClass(y=2)
    assert obj5 != obj7

    # Test with all fields missing
    obj8 = TestClass()
    obj9 = TestClass()
    assert obj8 == obj9


# LLM-generated content at query #20
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"

    instance_empty = TestClass(x=1)
    assert repr(instance_empty) == "TestClass(x=1)"

    class TestClassNoFields(PClass):
        pass

    instance_no_fields = TestClassNoFields()
    assert repr(instance_no_fields) == "TestClassNoFields()"


# LLM-generated content at query #21
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test setting with keyword arguments
    obj = TestPClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test setting with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError):
        obj.set(z=3)

    # Test setting with factory fields
    class TestPClassWithFactory(PClass):
        x = field()
        y = field(initial=0)

    obj_factory = TestPClassWithFactory(x=5)
    new_obj_factory = obj_factory.set(y=15)
    assert new_obj_factory.x == 5
    assert new_obj_factory.y == 15


# LLM-generated content at query #22
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestPClass(x=1, y=2)
    serialized = instance.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestPClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x + fmt)

    instance_with_format = TestPClassWithFormat(x=1)
    serialized_with_format = instance_with_format.serialize(format='_test')

    assert serialized_with_format == {'x': '1_test'}

    class TestPClassNoSerializer(PClass):
        x = field()
        y = field()

    instance_no_serializer = TestPClassNoSerializer(x=1, y=2)
    serialized_no_serializer = instance_no_serializer.serialize()

    assert serialized_no_serializer == {'x': 1, 'y': 2}


# LLM-generated content at query #23
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=2, y=1)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


# LLM-generated content at query #24
#--------------------------

```python
def test_PClass___eq__():
    # Test equality with same values
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert not (obj1 == obj3)

    # Test inequality with different types
    assert not (obj1 == 1)
    assert not (obj1 == "string")
    assert not (obj1 == None)

    # Test with missing fields
    class TestClass2(PClass):
        x = field()

    obj4 = TestClass2(x=1)
    obj5 = TestClass2(x=1)
    assert obj4 == obj5

    # Test with different field sets
    class TestClass3(PClass):
        x = field()
        y = field()
        z = field()

    obj6 = TestClass3(x=1, y=2, z=3)
    obj7 = TestClass3(x=1, y=2, z=3)
    assert obj6 == obj7

    # Test with different field values
    obj8 = TestClass3(x=1, y=2, z=4)
    assert not (obj6 == obj8)

    # Test with empty PClass
    class EmptyClass(PClass):
        pass

    obj9 = EmptyClass()
    obj10 = EmptyClass()
    assert obj9 == obj10


# LLM-generated content at query #25
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClassWithDefaults(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClassWithCallableInitial(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field(ignore_extra=True)

    instance = TestClassWithIgnoreExtra(x=1, y=2, z=3)
    assert instance.x == 1
    assert instance.y == 2

    # Test with factory_fields
    class TestClassWithFactoryFields(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactoryFields._factory_fields={'x'}, x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with invariant
    class TestClassWithInvariant(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with global invariant
    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()

        __invariant__ = lambda self: (self.x + self.y > 0, "sum must be positive")

    instance = TestClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1, y=-2)


# LLM-generated content at query #26
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    assert repr(obj) == "TestClass(x=1, y=2)"

    obj2 = TestClass(x="hello", y=None)
    assert repr(obj2) == "TestClass(x='hello', y=None)"

    class EmptyClass(PClass):
        pass

    obj3 = EmptyClass()
    assert repr(obj3) == "EmptyClass()"


# LLM-generated content at query #27
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    restore_func, args = instance.__reduce__()

    assert restore_func == _restore_pickle
    assert len(args) == 2
    assert args[0] == TestClass
    assert args[1] == {'x': 1, 'y': 2}

    restored_instance = restore_func(*args)
    assert restored_instance == instance
    assert restored_instance.x == 1
    assert restored_instance.y == 2


# LLM-generated content at query #28
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field
    try:
        TestClass(x=1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 1
        assert 'TestClass.y' in e.missing_fields

    # Test extra fields
    try:
        TestClass(x=1, y=2, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

    # Test with initial values
    class TestClassWithInitial(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClassWithInitial(y=2)
    assert instance.x == 0
    assert instance.y == 2

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClassWithCallableInitial(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassIgnoreExtra.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with factory_fields
    class TestClassFactory(PClass):
        x = field()
        y = field()

    instance = TestClassFactory._factory_fields={'x'}, x=1, y=2
    assert instance.x == 1
    assert instance.y == 2

    # Test invariant failure
    def invariant(value):
        return value > 0, "Value must be positive"

    class TestClassInvariant(PClass):
        x = field(invariant=invariant)

    try:
        TestClassInvariant(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Value must be positive" in e.invariant_errors

    # Test successful instantiation with invariant
    instance = TestClassInvariant(x=1)
    assert instance.x == 1


# LLM-generated content at query #29
#--------------------------

```python
def test_PClass___new__():
    # Test basic creation
    class TestClass(PClass):
        x = field()
        y = field(mandatory=True)

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test mandatory field missing
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test extra field provided
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test field with initial value
    class TestClassWithInitial(PClass):
        x = field(initial=10)
        y = field(mandatory=True)

    instance = TestClassWithInitial(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test field with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 20)
        y = field(mandatory=True)

    instance = TestClassWithCallableInitial(y=2)
    assert instance.x == 20
    assert instance.y == 2

    # Test factory_fields
    class TestClassWithFactory(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactory(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2

    # Test invariant failure
    class TestClassWithInvariant(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassIgnoreExtra.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, "z")

    # Test frozen attribute
    instance = TestClass(x=1, y=2)
    with pytest.raises(AttributeError):
        instance.x = 10

    # Test weakref
    class TestClassWeakRef(PClass):
        x = field()

    instance = TestClassWeakRef(x=1)
    weak_ref = weakref.ref(instance)
    assert weak_ref() is instance

    # Test global invariants
    def global_invariant(obj):
        return obj.x != obj.y, "x and y must be different"

    class TestClassGlobalInvariant(PClass):
        __invariant__ = global_invariant
        x = field()
        y = field()

    with pytest.raises(InvariantException):
        TestClassGlobalInvariant(x=1, y=1)

    # Test successful creation with global invariant
    instance = TestClassGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #30
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__slots__' in TestClass.__dict__
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'x' in TestClass.__slots__
    assert 'y' in TestClass.__slots__

    # Test __weakref__ slot for direct PClass subclass
    assert '__weakref__' in TestClass.__slots__

    # Test nested PClass
    class NestedClass(TestClass):
        z = field()

    assert hasattr(NestedClass, '_pclass_fields')
    assert hasattr(NestedClass, '_pclass_invariants')
    assert '__slots__' in NestedClass.__dict__
    assert '_pclass_frozen' in NestedClass.__slots__
    assert 'z' in NestedClass.__slots__

    # Test that nested class doesn't get __weakref__ slot
    assert '__weakref__' not in NestedClass.__slots__

    # Test field inheritance
    assert 'x' in NestedClass._pclass_fields
    assert 'y' in NestedClass._pclass_fields
    assert 'z' in NestedClass._pclass_fields

    # Test invariant inheritance
    class InvariantClass(PClass):
        __invariant__ = lambda self: (True, None)
        a = field()

    assert len(InvariantClass._pclass_invariants) == 1

    class NestedInvariantClass(InvariantClass):
        b = field()

    assert len(NestedInvariantClass._pclass_invariants) == 1


# LLM-generated content at query #31
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test setting with keyword arguments
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test setting with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        e = field(invariant=positive_invariant)
        f = field()

    obj_inv = InvariantClass(e=1, f=2)
    new_obj_inv = obj_inv.set(e=5)
    assert new_obj_inv.e == 5

    try:
        obj_inv.set(e=-1)
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestClass2(PClass):
        x = field()
        y = field(mandatory=False)

    obj4 = TestClass2(x=1)
    obj5 = TestClass2(x=1)
    assert obj4 == obj5

    obj6 = TestClass2(x=1, y=2)
    assert obj4 != obj6

    # Test with different classes
    class TestClass3(PClass):
        x = field()

    obj7 = TestClass3(x=1)
    assert obj1 != obj7


# LLM-generated content at query #33
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1  # Original unchanged

    # Test set with positional arguments
    new_instance2 = instance.set('y', 20)
    assert new_instance2.y == 20
    assert new_instance2.x == 1

    # Test setting multiple fields
    new_instance3 = instance.set(x=100, y=200)
    assert new_instance3.x == 100
    assert new_instance3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        instance.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    inst = MandatoryClass(a=1, b=2)
    new_inst = inst.set(b=20)
    assert new_inst.a == 1
    assert new_inst.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    inst2 = InitialClass(d=3)
    assert inst2.c == 0
    new_inst2 = inst2.set(c=5)
    assert new_inst2.c == 5
    assert new_inst2.d == 3

    # Test with factory field
    class FactoryClass(PClass):
        e = field(factory=lambda x: x * 2)
        f = field()

    inst3 = FactoryClass(f=4)
    assert inst3.e is None  # Factory not called without value
    new_inst3 = inst3.set(e=6)
    assert new_inst3.e == 6
    assert new_inst3.f == 4


# LLM-generated content at query #34
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestClassWithOptional(PClass):
        x = field()
        y = field(initial=0)

    obj4 = TestClassWithOptional(x=1)
    obj5 = TestClassWithOptional(x=1, y=0)
    assert obj4 == obj5

    # Test with different classes
    class AnotherClass(PClass):
        x = field()
        y = field()

    obj6 = AnotherClass(x=1, y=2)
    assert obj1 != obj6

    # Test with NotImplemented
    class CustomClass:
        def __eq__(self, other):
            return NotImplemented

    custom_obj = CustomClass()
    assert obj1 != custom_obj


# LLM-generated content at query #35
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__slots__' in TestClass.__dict__
    assert '_pclass_frozen' in TestClass.__slots__

    # Test with multiple fields
    class MultiFieldClass(PClass):
        x = field()
        y = field()

    assert len(MultiFieldClass._pclass_fields) == 2
    assert 'x' in MultiFieldClass._pclass_fields
    assert 'y' in MultiFieldClass._pclass_fields

    # Test with invariants
    class InvariantClass(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))
        __invariant__ = lambda self: (self.x < 100, "x must be less than 100")

    assert len(InvariantClass._pclass_invariants) == 1

    # Test that __weakref__ is added for direct PClass subclasses
    assert '__weakref__' in TestClass.__slots__

    # Test that __weakref__ is not added for non-direct subclasses
    class BaseClass(PClass):
        pass

    class DerivedClass(BaseClass):
        pass

    assert '__weakref__' not in DerivedClass.__slots__


# LLM-generated content at query #36
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    serialized = instance.serialize()

    assert serialized == {'x': 1, 'y': 2}

    class TestClassWithSerializer(PClass):
        x = field(serializer=lambda v: v * 2)
        y = field()

    instance_with_serializer = TestClassWithSerializer(x=1, y=2)
    serialized_with_serializer = instance_with_serializer.serialize()

    assert serialized_with_serializer == {'x': 2, 'y': 2}

    class TestClassWithFormatSerializer(PClass):
        x = field(serializer=lambda v, fmt: v * 2 if fmt == 'test' else v)
        y = field()

    instance_with_format_serializer = TestClassWithFormatSerializer(x=1, y=2)
    serialized_with_format = instance_with_format_serializer.serialize(format='test')

    assert serialized_with_format == {'x': 2, 'y': 2}

    serialized_without_format = instance_with_format_serializer.serialize()

    assert serialized_without_format == {'x': 1, 'y': 2}


# LLM-generated content at query #37
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2

    # Test with factory field
    class FactoryClass(PClass):
        e = field(factory=lambda x: x * 2)
        f = field()

    obj_factory = FactoryClass(e=5, f=2)
    assert obj_factory.e == 10
    new_obj_factory = obj_factory.set(e=3)
    assert new_obj_factory.e == 6
    assert new_obj_factory.f == 2


# LLM-generated content at query #38
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra field
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default value
    class TestClassWithDefault(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClassWithDefault(y=2)
    assert instance.x == 0
    assert instance.y == 2

    # Test with callable default
    class TestClassWithCallableDefault(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClassWithCallableDefault(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with factory fields
    class TestClassWithFactory(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactory._create({'x': 1, 'y': 2}, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

    # Test with ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassWithIgnoreExtra._create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with global invariant
    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = TestClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1, y=-2)


# LLM-generated content at query #39
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)
    serialized = instance.serialize()

    assert serialized == {'x': 1, 'y': 4}

    # Test with format parameter
    class TestClassWithFormat(PClass):
        z = field(serializer=lambda x, fmt: x.upper() if fmt == 'upper' else x)

    instance_with_format = TestClassWithFormat(z='hello')
    serialized_upper = instance_with_format.serialize(format='upper')

    assert serialized_upper == {'z': 'HELLO'}

    # Test with missing optional field
    class TestClassOptional(PClass):
        a = field()
        b = field(initial=10)

    instance_optional = TestClassOptional(a=5)
    serialized_optional = instance_optional.serialize()

    assert serialized_optional == {'a': 5, 'b': 10}

    # Test with no serializer
    class TestClassNoSerializer(PClass):
        c = field()

    instance_no_serializer = TestClassNoSerializer(c='test')
    serialized_no_serializer = instance_no_serializer.serialize()

    assert serialized_no_serializer == {'c': 'test'}


# LLM-generated content at query #40
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    serialized = obj.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x + fmt)

    obj_with_format = TestClassWithFormat(x=1)
    serialized_with_format = obj_with_format.serialize(format='_test')

    assert serialized_with_format == {'x': '1_test'}

    class TestClassNoSerializer(PClass):
        x = field()

    obj_no_serializer = TestClassNoSerializer(x=1)
    serialized_no_serializer = obj_no_serializer.serialize()

    assert serialized_no_serializer == {'x': 1}


# LLM-generated content at query #41
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestPClass(x=1, y=2)

    # Test basic serialization
    serialized = instance.serialize()
    assert serialized == {'x': 1, 'y': 4}  # y is serialized with custom serializer

    # Test with format parameter
    serialized_with_format = instance.serialize(format='json')
    assert serialized_with_format == {'x': 1, 'y': 4}

    # Test with missing optional field
    class TestPClassOptional(PClass):
        x = field()
        z = field(initial=None)

    instance_optional = TestPClassOptional(x=5)
    serialized_optional = instance_optional.serialize()
    assert serialized_optional == {'x': 5, 'z': None}

    # Test with no fields set (all optional with initial values)
    class TestPClassAllOptional(PClass):
        a = field(initial=0)
        b = field(initial="default")

    instance_all_optional = TestPClassAllOptional()
    serialized_all_optional = instance_all_optional.serialize()
    assert serialized_all_optional == {'a': 0, 'b': "default"}


# LLM-generated content at query #42
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    assert repr(obj) == "TestClass(x=1, y=2)"

    obj_empty = TestClass()
    assert repr(obj_empty) == "TestClass()"

    class TestClassWithString(PClass):
        name = field()

    obj_string = TestClassWithString(name="test")
    assert repr(obj_string) == "TestClassWithString(name='test')"


# LLM-generated content at query #43
#--------------------------

```python
def test_PClassMeta___new__():
    class TestPClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestPClass, '_pclass_fields')
    assert hasattr(TestPClass, '_pclass_invariants')
    assert '__slots__' in TestPClass.__dict__
    assert '_pclass_frozen' in TestPClass.__slots__
    assert 'x' in TestPClass.__slots__
    assert 'y' in TestPClass.__slots__
    assert '__weakref__' in TestPClass.__slots__

    class TestPClass2(TestPClass):
        z = field()

    assert hasattr(TestPClass2, '_pclass_fields')
    assert hasattr(TestPClass2, '_pclass_invariants')
    assert '__slots__' in TestPClass2.__dict__
    assert '_pclass_frozen' in TestPClass2.__slots__
    assert 'x' in TestPClass2.__slots__
    assert 'y' in TestPClass2.__slots__
    assert 'z' in TestPClass2.__slots__
    assert '__weakref__' not in TestPClass2.__slots__


# LLM-generated content at query #44
#--------------------------

```python
def test_PClass___reduce__():
    # Setup
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=10, y=20)

    # Exercise
    restore_func, args = original.__reduce__()

    # Verify
    assert restore_func == _restore_pickle
    assert len(args) == 2
    assert args[0] == TestClass
    assert args[1] == {'x': 10, 'y': 20}

    # Verify restoration
    restored = restore_func(*args)
    assert restored == original
    assert restored.x == 10
    assert restored.y == 20


# LLM-generated content at query #45
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert '__slots__' in TestClass.__dict__
    assert '_pclass_frozen' in TestClass.__slots__

    # Test with multiple fields
    class MultiFieldClass(PClass):
        a = field()
        b = field()
        c = field()

    assert len(MultiFieldClass._pclass_fields) == 3
    assert all(f in MultiFieldClass.__slots__ for f in ['a', 'b', 'c'])

    # Test with invariants
    class InvariantClass(PClass):
        x = field(invariant=lambda x: (x > 0, "Must be positive"))
        __invariant__ = lambda self: (self.x < 100, "Must be less than 100")

    assert '_pclass_invariants' in InvariantClass.__dict__
    assert len(InvariantClass._pclass_invariants) == 1

    # Test __weakref__ slot
    class WeakRefClass(PClass):
        pass

    assert '__weakref__' in WeakRefClass.__slots__

    # Test inheritance
    class BaseClass(PClass):
        base_field = field()

    class DerivedClass(BaseClass):
        derived_field = field()

    assert 'base_field' in DerivedClass._pclass_fields
    assert 'derived_field' in DerivedClass._pclass_fields


# LLM-generated content at query #46
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restored_obj = pickle.loads(pickle.dumps(obj))

    assert restored_obj.x == 1
    assert restored_obj.y == 2
    assert isinstance(restored_obj, TestClass)


# LLM-generated content at query #47
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=10)
        y = field(initial=20)

    instance = TestClassWithDefaults()
    assert instance.x == 10
    assert instance.y == 20

    # Test with factory fields
    class TestClassWithFactory(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactory._create({'x': 1, 'y': 2}, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

    # Test with ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassIgnoreExtra._create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with invariant violation
    class TestClassWithInvariant(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 42)

    instance = TestClassWithCallableInitial()
    assert instance.x == 42

    # Test with frozen attribute
    instance = TestClass(x=1, y=2)
    with pytest.raises(AttributeError):
        instance.x = 3


# LLM-generated content at query #48
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    result = obj.serialize()

    assert result == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x + fmt)

    obj_with_format = TestClassWithFormat(x=1)
    result_with_format = obj_with_format.serialize(format='test')

    assert result_with_format == {'x': '1test'}

    class TestClassNoSerializer(PClass):
        x = field()

    obj_no_serializer = TestClassNoSerializer(x=1)
    result_no_serializer = obj_no_serializer.serialize()

    assert result_no_serializer == {'x': 1}


# LLM-generated content at query #49
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restored_obj = pickle.loads(pickle.dumps(obj))

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2


# LLM-generated content at query #50
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    restore_func, args = instance.__reduce__()

    assert restore_func == _restore_pickle
    assert len(args) == 2
    assert args[0] == TestClass
    assert args[1] == {'x': 1, 'y': 2}

    restored_instance = restore_func(*args)
    assert restored_instance.x == 1
    assert restored_instance.y == 2
    assert isinstance(restored_instance, TestClass)


# LLM-generated content at query #51
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestClassWithDefault(PClass):
        x = field(initial=0)
        y = field()

    obj4 = TestClassWithDefault(y=2)
    obj5 = TestClassWithDefault(y=2)
    assert obj4 == obj5

    # Test with different classes
    class AnotherClass(PClass):
        x = field()
        y = field()

    obj6 = AnotherClass(x=1, y=2)
    assert obj1 != obj6


# LLM-generated content at query #52
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)

    # Test with different types
    class MixedClass(PClass):
        a = field()
        b = field(type=str)
        c = field(type=float)

    obj4 = MixedClass(a=10, b="test", c=3.14)
    obj5 = MixedClass(a=10, b="test", c=3.14)
    obj6 = MixedClass(a=10, b="different", c=3.14)

    assert hash(obj4) == hash(obj5)
    assert hash(obj4) != hash(obj6)

    # Test with missing optional fields
    class OptionalClass(PClass):
        required = field()
        optional = field(initial=None)

    obj7 = OptionalClass(required=1)
    obj8 = OptionalClass(required=1)
    obj9 = OptionalClass(required=1, optional="value")

    assert hash(obj7) == hash(obj8)
    assert hash(obj7) != hash(obj9)


# LLM-generated content at query #53
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restore_func, args = obj.__reduce__()
    restored_obj = restore_func(*args)

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2
    assert obj == restored_obj


# LLM-generated content at query #54
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2

    # Test with invariant
    def positive_invariant(x):
        return x > 0, "Must be positive"

    class InvariantClass(PClass):
        e = field(invariant=positive_invariant)
        f = field()

    obj_inv = InvariantClass(e=5, f=10)
    new_obj_inv = obj_inv.set(e=3)
    assert new_obj_inv.e == 3
    assert new_obj_inv.f == 10

    # Test invariant failure
    try:
        obj_inv.set(e=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #55
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda v: v * 2)

    obj = TestClass(x=1, y=2)
    assert obj.serialize() == {'x': 1, 'y': 4}

    obj = TestClass(x='a', y='b')
    assert obj.serialize() == {'x': 'a', 'y': 'bb'}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda v, fmt: v.upper() if fmt == 'upper' else v)

    obj = TestClassWithFormat(x='hello')
    assert obj.serialize() == {'x': 'hello'}
    assert obj.serialize(format='upper') == {'x': 'HELLO'}

    class TestClassWithMissing(PClass):
        x = field()
        y = field()

    obj = TestClassWithMissing(x=1)
    assert obj.serialize() == {'x': 1}


# LLM-generated content at query #56
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with kwargs
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with args
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=4)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 4

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        e = field(invariant=positive_invariant)
        f = field()

    obj_inv = InvariantClass(e=5, f=6)
    new_obj_inv = obj_inv.set(e=10)
    assert new_obj_inv.e == 10

    # Test invariant failure
    try:
        obj_inv.set(e=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with factory
    def double_factory(value):
        return value * 2

    class FactoryClass(PClass):
        g = field(factory=double_factory)
        h = field()

    obj_fact = FactoryClass(g=3, h=4)
    assert obj_fact.g == 6
    new_obj_fact = obj_fact.set(g=5)
    assert new_obj_fact.g == 10
    assert new_obj_fact.h == 4

    # Test with ignore_extra
    class IgnoreExtraClass(PClass):
        i = field(ignore_extra=True)
        j = field()

    obj_ignore = IgnoreExtraClass(i={'a': 1, 'b': 2}, j=3)
    new_obj_ignore = obj_ignore.set(i={'a': 10, 'b': 20, 'c': 30})
    assert new_obj_ignore.i == {'a': 10, 'b': 20, 'c': 30}
    assert new_obj_ignore.j == 3


# LLM-generated content at query #57
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y='test')
    assert repr(instance) == "TestClass(x=1, y='test')"

    instance_empty = TestClass(x=0, y=None)
    assert repr(instance_empty) == "TestClass(x=0, y=None)"

    class SingleField(PClass):
        value = field()

    single_instance = SingleField(value=42)
    assert repr(single_instance) == "SingleField(value=42)"


# LLM-generated content at query #58
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with kwargs
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with args
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2

    # Test with factory
    class FactoryClass(PClass):
        e = field(factory=lambda x: x * 2)
        f = field()

    obj_factory = FactoryClass(e=5, f=2)
    assert obj_factory.e == 10
    new_obj_factory = obj_factory.set(e=3)
    assert new_obj_factory.e == 6
    assert new_obj_factory.f == 2


# LLM-generated content at query #59
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restore_func, args = obj.__reduce__()

    assert restore_func == _restore_pickle
    assert len(args) == 2
    assert args[0] == TestClass
    assert args[1] == {'x': 1, 'y': 2}

    restored_obj = restore_func(*args)
    assert restored_obj.x == obj.x
    assert restored_obj.y == obj.y
    assert restored_obj == obj


# LLM-generated content at query #60
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('x', 20)
    assert new_obj2.x == 20
    assert new_obj2.y == 2

    # Test setting multiple fields
    new_obj3 = obj.set(x=30, y=40)
    assert new_obj3.x == 30
    assert new_obj3.y == 40

    # Test setting non-existent field
    try:
        obj.set(z=5)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=10)
    new_obj_init = obj_init.set(c=5)
    assert new_obj_init.c == 5
    assert new_obj_init.d == 10

    # Test with factory field
    class FactoryClass(PClass):
        e = field(factory=lambda x: x * 2)
        f = field()

    obj_fact = FactoryClass(e=3, f=4)
    new_obj_fact = obj_fact.set(e=5)
    assert new_obj_fact.e == 10  # factory doubles the value
    assert new_obj_fact.f == 4


# LLM-generated content at query #61
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestPClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError):
        obj.set(z=3)

    # Test with mandatory field
    class MandatoryPClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryPClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value
    class InitialPClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialPClass(d=4)
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 4

    # Test with type checking
    class TypedPClass(PClass):
        e = field(type=int)
        f = field()

    obj_typed = TypedPClass(e=5, f=6)
    new_obj_typed = obj_typed.set(e=50)
    assert new_obj_typed.e == 50

    with pytest.raises(TypeError):
        obj_typed.set(e="not an int")

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantPClass(PClass):
        g = field(invariant=positive_invariant)
        h = field()

    obj_inv = InvariantPClass(g=1, h=8)
    new_obj_inv = obj_inv.set(g=2)
    assert new_obj_inv.g == 2

    with pytest.raises(InvariantException):
        obj_inv.set(g=-1)

    # Test with serializer
    class SerializedPClass(PClass):
        i = field(serializer=lambda x: str(x))
        j = field()

    obj_ser = SerializedPClass(i=9, j=10)
    new_obj_ser = obj_ser.set(i=90)
    assert new_obj_ser.i == 90
    assert obj_ser.serialize()['i'] == '9'
    assert new_obj_ser.serialize()['i'] == '90'


# LLM-generated content at query #62
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestClass2(PClass):
        a = field()

    obj4 = TestClass2(a=1)
    obj5 = TestClass2(a=1)
    assert obj4 == obj5

    # Test with different number of fields
    class TestClass3(PClass):
        x = field()
        y = field()
        z = field()

    obj6 = TestClass3(x=1, y=2, z=3)
    obj7 = TestClass3(x=1, y=2, z=3)
    assert obj6 == obj7
    assert obj1 != obj6

    # Test with optional fields
    class TestClass4(PClass):
        x = field()
        y = field(initial=0)

    obj8 = TestClass4(x=1)
    obj9 = TestClass4(x=1)
    assert obj8 == obj9


# LLM-generated content at query #63
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClassWithDefaults(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClassWithCallableInitial(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()
        y = field(ignore_extra=True)

    instance = TestClassIgnoreExtra(x=1, y=2, z=3)
    assert instance.x == 1
    assert instance.y == 2

    # Test with factory_fields
    class TestClassFactoryFields(PClass):
        x = field()
        y = field()

    instance = TestClassFactoryFields._factory_fields={'x'}, x=1, y=2
    assert instance.x == 1
    assert instance.y == 2

    # Test with invariant
    def invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=invariant)

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with global invariant
    def global_invariant(instance):
        return instance.x > 0, "x must be positive"

    class TestClassWithGlobalInvariant(PClass):
        __invariant__ = global_invariant
        x = field()

    instance = TestClassWithGlobalInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1)


# LLM-generated content at query #64
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__weakref__' in TestClass.__slots__

    # Test field initialization
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test extra field
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test invariant violation
    class TestClassWithInvariant(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test default values
    class TestClassWithDefaults(PClass):
        x = field(initial=0)
        y = field(initial=lambda: "default")

    instance = TestClassWithDefaults()
    assert instance.x == 0
    assert instance.y == "default"

    # Test inheritance
    class BaseClass(PClass):
        x = field()

    class DerivedClass(BaseClass):
        y = field()

    instance = DerivedClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #65
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)
    assert instance.serialize() == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x.upper() if fmt == 'upper' else x)

    instance = TestClassWithFormat(x='hello')
    assert instance.serialize() == {'x': 'hello'}
    assert instance.serialize(format='upper') == {'x': 'HELLO'}

    class TestClassNoSerializer(PClass):
        x = field()

    instance = TestClassNoSerializer(x=1)
    assert instance.serialize() == {'x': 1}


# LLM-generated content at query #66
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    assert instance.x == 1

    # Test with multiple fields
    class MultiFieldClass(PClass):
        x = field()
        y = field()

    instance = MultiFieldClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    class MandatoryFieldClass(PClass):
        x = field(mandatory=True)

    try:
        instance = MandatoryFieldClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 1
        assert 'MandatoryFieldClass.x' in e.missing_fields

    # Test with default value
    class DefaultValueClass(PClass):
        x = field(initial=10)

    instance = DefaultValueClass()
    assert instance.x == 10

    # Test with callable default value
    class CallableDefaultClass(PClass):
        x = field(initial=lambda: 20)

    instance = CallableDefaultClass()
    assert instance.x == 20

    # Test with extra fields
    class ExtraFieldClass(PClass):
        x = field()

    try:
        instance = ExtraFieldClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

    # Test with ignore_extra=True
    instance = ExtraFieldClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

    # Test with factory fields
    class FactoryFieldClass(PClass):
        x = field()

    instance = FactoryFieldClass(x=1)
    new_instance = instance.set(x=2)
    assert new_instance.x == 2

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        x = field(invariant=positive_invariant)

    try:
        instance = InvariantClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Value must be positive" in e.invariant_errors

    # Test with global invariant
    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class GlobalInvariantClass(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    try:
        instance = GlobalInvariantClass(x=-1, y=-2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Sum must be positive" in e.invariant_errors

    # Test with frozen attribute
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

    # Test with weakref
    class WeakRefClass(PClass):
        x = field()

    instance = WeakRefClass(x=1)
    weak_ref = weakref.ref(instance)
    assert weak_ref() is instance

    # Test with pickle
    import pickle
    instance = TestClass(x=1)
    pickled = pickle.dumps(instance)
    unpickled = pickle.loads(pickled)
    assert unpickled.x == 1


# LLM-generated content at query #67
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x.upper() if isinstance(x, str) else x)

    obj = TestClass(x=10, y="hello")

    # Test basic serialization
    serialized = obj.serialize()
    assert serialized == {'x': 10, 'y': "HELLO"}

    # Test with format parameter
    serialized_with_format = obj.serialize(format='json')
    assert serialized_with_format == {'x': 10, 'y': "HELLO"}

    # Test with missing optional field
    class TestClass2(PClass):
        a = field()
        b = field(initial=42)

    obj2 = TestClass2(a=5)
    serialized2 = obj2.serialize()
    assert serialized2 == {'a': 5, 'b': 42}

    # Test with custom serializer that uses format
    class TestClass3(PClass):
        data = field(serializer=lambda x, fmt: f"{fmt}:{x}" if fmt else x)

    obj3 = TestClass3(data="test")
    assert obj3.serialize() == {'data': "test"}
    assert obj3.serialize(format='csv') == {'data': "csv:test"}


# LLM-generated content at query #68
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restored_obj = pickle.loads(pickle.dumps(obj))

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2


# LLM-generated content at query #69
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    instance = TestPClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1  # Original should be unchanged

    # Test set with positional arguments
    new_instance2 = instance.set('y', 20)
    assert new_instance2.y == 20
    assert new_instance2.x == 1

    # Test setting multiple fields
    new_instance3 = instance.set(x=100, y=200)
    assert new_instance3.x == 100
    assert new_instance3.y == 200

    # Test setting non-existent field
    try:
        instance.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryPClass(PClass):
        a = field(mandatory=True)
        b = field()

    mandatory_instance = MandatoryPClass(a=1, b=2)
    new_mandatory = mandatory_instance.set(b=20)
    assert new_mandatory.a == 1
    assert new_mandatory.b == 20

    # Test with initial value field
    class InitialPClass(PClass):
        c = field(initial=0)
        d = field()

    initial_instance = InitialPClass(d=2)
    assert initial_instance.c == 0
    new_initial = initial_instance.set(c=10)
    assert new_initial.c == 10
    assert new_initial.d == 2

    # Test with type checking
    class TypedPClass(PClass):
        e = field(type=int)
        f = field()

    typed_instance = TypedPClass(e=1, f="test")
    new_typed = typed_instance.set(e=2)
    assert new_typed.e == 2
    assert new_typed.f == "test"

    try:
        typed_instance.set(e="not an int")
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantPClass(PClass):
        g = field(invariant=positive_invariant)
        h = field()

    invariant_instance = InvariantPClass(g=1, h=2)
    new_invariant = invariant_instance.set(g=5)
    assert new_invariant.g == 5
    assert new_invariant.h == 2

    try:
        invariant_instance.set(g=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #70
#--------------------------

```python
def test_PClass___eq__():
    # Test equality between two PClass instances with same attributes
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality when attributes differ
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different class
    class OtherClass(PClass):
        x = field()

    obj4 = OtherClass(x=1)
    assert obj1 != obj4

    # Test inequality with non-PClass object
    assert obj1 != "not a PClass"

    # Test with missing attributes (should still compare correctly)
    class PartialClass(PClass):
        x = field()
        y = field(mandatory=False)

    obj5 = PartialClass(x=1)
    obj6 = PartialClass(x=1)
    assert obj5 == obj6

    # Test with different missing attributes
    obj7 = PartialClass(x=1, y=2)
    assert obj5 != obj7

    # Test with None values
    class NullableClass(PClass):
        x = field()
        y = field(mandatory=False)

    obj8 = NullableClass(x=None, y=None)
    obj9 = NullableClass(x=None, y=None)
    assert obj8 == obj9

    # Test with different None values
    obj10 = NullableClass(x=None, y=1)
    assert obj8 != obj10


# LLM-generated content at query #71
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with kwargs
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with args
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value
    class InitialClass(PClass):
        x = field(initial=0)
        y = field()

    obj_init = InitialClass(y=5)
    assert obj_init.x == 0
    new_obj_init = obj_init.set(x=10)
    assert new_obj_init.x == 10
    assert new_obj_init.y == 5

    # Test with factory
    class FactoryClass(PClass):
        x = field(factory=lambda v: v * 2)
        y = field()

    obj_fact = FactoryClass(x=3, y=4)
    assert obj_fact.x == 6
    new_obj_fact = obj_fact.set(x=5)
    assert new_obj_fact.x == 10
    assert new_obj_fact.y == 4


# LLM-generated content at query #72
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=0)
        y = field(initial=1)

    instance = TestClassWithDefaults()
    assert instance.x == 0
    assert instance.y == 1

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 0)
        y = field(initial=lambda: 1)

    instance = TestClassWithCallableInitial()
    assert instance.x == 0
    assert instance.y == 1

    # Test with ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassIgnoreExtra.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2

    # Test with factory_fields
    class TestClassFactoryFields(PClass):
        x = field()
        y = field()

    instance = TestClassFactoryFields(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

    # Test with invariant
    def invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=invariant)

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with global invariant
    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = TestClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1, y=-2)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PClassEvolver_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    evolver = original.evolver()

    # Test setting existing field
    evolver.set('x', 10)
    assert evolver['x'] == 10
    assert evolver._pclass_evolver_data_is_dirty
    assert 'x' in evolver._factory_fields

    # Test setting new field
    evolver.set('z', 30)
    assert evolver['z'] == 30
    assert evolver._pclass_evolver_data_is_dirty
    assert 'z' in evolver._factory_fields

    # Test setting same value (should not mark as dirty)
    evolver._pclass_evolver_data_is_dirty = False
    evolver.set('x', 10)
    assert not evolver._pclass_evolver_data_is_dirty

    # Test persistent after modification
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert hasattr(new_instance, 'z')
    assert new_instance.z == 30

    # Test that original is unchanged
    assert original.x == 1
    assert original.y == 2
    assert not hasattr(original, 'z')


# LLM-generated content at query #2
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestPClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('x', 20)
    assert new_obj2.x == 20
    assert new_obj2.y == 2

    # Test setting multiple fields
    new_obj3 = obj.set(x=30, y=40)
    assert new_obj3.x == 30
    assert new_obj3.y == 40

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=5)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryPClass(PClass):
        a = field(mandatory=True)
        b = field()

    mand_obj = MandatoryPClass(a=1, b=2)
    new_mand_obj = mand_obj.set(a=100)
    assert new_mand_obj.a == 100
    assert new_mand_obj.b == 2

    # Test with field that has initial value
    class InitialPClass(PClass):
        c = field(initial=0)
        d = field()

    init_obj = InitialPClass(d=5)
    assert init_obj.c == 0
    new_init_obj = init_obj.set(c=10)
    assert new_init_obj.c == 10
    assert new_init_obj.d == 5

    # Test with field that has invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantPClass(PClass):
        e = field(invariant=positive_invariant)
        f = field()

    inv_obj = InvariantPClass(e=5, f=10)
    new_inv_obj = inv_obj.set(e=15)
    assert new_inv_obj.e == 15
    assert new_inv_obj.f == 10

    # Test that invariant is checked when setting
    try:
        inv_obj.set(e=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with factory field
    class FactoryPClass(PClass):
        g = field(factory=lambda x: x * 2)
        h = field()

    fact_obj = FactoryPClass(g=3, h=4)
    assert fact_obj.g == 6  # Factory doubles the value
    new_fact_obj = fact_obj.set(g=5)
    assert new_fact_obj.g == 10  # Factory still applies
    assert new_fact_obj.h == 4


# LLM-generated content at query #3
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y="test")
    assert repr(obj) == "TestClass(x=1, y='test')"

    obj2 = TestClass(x=None, y=0)
    assert repr(obj2) == "TestClass(x=None, y=0)"

    class EmptyClass(PClass):
        pass

    obj3 = EmptyClass()
    assert repr(obj3) == "EmptyClass()"


# LLM-generated content at query #4
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test default values
    class TestClassWithDefaults(PClass):
        x = field(initial=10)
        y = field(initial=20)

    instance = TestClassWithDefaults()
    assert instance.x == 10
    assert instance.y == 20

    # Test callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 100)

    instance = TestClassWithCallableInitial()
    assert instance.x == 100

    # Test invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test factory fields
    class TestClassWithFactory(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactory(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

    # Test ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassWithIgnoreExtra.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test frozen attribute
    instance = TestClass(x=1, y=2)
    with pytest.raises(AttributeError):
        instance.x = 3

    # Test weakref
    class TestClassWithWeakref(PClass):
        x = field()

    instance = TestClassWithWeakref(x=1)
    weak_ref = weakref.ref(instance)
    assert weak_ref() is instance

    # Test global invariants
    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = TestClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1, y=-2)


# LLM-generated content at query #5
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError):
        obj.set(z=3)

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    new_m_obj = m_obj.set(b=20)
    assert new_m_obj.a == 1
    assert new_m_obj.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=5)
    assert i_obj.c == 0
    new_i_obj = i_obj.set(c=10)
    assert new_i_obj.c == 10
    assert new_i_obj.d == 5

    # Test with type checking
    class TypedClass(PClass):
        e = field(type=int)
        f = field()

    t_obj = TypedClass(e=1, f="test")
    new_t_obj = t_obj.set(e=2)
    assert new_t_obj.e == 2

    with pytest.raises(TypeError):
        t_obj.set(e="not an int")

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        g = field(invariant=positive_invariant)
        h = field()

    inv_obj = InvariantClass(g=5, h=10)
    new_inv_obj = inv_obj.set(g=3)
    assert new_inv_obj.g == 3

    with pytest.raises(InvariantException):
        inv_obj.set(g=-1)

    # Test equality after set
    obj1 = TestClass(x=1, y=2)
    obj2 = obj1.set(x=1, y=2)
    assert obj1 == obj2
    assert obj1 is not obj2

    # Test hash after set
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #6
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClassWithDefaults(y=2)
    assert instance.x == 0
    assert instance.y == 2

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = TestClassWithCallableInitial(y=2)
    assert instance.x == 42
    assert instance.y == 2

    # Test with ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassWithIgnoreExtra(x=1, y=2, ignore_extra=True, z=3)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with factory_fields
    class TestClassWithFactoryFields(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactoryFields(x=1, y=2, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    instance = TestClassWithInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1, y=2)

    # Test with global invariant
    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = TestClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1, y=2)


# LLM-generated content at query #7
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test setting with keyword arguments
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test setting with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1
    assert obj.y == 2  # Original unchanged

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2


# LLM-generated content at query #8
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with field that has initial value
    class InitialClass(PClass):
        x = field(initial=0)
        y = field()

    obj_init = InitialClass(y=5)
    assert obj_init.x == 0
    new_obj_init = obj_init.set(x=10)
    assert new_obj_init.x == 10
    assert new_obj_init.y == 5

    # Test that set returns new instance
    obj_test = TestClass(x=1, y=2)
    new_obj_test = obj_test.set(x=1)
    assert obj_test is not new_obj_test


# LLM-generated content at query #9
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__slots__' in TestClass.__dict__
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'x' in TestClass.__slots__
    assert 'y' in TestClass.__slots__

    # Test with single inheritance from PClass
    class SingleInheritance(TestClass):
        z = field()

    assert '_pclass_fields' in SingleInheritance.__dict__
    assert 'z' in SingleInheritance._pclass_fields
    assert '__weakref__' in SingleInheritance.__slots__

    # Test with multiple inheritance (should raise TypeError)
    with pytest.raises(TypeError):
        class MultipleInheritance(TestClass, PClass):
            pass

    # Test field inheritance
    assert 'x' in SingleInheritance._pclass_fields
    assert 'y' in SingleInheritance._pclass_fields

    # Test invariant storage
    def test_invariant(instance):
        return True, None

    class InvariantClass(PClass):
        __invariant__ = test_invariant
        a = field()

    assert '_pclass_invariants' in InvariantClass.__dict__
    assert test_invariant in InvariantClass._pclass_invariants


# LLM-generated content at query #10
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__

    # Test field initialization
    instance = TestClass(x=1)
    assert instance.x == 1

    # Test mandatory field
    class TestClassMandatory(PClass):
        x = field(mandatory=True)

    with pytest.raises(InvariantException):
        TestClassMandatory()

    # Test default value
    class TestClassDefault(PClass):
        x = field(initial=0)

    instance = TestClassDefault()
    assert instance.x == 0

    # Test invariant
    def invariant(x):
        return x > 0, "x must be positive"

    class TestClassInvariant(PClass):
        x = field(invariant=invariant)

    with pytest.raises(InvariantException):
        TestClassInvariant(x=-1)

    # Test multiple inheritance (should not be a PClass)
    class BaseClass:
        pass

    class TestClassMultipleInheritance(BaseClass, PClass):
        x = field()

    assert not hasattr(TestClassMultipleInheritance, '__weakref__')


# LLM-generated content at query #11
#--------------------------

```python
def test_PClass___eq__():
    # Test equality with same instance
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    assert instance1 == instance1

    # Test equality with different instances with same values
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

    # Test inequality with different values
    instance3 = TestClass(x=1, y=3)
    assert instance1 != instance3

    # Test inequality with different classes
    class AnotherClass(PClass):
        x = field()
        y = field()

    instance4 = AnotherClass(x=1, y=2)
    assert instance1 != instance4

    # Test inequality with non-PClass object
    assert instance1 != 1
    assert instance1 != "string"
    assert instance1 != None


# LLM-generated content at query #12
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert 'x' in TestClass._pclass_fields
    assert TestClass.__slots__ == ('_pclass_frozen', 'x', '__weakref__')

    # Test with multiple fields
    class MultiFieldClass(PClass):
        x = field()
        y = field()

    assert 'x' in MultiFieldClass._pclass_fields
    assert 'y' in MultiFieldClass._pclass_fields
    assert MultiFieldClass.__slots__ == ('_pclass_frozen', 'x', 'y', '__weakref__')

    # Test with invariants
    class InvariantClass(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))
        __invariant__ = lambda self: (self.x < 100, "x must be less than 100")

    assert '_pclass_invariants' in InvariantClass.__dict__
    assert InvariantClass.__slots__ == ('_pclass_frozen', 'x', '__weakref__')

    # Test without PClass base (should not add __weakref__)
    class NonPClassBase(CheckedType, metaclass=PClassMeta):
        x = field()

    assert NonPClassBase.__slots__ == ('_pclass_frozen', 'x')


# LLM-generated content at query #13
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    original = TestPClass(x=1, y=2)
    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, TestPClass)
    assert restored.x == 1
    assert restored.y == 2
    assert restored == original


# LLM-generated content at query #14
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, TestClass)
    assert restored.x == 1
    assert restored.y == 2


# LLM-generated content at query #15
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestPClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestPClass, '_pclass_fields')
    assert hasattr(TestPClass, '_pclass_invariants')
    assert hasattr(TestPClass, '__slots__')
    assert '__weakref__' in TestPClass.__slots__

    # Test field initialization
    instance = TestPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test mandatory field error
    with pytest.raises(InvariantException):
        TestPClass(x=1)

    # Test extra field error
    with pytest.raises(AttributeError):
        TestPClass(x=1, y=2, z=3)

    # Test invariant error
    class TestPClassWithInvariant(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    with pytest.raises(InvariantException):
        TestPClassWithInvariant(x=-1)

    # Test default value
    class TestPClassWithDefault(PClass):
        x = field(initial=0)
        y = field()

    instance = TestPClassWithDefault(y=1)
    assert instance.x == 0
    assert instance.y == 1

    # Test callable default
    class TestPClassWithCallableDefault(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = TestPClassWithCallableDefault(y=1)
    assert instance.x == 42
    assert instance.y == 1

    # Test ignore_extra
    class TestPClassIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestPClassIgnoreExtra.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test factory_fields
    class TestPClassFactoryFields(PClass):
        x = field()
        y = field()

    instance = TestPClassFactoryFields._factory_fields={'x'}, x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test global invariants
    def global_invariant(obj):
        return obj.x + obj.y > 0, "sum must be positive"

    class TestPClassGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    with pytest.raises(InvariantException):
        TestPClassGlobalInvariant(x=-1, y=0)


# LLM-generated content at query #16
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test extra field
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test default value
    class TestClassDefault(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClassDefault(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test callable default
    class TestClassCallable(PClass):
        x = field(initial=lambda: 20)
        y = field()

    instance = TestClassCallable(y=2)
    assert instance.x == 20
    assert instance.y == 2

    # Test invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassInvariant(PClass):
        x = field(invariant=positive_invariant)

    with pytest.raises(InvariantException):
        TestClassInvariant(x=-1)

    # Test factory
    class TestClassFactory(PClass):
        x = field(factory=lambda v: v * 2)

    instance = TestClassFactory(x=5)
    assert instance.x == 10

    # Test ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field(ignore_extra=True)

    instance = TestClassIgnoreExtra(x=1, y=2)
    assert instance.x == 1


# LLM-generated content at query #17
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    new_m_obj = m_obj.set(b=20)
    assert new_m_obj.a == 1
    assert new_m_obj.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=1)
    new_i_obj = i_obj.set(c=10)
    assert new_i_obj.c == 10
    assert new_i_obj.d == 1

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        e = field(invariant=positive_invariant)
        f = field()

    inv_obj = InvariantClass(e=5, f=10)
    new_inv_obj = inv_obj.set(e=15)
    assert new_inv_obj.e == 15
    assert new_inv_obj.f == 10

    # Test invariant violation
    try:
        inv_obj.set(e=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)
    serialized = instance.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field()
        y = field(serializer=lambda x, fmt: x * 2 if fmt == 'test' else x)

    instance_with_format = TestClassWithFormat(x=1, y=2)
    serialized_with_format = instance_with_format.serialize(format='test')

    assert serialized_with_format == {'x': 1, 'y': 4}

    serialized_without_format = instance_with_format.serialize()
    assert serialized_without_format == {'x': 1, 'y': 2}


# LLM-generated content at query #19
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restored_obj = pickle.loads(pickle.dumps(obj))

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2


# LLM-generated content at query #20
#--------------------------

```python
def test_PClassMeta___new__():
    class TestPClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestPClass, '_pclass_fields')
    assert hasattr(TestPClass, '_pclass_invariants')
    assert '__weakref__' in TestPClass.__slots__
    assert '_pclass_frozen' in TestPClass.__slots__
    assert 'x' in TestPClass.__slots__
    assert 'y' in TestPClass.__slots__

    class TestPClassNoFields(PClass):
        pass

    assert hasattr(TestPClassNoFields, '_pclass_fields')
    assert hasattr(TestPClassNoFields, '_pclass_invariants')
    assert '__weakref__' in TestPClassNoFields.__slots__
    assert '_pclass_frozen' in TestPClassNoFields.__slots__

    class TestPClassWithInvariant(PClass):
        x = field()
        __invariant__ = lambda self: (True, None)

    assert hasattr(TestPClassWithInvariant, '_pclass_invariants')
    assert len(TestPClassWithInvariant._pclass_invariants) == 1


# LLM-generated content at query #21
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__weakref__' in TestClass.__slots__

    # Test field initialization
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test extra field
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test field with default value
    class TestClassWithDefault(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClassWithDefault(y=2)
    assert instance.x == 0
    assert instance.y == 2

    # Test field with callable default
    class TestClassWithCallableDefault(PClass):
        x = field(initial=lambda: 0)
        y = field()

    instance = TestClassWithCallableDefault(y=2)
    assert instance.x == 0
    assert instance.y == 2

    # Test invariant
    class TestClassWithInvariant(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test global invariant
    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()

        @classmethod
        def __invariant__(cls, instance):
            return instance.x + instance.y > 0, "sum must be positive"

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1, y=0)

    # Test weakref slot
    import weakref
    instance = TestClass(x=1, y=2)
    ref = weakref.ref(instance)
    assert ref() is instance


# LLM-generated content at query #22
#--------------------------

```python
def test_PClass___eq__():
    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestClass2(PClass):
        x = field()
        y = field(mandatory=False)

    obj4 = TestClass2(x=1)
    obj5 = TestClass2(x=1)
    assert obj4 == obj5

    obj6 = TestClass2(x=1, y=2)
    assert obj4 != obj6

    # Test with different field names
    class TestClass3(PClass):
        a = field()
        b = field()

    obj7 = TestClass3(a=1, b=2)
    assert obj1 != obj7

    # Test with sub-classes
    class TestSubClass(TestClass):
        z = field()

    obj8 = TestSubClass(x=1, y=2, z=3)
    assert obj1 != obj8


# LLM-generated content at query #23
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    # Test that equal objects have the same hash
    assert hash(obj1) == hash(obj2)

    # Test that different objects have different hashes
    assert hash(obj1) != hash(obj3)

    # Test that hash is consistent
    assert hash(obj1) == hash(obj1)


# LLM-generated content at query #24
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    new_m_obj = m_obj.set(b=20)
    assert new_m_obj.a == 1
    assert new_m_obj.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=3)
    assert i_obj.c == 0
    new_i_obj = i_obj.set(c=10)
    assert new_i_obj.c == 10
    assert new_i_obj.d == 3

    # Test with invariant
    def positive_invariant(val):
        return val > 0, "Must be positive"

    class InvariantClass(PClass):
        e = field(invariant=positive_invariant)
        f = field()

    inv_obj = InvariantClass(e=5, f=6)
    new_inv_obj = inv_obj.set(e=10)
    assert new_inv_obj.e == 10

    try:
        inv_obj.set(e=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestPClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestPClass(x=1, y=2, z=3)

    # Test with default values
    class TestPClassWithDefaults(PClass):
        x = field(initial=10)
        y = field(initial=20)

    instance = TestPClassWithDefaults()
    assert instance.x == 10
    assert instance.y == 20

    # Test with callable initial
    class TestPClassWithCallableInitial(PClass):
        x = field(initial=lambda: 100)
        y = field(initial=lambda: 200)

    instance = TestPClassWithCallableInitial()
    assert instance.x == 100
    assert instance.y == 200

    # Test with factory fields
    class TestPClassWithFactory(PClass):
        x = field(factory=int)
        y = field(factory=str)

    instance = TestPClassWithFactory(x='1', y=2)
    assert instance.x == 1
    assert instance.y == '2'

    # Test with ignore_extra
    class TestPClassWithIgnoreExtra(PClass):
        x = field(ignore_extra=True)
        y = field(ignore_extra=True)

    instance = TestPClassWithIgnoreExtra(x=1, y=2, z=3, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2

    # Test with invariant
    class TestPClassWithInvariant(PClass):
        x = field(invariant=lambda x: (x > 0, 'x must be positive'))

    instance = TestPClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestPClassWithInvariant(x=-1)

    # Test with global invariant
    class TestPClassWithGlobalInvariant(PClass):
        x = field()
        y = field()

        @__invariant__
        def check_sum(self):
            return self.x + self.y > 0, 'sum must be positive'

    instance = TestPClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestPClassWithGlobalInvariant(x=-1, y=-2)

    # Test with frozen attribute
    instance = TestPClass(x=1, y=2)
    with pytest.raises(AttributeError):
        instance.x = 3

    # Test with weakref
    import weakref
    instance = TestPClass(x=1, y=2)
    ref = weakref.ref(instance)
    assert ref() is instance

    # Test with pickle
    import pickle
    instance = TestPClass(x=1, y=2)
    pickled = pickle.dumps(instance)
    unpickled = pickle.loads(pickled)
    assert unpickled.x == 1
    assert unpickled.y == 2


# LLM-generated content at query #26
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__

    # Test field initialization
    instance = TestClass(x=1)
    assert instance.x == 1

    # Test mandatory field
    with pytest.raises(InvariantException):
        class TestMandatory(PClass):
            x = field(mandatory=True)

        TestMandatory()

    # Test invariant failure
    with pytest.raises(InvariantException):
        class TestInvariant(PClass):
            x = field(invariant=lambda x: (x > 0, "x must be positive"))

        TestInvariant(x=-1)

    # Test extra fields
    with pytest.raises(AttributeError):
        class TestExtra(PClass):
            x = field()

        TestExtra(x=1, y=2)

    # Test ignore_extra
    class TestIgnoreExtra(PClass):
        x = field()

    instance = TestIgnoreExtra.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

    # Test global invariants
    with pytest.raises(InvariantException):
        class TestGlobalInvariant(PClass):
            x = field()

            @__invariant__
            def check_x(self):
                return self.x > 0, "x must be positive"

        TestGlobalInvariant(x=-1)

    # Test initial value
    class TestInitial(PClass):
        x = field(initial=0)

    instance = TestInitial()
    assert instance.x == 0

    # Test callable initial
    class TestCallableInitial(PClass):
        x = field(initial=lambda: 42)

    instance = TestCallableInitial()
    assert instance.x == 42

    # Test factory fields
    class TestFactory(PClass):
        x = field()

    instance = TestFactory(x=1)
    new_instance = instance.set(x=2)
    assert new_instance.x == 2
    assert instance.x == 1


# LLM-generated content at query #27
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y="test")
    assert repr(instance) == "TestPClass(x=1, y='test')"

    instance_empty = TestPClass(x=0, y=None)
    assert repr(instance_empty) == "TestPClass(x=0, y=None)"

    class SingleFieldPClass(PClass):
        single = field()

    single_instance = SingleFieldPClass(single="value")
    assert repr(single_instance) == "SingleFieldPClass(single='value')"

    class NoFieldsPClass(PClass):
        pass

    no_fields_instance = NoFieldsPClass()
    assert repr(no_fields_instance) == "NoFieldsPClass()"


# LLM-generated content at query #28
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__weakref__' in TestClass.__slots__

    # Test field setting
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test invariant failure
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestInvariantClass(PClass):
        x = field(invariant=positive_invariant)

    with pytest.raises(InvariantException):
        TestInvariantClass(x=-1)

    # Test initial values
    class TestInitialClass(PClass):
        x = field(initial=0)
        y = field(initial=lambda: "default")

    instance = TestInitialClass()
    assert instance.x == 0
    assert instance.y == "default"

    # Test inheritance
    class BaseClass(PClass):
        x = field()

    class DerivedClass(BaseClass):
        y = field()

    instance = DerivedClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #29
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test setting with keyword arguments
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test setting with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2


# LLM-generated content at query #30
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class SimplePClass(PClass):
        x = field()

    obj = SimplePClass(x=1)
    assert obj.x == 1
    assert obj._pclass_frozen is True

    # Test with mandatory field
    class MandatoryPClass(PClass):
        x = field(mandatory=True)

    obj = MandatoryPClass(x=1)
    assert obj.x == 1

    # Test missing mandatory field raises error
    with pytest.raises(InvariantException):
        MandatoryPClass()

    # Test with initial value
    class InitialPClass(PClass):
        x = field(initial=0)

    obj = InitialPClass()
    assert obj.x == 0

    # Test with callable initial
    class CallableInitialPClass(PClass):
        x = field(initial=lambda: 42)

    obj = CallableInitialPClass()
    assert obj.x == 42

    # Test with multiple fields
    class MultiFieldPClass(PClass):
        x = field()
        y = field()
        z = field(initial=10)

    obj = MultiFieldPClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 10

    # Test with extra fields raises error
    with pytest.raises(AttributeError):
        MultiFieldPClass(x=1, y=2, extra=3)

    # Test with ignore_extra
    obj = MultiFieldPClass.create({'x': 1, 'y': 2, 'extra': 3}, ignore_extra=True)
    assert obj.x == 1
    assert obj.y == 2
    assert not hasattr(obj, 'extra')

    # Test with factory fields
    class FactoryPClass(PClass):
        x = field()
        y = field()

    obj = FactoryPClass._factory_fields={'x'}, x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2

    # Test invariant failure
    class InvariantPClass(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    with pytest.raises(InvariantException):
        InvariantPClass(x=-1)

    # Test global invariant
    def global_inv(obj):
        return obj.x + obj.y > 0, "sum must be positive"

    class GlobalInvPClass(PClass):
        __invariant__ = global_inv
        x = field()
        y = field()

    obj = GlobalInvPClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2

    with pytest.raises(InvariantException):
        GlobalInvPClass(x=-1, y=-2)

    # Test with weakref slot
    class WeakRefPClass(PClass):
        x = field()

    obj = WeakRefPClass(x=1)
    assert hasattr(obj, '__weakref__')

    # Test with no weakref slot (multiple inheritance)
    class OtherBase:
        pass

    class NoWeakRefPClass(OtherBase, PClass):
        x = field()

    obj = NoWeakRefPClass(x=1)
    assert not hasattr(obj, '__weakref__')


# LLM-generated content at query #31
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)

    assert obj.serialize() == {'x': 1, 'y': 4}
    assert obj.serialize(format='test') == {'x': 1, 'y': 4}

    class TestClassWithMissing(PClass):
        a = field()
        b = field()

    obj2 = TestClassWithMissing(a=5)
    assert obj2.serialize() == {'a': 5}


# LLM-generated content at query #32
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert 'x' in TestClass._pclass_fields
    assert '__weakref__' in TestClass.__slots__

    # Test field initialization
    instance = TestClass(x=1)
    assert instance.x == 1

    # Test mandatory field
    class TestClassMandatory(PClass):
        x = field(mandatory=True)

    with pytest.raises(InvariantException):
        TestClassMandatory()

    # Test field with initial value
    class TestClassInitial(PClass):
        x = field(initial=0)

    instance = TestClassInitial()
    assert instance.x == 0

    # Test field with callable initial
    class TestClassCallableInitial(PClass):
        x = field(initial=lambda: 42)

    instance = TestClassCallableInitial()
    assert instance.x == 42

    # Test field with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassInvariant(PClass):
        x = field(invariant=positive_invariant)

    with pytest.raises(InvariantException):
        TestClassInvariant(x=-1)

    # Test field with type check
    class TestClassType(PClass):
        x = field(type=int)

    with pytest.raises(TypeError):
        TestClassType(x="not an int")

    # Test extra fields
    class TestClassExtra(PClass):
        x = field()

    with pytest.raises(AttributeError):
        TestClassExtra(x=1, y=2)

    # Test ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()

    instance = TestClassIgnoreExtra.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

    # Test factory_fields
    class TestClassFactory(PClass):
        x = field()

    instance = TestClassFactory(x=1)
    new_instance = instance.set(x=2)
    assert new_instance.x == 2
    assert instance.x == 1

    # Test global invariant
    def global_invariant(obj):
        return obj.x > 0, "x must be positive"

    class TestClassGlobalInvariant(PClass):
        __invariant__ = global_invariant
        x = field()

    with pytest.raises(InvariantException):
        TestClassGlobalInvariant(x=-1)

    # Test frozen attribute
    instance = TestClass(x=1)
    with pytest.raises(AttributeError):
        instance.x = 2

    # Test weakref
    import weakref
    instance = TestClass(x=1)
    ref = weakref.ref(instance)
    assert ref() is instance


# LLM-generated content at query #33
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        a = field()
        b = field(serializer=lambda x: x * 2)

    instance = TestClass(a=1, b=2)
    result = instance.serialize()
    assert result == {'a': 1, 'b': 4}

    class TestClassWithFormat(PClass):
        a = field(serializer=lambda x, fmt: x + fmt)

    instance = TestClassWithFormat(a=1)
    result = instance.serialize(format='test')
    assert result == {'a': '1test'}

    class TestClassNoSerializer(PClass):
        a = field()
        b = field()

    instance = TestClassNoSerializer(a=1, b=2)
    result = instance.serialize()
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #34
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test extra field
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test default value
    class TestClassWithDefault(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClassWithDefault(y=2)
    assert instance.x == 0
    assert instance.y == 2

    # Test callable default
    class TestClassWithCallableDefault(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClassWithCallableDefault(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test invariant failure
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test successful instantiation with invariant
    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    # Test factory fields
    class TestClassWithFactory(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactory(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

    # Test ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field(ignore_extra=True)

    instance = TestClassWithIgnoreExtra(x=1, y=2, z=3)
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #35
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises error
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    new_m_obj = m_obj.set(b=20)
    assert new_m_obj.a == 1
    assert new_m_obj.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=5)
    assert i_obj.c == 0
    new_i_obj = i_obj.set(c=10)
    assert new_i_obj.c == 10
    assert new_i_obj.d == 5

    # Test with invariant
    def positive_invariant(x):
        return x > 0, "Must be positive"

    class InvariantClass(PClass):
        e = field(invariant=positive_invariant)

    inv_obj = InvariantClass(e=5)
    try:
        inv_obj.set(e=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with factory
    class FactoryClass(PClass):
        f = field(factory=lambda x: x * 2)

    fac_obj = FactoryClass(f=3)
    assert fac_obj.f == 6
    new_fac_obj = fac_obj.set(f=4)
    assert new_fac_obj.f == 8


# LLM-generated content at query #36
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)
    assert isinstance(hash(obj1), int)


# LLM-generated content at query #37
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test setting with keyword arguments
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test setting with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1
    assert obj.y == 2  # Original unchanged

    # Test setting multiple fields
    new_obj3 = obj.set(x=30, y=40)
    assert new_obj3.x == 30
    assert new_obj3.y == 40

    # Test setting non-existent field
    try:
        obj.set(z=5)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=10)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=5)
    assert new_obj_init.c == 5
    assert new_obj_init.d == 10


# LLM-generated content at query #38
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    new_m_obj = m_obj.set(b=20)
    assert new_m_obj.a == 1
    assert new_m_obj.b == 20

    # Test with field that has initial value
    class InitialClass(PClass):
        x = field(initial=0)
        y = field()

    i_obj = InitialClass(y=5)
    new_i_obj = i_obj.set(x=10)
    assert new_i_obj.x == 10
    assert new_i_obj.y == 5

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Must be positive"

    class InvariantClass(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    inv_obj = InvariantClass(x=1, y=2)
    new_inv_obj = inv_obj.set(x=5)
    assert new_inv_obj.x == 5

    try:
        inv_obj.set(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with factory
    def double_factory(value):
        return value * 2

    class FactoryClass(PClass):
        x = field(factory=double_factory)
        y = field()

    f_obj = FactoryClass(x=1, y=2)
    new_f_obj = f_obj.set(x=3)
    assert new_f_obj.x == 6  # Factory doubles the value
    assert new_f_obj.y == 2

    # Test equality after set
    obj1 = TestClass(x=1, y=2)
    obj2 = obj1.set(x=1, y=2)
    assert obj1 == obj2
    assert obj1 is not obj2

    # Test hash after set
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #39
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestPClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError):
        obj.set(z=3)

    # Test with mandatory field
    class MandatoryPClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryPClass(a=1, b=2)
    m_new = m_obj.set(b=20)
    assert m_new.a == 1
    assert m_new.b == 20

    # Test with initial value field
    class InitialPClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialPClass(d=5)
    assert i_obj.c == 0
    i_new = i_obj.set(c=10)
    assert i_new.c == 10
    assert i_new.d == 5

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Must be positive"

    class InvariantPClass(PClass):
        e = field(invariant=positive_invariant)

    inv_obj = InvariantPClass(e=5)
    inv_new = inv_obj.set(e=10)
    assert inv_new.e == 10

    with pytest.raises(InvariantException):
        inv_obj.set(e=-1)

    # Test with factory
    def double_factory(value):
        return value * 2

    class FactoryPClass(PClass):
        f = field(factory=double_factory)

    fac_obj = FactoryPClass(f=5)
    assert fac_obj.f == 10  # Factory doubles the value
    fac_new = fac_obj.set(f=3)
    assert fac_new.f == 6


# LLM-generated content at query #40
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__weakref__' in TestClass.__slots__

    # Test field inheritance
    class ParentClass(PClass):
        x = field()

    class ChildClass(ParentClass):
        y = field()

    assert 'x' in ChildClass._pclass_fields
    assert 'y' in ChildClass._pclass_fields
    assert '__weakref__' not in ChildClass.__slots__  # Only in top-level class

    # Test invariant storage
    def test_invariant(instance):
        return True, "OK"

    class InvariantClass(PClass):
        __invariant__ = test_invariant
        x = field()

    assert InvariantClass._pclass_invariants == (test_invariant,)

    # Test slots creation
    class SlotsClass(PClass):
        a = field()
        b = field()

    expected_slots = ('_pclass_frozen', 'a', 'b', '__weakref__')
    assert SlotsClass.__slots__ == expected_slots

    # Test multiple inheritance (should fail)
    try:
        class BadClass(PClass, object):
            x = field()
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #41
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)
    assert hash(obj1) == hash(TestClass(x=1, y=2))


# LLM-generated content at query #42
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    serialized = obj.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x.upper() if fmt == 'upper' else x)

    obj_with_format = TestClassWithFormat(x='hello')
    assert obj_with_format.serialize() == {'x': 'hello'}
    assert obj_with_format.serialize(format='upper') == {'x': 'HELLO'}

    class TestClassWithMissing(PClass):
        x = field()
        y = field()

    obj_missing = TestClassWithMissing(x=1)
    assert obj_missing.serialize() == {'x': 1}


# LLM-generated content at query #43
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field, invariant

    # Test basic instantiation
    class SimplePClass(PClass):
        x = field()
        y = field()

    instance = SimplePClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with mandatory field
    class MandatoryPClass(PClass):
        x = field(mandatory=True)
        y = field()

    instance = MandatoryPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field raises error
    with pytest.raises(InvariantException):
        MandatoryPClass(y=2)

    # Test with initial value
    class InitialPClass(PClass):
        x = field(initial=0)
        y = field()

    instance = InitialPClass(y=2)
    assert instance.x == 0
    assert instance.y == 2

    # Test with callable initial
    class CallableInitialPClass(PClass):
        x = field(initial=lambda: 0)
        y = field()

    instance = CallableInitialPClass(y=2)
    assert instance.x == 0
    assert instance.y == 2

    # Test with invariant
    def check_positive(value):
        return value > 0, "Value must be positive"

    class InvariantPClass(PClass):
        x = field(invariant=check_positive)
        y = field()

    instance = InvariantPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test failing invariant
    with pytest.raises(InvariantException):
        InvariantPClass(x=-1, y=2)

    # Test with extra fields raises error
    with pytest.raises(AttributeError):
        SimplePClass(x=1, y=2, z=3)

    # Test with ignore_extra
    class IgnoreExtraPClass(PClass):
        x = field()
        y = field()

    instance = IgnoreExtraPClass.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2

    # Test with factory_fields
    class FactoryPClass(PClass):
        x = field()
        y = field()

    instance = FactoryPClass(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

    # Test with global invariant
    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class GlobalInvariantPClass(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = GlobalInvariantPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test failing global invariant
    with pytest.raises(InvariantException):
        GlobalInvariantPClass(x=-1, y=-2)


# LLM-generated content at query #44
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestPClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryPClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryPClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialPClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialPClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2

    # Test with factory fields
    class FactoryPClass(PClass):
        e = field()
        f = field()

    obj_factory = FactoryPClass(e=1, f=2)
    new_obj_factory = obj_factory.set(e=10)
    assert new_obj_factory.e == 10
    assert new_obj_factory.f == 2


# LLM-generated content at query #45
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    new_m_obj = m_obj.set(b=20)
    assert new_m_obj.a == 1
    assert new_m_obj.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        x = field(initial=0)
        y = field()

    i_obj = InitialClass(y=5)
    assert i_obj.x == 0
    new_i_obj = i_obj.set(x=10)
    assert new_i_obj.x == 10
    assert new_i_obj.y == 5

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Must be positive"

    class InvariantClass(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    inv_obj = InvariantClass(x=5, y=10)
    new_inv_obj = inv_obj.set(x=3)
    assert new_inv_obj.x == 3

    try:
        inv_obj.set(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

    # Test with factory
    def double_factory(value):
        return value * 2

    class FactoryClass(PClass):
        x = field(factory=double_factory)
        y = field()

    f_obj = FactoryClass(x=5, y=10)
    assert f_obj.x == 10  # 5 * 2
    new_f_obj = f_obj.set(x=3)
    assert new_f_obj.x == 6  # 3 * 2

    # Test equality after set
    obj1 = TestClass(x=1, y=2)
    obj2 = obj1.set(x=1, y=2)
    assert obj1 == obj2
    assert hash(obj1) == hash(obj2)

    # Test inequality after set
    obj3 = obj1.set(x=3)
    assert obj1 != obj3


# LLM-generated content at query #46
#--------------------------

```python
def test_PClass___new__():
    # Test basic creation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=10)
        y = field(initial=20)

    instance = TestClassWithDefaults()
    assert instance.x == 10
    assert instance.y == 20

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 100)

    instance = TestClassWithCallableInitial()
    assert instance.x == 100

    # Test with ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()
        y = field(ignore_extra=True)

    instance = TestClassIgnoreExtra(x=1, y=2, z=3, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2

    # Test with factory_fields
    class TestClassFactoryFields(PClass):
        x = field()
        y = field()

    instance = TestClassFactoryFields(x=1, y=2, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

    # Test with invariant
    def invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=invariant)

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with global invariant
    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = TestClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1, y=-2)


# LLM-generated content at query #47
#--------------------------

```python
def test_PClass___eq__():
    class TestPClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestPClass(x=1, y=2)
    obj2 = TestPClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestPClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestPClassOptional(PClass):
        x = field()
        y = field(initial=0)

    obj4 = TestPClassOptional(x=1)
    obj5 = TestPClassOptional(x=1, y=0)
    assert obj4 == obj5

    # Test with different number of fields
    class TestPClassExtra(PClass):
        x = field()
        y = field()
        z = field()

    obj6 = TestPClass(x=1, y=2)
    obj7 = TestPClassExtra(x=1, y=2, z=3)
    assert obj6 != obj7

    # Test with same object
    assert obj1 == obj1


# LLM-generated content at query #48
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    assert repr(obj) == "TestClass(x=1, y=2)"

    obj_empty = TestClass()
    assert repr(obj_empty) == "TestClass()"

    class TestClassWithString(PClass):
        name = field(type=str)

    obj_str = TestClassWithString(name="test")
    assert repr(obj_str) == "TestClassWithString(name='test')"

    class TestClassWithNone(PClass):
        value = field()

    obj_none = TestClassWithNone(value=None)
    assert repr(obj_none) == "TestClassWithNone(value=None)"


# LLM-generated content at query #49
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda v: v * 2)

    instance = TestClass(x=1, y=2)
    serialized = instance.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field()
        y = field(serializer=lambda v, fmt: v * 2 if fmt == 'double' else v)

    instance_with_format = TestClassWithFormat(x=1, y=2)
    serialized_with_format = instance_with_format.serialize(format='double')

    assert serialized_with_format == {'x': 1, 'y': 4}

    serialized_without_format = instance_with_format.serialize()
    assert serialized_without_format == {'x': 1, 'y': 2}


# LLM-generated content at query #50
#--------------------------

```python
def test_PClass___eq__():
    # Test basic equality
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with different classes
    class AnotherClass(PClass):
        x = field()
        y = field()

    obj4 = AnotherClass(x=1, y=2)
    assert obj1 != obj4

    # Test with missing fields
    class PartialClass(PClass):
        x = field()
        y = field(mandatory=False)

    obj5 = PartialClass(x=1)
    obj6 = PartialClass(x=1)
    assert obj5 == obj6

    obj7 = PartialClass(x=1, y=2)
    assert obj5 != obj7

    # Test with custom types
    class CustomClass(PClass):
        x = field(type=int)
        y = field(type=str)

    obj8 = CustomClass(x=1, y="test")
    obj9 = CustomClass(x=1, y="test")
    assert obj8 == obj9

    obj10 = CustomClass(x=2, y="test")
    assert obj8 != obj10


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__PClassEvolver_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    evolver = original.evolver()

    # Test setting existing field
    evolver.set('x', 10)
    assert evolver['x'] == 10
    assert evolver._pclass_evolver_data_is_dirty

    # Test setting new field
    evolver.set('z', 30)
    assert evolver['z'] == 30
    assert 'z' in evolver._factory_fields

    # Test setting same value doesn't mark as dirty
    evolver.set('y', 2)
    assert not evolver._pclass_evolver_data_is_dirty

    # Test persistent creates new instance
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert hasattr(new_instance, 'z')
    assert new_instance.z == 30

    # Test original remains unchanged
    assert original.x == 1
    assert original.y == 2
    assert not hasattr(original, 'z')


# LLM-generated content at query #2
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test setting with keyword arguments
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test setting with positional arguments
    new_obj2 = obj.set('x', 20)
    assert new_obj2.x == 20
    assert new_obj2.y == 2

    # Test setting multiple fields
    new_obj3 = obj.set(x=30, y=40)
    assert new_obj3.x == 30
    assert new_obj3.y == 40

    # Test setting non-existent field
    try:
        obj.set(z=5)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=100)
        d = field()

    obj_init = InitialClass(d=2)
    assert obj_init.c == 100
    new_obj_init = obj_init.set(c=200)
    assert new_obj_init.c == 200
    assert new_obj_init.d == 2


# LLM-generated content at query #3
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"

    instance_empty = TestClass()
    assert repr(instance_empty) == "TestClass()"

    class TestClassWithString(PClass):
        name = field()

    instance_string = TestClassWithString(name="test")
    assert repr(instance_string) == "TestClassWithString(name='test')"


# LLM-generated content at query #4
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field, invariant

    # Test basic instantiation
    class SimplePClass(PClass):
        x = field()
        y = field()

    instance = SimplePClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with mandatory field
    class MandatoryPClass(PClass):
        x = field(mandatory=True)
        y = field()

    instance = MandatoryPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field raises error
    try:
        MandatoryPClass(y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 1
        assert 'MandatoryPClass.x' in e.missing_fields

    # Test with initial value
    class InitialPClass(PClass):
        x = field(initial=10)
        y = field()

    instance = InitialPClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with callable initial
    class CallableInitialPClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = CallableInitialPClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with invariant
    def positive_invariant(inst, field, value):
        return value > 0, "Value must be positive"

    class InvariantPClass(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    instance = InvariantPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test invariant failure
    try:
        InvariantPClass(x=-1, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Value must be positive" in e.invariant_errors

    # Test with extra fields raises error
    try:
        SimplePClass(x=1, y=2, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

    # Test with ignore_extra
    class IgnoreExtraPClass(PClass):
        x = field()
        y = field()

    instance = IgnoreExtraPClass.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with factory_fields
    class FactoryPClass(PClass):
        x = field()
        y = field()

    instance = FactoryPClass(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

    # Test frozen attribute
    instance = SimplePClass(x=1, y=2)
    assert instance._pclass_frozen is True

    # Test setting attribute after frozen raises error
    try:
        instance.x = 3
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

    # Test with global invariant
    def global_invariant(inst):
        return inst.x + inst.y > 0, "Sum must be positive"

    class GlobalInvariantPClass(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = GlobalInvariantPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test global invariant failure
    try:
        GlobalInvariantPClass(x=-1, y=-2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Sum must be positive" in e.invariant_errors


# LLM-generated content at query #5
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with kwargs
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with args
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2

    # Test with factory field
    class FactoryClass(PClass):
        e = field(factory=lambda x: x * 2)
        f = field()

    obj_factory = FactoryClass(e=5, f=2)
    new_obj_factory = obj_factory.set(e=10)
    assert new_obj_factory.e == 20  # Factory doubles the value
    assert new_obj_factory.f == 2

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Must be positive"

    class InvariantClass(PClass):
        g = field(invariant=positive_invariant)
        h = field()

    obj_inv = InvariantClass(g=5, h=2)
    new_obj_inv = obj_inv.set(g=10)
    assert new_obj_inv.g == 10

    try:
        obj_inv.set(g=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    # Test that equal objects have the same hash
    assert hash(obj1) == hash(obj2)

    # Test that different objects have different hashes
    assert hash(obj1) != hash(obj3)

    # Test that hash is consistent
    assert hash(obj1) == hash(obj1)


# LLM-generated content at query #7
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=10)
        y = field(initial=20)

    instance = TestClassWithDefaults()
    assert instance.x == 10
    assert instance.y == 20

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 100)
        y = field(initial=lambda: 200)

    instance = TestClassWithCallableInitial()
    assert instance.x == 100
    assert instance.y == 200

    # Test with factory fields
    class TestClassWithFactory(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactory(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2

    # Test with ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassWithIgnoreExtra.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    instance = TestClassWithInvariant(x=1, y=2)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1, y=2)

    # Test with global invariant
    def sum_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = sum_invariant

    instance = TestClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1, y=-2)


# LLM-generated content at query #8
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test that missing fields are preserved
    partial_obj = TestClass(x=5)
    new_partial = partial_obj.set(y=15)
    assert new_partial.x == 5
    assert new_partial.y == 15

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    mand_obj = MandatoryClass(a=1, b=2)
    new_mand = mand_obj.set(b=20)
    assert new_mand.a == 1
    assert new_mand.b == 20

    # Test that set doesn't allow new fields
    try:
        obj.set(z=30)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    init_obj = InitialClass(d=10)
    new_init = init_obj.set(c=5)
    assert new_init.c == 5
    assert new_init.d == 10

    # Test equality after set
    obj1 = TestClass(x=1, y=2)
    obj2 = obj1.set(x=1, y=2)
    assert obj1 == obj2
    assert obj1 is not obj2

    # Test with factory fields
    class FactoryClass(PClass):
        e = field()
        f = field()

    fact_obj = FactoryClass(e=1, f=2)
    new_fact = fact_obj.set(e=10)
    assert isinstance(new_fact, FactoryClass)
    assert new_fact.e == 10
    assert new_fact.f == 2


# LLM-generated content at query #9
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__slots__' in TestClass.__dict__
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'x' in TestClass.__slots__

    # Test inheritance
    class ParentClass(PClass):
        y = field()

    class ChildClass(ParentClass):
        z = field()

    assert hasattr(ChildClass, '_pclass_fields')
    assert hasattr(ChildClass, '_pclass_invariants')
    assert 'y' in ChildClass._pclass_fields
    assert 'z' in ChildClass._pclass_fields
    assert '__weakref__' in ChildClass.__slots__

    # Test field initialization
    instance = TestClass(x=1)
    assert instance.x == 1
    assert instance._pclass_frozen

    # Test missing mandatory field
    with pytest.raises(InvariantException):
        TestClass()

    # Test extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, extra=2)

    # Test invariants
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        a = field(invariant=positive_invariant)

    with pytest.raises(InvariantException):
        InvariantClass(a=-1)

    # Test weakref slot
    class WeakRefClass(PClass):
        b = field()

    instance = WeakRefClass(b=1)
    weak_ref = weakref.ref(instance)
    assert weak_ref() is instance


# LLM-generated content at query #10
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)
    assert hash(obj1) == hash(TestClass(x=1, y=2))


# LLM-generated content at query #11
#--------------------------

```python
def test_PClass___eq__():
    # Test equality with same values
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different types
    assert obj1 != 42
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestClass2(PClass):
        x = field()
        y = field(mandatory=False)

    obj4 = TestClass2(x=1)
    obj5 = TestClass2(x=1)
    assert obj4 == obj5

    obj6 = TestClass2(x=1, y=2)
    assert obj4 != obj6

    # Test with different classes
    class TestClass3(PClass):
        x = field()
        z = field()

    obj7 = TestClass3(x=1, z=2)
    assert obj1 != obj7


# LLM-generated content at query #12
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=30, y=40)
    assert new_obj3.x == 30
    assert new_obj3.y == 40

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=100)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    m_new = m_obj.set(b=20)
    assert m_new.a == 1
    assert m_new.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=5)
    assert i_obj.c == 0
    i_new = i_obj.set(c=10)
    assert i_new.c == 10
    assert i_new.d == 5


# LLM-generated content at query #13
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


# LLM-generated content at query #14
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=1)
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 1

    # Test that set returns new instance
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=5)
    assert obj is not new_obj
    assert isinstance(new_obj, TestClass)


# LLM-generated content at query #15
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)
    serialized = instance.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x.upper() if fmt == 'upper' else x)

    instance = TestClassWithFormat(x='hello')
    serialized = instance.serialize(format='upper')

    assert serialized == {'x': 'HELLO'}

    class TestClassWithNoSerializer(PClass):
        x = field()

    instance = TestClassWithNoSerializer(x='test')
    serialized = instance.serialize()

    assert serialized == {'x': 'test'}


# LLM-generated content at query #16
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestClassWithOptional(PClass):
        x = field()
        y = field(initial=0)

    obj4 = TestClassWithOptional(x=1)
    obj5 = TestClassWithOptional(x=1, y=0)
    assert obj4 == obj5

    # Test with different classes
    class AnotherClass(PClass):
        x = field()
        y = field()

    obj6 = AnotherClass(x=1, y=2)
    assert obj1 != obj6


# LLM-generated content at query #17
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=0)
        y = field(initial=1)

    instance = TestClassWithDefaults()
    assert instance.x == 0
    assert instance.y == 1

    # Test with factory fields
    class TestClassWithFactory(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactory._create({'x': 1, 'y': 2}, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

    # Test with ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassWithIgnoreExtra._create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with invariant
    def invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=invariant)

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 0)

    instance = TestClassWithCallableInitial()
    assert instance.x == 0


# LLM-generated content at query #18
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError):
        obj.set(z=3)

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2

    # Test with type checking
    class TypedClass(PClass):
        e = field(type=int)
        f = field()

    obj_typed = TypedClass(e=1, f="test")
    new_obj_typed = obj_typed.set(e=2)
    assert new_obj_typed.e == 2

    with pytest.raises(TypeError):
        obj_typed.set(e="not an int")

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        g = field(invariant=positive_invariant)
        h = field()

    obj_inv = InvariantClass(g=1, h=2)
    new_obj_inv = obj_inv.set(g=5)
    assert new_obj_inv.g == 5

    with pytest.raises(InvariantException):
        obj_inv.set(g=-1)

    # Test with factory
    def double_factory(value):
        return value * 2

    class FactoryClass(PClass):
        i = field(factory=double_factory)
        j = field()

    obj_fact = FactoryClass(i=1, j=2)
    assert obj_fact.i == 2  # Factory doubles the value
    new_obj_fact = obj_fact.set(i=3)
    assert new_obj_fact.i == 6  # Factory doubles the new value


# LLM-generated content at query #19
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field, optional, invariant

    class TestClass(PClass):
        x = field()
        y = field(type=int, serializer=lambda x: x * 2)
        z = field(type=str, serializer=lambda s: s.upper())

    instance = TestClass(x=1, y=5, z="hello")

    # Test basic serialization
    serialized = instance.serialize()
    assert serialized == {'x': 1, 'y': 10, 'z': 'HELLO'}

    # Test with missing optional field
    class TestClassOptional(PClass):
        a = field()
        b = optional(int)

    instance_optional = TestClassOptional(a=10)
    serialized_optional = instance_optional.serialize()
    assert serialized_optional == {'a': 10}

    # Test with format parameter
    class TestClassFormat(PClass):
        name = field(serializer=lambda x, fmt: f"{fmt}:{x}" if fmt else x)

    instance_format = TestClassFormat(name="test")
    assert instance_format.serialize() == {'name': 'test'}
    assert instance_format.serialize(format="prefix") == {'name': 'prefix:test'}

    # Test with None values
    class TestClassNone(PClass):
        value = field()

    instance_none = TestClassNone(value=None)
    serialized_none = instance_none.serialize()
    assert serialized_none == {'value': None}

    # Test empty serialization
    class EmptyClass(PClass):
        pass

    empty_instance = EmptyClass()
    assert empty_instance.serialize() == {}


# LLM-generated content at query #20
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test setting a field with keyword argument
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1  # Original unchanged

    # Test setting a field with positional arguments
    new_instance2 = instance.set('x', 20)
    assert new_instance2.x == 20
    assert new_instance2.y == 2

    # Test setting multiple fields
    new_instance3 = instance.set(x=30, y=40)
    assert new_instance3.x == 30
    assert new_instance3.y == 40

    # Test setting non-existent field raises AttributeError
    try:
        instance.set(z=5)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    inst = MandatoryClass(a=1, b=2)
    new_inst = inst.set(a=10)
    assert new_inst.a == 10
    assert new_inst.b == 2

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    inst_initial = InitialClass(d=5)
    assert inst_initial.c == 0
    new_inst_initial = inst_initial.set(c=10)
    assert new_inst_initial.c == 10
    assert new_inst_initial.d == 5

    # Test with factory field
    class FactoryClass(PClass):
        e = field(factory=lambda x: x * 2)
        f = field()

    inst_factory = FactoryClass(e=3, f=6)
    assert inst_factory.e == 6  # 3 * 2
    new_inst_factory = inst_factory.set(e=4)
    assert new_inst_factory.e == 8  # 4 * 2
    assert new_inst_factory.f == 6


# LLM-generated content at query #21
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    mand_obj = MandatoryClass(a=1, b=2)
    new_mand = mand_obj.set(b=20)
    assert new_mand.a == 1
    assert new_mand.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    init_obj = InitialClass(d=5)
    assert init_obj.c == 0
    new_init = init_obj.set(c=10)
    assert new_init.c == 10
    assert new_init.d == 5

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Must be positive"

    class InvariantClass(PClass):
        e = field(invariant=positive_invariant)
        f = field()

    inv_obj = InvariantClass(e=5, f=10)
    new_inv = inv_obj.set(e=10)
    assert new_inv.e == 10

    try:
        inv_obj.set(e=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with factory
    def double_factory(value):
        return value * 2

    class FactoryClass(PClass):
        g = field(factory=double_factory)
        h = field()

    fact_obj = FactoryClass(g=5, h=10)
    assert fact_obj.g == 10
    new_fact = fact_obj.set(g=3)
    assert new_fact.g == 6

    # Test equality after set
    obj1 = TestClass(x=1, y=2)
    obj2 = obj1.set(x=1, y=2)
    assert obj1 == obj2
    assert hash(obj1) == hash(obj2)

    # Test inequality after set
    obj3 = obj1.set(x=3)
    assert obj1 != obj3


# LLM-generated content at query #22
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.x == 1
    assert new_obj2.y == 20

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2

    # Test with type checking
    class TypedClass(PClass):
        e = field(type=int)
        f = field()

    obj_typed = TypedClass(e=1, f=2)
    try:
        obj_typed.set(e="not an int")
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with invariant
    def positive_invariant(x):
        return x > 0, "Must be positive"

    class InvariantClass(PClass):
        g = field(invariant=positive_invariant)
        h = field()

    obj_inv = InvariantClass(g=1, h=2)
    try:
        obj_inv.set(g=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with serializer
    def custom_serializer(x):
        return str(x)

    class SerializerClass(PClass):
        i = field(serializer=custom_serializer)
        j = field()

    obj_ser = SerializerClass(i=1, j=2)
    new_obj_ser = obj_ser.set(i=10)
    assert new_obj_ser.serialize() == {'i': '10', 'j': 2}

    # Test with factory
    def double_factory(x):
        return x * 2

    class FactoryClass(PClass):
        k = field(factory=double_factory)
        l = field()

    obj_fact = FactoryClass(k=1, l=2)
    new_obj_fact = obj_fact.set(k=5)
    assert new_obj_fact.k == 10  # Factory doubles the value

    # Test with ignore_extra
    class IgnoreExtraClass(PClass):
        m = field(ignore_extra=True)
        n = field()

    obj_ignore = IgnoreExtraClass(m={'a': 1, 'b': 2}, n=3)
    new_obj_ignore = obj_ignore.set(m={'a': 10, 'b': 20, 'c': 30})
    assert new_obj_ignore.m == {'a': 10, 'b': 20, 'c': 30}


# LLM-generated content at query #23
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test default values
    class TestClassWithDefaults(PClass):
        x = field(initial=0)
        y = field(initial=1)

    instance = TestClassWithDefaults()
    assert instance.x == 0
    assert instance.y == 1

    # Test callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 42)

    instance = TestClassWithCallableInitial()
    assert instance.x == 42

    # Test invariant failure
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test factory fields
    class TestClassWithFactory(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactory(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

    # Test ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field(ignore_extra=True)

    instance = TestClassWithIgnoreExtra(x=1, y={'a': 1, 'b': 2})
    assert instance.x == 1
    assert instance.y == {'a': 1, 'b': 2}

    # Test global invariants
    def global_invariant(instance):
        return instance.x + instance.y == 10, "Sum must be 10"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = TestClassWithGlobalInvariant(x=3, y=7)
    assert instance.x == 3
    assert instance.y == 7

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=1, y=2)

    # Test frozen attribute
    instance = TestClass(x=1, y=2)
    with pytest.raises(AttributeError):
        instance.x = 3


# LLM-generated content at query #24
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field, invariant

    # Test basic PClass creation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with mandatory field
    class TestMandatory(PClass):
        x = field(mandatory=True)
        y = field()

    instance = TestMandatory(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field raises error
    try:
        TestMandatory(y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 1
        assert 'TestMandatory.x' in e.missing_fields

    # Test with default value
    class TestDefault(PClass):
        x = field(initial=10)
        y = field()

    instance = TestDefault(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with callable default
    class TestCallableDefault(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestCallableDefault(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test extra fields raise error
    try:
        TestClass(x=1, y=2, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

    # Test with ignore_extra
    instance = TestClass.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with invariant
    def check_positive(value):
        return value > 0, "Value must be positive"

    class TestInvariant(PClass):
        x = field(invariant=check_positive)
        y = field()

    instance = TestInvariant(x=1, y=2)
    assert instance.x == 1

    # Test invariant failure
    try:
        TestInvariant(x=-1, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Value must be positive" in e.invariant_errors

    # Test with factory fields
    class TestFactory(PClass):
        x = field()
        y = field()

    instance = TestFactory(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

    # Test frozen attribute
    instance = TestClass(x=1, y=2)
    try:
        instance.x = 3
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

    # Test global invariant
    @invariant
    def global_inv(obj):
        return obj.x + obj.y > 0, "Sum must be positive"

    class TestGlobalInvariant(PClass):
        x = field()
        y = field()

    instance = TestGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    try:
        TestGlobalInvariant(x=-1, y=-2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Sum must be positive" in e.invariant_errors


# LLM-generated content at query #25
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    assert instance.x == 1

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass()

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2)

    # Test with initial value
    class TestClassWithInitial(PClass):
        x = field(initial=0)

    instance = TestClassWithInitial()
    assert instance.x == 0

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 42)

    instance = TestClassWithCallableInitial()
    assert instance.x == 42

    # Test with factory fields
    class TestClassWithFactory(PClass):
        x = field()

    instance = TestClassWithFactory._create({'x': 1}, _factory_fields={'x'})
    assert instance.x == 1

    # Test with ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()

    instance = TestClassWithIgnoreExtra._create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

    # Test with invariant
    def invariant(x):
        return x > 0, "x must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=invariant)

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with global invariant
    def global_invariant(self):
        return self.x > 0, "x must be positive"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        __invariant__ = global_invariant

    instance = TestClassWithGlobalInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1)


# LLM-generated content at query #26
#--------------------------

```python
def test_PClass___eq__():
    # Test basic equality
    class TestClass(PClass):
        x = field()
        y = field()

    a = TestClass(x=1, y=2)
    b = TestClass(x=1, y=2)
    c = TestClass(x=3, y=4)

    assert a == b
    assert not (a == c)
    assert not (b == c)

    # Test with different types
    assert not (a == 1)
    assert not (a == "string")
    assert not (a == None)

    # Test with missing fields
    class TestClass2(PClass):
        x = field()
        y = field(mandatory=False)

    d = TestClass2(x=1)
    e = TestClass2(x=1)
    f = TestClass2(x=1, y=2)

    assert d == e
    assert not (d == f)
    assert not (e == f)

    # Test with different field values
    g = TestClass(x=1, y=2)
    h = TestClass(x=1, y=3)

    assert not (g == h)

    # Test with NotImplemented
    assert a.__eq__(None) is NotImplemented


# LLM-generated content at query #27
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    func, args = obj.__reduce__()

    assert func == _restore_pickle
    assert len(args) == 2
    assert args[0] == TestClass
    assert args[1] == {'x': 1, 'y': 2}

    restored_obj = func(*args)
    assert restored_obj.x == 1
    assert restored_obj.y == 2
    assert restored_obj == obj


# LLM-generated content at query #28
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)
    serialized = instance.serialize()

    assert serialized == {'x': 1, 'y': 4}

    # Test with format parameter
    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x * 3 if fmt == 'special' else x)

    instance2 = TestClassWithFormat(x=5)
    assert instance2.serialize() == {'x': 5}
    assert instance2.serialize(format='special') == {'x': 15}

    # Test with missing optional field
    class TestClassOptional(PClass):
        x = field()
        y = field(initial=10)

    instance3 = TestClassOptional(x=3)
    assert instance3.serialize() == {'x': 3, 'y': 10}

    # Test with no fields
    class EmptyClass(PClass):
        pass

    instance4 = EmptyClass()
    assert instance4.serialize() == {}


# LLM-generated content at query #29
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restored_obj = pickle.loads(pickle.dumps(obj))

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2


# LLM-generated content at query #30
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    serialized = obj.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x.upper() if fmt == 'upper' else x)

    obj2 = TestClassWithFormat(x='hello')
    serialized2 = obj2.serialize(format='upper')

    assert serialized2 == {'x': 'HELLO'}

    class TestClassNoSerializer(PClass):
        x = field()

    obj3 = TestClassNoSerializer(x=10)
    serialized3 = obj3.serialize()

    assert serialized3 == {'x': 10}


# LLM-generated content at query #31
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestPClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryPClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryPClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialPClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialPClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2

    # Test with factory field
    class FactoryPClass(PClass):
        e = field(factory=lambda x: x * 2)
        f = field()

    obj_factory = FactoryPClass(e=5, f=2)
    assert obj_factory.e == 10  # Factory doubles the value
    new_obj_factory = obj_factory.set(e=3)
    assert new_obj_factory.e == 6  # Factory still applies


# LLM-generated content at query #32
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with kwargs
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with args
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    new_m_obj = m_obj.set(b=20)
    assert new_m_obj.a == 1
    assert new_m_obj.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=1)
    new_i_obj = i_obj.set(c=10)
    assert new_i_obj.c == 10
    assert new_i_obj.d == 1

    # Test with type checking
    class TypedClass(PClass):
        e = field(type=int)
        f = field()

    t_obj = TypedClass(e=1, f="test")
    new_t_obj = t_obj.set(e=2)
    assert new_t_obj.e == 2
    assert new_t_obj.f == "test"

    try:
        t_obj.set(e="not an int")
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        g = field(invariant=positive_invariant)
        h = field()

    inv_obj = InvariantClass(g=1, h=2)
    new_inv_obj = inv_obj.set(g=5)
    assert new_inv_obj.g == 5
    assert new_inv_obj.h == 2

    try:
        inv_obj.set(g=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #33
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y="test")
    assert repr(instance) == "TestClass(x=1, y='test')"

    instance_empty = TestClass()
    assert repr(instance_empty) == "TestClass()"

    class AnotherClass(PClass):
        a = field()
        b = field(initial=10)

    instance_with_initial = AnotherClass(a=5)
    assert repr(instance_with_initial) == "AnotherClass(a=5, b=10)"


# LLM-generated content at query #34
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    # Test basic serialization
    obj = TestPClass(x=1, y=2)
    serialized = obj.serialize()
    assert serialized == {'x': 1, 'y': 4}  # y is serialized with custom serializer

    # Test with missing optional field
    class TestPClassOptional(PClass):
        x = field()
        y = field(initial=0)

    obj_optional = TestPClassOptional(x=5)
    serialized_optional = obj_optional.serialize()
    assert serialized_optional == {'x': 5, 'y': 0}

    # Test with format parameter
    class TestPClassFormat(PClass):
        x = field(serializer=lambda x, fmt: str(x) if fmt == 'str' else x)

    obj_format = TestPClassFormat(x=10)
    serialized_format = obj_format.serialize(format='str')
    assert serialized_format == {'x': '10'}

    # Test empty serialization
    class EmptyPClass(PClass):
        pass

    empty_obj = EmptyPClass()
    assert empty_obj.serialize() == {}


# LLM-generated content at query #35
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    assert repr(obj) == "TestClass(x=1, y=2)"

    obj_empty = TestClass()
    assert repr(obj_empty) == "TestClass()"

    class TestClassWithString(PClass):
        name = field(type=str)

    obj_str = TestClassWithString(name="test")
    assert repr(obj_str) == "TestClassWithString(name='test')"

    class TestClassWithNone(PClass):
        value = field()

    obj_none = TestClassWithNone(value=None)
    assert repr(obj_none) == "TestClassWithNone(value=None)"


# LLM-generated content at query #36
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restore_func, args = obj.__reduce__()
    restored_obj = restore_func(*args)

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2


# LLM-generated content at query #37
#--------------------------

```python
def test_PClassMeta___new__():
    class TestPClass(PClass):
        x = field()
        y = field(initial=0)

    assert hasattr(TestPClass, '_pclass_fields')
    assert hasattr(TestPClass, '_pclass_invariants')
    assert hasattr(TestPClass, '__slots__')
    assert '_pclass_frozen' in TestPClass.__slots__
    assert 'x' in TestPClass.__slots__
    assert 'y' in TestPClass.__slots__
    assert '__weakref__' in TestPClass.__slots__

    instance = TestPClass(x=1)
    assert instance.x == 1
    assert instance.y == 0
    assert instance._pclass_frozen is True

    with pytest.raises(AttributeError):
        instance.x = 2

    with pytest.raises(AttributeError):
        TestPClass(z=1)


# LLM-generated content at query #38
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field

    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default value
    class TestClassWithDefault(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClassWithDefault(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClassWithCallableInitial(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassIgnoreExtra(x=1, y=2, ignore_extra=True, z=3)
    assert instance.x == 1
    assert instance.y == 2

    # Test with factory_fields
    class TestClassFactoryFields(PClass):
        x = field()
        y = field()

    instance = TestClassFactoryFields(x=1, y=2, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

    # Test with invariant
    def invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=invariant)
        y = field()

    instance = TestClassWithInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1, y=2)

    # Test with global invariant
    def global_invariant(instance):
        return instance.x != instance.y, "x and y must be different"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = TestClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=1, y=1)


# LLM-generated content at query #39
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)

    # Test basic serialization
    assert obj.serialize() == {'x': 1, 'y': 4}

    # Test serialization with format
    assert obj.serialize(format='json') == {'x': 1, 'y': 4}

    # Test with missing field
    class TestClass2(PClass):
        a = field()
        b = field()

    obj2 = TestClass2(a=5)
    assert obj2.serialize() == {'a': 5}

    # Test with no fields
    class EmptyClass(PClass):
        pass

    empty_obj = EmptyClass()
    assert empty_obj.serialize() == {}


# LLM-generated content at query #40
#--------------------------

```python
def test_PClass___eq__():
    # Test basic equality
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different class
    class AnotherClass(PClass):
        x = field()
        y = field()

    obj4 = AnotherClass(x=1, y=2)
    assert obj1 != obj4

    # Test with missing fields
    obj5 = TestClass(x=1)
    obj6 = TestClass(x=1)
    assert obj5 == obj6

    # Test with NotImplemented for non-PClass objects
    assert obj1 != "not a PClass"
    assert obj1 != 42
    assert obj1 != None


# LLM-generated content at query #41
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field, invariant

    # Test basic PClass creation
    class SimpleClass(PClass):
        x = field()

    obj = SimpleClass(x=1)
    assert obj.x == 1
    assert obj._pclass_frozen is True

    # Test with multiple fields
    class MultiFieldClass(PClass):
        x = field()
        y = field()

    obj = MultiFieldClass(x=1, y=2)
    assert obj.x == 1
    assert obj.y == 2

    # Test with mandatory field
    class MandatoryFieldClass(PClass):
        x = field(mandatory=True)

    obj = MandatoryFieldClass(x=1)
    assert obj.x == 1

    # Test missing mandatory field raises error
    try:
        MandatoryFieldClass()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 1
        assert 'MandatoryFieldClass.x' in e.missing_fields

    # Test with default value
    class DefaultValueClass(PClass):
        x = field(initial=0)

    obj = DefaultValueClass()
    assert obj.x == 0

    # Test with callable initial
    class CallableInitialClass(PClass):
        x = field(initial=lambda: 42)

    obj = CallableInitialClass()
    assert obj.x == 42

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        x = field(invariant=positive_invariant)

    obj = InvariantClass(x=1)
    assert obj.x == 1

    # Test invariant failure
    try:
        InvariantClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Value must be positive" in e.invariant_errors

    # Test with extra fields raises error
    try:
        SimpleClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "y" in str(e)
        assert "SimpleClass" in str(e)

    # Test with ignore_extra
    class IgnoreExtraClass(PClass):
        x = field()

    obj = IgnoreExtraClass(x=1, y=2, ignore_extra=True)
    assert obj.x == 1
    assert not hasattr(obj, 'y')

    # Test with factory_fields
    class FactoryFieldClass(PClass):
        x = field()

    obj = FactoryFieldClass(x=1, _factory_fields={'x'})
    assert obj.x == 1

    # Test with global invariant
    def global_invariant(instance):
        return instance.x > 0, "x must be positive"

    class GlobalInvariantClass(PClass):
        x = field()
        __invariant__ = global_invariant

    obj = GlobalInvariantClass(x=1)
    assert obj.x == 1

    # Test global invariant failure
    try:
        GlobalInvariantClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "x must be positive" in e.invariant_errors


# LLM-generated content at query #42
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestPClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('x', 20)
    assert new_obj2.x == 20
    assert new_obj2.y == 2

    # Test setting multiple fields
    new_obj3 = obj.set(x=30, y=40)
    assert new_obj3.x == 30
    assert new_obj3.y == 40

    # Test setting non-existent field
    try:
        obj.set(z=5)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryPClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mandatory = MandatoryPClass(a=1, b=2)
    new_obj_mandatory = obj_mandatory.set(b=20)
    assert new_obj_mandatory.a == 1
    assert new_obj_mandatory.b == 20

    # Test with initial value field
    class InitialPClass(PClass):
        c = field(initial=0)
        d = field()

    obj_initial = InitialPClass(d=1)
    new_obj_initial = obj_initial.set(c=10)
    assert new_obj_initial.c == 10
    assert new_obj_initial.d == 1

    # Test with factory field
    class FactoryPClass(PClass):
        e = field(factory=lambda x: x * 2)
        f = field()

    obj_factory = FactoryPClass(f=1)
    new_obj_factory = obj_factory.set(e=5)
    assert new_obj_factory.e == 10  # Factory doubles the value
    assert new_obj_factory.f == 1


# LLM-generated content at query #43
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestPClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestPClass, '_pclass_fields')
    assert hasattr(TestPClass, '_pclass_invariants')
    assert '__weakref__' in TestPClass.__slots__

    # Test field initialization
    instance = TestPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field
    with pytest.raises(InvariantException):
        TestPClass(x=1)

    # Test extra fields
    with pytest.raises(AttributeError):
        TestPClass(x=1, y=2, z=3)

    # Test field invariants
    class TestPClassWithInvariant(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    with pytest.raises(InvariantException):
        TestPClassWithInvariant(x=-1)

    # Test global invariants
    class TestPClassWithGlobalInvariant(PClass):
        x = field()
        y = field()

        @__invariant__
        def check_sum(self):
            return self.x + self.y > 0, "sum must be positive"

    with pytest.raises(InvariantException):
        TestPClassWithGlobalInvariant(x=-1, y=-2)

    # Test initial values
    class TestPClassWithInitial(PClass):
        x = field(initial=0)
        y = field(initial=lambda: 1)

    instance = TestPClassWithInitial()
    assert instance.x == 0
    assert instance.y == 1

    # Test factory fields
    class TestPClassWithFactory(PClass):
        x = field()
        y = field()

    instance = TestPClassWithFactory(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

    # Test weakref slot
    import weakref
    instance = TestPClass(x=1, y=2)
    ref = weakref.ref(instance)
    assert ref() is instance


# LLM-generated content at query #44
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)

    obj4 = TestClass(x=1, y=2)
    assert hash(obj1) == hash(obj4)

    class TestClass2(PClass):
        a = field()
        b = field()

    obj5 = TestClass2(a=1, b=2)
    assert hash(obj1) != hash(obj5)


# LLM-generated content at query #45
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"

    instance_empty = TestClass(x=1)
    assert repr(instance_empty) == "TestClass(x=1)"

    class TestClassWithString(PClass):
        name = field()

    instance_string = TestClassWithString(name="test")
    assert repr(instance_string) == "TestClassWithString(name='test')"


# LLM-generated content at query #46
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1  # Original unchanged

    # Test set with positional arguments
    new_instance2 = instance.set('x', 20)
    assert new_instance2.x == 20
    assert new_instance2.y == 2

    # Test setting multiple fields
    new_instance3 = instance.set(x=30, y=40)
    assert new_instance3.x == 30
    assert new_instance3.y == 40

    # Test setting non-existent field raises AttributeError
    try:
        instance.set(z=5)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    inst = MandatoryClass(a=1, b=2)
    new_inst = inst.set(b=20)
    assert new_inst.a == 1
    assert new_inst.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    inst2 = InitialClass(d=3)
    assert inst2.c == 0
    new_inst2 = inst2.set(c=10)
    assert new_inst2.c == 10
    assert new_inst2.d == 3


# LLM-generated content at query #47
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    assert obj.serialize() == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x.upper() if fmt == 'upper' else x)

    obj2 = TestClassWithFormat(x='hello')
    assert obj2.serialize() == {'x': 'hello'}
    assert obj2.serialize(format='upper') == {'x': 'HELLO'}

    class TestClassNoSerializer(PClass):
        x = field()

    obj3 = TestClassNoSerializer(x=1)
    assert obj3.serialize() == {'x': 1}


# LLM-generated content at query #48
#--------------------------

```python
def test_PClass___repr__():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    assert repr(instance) == "TestPClass(x=1, y=2)"

    instance_empty = TestPClass()
    assert repr(instance_empty) == "TestPClass()"

    class TestPClassWithString(PClass):
        name = field(type=str)

    instance_string = TestPClassWithString(name="test")
    assert repr(instance_string) == "TestPClassWithString(name='test')"


# LLM-generated content at query #49
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    serialized = obj.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field()
        y = field(serializer=lambda x, fmt: x * 2 if fmt == 'test' else x)

    obj2 = TestClassWithFormat(x=1, y=2)
    serialized2 = obj2.serialize(format='test')

    assert serialized2 == {'x': 1, 'y': 4}

    serialized3 = obj2.serialize(format='other')
    assert serialized3 == {'x': 1, 'y': 2}


# LLM-generated content at query #50
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda v: v * 2)

    obj = TestClass(x=1, y=2)
    serialized = obj.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda v, fmt: v + fmt)

    obj_with_format = TestClassWithFormat(x=1)
    serialized_with_format = obj_with_format.serialize(format='test')

    assert serialized_with_format == {'x': '1test'}

    class TestClassWithNoneSerializer(PClass):
        x = field(serializer=None)

    obj_none_serializer = TestClassWithNoneSerializer(x=1)
    serialized_none = obj_none_serializer.serialize()

    assert serialized_none == {'x': 1}


# LLM-generated content at query #51
#--------------------------

```python
def test_PClass___repr__():
    class TestPClass(PClass):
        x = field()
        y = field()

    obj = TestPClass(x=1, y='test')
    assert repr(obj) == "TestPClass(x=1, y='test')"

    obj_empty = TestPClass(x=1, y=None)
    assert repr(obj_empty) == "TestPClass(x=1, y=None)"

    class EmptyPClass(PClass):
        pass

    empty_obj = EmptyPClass()
    assert repr(empty_obj) == "EmptyPClass()"


# LLM-generated content at query #52
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with identical instances
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

    # Test inequality with different values
    instance3 = TestClass(x=1, y=3)
    assert instance1 != instance3

    # Test inequality with different types
    assert instance1 != 1
    assert instance1 != "string"
    assert instance1 != None

    # Test with missing fields
    class TestClassWithOptional(PClass):
        x = field()
        y = field(initial=0)

    instance4 = TestClassWithOptional(x=1)
    instance5 = TestClassWithOptional(x=1, y=0)
    assert instance4 == instance5

    # Test with different field values
    instance6 = TestClassWithOptional(x=1, y=1)
    assert instance4 != instance6

    # Test with different class types
    class AnotherClass(PClass):
        x = field()
        y = field()

    instance7 = AnotherClass(x=1, y=2)
    assert instance1 != instance7


# LLM-generated content at query #53
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field, invariant

    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test mandatory field
    class TestMandatory(PClass):
        mandatory_field = field(mandatory=True)

    try:
        TestMandatory()
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "TestMandatory.mandatory_field" in str(e)

    # Test default value
    class TestDefault(PClass):
        default_field = field(initial=42)

    instance = TestDefault()
    assert instance.default_field == 42

    # Test callable default
    class TestCallableDefault(PClass):
        callable_default = field(initial=lambda: "default")

    instance = TestCallableDefault()
    assert instance.callable_default == "default"

    # Test invariant
    def positive_invariant(inst, val):
        return val > 0, "Value must be positive"

    class TestInvariant(PClass):
        positive_field = field(invariant=positive_invariant)

    try:
        TestInvariant(positive_field=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Value must be positive" in str(e)

    # Test extra fields
    class TestExtraFields(PClass):
        x = field()

    try:
        TestExtraFields(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

    # Test ignore_extra
    instance = TestExtraFields.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "y")

    # Test factory_fields
    class TestFactory(PClass):
        x = field()
        y = field()

    instance = TestFactory(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2

    # Test frozen attribute
    instance = TestClass(x=1, y=2)
    try:
        instance.x = 10
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test global invariant
    @invariant
    def global_invariant(inst):
        return inst.x + inst.y > 0, "Sum must be positive"

    class TestGlobalInvariant(PClass):
        x = field()
        y = field()

    try:
        TestGlobalInvariant(x=-1, y=-2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Sum must be positive" in str(e)


# LLM-generated content at query #54
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y="test")
    assert repr(obj) == "TestClass(x=1, y='test')"

    obj2 = TestClass(x=None, y=2.5)
    assert repr(obj2) == "TestClass(x=None, y=2.5)"

    class EmptyClass(PClass):
        pass

    empty_obj = EmptyClass()
    assert repr(empty_obj) == "EmptyClass()"


# LLM-generated content at query #55
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    assert obj.serialize() == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x + fmt)

    obj = TestClassWithFormat(x=1)
    assert obj.serialize('a') == {'x': '1a'}


# LLM-generated content at query #56
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    assert instance.x == 1

    # Test with missing mandatory field
    class TestClass2(PClass):
        x = field(mandatory=True)

    try:
        TestClass2()
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ('TestClass2.x',)

    # Test with invalid field value
    class TestClass3(PClass):
        x = field(invariant=lambda x: (x > 0, "Must be positive"))

    try:
        TestClass3(x=-1)
        assert False, "Should raise InvariantException"
    except InvariantException as e:
        assert e.invariant_errors == ("Must be positive",)

    # Test with extra fields
    class TestClass4(PClass):
        x = field()

    try:
        TestClass4(x=1, y=2)
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

    # Test with default value
    class TestClass5(PClass):
        x = field(initial=5)

    instance = TestClass5()
    assert instance.x == 5

    # Test with callable initial
    class TestClass6(PClass):
        x = field(initial=lambda: 10)

    instance = TestClass6()
    assert instance.x == 10

    # Test with factory_fields
    class TestClass7(PClass):
        x = field()

    instance = TestClass7._create({'x': 1}, _factory_fields={'x'})
    assert instance.x == 1

    # Test with ignore_extra
    class TestClass8(PClass):
        x = field()

    instance = TestClass8._create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

    # Test with global invariant
    def global_inv(obj):
        if obj.x != obj.y:
            raise ValueError("x and y must be equal")

    class TestClass9(PClass):
        __invariant__ = global_inv
        x = field()
        y = field()

    try:
        TestClass9(x=1, y=2)
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass

    instance = TestClass9(x=1, y=1)
    assert instance.x == 1
    assert instance.y == 1


# LLM-generated content at query #57
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test setting with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test setting with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1
    assert obj.y == 2  # Original unchanged

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError):
        obj.set(z=3)

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2


# LLM-generated content at query #58
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field, s

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)
    assert instance.serialize() == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x.upper() if fmt == 'upper' else x.lower())

    instance = TestClassWithFormat(x='hello')
    assert instance.serialize(format='upper') == {'x': 'HELLO'}
    assert instance.serialize(format='lower') == {'x': 'hello'}

    class TestClassNoSerializer(PClass):
        x = field()

    instance = TestClassNoSerializer(x=1)
    assert instance.serialize() == {'x': 1}


# LLM-generated content at query #59
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restore_func, args = obj.__reduce__()
    restored_obj = restore_func(*args)

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2
    assert obj == restored_obj


# LLM-generated content at query #60
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restore_func, args = obj.__reduce__()
    restored_obj = restore_func(*args)

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2
    assert restored_obj == obj


# LLM-generated content at query #61
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassDefaults(PClass):
        x = field(initial=10)
        y = field(initial=20)

    instance = TestClassDefaults()
    assert instance.x == 10
    assert instance.y == 20

    # Test with factory fields
    class TestClassFactory(PClass):
        x = field()
        y = field()

    instance = TestClassFactory(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

    # Test with ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassIgnoreExtra.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with invariant
    class TestClassInvariant(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    instance = TestClassInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassInvariant(x=-1)

    # Test with global invariant
    class TestClassGlobalInvariant(PClass):
        x = field()
        y = field()

        @invariant
        def check_sum(self):
            return self.x + self.y > 0, "sum must be positive"

    instance = TestClassGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassGlobalInvariant(x=-1, y=-2)

    # Test with weakref
    class TestClassWeakRef(PClass):
        x = field()

    instance = TestClassWeakRef(x=1)
    weak_ref = weakref.ref(instance)
    assert weak_ref() is instance

    # Test with frozen attribute
    instance = TestClass(x=1, y=2)
    with pytest.raises(AttributeError):
        instance.x = 3

    # Test with pickle
    instance = TestClass(x=1, y=2)
    pickled = pickle.dumps(instance)
    unpickled = pickle.loads(pickled)
    assert unpickled.x == 1
    assert unpickled.y == 2


# LLM-generated content at query #62
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)

    assert obj.serialize() == {'x': 1, 'y': 4}
    assert obj.serialize(format='test') == {'x': 1, 'y': 4}

    class TestClassWithMissing(PClass):
        a = field()
        b = field()

    obj_missing = TestClassWithMissing(a=10)
    assert obj_missing.serialize() == {'a': 10}


# LLM-generated content at query #63
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestClassWithOptional(PClass):
        x = field()
        y = field(initial=0)

    obj4 = TestClassWithOptional(x=1)
    obj5 = TestClassWithOptional(x=1, y=0)
    assert obj4 == obj5

    # Test with different number of fields
    class TestClassExtended(TestClass):
        z = field(initial=0)

    obj6 = TestClassExtended(x=1, y=2)
    assert obj1 != obj6


# LLM-generated content at query #64
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    restore_func, args = instance.__reduce__()
    restored_instance = restore_func(*args)

    assert isinstance(restored_instance, TestClass)
    assert restored_instance.x == 1
    assert restored_instance.y == 2
    assert restored_instance == instance


# LLM-generated content at query #65
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra field
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with initial value
    class TestClassWithInitial(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClassWithInitial(y=2)
    assert instance.x == 0
    assert instance.y == 2

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClassWithCallableInitial(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassWithIgnoreExtra(x=1, y=2, z=3, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with factory fields
    class TestClassWithFactory(PClass):
        x = field()
        y = field()

    instance = TestClassWithFactory(x=1, y=2, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    instance = TestClassWithInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1, y=2)

    # Test with global invariant
    def global_invariant(obj):
        return obj.x + obj.y > 0, "Sum must be positive"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = TestClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=-1, y=-2)


# LLM-generated content at query #66
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)

    # Test basic serialization
    serialized = instance.serialize()
    assert serialized == {'x': 1, 'y': 4}

    # Test with custom format
    serialized_format = instance.serialize(format='json')
    assert serialized_format == {'x': 1, 'y': 4}

    # Test with missing optional field
    class TestClassOptional(PClass):
        x = field()
        y = field(initial=5)

    instance_optional = TestClassOptional(x=1)
    serialized_optional = instance_optional.serialize()
    assert serialized_optional == {'x': 1, 'y': 10}

    # Test with no fields
    class EmptyClass(PClass):
        pass

    empty_instance = EmptyClass()
    serialized_empty = empty_instance.serialize()
    assert serialized_empty == {}


# LLM-generated content at query #67
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=0)
        y = field(initial=lambda: "default")

    instance = TestClassWithDefaults()
    assert instance.x == 0
    assert instance.y == "default"

    # Test with field invariants
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with factory fields
    class TestClassWithFactory(PClass):
        x = field()

    instance = TestClassWithFactory(x=1)
    new_instance = instance.set(x=2)
    assert new_instance.x == 2
    assert instance.x == 1

    # Test with ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()

    instance = TestClassWithIgnoreExtra.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "y")

    # Test with global invariants
    def global_invariant(instance):
        return instance.x != instance.y, "x and y must be different"

    class TestClassWithGlobalInvariant(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = TestClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassWithGlobalInvariant(x=1, y=1)


# LLM-generated content at query #68
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restored_func, restored_args = obj.__reduce__()
    restored_obj = restored_func(*restored_args)

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2


# LLM-generated content at query #69
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field, optional, invariant

    class TestClass(PClass):
        x = field()
        y = field(type=int)
        z = field(serializer=lambda x: x.upper() if isinstance(x, str) else x)

    # Test basic serialization
    obj = TestClass(x=1, y=2, z='hello')
    serialized = obj.serialize()
    assert serialized == {'x': 1, 'y': 2, 'z': 'HELLO'}

    # Test with missing optional field
    class OptionalClass(PClass):
        a = field()
        b = optional(field(type=str))

    obj2 = OptionalClass(a=10)
    serialized2 = obj2.serialize()
    assert serialized2 == {'a': 10}

    # Test with custom format
    class FormatClass(PClass):
        data = field(serializer=lambda x, fmt: x * 2 if fmt == 'double' else x)

    obj3 = FormatClass(data=5)
    assert obj3.serialize() == {'data': 5}
    assert obj3.serialize(format='double') == {'data': 10}

    # Test empty serialization
    class EmptyClass(PClass):
        pass

    obj4 = EmptyClass()
    assert obj4.serialize() == {}


# LLM-generated content at query #70
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestPClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestPClass, '_pclass_fields')
    assert hasattr(TestPClass, '_pclass_invariants')
    assert '__weakref__' in TestPClass.__slots__

    # Test field initialization
    instance = TestPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test mandatory field error
    with pytest.raises(InvariantException):
        TestPClass()

    # Test extra field error
    with pytest.raises(AttributeError):
        TestPClass(x=1, y=2, z=3)

    # Test invariant error
    class TestPClassWithInvariant(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    with pytest.raises(InvariantException):
        TestPClassWithInvariant(x=-1)

    # Test weakref slot
    class TestPClassWithWeakref(PClass):
        x = field()

    instance = TestPClassWithWeakref(x=1)
    assert '__weakref__' in TestPClassWithWeakref.__slots__

    # Test inheritance
    class ParentPClass(PClass):
        x = field()

    class ChildPClass(ParentPClass):
        y = field()

    child_instance = ChildPClass(x=1, y=2)
    assert child_instance.x == 1
    assert child_instance.y == 2


# LLM-generated content at query #71
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, TestClass)
    assert restored.x == 1
    assert restored.y == 2
    assert restored == original


# LLM-generated content at query #72
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        a = field()
        b = field(serializer=lambda x: x * 2)

    instance = TestClass(a=1, b=2)
    serialized = instance.serialize()

    assert serialized == {'a': 1, 'b': 4}

    class TestClassWithFormat(PClass):
        a = field(serializer=lambda x, fmt: x + fmt)

    instance_with_format = TestClassWithFormat(a=5)
    serialized_with_format = instance_with_format.serialize(format='_test')

    assert serialized_with_format == {'a': '5_test'}

    class TestClassWithMissing(PClass):
        a = field()
        b = field()

    instance_missing = TestClassWithMissing(a=10)
    serialized_missing = instance_missing.serialize()

    assert serialized_missing == {'a': 10}


# LLM-generated content at query #73
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestClassWithOptional(PClass):
        x = field()
        y = field(initial=0)

    obj4 = TestClassWithOptional(x=1)
    obj5 = TestClassWithOptional(x=1, y=0)
    assert obj4 == obj5

    # Test with different classes
    class AnotherClass(PClass):
        x = field()
        y = field()

    obj6 = AnotherClass(x=1, y=2)
    assert obj1 != obj6

    # Test with NotImplemented
    assert obj1 != []
    assert obj1 != {}


# LLM-generated content at query #74
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(b=20)
    assert new_obj_mand.a == 1
    assert new_obj_mand.b == 20

    # Test with field that has initial value
    class InitialClass(PClass):
        x = field(initial=0)
        y = field()

    obj_init = InitialClass(y=5)
    assert obj_init.x == 0
    new_obj_init = obj_init.set(x=10)
    assert new_obj_init.x == 10
    assert new_obj_init.y == 5

    # Test with field that has invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    obj_inv = InvariantClass(x=1, y=2)
    new_obj_inv = obj_inv.set(x=5)
    assert new_obj_inv.x == 5

    try:
        obj_inv.set(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with field that has factory
    class FactoryClass(PClass):
        x = field(factory=lambda v: v * 2)
        y = field()

    obj_fact = FactoryClass(x=1, y=2)
    assert obj_fact.x == 2
    new_obj_fact = obj_fact.set(x=3)
    assert new_obj_fact.x == 6


# LLM-generated content at query #75
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    obj = TestPClass(x=1, y=2)
    restore_func, args = obj.__reduce__()

    assert restore_func == _restore_pickle
    assert args[0] == TestPClass
    assert args[1] == {'x': 1, 'y': 2}

    restored_obj = restore_func(*args)
    assert restored_obj.x == 1
    assert restored_obj.y == 2
    assert isinstance(restored_obj, TestPClass)


# LLM-generated content at query #76
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__weakref__' in TestClass.__slots__

    # Test field initialization
    instance = TestClass(x=1)
    assert instance.x == 1

    # Test mandatory field
    with pytest.raises(InvariantException):
        class TestClass2(PClass):
            x = field(mandatory=True)

        TestClass2()

    # Test invariant
    with pytest.raises(InvariantException):
        class TestClass3(PClass):
            x = field(invariant=lambda x: (x > 0, "x must be positive"))

        TestClass3(x=-1)

    # Test extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2)

    # Test inheritance
    class BaseClass(PClass):
        x = field()

    class DerivedClass(BaseClass):
        y = field()

    derived = DerivedClass(x=1, y=2)
    assert derived.x == 1
    assert derived.y == 2

    # Test weakref slot
    assert hasattr(derived, '__weakref__')


# LLM-generated content at query #77
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__slots__' in TestClass.__dict__
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'x' in TestClass.__slots__

    # Test with multiple fields
    class MultiFieldClass(PClass):
        a = field()
        b = field()
        c = field()

    assert 'a' in MultiFieldClass.__slots__
    assert 'b' in MultiFieldClass.__slots__
    assert 'c' in MultiFieldClass.__slots__

    # Test inheritance
    class ParentClass(PClass):
        parent_field = field()

    class ChildClass(ParentClass):
        child_field = field()

    assert 'parent_field' in ChildClass.__slots__
    assert 'child_field' in ChildClass.__slots__
    assert '__weakref__' in ChildClass.__slots__

    # Test with invariants
    class InvariantClass(PClass):
        x = field(invariant=lambda v: (v > 0, "Must be positive"))

    assert hasattr(InvariantClass, '_pclass_invariants')
    assert len(InvariantClass._pclass_invariants) > 0

    # Test field initialization
    class InitClass(PClass):
        x = field(initial=42)

    obj = InitClass()
    assert obj.x == 42

    # Test mandatory field
    class MandatoryClass(PClass):
        x = field(mandatory=True)

    with pytest.raises(InvariantException):
        MandatoryClass()

    # Test with extra kwargs
    class ExtraClass(PClass):
        x = field()

    with pytest.raises(AttributeError):
        ExtraClass(x=1, y=2)

    # Test with ignore_extra
    obj = ExtraClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert obj.x == 1


# LLM-generated content at query #78
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field, invariant

    # Test basic instantiation
    class SimpleClass(PClass):
        x = field()
        y = field()

    instance = SimpleClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with mandatory field
    class MandatoryClass(PClass):
        x = field(mandatory=True)
        y = field()

    instance = MandatoryClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test missing mandatory field raises error
    try:
        MandatoryClass(y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "SimpleClass.x" in str(e.missing_fields)

    # Test with initial value
    class InitialClass(PClass):
        x = field(initial=10)
        y = field()

    instance = InitialClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with callable initial
    class CallableInitialClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = CallableInitialClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    instance = InvariantClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test invariant failure
    try:
        InvariantClass(x=-1, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Value must be positive" in e.invariant_errors

    # Test extra fields raise error
    try:
        SimpleClass(x=1, y=2, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

    # Test ignore_extra parameter
    instance = SimpleClass.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, "z")

    # Test factory_fields parameter
    class FactoryClass(PClass):
        x = field()
        y = field()

    instance = FactoryClass._factory_fields={"x"}, x=1, y=2
    assert instance.x == 1
    assert instance.y == 2

    # Test frozen attribute
    instance = SimpleClass(x=1, y=2)
    assert instance._pclass_frozen is True

    # Test setting attribute after frozen raises error
    try:
        instance.x = 3
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

    # Test global invariants
    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class GlobalInvariantClass(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = GlobalInvariantClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    try:
        GlobalInvariantClass(x=-1, y=-2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Sum must be positive" in str(e.invariant_errors)


# LLM-generated content at query #79
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test setting a field with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test setting a field with positional argument
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1
    assert obj.y == 2  # Original unchanged

    # Test setting multiple fields
    new_obj3 = obj.set(x=30, y=40)
    assert new_obj3.x == 30
    assert new_obj3.y == 40

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=100)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    m_new = m_obj.set(b=20)
    assert m_new.a == 1
    assert m_new.b == 20

    # Test with initial value
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=10)
    i_new = i_obj.set(c=5)
    assert i_new.c == 5
    assert i_new.d == 10

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        e = field(invariant=positive_invariant)
        f = field()

    inv_obj = InvariantClass(e=5, f=10)
    inv_new = inv_obj.set(e=10)
    assert inv_new.e == 10

    try:
        inv_obj.set(e=-1)
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #80
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restored_obj = pickle.loads(pickle.dumps(obj))

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2


# LLM-generated content at query #81
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic hash
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=2, y=1)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)

    # Test with different types
    class MixedClass(PClass):
        a = field()
        b = field(type=str)
        c = field(type=float)

    obj4 = MixedClass(a=10, b="test", c=3.14)
    obj5 = MixedClass(a=10, b="test", c=3.14)
    obj6 = MixedClass(a=10, b="test", c=3.15)

    assert hash(obj4) == hash(obj5)
    assert hash(obj4) != hash(obj6)

    # Test with missing optional fields
    class OptionalClass(PClass):
        required = field(mandatory=True)
        optional = field(initial=0)

    obj7 = OptionalClass(required=1)
    obj8 = OptionalClass(required=1)
    obj9 = OptionalClass(required=1, optional=5)

    assert hash(obj7) == hash(obj8)
    assert hash(obj7) != hash(obj9)

    # Test that hash works with evolver
    obj10 = TestClass(x=1, y=2)
    evolver = obj10.evolver()
    evolver.set('x', 3)
    obj11 = evolver.persistent()

    assert hash(obj10) != hash(obj11)


# LLM-generated content at query #82
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, TestClass)
    assert restored.x == 1
    assert restored.y == 2


# LLM-generated content at query #83
#--------------------------

```python
def test_PClass___new__():
    from pyrsistent import PClass, field, invariant

    # Test basic instantiation
    class SimpleClass(PClass):
        x = field()
        y = field()

    instance = SimpleClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    class MandatoryClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        MandatoryClass(y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.missing_fields) == 1
        assert 'MandatoryClass.x' in e.missing_fields

    # Test with default value
    class DefaultClass(PClass):
        x = field(initial=0)
        y = field()

    instance = DefaultClass(y=2)
    assert instance.x == 0
    assert instance.y == 2

    # Test with callable default
    class CallableDefaultClass(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = CallableDefaultClass(y=2)
    assert instance.x == 42
    assert instance.y == 2

    # Test with extra fields
    class ExtraFieldClass(PClass):
        x = field()
        y = field()

    try:
        ExtraFieldClass(x=1, y=2, z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

    # Test with ignore_extra
    instance = ExtraFieldClass.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with field invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    try:
        InvariantClass(x=-1, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Value must be positive" in e.invariant_errors

    # Test with global invariant
    @invariant
    def sum_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class GlobalInvariantClass(PClass):
        x = field()
        y = field()
        __invariant__ = sum_invariant

    try:
        GlobalInvariantClass(x=-3, y=2)
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Sum must be positive" in e.invariant_errors

    # Test with factory fields
    class FactoryClass(PClass):
        x = field()
        y = field()

    instance = FactoryClass._create({'x': 1, 'y': 2}, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

    # Test frozen attribute
    instance = SimpleClass(x=1, y=2)
    assert instance._pclass_frozen is True

    try:
        instance.x = 3
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #84
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1  # Original unchanged

    # Test set with positional arguments
    new_instance2 = instance.set('y', 20)
    assert new_instance2.y == 20
    assert new_instance2.x == 1
    assert instance.y == 2  # Original unchanged

    # Test setting multiple fields
    new_instance3 = instance.set(x=100, y=200)
    assert new_instance3.x == 100
    assert new_instance3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        instance.set(z=3)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    inst = MandatoryClass(a=1, b=2)
    new_inst = inst.set(b=20)
    assert new_inst.a == 1
    assert new_inst.b == 20

    # Test with field that has initial value
    class InitialClass(PClass):
        x = field(initial=0)
        y = field()

    init_inst = InitialClass(y=5)
    assert init_inst.x == 0
    new_init_inst = init_inst.set(x=10)
    assert new_init_inst.x == 10
    assert new_init_inst.y == 5


# LLM-generated content at query #85
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with kwargs
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with args
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    new_m_obj = m_obj.set(b=20)
    assert new_m_obj.a == 1
    assert new_m_obj.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=3)
    assert i_obj.c == 0
    new_i_obj = i_obj.set(c=10)
    assert new_i_obj.c == 10
    assert new_i_obj.d == 3


# LLM-generated content at query #86
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    obj_mand = MandatoryClass(a=1, b=2)
    new_obj_mand = obj_mand.set(a=10)
    assert new_obj_mand.a == 10
    assert new_obj_mand.b == 2

    # Test with field that has initial value
    class InitialClass(PClass):
        x = field(initial=0)
        y = field()

    obj_init = InitialClass(y=5)
    assert obj_init.x == 0
    new_obj_init = obj_init.set(x=10)
    assert new_obj_init.x == 10
    assert new_obj_init.y == 5

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    obj_inv = InvariantClass(x=5, y=10)
    try:
        obj_inv.set(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test with factory
    def double_factory(value):
        return value * 2

    class FactoryClass(PClass):
        x = field(factory=double_factory)
        y = field()

    obj_fact = FactoryClass(x=5, y=10)
    assert obj_fact.x == 10  # Factory doubles the value
    new_obj_fact = obj_fact.set(x=3)
    assert new_obj_fact.x == 6  # Factory doubles the new value


# LLM-generated content at query #87
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    assert obj.serialize() == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x + fmt)

    obj = TestClassWithFormat(x=1)
    assert obj.serialize(format='a') == {'x': '1a'}

    class TestClassWithNoSerializer(PClass):
        x = field()

    obj = TestClassWithNoSerializer(x=1)
    assert obj.serialize() == {'x': 1}

    class TestClassWithMissingValue(PClass):
        x = field()
        y = field()

    obj = TestClassWithMissingValue(x=1)
    assert obj.serialize() == {'x': 1}


# LLM-generated content at query #88
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    obj = TestPClass(x=1, y=2)
    restore_func, args = obj.__reduce__()

    assert restore_func == _restore_pickle
    assert len(args) == 2
    assert args[0] == TestPClass
    assert args[1] == {'x': 1, 'y': 2}

    restored_obj = restore_func(*args)
    assert restored_obj.x == 1
    assert restored_obj.y == 2
    assert isinstance(restored_obj, TestPClass)


# LLM-generated content at query #89
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    # Test basic serialization
    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    assert obj.serialize() == {'x': 1, 'y': 2}

    # Test with missing field
    class TestClass2(PClass):
        a = field()
        b = field(initial=0)

    obj2 = TestClass2(a=10)
    assert obj2.serialize() == {'a': 10, 'b': 0}

    # Test with custom serializer
    def custom_serializer(value):
        return str(value) + "_serialized"

    class TestClass3(PClass):
        z = field(serializer=custom_serializer)

    obj3 = TestClass3(z=42)
    assert obj3.serialize() == {'z': '42_serialized'}

    # Test with format parameter
    def format_serializer(format, value):
        if format == 'json':
            return {'value': value, 'format': format}
        return value

    class TestClass4(PClass):
        w = field(serializer=format_serializer)

    obj4 = TestClass4(w=100)
    assert obj4.serialize() == {'w': 100}
    assert obj4.serialize(format='json') == {'w': {'value': 100, 'format': 'json'}}


# LLM-generated content at query #90
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__weakref__' in TestClass.__slots__

    # Test PClass with multiple fields
    class TestClass2(PClass):
        x = field()
        y = field()

    assert 'x' in TestClass2._pclass_fields
    assert 'y' in TestClass2._pclass_fields

    # Test PClass with invariants
    class TestClass3(PClass):
        x = field()
        __invariant__ = lambda self: (True, None)

    assert hasattr(TestClass3, '_pclass_invariants')
    assert len(TestClass3._pclass_invariants) == 1

    # Test PClass with ignore_extra
    class TestClass4(PClass):
        x = field()

    instance = TestClass4(x=1, ignore_extra=True)
    assert instance.x == 1

    # Test PClass with missing mandatory field
    class TestClass5(PClass):
        x = field(mandatory=True)

    with pytest.raises(InvariantException):
        TestClass5()

    # Test PClass with invalid field value
    class TestClass6(PClass):
        x = field(invariant=lambda x: (x > 0, "x must be positive"))

    with pytest.raises(InvariantException):
        TestClass6(x=-1)

    # Test PClass with valid field value
    instance = TestClass6(x=1)
    assert instance.x == 1

    # Test PClass with initial value
    class TestClass7(PClass):
        x = field(initial=0)

    instance = TestClass7()
    assert instance.x == 0

    # Test PClass with callable initial value
    class TestClass8(PClass):
        x = field(initial=lambda: 0)

    instance = TestClass8()
    assert instance.x == 0

    # Test PClass with extra kwargs
    class TestClass9(PClass):
        x = field()

    with pytest.raises(AttributeError):
        TestClass9(x=1, y=2)

    # Test PClass with factory_fields
    class TestClass10(PClass):
        x = field()

    instance = TestClass10(x=1)
    instance2 = instance.set(x=2)
    assert instance2.x == 2


# LLM-generated content at query #91
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y='test')
    assert repr(obj) == "TestClass(x=1, y='test')"

    obj_empty = TestClass(x=0, y='')
    assert repr(obj_empty) == "TestClass(x=0, y='')"

    class SingleField(PClass):
        value = field()

    single_obj = SingleField(value=42)
    assert repr(single_obj) == "SingleField(value=42)"


# LLM-generated content at query #92
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)

    # Test basic serialization
    assert instance.serialize() == {'x': 1, 'y': 4}  # y is doubled by serializer

    # Test with format parameter
    assert instance.serialize(format='test') == {'x': 1, 'y': 4}

    # Test with missing optional field
    class TestClass2(PClass):
        a = field()
        b = field(mandatory=False)

    instance2 = TestClass2(a=1)
    assert instance2.serialize() == {'a': 1}

    # Test with custom serializer that uses format
    class TestClass3(PClass):
        z = field(serializer=lambda x, fmt: f"{x}-{fmt}" if fmt else str(x))

    instance3 = TestClass3(z=5)
    assert instance3.serialize() == {'z': '5'}
    assert instance3.serialize(format='json') == {'z': '5-json'}


# LLM-generated content at query #93
#--------------------------

```python
def test_PClassMeta___new__():
    class TestPClass(PClass):
        x = field()
        y = field()

    assert hasattr(TestPClass, '_pclass_fields')
    assert hasattr(TestPClass, '_pclass_invariants')
    assert hasattr(TestPClass, '__slots__')
    assert '__weakref__' in TestPClass.__slots__
    assert '_pclass_frozen' in TestPClass.__slots__
    assert 'x' in TestPClass.__slots__
    assert 'y' in TestPClass.__slots__

    class TestPClassNoWeakref(PClass):
        pass

    assert '__weakref__' not in TestPClassNoWeakref.__slots__

    class TestPClassWithInvariant(PClass):
        __invariant__ = lambda self: (True, None)
        x = field()

    assert hasattr(TestPClassWithInvariant, '_pclass_invariants')
    assert len(TestPClassWithInvariant._pclass_invariants) == 1


# LLM-generated content at query #94
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with different class instances
    class AnotherClass(PClass):
        x = field()
        y = field()

    obj4 = AnotherClass(x=1, y=2)
    assert obj1 != obj4

    # Test with missing fields
    class PartialClass(PClass):
        x = field()

    obj5 = PartialClass(x=1)
    obj6 = PartialClass(x=1)
    assert obj5 == obj6


# LLM-generated content at query #95
#--------------------------

```python
def test_PClass___eq__():
    class TestPClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    a = TestPClass(x=1, y=2)
    b = TestPClass(x=1, y=2)
    assert a == b

    # Test inequality with different values
    c = TestPClass(x=1, y=3)
    assert a != c

    # Test inequality with different types
    assert a != 1
    assert a != "string"
    assert a != None

    # Test with different number of fields
    class TestPClass2(PClass):
        x = field()
        y = field()
        z = field()

    d = TestPClass2(x=1, y=2, z=3)
    assert a != d

    # Test with missing fields
    class TestPClass3(PClass):
        x = field()

    e = TestPClass3(x=1)
    f = TestPClass3(x=1)
    assert e == f


# LLM-generated content at query #96
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)
    serialized = instance.serialize()

    assert serialized == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field()
        y = field(serializer=lambda x, fmt: x * 2 if fmt == 'test' else x)

    instance_with_format = TestClassWithFormat(x=1, y=2)
    serialized_with_format = instance_with_format.serialize(format='test')

    assert serialized_with_format == {'x': 1, 'y': 4}

    serialized_without_format = instance_with_format.serialize()
    assert serialized_without_format == {'x': 1, 'y': 2}


# LLM-generated content at query #97
#--------------------------

```python
def test_PClass___hash__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    # Test that equal objects have the same hash
    assert hash(obj1) == hash(obj2)

    # Test that different objects have different hashes
    assert hash(obj1) != hash(obj3)

    # Test that hash is consistent
    assert hash(obj1) == hash(obj1)

    # Test that hash works with missing fields
    class TestClass2(PClass):
        x = field()
        y = field(mandatory=False)

    obj4 = TestClass2(x=1)
    obj5 = TestClass2(x=1)
    assert hash(obj4) == hash(obj5)

    # Test that hash works with no fields
    class EmptyClass(PClass):
        pass

    obj6 = EmptyClass()
    obj7 = EmptyClass()
    assert hash(obj6) == hash(obj7)


# LLM-generated content at query #98
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('x', 20)
    assert new_obj2.x == 20
    assert new_obj2.y == 2

    # Test setting multiple fields
    new_obj3 = obj.set(x=30, y=40)
    assert new_obj3.x == 30
    assert new_obj3.y == 40

    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError):
        obj.set(z=5)

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    m_new = m_obj.set(a=100)
    assert m_new.a == 100
    assert m_new.b == 2

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=3)
    assert i_obj.c == 0
    i_new = i_obj.set(c=5)
    assert i_new.c == 5
    assert i_new.d == 3

    # Test that set returns new instance
    original_id = id(obj)
    new_id = id(obj.set(x=5))
    assert original_id != new_id


# LLM-generated content at query #99
#--------------------------

```python
def test_PClass___new__():
    # Test basic instantiation
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestPClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestPClass(x=1, y=2, z=3)

    # Test with default values
    class TestPClassWithDefaults(PClass):
        x = field(initial=0)
        y = field(initial=lambda: "default")

    instance = TestPClassWithDefaults()
    assert instance.x == 0
    assert instance.y == "default"

    # Test with factory fields
    class TestPClassWithFactory(PClass):
        x = field(factory=lambda val, ignore_extra=False: val * 2)

    instance = TestPClassWithFactory(x=5)
    assert instance.x == 10

    # Test with ignore_extra
    class TestPClassWithIgnoreExtra(PClass):
        x = field(ignore_extra=True)

    instance = TestPClassWithIgnoreExtra(x=1, z=3)
    assert instance.x == 1

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestPClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)

    instance = TestPClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestPClassWithInvariant(x=-1)

    # Test with global invariant
    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class TestPClassWithGlobalInvariant(PClass):
        __invariant__ = global_invariant
        x = field()
        y = field()

    instance = TestPClassWithGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestPClassWithGlobalInvariant(x=-1, y=-2)


# LLM-generated content at query #100
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restore_func, args = obj.__reduce__()
    restored_obj = restore_func(*args)

    assert isinstance(restored_obj, TestClass)
    assert restored_obj.x == 1
    assert restored_obj.y == 2


# LLM-generated content at query #101
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    serialized = obj.serialize()

    assert serialized == {'x': 1, 'y': 4}
    assert isinstance(serialized, dict)


# LLM-generated content at query #102
#--------------------------

```python
def test_PClass___eq__():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test equality with same values
    obj1 = TestPClass(x=1, y=2)
    obj2 = TestPClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestPClass(x=1, y=3)
    assert obj1 != obj3

    # Test inequality with different type
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class TestPClassWithOptional(PClass):
        x = field()
        y = field(initial=0)

    obj4 = TestPClassWithOptional(x=1)
    obj5 = TestPClassWithOptional(x=1, y=0)
    assert obj4 == obj5

    # Test with different classes
    class AnotherPClass(PClass):
        x = field()

    obj6 = AnotherPClass(x=1)
    assert obj1 != obj6


# LLM-generated content at query #103
#--------------------------

```python
def test_PClass___eq__():
    # Test equality with same values
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2

    # Test inequality with different values
    obj3 = TestClass(x=1, y=3)
    assert obj1 != obj3

    # Test with different types
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with subclass
    class SubClass(TestClass):
        z = field()

    obj4 = SubClass(x=1, y=2, z=3)
    assert obj1 != obj4

    # Test with missing fields
    class TestClass2(PClass):
        x = field()
        y = field(mandatory=False)

    obj5 = TestClass2(x=1)
    obj6 = TestClass2(x=1, y=None)
    assert obj5 == obj6

    # Test with different field values
    obj7 = TestClass2(x=2)
    assert obj5 != obj7


# LLM-generated content at query #104
#--------------------------

```python
def test_PClassMeta___new__():
    # Test basic PClass creation
    class TestClass(PClass):
        x = field()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert '__slots__' in TestClass.__dict__
    assert '_pclass_frozen' in TestClass.__slots__

    # Test that __weakref__ is added when inheriting from PClass
    class ChildClass(TestClass):
        y = field()

    assert '__weakref__' in ChildClass.__slots__

    # Test that fields are properly set
    instance = TestClass(x=1)
    assert instance.x == 1
    assert instance._pclass_frozen is True

    # Test invariant checking
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        x = field(invariant=positive_invariant)

    with pytest.raises(InvariantException):
        InvariantClass(x=-1)

    # Test mandatory field checking
    class MandatoryClass(PClass):
        x = field(mandatory=True)

    with pytest.raises(InvariantException):
        MandatoryClass()

    # Test that extra fields are rejected
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2)

    # Test that ignore_extra works
    class IgnoreExtraClass(PClass):
        x = field(ignore_extra=True)

    instance = IgnoreExtraClass(x=1, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

    # Test that initial values work
    class InitialClass(PClass):
        x = field(initial=0)

    instance = InitialClass()
    assert instance.x == 0

    # Test that callable initial values work
    class CallableInitialClass(PClass):
        x = field(initial=lambda: 42)

    instance = CallableInitialClass()
    assert instance.x == 42


# LLM-generated content at query #105
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test setting with keyword arguments
    obj = TestPClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test setting with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1
    assert obj.y == 2  # Original unchanged

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError):
        obj.set(z=3)

    # Test setting with factory fields
    class TestPClassWithFactory(PClass):
        x = field(initial=0)
        y = field()

    obj_factory = TestPClassWithFactory(y=5)
    new_obj_factory = obj_factory.set(x=10)
    assert new_obj_factory.x == 10
    assert new_obj_factory.y == 5

    # Test immutability
    with pytest.raises(AttributeError):
        obj.x = 20


# LLM-generated content at query #106
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test basic set with keyword argument
    obj = TestClass(x=1, y=2)
    new_obj = obj.set(x=10)
    assert new_obj.x == 10
    assert new_obj.y == 2
    assert obj.x == 1  # Original unchanged

    # Test set with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1

    # Test setting multiple fields
    new_obj3 = obj.set(x=100, y=200)
    assert new_obj3.x == 100
    assert new_obj3.y == 200

    # Test setting non-existent field raises AttributeError
    try:
        obj.set(z=3)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    m_obj = MandatoryClass(a=1, b=2)
    new_m_obj = m_obj.set(b=20)
    assert new_m_obj.a == 1
    assert new_m_obj.b == 20

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=1)
    new_i_obj = i_obj.set(c=10)
    assert new_i_obj.c == 10
    assert new_i_obj.d == 1

    # Test with factory field
    class FactoryClass(PClass):
        e = field(factory=lambda x: x * 2)
        f = field()

    f_obj = FactoryClass(f=1)
    new_f_obj = f_obj.set(e=5)
    assert new_f_obj.e == 5
    assert new_f_obj.f == 1

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Must be positive"

    class InvariantClass(PClass):
        g = field(invariant=positive_invariant)
        h = field()

    inv_obj = InvariantClass(g=1, h=2)
    new_inv_obj = inv_obj.set(g=2)
    assert new_inv_obj.g == 2
    assert new_inv_obj.h == 2

    # Test invariant violation
    try:
        inv_obj.set(g=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


