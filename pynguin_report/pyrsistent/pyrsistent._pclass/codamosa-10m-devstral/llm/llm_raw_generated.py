####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    restore_func, args = original.__reduce__()
    restored = restore_func(*args)

    assert isinstance(restored, TestClass)
    assert restored.x == 1
    assert restored.y == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    # Test setting with keyword argument
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
    try:
        obj.set(z=3)
        assert False, "Expected AttributeError"
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

    # Test with initial value
    class InitialPClass(PClass):
        x = field(initial=0)
        y = field()

    obj_initial = InitialPClass(y=2)
    new_obj_initial = obj_initial.set(x=10)
    assert new_obj_initial.x == 10
    assert new_obj_initial.y == 2

    # Test with factory field
    class FactoryPClass(PClass):
        x = field(factory=lambda x: x * 2)
        y = field()

    obj_factory = FactoryPClass(x=1, y=2)
    new_obj_factory = obj_factory.set(x=5)
    assert new_obj_factory.x == 10  # Factory doubles the value
    assert new_obj_factory.y == 2


# LLM-generated content at query #3
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
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test__PClassEvolver_remove():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    evolver = original.evolver()

    # Test removing existing attribute
    evolver.remove('x')
    result = evolver.persistent()
    assert result.x is None
    assert result.y == 2

    # Test removing non-existing attribute
    with pytest.raises(AttributeError):
        evolver.remove('z')

    # Test that original is unchanged
    assert original.x == 1
    assert original.y == 2


# LLM-generated content at query #5
#--------------------------

```python
def test_PClass_set():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    # Test setting with keyword arguments
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1  # Original unchanged

    # Test setting with positional arguments
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
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

    # Test with mandatory field
    class MandatoryClass(PClass):
        a = field(mandatory=True)
        b = field()

    mandatory_instance = MandatoryClass(a=5, b=10)
    new_mandatory = mandatory_instance.set(b=15)
    assert new_mandatory.a == 5
    assert new_mandatory.b == 15

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    initial_instance = InitialClass(d=20)
    assert initial_instance.c == 0
    new_initial = initial_instance.set(c=5)
    assert new_initial.c == 5
    assert new_initial.d == 20

    # Test with factory field
    class FactoryClass(PClass):
        e = field(factory=lambda x: x * 2)
        f = field()

    factory_instance = FactoryClass(e=3, f=4)
    assert factory_instance.e == 6
    new_factory = factory_instance.set(e=5)
    assert new_factory.e == 10
    assert new_factory.f == 4


# LLM-generated content at query #6
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

    obj = TestClassWithFormat(x='hello')
    assert obj.serialize() == {'x': 'hello'}
    assert obj.serialize(format='upper') == {'x': 'HELLO'}

    class TestClassNoSerializer(PClass):
        x = field()

    obj = TestClassNoSerializer(x=1)
    assert obj.serialize() == {'x': 1}


# LLM-generated content at query #7
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    assert obj.serialize() == {'x': 1, 'y': 4}

    class CustomSerializer:
        def __init__(self, value):
            self.value = value

        def serialize(self, format=None):
            return self.value.upper() if format == 'upper' else self.value.lower()

    class TestClassWithCustomSerializer(PClass):
        z = field(serializer=lambda x, format: x.serialize(format))

    obj2 = TestClassWithCustomSerializer(z=CustomSerializer('Hello'))
    assert obj2.serialize() == {'z': 'hello'}
    assert obj2.serialize(format='upper') == {'z': 'HELLO'}

    class TestClassWithMissingValue(PClass):
        a = field()
        b = field()

    obj3 = TestClassWithMissingValue(a=1)
    assert obj3.serialize() == {'a': 1}


# LLM-generated content at query #8
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field, s

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)
    serialized = instance.serialize()

    assert serialized == {'x': 1, 'y': 4}

    # Test with format
    serialized_with_format = instance.serialize(format='json')
    assert serialized_with_format == {'x': 1, 'y': 4}

    # Test with missing field
    class TestClassWithMissing(PClass):
        a = field()
        b = field()

    instance_missing = TestClassWithMissing(a=10)
    serialized_missing = instance_missing.serialize()
    assert serialized_missing == {'a': 10}


# LLM-generated content at query #9
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
    instance = TestClassWithFactory.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with invariant
    def positive_invariant(value):
        return value > 0, 'Value must be positive'

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with callable initial
    class TestClassWithCallableInitial(PClass):
        x = field(initial=lambda: 10)

    instance = TestClassWithCallableInitial()
    assert instance.x == 10


# LLM-generated content at query #10
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
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != {"x": 1, "y": 2}

    # Test with missing fields
    class TestClass2(PClass):
        a = field()
        b = field(initial=0)

    obj4 = TestClass2(a=1)
    obj5 = TestClass2(a=1)
    assert obj4 == obj5

    obj6 = TestClass2(a=1, b=0)
    assert obj4 == obj6

    # Test with NotImplemented
    assert obj1.__eq__(None) is NotImplemented


# LLM-generated content at query #11
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y="test")
    assert repr(instance) == "TestClass(x=1, y='test')"

    instance_empty = TestClass(x=0, y=None)
    assert repr(instance_empty) == "TestClass(x=0, y=None)"

    class SingleFieldClass(PClass):
        name = field()

    single_instance = SingleFieldClass(name="value")
    assert repr(single_instance) == "SingleFieldClass(name='value')"

    class NoFieldsClass(PClass):
        pass

    no_fields_instance = NoFieldsClass()
    assert repr(no_fields_instance) == "NoFieldsClass()"


# LLM-generated content at query #12
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

    # Test with missing mandatory field
    with pytest.raises(InvariantException):
        TestClass(x=1)

    # Test with extra fields
    with pytest.raises(AttributeError):
        TestClass(x=1, y=2, z=3)

    # Test with default values
    class TestClassWithDefaults(PClass):
        x = field(initial=10)
        y = field(initial=lambda: 20)

    instance = TestClassWithDefaults()
    assert instance.x == 10
    assert instance.y == 20

    # Test with custom invariant
    def check_positive(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=check_positive)

    instance = TestClassWithInvariant(x=5)
    assert instance.x == 5

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with ignore_extra
    class TestClassIgnoreExtra(PClass):
        x = field()

    instance = TestClassIgnoreExtra(x=1, ignore_extra=True)
    assert instance.x == 1

    # Test with factory_fields
    class TestClassFactoryFields(PClass):
        x = field()

    instance = TestClassFactoryFields(x=1)
    new_instance = instance.set(x=2)
    assert new_instance.x == 2
    assert instance.x == 1

    # Test with global invariant
    @invariant
    def check_sum(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class TestClassGlobalInvariant(PClass):
        x = field()
        y = field()

    instance = TestClassGlobalInvariant(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        TestClassGlobalInvariant(x=-1, y=-2)


# LLM-generated content at query #13
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
    assert restored == original


# LLM-generated content at query #15
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
    assert obj.x == 1  # Original object unchanged

    # Test setting a field with positional arguments
    new_obj2 = obj.set('y', 20)
    assert new_obj2.y == 20
    assert new_obj2.x == 1
    assert obj.y == 2  # Original object unchanged

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

    # Test setting field with None value
    new_obj4 = obj.set(x=None)
    assert new_obj4.x is None
    assert new_obj4.y == 2


# LLM-generated content at query #16
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

    # Test weakref slot
    class WeakRefClass(PClass):
        x = field()

    assert '__weakref__' in WeakRefClass.__slots__

    # Test inheritance
    class ParentClass(PClass):
        x = field()

    class ChildClass(ParentClass):
        y = field()

    assert len(ChildClass._pclass_fields) == 2
    assert 'x' in ChildClass._pclass_fields
    assert 'y' in ChildClass._pclass_fields

    # Test invariant storage
    def test_invariant(inst):
        return True, "OK"

    class InvariantClass(PClass):
        __invariant__ = test_invariant
        x = field()

    assert len(InvariantClass._pclass_invariants) == 1


# LLM-generated content at query #17
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

    # Test setting non-existent field raises AttributeError
    with pytest.raises(AttributeError):
        obj.set(z=100)

    # Test setting with factory fields
    class TestClassWithFactory(PClass):
        x = field()
        y = field()

    obj_factory = TestClassWithFactory(x=1, y=2)
    new_obj_factory = obj_factory.set(x=100)
    assert new_obj_factory.x == 100
    assert new_obj_factory.y == 2


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

    class TestClassWithMissing(PClass):
        a = field()
        b = field()

    instance_missing = TestClassWithMissing(a=10)
    serialized_missing = instance_missing.serialize()

    assert serialized_missing == {'a': 10}


# LLM-generated content at query #19
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

    with pytest.raises(InvariantException):
        MandatoryClass(y=2)

    # Test with initial value
    class InitialClass(PClass):
        x = field(initial=10)
        y = field()

    instance = InitialClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

    # Test with callable initial
    class CallableInitialClass(PClass):
        x = field(initial=lambda: 20)
        y = field()

    instance = CallableInitialClass(y=2)
    assert instance.x == 20
    assert instance.y == 2

    # Test with invariant
    def check_positive(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        x = field(invariant=check_positive)
        y = field()

    instance = InvariantClass(x=1, y=2)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        InvariantClass(x=-1, y=2)

    # Test with extra fields
    with pytest.raises(AttributeError):
        SimpleClass(x=1, y=2, z=3)

    # Test with ignore_extra
    class IgnoreExtraClass(PClass):
        x = field()
        y = field()

    instance = IgnoreExtraClass.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with factory_fields
    class FactoryClass(PClass):
        x = field()
        y = field()

    instance = FactoryClass(x=1, y=2)
    evolver = instance.evolver()
    evolver.set('x', 10)
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 2

    # Test with global invariant
    def global_invariant(instance):
        return instance.x != instance.y, "x and y must be different"

    class GlobalInvariantClass(PClass):
        x = field()
        y = field()
        __invariant__ = global_invariant

    instance = GlobalInvariantClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    with pytest.raises(InvariantException):
        GlobalInvariantClass(x=1, y=1)

    # Test with serializer
    class SerializerClass(PClass):
        x = field(serializer=lambda x: str(x))
        y = field()

    instance = SerializerClass(x=1, y=2)
    serialized = instance.serialize()
    assert serialized['x'] == '1'
    assert serialized['y'] == 2

    # Test with transform
    class TransformClass(PClass):
        x = field()
        y = field()

    instance = TransformClass(x=1, y=2)
    transformed = instance.transform(lambda x: x.set('x', 10))
    assert transformed.x == 10
    assert transformed.y == 2

    # Test with equality
    instance1 = SimpleClass(x=1, y=2)
    instance2 = SimpleClass(x=1, y=2)
    assert instance1 == instance2

    # Test with inequality
    instance3 = SimpleClass(x=1, y=3)
    assert instance1 != instance3

    # Test with hash
    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)

    # Test with repr
    assert repr(instance1) == "SimpleClass(x=1, y=2)"

    # Test with pickle
    import pickle
    pickled = pickle.dumps(instance1)
    unpickled = pickle.loads(pickled)
    assert unpickled == instance1

    # Test with evolver
    evolver = instance1.evolver()
    evolver['x'] = 10
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 2

    # Test with remove
    instance4 = SimpleClass(x=1, y=2)
    instance5 = instance4.remove('x')
    assert not hasattr(instance5, 'x')
    assert instance5.y == 2

    # Test with set
    instance6 = instance1.set(x=10)
    assert instance6.x == 10
    assert instance6.y == 2

    # Test with create
    instance7 = SimpleClass.create({'x': 1, 'y': 2})
    assert instance7.x == 1
    assert instance7.y == 2

    # Test with frozen
    with pytest.raises(AttributeError):
        instance1.x = 10

    # Test with delete
    with pytest.raises(AttributeError):
        del instance1.x


# LLM-generated content at query #20
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


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    assert obj.serialize() == {'x': 1, 'y': 4}

    obj = TestClass(x='a', y='b')
    assert obj.serialize() == {'x': 'a', 'y': 'bb'}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda x, fmt: x.upper() if fmt == 'upper' else x)

    obj = TestClassWithFormat(x='hello')
    assert obj.serialize() == {'x': 'hello'}
    assert obj.serialize(format='upper') == {'x': 'HELLO'}

    class TestClassWithMissing(PClass):
        x = field()
        y = field()

    obj = TestClassWithMissing(x=1)
    assert obj.serialize() == {'x': 1}


# LLM-generated content at query #2
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
    obj4 = TestPClass(x=1)
    obj5 = TestPClass(x=1)
    assert obj4 == obj5

    # Test with different number of fields
    obj6 = TestPClass(x=1, y=2)
    obj7 = TestPClass(x=1)
    assert obj6 != obj7

    # Test with different field names
    class AnotherPClass(PClass):
        a = field()
        b = field()

    obj8 = AnotherPClass(a=1, b=2)
    assert obj1 != obj8


# LLM-generated content at query #3
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
    evolver.set('y', 20)
    assert evolver['y'] == 20
    assert evolver._pclass_evolver_data_is_dirty
    assert 'y' in evolver._factory_fields

    # Test setting same value (should not mark as dirty)
    evolver._pclass_evolver_data_is_dirty = False
    evolver.set('x', 10)
    assert not evolver._pclass_evolver_data_is_dirty

    # Test persistent after modifications
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert isinstance(new_instance, TestClass)


# LLM-generated content at query #4
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

    # Test inequality with non-PClass object
    assert obj1 != "not a PClass"

    # Test with missing fields
    class PartialClass(PClass):
        x = field()
        y = field(mandatory=False)

    obj5 = PartialClass(x=1)
    obj6 = PartialClass(x=1)
    assert obj5 == obj6

    # Test with different missing fields
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
    obj10 = NullableClass(x=1, y=None)
    assert obj8 != obj10


# LLM-generated content at query #5
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
    evolver.remove('x')
    result = evolver.persistent()
    assert result.x is None
    assert result.y == 2

    # Test removal of non-existent field
    with pytest.raises(AttributeError):
        evolver.remove('z')

    # Test removal using __delitem__
    evolver = original.evolver()
    del evolver['y']
    result = evolver.persistent()
    assert result.x == 1
    assert result.y is None


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
    obj3 = TestClass(x=3, y=4)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


# LLM-generated content at query #7
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    instance = TestClass(x=1, y=2)

    # Test basic serialization
    assert instance.serialize() == {'x': 1, 'y': 4}

    # Test with custom format
    assert instance.serialize(format='test') == {'x': 1, 'y': 4}

    # Test with missing fields
    class TestClass2(PClass):
        a = field()
        b = field()

    instance2 = TestClass2(a=10)
    assert instance2.serialize() == {'a': 10}

    # Test with no fields
    class EmptyClass(PClass):
        pass

    empty_instance = EmptyClass()
    assert empty_instance.serialize() == {}


# LLM-generated content at query #8
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
    evolver.set('y', 20)
    assert evolver['y'] == 20
    assert evolver._pclass_evolver_data_is_dirty

    # Test setting same value doesn't mark as dirty
    evolver.set('x', 10)
    assert not evolver._pclass_evolver_data_is_dirty

    # Test persistent creates new instance
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert new_instance is not original

    # Test factory_fields tracking
    assert 'x' in evolver._factory_fields
    assert 'y' in evolver._factory_fields


# LLM-generated content at query #9
#--------------------------

```python
def test__PClassEvolver_set():
    # Setup
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    original = TestPClass(x=1, y=2)
    evolver = original.evolver()

    # Test setting existing field
    evolver.set('x', 10)
    assert evolver['x'] == 10
    assert evolver._pclass_evolver_data_is_dirty
    assert 'x' in evolver._factory_fields

    # Test setting new value (same as existing)
    evolver.set('x', 10)
    assert evolver['x'] == 10
    assert not evolver._pclass_evolver_data_is_dirty  # Should not be dirty since value didn't change

    # Test setting different field
    evolver.set('y', 20)
    assert evolver['y'] == 20
    assert evolver._pclass_evolver_data_is_dirty
    assert 'y' in evolver._factory_fields

    # Test persistent creates new instance
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert original.x == 1
    assert original.y == 2

    # Test using __setitem__ syntax
    evolver['x'] = 100
    assert evolver['x'] == 100
    assert evolver._pclass_evolver_data_is_dirty


# LLM-generated content at query #10
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    assert repr(obj) == "TestClass(x=1, y=2)"

    obj_empty = TestClass(x=1)
    assert repr(obj_empty) == "TestClass(x=1)"

    class ComplexClass(PClass):
        name = field()
        value = field()
        nested = field()

    complex_obj = ComplexClass(name="test", value=42, nested={"a": 1})
    assert repr(complex_obj) == "ComplexClass(name='test', value=42, nested={'a': 1})"


# LLM-generated content at query #11
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

    # Test field invariants
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestInvariant(PClass):
        x = field(invariant=positive_invariant)

    with pytest.raises(InvariantException):
        TestInvariant(x=-1)

    # Test initial values
    class TestInitial(PClass):
        x = field(initial=5)
        y = field()

    instance = TestInitial(y=10)
    assert instance.x == 5
    assert instance.y == 10

    # Test factory fields
    class TestFactory(PClass):
        x = field()
        y = field()

    instance = TestFactory(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

    # Test ignore_extra
    class TestIgnoreExtra(PClass):
        x = field()
        y = field(ignore_extra=True)

    instance = TestIgnoreExtra(x=1, y=2, z=3)
    assert instance.x == 1
    assert instance.y == 2

    # Test frozen attribute
    instance = TestClass(x=1, y=2)
    with pytest.raises(AttributeError):
        instance.x = 3

    # Test weakref
    class TestWeakRef(PClass):
        x = field()

    instance = TestWeakRef(x=1)
    assert hasattr(instance, '__weakref__')


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
        c = field(initial=0)
        d = field()

    i_obj = InitialClass(d=3)
    assert i_obj.c == 0
    new_i_obj = i_obj.set(c=5)
    assert new_i_obj.c == 5
    assert new_i_obj.d == 3

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantClass(PClass):
        e = field(invariant=positive_invariant)
        f = field()

    inv_obj = InvariantClass(e=1, f=4)
    new_inv_obj = inv_obj.set(e=2)
    assert new_inv_obj.e == 2

    try:
        inv_obj.set(e=-1)
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #13
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

    # Test with field that has initial value
    class InitialClass(PClass):
        x = field(initial=0)
        y = field()

    obj_init = InitialClass(y=5)
    new_obj_init = obj_init.set(x=10)
    assert new_obj_init.x == 10
    assert new_obj_init.y == 5

    # Test that set preserves other fields
    obj_full = TestClass(x=1, y=2)
    new_obj_full = obj_full.set(x=99)
    assert hasattr(new_obj_full, 'y')
    assert new_obj_full.y == 2

    # Test with factory fields
    class FactoryClass(PClass):
        x = field()
        y = field()

    obj_factory = FactoryClass(x=1, y=2)
    new_obj_factory = obj_factory.set(x=10)
    assert new_obj_factory.x == 10
    assert new_obj_factory.y == 2

    # Test equality after set
    obj1 = TestClass(x=1, y=2)
    obj2 = obj1.set(x=1, y=2)
    assert obj1 == obj2


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda v: v * 2)

    obj = TestClass(x=1, y=2)
    assert obj.serialize() == {'x': 1, 'y': 4}

    class TestClassWithFormat(PClass):
        x = field(serializer=lambda v, fmt: v + fmt)
        y = field()

    obj2 = TestClassWithFormat(x=1, y=2)
    assert obj2.serialize(format='_test') == {'x': '1_test', 'y': 2}

    class TestClassNoSerializer(PClass):
        x = field()
        y = field()

    obj3 = TestClassNoSerializer(x=1, y=2)
    assert obj3.serialize() == {'x': 1, 'y': 2}


# LLM-generated content at query #2
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


# LLM-generated content at query #3
#--------------------------

```python
def test__PClassEvolver_set():
    class TestPClass(PClass):
        x = field()
        y = field()

    original = TestPClass(x=1, y=2)
    evolver = original.evolver()

    # Test setting a new value
    evolver.set('x', 10)
    assert evolver['x'] == 10
    assert evolver._pclass_evolver_data_is_dirty
    assert 'x' in evolver._factory_fields

    # Test setting the same value (should not mark as dirty)
    evolver.set('x', 10)
    assert not evolver._pclass_evolver_data_is_dirty

    # Test setting another field
    evolver.set('y', 20)
    assert evolver['y'] == 20
    assert evolver._pclass_evolver_data_is_dirty
    assert 'y' in evolver._factory_fields

    # Test persistent() returns new instance with updated values
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert original.x == 1
    assert original.y == 2


# LLM-generated content at query #4
#--------------------------

```python
def test_PClass___eq__():
    # Test basic equality
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=3, y=4)

    assert obj1 == obj2
    assert not (obj1 == obj3)

    # Test with different types
    assert not (obj1 == 1)
    assert not (obj1 == "string")
    assert not (obj1 == None)

    # Test with missing fields
    class TestClass2(PClass):
        a = field(initial=0)
        b = field()

    obj4 = TestClass2(b=5)
    obj5 = TestClass2(b=5)
    obj6 = TestClass2(b=6)

    assert obj4 == obj5
    assert not (obj4 == obj6)

    # Test with different classes
    class TestClass3(PClass):
        x = field()

    obj7 = TestClass(x=1, y=2)
    obj8 = TestClass3(x=1)

    assert not (obj7 == obj8)


# LLM-generated content at query #5
#--------------------------

```python
def test__PClassEvolver_remove():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    evolver = original.evolver()

    # Test removing an existing field
    result = evolver.remove('x')
    assert isinstance(result, TestClass)
    assert not hasattr(result, 'x')
    assert result.y == 2

    # Test removing a non-existing field
    with pytest.raises(AttributeError):
        evolver.remove('z')

    # Test that original is unchanged
    assert original.x == 1
    assert original.y == 2


# LLM-generated content at query #6
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
    assert evolver['y'] == 2
    assert evolver._pclass_evolver_data_is_dirty

    # Test setting new field
    evolver.set('z', 30)
    assert evolver['z'] == 30
    assert evolver._pclass_evolver_data_is_dirty

    # Test setting same value doesn't dirty
    evolver.set('x', 10)
    assert not evolver._pclass_evolver_data_is_dirty

    # Test persistent creates new instance
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert hasattr(new_instance, 'z')  # Assuming dynamic fields are allowed

    # Test original remains unchanged
    assert original.x == 1
    assert original.y == 2
    assert not hasattr(original, 'z')


# LLM-generated content at query #7
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

    obj_init = InitialClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2


# LLM-generated content at query #8
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    assert repr(obj) == "TestClass(x=1, y=2)"

    obj2 = TestClass(x="a", y="b")
    assert repr(obj2) == "TestClass(x='a', y='b')"

    class EmptyClass(PClass):
        pass

    obj3 = EmptyClass()
    assert repr(obj3) == "EmptyClass()"


# LLM-generated content at query #9
#--------------------------

```python
def test__PClassEvolver_remove():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    evolver = original.evolver()

    # Test removing an existing attribute
    evolver.remove('x')
    result = evolver.persistent()
    assert result.x is None
    assert result.y == 2

    # Test removing a non-existing attribute
    evolver = original.evolver()
    with pytest.raises(AttributeError):
        evolver.remove('z')

    # Test removing all attributes
    evolver = original.evolver()
    evolver.remove('x')
    evolver.remove('y')
    result = evolver.persistent()
    assert result.x is None
    assert result.y is None


# LLM-generated content at query #10
#--------------------------

```python
def test__PClassEvolver_set():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    original = TestPClass(x=1, y=2)
    evolver = original.evolver()

    # Test setting existing field
    evolver.set('x', 10)
    assert evolver['x'] == 10
    assert evolver._pclass_evolver_data_is_dirty

    # Test setting new field
    evolver.set('y', 20)
    assert evolver['y'] == 20
    assert evolver._pclass_evolver_data_is_dirty

    # Test setting same value doesn't mark as dirty
    evolver.set('x', 10)
    assert not evolver._pclass_evolver_data_is_dirty

    # Test persistent creates new instance
    new_instance = evolver.persistent()
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert new_instance is not original

    # Test factory_fields tracking
    assert 'x' in evolver._factory_fields
    assert 'y' in evolver._factory_fields


# LLM-generated content at query #11
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

    # Test PClass with multiple fields
    class MultiFieldClass(PClass):
        x = field()
        y = field()
        z = field()

    assert len(MultiFieldClass._pclass_fields) == 3
    assert 'x' in MultiFieldClass._pclass_fields
    assert 'y' in MultiFieldClass._pclass_fields
    assert 'z' in MultiFieldClass._pclass_fields

    # Test PClass with invariants
    def invariant_func(instance):
        return True, None

    class InvariantClass(PClass):
        __invariant__ = invariant_func
        x = field()

    assert InvariantClass._pclass_invariants == (invariant_func,)

    # Test PClass without PClass base (should not have __weakref__)
    class NonPClassBase:
        pass

    class NonPClassChild(NonPClassBase, PClass):
        x = field()

    assert '__weakref__' not in NonPClassChild.__slots__

    # Test field initialization
    class InitClass(PClass):
        x = field(initial=10)
        y = field()

    obj = InitClass(y=20)
    assert obj.x == 10
    assert obj.y == 20

    # Test mandatory field
    class MandatoryClass(PClass):
        x = field(mandatory=True)

    try:
        MandatoryClass()
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test field type checking
    class TypedClass(PClass):
        x = field(type=int)

    obj = TypedClass(x=10)
    assert obj.x == 10

    try:
        TypedClass(x="not an int")
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test field invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class InvariantFieldClass(PClass):
        x = field(invariant=positive_invariant)

    obj = InvariantFieldClass(x=10)
    assert obj.x == 10

    try:
        InvariantFieldClass(x=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test global invariant
    def global_invariant(instance):
        return instance.x + instance.y > 0, "Sum must be positive"

    class GlobalInvariantClass(PClass):
        __invariant__ = global_invariant
        x = field()
        y = field()

    obj = GlobalInvariantClass(x=1, y=1)
    assert obj.x == 1
    assert obj.y == 1

    try:
        GlobalInvariantClass(x=-1, y=-1)
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test extra fields
    class ExtraFieldClass(PClass):
        x = field()

    try:
        ExtraFieldClass(x=1, y=2)
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

    # Test ignore_extra
    class IgnoreExtraClass(PClass):
        x = field()

    obj = IgnoreExtraClass(x=1, y=2, ignore_extra=True)
    assert obj.x == 1
    assert not hasattr(obj, 'y')


# LLM-generated content at query #12
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


# LLM-generated content at query #13
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

    i_obj = InitialClass(d=2)
    assert i_obj.c == 0
    new_i_obj = i_obj.set(c=10)
    assert new_i_obj.c == 10
    assert new_i_obj.d == 2


# LLM-generated content at query #14
#--------------------------

```python
def test_PClassMeta___new__():
    class TestPClass(PClass):
        x = field()
        y = field()

    # Test that the metaclass sets up the fields correctly
    assert hasattr(TestPClass, '_pclass_fields')
    assert 'x' in TestPClass._pclass_fields
    assert 'y' in TestPClass._pclass_fields

    # Test that the metaclass sets up the invariants correctly
    assert hasattr(TestPClass, '_pclass_invariants')

    # Test that the metaclass sets up the slots correctly
    assert '__slots__' in TestPClass.__dict__
    assert '_pclass_frozen' in TestPClass.__slots__
    assert 'x' in TestPClass.__slots__
    assert 'y' in TestPClass.__slots__

    # Test that the metaclass adds __weakref__ to the top level class
    assert '__weakref__' in TestPClass.__slots__

    # Test that the metaclass does not add __weakref__ to subclasses
    class SubTestPClass(TestPClass):
        z = field()

    assert '__weakref__' not in SubTestPClass.__slots__


# LLM-generated content at query #15
#--------------------------

```python
def test__PClassEvolver_remove():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    evolver = obj.evolver()

    # Test successful removal
    result = evolver.remove('x')
    assert isinstance(result, TestClass)
    assert not hasattr(result, 'x')
    assert result.y == 2

    # Test removal of non-existent attribute
    with pytest.raises(AttributeError):
        evolver.remove('z')

    # Test that original object is unchanged
    assert obj.x == 1
    assert obj.y == 2


# LLM-generated content at query #16
#--------------------------

```python
def test_PClass___repr__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y="test")
    assert repr(instance) == "TestClass(x=1, y='test')"

    instance_empty = TestClass(x=None, y=None)
    assert repr(instance_empty) == "TestClass(x=None, y=None)"

    class SingleFieldClass(PClass):
        value = field()

    single_instance = SingleFieldClass(value=42)
    assert repr(single_instance) == "SingleFieldClass(value=42)"


# LLM-generated content at query #17
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

    # Test inequality with non-PClass object
    assert obj1 != 1
    assert obj1 != "string"
    assert obj1 != None

    # Test with missing fields
    class OptionalClass(PClass):
        x = field()
        y = field(initial=0)

    obj5 = OptionalClass(x=1)
    obj6 = OptionalClass(x=1, y=0)
    assert obj5 == obj6

    # Test with NotImplemented
    assert (obj1 == "not_a_pclass") == NotImplemented


# LLM-generated content at query #18
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)
    assert obj.serialize() == {'x': 1, 'y': 4}

    obj2 = TestClass(x=5, y=10)
    assert obj2.serialize() == {'x': 5, 'y': 20}

    class TestClass2(PClass):
        a = field(serializer=lambda x, fmt: f"{x}_{fmt}" if fmt else str(x))

    obj3 = TestClass2(a=100)
    assert obj3.serialize() == {'a': '100'}
    assert obj3.serialize(format='json') == {'a': '100_json'}

    class TestClass3(PClass):
        pass

    obj4 = TestClass3()
    assert obj4.serialize() == {}


# LLM-generated content at query #19
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
        x = field(initial=lambda: 10)
        y = field(initial=lambda: 20)

    instance = TestClassWithCallableInitial()
    assert instance.x == 10
    assert instance.y == 20

    # Test with invariant
    def positive_invariant(value):
        return value > 0, "Value must be positive"

    class TestClassWithInvariant(PClass):
        x = field(invariant=positive_invariant)

    instance = TestClassWithInvariant(x=1)
    assert instance.x == 1

    with pytest.raises(InvariantException):
        TestClassWithInvariant(x=-1)

    # Test with factory
    class TestClassWithFactory(PClass):
        x = field(factory=lambda v: v * 2)

    instance = TestClassWithFactory(x=5)
    assert instance.x == 10

    # Test with ignore_extra
    class TestClassWithIgnoreExtra(PClass):
        x = field()
        y = field()

    instance = TestClassWithIgnoreExtra(x=1, y=2, z=3, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

    # Test with _factory_fields
    class TestClassWithFactoryFields(PClass):
        x = field(factory=lambda v: v * 2)
        y = field()

    instance = TestClassWithFactoryFields(x=5, y=10, _factory_fields={'x'})
    assert instance.x == 10
    assert instance.y == 10

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


# LLM-generated content at query #20
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
    obj4 = TestClass(x=1)
    obj5 = TestClass(x=1)
    assert obj4 == obj5

    # Test with different field sets
    obj6 = TestClass(x=1, y=2)
    obj7 = TestClass(x=1)
    assert obj6 != obj7

    # Test with None values
    obj8 = TestClass(x=None, y=None)
    obj9 = TestClass(x=None, y=None)
    assert obj8 == obj9

    # Test with one None and one non-None
    obj10 = TestClass(x=1, y=None)
    obj11 = TestClass(x=1, y=2)
    assert obj10 != obj11


# LLM-generated content at query #21
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

    # Test with initial value field
    class InitialClass(PClass):
        c = field(initial=0)
        d = field()

    obj_init = InitialClass(d=2)
    assert obj_init.c == 0
    new_obj_init = obj_init.set(c=10)
    assert new_obj_init.c == 10
    assert new_obj_init.d == 2


# LLM-generated content at query #22
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

    class TestClassWithMissing(PClass):
        a = field()
        b = field()

    obj_missing = TestClassWithMissing(a=5)
    serialized_missing = obj_missing.serialize()

    assert serialized_missing == {'a': 5}


# LLM-generated content at query #23
#--------------------------

```python
def test_PClass___reduce__():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    restored = pickle.loads(pickle.dumps(instance))

    assert isinstance(restored, TestClass)
    assert restored.x == 1
    assert restored.y == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_PClass_serialize():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field(serializer=lambda x: x * 2)

    obj = TestClass(x=1, y=2)

    # Test default serialization
    assert obj.serialize() == {'x': 1, 'y': 2}

    # Test custom serializer
    assert obj.serialize() == {'x': 1, 'y': 4}

    # Test with missing optional field
    class TestClass2(PClass):
        x = field()
        z = field(initial=5)

    obj2 = TestClass2(x=10)
    assert obj2.serialize() == {'x': 10, 'z': 5}

    # Test with format parameter (assuming it's passed to serializer)
    class TestClass3(PClass):
        a = field(serializer=lambda val, fmt: f"{val}-{fmt}" if fmt else str(val))

    obj3 = TestClass3(a=100)
    assert obj3.serialize() == {'a': '100'}
    assert obj3.serialize(format='test') == {'a': '100-test'}


# LLM-generated content at query #25
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

    # Test that the restored object is equal to the original
    restored_obj = restore_func(*args)
    assert restored_obj == obj
    assert restored_obj.x == 1
    assert restored_obj.y == 2


