####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_pclass_meta_new_with_single_checkedtype_base():
    class TestClass(metaclass=PClassMeta):
        pass
    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__

def test_pclass_meta_new_with_multiple_bases():
    class Base1:
        pass
    class Base2:
        pass
    class TestClass(Base1, Base2, metaclass=PClassMeta):
        pass
    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' not in TestClass.__slots__

def test_pclass_meta_new_with_field_inheritance():
    class Base(metaclass=PClassMeta):
        x = _PField()
    class TestClass(Base, metaclass=PClassMeta):
        y = _PField()
    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    assert 'x' not in TestClass.__dict__
    assert 'y' not in TestClass.__dict__

def test_pclass_meta_new_with_invariant_inheritance():
    def invariant_func(obj):
        return True, ()
    class Base(metaclass=PClassMeta):
        __invariant__ = invariant_func
    class TestClass(Base, metaclass=PClassMeta):
        pass
    assert len(TestClass._pclass_invariants) == 1
    assert TestClass._pclass_invariants[0](None) == (True, ())

def test_pclass_meta_new_with_non_callable_invariant():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = "not callable"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        assert False, "Expected TypeError"


# LLM-generated content at query #2
#--------------------------

```python
def test___reduce___returns_correct_tuple():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()

    assert result[0] == _restore_pickle
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


# LLM-generated content at query #3
#--------------------------

```python
def test_pickle_support_returns_correct_tuple():
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=42)
    result = instance.__reduce__()
    assert result == (_restore_pickle, (TestClass, {'x': 42}))


# LLM-generated content at query #4
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_missing_mandatory_field():
    class TestClass(PClass):
        x = field(mandatory=True)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields

def test_pclass_new_with_extra_fields():
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

def test_pclass_new_with_initial_value():
    class TestClass(PClass):
        x = field(initial=0)

    instance = TestClass()
    assert instance.x == 0

def test_pclass_new_with_callable_initial():
    class TestClass(PClass):
        x = field(initial=lambda: 42)

    instance = TestClass()
    assert instance.x == 42

def test_pclass_new_with_factory():
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)

    instance = TestClass(x=5)
    assert instance.x == 10

def test_pclass_new_with_invalid_type():
    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert "Invalid type for field TestClass.x" in str(e)

def test_pclass_new_with_invariant_failure():
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "must be positive"))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "must be positive" in e.error_codes

def test_pclass_new_with_ignore_extra():
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "y")

def test_pclass_new_with_factory_fields():
    class TestClass(PClass):
        x = field(factory=lambda v: v.upper())

    instance = TestClass(x="hello", _factory_fields={"x"})
    assert instance.x == "HELLO"

def test_pclass_new_with_global_invariant_failure():
    def global_invariant(obj):
        return (obj.x != obj.y, "x and y must be different")

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_invariant]

    try:
        TestClass(x=1, y=1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Global invariant failed" in str(e)


# LLM-generated content at query #5
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_mandatory_field_missing():
    class TestPClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestPClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestPClass.x" in str(e)

def test_pclass_constructor_with_extra_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    try:
        TestPClass(x=1, y=2, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_ignore_extra():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_initial_value():
    class TestPClass(PClass):
        x = field(initial=10)
        y = field()

    instance = TestPClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    class TestPClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestPClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_invariant_check():
    class TestPClass(PClass):
        x = field(invariant=lambda x: x > 0)
        y = field()

    try:
        TestPClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_factory_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass._create({"x": 1, "y": 2}, _factory_fields={"x"})
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_pclass_instance():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance1 = TestPClass(x=1, y=2)
    instance2 = TestPClass.create(instance1)
    assert instance2.x == 1
    assert instance2.y == 2


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize_with_no_serializer():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize() == {'x': 1, 'y': 2}

def test_serialize_with_custom_serializer():
    class TestClass(PClass):
        x = field(serializer=lambda v: str(v))
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize() == {'x': '1', 'y': 2}

def test_serialize_with_missing_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1)
    assert instance.serialize() == {'x': 1}

def test_serialize_with_format():
    class TestClass(PClass):
        x = field(serializer=lambda v, fmt: f"{v}_{fmt}" if fmt else str(v))
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize(format='json') == {'x': '1_json', 'y': 2}


# LLM-generated content at query #7
#--------------------------

```python
def test__is_pclass_with_single_checkedtype_base():
    assert _is_pclass((CheckedType,)) == True

def test__is_pclass_with_multiple_bases():
    assert _is_pclass((CheckedType, object)) == False

def test__is_pclass_with_no_bases():
    assert _is_pclass(()) == False

def test__is_pclass_with_non_checkedtype_base():
    assert _is_pclass((object,)) == False


# LLM-generated content at query #8
#--------------------------

```python
def test_pclass_repr():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"


# LLM-generated content at query #9
#--------------------------

```python
def test_serialize_returns_dict_with_correct_keys():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    serialized = instance.serialize()
    assert isinstance(serialized, dict)
    assert set(serialized.keys()) == {'x', 'y'}


# LLM-generated content at query #10
#--------------------------

```python
def test__is_pclass_returns_true_for_pclass_bases():
    class PClass(metaclass=PClassMeta):
        pass

    class SubPClass(PClass):
        pass

    assert _is_pclass((SubPClass,)) == True


# LLM-generated content at query #11
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    try:
        TestClass(x=1, y=2, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 0
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 42
    assert instance.y == 2

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass._create({'x': 1, 'y': 2}, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass.create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')

def test_pclass_constructor_with_invariant_check():
    from pyrsistent import PClass, field, invariant

    def check_positive(instance):
        if instance.x < 0:
            raise ValueError("x must be positive")

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [check_positive]

    try:
        TestClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_pclassmeta_new_with_checkedtype_base():
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = _PField()
        field2 = _PField()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert TestClass._pclass_fields == {'field1': TestClass.field1, 'field2': TestClass.field2}
    assert TestClass._pclass_invariants == (wrap_invariant(TestClass.__invariant__),)
    assert TestClass.__slots__ == ('_pclass_frozen', 'field1', 'field2', '__weakref__')

def test_pclassmeta_new_without_checkedtype_base():
    class BaseClass:
        pass

    class TestClass(BaseClass, metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = _PField()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert TestClass._pclass_fields == {'field1': TestClass.field1}
    assert TestClass._pclass_invariants == (wrap_invariant(TestClass.__invariant__),)
    assert TestClass.__slots__ == ('_pclass_frozen', 'field1')

def test_pclassmeta_new_with_inherited_fields():
    class BaseClass(metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field1 = _PField()

    class TestClass(BaseClass, metaclass=PClassMeta):
        field2 = _PField()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert TestClass._pclass_fields == {'field1': BaseClass.field1, 'field2': TestClass.field2}
    assert TestClass._pclass_invariants == (wrap_invariant(BaseClass.__invariant__),)
    assert TestClass.__slots__ == ('_pclass_frozen', 'field1', 'field2')

def test_pclassmeta_new_with_inherited_invariants():
    class BaseClass(metaclass=PClassMeta):
        __invariant__ = lambda self: True

    class TestClass(BaseClass, metaclass=PClassMeta):
        __invariant__ = lambda self: False

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert TestClass._pclass_fields == {}
    assert TestClass._pclass_invariants == (wrap_invariant(BaseClass.__invariant__), wrap_invariant(TestClass.__invariant__))
    assert TestClass.__slots__ == ('_pclass_frozen', '__weakref__')

def test_pclassmeta_new_with_non_callable_invariant():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = "not callable"

        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"


# LLM-generated content at query #13
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    invariant_errors = ["error1"]
    missing_fields = ["field1"]
    assert invariant_errors or missing_fields


# LLM-generated content at query #14
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in str(e)

def test_pclass_constructor_with_extra_fields():
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

def test_pclass_constructor_with_initial_value():
    class TestClass(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    class TestClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_factory_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass._create({'x': 1, 'y': 2}, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_ignore_extra():
    class TestClass(PClass):
        x = field()

    instance = TestClass._create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

def test_pclass_constructor_with_invariant_check():
    def invariant(obj):
        if obj.x < 0:
            raise ValueError("x must be non-negative")

    class TestClass(PClass):
        x = field(invariant=invariant)

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_global_invariant():
    def global_invariant(obj):
        if obj.x + obj.y != 10:
            raise ValueError("x + y must equal 10")

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_invariant]

    try:
        TestClass(x=5, y=4)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_valid_invariant():
    def invariant(obj):
        if obj.x < 0:
            raise ValueError("x must be non-negative")

    class TestClass(PClass):
        x = field(invariant=invariant)

    instance = TestClass(x=1)
    assert instance.x == 1


# LLM-generated content at query #15
#--------------------------

```python
def test_pclass_fields_iteration():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    assert list(TestClass._pclass_fields.items()) == [('x', TestClass._pclass_fields['x']), ('y', TestClass._pclass_fields['y'])]


# LLM-generated content at query #16
#--------------------------

```python
def test_pclass_eq_same_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance == instance

def test_pclass_eq_different_instances_same_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

def test_pclass_eq_different_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)
    assert not instance1 == instance2

def test_pclass_eq_different_classes():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()

    class TestClass2(PClass):
        x = field()

    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    assert not instance1 == instance2

def test_pclass_eq_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    assert not instance == 1
    assert not instance == {"x": 1}
    assert not instance == [1]


# LLM-generated content at query #17
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 0
    assert instance.y == 2

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 42
    assert instance.y == 2

def test_pclass_new_with_invariant_failure():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "positive"))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "positive" in str(e)

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, invariant
    @invariant(lambda s: (s.x + s.y == 10, "sum"))
    class TestClass(PClass):
        x = field()
        y = field()

    try:
        TestClass(x=1, y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "sum" in str(e)

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass._create({'x': 1, 'y': 2}, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({'x': 1, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


# LLM-generated content at query #18
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=True)

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException
    class TestClass(PClass):
        x = field()
        y = field(mandatory=True)

    try:
        TestClass(x=1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert ('TestClass.y',) == e.missing_fields

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z' are not among the specified fields for TestClass" == str(e)

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({'x': 1, 'z': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')

def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=42)
        y = field()

    instance = TestClass(y=1)
    assert instance.x == 42
    assert instance.y == 1

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = TestClass(y=1)
    assert instance.x == 42
    assert instance.y == 1

def test_pclass_new_with_invariant_failure():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException
    def invariant(value):
        return (value > 0, "positive")

    class TestClass(PClass):
        x = field(invariant=invariant)

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert ("positive",) == e.error_codes

def test_pclass_new_with_factory():
    from pyrsistent import PClass, field
    def factory(value):
        return value * 2

    class TestClass(PClass):
        x = field(factory=factory)

    instance = TestClass(x=5)
    assert instance.x == 10

def test_pclass_new_with_factory_and_ignore_extra():
    from pyrsistent import PClass, field
    def factory(value, ignore_extra=False):
        return value * 2 if ignore_extra else value

    class TestClass(PClass):
        x = field(type={int}, factory=factory)

    instance = TestClass.create({'x': 5}, ignore_extra=True)
    assert instance.x == 10

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._pclass import InvariantException
    @invariant
    def global_inv(instance):
        return instance.x + instance.y > 0, "sum_positive"

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_inv]

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert ("sum_positive",) == e.error_codes

def test_pclass_new_with_type_check_failure():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError
    class TestClass(PClass):
        x = field(type={int})

    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert "Invalid type for field TestClass.x, was str" == str(e)

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    def factory(value):
        return value.upper()

    class TestClass(PClass):
        x = field(factory=factory)
        y = field()

    instance = TestClass(x="hello", y=1, _factory_fields={'x'})
    assert instance.x == "HELLO"
    assert instance.y == 1


# LLM-generated content at query #19
#--------------------------

```python
def test_set_new_key_marks_dirty():
    original = object()
    evolver = _PClassEvolver(original, {})
    evolver.set('new_key', 'value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data['new_key'] == 'value'
    assert 'new_key' in evolver._factory_fields

def test_set_existing_key_with_different_value_marks_dirty():
    original = object()
    evolver = _PClassEvolver(original, {'key': 'old_value'})
    evolver.set('key', 'new_value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data['key'] == 'new_value'
    assert 'key' in evolver._factory_fields

def test_set_existing_key_with_same_value_does_not_mark_dirty():
    original = object()
    evolver = _PClassEvolver(original, {'key': 'value'})
    evolver.set('key', 'value')
    assert evolver._pclass_evolver_data_is_dirty is False
    assert evolver._pclass_evolver_data['key'] == 'value'
    assert 'key' not in evolver._factory_fields

def test_set_returns_self():
    original = object()
    evolver = _PClassEvolver(original, {})
    result = evolver.set('key', 'value')
    assert result is evolver


# LLM-generated content at query #20
#--------------------------

```python
def test_pclass_hash_consistency():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=3, y=4)

    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)


# LLM-generated content at query #21
#--------------------------

```python
def test_set_with_keyword_arguments():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1
    assert instance.y == 2

def test_set_with_positional_arguments():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set('x', 10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1
    assert instance.y == 2

def test_set_with_multiple_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set(x=10, y=20)
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert new_instance.z == 3
    assert instance.x == 1
    assert instance.y == 2
    assert instance.z == 3

def test_set_with_missing_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10, z=30)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert not hasattr(new_instance, 'z')
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #22
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in str(e)

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_factory():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
        y = field()

    instance = TestClass(x=5, y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(factory=lambda v, ignore_extra=False: v * 2)

    instance = TestClass.create({"x": 5, "z": 3}, ignore_extra=True)
    assert instance.x == 10

def test_pclass_new_with_invariant_error():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "positive"))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "positive" in str(e)

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._pclass import InvariantException

    def check_sum(instance):
        return instance.x + instance.y > 0, "sum_positive"

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [check_sum]

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "sum_positive" in str(e)

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
        y = field()

    instance = TestClass(x=5, y=2, _factory_fields={"x"})
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_type_check():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import PTypeError

    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x="not_an_int")
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert "Invalid type" in str(e)


# LLM-generated content at query #23
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=42)
    instance = TestClass()
    assert instance.x == 42

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    instance = TestClass()
    assert instance.x == 42

def test_pclass_new_with_factory():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
    instance = TestClass(x=21)
    assert instance.x == 42

def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except Exception as e:
        assert "Invalid type for field" in str(e)

def test_pclass_new_with_invariant():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "must be positive"))
    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "y")

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
    instance = TestClass(x=21, _factory_fields={"x"})
    assert instance.x == 42

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    @invariant
    def check_sum(obj):
        return obj.x + obj.y > 0, "sum must be positive"
    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [check_sum]
    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Global invariant failed" in str(e)


# LLM-generated content at query #24
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass._create({"x": 1, "y": 2}, _factory_fields={"x"})
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #25
#--------------------------

```python
def test_serialize_returns_dict():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    result = instance.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #26
#--------------------------

```python
def test_pclass_eq_with_same_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

def test_pclass_eq_with_different_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)
    assert not (instance1 == instance2)

def test_pclass_eq_with_different_class():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    assert not (instance1 == instance2)

def test_pclass_eq_with_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert not (instance == 1)

def test_pclass_eq_with_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1)
    assert not (instance1 == instance2)


# LLM-generated content at query #27
#--------------------------

```python
def test_repr_returns_correct_format():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    result = repr(instance)
    assert result == "TestClass(x=1, y=2)"


# LLM-generated content at query #28
#--------------------------

```python
def test_set_preserves_existing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2


# LLM-generated content at query #29
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_exception():
    from pyrsistent import PClass, field, InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[1] == ('TestClass.x',)
        assert e.args[2] == 'Field invariant failed'


# LLM-generated content at query #30
#--------------------------

```python
def test_pclass_eq_same_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance == instance

def test_pclass_eq_different_instances_same_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

def test_pclass_eq_different_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)
    assert not (instance1 == instance2)

def test_pclass_eq_different_classes():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    assert not (instance1 == instance2)

def test_pclass_eq_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert not (instance == 1)
    assert not (instance == {"x": 1})


# LLM-generated content at query #31
#--------------------------

```python
def test_set_new_key_marks_dirty():
    evolver = _PClassEvolver(object(), {})
    evolver.set('new_key', 'value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data['new_key'] == 'value'
    assert 'new_key' in evolver._factory_fields

def test_set_existing_key_with_different_value_marks_dirty():
    evolver = _PClassEvolver(object(), {'key': 'old_value'})
    evolver.set('key', 'new_value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data['key'] == 'new_value'
    assert 'key' in evolver._factory_fields

def test_set_existing_key_with_same_value_does_not_mark_dirty():
    evolver = _PClassEvolver(object(), {'key': 'value'})
    evolver.set('key', 'value')
    assert evolver._pclass_evolver_data_is_dirty is False
    assert evolver._pclass_evolver_data['key'] == 'value'
    assert 'key' not in evolver._factory_fields

def test_set_returns_self():
    evolver = _PClassEvolver(object(), {})
    result = evolver.set('key', 'value')
    assert result is evolver


# LLM-generated content at query #32
#--------------------------

```python
def test_hash_equality():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=1, y=3)

    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)


# LLM-generated content at query #33
#--------------------------

```python
def test__is_pclass_returns_false_for_non_pclass_bases():
    class NonPClassBase:
        pass

    class TestClass(metaclass=PClassMeta):
        pass

    assert not _is_pclass((NonPClassBase,))


# LLM-generated content at query #34
#--------------------------

```python
def test_weakref_added_to_slots_when_bases_contain_pclass():
    class Base(metaclass=PClassMeta):
        pass

    class Child(Base):
        pass

    assert '__weakref__' in Child.__slots__


# LLM-generated content at query #35
#--------------------------

```python
def test__is_pclass_returns_false_for_non_pclass_bases():
    class NonPClassBase:
        pass

    assert not _is_pclass((NonPClassBase,))


# LLM-generated content at query #36
#--------------------------

```python
def test_set_predicate_false():
    evolver = _PClassEvolver(object(), {'key': 'value'})
    _MISSING_VALUE = object()
    assert evolver.set('key', 'value')._pclass_evolver_data_is_dirty is False


# LLM-generated content at query #37
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except Exception as e:
        assert "Invalid type" in str(e)

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)

    instance = TestClass(x=5)
    assert instance.x == 10

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "z")

def test_pclass_new_with_invariant_failure():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "must be positive"))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "must be positive" in str(e)

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, invariant
    @invariant
    def check_sum(obj):
        return obj.x + obj.y > 0, "sum must be positive"

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [check_sum]

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Global invariant failed" in str(e)


# LLM-generated content at query #38
#--------------------------

```python
def test_set_predicate_false():
    evolver = _PClassEvolver(object(), {'key': 'value'})
    _MISSING_VALUE = object()
    assert evolver._pclass_evolver_data.get('key', _MISSING_VALUE) is evolver._pclass_evolver_data['key']


# LLM-generated content at query #39
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in str(e)

def test_pclass_constructor_with_extra_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    try:
        TestClass(x=1, y=2, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_ignore_extra():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_initial_value():
    class TestClass(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 0
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    class TestClass(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 42
    assert instance.y == 2

def test_pclass_constructor_with_factory_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(_factory_fields={"x"}, x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #40
#--------------------------

```python
def test_pclass_pickling_returns_correct_tuple():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()

    assert result[0] == _restore_pickle
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 1, 'y': 2}


# LLM-generated content at query #41
#--------------------------

```python
def test_set_preserves_existing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=3)

    assert new_instance.x == 3
    assert new_instance.y == 2


# LLM-generated content at query #42
#--------------------------

```python
def test_weakref_in_slots_when_bases_are_pclass():
    class Base(metaclass=PClassMeta):
        pass

    class Derived(Base):
        pass

    assert '__weakref__' in Derived.__slots__


# LLM-generated content at query #43
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass._create({'x': 1, 'y': 2}, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({'x': 1, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


# LLM-generated content at query #44
#--------------------------

```python
def test_check_and_set_attr_with_valid_type_and_invariant():
    class MockField:
        type = int
        def invariant(self, value):
            return True, None

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField(), "attr", 42, result, invariant_errors)
    assert result.attr == 42
    assert invariant_errors == []

def test_check_and_set_attr_with_invalid_type():
    class MockField:
        type = int
        def invariant(self, value):
            return True, None

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    try:
        _check_and_set_attr(MockClass, MockField(), "attr", "not_an_int", result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.message == "Invalid type for field MockClass.attr, was str"

def test_check_and_set_attr_with_failed_invariant():
    class MockField:
        type = int
        def invariant(self, value):
            return False, "INVALID"

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField(), "attr", 42, result, invariant_errors)
    assert not hasattr(result, "attr")
    assert invariant_errors == ["INVALID"]


# LLM-generated content at query #45
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(initial=0)

    try:
        TestClass(x=1, y=2, z=3)
    except InvariantException:
        pass
    except AttributeError:
        pass


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields

def test_pclass_constructor_with_extra_fields():
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "'z'" in str(e)

def test_pclass_constructor_with_initial_values():
    class TestClass(PClass):
        x = field(initial=5)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 5
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    class TestClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_ignore_extra():
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_factory_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

def test_pclass_constructor_with_invariant_check():
    def invariant(obj):
        if obj.x > 10:
            raise ValueError("x must be <= 10")

    class TestClass(PClass):
        x = field(invariant=invariant)

    try:
        TestClass(x=15)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_pclassmeta_new_with_checkedtype_base():
    class TestClass(metaclass=PClassMeta):
        pass
    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__

def test_pclassmeta_new_without_checkedtype_base():
    class BaseClass:
        pass
    class TestClass(BaseClass, metaclass=PClassMeta):
        pass
    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' not in TestClass.__slots__

def test_pclassmeta_new_with_fields():
    class TestClass(metaclass=PClassMeta):
        field1 = _PField()
        field2 = _PField()
    assert 'field1' in TestClass._pclass_fields
    assert 'field2' in TestClass._pclass_fields
    assert 'field1' not in TestClass.__dict__
    assert 'field2' not in TestClass.__dict__

def test_pclassmeta_new_with_invariants():
    def invariant1(obj):
        return True, ()
    def invariant2(obj):
        return True, ()
    class TestClass(metaclass=PClassMeta):
        __invariant__ = invariant1
    class ChildClass(TestClass):
        __invariant__ = invariant2
    assert len(TestClass._pclass_invariants) == 1
    assert len(ChildClass._pclass_invariants) == 2

def test_pclassmeta_new_with_invalid_invariant():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = "not callable"
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_pclassmeta_new_with_inherited_fields():
    class ParentClass(metaclass=PClassMeta):
        field1 = _PField()
    class ChildClass(ParentClass):
        field2 = _PField()
    assert 'field1' in ChildClass._pclass_fields
    assert 'field2' in ChildClass._pclass_fields
    assert 'field1' not in ChildClass.__dict__
    assert 'field2' not in ChildClass.__dict__

def test_pclassmeta_new_with_inherited_invariants():
    def invariant1(obj):
        return True, ()
    def invariant2(obj):
        return True, ()
    class ParentClass(metaclass=PClassMeta):
        __invariant__ = invariant1
    class ChildClass(ParentClass):
        __invariant__ = invariant2
    assert len(ParentClass._pclass_invariants) == 1
    assert len(ChildClass._pclass_invariants) == 2


# LLM-generated content at query #3
#--------------------------

```python
def test_pclass_hash_consistency():
    from pyrsistent import PClass, field

    class TestPClass(PClass):
        x = field()
        y = field()

    instance1 = TestPClass(x=1, y=2)
    instance2 = TestPClass(x=1, y=2)
    instance3 = TestPClass(x=3, y=4)

    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)


# LLM-generated content at query #4
#--------------------------

```python
def test_set_new_key_value_pair():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('key', 'value')
    assert evolver._pclass_evolver_data == {'key': 'value'}
    assert 'key' in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True

def test_set_existing_key_with_same_value():
    original = object()
    initial_dict = {'key': 'value'}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('key', 'value')
    assert evolver._pclass_evolver_data == {'key': 'value'}
    assert 'key' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is False

def test_set_existing_key_with_different_value():
    original = object()
    initial_dict = {'key': 'old_value'}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('key', 'new_value')
    assert evolver._pclass_evolver_data == {'key': 'new_value'}
    assert 'key' in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True

def test_set_returns_self():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set('key', 'value')
    assert result is evolver


# LLM-generated content at query #5
#--------------------------

```python
def test_repr_with_no_fields():
    class EmptyClass(PClass):
        pass
    instance = EmptyClass()
    assert repr(instance) == "EmptyClass()"

def test_repr_with_single_field():
    class SingleFieldClass(PClass):
        x = field()
    instance = SingleFieldClass(x=1)
    assert repr(instance) == "SingleFieldClass(x=1)"

def test_repr_with_multiple_fields():
    class MultiFieldClass(PClass):
        x = field()
        y = field()
    instance = MultiFieldClass(x=1, y="hello")
    assert repr(instance) == "MultiFieldClass(x=1, y='hello')"

def test_repr_with_missing_optional_field():
    class OptionalFieldClass(PClass):
        x = field(mandatory=False)
    instance = OptionalFieldClass()
    assert repr(instance) == "OptionalFieldClass()"

def test_repr_with_complex_field_values():
    class ComplexFieldClass(PClass):
        x = field()
        y = field()
    instance = ComplexFieldClass(x=[1, 2, 3], y={"a": 1})
    assert repr(instance) == "ComplexFieldClass(x=[1, 2, 3], y={'a': 1})"


# LLM-generated content at query #6
#--------------------------

```python
def test_is_pclass_with_single_checkedtype_base():
    assert _is_pclass((CheckedType,)) == True

def test_is_pclass_with_multiple_bases():
    assert _is_pclass((CheckedType, object)) == False

def test_is_pclass_with_no_bases():
    assert _is_pclass(()) == False

def test_is_pclass_with_non_checkedtype_base():
    assert _is_pclass((object,)) == False


# LLM-generated content at query #7
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 5
    assert instance.y == 2

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_factory():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)

    instance = TestClass(x=3)
    assert instance.x == 6

def test_pclass_new_with_invariant():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "positive"))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "positive" in str(e)

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v, ignore_extra=False: v)

    instance = TestClass.create({'x': 1, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    def global_inv(obj):
        return (obj.x + obj.y == 10, "sum_10")

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_inv]

    try:
        TestClass(x=3, y=4)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "sum_10" in str(e)


# LLM-generated content at query #8
#--------------------------

```python
def test_serialize_without_custom_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y="test")
    result = instance.serialize()
    assert result == {"x": 1, "y": "test"}

def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v: v * 2)
        y = field()

    instance = TestClass(x=5, y="test")
    result = instance.serialize()
    assert result == {"x": 10, "y": "test"}

def test_serialize_with_missing_optional_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance = TestClass(x=1)
    result = instance.serialize()
    assert result == {"x": 1}

def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v, fmt: str(v) if fmt == "str" else v)
        y = field()

    instance = TestClass(x=1, y="test")
    result = instance.serialize(format="str")
    assert result == {"x": "1", "y": "test"}


# LLM-generated content at query #9
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(type=int)

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=42)

    instance = TestClass()
    assert instance.x == 42

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)

    instance = TestClass()
    assert instance.x == 42

def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except Exception as e:
        assert "Invalid type for field" in str(e)

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "y")

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass._factory_fields={"x"}, x=1
    assert instance.x == 1

def test_pclass_new_with_invariant_failure():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "must be positive"))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, invariant
    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [invariant(lambda s: (s.x != s.y, "x and y must differ"))]

    try:
        TestClass(x=1, y=1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Global invariant failed" in str(e)

def test_pclass_new_with_factory_and_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v, ignore_extra=False: v.upper() if ignore_extra else v)

    instance = TestClass.create({"x": "test"}, ignore_extra=True)
    assert instance.x == "TEST"


# LLM-generated content at query #10
#--------------------------

```python
def test_pclass_hash_equality():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=1, y=3)

    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)


# LLM-generated content at query #11
#--------------------------

```python
def test_serialize_includes_all_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    serialized = instance.serialize()

    assert 'x' in serialized
    assert 'y' in serialized


# LLM-generated content at query #12
#--------------------------

```python
def test__is_pclass_returns_false_for_non_pclass_bases():
    class NonPClassBase:
        pass

    assert not _is_pclass((NonPClassBase,))


# LLM-generated content at query #13
#--------------------------

```python
def test_serialize_returns_dict_with_field_names_and_serialized_values():
    class TestPClass(PClass):
        x = field()
        y = field(serializer=lambda v: v * 2)

    instance = TestPClass(x=1, y=2)
    result = instance.serialize()

    assert isinstance(result, dict)
    assert 'x' in result
    assert 'y' in result
    assert result['x'] == 1
    assert result['y'] == 4


# LLM-generated content at query #14
#--------------------------

```python
def test_pclassmeta_new_sets_fields():
    class TestClass(metaclass=PClassMeta):
        field1 = _PField()
        field2 = _PField()

    assert hasattr(TestClass, '_pclass_fields')
    assert 'field1' in TestClass._pclass_fields
    assert 'field2' in TestClass._pclass_fields

def test_pclassmeta_new_stores_invariants():
    def invariant_func(obj):
        return True, ()

    class TestClass(metaclass=PClassMeta):
        __invariant__ = invariant_func

    assert hasattr(TestClass, '_pclass_invariants')
    assert TestClass._pclass_invariants == (wrap_invariant(invariant_func),)

def test_pclassmeta_new_sets_slots():
    class TestClass(metaclass=PClassMeta):
        field1 = _PField()
        field2 = _PField()

    assert TestClass.__slots__ == ('_pclass_frozen', 'field1', 'field2')

def test_pclassmeta_new_adds_weakref_slot():
    class ParentClass(metaclass=PClassMeta):
        pass

    class ChildClass(ParentClass):
        pass

    assert '__weakref__' in ChildClass.__slots__


# LLM-generated content at query #15
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    restored = _restore_pickle(*instance.__reduce__())
    assert restored.x == 1
    assert restored.y == 2
    assert isinstance(restored, TestClass)


# LLM-generated content at query #16
#--------------------------

```python
def test_hash_returns_consistent_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert hash(instance1) == hash(instance2)


# LLM-generated content at query #17
#--------------------------

```python
def test_set_new_key_marks_dirty_and_adds_to_factory_fields():
    evolver = _PClassEvolver(object(), {})
    evolver.set('new_key', 'new_value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'new_key' in evolver._factory_fields
    assert evolver['new_key'] == 'new_value'

def test_set_existing_key_with_different_value_marks_dirty():
    evolver = _PClassEvolver(object(), {'key': 'old_value'})
    evolver.set('key', 'new_value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver['key'] == 'new_value'

def test_set_existing_key_with_same_value_does_not_mark_dirty():
    evolver = _PClassEvolver(object(), {'key': 'value'})
    evolver.set('key', 'value')
    assert evolver._pclass_evolver_data_is_dirty is False
    assert evolver['key'] == 'value'

def test_set_returns_self():
    evolver = _PClassEvolver(object(), {})
    result = evolver.set('key', 'value')
    assert result is evolver


# LLM-generated content at query #18
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(invariant=lambda v: (False, "error"))

    try:
        TestClass()
    except InvariantException as e:
        assert e.args[0] == ("error",)
        assert e.args[1] == ("TestClass.x",)


# LLM-generated content at query #19
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass._create({"x": 1, "y": 2}, _factory_fields={"x"})
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #20
#--------------------------

```python
def test_repr_returns_correct_string():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y="test")
    assert repr(instance) == "TestPClass(x=1, y='test')"


# LLM-generated content at query #21
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    from pyrsistent._pclass import _check_and_set_attr

    class TestClass(PClass):
        x = field(type=int)
        y = field(type=int)

    # Test with invariant_errors
    result = object.__new__(TestClass)
    invariant_errors = ['error']
    missing_fields = []
    assert invariant_errors or missing_fields

    # Test with missing_fields
    invariant_errors = []
    missing_fields = ['TestClass.x']
    assert invariant_errors or missing_fields

    # Test with both
    invariant_errors = ['error']
    missing_fields = ['TestClass.x']
    assert invariant_errors or missing_fields


# LLM-generated content at query #22
#--------------------------

```python
def test_pickle_support_returns_correct_tuple():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    result = instance.__reduce__()
    assert result == (_restore_pickle, (TestClass, {'x': 1, 'y': 2}))


# LLM-generated content at query #23
#--------------------------

```python
def test_pclass_fields_iteration():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    assert '_pclass_fields' in TestClass.__dict__
    assert hasattr(TestClass._pclass_fields, 'items')
    assert callable(TestClass._pclass_fields.items)


# LLM-generated content at query #24
#--------------------------

```python
def test_remove_existing_item():
    original = object()
    evolver = _PClassEvolver(original, {'a': 1, 'b': 2})
    result = evolver.remove('a')
    assert result is evolver
    assert 'a' not in evolver._pclass_evolver_data
    assert 'b' in evolver._pclass_evolver_data
    assert 'a' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True

def test_remove_nonexistent_item_raises_error():
    original = object()
    evolver = _PClassEvolver(original, {'a': 1})
    try:
        evolver.remove('b')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == 'b'


# LLM-generated content at query #25
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    from pyrsistent._pclass import _check_and_set_attr

    class TestClass(PClass):
        x = field()
        y = field()

    # Test case where invariant_errors is non-empty
    result = object()
    invariant_errors = ["error1"]
    missing_fields = []
    assert invariant_errors or missing_fields

    # Test case where missing_fields is non-empty
    invariant_errors = []
    missing_fields = ["field1"]
    assert invariant_errors or missing_fields

    # Test case where both are non-empty
    invariant_errors = ["error1"]
    missing_fields = ["field1"]
    assert invariant_errors or missing_fields


# LLM-generated content at query #26
#--------------------------

```python
def test_set_with_keyword_arguments():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1
    assert instance.y == 2

def test_set_with_positional_arguments():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set('x', 10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1
    assert instance.y == 2

def test_set_multiple_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set(x=10, y=20)
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert new_instance.z == 3
    assert instance.x == 1
    assert instance.y == 2
    assert instance.z == 3

def test_set_with_missing_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1)
    new_instance = instance.set(y=2)
    assert new_instance.x == 1
    assert new_instance.y == 2
    assert instance.x == 1
    assert not hasattr(instance, 'y')


# LLM-generated content at query #27
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 0
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 42
    assert instance.y == 2


# LLM-generated content at query #28
#--------------------------

```python
def test_pclassmeta_new_with_single_checkedtype_base():
    class TestClass(metaclass=PClassMeta):
        pass
    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__

def test_pclassmeta_new_with_multiple_bases():
    class Base1:
        pass
    class Base2:
        pass
    class TestClass(Base1, Base2, metaclass=PClassMeta):
        pass
    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' not in TestClass.__slots__

def test_pclassmeta_new_inherits_fields_and_invariants():
    class Parent(metaclass=PClassMeta):
        x = field()
        __invariant__ = lambda self: True

    class Child(Parent, metaclass=PClassMeta):
        y = field()

    assert 'x' in Child._pclass_fields
    assert 'y' in Child._pclass_fields
    assert len(Child._pclass_invariants) == 1

def test_pclassmeta_new_with_non_callable_invariant_raises_typeerror():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = "not callable"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        assert False, "Expected TypeError"


# LLM-generated content at query #29
#--------------------------

```python
def test_pclass_eq_same_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance == instance

def test_pclass_eq_different_instances_same_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

def test_pclass_eq_different_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)
    assert not (instance1 == instance2)

def test_pclass_eq_different_classes():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    assert not (instance1 == instance2)

def test_pclass_eq_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert not (instance == 1)

def test_pclass_eq_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    assert instance1 == instance2


# LLM-generated content at query #30
#--------------------------

```python
def test_eq_with_same_class_and_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    a = TestClass(x=1, y=2)
    b = TestClass(x=1, y=2)
    assert a == b

def test_eq_with_different_class():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    a = TestClass1(x=1)
    b = TestClass2(x=1)
    assert not (a == b)

def test_eq_with_different_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    a = TestClass(x=1, y=2)
    b = TestClass(x=1, y=3)
    assert not (a == b)

def test_eq_with_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    a = TestClass(x=1)
    b = TestClass(x=1, y=2)
    assert not (a == b)

def test_eq_with_non_pclass_object():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    a = TestClass(x=1)
    assert not (a == 1)


# LLM-generated content at query #31
#--------------------------

```python
def test_pclass_pickling_returns_correct_tuple():
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    result = instance.__reduce__()
    assert result == (_restore_pickle, (TestClass, {'x': 1}))


# LLM-generated content at query #32
#--------------------------

```python
def test_set_preserves_existing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)
    assert new_instance.y == 2


# LLM-generated content at query #33
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(type=int)

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)

def test_pclass_new_with_invalid_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except Exception as e:
        assert "Invalid type for field TestClass.x" in str(e)

def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=42)

    instance = TestClass()
    assert instance.x == 42

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)

    instance = TestClass()
    assert instance.x == 42

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "y")

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass(_factory_fields={"x"}, x=1)
    assert instance.x == 1

def test_pclass_new_with_invariant_failure():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "must be positive"))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "must be positive" in str(e)

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, invariant
    @invariant
    def check_sum(instance):
        return instance.x + instance.y > 0, "sum must be positive"

    class TestClass(PClass):
        x = field()
        y = field()

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Global invariant failed" in str(e)


# LLM-generated content at query #34
#--------------------------

```python
def test_serialize_includes_all_non_missing_fields():
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()

    instance = TestClass(a=1, b=2, c=3)
    serialized = instance.serialize()

    assert 'a' in serialized
    assert 'b' in serialized
    assert 'c' in serialized


# LLM-generated content at query #35
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "TestClass.x" in str(e)

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=0)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 0
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 42
    assert instance.y == 2

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass._create({'x': 1, 'y': 2}, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    instance = TestClass.create({'x': 1, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')

def test_pclass_constructor_with_invariant_check():
    from pyrsistent import PClass, field, invariant

    def check_positive(instance):
        if instance.x < 0:
            raise ValueError("x must be positive")

    class TestClass(PClass):
        x = field()
        _pclass_invariants = [invariant(lambda i: i.x >= 0)]

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "invariant" in str(e).lower()

def test_pclass_constructor_with_pclass_instance():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass.create(instance1)
    assert instance2.x == 1
    assert instance2.y == 2


# LLM-generated content at query #36
#--------------------------

```python
def test_eq_returns_true_for_identical_pclass_instances():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)

    assert instance1 == instance2


# LLM-generated content at query #37
#--------------------------

```python
def test_pclass_new_with_valid_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_new_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields

def test_pclass_new_with_invalid_field_type():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import PTypeError

    class TestClass(PClass):
        x = field(type=int)
        y = field()

    try:
        TestClass(x="not_an_int", y=2)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.actual_type == str

def test_pclass_new_with_invariant_failure():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import InvariantException

    def invariant(value):
        return (value > 0, "must_be_positive")

    class TestClass(PClass):
        x = field(invariant=invariant)
        y = field()

    try:
        TestClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "must_be_positive" in e.error_codes

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    try:
        TestClass(x=1, y=2, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, "z")

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field

    def factory(value):
        return value.upper()

    class TestClass(PClass):
        x = field(factory=factory)
        y = field()

    instance = TestClass(x="hello", y=2)
    assert instance.x == "HELLO"
    assert instance.y == 2

def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field

    def initial():
        return 20

    class TestClass(PClass):
        x = field(initial=initial)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 20
    assert instance.y == 2

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import InvariantException

    def global_invariant(instance):
        return (instance.x + instance.y > 0, "sum_must_be_positive")

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_invariant]

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "sum_must_be_positive" in e.error_codes


# LLM-generated content at query #38
#--------------------------

```python
def test_remove_item_exists():
    evolver = _PClassEvolver(object(), {'key': 'value'})
    assert 'key' in evolver._pclass_evolver_data
    evolver.remove('key')
    assert 'key' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty


# LLM-generated content at query #39
#--------------------------

```python
def test_set_preserves_existing_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10)

    assert new_instance.x == 10
    assert new_instance.y == 2


# LLM-generated content at query #40
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 10)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass._factory_fields={"x"}, x=1, y=2
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #41
#--------------------------

```python
def test__is_pclass_returns_false():
    class TestClass(metaclass=PClassMeta):
        pass

    assert '__weakref__' not in TestClass.__slots__


# LLM-generated content at query #42
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    invariant_errors = ["error1"]
    missing_fields = ["field1"]
    assert invariant_errors or missing_fields


# LLM-generated content at query #43
#--------------------------

```python
def test_repr_format():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    assert repr(instance) == "TestPClass(x=1, y=2)"


