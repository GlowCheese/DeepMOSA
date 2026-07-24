####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_set_new_key_marks_dirty():
    evolver = _PClassEvolver(object(), {})
    evolver.set('new_key', 'new_value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._pclass_evolver_data['new_key'] == 'new_value'
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


# LLM-generated content at query #2
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

def test_set_preserves_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance = TestClass(x=1)
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert not hasattr(new_instance, 'y')
    assert instance.x == 1
    assert not hasattr(instance, 'y')


# LLM-generated content at query #3
#--------------------------

```python
def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field, serialize

    class CustomSerializer:
        def __call__(self, value):
            return f"serialized_{value}"

    class TestClass(PClass):
        x = field(serializer=CustomSerializer())

    instance = TestClass(x=10)
    result = instance.serialize()
    assert result == {"x": "serialized_10"}

def test_serialize_without_serializer():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    instance = TestClass(x=10)
    result = instance.serialize()
    assert result == {"x": 10}

def test_serialize_with_missing_value():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=10)
    result = instance.serialize()
    assert result == {"x": 10}

def test_serialize_with_format():
    from pyrsistent import PClass, field, serialize

    class TestClass(PClass):
        x = field()

    instance = TestClass(x=10)
    result = instance.serialize(format="json")
    assert result == {"x": serialize(None, "json", 10)}


# LLM-generated content at query #4
#--------------------------

```python
def test_check_and_set_attr_with_valid_type_and_invariant():
    class TestClass:
        pass

    class Field:
        type = int
        def invariant(self, value):
            return True, None

    result = TestClass()
    invariant_errors = []
    _check_and_set_attr(TestClass, Field(), "test_field", 42, result, invariant_errors)
    assert result.test_field == 42
    assert invariant_errors == []

def test_check_and_set_attr_with_invalid_type():
    class TestClass:
        pass

    class Field:
        type = int
        def invariant(self, value):
            return True, None

    result = TestClass()
    invariant_errors = []
    try:
        _check_and_set_attr(TestClass, Field(), "test_field", "not_an_int", result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.cls == TestClass
        assert e.field_name == "test_field"
        assert e.expected_type == int
        assert e.actual_type == str

def test_check_and_set_attr_with_failed_invariant():
    class TestClass:
        pass

    class Field:
        type = int
        def invariant(self, value):
            return False, "INVALID"

    result = TestClass()
    invariant_errors = []
    _check_and_set_attr(TestClass, Field(), "test_field", 42, result, invariant_errors)
    assert not hasattr(result, "test_field")
    assert invariant_errors == ["INVALID"]


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

```python
def test_serialize_without_custom_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y="test")
    assert instance.serialize() == {"x": 1, "y": "test"}

def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field
    def custom_serializer(value):
        return str(value).upper()

    class TestClass(PClass):
        x = field(serializer=custom_serializer)
        y = field()

    instance = TestClass(x="hello", y=123)
    assert instance.serialize() == {"x": "HELLO", "y": 123}

def test_serialize_with_missing_optional_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance = TestClass(x=1)
    assert instance.serialize() == {"x": 1}

def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y="test")
    assert instance.serialize(format="json") == {"x": 1, "y": "test"}


# LLM-generated content at query #7
#--------------------------

```python
def test_pclassmeta_new_with_single_checkedtype_base():
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: True
        field = _PField()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' in TestClass.__slots__
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'field' in TestClass.__slots__
    assert TestClass._pclass_fields['field'] is field
    assert TestClass._pclass_invariants == (wrap_invariant(lambda self: True),)

def test_pclassmeta_new_with_multiple_bases():
    class Base1:
        __invariant__ = lambda self: True
        field1 = _PField()

    class Base2:
        __invariant__ = lambda self: False
        field2 = _PField()

    class TestClass(Base1, Base2, metaclass=PClassMeta):
        field3 = _PField()

    assert hasattr(TestClass, '_pclass_fields')
    assert hasattr(TestClass, '_pclass_invariants')
    assert hasattr(TestClass, '__slots__')
    assert '__weakref__' not in TestClass.__slots__
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'field1' in TestClass.__slots__
    assert 'field2' in TestClass.__slots__
    assert 'field3' in TestClass.__slots__
    assert TestClass._pclass_fields['field1'] is field1
    assert TestClass._pclass_fields['field2'] is field2
    assert TestClass._pclass_fields['field3'] is field3
    assert len(TestClass._pclass_invariants) == 2

def test_pclassmeta_new_with_non_callable_invariant():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = "not callable"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    else:
        assert False, "Expected TypeError"


# LLM-generated content at query #8
#--------------------------

```python
def test_pclass_hash_returns_consistent_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=20)
    assert hash(instance1) == hash(instance2)

def test_pclass_hash_different_for_different_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=21)
    assert hash(instance1) != hash(instance2)

def test_pclass_hash_with_missing_optional_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance1 = TestClass(x=10)
    instance2 = TestClass(x=10)
    assert hash(instance1) == hash(instance2)

def test_pclass_hash_with_different_optional_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance1 = TestClass(x=10)
    instance2 = TestClass(x=10, y=20)
    assert hash(instance1) != hash(instance2)


# LLM-generated content at query #9
#--------------------------

```python
def test_set_preserves_existing_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()

    a = AClass(x=1, y=2)
    a2 = a.set(x=3)
    assert a2.y == 2


# LLM-generated content at query #10
#--------------------------

```python
def test_pclass_eq_returns_true_for_equal_instances():
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

def test_pclass_eq_returns_false_for_different_instances():
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    assert not (instance1 == instance2)

def test_pclass_eq_returns_false_for_different_classes():
    class TestClass1(PClass):
        x = field()

    class TestClass2(PClass):
        x = field()

    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    assert not (instance1 == instance2)

def test_pclass_eq_returns_not_implemented_for_non_pclass():
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    assert instance == "not_a_pclass" is NotImplemented


# LLM-generated content at query #11
#--------------------------

```python
def test_set_new_key_marks_dirty_and_adds_to_factory_fields():
    evolver = _PClassEvolver(object(), {})
    result = evolver.set('new_key', 'new_value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'new_key' in evolver._factory_fields
    assert evolver._pclass_evolver_data['new_key'] == 'new_value'
    assert result is evolver

def test_set_existing_key_with_different_value_marks_dirty():
    evolver = _PClassEvolver(object(), {'existing_key': 'old_value'})
    result = evolver.set('existing_key', 'new_value')
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'existing_key' in evolver._factory_fields
    assert evolver._pclass_evolver_data['existing_key'] == 'new_value'
    assert result is evolver

def test_set_existing_key_with_same_value_does_not_mark_dirty():
    evolver = _PClassEvolver(object(), {'existing_key': 'same_value'})
    result = evolver.set('existing_key', 'same_value')
    assert evolver._pclass_evolver_data_is_dirty is False
    assert 'existing_key' not in evolver._factory_fields
    assert evolver._pclass_evolver_data['existing_key'] == 'same_value'
    assert result is evolver


# LLM-generated content at query #12
#--------------------------

```python
def test_repr_empty_pclass():
    class EmptyPClass(PClass):
        pass

    instance = EmptyPClass()
    assert repr(instance) == "EmptyPClass()"

def test_repr_single_field():
    class SingleFieldPClass(PClass):
        x = field()

    instance = SingleFieldPClass(x=42)
    assert repr(instance) == "SingleFieldPClass(x=42)"

def test_repr_multiple_fields():
    class MultiFieldPClass(PClass):
        x = field()
        y = field()
        z = field()

    instance = MultiFieldPClass(x=1, y="hello", z=[1, 2, 3])
    assert repr(instance) == "MultiFieldPClass(x=1, y='hello', z=[1, 2, 3])"

def test_repr_with_missing_optional_field():
    class OptionalFieldPClass(PClass):
        x = field(mandatory=True)
        y = field()

    instance = OptionalFieldPClass(x=10)
    assert repr(instance) == "OptionalFieldPClass(x=10)"

def test_repr_with_string_escaping():
    class StringFieldPClass(PClass):
        name = field()

    instance = StringFieldPClass(name="O'Reilly")
    assert repr(instance) == "StringFieldPClass(name=\"O'Reilly\")"

def test_repr_with_nested_pclass():
    class InnerPClass(PClass):
        value = field()

    class OuterPClass(PClass):
        inner = field(type=InnerPClass)

    inner = InnerPClass(value=100)
    outer = OuterPClass(inner=inner)
    assert repr(outer) == "OuterPClass(inner=InnerPClass(value=100))"


# LLM-generated content at query #13
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
        assert "'z'" in str(e)

def test_pclass_new_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=42)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 42
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

def test_pclass_new_with_factory_and_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, factory=lambda v, ignore_extra=False: v * 2)

    instance = TestClass.create({"x": 5, "z": 10}, ignore_extra=True)
    assert instance.x == 10

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, invariant
    @invariant
    def check_sum(obj):
        return obj.x + obj.y > 0, "sum_positive"

    class TestClass(PClass):
        x = field()
        y = field()

    try:
        TestClass(x=-5, y=-3)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "sum_positive" in str(e)


# LLM-generated content at query #14
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
        x = field(initial=0)

    instance = TestClass()
    assert instance.x == 0

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

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass(_factory_fields={'x'}, x=1)
    assert instance.x == 1

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

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


# LLM-generated content at query #15
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
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
        x = field(initial=0)
        y = field()

    instance = TestPClass(y=2)
    assert instance.x == 0
    assert instance.y == 2

def test_pclass_constructor_with_callable_initial():
    class TestPClass(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = TestPClass(y=2)
    assert instance.x == 42
    assert instance.y == 2


# LLM-generated content at query #16
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

def test_pclass_constructor_with_mandatory_field_missing():
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
        x = field(initial=42)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 42
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


# LLM-generated content at query #17
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
        x = field(initial=42)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 42
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
        _pclass_invariants = [check_positive]

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "x must be positive" in str(e)


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_false():
    class TestClass(metaclass=PClassMeta):
        pass

    assert '__weakref__' not in TestClass.__slots__


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_false():
    class TestClass(metaclass=PClassMeta):
        pass

    assert '__weakref__' not in TestClass.__slots__


# LLM-generated content at query #20
#--------------------------

```python
def test__is_pclass_returns_false_for_non_pclass_bases():
    class NonPClass:
        pass

    bases = (NonPClass,)
    assert not _is_pclass(bases)


# LLM-generated content at query #21
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    from pyrsistent._field_common import PFIELD_NO_INITIAL

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass()
    except InvariantException:
        pass
    else:
        assert False, "Expected InvariantException for missing mandatory field"

    class TestClassWithInvariant(PClass):
        x = field()
        y = field()

        @classmethod
        def _pclass_invariant(cls, instance):
            return (instance.x != 0, "x cannot be 0")

    try:
        TestClassWithInvariant(x=0, y=1)
    except InvariantException:
        pass
    else:
        assert False, "Expected InvariantException for invariant violation"


# LLM-generated content at query #22
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, PFIELD_NO_INITIAL
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(type=int, mandatory=True)
        y = field(type=int, initial=lambda: 0)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[1] == ('TestClass.x',), "Expected missing field error for 'x'"


# LLM-generated content at query #23
#--------------------------

```python
def test_pclass_hash_returns_consistent_value():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=2, y=1)

    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)


# LLM-generated content at query #24
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

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

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

def test_pclass_new_with_invariant():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "must be positive"))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "must be positive" in str(e)

def test_pclass_new_with_global_invariant():
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
        assert "sum must be positive" in str(e)

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

    instance = TestClass._factory_fields={"x"}, x=21
    assert instance.x == 42


# LLM-generated content at query #25
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    restored = _restore_pickle(*obj.__reduce__())
    assert restored == obj
    assert restored.x == 1
    assert restored.y == 2


# LLM-generated content at query #26
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    from pyrsistent._field_common import PFIELD_NO_INITIAL

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(initial=0)

    try:
        TestClass(x=1, y=2, z=3)
    except InvariantException as e:
        assert e.error_codes == ('TestClass.x',)
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #27
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
    class TestClass(metaclass=PClassMeta):
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

    class TestSubClass(TestClass):
        __invariant__ = invariant2

    assert len(TestSubClass._pclass_invariants) == 2
    assert all(callable(inv) for inv in TestSubClass._pclass_invariants)

def test_pclassmeta_new_with_non_callable_invariant():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = "not callable"

        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_pclassmeta_new_with_inherited_fields():
    class BaseClass(metaclass=PClassMeta):
        field1 = _PField()

    class TestClass(BaseClass):
        field2 = _PField()

    assert 'field1' in TestClass._pclass_fields
    assert 'field2' in TestClass._pclass_fields
    assert 'field1' not in TestClass.__dict__
    assert 'field2' not in TestClass.__dict__

def test_pclassmeta_new_with_inherited_invariants():
    def invariant1(obj):
        return True, ()

    def invariant2(obj):
        return True, ()

    class BaseClass(metaclass=PClassMeta):
        __invariant__ = invariant1

    class TestClass(BaseClass):
        __invariant__ = invariant2

    assert len(TestClass._pclass_invariants) == 2
    assert all(callable(inv) for inv in TestClass._pclass_invariants)


# LLM-generated content at query #28
#--------------------------

```python
def test_serialize_with_no_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y="test")
    assert instance.serialize() == {"x": 1, "y": "test"}

def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v: str(v))
        y = field()

    instance = TestClass(x=1, y="test")
    assert instance.serialize() == {"x": "1", "y": "test"}

def test_serialize_with_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1)
    assert instance.serialize() == {"x": 1}

def test_serialize_with_format():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v, fmt: v if fmt is None else str(v))
        y = field()

    instance = TestClass(x=1, y="test")
    assert instance.serialize(format="custom") == {"x": "1", "y": "test"}


# LLM-generated content at query #29
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    invariant_errors = ["error1"]
    missing_fields = ["field1"]
    assert invariant_errors or missing_fields


# LLM-generated content at query #30
#--------------------------

```python
def test_equality_with_same_class_and_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2


# LLM-generated content at query #31
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


# LLM-generated content at query #32
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

def test_pclass_constructor_with_mandatory_field_missing():
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
        y = field()

    instance = TestClass.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2, _factory_fields={"x"})
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_invariant_check():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException

    def positive_invariant(inst, field, value):
        if value <= 0:
            raise ValueError("Value must be positive")

    class TestClass(PClass):
        x = field(invariant=positive_invariant)
        y = field()

    try:
        TestClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._pclass import InvariantException

    @invariant
    def sum_invariant(inst):
        if inst.x + inst.y != 10:
            raise ValueError("Sum must be 10")

    class TestClass(PClass):
        x = field()
        y = field()

    TestClass._pclass_invariants = [sum_invariant]

    try:
        TestClass(x=5, y=4)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Field invariant failed" in str(e)


# LLM-generated content at query #33
#--------------------------

```python
def test_repr_returns_correct_string():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    assert repr(instance) == "TestPClass(x=1, y=2)"


# LLM-generated content at query #34
#--------------------------

```python
def test_repr_returns_correct_string():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y="test")
    assert repr(instance) == "TestPClass(x=1, y='test')"


# LLM-generated content at query #35
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    restored_class, restored_data = instance.__reduce__()
    assert restored_class == _restore_pickle
    assert restored_data == (TestClass, {'x': 1, 'y': 2})


# LLM-generated content at query #36
#--------------------------

```python
def test_pclass_reduce_with_pickling():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    obj = TestClass(x=1, y=2)
    reduced = obj.__reduce__()
    assert reduced[0] == _restore_pickle
    assert reduced[1][0] == TestClass
    assert reduced[1][1] == {'x': 1, 'y': 2}


# LLM-generated content at query #37
#--------------------------

```python
def test_check_and_set_attr_with_valid_invariant():
    class MockField:
        type = None
        def invariant(self, value):
            return True, None

    class MockCls:
        pass

    result = MockCls()
    _check_and_set_attr(MockCls, MockField(), "attr", "value", result, [])
    assert hasattr(result, "attr")
    assert result.attr == "value"


# LLM-generated content at query #38
#--------------------------

```python
def test_weakref_in_slots_when_bases_are_pclass():
    class Parent(metaclass=PClassMeta):
        pass

    class Child(Parent):
        pass

    assert '__weakref__' in Child.__slots__


# LLM-generated content at query #39
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

def test_pclass_new_with_invalid_field_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="not an int")
        assert False, "Expected PTypeError"
    except Exception as e:
        assert "Invalid type for field" in str(e)

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
        assert "Field invariant failed" in str(e)

def test_pclass_new_with_global_invariant_failure():
    from pyrsistent import PClass, field, invariant
    class TestClass(PClass):
        x = field()
        y = field()
        _invariant = invariant(lambda s: (s.x != s.y, "x and y must differ"))
    try:
        TestClass(x=1, y=1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Global invariant failed" in str(e)


# LLM-generated content at query #40
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    from pyrsistent._field_common import PFIELD_NO_INITIAL

    class TestClass(PClass):
        x = field(mandatory=True)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.missing_fields == ("TestClass.x",)


# LLM-generated content at query #41
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(invariant=lambda v: (v > 0, "positive"))

    try:
        TestClass(x=1, y=-1)
    except InvariantException as e:
        assert e.args[0] == ("positive",)
        assert e.args[1] == ()
        assert e.args[2] == 'Field invariant failed'

    try:
        TestClass(y=1)
    except InvariantException as e:
        assert e.args[0] == ()
        assert e.args[1] == ("TestClass.x",)
        assert e.args[2] == 'Field invariant failed'


# LLM-generated content at query #42
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


# LLM-generated content at query #43
#--------------------------

```python
def test_set_with_keyword_arguments():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=10, y=20)
    assert new_instance.x == 10
    assert new_instance.y == 20
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

def test_set_with_mixed_arguments():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set('x', 10, y=20)
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
    new_instance = instance.set(x=10)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert instance.x == 1
    assert instance.y == 2

def test_set_with_extra_field():
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


# LLM-generated content at query #44
#--------------------------

```python
def test_equality_with_same_class_and_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert obj1 == obj2


# LLM-generated content at query #45
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
        assert "TestClass.x" in str(e.missing_fields)

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


# LLM-generated content at query #46
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=True)

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=True)

    try:
        TestClass(x=1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "z" in str(e)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=42)

    instance = TestClass()
    assert instance.x == 42

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)

    instance = TestClass()
    assert instance.x == 42

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "z": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2, _factory_fields={"x"})
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #47
#--------------------------

```python
def test_pclass_meta_new_with_single_checked_type_base():
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
    class Base:
        x = _PField(invariant=lambda x: x > 0)

    class TestClass(Base, metaclass=PClassMeta):
        y = _PField()

    assert 'x' in TestClass._pclass_fields
    assert 'y' in TestClass._pclass_fields
    assert 'x' not in TestClass.__dict__
    assert 'y' not in TestClass.__dict__

def test_pclass_meta_new_with_invariant_inheritance():
    def test_invariant(obj):
        return True, ()

    class Base:
        __invariant__ = test_invariant

    class TestClass(Base, metaclass=PClassMeta):
        pass

    assert len(TestClass._pclass_invariants) == 1
    assert TestClass._pclass_invariants[0] == wrap_invariant(test_invariant)

def test_pclass_meta_new_with_non_callable_invariant():
    class Base:
        __invariant__ = "not callable"

    try:
        class TestClass(Base, metaclass=PClassMeta):
            pass
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "Invariants must be callable"


# LLM-generated content at query #48
#--------------------------

```python
def test_pclass_fields_iteration():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    assert list(TestClass._pclass_fields.items()) == [('x', TestClass._pclass_fields['x'])]


# LLM-generated content at query #49
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
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()

    try:
        TestClass(y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields

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

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2, _factory_fields={"x"})
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #50
#--------------------------

```python
def test_invariant_check_fails():
    field = Mock(invariant=lambda x: (False, "error"))
    invariant_errors = []
    _check_and_set_attr(Mock, field, "attr", "value", Mock, invariant_errors)
    assert invariant_errors == ["error"]


# LLM-generated content at query #51
#--------------------------

```python
def test_set_with_kwargs():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(x=3)
    assert a2.x == 3
    assert a2.y == 2
    assert a.x == 1
    assert a.y == 2

def test_set_with_args():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set('x', 2)
    assert a2.x == 2
    assert a.x == 1

def test_set_multiple_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
        z = field()
    a = AClass(x=1, y=2, z=3)
    a2 = a.set(x=4, y=5)
    assert a2.x == 4
    assert a2.y == 5
    assert a2.z == 3
    assert a.x == 1
    assert a.y == 2
    assert a.z == 3

def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a is not a2
    assert a.x == 1
    assert a2.x == 2


# LLM-generated content at query #52
#--------------------------

```python
def test_equality_with_same_class_and_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance1 = TestPClass(x=1, y=2)
    instance2 = TestPClass(x=1, y=2)
    assert instance1 == instance2


# LLM-generated content at query #53
#--------------------------

```python
def test_check_and_set_attr_with_invariant_failure():
    class MockField:
        type = None
        def invariant(self, value):
            return False, "Error"

    cls = type('MockClass', (), {})
    field = MockField()
    name = "test_field"
    value = "test_value"
    result = object()
    invariant_errors = []

    _check_and_set_attr(cls, field, name, value, result, invariant_errors)

    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "Error"


# LLM-generated content at query #54
#--------------------------

```python
def test_repr_contains_class_name_and_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    repr_str = repr(instance)

    assert "TestClass" in repr_str
    assert "x=1" in repr_str
    assert "y=2" in repr_str


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_remove_existing_item():
    original = object()
    evolver = _PClassEvolver(original, {'a': 1, 'b': 2})
    result = evolver.remove('a')
    assert result is evolver
    assert 'a' not in evolver._pclass_evolver_data
    assert 'b' in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'a' not in evolver._factory_fields

def test_remove_nonexistent_item_raises_attribute_error():
    original = object()
    evolver = _PClassEvolver(original, {'a': 1})
    try:
        evolver.remove('b')
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == 'b'


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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
        x = field()
        y = field(mandatory=True)

    try:
        TestClass(x=1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.y" in e.missing_fields

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
        x = field(initial=42)
        y = field()

    instance = TestClass(y=10)
    assert instance.x == 42
    assert instance.y == 10

def test_pclass_new_with_callable_initial():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(initial=lambda: 42)
        y = field()

    instance = TestClass(y=10)
    assert instance.x == 42
    assert instance.y == 10

def test_pclass_new_with_factory():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
        y = field()

    instance = TestClass(x=5, y=10)
    assert instance.x == 10
    assert instance.y == 10

def test_pclass_new_with_invariant():
    from pyrsistent import PClass, field
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "positive"))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "positive" in e.error_codes

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    from pyrsistent._pclass import InvariantException

    @invariant
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
        assert "sum_positive" in e.error_codes

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    instance = TestClass.create({"x": 1, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, "z")

def test_pclass_new_with_factory_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
        y = field()

    instance = TestClass.create({"x": 5, "y": 10}, _factory_fields={"x"})
    assert instance.x == 10
    assert instance.y == 10

def test_pclass_new_with_ignore_extra_and_factory():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(factory=lambda v, ignore_extra=False: v * (3 if ignore_extra else 2))
        y = field()

    instance = TestClass.create({"x": 5, "y": 10, "z": 3}, ignore_extra=True, _factory_fields={"x"})
    assert instance.x == 15
    assert instance.y == 10
    assert not hasattr(instance, "z")


# LLM-generated content at query #4
#--------------------------

```python
def test_set_with_keyword_argument():
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

def test_set_with_multiple_updates():
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

def test_set_with_no_changes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=1)
    assert new_instance.x == 1
    assert new_instance.y == 2
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #5
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

def test_pclass_constructor_with_mandatory_field_missing():
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
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

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

    instance = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


# LLM-generated content at query #6
#--------------------------

```python
def test_eq_same_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance == instance

def test_eq_different_instances_same_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2

def test_eq_different_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    assert not (instance1 == instance2)

def test_eq_different_types():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    assert not (instance1 == instance2)

def test_eq_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1, y=2)
    assert not (instance1 == instance2)


# LLM-generated content at query #7
#--------------------------

```python
def test_pclassmeta_new_with_single_checkedtype_base():
    bases = (CheckedType,)
    dct = {'a': _PField(1), 'b': _PField(2)}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert result._pclass_fields == {'a': _PField(1), 'b': _PField(2)}
    assert result.__slots__ == ('_pclass_frozen', 'a', 'b', '__weakref__')
    assert result._pclass_invariants == ()

def test_pclassmeta_new_with_multiple_bases():
    class Base1(CheckedType):
        __invariant__ = lambda self: True

    class Base2(CheckedType):
        __invariant__ = lambda self: True

    bases = (Base1, Base2)
    dct = {'c': _PField(3)}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert 'c' in result._pclass_fields
    assert '__weakref__' not in result.__slots__

def test_pclassmeta_new_with_invariant_in_base():
    class Base(CheckedType):
        __invariant__ = lambda self: (True, 'test')

    bases = (Base,)
    dct = {}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert len(result._pclass_invariants) == 1
    assert callable(result._pclass_invariants[0])

def test_pclassmeta_new_with_non_callable_invariant():
    class Base(CheckedType):
        __invariant__ = "not callable"

    bases = (Base,)
    dct = {}
    try:
        PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test_pclassmeta_new_with_field_inheritance():
    class Base(CheckedType):
        x = _PField(1)

    bases = (Base,)
    dct = {'y': _PField(2)}
    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert 'x' in result._pclass_fields
    assert 'y' in result._pclass_fields
    assert result._pclass_fields['x'] == Base.x
    assert result._pclass_fields['y'] == dct['y']


# LLM-generated content at query #8
#--------------------------

```python
def test_set_with_kwargs():
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

def test_set_with_args():
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


# LLM-generated content at query #9
#--------------------------

```python
def test_is_pclass_with_single_checked_type_base():
    assert _is_pclass((CheckedType,)) == True

def test_is_pclass_with_multiple_bases():
    assert _is_pclass((CheckedType, object)) == False

def test_is_pclass_with_no_bases():
    assert _is_pclass(()) == False

def test_is_pclass_with_non_checked_type_base():
    assert _is_pclass((object,)) == False


# LLM-generated content at query #10
#--------------------------

```python
def test__is_pclass_predicate_false():
    class TestClass(metaclass=PClassMeta):
        pass

    assert '__weakref__' not in TestClass.__slots__


# LLM-generated content at query #11
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
        assert "TestClass" in str(e)

def test_pclass_constructor_with_initial_values():
    class TestClass(PClass):
        x = field(initial=0)
        y = field(initial=lambda: "default")

    instance = TestClass()
    assert instance.x == 0
    assert instance.y == "default"

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

    instance = TestClass._create({'x': 1, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


# LLM-generated content at query #12
#--------------------------

```python
def test_pclass_reduce_with_pickling():
    from pyrsistent import PClass, field, v
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_pickle
    assert reduced[1][0] == TestClass
    assert reduced[1][1] == {'x': 1, 'y': 2}

def test_pclass_reduce_with_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance = TestClass(x=1)
    reduced = instance.__reduce__()
    assert reduced[0] == _restore_pickle
    assert reduced[1][0] == TestClass
    assert reduced[1][1] == {'x': 1}


# LLM-generated content at query #13
#--------------------------

```python
def test_repr_with_single_field():
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    assert repr(instance) == "TestClass(x=1)"

def test_repr_with_multiple_fields():
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    instance = TestClass(x=1, y="hello", z=[1, 2, 3])
    assert repr(instance) == "TestClass(x=1, y='hello', z=[1, 2, 3])"

def test_repr_with_missing_optional_field():
    class TestClass(PClass):
        x = field()
        y = field(initial=0)

    instance = TestClass(x=1)
    assert repr(instance) == "TestClass(x=1, y=0)"

def test_repr_with_none_value():
    class TestClass(PClass):
        x = field()

    instance = TestClass(x=None)
    assert repr(instance) == "TestClass(x=None)"

def test_repr_with_complex_object():
    class TestClass(PClass):
        x = field()

    instance = TestClass(x={"a": 1, "b": [2, 3]})
    assert repr(instance) == "TestClass(x={'a': 1, 'b': [2, 3]})"


# LLM-generated content at query #14
#--------------------------

```python
def test_pclass_hash_returns_consistent_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert hash(instance1) == hash(instance2)

def test_pclass_hash_different_for_different_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)
    assert hash(instance1) != hash(instance2)

def test_pclass_hash_includes_all_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()

    instance1 = TestClass(x=1, y=2, z=3)
    instance2 = TestClass(x=1, y=2, z=4)
    assert hash(instance1) != hash(instance2)

def test_pclass_hash_with_missing_optional_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1, y=2)
    assert hash(instance1) != hash(instance2)


# LLM-generated content at query #15
#--------------------------

```python
def test_set_preserves_existing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    original = TestClass(x=1, y=2)
    modified = original.set(x=10)
    assert modified.y == 2


# LLM-generated content at query #16
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
        assert "Field invariant failed" in str(e)

def test_pclass_new_with_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()

    try:
        TestClass(x=1, z=3)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

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

    instance = TestClass(x=5)
    assert instance.x == 10

def test_pclass_new_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v, ignore_extra=False: v, type={TestClass})

    instance = TestClass.create({"x": 1, "z": 3}, ignore_extra=True)
    assert instance.x == 1

def test_pclass_new_with_invariant_error():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(invariant=lambda v: (v > 0, "must be positive"))

    try:
        TestClass(x=-1)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test_pclass_new_with_global_invariant():
    from pyrsistent import PClass, field, invariant
    def check_sum(instance):
        return instance.x + instance.y > 0, "sum must be positive"

    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [check_sum]

    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

    try:
        TestClass(x=-1, y=-2)
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Global invariant failed" in str(e)


# LLM-generated content at query #17
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

def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=0)
        y = field(initial=lambda: "default")

    instance = TestClass()
    assert instance.x == 0
    assert instance.y == "default"

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

    instance = TestClass._create({'x': 1, 'y': 2, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')


# LLM-generated content at query #18
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
        x = field(initial=42)
        y = field()

    instance = TestClass(y=2)
    assert instance.x == 42
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


# LLM-generated content at query #19
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_exception():
    from pyrsistent import PClass, field, mandatory, invariant
    from pyrsistent._pclass import InvariantException

    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(invariant=lambda v: (False, "error"))

    try:
        TestClass()
    except InvariantException as e:
        assert e.args[0] == ("error",)
        assert e.args[1] == ("TestClass.x",)
        assert e.args[2] == 'Field invariant failed'


# LLM-generated content at query #20
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

def test_pclass_constructor_with_initial_values():
    class TestClass(PClass):
        x = field(initial=0)
        y = field(initial=lambda: "default")

    instance = TestClass()
    assert instance.x == 0
    assert instance.y == "default"

def test_pclass_constructor_with_factory_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(_factory_fields={"x"}, x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

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

def test_pclass_constructor_with_valid_invariant():
    def invariant(obj):
        if obj.x < 0:
            raise ValueError("x must be non-negative")

    class TestClass(PClass):
        x = field(invariant=invariant)

    instance = TestClass(x=1)
    assert instance.x == 1

def test_pclass_constructor_with_serializer():
    class TestClass(PClass):
        x = field(serializer=lambda v: v * 2)

    instance = TestClass(x=5)
    assert instance.serialize() == {"x": 10}

def test_pclass_constructor_with_pclass_instance():
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass.create(instance1)
    assert instance2.x == 1
    assert instance2.y == 2


# LLM-generated content at query #21
#--------------------------

```python
def test_pclass_fields_iteration():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    assert '_pclass_fields' in TestClass.__dict__
    assert len(TestClass._pclass_fields) > 0
    assert 'x' in TestClass._pclass_fields


# LLM-generated content at query #22
#--------------------------

```python
def test_equality_with_same_class_instance():
    class TestClass(PClass):
        x = field()
        y = field()

    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2


# LLM-generated content at query #23
#--------------------------

```python
def test__is_pclass_returns_false_for_non_pclass_bases():
    class NonPClassBase:
        pass

    class TestClass(metaclass=PClassMeta):
        pass

    assert not _is_pclass((NonPClassBase,))


# LLM-generated content at query #24
#--------------------------

```python
def test_pclass_hash_equality():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


# LLM-generated content at query #25
#--------------------------

```python
def test_pclass_hash_equality():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=1, y=3)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


# LLM-generated content at query #26
#--------------------------

```python
def test_repr_returns_correct_string():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y="test")
    assert repr(instance) == "TestClass(x=1, y='test')"

def test_repr_with_missing_optional_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)

    instance = TestClass(x=1)
    assert repr(instance) == "TestClass(x=1)"

def test_repr_with_complex_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=[1, 2, 3], y={"a": 1})
    assert repr(instance) == "TestClass(x=[1, 2, 3], y={'a': 1})"


# LLM-generated content at query #27
#--------------------------

```python
def test_pclass_fields_iteration():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    assert list(TestClass._pclass_fields.items()) == [('x', TestClass._pclass_fields['x']), ('y', TestClass._pclass_fields['y'])]


# LLM-generated content at query #28
#--------------------------

```python
def test_pickle_reduce_returns_tuple_with_restore_pickle_and_class_data():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    instance = TestClass(x=42)
    result = instance.__reduce__()

    assert result[0] == _restore_pickle
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 42}


# LLM-generated content at query #29
#--------------------------

```python
def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field, serialize
    class TestClass(PClass):
        x = field(serializer=lambda v: v * 2)
    instance = TestClass(x=5)
    assert instance.serialize() == {'x': 10}

def test_serialize_without_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=5)
    assert instance.serialize() == {'x': 5}

def test_serialize_with_missing_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=5)
    assert instance.serialize() == {'x': 5}

def test_serialize_with_format():
    from pyrsistent import PClass, field, serialize
    class TestClass(PClass):
        x = field(serializer=lambda v, fmt: str(v) if fmt == 'str' else v)
    instance = TestClass(x=5)
    assert instance.serialize(format='str') == {'x': '5'}


# LLM-generated content at query #30
#--------------------------

```python
def test_set_with_keyword_arguments():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()

    a = AClass(x=1, y=2)
    a2 = a.set(x=10)
    assert a.x == 1
    assert a.y == 2
    assert a2.x == 10
    assert a2.y == 2

def test_set_with_positional_arguments():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()

    a = AClass(x=1, y=2)
    a2 = a.set('x', 10)
    assert a.x == 1
    assert a.y == 2
    assert a2.x == 10
    assert a2.y == 2

def test_set_preserves_other_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
        z = field()

    a = AClass(x=1, y=2, z=3)
    a2 = a.set(x=10)
    assert a2.x == 10
    assert a2.y == 2
    assert a2.z == 3

def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()

    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a is not a2
    assert a.x == 1
    assert a2.x == 2


# LLM-generated content at query #31
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_missing_mandatory_field():
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

def test_pclass_constructor_with_initial_values():
    class TestPClass(PClass):
        x = field(initial=10)
        y = field(initial=lambda: 20)

    instance = TestPClass()
    assert instance.x == 10
    assert instance.y == 20

def test_pclass_constructor_with_factory_fields():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass._factory_fields={'x'}, x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_ignore_extra():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(ignore_extra=True, x=1, y=2, z=3)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, 'z')


# LLM-generated content at query #32
#--------------------------

```python
def test_pickle_support_returns_correct_tuple():
    class TestPClass(PClass):
        x = field()
        y = field()

    instance = TestPClass(x=1, y=2)
    result = instance.__reduce__()
    assert result == (_restore_pickle, (TestPClass, {'x': 1, 'y': 2}))


# LLM-generated content at query #33
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

def test_pclass_constructor_with_initial_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field(initial=lambda: 20)
    instance = TestClass()
    assert instance.x == 10
    assert instance.y == 20

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
    instance = TestClass(x=1, y=2, _factory_fields={"x"})
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #34
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_exception():
    from pyrsistent import PClass, field, invariant, v, s, m

    class TestClass(PClass):
        x = field(type=int)
        y = field(type=int, mandatory=True)

        @invariant
        def check_x(self):
            return self.x > 0, "x must be positive"

    with pytest.raises(InvariantException):
        TestClass(x=-1, y=1)

    with pytest.raises(InvariantException):
        TestClass(x=1)


# LLM-generated content at query #35
#--------------------------

```python
def test_check_and_set_attr_with_valid_type_and_invariant():
    class MockField:
        type = int
        invariant = lambda self, value: (True, None)

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
        invariant = lambda self, value: (True, None)

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []

    try:
        _check_and_set_attr(MockClass, MockField(), "attr", "not_an_int", result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert str(e) == "Invalid type for field MockClass.attr, was str"

def test_check_and_set_attr_with_failed_invariant():
    class MockField:
        type = int
        invariant = lambda self, value: (False, "INVALID")

    class MockClass:
        pass

    result = MockClass()
    invariant_errors = []

    _check_and_set_attr(MockClass, MockField(), "attr", 42, result, invariant_errors)

    assert not hasattr(result, "attr")
    assert invariant_errors == ["INVALID"]


# LLM-generated content at query #36
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    invariant_errors = ["error1"]
    missing_fields = []
    assert invariant_errors or missing_fields


# LLM-generated content at query #37
#--------------------------

```python
def test_serialize_returns_dict():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()

    instance = TestClass(x=1)
    result = instance.serialize()
    assert isinstance(result, dict)


# LLM-generated content at query #38
#--------------------------

```python
def test_pclass_hash_consistency():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    obj3 = TestClass(x=3, y=4)

    assert hash(obj1) == hash(obj2)
    assert hash(obj1) != hash(obj3)


# LLM-generated content at query #39
#--------------------------

```python
def test__is_pclass_returns_true_for_pclass_bases():
    class PClassBase(metaclass=PClassMeta):
        pass

    class PClassChild(PClassBase):
        pass

    assert _is_pclass((PClassBase,)) == True


# LLM-generated content at query #40
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

def test_pclass_constructor_with_initial_values():
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

def test_pclass_constructor_with_ignore_extra():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass.create({"x": 1, "y": 2, "z": 3}, ignore_extra=True)
    assert instance.x == 1
    assert instance.y == 2
    assert not hasattr(instance, "z")

def test_pclass_constructor_with_factory_fields():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass._factory_fields={"x"}, x=1, y=2
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_invariant_check():
    class TestClass(PClass):
        x = field(invariant=lambda x: x > 0)
        y = field()

    try:
        TestClass(x=-1, y=2)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "invariant" in str(e).lower()

def test_pclass_constructor_with_global_invariant():
    def global_invariant(instance):
        if instance.x + instance.y != 10:
            raise ValueError("Sum must be 10")

    class TestClass(PClass):
        _pclass_invariants = [global_invariant]
        x = field()
        y = field()

    try:
        TestClass(x=5, y=4)
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "Sum must be 10" in str(e)


# LLM-generated content at query #41
#--------------------------

```python
def test_serialize_with_no_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize() == {'x': 1, 'y': 2}

def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v: str(v))
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize() == {'x': '1', 'y': 2}

def test_serialize_with_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1)
    assert instance.serialize() == {'x': 1}

def test_serialize_with_format():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v, fmt: v if fmt is None else str(v))
        y = field()

    instance = TestClass(x=1, y=2)
    assert instance.serialize(format='json') == {'x': '1', 'y': 2}


# LLM-generated content at query #42
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    from pyrsistent._field_common import PFIELD_NO_INITIAL

    class TestClass(PClass):
        x = field(mandatory=True)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert e.args[1] == ('TestClass.x',)


# LLM-generated content at query #43
#--------------------------

```python
def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field, v, s, m

    class TestClass(PClass):
        x = field(type=int, initial=0)
        y = field(type=str, mandatory=True)

    instance = TestClass(y="test")
    assert instance.x == 0
    assert instance.y == "test"

def test_pclass_constructor_with_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException

    class TestClass(PClass):
        x = field(type=int, mandatory=True)

    try:
        TestClass()
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields

def test_pclass_constructor_with_extra_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x=1, y=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "y" in str(e)

def test_pclass_constructor_with_invalid_field_value():
    from pyrsistent import PClass, field, InvariantException

    class TestClass(PClass):
        x = field(type=int)

    try:
        TestClass(x="not an int")
        assert False, "Expected InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(type=int)
        y = field(type=str)

    instance = TestClass._create({'x': 1, 'y': 'test'}, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 'test'

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field(type=int)

    instance = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


# LLM-generated content at query #44
#--------------------------

```python
def test_repr_returns_correct_string():
    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"


# LLM-generated content at query #45
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

    instance = TestClass.create({'x': 1, 'z': 3}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')


# LLM-generated content at query #46
#--------------------------

```python
def test_weakref_in_slots_when_bases_are_pclass():
    bases = (object,)
    dct = {'__slots__': ()}

    result = PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)

    assert '__weakref__' in result.__slots__


# LLM-generated content at query #47
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    from pyrsistent import PClass, field, InvariantException
    from pyrsistent._field_common import PFIELD_NO_INITIAL

    class TestClass(PClass):
        x = field(type=int, invariant=lambda v: (v > 0, "Value must be positive"))
        y = field(mandatory=True)

    try:
        TestClass(x=-1, y=2)
    except InvariantException:
        pass
    else:
        assert False, "Expected InvariantException for invariant error"

    try:
        TestClass(x=1)
    except InvariantException:
        pass
    else:
        assert False, "Expected InvariantException for missing mandatory field"


# LLM-generated content at query #48
#--------------------------

```python
def test_invariant_errors_or_missing_fields():
    class TestPClass(PClass):
        x = field(type=int, mandatory=True)
        y = field(invariant=lambda v: (False, "error") if v < 0 else (True, None))

    with pytest.raises(InvariantException):
        TestPClass(x=1, y=-1)


# LLM-generated content at query #49
#--------------------------

```python
def test_pclass_reduce():
    from pyrsistent import PClass, field

    class TestClass(PClass):
        x = field()
        y = field()

    instance = TestClass(x=1, y=2)
    restored_class, restored_data = instance.__reduce__()
    assert restored_class == _restore_pickle
    assert restored_data == (TestClass, {'x': 1, 'y': 2})


# LLM-generated content at query #50
#--------------------------

```python
def test__is_pclass_returns_true_for_pclass_bases():
    class Base(metaclass=PClassMeta):
        pass

    assert _is_pclass((Base,))


