####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_set_updates_data_when_value_is_different():
    original = type('Original', (), {})()
    evolver = _PClassEvolver(original, {'a': 1})
    evolver.set('a', 2)
    result = evolver._pclass_evolver_data['a']
    assert result == 2

def test_set_marks_data_as_dirty_when_value_is_different():
    original = type('Original', (), {})()
    evolver = _PClassEvolver(original, {'a': 1})
    evolver.set('a', 2)
    result = evolver._pclass_evolver_data_is_dirty
    assert result is True

def test_set_adds_key_to_factory_fields_when_value_is_different():
    original = type('Original', (), {})()
    evolver = _PClassEvolver(original, {'a': 1})
    evolver.set('a', 2)
    result = 'a' in evolver._factory_fields
    assert result is True

def test_set_does_not_update_data_when_value_is_same():
    original = type('Original', (), {})()
    evolver = _PClassEvolver(original, {'a': 1})
    evolver.set('a', 1)
    result = evolver._pclass_evolver_data_is_dirty
    assert result is False

def test_set_returns_self():
    original = type('Original', (), {})()
    evolver = _PClassEvolver(original, {'a': 1})
    result = evolver.set('a', 2)
    assert result is evolver

def test_set_with_new_key_updates_data():
    original = type('Original', (), {})()
    evolver = _PClassEvolver(original, {})
    evolver.set('b', 3)
    result = evolver._pclass_evolver_data['b']
    assert result == 3

def test_set_with_new_key_marks_data_as_dirty():
    original = type('Original', (), {})()
    evolver = _PClassEvolver(original, {})
    evolver.set('b', 3)
    result = evolver._pclass_evolver_data_is_dirty
    assert result is True

def test_set_with_new_key_adds_to_factory_fields():
    original = type('Original', (), {})()
    evolver = _PClassEvolver(original, {})
    evolver.set('b', 3)
    result = 'b' in evolver._factory_fields
    assert result is True


# LLM-generated content at query #2
#--------------------------

def test___new___single_inheritance():
    class CheckedType:
        pass
    class TestClass(CheckedType, metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    instance = TestClass()
    assert hasattr(instance, '_pclass_frozen')
    assert '_pclass_frozen' in TestClass.__slots__
    assert '__weakref__' in TestClass.__slots__

def test___new___multiple_inheritance():
    class Base1:
        pass
    class Base2:
        pass
    class TestClass(Base1, Base2, metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    instance = TestClass()
    assert hasattr(instance, '_pclass_frozen')
    assert '__weakref__' not in TestClass.__slots__

def test___new___with_fields():
    class CheckedType:
        pass
    class TestClass(CheckedType, metaclass=PClassMeta):
        field = _PField()
        __invariant__ = lambda self: (True, ())
    instance = TestClass()
    assert 'field' in TestClass._pclass_fields
    assert 'field' in TestClass.__slots__
    assert hasattr(instance, '_pclass_frozen')

def test___new___invariant_inheritance():
    class CheckedType:
        pass
    class Base(CheckedType, metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class Derived(Base):
        pass
    assert len(Derived._pclass_invariants) == 1
    instance = Derived()
    assert hasattr(instance, '_pclass_frozen')

def test___new___multiple_invariants():
    class CheckedType:
        pass
    class Base(CheckedType, metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class Derived(Base):
        __invariant__ = lambda self: (False, ('error',))
    assert len(Derived._pclass_invariants) == 2
    instance = Derived()
    assert hasattr(instance, '_pclass_frozen')

def test___new___non_callable_invariant():
    class CheckedType:
        pass
    try:
        class TestClass(CheckedType, metaclass=PClassMeta):
            __invariant__ = 'not callable'
        assert False
    except TypeError:
        pass

def test___new___no_invariant():
    class CheckedType:
        pass
    class TestClass(CheckedType, metaclass=PClassMeta):
        pass
    assert TestClass._pclass_invariants == ()
    instance = TestClass()
    assert hasattr(instance, '_pclass_frozen')

def test___new___field_inheritance():
    class CheckedType:
        pass
    class Base(CheckedType, metaclass=PClassMeta):
        base_field = _PField()
    class Derived(Base):
        derived_field = _PField()
    assert 'base_field' in Derived._pclass_fields
    assert 'derived_field' in Derived._pclass_fields
    assert 'base_field' in Derived.__slots__
    assert 'derived_field' in Derived.__slots__
    instance = Derived()
    assert hasattr(instance, '_pclass_frozen')


# LLM-generated content at query #3
#--------------------------

def test_repr_with_single_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    result = repr(instance)
    expected = "TestClass(x=10)"
    assert result == expected

def test_repr_with_multiple_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    result = repr(instance)
    expected = "TestClass(x=1, y=2)"
    assert result == expected

def test_repr_with_string_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        name = field()
    instance = TestClass(name="Alice")
    result = repr(instance)
    expected = "TestClass(name='Alice')"
    assert result == expected

def test_repr_with_none_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        value = field()
    instance = TestClass(value=None)
    result = repr(instance)
    expected = "TestClass(value=None)"
    assert result == expected

def test_repr_with_initial_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
    instance = TestClass()
    result = repr(instance)
    expected = "TestClass(x=5)"
    assert result == expected

def test_repr_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_repr_with_extra_fields_ignored():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass.create({'x': 1, 'y': 2}, ignore_extra=True)
    result = repr(instance)
    expected = "TestClass(x=1)"
    assert result == expected

def test_repr_after_set():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=3)
    result = repr(new_instance)
    expected = "TestClass(x=3, y=2)"
    assert result == expected

def test_repr_with_complex_object():
    from pyrsistent import PClass, field
    class Inner(PClass):
        a = field()
    class Outer(PClass):
        inner = field()
    inner_instance = Inner(a=42)
    outer_instance = Outer(inner=inner_instance)
    result = repr(outer_instance)
    expected = "Outer(inner=Inner(a=42))"
    assert result == expected

def test_repr_with_empty_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = repr(instance)
    expected = "TestClass()"
    assert result == expected


# LLM-generated content at query #4
#--------------------------

def test_set_updates_data_and_flags_when_value_is_different():
    original = object()
    initial_dict = {"a": 1}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set("b", 2)
    assert evolver._pclass_evolver_data == {"a": 1, "b": 2}
    assert evolver._factory_fields == {"b"}
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver

def test_set_does_not_update_when_value_is_same():
    original = object()
    initial_dict = {"a": 1}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set("a", 1)
    assert evolver._pclass_evolver_data == {"a": 1}
    assert evolver._factory_fields == set()
    assert evolver._pclass_evolver_data_is_dirty is False
    assert result is evolver

def test_set_updates_existing_key_with_new_value():
    original = object()
    initial_dict = {"a": 1}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set("a", 3)
    assert evolver._pclass_evolver_data == {"a": 3}
    assert evolver._factory_fields == {"a"}
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver

def test_set_handles_missing_value_sentinel():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set("c", 5)
    assert evolver._pclass_evolver_data == {"c": 5}
    assert evolver._factory_fields == {"c"}
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver


# LLM-generated content at query #5
#--------------------------

def test___hash___returns_same_hash_for_equal_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert hash(instance1) == hash(instance2)

def test___hash___returns_different_hash_for_different_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)
    assert hash(instance1) != hash(instance2)

def test___hash___handles_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    instance = TestClass(x=1)
    hash_value = hash(instance)
    assert isinstance(hash_value, int)

def test___hash___consistent_across_multiple_calls():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    instance = TestClass(a=10, b=20)
    first_hash = hash(instance)
    second_hash = hash(instance)
    assert first_hash == second_hash

def test___hash___with_nested_pclass():
    from pyrsistent import PClass, field
    class Inner(PClass):
        val = field()
    class Outer(PClass):
        inner = field()
    inner1 = Inner(val=5)
    outer1 = Outer(inner=inner1)
    inner2 = Inner(val=5)
    outer2 = Outer(inner=inner2)
    assert hash(outer1) == hash(outer2)

def test___hash___different_classes_same_values():
    from pyrsistent import PClass, field
    class ClassA(PClass):
        x = field()
    class ClassB(PClass):
        x = field()
    instance_a = ClassA(x=1)
    instance_b = ClassB(x=1)
    assert hash(instance_a) != hash(instance_b)


# LLM-generated content at query #6
#--------------------------

def test_serialize_without_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y="hello")
    result = instance.serialize()
    expected = {'x': 10, 'y': 'hello'}
    assert result == expected

def test_serialize_with_serializer():
    from pyrsistent import PClass, field
    def serialize_x(value, format):
        return value * 2
    def serialize_y(value, format):
        return value.upper()
    class TestClass(PClass):
        x = field(serializer=serialize_x)
        y = field(serializer=serialize_y)
    instance = TestClass(x=5, y="world")
    result = instance.serialize()
    expected = {'x': 10, 'y': 'WORLD'}
    assert result == expected

def test_serialize_with_format():
    from pyrsistent import PClass, field
    def custom_serializer(value, format):
        if format == 'double':
            return value * 2
        return value
    class TestClass(PClass):
        x = field(serializer=custom_serializer)
        y = field()
    instance = TestClass(x=7, y=3)
    result = instance.serialize(format='double')
    expected = {'x': 14, 'y': 3}
    assert result == expected

def test_serialize_missing_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    instance = TestClass(x=1)
    result = instance.serialize()
    expected = {'x': 1}
    assert result == expected

def test_serialize_empty():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = instance.serialize()
    expected = {}
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_serialize_without_custom_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y="hello")
    result = instance.serialize()
    expected = {'x': 10, 'y': 'hello'}
    assert result == expected

def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field
    def custom_serializer(value, format=None):
        return value * 2
    class TestClass(PClass):
        x = field(serializer=custom_serializer)
        y = field()
    instance = TestClass(x=5, y="test")
    result = instance.serialize()
    expected = {'x': 10, 'y': 'test'}
    assert result == expected

def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    def format_serializer(value, format=None):
        return f"{format}:{value}"
    class TestClass(PClass):
        x = field(serializer=format_serializer)
        y = field()
    instance = TestClass(x=100, y="world")
    result = instance.serialize(format="json")
    expected = {'x': 'json:100', 'y': 'world'}
    assert result == expected

def test_serialize_with_missing_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    instance = TestClass(x=1)
    result = instance.serialize()
    expected = {'x': 1}
    assert result == expected

def test_serialize_empty_pclass():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = instance.serialize()
    expected = {}
    assert result == expected

def test_serialize_with_initial_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=42)
        y = field()
    instance = TestClass(y="value")
    result = instance.serialize()
    expected = {'x': 42, 'y': 'value'}
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test___new___raises_on_invalid_type():
    from pyrsistent import PClass, field, PTypeError
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="string")
        assert False
    except PTypeError as e:
        assert e.__class__.__name__ == "PTypeError"

def test___new___raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
        assert False
    except InvariantException as e:
        assert "missing_fields" in str(e)

def test___new___uses_initial_for_non_provided_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5

def test___new___raises_on_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, y=2)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test___new___handles_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 100)
    instance = TestClass()
    assert instance.x == 100

def test___new___checks_global_invariants():
    from pyrsistent import PClass, field, InvariantException, invariant
    def check_positive(obj):
        return obj.x > 0, "not_positive"
    class TestClass(PClass):
        x = field()
        __invariants__ = [check_positive]
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___handles_factory_fields_parameter():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
    instance = TestClass(_factory_fields={'x'}, x=5)
    assert instance.x == 10

def test___new___handles_ignore_extra_parameter():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

def test___new___propagates_ignore_extra_to_factory():
    from pyrsistent import PClass, field
    def factory_with_ignore_extra(value, ignore_extra=False):
        return value if ignore_extra else value * 2
    class TestClass(PClass):
        x = field(factory=factory_with_ignore_extra)
    instance = TestClass(x=5, ignore_extra=True)
    assert instance.x == 5

def test___new___sets_frozen_attribute():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert instance._pclass_frozen == True

def test___new___handles_invariant_errors():
    from pyrsistent import PClass, field, InvariantException
    def always_false(value):
        return False, "error"
    class TestClass(PClass):
        x = field(invariant=always_false)
    try:
        TestClass(x=1)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)


# LLM-generated content at query #9
#--------------------------

def test_is_pclass_true():
    result = _is_pclass((CheckedType,))
    assert result == True

def test_is_pclass_false_multiple_bases():
    result = _is_pclass((CheckedType, object))
    assert result == False

def test_is_pclass_false_different_base():
    result = _is_pclass((object,))
    assert result == False

def test_is_pclass_false_empty():
    result = _is_pclass(())
    assert result == False


# LLM-generated content at query #10
#--------------------------

def test___eq___returns_true_for_same_class_and_equal_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    result = instance1.__eq__(instance2)
    assert result is True

def test___eq___returns_false_for_same_class_and_different_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    result = instance1.__eq__(instance2)
    assert result is False

def test___eq___returns_not_implemented_for_different_class():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    result = instance1.__eq__(instance2)
    assert result is NotImplemented

def test___eq___returns_not_implemented_for_non_pclass():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    other = object()
    result = instance.__eq__(other)
    assert result is NotImplemented

def test___eq___handles_missing_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    result = instance1.__eq__(instance2)
    assert result is True

def test___eq___returns_false_when_one_field_missing_in_one_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1)
    result = instance1.__eq__(instance2)
    assert result is False


# LLM-generated content at query #11
#--------------------------

def test___new___creates_pclass_with_fields():
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
        field = _PField()
    assert hasattr(TestClass, '_pclass_fields')
    assert 'field' in TestClass._pclass_fields
    assert TestClass._pclass_fields['field'] is field
    assert '_pclass_frozen' in TestClass.__slots__
    assert 'field' in TestClass.__slots__
    assert hasattr(TestClass, '_pclass_invariants')
    assert len(TestClass._pclass_invariants) == 1

def test___new___inherits_fields_from_base():
    class BaseClass(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
        base_field = _PField()
    class DerivedClass(BaseClass, metaclass=PClassMeta):
        derived_field = _PField()
    assert 'base_field' in DerivedClass._pclass_fields
    assert 'derived_field' in DerivedClass._pclass_fields
    assert DerivedClass._pclass_fields['base_field'] is BaseClass._pclass_fields['base_field']
    assert DerivedClass._pclass_fields['derived_field'] is derived_field

def test___new___collects_invariants_from_base():
    base_invariant = lambda self: (True, ())
    derived_invariant = lambda self: (True, ())
    class BaseClass(metaclass=PClassMeta):
        __invariant__ = base_invariant
    class DerivedClass(BaseClass, metaclass=PClassMeta):
        __invariant__ = derived_invariant
    assert len(DerivedClass._pclass_invariants) == 2
    assert DerivedClass._pclass_invariants[0].__wrapped__ is derived_invariant
    assert DerivedClass._pclass_invariants[1].__wrapped__ is base_invariant

def test___new___wraps_invariants():
    invariant = lambda self: ((True, ()), (False, ('error',)))
    class TestClass(metaclass=PClassMeta):
        __invariant__ = invariant
    result = TestClass._pclass_invariants[0](None)
    assert result == (False, ('error',))

def test___new___adds_weakref_only_for_top_pclass():
    class TopClass(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class SubClass(TopClass, metaclass=PClassMeta):
        pass
    assert '__weakref__' in TopClass.__slots__
    assert '__weakref__' not in SubClass.__slots__

def test___new___raises_type_error_for_non_callable_invariant():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = 'not callable'
        assert False
    except TypeError:
        pass

def test___new___handles_multiple_inheritance_fields():
    class Base1(metaclass=PClassMeta):
        field1 = _PField()
        __invariant__ = lambda self: (True, ())
    class Base2(metaclass=PClassMeta):
        field2 = _PField()
        __invariant__ = lambda self: (True, ())
    class Derived(Base1, Base2, metaclass=PClassMeta):
        field3 = _PField()
    assert 'field1' in Derived._pclass_fields
    assert 'field2' in Derived._pclass_fields
    assert 'field3' in Derived._pclass_fields

def test___new___removes_pfield_from_dict():
    field = _PField()
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
        my_field = field
    assert 'my_field' not in TestClass.__dict__
    assert TestClass._pclass_fields['my_field'] is field


# LLM-generated content at query #12
#--------------------------

```python
def test_invariant_errors_trigger_exception():
    from pyrsistent import PClass, field, InvariantException
    def invariant_fails(obj):
        return (False, "error_code")
    class TestClass(PClass):
        x = field(invariant=invariant_fails)
    try:
        TestClass(x=1)
    except InvariantException as e:
        assert e.error_codes == ("error_code",)
        assert e.missing_fields == ()
        assert str(e) == "Field invariant failed"

def test_missing_mandatory_fields_trigger_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
    except InvariantException as e:
        assert e.error_codes == ()
        assert e.missing_fields == ("TestClass.x",)
        assert str(e) == "Field invariant failed"

def test_both_invariant_errors_and_missing_fields_trigger_exception():
    from pyrsistent import PClass, field, InvariantException
    def invariant_fails(obj):
        return (False, "error_code")
    class TestClass(PClass):
        x = field(mandatory=True, invariant=invariant_fails)
        y = field(mandatory=True)
    try:
        TestClass()
    except InvariantException as e:
        assert "error_code" in e.error_codes
        assert "TestClass.x" in e.missing_fields
        assert "TestClass.y" in e.missing_fields
        assert str(e) == "Field invariant failed"


# LLM-generated content at query #13
#--------------------------

def test_check_and_set_attr_sets_value_when_valid():
    class MockField:
        type = [int]
        def invariant(self, value):
            return True, None
    class MockClass:
        pass
    field = MockField()
    name = "test_field"
    value = 42
    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    assert getattr(result, name) == value
    assert len(invariant_errors) == 0

def test_check_and_set_attr_raises_ptype_error_on_type_mismatch():
    class MockField:
        type = [int]
        def invariant(self, value):
            return True, None
    class MockClass:
        pass
    field = MockField()
    name = "test_field"
    value = "not_an_int"
    result = MockClass()
    invariant_errors = []
    try:
        _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
        assert False
    except PTypeError as e:
        assert e.destination_cls == MockClass
        assert e.field_name == name
        assert e.expected_type == [int]
        assert e.actual_type == str

def test_check_and_set_attr_appends_error_on_invariant_failure():
    class MockField:
        type = [int]
        def invariant(self, value):
            return False, "invariant_failed"
    class MockClass:
        pass
    field = MockField()
    name = "test_field"
    value = 42
    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    assert not hasattr(result, name)
    assert invariant_errors == ["invariant_failed"]

def test_check_and_set_attr_handles_multiple_types():
    class MockField:
        type = [int, str]
        def invariant(self, value):
            return True, None
    class MockClass:
        pass
    field = MockField()
    name = "test_field"
    value_str = "valid_string"
    result_str = MockClass()
    invariant_errors_str = []
    _check_and_set_attr(MockClass, field, name, value_str, result_str, invariant_errors_str)
    assert getattr(result_str, name) == value_str
    value_int = 123
    result_int = MockClass()
    invariant_errors_int = []
    _check_and_set_attr(MockClass, field, name, value_int, result_int, invariant_errors_int)
    assert getattr(result_int, name) == value_int

def test_check_and_set_attr_does_not_set_on_invariant_failure():
    class MockField:
        type = [int]
        def invariant(self, value):
            return False, "error"
    class MockClass:
        pass
    field = MockField()
    name = "test_field"
    value = 42
    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    assert not hasattr(result, name)
    assert invariant_errors == ["error"]


# LLM-generated content at query #14
#--------------------------

def test_hash_returns_same_value_for_equal_objects():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert hash(obj1) == hash(obj2)

def test_hash_returns_different_value_for_different_objects():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=3, y=4)
    assert hash(obj1) != hash(obj2)

def test_hash_consistent_across_multiple_calls():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    obj = TestClass(x=5)
    first_hash = hash(obj)
    second_hash = hash(obj)
    assert first_hash == second_hash

def test_hash_handles_missing_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    obj = TestClass(x=10)
    hash_value = hash(obj)
    assert isinstance(hash_value, int)

def test_hash_uses_all_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    obj1 = TestClass(a=1, b=2, c=3)
    obj2 = TestClass(a=1, b=2, c=4)
    assert hash(obj1) != hash(obj2)


# LLM-generated content at query #15
#--------------------------

def test___new___creates_instance_with_valid_kwargs():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test___new___raises_attribute_error_on_extra_kwargs():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, y=2)
        assert False
    except AttributeError as e:
        assert "'y' are not among the specified fields for TestClass" in str(e)

def test___new___sets_initial_values_for_non_provided_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5

def test___new___raises_invariant_exception_on_missing_mandatory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=5)
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)
        assert "TestClass.x" in str(e)

def test___new___raises_invariant_exception_on_field_invariant_failure():
    from pyrsistent import PClass, field
    def invariant_check(v):
        return v > 0, "value must be positive"
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    try:
        TestClass(x=-1)
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___raises_type_error_on_invalid_field_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="string")
        assert False
    except Exception as e:
        assert "Invalid type for field TestClass.x" in str(e)

def test___new___handles_callable_initial():
    from pyrsistent import PClass, field
    def get_default():
        return 42
    class TestClass(PClass):
        x = field(initial=get_default)
    instance = TestClass()
    assert instance.x == 42

def test___new___sets_factory_fields_when_provided():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test___new___ignores_extra_kwargs_when_ignore_extra_true():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(ignore_extra=True, x=1, y=2)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

def test___new___checks_global_invariants():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x != obj.y, "x and y must be different"
    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = (global_invariant,)
    try:
        TestClass(x=1, y=1)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___freezes_instance_after_creation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


# LLM-generated content at query #16
#--------------------------

def test___reduce___returns_correct_tuple():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 10, 'y': 20}

def test___reduce___handles_missing_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 10}

def test___reduce___with_no_fields():
    from pyrsistent import PClass
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert result[1][1] == {}

def test___reduce___preserves_field_order():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    instance = TestClass(a=1, b=2, c=3)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] is TestClass
    assert list(result[1][1].keys()) == ['a', 'b', 'c']
    assert result[1][1] == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #17
#--------------------------

def test_check_and_set_attr_invariant_failure():
    class MockField:
        def __init__(self, type, invariant_result):
            self.type = type
            self.invariant = lambda x: invariant_result
    class MockClass:
        pass
    field = MockField(type=None, invariant_result=(False, "error"))
    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, "test_field", "test_value", result, invariant_errors)
    assert not hasattr(result, "test_field")
    assert invariant_errors == ["error"]


# LLM-generated content at query #18
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_invariant_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True, invariant=lambda v: (v > 0, 'error1'))
    try:
        TestClass(x=-1)
    except InvariantException as e:
        assert e.error_codes == ('error1',)
        assert e.missing_fields == ()
        assert str(e) == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #19
#--------------------------

def test___new___creates_instance_with_valid_fields():
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test___new___raises_error_on_extra_fields():
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, y=2)
        assert False
    except AttributeError as e:
        assert "'y' are not among the specified fields" in str(e)

def test___new___applies_initial_values():
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test___new___raises_error_on_missing_mandatory_field():
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test___new___validates_field_type():
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="string")
        assert False
    except PTypeError as e:
        assert "Invalid type for field" in str(e)

def test___new___validates_field_invariant():
    def positive(value):
        return value > 0, "not_positive"
    class TestClass(PClass):
        x = field(invariant=positive)
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test___new___applies_global_invariants():
    def global_invariant(obj):
        return obj.x != obj.y, "x_equals_y"
    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_invariant]
    try:
        TestClass(x=1, y=1)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___handles_factory_fields():
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
    instance = TestClass(x=5)
    assert instance.x == 10

def test___new___handles_ignore_extra():
    class TestClass(PClass):
        x = field()
    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

def test___new___freezes_instance():
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


# LLM-generated content at query #20
#--------------------------

def test_set_updates_field_with_keyword_argument():
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a2.x == 2
    assert a.x == 1

def test_set_updates_field_with_positional_argument():
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set('x', 3)
    assert a2.x == 3
    assert a.x == 1

def test_set_returns_new_instance():
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a is not a2

def test_set_preserves_other_fields():
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=10)
    a2 = a.set(x=2)
    assert a2.x == 2
    assert a2.y == 10
    assert a.x == 1
    assert a.y == 10

def test_set_multiple_fields():
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=10)
    a2 = a.set(x=2, y=20)
    assert a2.x == 2
    assert a2.y == 20
    assert a.x == 1
    assert a.y == 10

def test_set_with_mandatory_field_missing_initial():
    class AClass(PClass):
        x = field(mandatory=True)
        y = field()
    a = AClass(x=1)
    a2 = a.set(y=5)
    assert a2.x == 1
    assert a2.y == 5

def test_set_with_factory_field():
    class AClass(PClass):
        x = field(factory=lambda v: v * 2)
    a = AClass(x=1)
    a2 = a.set(x=3)
    assert a2.x == 6

def test_set_raises_attribute_error_for_unknown_field():
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    try:
        a.set(z=5)
        assert False
    except AttributeError:
        pass

def test_set_with_initial_field():
    class AClass(PClass):
        x = field(initial=10)
    a = AClass()
    a2 = a.set(x=20)
    assert a2.x == 20
    assert a.x == 10

def test_set_on_empty_pclass():
    class AClass(PClass):
        pass
    a = AClass()
    a2 = a.set()
    assert a == a2

def test_set_maintains_invariants():
    class AClass(PClass):
        x = field(invariant=lambda x: (x > 0, 'x must be positive'))
    a = AClass(x=1)
    try:
        a.set(x=-1)
        assert False
    except InvariantException:
        pass

def test_set_with_ignore_extra_in_factory():
    class Inner(PClass):
        a = field()
    class AClass(PClass):
        x = field(type=Inner)
    inner = Inner(a=1)
    a = AClass(x=inner)
    a2 = a.set(x={'a': 2, 'b': 3})
    assert a2.x.a == 2

def test_set_equality():
    class AClass(PClass):
        x = field()
        y = field()
    a1 = AClass(x=1, y=2)
    a2 = a1.set(x=3)
    a3 = AClass(x=3, y=2)
    assert a2 == a3
    assert a1 != a2

def test_set_hash_consistency():
    class AClass(PClass):
        x = field()
        y = field()
    a1 = AClass(x=1, y=2)
    a2 = a1.set(x=3)
    a3 = AClass(x=3, y=2)
    assert hash(a2) == hash(a3)

def test_set_with_none_value():
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=None)
    assert a2.x is None

def test_set_with_false_value():
    class AClass(PClass):
        x = field()
    a = AClass(x=True)
    a2 = a.set(x=False)
    assert a2.x is False

def test_set_with_empty_string():
    class AClass(PClass):
        x = field()
    a = AClass(x="hello")
    a2 = a.set(x="")
    assert a2.x == ""

def test_set_with_list_value():
    class AClass(PClass):
        x = field()
    a = AClass(x=[1, 2])
    a2 = a.set(x=[3, 4])
    assert a2.x == [3, 4]

def test_set_with_dict_value():
    class AClass(PClass):
        x = field()
    a = AClass(x={'a': 1})
    a2 = a.set(x={'b': 2})
    assert a2.x == {'b': 2}

def test_set_with_custom_object():
    class Inner:
        pass
    class AClass(PClass):
        x = field()
    inner1 = Inner()
    inner2 = Inner()
    a = AClass(x=inner1)
    a2 = a.set(x=inner2)
    assert a2.x is inner2

def test_set_does_not_modify_original():
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    original_x = a.x
    original_y = a.y
    a.set(x=10, y=20)
    assert a.x == original_x
    assert a.y == original_y


# LLM-generated content at query #21
#--------------------------

def test_set_updates_field_with_keyword_argument():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    updated = instance.set(x=2)
    assert updated.x == 2
    assert instance.x == 1

def test_set_updates_field_with_positional_arguments():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    updated = instance.set('x', 2)
    assert updated.x == 2
    assert instance.x == 1

def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    updated = instance.set(x=3)
    assert updated is not instance
    assert updated.x == 3
    assert updated.y == 2

def test_set_preserves_other_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    updated = instance.set(x=3)
    assert updated.y == 2

def test_set_with_multiple_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    updated = instance.set(x=3, y=4)
    assert updated.x == 3
    assert updated.y == 4

def test_set_raises_attribute_error_for_unknown_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.set(z=2)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass

def test_set_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, factory=int)
    instance = TestClass(x=1)
    updated = instance.set(x='2')
    assert updated.x == 2

def test_set_maintains_immutability():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    updated = instance.set(x=2)
    try:
        instance.x = 3
        assert False, "Expected AttributeError"
    except AttributeError:
        pass
    try:
        updated.x = 3
        assert False, "Expected AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #22
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
        y = field(type=str, mandatory=True)
        z = field(initial=10)
    instance = TestClass(x=5, y="test")
    assert instance.x == 5
    assert instance.y == "test"
    assert instance.z == 10

def test___new___raises_type_error_for_invalid_field_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="not_an_int")
        assert False
    except Exception as e:
        assert "Invalid type" in str(e)

def test___new___raises_invariant_exception_for_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___raises_attribute_error_for_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, y=2)
        assert False
    except Exception as e:
        assert "are not among the specified fields" in str(e)

def test___new___handles_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    instance = TestClass()
    assert instance.x == 42

def test___new___sets_factory_fields_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=dict, factory=lambda v, **kwargs: v)
    instance = TestClass(x={"a": 1}, ignore_extra=True)
    assert instance.x == {"a": 1}

def test___new___raises_invariant_exception_for_field_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(v):
        return v > 0, "value_not_positive"
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "value_not_positive" in e.invariant_errors

def test___new___checks_global_invariants():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x + obj.y == 10, "sum_not_ten"
    class TestClass(PClass):
        x = field()
        y = field()
        _invariants = [global_invariant]
    try:
        TestClass(x=3, y=8)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___freezes_instance_after_creation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


# LLM-generated content at query #23
#--------------------------

def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=20)
        assert False
    except Exception as e:
        assert isinstance(e, type(e).__module__ + '.' + type(e).__name__)

def test_pclass_constructor_with_extra_field_raises_attribute_error():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=10, z=30)
        assert False
    except AttributeError as e:
        assert "'z'" in str(e)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
        y = field()
    instance = TestClass(y=20)
    assert instance.x == 5
    assert instance.y == 20

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 100)
        y = field()
    instance = TestClass(y=200)
    assert instance.x == 100
    assert instance.y == 200

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10, z=30, ignore_extra=True)
    assert instance.x == 10
    assert not hasattr(instance, 'z')

def test_pclass_constructor_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(value):
        if value < 0:
            return (('value must be non-negative',),)
        return ()
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    try:
        TestClass(x=-5)
        assert False
    except InvariantException:
        pass

def test_pclass_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException, invariant
    class TestClass(PClass):
        x = field()
        y = field()
        __invariant__ = invariant(lambda d: d['x'] + d['y'] == 100, 'sum must be 100')
    try:
        TestClass(x=30, y=80)
        assert False
    except InvariantException:
        pass

def test_pclass_constructor_creates_frozen_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    try:
        instance.x = 20
        assert False
    except AttributeError:
        pass


# LLM-generated content at query #24
#--------------------------

def test_is_pclass_true_for_pclass_bases():
    from pyrsistent._pclass import _is_pclass
    class A(metaclass=PClassMeta):
        pass
    class B(A):
        pass
    result = _is_pclass((A,))
    assert result is True


# LLM-generated content at query #25
#--------------------------

def test___reduce___returns_correct_tuple():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 10, 'y': 20}


# LLM-generated content at query #26
#--------------------------

def test___eq___with_same_class_and_equal_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    result = obj1.__eq__(obj2)
    assert result is True

def test___eq___with_same_class_and_different_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=3, y=4)
    result = obj1.__eq__(obj2)
    assert result is False

def test___eq___with_different_class():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    obj1 = TestClass1(x=1)
    obj2 = TestClass2(x=1)
    result = obj1.__eq__(obj2)
    assert result is NotImplemented

def test___eq___with_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    obj = TestClass(x=1)
    other = object()
    result = obj.__eq__(other)
    assert result is NotImplemented

def test___eq___with_missing_field_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    result = obj1.__eq__(obj2)
    assert result is True

def test___eq___with_one_missing_field_and_one_present():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1)
    result = obj1.__eq__(obj2)
    assert result is False


# LLM-generated content at query #27
#--------------------------

def test_hash_returns_same_value_for_equal_objects():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert hash(obj1) == hash(obj2)

def test_hash_returns_different_value_for_different_objects():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=3, y=4)
    assert hash(obj1) != hash(obj2)

def test_hash_consistent_across_multiple_calls():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    obj = TestClass(x=5)
    first_hash = hash(obj)
    second_hash = hash(obj)
    assert first_hash == second_hash

def test_hash_handles_missing_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj = TestClass(x=10)
    hash_value = hash(obj)
    assert isinstance(hash_value, int)

def test_hash_uses_all_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    obj1 = TestClass(a=1, b=2, c=3)
    obj2 = TestClass(a=1, b=2, c=4)
    assert hash(obj1) != hash(obj2)

def test_hash_works_with_none_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=None, y=5)
    obj2 = TestClass(x=None, y=5)
    assert hash(obj1) == hash(obj2)

def test_hash_works_with_complex_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    obj1 = TestClass(x=[1, 2, 3])
    obj2 = TestClass(x=[1, 2, 3])
    assert hash(obj1) == hash(obj2)

def test_hash_works_with_custom_objects():
    from pyrsistent import PClass, field
    class Inner:
        def __init__(self, val):
            self.val = val
        def __eq__(self, other):
            return self.val == other.val
        def __hash__(self):
            return hash(self.val)
    class TestClass(PClass):
        x = field()
    inner1 = Inner(10)
    inner2 = Inner(10)
    obj1 = TestClass(x=inner1)
    obj2 = TestClass(x=inner2)
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #28
#--------------------------

def test_hash_returns_same_value_for_equal_objects():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    assert hash(obj1) == hash(obj2)

def test_hash_returns_different_value_for_different_objects():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=3, y=4)
    assert hash(obj1) != hash(obj2)

def test_hash_uses_all_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    obj = TestClass(a=10, b=20)
    expected_hash = hash(tuple([('a', 10), ('b', 20)]))
    assert hash(obj) == expected_hash

def test_hash_works_with_missing_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(initial=5)
    obj = TestClass(x=1)
    assert hash(obj) == hash(tuple([('x', 1), ('y', 5)]))

def test_hash_consistent_across_multiple_calls():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    obj = TestClass(x=42)
    first_hash = hash(obj)
    second_hash = hash(obj)
    assert first_hash == second_hash


# LLM-generated content at query #29
#--------------------------

def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=20)
        assert False
    except Exception as e:
        assert "missing_fields" in str(e)

def test_pclass_constructor_with_extra_field_raises_attribute_error():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=10, z=30)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
        y = field()
    instance = TestClass(y=20)
    assert instance.x == 5
    assert instance.y == 20

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 100)
        y = field()
    instance = TestClass(y=50)
    assert instance.x == 100
    assert instance.y == 50

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(ignore_extra=True, x=10, z=30)
    assert instance.x == 10
    assert not hasattr(instance, 'z')

def test_pclass_constructor_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(value):
        if value < 0:
            return (f"Value must be non-negative, got {value}",)
        return ()
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    try:
        TestClass(x=-5)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException, invariant
    class TestClass(PClass):
        x = field()
        y = field()
        __invariant__ = invariant(lambda obj: obj.x + obj.y == 10, "Sum must be 10")
    try:
        TestClass(x=3, y=8)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_factory_and_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, ignore_extra=True, x=10, y=20, z=30)
    assert instance.x == 10
    assert instance.y == 20
    assert not hasattr(instance, 'z')


# LLM-generated content at query #30
#--------------------------

def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=20)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_with_extra_field_raises_attribute_error():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=10, z=30)
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
        y = field()
    instance = TestClass(y=20)
    assert instance.x == 5
    assert instance.y == 20

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 100)
        y = field()
    instance = TestClass(y=50)
    assert instance.x == 100
    assert instance.y == 50

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_pclass_constructor_with_ignore_extra_true():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(ignore_extra=True, x=10, z=30)
    assert instance.x == 10
    assert not hasattr(instance, 'z')

def test_pclass_constructor_with_ignore_extra_false():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(ignore_extra=False, x=10, z=30)
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)

def test_pclass_constructor_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def check_positive(value):
        return value > 0, "Value must be positive"
    class TestClass(PClass):
        x = field(invariant=check_positive)
    try:
        TestClass(x=-5)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x + obj.y > 0, "Sum must be positive"
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [global_invariant]
    try:
        TestClass(x=-10, y=5)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    try:
        instance.x = 20
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)

def test_pclass_constructor_equality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=20)
    assert instance1 == instance2

def test_pclass_constructor_inequality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=30)
    assert instance1 != instance2

def test_pclass_constructor_hash_equality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=20)
    assert hash(instance1) == hash(instance2)

def test_pclass_constructor_pickle_support():
    import pickle
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    pickled = pickle.dumps(instance)
    unpickled = pickle.loads(pickled)
    assert instance == unpickled
    assert instance.x == unpickled.x
    assert instance.y == unpickled.y


# LLM-generated content at query #31
#--------------------------

def test_repr_returns_correct_format():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y="hello")
    result = repr(instance)
    expected = "TestClass(x=10, y='hello')"
    assert result == expected

def test_repr_with_no_fields():
    from pyrsistent import PClass, field
    class EmptyClass(PClass):
        pass
    instance = EmptyClass()
    result = repr(instance)
    expected = "EmptyClass()"
    assert result == expected

def test_repr_with_one_field():
    from pyrsistent import PClass, field
    class SingleFieldClass(PClass):
        name = field()
    instance = SingleFieldClass(name="test")
    result = repr(instance)
    expected = "SingleFieldClass(name='test')"
    assert result == expected

def test_repr_with_special_characters_in_field_value():
    from pyrsistent import PClass, field
    class SpecialClass(PClass):
        text = field()
    instance = SpecialClass(text="line\nbreak")
    result = repr(instance)
    expected = "SpecialClass(text='line\\nbreak')"
    assert result == expected

def test_repr_with_numeric_field_names_and_values():
    from pyrsistent import PClass, field
    class NumericClass(PClass):
        a = field()
        b = field()
    instance = NumericClass(a=1, b=2.5)
    result = repr(instance)
    expected = "NumericClass(a=1, b=2.5)"
    assert result == expected

def test_repr_uses_class_name_correctly():
    from pyrsistent import PClass, field
    class DifferentName(PClass):
        value = field()
    instance = DifferentName(value=42)
    result = repr(instance)
    expected = "DifferentName(value=42)"
    assert result == expected

def test_repr_with_boolean_and_none_values():
    from pyrsistent import PClass, field
    class BoolNoneClass(PClass):
        flag = field()
        empty = field()
    instance = BoolNoneClass(flag=True, empty=None)
    result = repr(instance)
    expected = "BoolNoneClass(flag=True, empty=None)"
    assert result == expected

def test_repr_with_list_field_value():
    from pyrsistent import PClass, field
    class ListClass(PClass):
        items = field()
    instance = ListClass(items=[1, 2, 3])
    result = repr(instance)
    expected = "ListClass(items=[1, 2, 3])"
    assert result == expected

def test_repr_with_dict_field_value():
    from pyrsistent import PClass, field
    class DictClass(PClass):
        mapping = field()
    instance = DictClass(mapping={'a': 1})
    result = repr(instance)
    expected = "DictClass(mapping={'a': 1})"
    assert result == expected

def test_repr_with_initial_field_values():
    from pyrsistent import PClass, field
    class InitialClass(PClass):
        x = field(initial=5)
        y = field(initial='default')
    instance = InitialClass()
    result = repr(instance)
    expected = "InitialClass(x=5, y='default')"
    assert result == expected


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_remove_existing_item():
    original = object()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.remove('a')
    assert result is evolver
    assert 'a' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._factory_fields == set()

def test_remove_non_existing_item():
    original = object()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    try:
        evolver.remove('b')
        assert False
    except AttributeError as e:
        assert str(e) == 'b'

def test_remove_clears_factory_fields():
    original = object()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 3)
    evolver.remove('a')
    assert 'a' not in evolver._pclass_evolver_data
    assert 'a' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True

def test_remove_does_not_affect_other_items():
    original = object()
    initial_dict = {'a': 1, 'b': 2, 'c': 3}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.remove('b')
    assert evolver._pclass_evolver_data == {'a': 1, 'c': 3}
    assert evolver._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #2
#--------------------------

def test_set_with_keyword_argument():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a.x == 1
    assert a2.x == 2

def test_set_with_positional_arguments():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set('x', 3)
    assert a.x == 1
    assert a2.x == 3

def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a is not a2

def test_set_preserves_other_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=10)
    a2 = a.set(x=2)
    assert a.x == 1
    assert a.y == 10
    assert a2.x == 2
    assert a2.y == 10

def test_set_with_multiple_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=10)
    a2 = a.set(x=2, y=20)
    assert a.x == 1
    assert a.y == 10
    assert a2.x == 2
    assert a2.y == 20

def test_set_on_pclass_with_mandatory_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(mandatory=True)
        y = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a.x == 1
    assert a2.x == 2

def test_set_with_initial_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(initial=5)
    a = AClass()
    a2 = a.set(x=10)
    assert a.x == 5
    assert a2.x == 10

def test_set_raises_attribute_error_for_unknown_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    try:
        a.set(z=2)
        assert False
    except AttributeError:
        pass

def test_set_with_factory_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(type=int, factory=int)
    a = AClass(x=1)
    a2 = a.set(x='2')
    assert a2.x == 2

def test_set_maintains_immutability():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    try:
        a.x = 3
        assert False
    except AttributeError:
        pass
    try:
        a2.x = 3
        assert False
    except AttributeError:
        pass


# LLM-generated content at query #3
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
        y = field(type=str, mandatory=True)
    instance = TestClass(x=10, y="test")
    assert instance.x == 10
    assert instance.y == "test"

def test___new___raises_InvariantException_on_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
        assert False
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields

def test___new___raises_InvariantException_on_field_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(value):
        return value > 0, "value must be positive"
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    try:
        TestClass(x=-5)
        assert False
    except InvariantException as e:
        assert "value must be positive" in e.invariant_errors

def test___new___raises_AttributeError_on_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, y=2)
        assert False
    except AttributeError as e:
        assert "y" in str(e)

def test___new___uses_initial_value_when_field_not_provided():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=100)
        y = field(initial=lambda: "default")
    instance = TestClass()
    assert instance.x == 100
    assert instance.y == "default"

def test___new___raises_PTypeError_on_type_mismatch():
    from pyrsistent import PClass, field, PTypeError
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="not_an_int")
        assert False
    except PTypeError as e:
        assert "Invalid type for field TestClass.x" in str(e)

def test___new___handles_factory_fields_with_ignore_extra():
    from pyrsistent import PClass, field
    def factory(value, ignore_extra=False):
        return value * 2
    class TestClass(PClass):
        x = field(type=int, factory=factory)
    instance = TestClass(x=5, ignore_extra=True)
    assert instance.x == 10

def test___new___sets_frozen_attribute():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert instance._pclass_frozen == True

def test___new___checks_global_invariants():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x + obj.y == 10, "sum must be 10"
    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = (global_invariant,)
    try:
        TestClass(x=3, y=4)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)
    valid_instance = TestClass(x=6, y=4)
    assert valid_instance.x == 6
    assert valid_instance.y == 4

def test___new___with_factory_fields_parameter():
    from pyrsistent import PClass, field
    def factory(value):
        return value.upper()
    class TestClass(PClass):
        x = field(factory=factory)
        y = field()
    instance = TestClass(_factory_fields={'x'}, x="hello", y="world")
    assert instance.x == "HELLO"
    assert instance.y == "world"


# LLM-generated content at query #4
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, mandatory=True)
        y = field(type=int, initial=10)
    instance = TestClass(x=5)
    assert instance.x == 5
    assert instance.y == 10

def test___new___raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, mandatory=True)
    try:
        TestClass()
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___raises_on_invalid_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="string")
        assert False
    except Exception as e:
        assert "Invalid type" in str(e)

def test___new___raises_on_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x=1, y=2)
        assert False
    except Exception as e:
        assert "are not among the specified fields" in str(e)

def test___new___handles_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, factory=lambda v: v * 2)
    instance = TestClass(x=5)
    assert instance.x == 10

def test___new___handles_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

def test___new___sets_initial_from_callable():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    instance = TestClass()
    assert instance.x == 42

def test___new___invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(value):
        return value > 0, "value must be positive"
    class TestClass(PClass):
        x = field(type=int, invariant=invariant_check)
    try:
        TestClass(x=-1)
        assert False
    except InvariantException:
        pass

def test___new___global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException, invariant
    class TestClass(PClass):
        x = field(type=int)
        y = field(type=int)
        @invariant(lambda self: self.x + self.y > 0)
        def sum_positive(self):
            return self.x + self.y > 0, "sum must be positive"
    try:
        TestClass(x=-5, y=2)
        assert False
    except InvariantException:
        pass

def test___new___freezes_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError:
        pass


# LLM-generated content at query #5
#--------------------------

def test___hash___returns_same_hash_for_equal_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=20)
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 == hash2

def test___hash___returns_different_hash_for_different_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=30, y=40)
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 != hash2

def test___hash___works_with_missing_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    instance = TestClass(x=10)
    hash_value = instance.__hash__()
    assert isinstance(hash_value, int)

def test___hash___consistent_across_multiple_calls():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    instance = TestClass(a=5, b=15)
    hash1 = instance.__hash__()
    hash2 = instance.__hash__()
    assert hash1 == hash2

def test___hash___handles_nested_structures():
    from pyrsistent import PClass, field, pvector
    class InnerClass(PClass):
        val = field()
    class OuterClass(PClass):
        inner = field()
        vec = field()
    inner = InnerClass(val=100)
    vec = pvector([1, 2, 3])
    instance = OuterClass(inner=inner, vec=vec)
    hash_value = instance.__hash__()
    assert isinstance(hash_value, int)

def test___hash___with_complex_types():
    from pyrsistent import PClass, field, pmap
    class TestClass(PClass):
        mapping = field()
        number = field()
    mapping = pmap({'a': 1, 'b': 2})
    instance = TestClass(mapping=mapping, number=42)
    hash_value = instance.__hash__()
    assert isinstance(hash_value, int)


# LLM-generated content at query #6
#--------------------------

def test___new___single_inheritance_without_fields():
    class Base(metaclass=PClassMeta):
        pass
    class Derived(Base):
        pass
    assert Derived._pclass_fields == {}
    assert Derived._pclass_invariants == ()
    assert Derived.__slots__ == ('_pclass_frozen', '__weakref__')

def test___new___multiple_inheritance_without_fields():
    class A(metaclass=PClassMeta):
        pass
    class B(metaclass=PClassMeta):
        pass
    class C(A, B):
        pass
    assert C._pclass_fields == {}
    assert C._pclass_invariants == ()
    assert C.__slots__ == ('_pclass_frozen', '__weakref__')

def test___new___with_fields():
    from pyrsistent import field
    class MyClass(metaclass=PClassMeta):
        x = field()
        y = field()
    assert 'x' in MyClass._pclass_fields
    assert 'y' in MyClass._pclass_fields
    assert MyClass._pclass_fields['x'].__class__.__name__ == '_PField'
    assert MyClass._pclass_fields['y'].__class__.__name__ == '_PField'
    assert MyClass._pclass_invariants == ()
    assert '_pclass_frozen' in MyClass.__slots__
    assert 'x' not in MyClass.__slots__
    assert 'y' not in MyClass.__slots__
    assert '__weakref__' in MyClass.__slots__

def test___new___inherits_fields():
    from pyrsistent import field
    class Base(metaclass=PClassMeta):
        a = field()
    class Derived(Base):
        b = field()
    assert 'a' in Derived._pclass_fields
    assert 'b' in Derived._pclass_fields
    assert Derived._pclass_fields['a'].__class__.__name__ == '_PField'
    assert Derived._pclass_fields['b'].__class__.__name__ == '_PField'
    assert Derived._pclass_invariants == ()
    assert '_pclass_frozen' in Derived.__slots__
    assert '__weakref__' in Derived.__slots__

def test___new___with_invariant():
    def my_invariant(instance):
        return True, ()
    class MyClass(metaclass=PClassMeta):
        __invariant__ = my_invariant
    assert len(MyClass._pclass_invariants) == 1
    assert callable(MyClass._pclass_invariants[0])
    assert MyClass._pclass_fields == {}
    assert '_pclass_frozen' in MyClass.__slots__
    assert '__weakref__' in MyClass.__slots__

def test___new___inherits_invariants():
    def inv1(instance):
        return True, ()
    def inv2(instance):
        return True, ()
    class Base(metaclass=PClassMeta):
        __invariant__ = inv1
    class Derived(Base):
        __invariant__ = inv2
    assert len(Derived._pclass_invariants) == 2
    assert callable(Derived._pclass_invariants[0])
    assert callable(Derived._pclass_invariants[1])
    assert Derived._pclass_fields == {}
    assert '_pclass_frozen' in Derived.__slots__
    assert '__weakref__' in Derived.__slots__

def test___new___non_callable_invariant_raises():
    class BadClass(metaclass=PClassMeta):
        __invariant__ = "not callable"
    raised = False
    try:
        class Test(BadClass):
            pass
    except TypeError:
        raised = True
    assert raised

def test___new___slots_contains_only_frozen_and_weakref_for_top_pclass():
    class Top(metaclass=PClassMeta):
        pass
    assert Top.__slots__ == ('_pclass_frozen', '__weakref__')

def test___new___slots_excludes_fields():
    from pyrsistent import field
    class WithFields(metaclass=PClassMeta):
        f = field()
    assert 'f' not in WithFields.__slots__
    assert '_pclass_frozen' in WithFields.__slots__
    assert '__weakref__' in WithFields.__slots__

def test___new___fields_dict_cleaned():
    from pyrsistent import field
    class MyClass(metaclass=PClassMeta):
        f = field()
    assert 'f' not in MyClass.__dict__
    assert 'f' in MyClass._pclass_fields


# LLM-generated content at query #7
#--------------------------

def test_PClassMeta_new_with_inherited_fields():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class CheckedType:
        pass
    class Parent(CheckedType, metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
        x = _PField()
    class Child(Parent):
        y = _PField()
    assert '_pclass_fields' in Child.__dict__
    assert 'x' in Child._pclass_fields
    assert 'y' in Child._pclass_fields
    assert '_pclass_invariants' in Child.__dict__
    assert len(Child._pclass_invariants) == 1
    assert '__slots__' in Child.__dict__
    assert '_pclass_frozen' in Child.__slots__
    assert 'x' in Child.__slots__
    assert 'y' in Child.__slots__

def test_PClassMeta_new_without_inheritance():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class CheckedType:
        pass
    class Single(CheckedType, metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
        a = _PField()
    assert '_pclass_fields' in Single.__dict__
    assert 'a' in Single._pclass_fields
    assert '_pclass_invariants' in Single.__dict__
    assert len(Single._pclass_invariants) == 1
    assert '__slots__' in Single.__dict__
    assert '_pclass_frozen' in Single.__slots__
    assert 'a' in Single.__slots__
    assert '__weakref__' in Single.__slots__

def test_PClassMeta_new_with_multiple_invariants():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class CheckedType:
        pass
    class Base(CheckedType, metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class Derived(Base):
        __invariant__ = lambda self: (False, ('error',))
    assert '_pclass_invariants' in Derived.__dict__
    assert len(Derived._pclass_invariants) == 2

def test_PClassMeta_new_with_non_callable_invariant():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class CheckedType:
        pass
    try:
        class Invalid(CheckedType, metaclass=PClassMeta):
            __invariant__ = 'not callable'
        assert False
    except TypeError:
        pass

def test_PClassMeta_new_without_fields():
    from pyrsistent._field_common import set_fields
    from pyrsistent._checked_types import store_invariants
    class CheckedType:
        pass
    class NoFields(CheckedType, metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    assert '_pclass_fields' in NoFields.__dict__
    assert NoFields._pclass_fields == {}
    assert '__slots__' in NoFields.__dict__
    assert '_pclass_frozen' in NoFields.__slots__
    assert len(NoFields.__slots__) == 2


# LLM-generated content at query #8
#--------------------------

def test___reduce___returns_correct_tuple():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 10, 'y': 20}

def test___reduce___with_missing_attribute():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 10}

def test___reduce___with_no_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=False)
        y = field(mandatory=False)
    instance = TestClass()
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {}

def test___reduce___with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 100)
        y = field(initial=200)
    instance = TestClass()
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 100, 'y': 200}

def test___reduce___after_set_operation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=3)
    result = new_instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 3, 'y': 2}


# LLM-generated content at query #9
#--------------------------

def test_set_with_keyword_argument():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a.x == 1
    assert a2.x == 2

def test_set_with_positional_arguments():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set('x', 3)
    assert a.x == 1
    assert a2.x == 3

def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=10)
    a2 = a.set(x=2)
    assert a is not a2
    assert a.x == 1
    assert a.y == 10
    assert a2.x == 2
    assert a2.y == 10

def test_set_preserves_other_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=20)
    a2 = a.set(x=2)
    assert a.y == 20
    assert a2.y == 20

def test_set_with_multiple_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(x=10, y=20)
    assert a.x == 1
    assert a.y == 2
    assert a2.x == 10
    assert a2.y == 20

def test_set_on_pclass_with_mandatory_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(mandatory=True)
        y = field()
    a = AClass(x=1)
    a2 = a.set(x=5)
    assert a.x == 1
    assert a2.x == 5

def test_set_with_factory_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(factory=lambda v: v * 2)
    a = AClass(x=1)
    a2 = a.set(x=3)
    assert a.x == 2
    assert a2.x == 6

def test_set_raises_no_error_for_extra_keyword():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2, extra=99)
    assert a2.x == 2

def test_set_maintains_immutability():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    try:
        a.x = 3
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass
    try:
        a2.x = 4
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass

def test_set_with_initial_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(initial=5)
        y = field()
    a = AClass(y=10)
    a2 = a.set(x=7)
    assert a.x == 5
    assert a.y == 10
    assert a2.x == 7
    assert a2.y == 10


# LLM-generated content at query #10
#--------------------------

def test_repr_with_single_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    result = repr(instance)
    expected = "TestClass(x=10)"
    assert result == expected

def test_repr_with_multiple_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    result = repr(instance)
    expected = "TestClass(x=1, y=2)"
    assert result == expected

def test_repr_with_string_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        name = field()
    instance = TestClass(name="test")
    result = repr(instance)
    expected = "TestClass(name='test')"
    assert result == expected

def test_repr_with_none_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        value = field()
    instance = TestClass(value=None)
    result = repr(instance)
    expected = "TestClass(value=None)"
    assert result == expected

def test_repr_with_list_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        items = field()
    instance = TestClass(items=[1, 2, 3])
    result = repr(instance)
    expected = "TestClass(items=[1, 2, 3])"
    assert result == expected

def test_repr_with_dict_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        data = field()
    instance = TestClass(data={'a': 1})
    result = repr(instance)
    expected = "TestClass(data={'a': 1})"
    assert result == expected

def test_repr_with_initial_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
    instance = TestClass()
    result = repr(instance)
    expected = "TestClass(x=5)"
    assert result == expected

def test_repr_with_mandatory_field_missing():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
    except InvariantException:
        pass

def test_repr_with_extra_fields_ignored():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass.create({'x': 1, 'extra': 2}, ignore_extra=True)
    result = repr(instance)
    expected = "TestClass(x=1)"
    assert result == expected

def test_repr_after_set():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=3)
    result = repr(new_instance)
    expected = "TestClass(x=3, y=2)"
    assert result == expected


# LLM-generated content at query #11
#--------------------------

def test___reduce___returns_correct_tuple():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 10, 'y': 20}

def test___reduce___with_missing_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10)
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 10}

def test___reduce___with_no_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass()
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert result[1][0] is TestClass
    assert result[1][1] == {}

def test___reduce___preserves_frozen_state():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=5)
    reduced = instance.__reduce__()
    restored = reduced[0](*reduced[1])
    assert restored == instance
    assert restored.x == 5

def test___reduce___works_with_custom_field_types():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
        y = field(type=str)
    instance = TestClass(x=42, y='hello')
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 42, 'y': 'hello'}


# LLM-generated content at query #12
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true_when_conditions_met():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    class TestClass(PClass):
        x = field(type=dict, factory=lambda v, ignore_extra=False: v)
    field_obj = TestClass._pclass_fields['x']
    result = is_field_ignore_extra_complaint(PClass, field_obj, True)
    assert result == True


# LLM-generated content at query #13
#--------------------------

def test_is_pclass_true():
    result = _is_pclass((CheckedType,))
    assert result == True

def test_is_pclass_false_multiple_bases():
    result = _is_pclass((CheckedType, object))
    assert result == False

def test_is_pclass_false_different_base():
    result = _is_pclass((object,))
    assert result == False

def test_is_pclass_false_empty():
    result = _is_pclass(())
    assert result == False


# LLM-generated content at query #14
#--------------------------

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
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_with_extra_field_raises_attribute_error():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, z=3)
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)

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
        x = field(initial=lambda: 100)
    instance = TestClass()
    assert instance.x == 100

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=5, y=10)
    assert instance.x == 5
    assert instance.y == 10

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1, z=3, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')

def test_pclass_constructor_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(value):
        if value < 0:
            return (f'Value must be non-negative, got {value}',)
        return ()
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    try:
        TestClass(x=-5)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException, invariant
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [invariant(lambda c: c.x + c.y == 10, 'Sum must be 10')]
    try:
        TestClass(x=3, y=8)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_creates_frozen_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)

def test_pclass_constructor_with_no_fields():
    from pyrsistent import PClass
    class TestClass(PClass):
        pass
    instance = TestClass()
    assert isinstance(instance, TestClass)

def test_pclass_constructor_with_field_serializer():
    from pyrsistent import PClass, field
    def serializer(format, value):
        return value * 2
    class TestClass(PClass):
        x = field(serializer=serializer)
    instance = TestClass(x=5)
    assert instance.serialize() == {'x': 10}

def test_pclass_constructor_equality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=3, y=4)
    assert instance1 == instance2
    assert instance1 != instance3

def test_pclass_constructor_hash():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert hash(instance1) == hash(instance2)

def test_pclass_constructor_pickle_support():
    import pickle
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    pickled = pickle.dumps(instance)
    unpickled = pickle.loads(pickled)
    assert unpickled == instance

def test_pclass_constructor_with_missing_value_handling():
    from pyrsistent import PClass, field, _MISSING_VALUE
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1)
    assert getattr(instance, 'y', _MISSING_VALUE) is _MISSING_VALUE

def test_pclass_constructor_repr():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert repr(instance) == "TestClass(x=1, y=2)"

def test_pclass_constructor_with_factory_fields_and_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=5, y=10, z=15, ignore_extra=True)
    assert instance.x == 5
    assert instance.y == 10
    assert not hasattr(instance, 'z')


# LLM-generated content at query #15
#--------------------------

def test_is_pclass_false_for_non_pclass_bases():
    class NonPClassBase:
        pass
    class TestClass(metaclass=PClassMeta):
        __slots__ = ()
        _pclass_fields = {}
        _pclass_invariants = ()
    bases = (NonPClassBase,)
    result = _is_pclass(bases)
    assert result is False


# LLM-generated content at query #16
#--------------------------

def test_repr_returns_correct_format():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y="hello")
    result = repr(instance)
    expected = "TestClass(x=10, y='hello')"
    assert result == expected


# LLM-generated content at query #17
#--------------------------

def test_serialize_without_custom_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y="hello")
    result = instance.serialize()
    expected = {"x": 10, "y": "hello"}
    assert result == expected

def test_serialize_with_custom_serializer():
    from pyrsistent import PClass, field
    def serialize_x(value, format):
        return value * 2
    class TestClass(PClass):
        x = field(type=int, serializer=serialize_x)
        y = field()
    instance = TestClass(x=5, y="world")
    result = instance.serialize()
    expected = {"x": 10, "y": "world"}
    assert result == expected

def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    def serializer_with_format(value, format):
        return f"{format}:{value}"
    class TestClass(PClass):
        data = field(serializer=serializer_with_format)
    instance = TestClass(data="test")
    result = instance.serialize(format="json")
    expected = {"data": "json:test"}
    assert result == expected

def test_serialize_missing_field_with_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=100)
        y = field()
    instance = TestClass(y=200)
    result = instance.serialize()
    expected = {"x": 100, "y": 200}
    assert result == expected

def test_serialize_empty_pclass():
    from pyrsistent import PClass, field
    class EmptyClass(PClass):
        pass
    instance = EmptyClass()
    result = instance.serialize()
    expected = {}
    assert result == expected

def test_serialize_with_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        required = field(mandatory=True)
        optional = field(initial=0)
    instance = TestClass(required="mandatory_value")
    result = instance.serialize()
    expected = {"required": "mandatory_value", "optional": 0}
    assert result == expected


# LLM-generated content at query #18
#--------------------------

def test_pclass_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_pclass_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=20)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_with_extra_field_raises_attribute_error():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=10, z=30)
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)

def test_pclass_constructor_with_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
        y = field()
    instance = TestClass(y=20)
    assert instance.x == 5
    assert instance.y == 20

def test_pclass_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 100)
        y = field()
    instance = TestClass(y=20)
    assert instance.x == 100
    assert instance.y == 20

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10, z=30, ignore_extra=True)
    assert instance.x == 10
    assert not hasattr(instance, 'z')

def test_pclass_constructor_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(value):
        if value < 0:
            return (f'Value must be non-negative, got {value}',)
        return ()
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    try:
        TestClass(x=-5)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        if obj.x + obj.y != 100:
            return ('Sum must be 100',)
        return ()
    class TestClass(PClass):
        x = field()
        y = field()
        _pclass_invariants = [global_invariant]
    try:
        TestClass(x=30, y=80)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    try:
        instance.x = 20
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)

def test_pclass_constructor_with_no_fields():
    from pyrsistent import PClass
    class TestClass(PClass):
        pass
    instance = TestClass()
    assert isinstance(instance, TestClass)

def test_pclass_constructor_with_factory_and_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=10, y=20, z=30, ignore_extra=True)
    assert instance.x == 10
    assert instance.y == 20
    assert not hasattr(instance, 'z')

def test_pclass_constructor_with_missing_mandatory_and_invariant():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(value):
        if value < 0:
            return (f'Value must be non-negative, got {value}',)
        return ()
    class TestClass(PClass):
        x = field(mandatory=True, invariant=invariant_check)
        y = field()
    try:
        TestClass(y=20)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)


# LLM-generated content at query #19
#--------------------------

def test_check_and_set_attr_valid():
    class MockField:
        type = (int,)
        def invariant(self, value):
            return True, None
    class MockClass:
        __name__ = "MockClass"
    field = MockField()
    name = "test_field"
    value = 42
    result = type('result', (), {})()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    assert getattr(result, name) == value
    assert invariant_errors == []

def test_check_and_set_attr_invalid_type():
    class MockField:
        type = (int,)
        def invariant(self, value):
            return True, None
    class MockClass:
        __name__ = "MockClass"
    field = MockField()
    name = "test_field"
    value = "not_an_int"
    result = type('result', (), {})()
    invariant_errors = []
    try:
        _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
        assert False
    except PTypeError as e:
        assert e.destination_cls == MockClass
        assert e.field_name == name
        assert e.expected_type == (int,)
        assert e.actual_type == str

def test_check_and_set_attr_invariant_fails():
    class MockField:
        type = (int,)
        def invariant(self, value):
            return False, "invariant_error"
    class MockClass:
        __name__ = "MockClass"
    field = MockField()
    name = "test_field"
    value = 42
    result = type('result', (), {})()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    assert not hasattr(result, name)
    assert invariant_errors == ["invariant_error"]

def test_check_and_set_attr_no_type_check():
    class MockField:
        type = None
        def invariant(self, value):
            return True, None
    class MockClass:
        __name__ = "MockClass"
    field = MockField()
    name = "test_field"
    value = "any_value"
    result = type('result', (), {})()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    assert getattr(result, name) == value
    assert invariant_errors == []

def test_check_and_set_attr_multiple_types_valid():
    class MockField:
        type = (int, str)
        def invariant(self, value):
            return True, None
    class MockClass:
        __name__ = "MockClass"
    field = MockField()
    name = "test_field"
    value = "a_string"
    result = type('result', (), {})()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    assert getattr(result, name) == value
    assert invariant_errors == []


# LLM-generated content at query #20
#--------------------------

def test___reduce___returns_correct_tuple():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 10, 'y': 20}

def test___reduce___handles_missing_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10)
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 10}

def test___reduce___with_no_fields():
    from pyrsistent import PClass
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert result[1][0] is TestClass
    assert result[1][1] == {}

def test___reduce___preserves_field_order():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    instance = TestClass(a=1, b=2, c=3)
    result = instance.__reduce__()
    data = result[1][1]
    assert list(data.keys()) == ['a', 'b', 'c']
    assert data['a'] == 1
    assert data['b'] == 2
    assert data['c'] == 3


# LLM-generated content at query #21
#--------------------------

def test___reduce___returns_correct_tuple():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 10, 'y': 20}

def test___reduce___with_missing_attribute():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10)
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 10}

def test___reduce___with_no_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=False)
        y = field(mandatory=False)
    instance = TestClass()
    result = instance.__reduce__()
    assert result[0] is _restore_pickle
    assert len(result[1]) == 2
    assert result[1][0] is TestClass
    assert result[1][1] == {}

def test___reduce___pickle_roundtrip():
    import pickle
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    original = TestClass(a=100, b=200)
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)
    assert restored == original
    assert restored is not original
    assert restored.a == 100
    assert restored.b == 200


# LLM-generated content at query #22
#--------------------------

def test_PClassMeta_new_sets_fields_and_invariants():
    class MockCheckedType:
        pass
    class MockBase:
        _pclass_fields = {'base_field': 'base_value'}
        __invariant__ = lambda self: (True, ())
    bases = (MockBase,)
    dct = {'field1': 'value1', '__invariant__': lambda self: (True, ())}
    PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert '_pclass_fields' in dct
    assert 'field1' not in dct
    assert dct['_pclass_fields'] == {'base_field': 'base_value'}
    assert '_pclass_invariants' in dct
    assert len(dct['_pclass_invariants']) == 2
    assert callable(dct['_pclass_invariants'][0])
    assert callable(dct['_pclass_invariants'][1])
    assert '__slots__' in dct
    assert '_pclass_frozen' in dct['__slots__']
    assert 'base_field' in dct['__slots__']

def test_PClassMeta_new_with_checked_type_base():
    class MockCheckedType:
        pass
    bases = (MockCheckedType,)
    dct = {}
    PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert '__slots__' in dct
    assert '__weakref__' in dct['__slots__']

def test_PClassMeta_new_without_checked_type_base():
    class MockBase:
        pass
    bases = (MockBase,)
    dct = {}
    PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert '__slots__' in dct
    assert '__weakref__' not in dct['__slots__']

def test_PClassMeta_new_handles_invalid_invariant():
    class MockBase:
        __invariant__ = 'not callable'
    bases = (MockBase,)
    dct = {}
    try:
        PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
        assert False
    except TypeError:
        pass

def test_PClassMeta_new_merges_invariants():
    class MockBase1:
        __invariant__ = lambda self: (True, ())
    class MockBase2:
        __invariant__ = lambda self: (False, ('error',))
    bases = (MockBase1, MockBase2)
    dct = {}
    PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert '_pclass_invariants' in dct
    invariants = dct['_pclass_invariants']
    assert len(invariants) == 2
    result1 = invariants[0]()
    assert result1 == (True, ())
    result2 = invariants[1]()
    assert result2 == (False, ('error',))

def test_PClassMeta_new_with_pfield():
    from pyrsistent._field_common import _PField
    class MockBase:
        pass
    bases = (MockBase,)
    pfield = _PField()
    dct = {'field1': pfield}
    PClassMeta.__new__(PClassMeta, 'TestClass', bases, dct)
    assert '_pclass_fields' in dct
    assert 'field1' not in dct
    assert dct['_pclass_fields']['field1'] is pfield
    assert 'field1' in dct['__slots__']


# LLM-generated content at query #23
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_invariant_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True, invariant=lambda x: (x > 0, 'error1'))
    try:
        TestClass(x=-1)
    except InvariantException as e:
        assert e.invariant_errors == ('error1',)
        assert e.missing_fields == ()
    else:
        assert False


# LLM-generated content at query #24
#--------------------------

def test___eq___returns_true_for_same_class_and_equal_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    result = instance1.__eq__(instance2)
    assert result is True

def test___eq___returns_false_for_same_class_and_different_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    result = instance1.__eq__(instance2)
    assert result is False

def test___eq___returns_not_implemented_for_different_class():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    result = instance1.__eq__(instance2)
    assert result is NotImplemented

def test___eq___returns_true_for_same_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    result = instance.__eq__(instance)
    assert result is True

def test___eq___handles_missing_fields_correctly():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    result = instance1.__eq__(instance2)
    assert result is True

def test___eq___returns_false_when_one_field_differs():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    instance1 = TestClass(a=1, b=2, c=3)
    instance2 = TestClass(a=1, b=2, c=4)
    result = instance1.__eq__(instance2)
    assert result is False


# LLM-generated content at query #25
#--------------------------

def test_is_pclass_false_for_non_pclass_bases():
    class Base:
        pass
    class TestClass(metaclass=PClassMeta):
        pass
    instance = TestClass()
    assert not hasattr(instance, '__weakref__')


# LLM-generated content at query #26
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true_when_conditions_met():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    class TestClass(PClass):
        x = field(type=dict, factory=lambda v, ignore_extra=False: v)
    instance = TestClass(x={})
    test_field = TestClass._pclass_fields['x']
    result = is_field_ignore_extra_complaint(PClass, test_field, True)
    assert result == True


# LLM-generated content at query #27
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, mandatory=True)
        y = field(type=int, initial=10)
    instance = TestClass(x=5)
    assert instance.x == 5
    assert instance.y == 10

def test___new___raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, mandatory=True)
    try:
        TestClass()
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___raises_on_invalid_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="string")
        assert False
    except Exception as e:
        assert "Invalid type" in str(e)

def test___new___raises_on_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x=1, y=2)
        assert False
    except Exception as e:
        assert "are not among the specified fields" in str(e)

def test___new___handles_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    instance = TestClass()
    assert instance.x == 42

def test___new___invariant_failure_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(value):
        return value > 0, "value must be positive"
    class TestClass(PClass):
        x = field(type=int, invariant=invariant_check)
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "value must be positive" in str(e)

def test___new___global_invariant_failure_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x + obj.y == 10, "sum must be 10"
    class TestClass(PClass):
        x = field(type=int)
        y = field(type=int)
        _pclass_invariants = [global_invariant]
    try:
        TestClass(x=3, y=8)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___with_factory_fields_and_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, factory=lambda v, ignore_extra=False: v * 2)
    instance = TestClass(x=5, ignore_extra=True)
    assert instance.x == 10

def test___new___sets_frozen_attribute():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    instance = TestClass(x=1)
    assert instance._pclass_frozen == True

def test___new___handles_factory_fields_parameter():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, factory=lambda v: v + 1)
        y = field(type=int)
    instance = TestClass(_factory_fields={'x'}, x=5, y=10)
    assert instance.x == 6
    assert instance.y == 10


# LLM-generated content at query #28
#--------------------------

def test_set_method_adds_missing_fields_from_pclass_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2
    assert instance.x == 1
    assert instance.y == 2


# LLM-generated content at query #29
#--------------------------

def test_check_type_raises_ptype_error_on_invalid_type():
    class MockField:
        type = (int,)
        def invariant(self, value):
            return (True, None)
    class MockClass:
        __name__ = "MockClass"
    field = MockField()
    name = "test_field"
    value = "not_an_int"
    result = type('result', (), {})()
    invariant_errors = []
    try:
        _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
        assert False
    except PTypeError as e:
        assert e.destination_cls == MockClass
        assert e.field_name == name
        assert e.expected_types == field.type
        assert e.actual_type == type(value)


# LLM-generated content at query #30
#--------------------------

def test_set_with_keyword_argument():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a.x == 1
    assert a2.x == 2

def test_set_with_positional_arguments():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set('x', 2)
    assert a.x == 1
    assert a2.x == 2

def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a is not a2

def test_set_preserves_other_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=10)
    a2 = a.set(x=2)
    assert a.x == 1
    assert a.y == 10
    assert a2.x == 2
    assert a2.y == 10

def test_set_with_multiple_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=10)
    a2 = a.set(x=2, y=20)
    assert a.x == 1
    assert a.y == 10
    assert a2.x == 2
    assert a2.y == 20

def test_set_with_factory_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(type=int)
        y = field(type=int)
    a = AClass(x=1, y=10)
    a2 = a.set(x=2)
    assert isinstance(a2.x, int)
    assert isinstance(a2.y, int)

def test_set_raises_attribute_error_for_unknown_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    try:
        a.set(z=2)
        assert False
    except AttributeError:
        pass

def test_set_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(mandatory=True)
        y = field()
    a = AClass(x=1)
    a2 = a.set(y=20)
    assert a2.x == 1
    assert a2.y == 20

def test_set_on_empty_pclass():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass()
    a2 = a.set(x=1)
    assert a2.x == 1

def test_set_with_initial_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(initial=5)
    a = AClass()
    a2 = a.set(x=10)
    assert a.x == 5
    assert a2.x == 10


# LLM-generated content at query #31
#--------------------------

def test_serialize_includes_all_fields_with_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    serialized = instance.serialize()
    assert 'x' in serialized
    assert 'y' in serialized
    assert serialized['x'] == 10
    assert serialized['y'] == 20

def test_serialize_excludes_missing_value_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    instance = TestClass(x=10)
    serialized = instance.serialize()
    assert 'x' in serialized
    assert 'y' not in serialized
    assert serialized['x'] == 10

def test_serialize_uses_custom_serializer():
    from pyrsistent import PClass, field
    def custom_serializer(value, format):
        return f"serialized_{value}"
    class TestClass(PClass):
        x = field(serializer=custom_serializer)
    instance = TestClass(x=5)
    serialized = instance.serialize()
    assert serialized['x'] == "serialized_5"

def test_serialize_with_format_passed_to_serializer():
    from pyrsistent import PClass, field
    def custom_serializer(value, format):
        return f"{format}:{value}"
    class TestClass(PClass):
        x = field(serializer=custom_serializer)
    instance = TestClass(x=100)
    serialized = instance.serialize(format='json')
    assert serialized['x'] == "json:100"

def test_serialize_empty_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        pass
    instance = TestClass()
    serialized = instance.serialize()
    assert serialized == {}


# LLM-generated content at query #32
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_invariant_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True, invariant=lambda x: (x > 0, 'error1'))
        y = field(mandatory=True)
    try:
        TestClass(x=-1)
    except InvariantException as e:
        assert e.error_codes == ('error1',)
        assert e.missing_fields == ('TestClass.y',)
        assert e.msg == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #33
#--------------------------

```python
def test_is_pclass_returns_true_for_pclass_bases():
    from pyrsistent._pclass import PClassMeta, _is_pclass
    class BasePClass(metaclass=PClassMeta):
        pass
    class DerivedPClass(BasePClass, metaclass=PClassMeta):
        pass
    bases = (BasePClass,)
    result = _is_pclass(bases)
    assert result == True


# LLM-generated content at query #34
#--------------------------

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
        assert False
    except Exception as e:
        assert "missing_fields" in str(e)

def test_pclass_constructor_with_extra_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, z=3)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

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
        x = field(initial=lambda: 100)
    instance = TestClass()
    assert instance.x == 100

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=5, y=10)
    assert instance.x == 5
    assert instance.y == 10

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1, z=3, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')

def test_pclass_constructor_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(value):
        if value < 0:
            return (False, "value must be non-negative")
        return (True, "")
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    try:
        TestClass(x=-5)
        assert False
    except InvariantException as e:
        assert "invariant_errors" in str(e)

def test_pclass_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException, invariant
    class TestClass(PClass):
        x = field()
        y = field()
        @invariant(lambda self: self.x + self.y == 10)
        def sum_invariant(self):
            return self.x + self.y == 10
    try:
        TestClass(x=3, y=4)
        assert False
    except InvariantException as e:
        assert "invariant_errors" in str(e)

def test_pclass_constructor_creates_frozen_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


# LLM-generated content at query #35
#--------------------------

def test_eq_returns_true_for_same_class_and_equal_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    result = instance1 == instance2
    assert result is True


# LLM-generated content at query #36
#--------------------------

def test_eq_returns_true_for_same_class_and_equal_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2


# LLM-generated content at query #37
#--------------------------

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
        assert False
    except Exception as e:
        assert "missing_fields" in str(e)

def test_pclass_constructor_with_extra_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, z=3)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test_pclass_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1, z=3, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')

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
        x = field(initial=lambda: 100)
    instance = TestClass()
    assert instance.x == 100

def test_pclass_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2, _factory_fields={'x'})
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def check_positive(value):
        return value > 0, "value must be positive"
    class TestClass(PClass):
        x = field(invariant=check_positive)
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "value must be positive" in str(e)

def test_pclass_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException, invariant
    def global_check(obj):
        return obj.x + obj.y > 0, "sum must be positive"
    class TestClass(PClass):
        __invariant__ = invariant(global_check)
        x = field()
        y = field()
    try:
        TestClass(x=-5, y=2)
        assert False
    except InvariantException as e:
        assert "sum must be positive" in str(e)

def test_pclass_constructor_creates_frozen_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


