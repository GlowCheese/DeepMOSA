####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_set_updates_data_when_value_different():
    original = object()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set('a', 2)
    assert evolver._pclass_evolver_data == {'a': 2}
    assert result is evolver

def test_set_marks_data_dirty_when_value_different():
    original = object()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 2)
    assert evolver._pclass_evolver_data_is_dirty is True

def test_set_adds_key_to_factory_fields_when_value_different():
    original = object()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 2)
    assert evolver._factory_fields == {'a'}

def test_set_does_not_update_data_when_value_same():
    original = object()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 1)
    assert evolver._pclass_evolver_data == {'a': 1}
    assert evolver._pclass_evolver_data_is_dirty is False

def test_set_does_not_add_key_to_factory_fields_when_value_same():
    original = object()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 1)
    assert evolver._factory_fields == set()

def test_set_adds_new_key():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('b', 3)
    assert evolver._pclass_evolver_data == {'b': 3}
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._factory_fields == {'b'}

def test_set_handles_missing_value_sentinel():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('c', _MISSING_VALUE)
    assert evolver._pclass_evolver_data == {'c': _MISSING_VALUE}
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._factory_fields == {'c'}


# LLM-generated content at query #2
#--------------------------

def test___new___sets_pclass_fields():
    class Base(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
        _pclass_fields = {}
    class Derived(Base):
        field = _PField(type=int, invariant=lambda x: (x > 0, "positive"))
    assert Derived._pclass_fields == {'field': _PField(type=int, invariant=lambda x: (x > 0, "positive"))}

def test___new___inherits_invariants():
    def base_invariant(self):
        return True, ()
    class Base(metaclass=PClassMeta):
        __invariant__ = base_invariant
        _pclass_fields = {}
    class Derived(Base):
        __invariant__ = lambda self: (True, ())
        _pclass_fields = {}
    assert len(Derived._pclass_invariants) == 2
    assert Derived._pclass_invariants[0] is base_invariant

def test___new___wraps_invariants():
    def invariant_returning_tuple(self):
        return (False, "error1"), (True, ())
    class TestClass(metaclass=PClassMeta):
        __invariant__ = invariant_returning_tuple
        _pclass_fields = {}
    result = TestClass._pclass_invariants[0](TestClass())
    assert result == (False, ("error1",))

def test___new___sets_slots():
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
        field = _PField(type=int)
    assert TestClass.__slots__ == ('_pclass_frozen', 'field')

def test___new___adds_weakref_for_top_level():
    class TopLevel(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
        _pclass_fields = {}
    assert '__weakref__' in TopLevel.__slots__

def test___new___no_weakref_for_inherited():
    class Base(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
        _pclass_fields = {}
    class Derived(Base):
        __invariant__ = lambda self: (True, ())
        _pclass_fields = {}
    assert '__weakref__' not in Derived.__slots__

def test___new___raises_on_non_callable_invariant():
    try:
        class Invalid(metaclass=PClassMeta):
            __invariant__ = "not callable"
            _pclass_fields = {}
        assert False
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test___new___merges_fields_from_bases():
    class Base1(metaclass=PClassMeta):
        field1 = _PField(type=int)
        __invariant__ = lambda self: (True, ())
    class Base2(metaclass=PClassMeta):
        field2 = _PField(type=str)
        __invariant__ = lambda self: (True, ())
    class Combined(Base1, Base2):
        field3 = _PField(type=float)
        __invariant__ = lambda self: (True, ())
    expected_keys = {'field1', 'field2', 'field3'}
    assert set(Combined._pclass_fields.keys()) == expected_keys


# LLM-generated content at query #3
#--------------------------

def test___hash___returns_same_hash_for_equal_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y="hello")
    instance2 = TestClass(x=10, y="hello")
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 == hash2

def test___hash___returns_different_hash_for_different_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y="hello")
    instance2 = TestClass(x=20, y="world")
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 != hash2

def test___hash___works_with_missing_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10)
    hash_value = instance.__hash__()
    assert isinstance(hash_value, int)

def test___hash___consistent_across_multiple_calls():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y="hello")
    hash1 = instance.__hash__()
    hash2 = instance.__hash__()
    assert hash1 == hash2

def test___hash___different_for_different_field_order():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    instance1 = TestClass(a=1, b=2)
    instance2 = TestClass(a=2, b=1)
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 != hash2

def test___hash___handles_none_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=None, y=None)
    hash_value = instance.__hash__()
    assert isinstance(hash_value, int)

def test___hash___handles_complex_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=[1, 2, 3], y={"key": "value"})
    hash_value = instance.__hash__()
    assert isinstance(hash_value, int)


# LLM-generated content at query #4
#--------------------------

def test_is_pclass_false_for_non_pclass_bases():
    class NonPClassBase:
        pass
    class TestClass(metaclass=PClassMeta):
        pass
    test_instance = TestClass()
    assert not hasattr(test_instance, '__weakref__')


# LLM-generated content at query #5
#--------------------------

def test_is_pclass_false_for_non_pclass_bases():
    class NonPClassBase:
        pass
    class TestClass(metaclass=PClassMeta):
        __slots__ = ()
    bases = (NonPClassBase,)
    result = _is_pclass(bases)
    assert result is False


# LLM-generated content at query #6
#--------------------------

def test_persistent_returns_original_if_not_dirty():
    original = object()
    evolver = _PClassEvolver(original, {})
    result = evolver.persistent()
    assert result is original

def test_persistent_returns_new_instance_if_dirty():
    class MockPClass:
        def __init__(self, _factory_fields=None, **kwargs):
            self.kwargs = kwargs
            self._factory_fields = _factory_fields
    original = MockPClass()
    evolver = _PClassEvolver(original, {'a': 1})
    evolver.set('b', 2)
    result = evolver.persistent()
    assert result is not original
    assert result.kwargs == {'a': 1, 'b': 2}
    assert result._factory_fields == {'b'}

def test_persistent_uses_original_class():
    class MockPClass:
        def __init__(self, _factory_fields=None, **kwargs):
            self.kwargs = kwargs
            self._factory_fields = _factory_fields
    original = MockPClass()
    evolver = _PClassEvolver(original, {})
    evolver.set('x', 10)
    result = evolver.persistent()
    assert isinstance(result, MockPClass)

def test_persistent_includes_all_data():
    class MockPClass:
        def __init__(self, _factory_fields=None, **kwargs):
            self.kwargs = kwargs
            self._factory_fields = _factory_fields
    original = MockPClass()
    evolver = _PClassEvolver(original, {'initial': 5})
    evolver.set('added', 6)
    evolver.remove('initial')
    result = evolver.persistent()
    assert 'initial' not in result.kwargs
    assert result.kwargs == {'added': 6}
    assert result._factory_fields == {'added'}

def test_persistent_after_multiple_modifications():
    class MockPClass:
        def __init__(self, _factory_fields=None, **kwargs):
            self.kwargs = kwargs
            self._factory_fields = _factory_fields
    original = MockPClass()
    evolver = _PClassEvolver(original, {'a': 1, 'b': 2})
    evolver.set('c', 3)
    evolver.remove('b')
    evolver.set('a', 10)
    result = evolver.persistent()
    assert result.kwargs == {'a': 10, 'c': 3}
    assert result._factory_fields == {'c', 'a'}


# LLM-generated content at query #7
#--------------------------

def test_is_pclass_true():
    bases = (CheckedType,)
    result = _is_pclass(bases)
    assert result == True

def test_is_pclass_false_multiple_bases():
    bases = (CheckedType, object)
    result = _is_pclass(bases)
    assert result == False

def test_is_pclass_false_different_base():
    bases = (object,)
    result = _is_pclass(bases)
    assert result == False

def test_is_pclass_false_empty():
    bases = ()
    result = _is_pclass(bases)
    assert result == False


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

def test___repr___returns_correct_string_for_simple_pclass():
    from pyrsistent import PClass, field
    class SimpleClass(PClass):
        x = field()
        y = field()
    instance = SimpleClass(x=10, y="hello")
    result = repr(instance)
    expected = "SimpleClass(x=10, y='hello')"
    assert result == expected

def test___repr___handles_empty_pclass():
    from pyrsistent import PClass, field
    class EmptyClass(PClass):
        pass
    instance = EmptyClass()
    result = repr(instance)
    expected = "EmptyClass()"
    assert result == expected

def test___repr___handles_pclass_with_mandatory_field_missing_but_with_initial():
    from pyrsistent import PClass, field
    class ClassWithInitial(PClass):
        x = field(initial=5)
        y = field()
    instance = ClassWithInitial(y=20)
    result = repr(instance)
    expected = "ClassWithInitial(x=5, y=20)"
    assert result == expected

def test___repr___handles_pclass_with_nested_structures():
    from pyrsistent import PClass, field, pvector
    class NestedClass(PClass):
        items = field()
        name = field()
    instance = NestedClass(items=pvector([1, 2, 3]), name="test")
    result = repr(instance)
    expected = "NestedClass(items=pvector([1, 2, 3]), name='test')"
    assert result == expected

def test___repr___handles_pclass_with_boolean_and_none_values():
    from pyrsistent import PClass, field
    class MixedClass(PClass):
        flag = field()
        value = field()
    instance = MixedClass(flag=True, value=None)
    result = repr(instance)
    expected = "MixedClass(flag=True, value=None)"
    assert result == expected

def test___repr___handles_pclass_with_custom_repr_in_field_values():
    from pyrsistent import PClass, field
    class CustomReprClass:
        def __repr__(self):
            return "Custom()"
    class ContainerClass(PClass):
        obj = field()
    custom_obj = CustomReprClass()
    instance = ContainerClass(obj=custom_obj)
    result = repr(instance)
    expected = "ContainerClass(obj=Custom())"
    assert result == expected

def test___repr___handles_pclass_with_multiple_fields_ordered_alphabetically():
    from pyrsistent import PClass, field
    class MultiFieldClass(PClass):
        z = field()
        a = field()
        m = field()
    instance = MultiFieldClass(z=3, a=1, m=2)
    result = repr(instance)
    expected = "MultiFieldClass(a=1, m=2, z=3)"
    assert result == expected


# LLM-generated content at query #10
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test___new___raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=2)
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___uses_initial_value_for_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test___new___raises_on_extra_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, z=3)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test___new___checks_type_and_raises_on_invalid():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="string")
        assert False
    except Exception as e:
        assert "Invalid type" in str(e)

def test___new___checks_field_invariant_and_raises():
    from pyrsistent import PClass, field, InvariantException
    def positive(value):
        return value > 0, "not_positive"
    class TestClass(PClass):
        x = field(invariant=positive)
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test___new___checks_global_invariant_and_raises():
    from pyrsistent import PClass, field, InvariantException
    def sum_positive(instance):
        return instance.x + instance.y > 0, "sum_not_positive"
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [sum_positive]
    try:
        TestClass(x=-5, y=2)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___handles_factory_fields_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=dict, factory=lambda v, ignore_extra=False: v)
    instance = TestClass(x={"a": 1}, ignore_extra=True)
    assert instance.x == {"a": 1}

def test___new___freezes_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

def test___new___creates_instance_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    instance = TestClass()
    assert instance.x == 42


# LLM-generated content at query #11
#--------------------------

def test_hash_returns_consistent_value_for_same_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    hash1 = hash(instance)
    hash2 = hash(instance)
    assert hash1 == hash2

def test_hash_equal_for_identical_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=20)
    assert hash(instance1) == hash(instance2)

def test_hash_different_for_different_field_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=30, y=40)
    assert hash(instance1) != hash(instance2)

def test_hash_different_for_different_field_names():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        a = field()
    class TestClass2(PClass):
        b = field()
    instance1 = TestClass1(a=10)
    instance2 = TestClass2(b=10)
    assert hash(instance1) != hash(instance2)

def test_hash_uses_all_fields_even_if_some_missing():
    from pyrsistent import PClass, field, _MISSING_VALUE
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    instance1 = TestClass(x=10)
    instance2 = TestClass(x=10, y=_MISSING_VALUE)
    assert hash(instance1) == hash(instance2)

def test_hash_handles_nested_pclass():
    from pyrsistent import PClass, field
    class Inner(PClass):
        a = field()
    class Outer(PClass):
        inner = field()
    inner1 = Inner(a=5)
    outer1 = Outer(inner=inner1)
    inner2 = Inner(a=5)
    outer2 = Outer(inner=inner2)
    assert hash(outer1) == hash(outer2)

def test_hash_consistent_with_equality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=20)
    assert instance1 == instance2
    assert hash(instance1) == hash(instance2)

def test_hash_different_when_equality_false():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=30)
    assert instance1 != instance2
    assert hash(instance1) != hash(instance2)

def test_hash_works_with_mandatory_fields_only():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(mandatory=True)
    instance = TestClass(x=1, y=2)
    assert isinstance(hash(instance), int)

def test_hash_works_with_optional_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(initial=0)
    instance = TestClass(x=1)
    assert isinstance(hash(instance), int)


# LLM-generated content at query #12
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true_when_conditions_met():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    class TestClass(PClass):
        x = field(type=dict, factory=lambda v, ignore_extra=False: v)
    field_instance = TestClass._pclass_fields['x']
    result = is_field_ignore_extra_complaint(PClass, field_instance, True)
    assert result == True


# LLM-generated content at query #13
#--------------------------

def test_hash_returns_same_value_for_equal_objects():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=10, y=20)
    obj2 = TestClass(x=10, y=20)
    assert hash(obj1) == hash(obj2)

def test_hash_returns_different_value_for_different_objects():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=10, y=20)
    obj2 = TestClass(x=30, y=40)
    assert hash(obj1) != hash(obj2)

def test_hash_uses_all_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    obj = TestClass(a=1, b=2)
    expected_hash = hash(tuple([('a', 1), ('b', 2)]))
    assert hash(obj) == expected_hash

def test_hash_handles_missing_values():
    from pyrsistent import PClass, field
    from pyrsistent import _MISSING_VALUE
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    obj = TestClass(x=5)
    expected_hash = hash(tuple([('x', 5), ('y', _MISSING_VALUE)]))
    assert hash(obj) == expected_hash

def test_hash_is_consistent_across_multiple_calls():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    obj = TestClass(x=100)
    first_hash = hash(obj)
    second_hash = hash(obj)
    assert first_hash == second_hash

def test_hash_works_with_nested_pclass():
    from pyrsistent import PClass, field
    class Inner(PClass):
        val = field()
    class Outer(PClass):
        inner = field()
    inner_obj = Inner(val=42)
    outer_obj = Outer(inner=inner_obj)
    outer_obj2 = Outer(inner=inner_obj)
    assert hash(outer_obj) == hash(outer_obj2)

def test_hash_differs_when_field_order_differs():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    obj1 = TestClass(a=1, b=2)
    obj2 = TestClass(b=2, a=1)
    assert hash(obj1) == hash(obj2)

def test_hash_uses_field_names_and_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        name = field()
        value = field()
    obj = TestClass(name='test', value=123)
    expected = hash(tuple([('name', 'test'), ('value', 123)]))
    assert hash(obj) == expected


# LLM-generated content at query #14
#--------------------------

def test_repr_returns_correct_format():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y="hello")
    expected = "TestClass(x=10, y='hello')"
    actual = repr(instance)
    assert actual == expected

def test_repr_with_no_fields():
    from pyrsistent import PClass, field
    class EmptyClass(PClass):
        pass
    instance = EmptyClass()
    expected = "EmptyClass()"
    actual = repr(instance)
    assert actual == expected

def test_repr_with_one_field():
    from pyrsistent import PClass, field
    class SingleFieldClass(PClass):
        a = field()
    instance = SingleFieldClass(a=42)
    expected = "SingleFieldClass(a=42)"
    actual = repr(instance)
    assert actual == expected

def test_repr_with_nested_values():
    from pyrsistent import PClass, field, pvector
    class NestedClass(PClass):
        items = field()
    instance = NestedClass(items=pvector([1, 2, 3]))
    expected = "NestedClass(items=pvector([1, 2, 3]))"
    actual = repr(instance)
    assert actual == expected

def test_repr_with_special_characters_in_field_value():
    from pyrsistent import PClass, field
    class SpecialClass(PClass):
        text = field()
    instance = SpecialClass(text="line1\nline2")
    expected = "SpecialClass(text='line1\\nline2')"
    actual = repr(instance)
    assert actual == expected

def test_repr_after_set_operation():
    from pyrsistent import PClass, field
    class UpdateClass(PClass):
        x = field()
        y = field()
    instance = UpdateClass(x=1, y=2)
    new_instance = instance.set(x=100)
    expected = "UpdateClass(x=100, y=2)"
    actual = repr(new_instance)
    assert actual == expected

def test_repr_with_boolean_and_none_values():
    from pyrsistent import PClass, field
    class MixedClass(PClass):
        flag = field()
        empty = field()
    instance = MixedClass(flag=True, empty=None)
    expected = "MixedClass(flag=True, empty=None)"
    actual = repr(instance)
    assert actual == expected

def test_repr_uses_to_dict_for_representation():
    from pyrsistent import PClass, field
    class ToDictClass(PClass):
        a = field()
        b = field()
    instance = ToDictClass(a=5, b=6)
    to_dict_result = instance._to_dict()
    expected_format = "{0}({1})".format(instance.__class__.__name__, ', '.join('{0}={1}'.format(k, repr(v)) for k, v in to_dict_result.items()))
    actual = repr(instance)
    assert actual == expected_format


# LLM-generated content at query #15
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test___new___raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=2)
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___uses_initial_value_for_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test___new___raises_on_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, z=3)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test___new___handles_factory_with_ignore_extra():
    from pyrsistent import PClass, field
    def factory(value, ignore_extra=False):
        return value * 2
    class TestClass(PClass):
        x = field(factory=factory, type=(int,))
    instance = TestClass(x=5, ignore_extra=True)
    assert instance.x == 10

def test___new___invokes_field_invariant():
    from pyrsistent import PClass, field
    def invariant(value):
        return value > 0, "value must be positive"
    class TestClass(PClass):
        x = field(invariant=invariant)
    try:
        TestClass(x=-1)
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___checks_global_invariants():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x + obj.y > 0, "sum must be positive"
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [global_invariant]
    try:
        TestClass(x=-5, y=2)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___sets_frozen_flag():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert instance._pclass_frozen == True

def test___new___handles_callable_initial():
    from pyrsistent import PClass, field
    def initial_func():
        return 42
    class TestClass(PClass):
        x = field(initial=initial_func)
    instance = TestClass()
    assert instance.x == 42

def test___new___propagates_factory_fields():
    from pyrsistent import PClass, field
    def factory(value):
        return value + 100
    class TestClass(PClass):
        x = field(factory=factory)
        y = field()
    instance = TestClass(x=5, y=10)
    assert instance.x == 105
    assert instance.y == 10


# LLM-generated content at query #16
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test___new___raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=2)
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___uses_initial_for_missing_non_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test___new___raises_on_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, z=3)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test___new___handles_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
    instance = TestClass(x=5)
    assert instance.x == 10

def test___new___handles_ignore_extra_with_factory():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v, ignore_extra=False: v if not ignore_extra else v * 2)
    instance = TestClass(x=5, ignore_extra=True)
    assert instance.x == 10

def test___new___invokes_global_invariants():
    from pyrsistent import PClass, field, InvariantException
    def check_positive(obj):
        return obj.x > 0, "not positive"
    class TestClass(PClass):
        x = field()
        __invariants__ = [check_positive]
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___sets_frozen_flag():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert instance._pclass_frozen == True

def test___new___handles_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    instance = TestClass()
    assert instance.x == 42

def test___new___propagates_invariant_errors():
    from pyrsistent import PClass, field, InvariantException
    def check_even(value):
        return value % 2 == 0, "not even"
    class TestClass(PClass):
        x = field(invariant=check_even)
    try:
        TestClass(x=3)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)


# LLM-generated content at query #17
#--------------------------

def test_eq_returns_true_for_same_class_and_equal_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2


# LLM-generated content at query #18
#--------------------------

def test_remove_existing_item():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict.copy())
    result = evolver.remove('a')
    assert result is evolver
    assert 'a' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'a' not in evolver._factory_fields

def test_remove_non_existing_item():
    original = type('Original', (), {})()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict.copy())
    try:
        evolver.remove('b')
        assert False
    except AttributeError as e:
        assert str(e) == 'b'

def test_remove_item_clears_factory_fields():
    original = type('Original', (), {})()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict.copy())
    evolver.set('a', 2)
    evolver.remove('a')
    assert 'a' not in evolver._factory_fields

def test_remove_preserves_other_items():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2, 'c': 3}
    evolver = _PClassEvolver(original, initial_dict.copy())
    evolver.remove('b')
    assert evolver._pclass_evolver_data == {'a': 1, 'c': 3}

def test_remove_does_not_mark_dirty_if_item_missing():
    original = type('Original', (), {})()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict.copy())
    evolver._pclass_evolver_data_is_dirty = False
    try:
        evolver.remove('b')
        assert False
    except AttributeError:
        pass
    assert evolver._pclass_evolver_data_is_dirty is False


# LLM-generated content at query #19
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
    def custom_serializer(value, format):
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
    def format_serializer(value, format):
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
        z = field(initial=0)
    instance = TestClass(x=1)
    result = instance.serialize()
    expected = {'x': 1, 'z': 0}
    assert result == expected

def test_serialize_empty_pclass():
    from pyrsistent import PClass, field
    class EmptyClass(PClass):
        pass
    instance = EmptyClass()
    result = instance.serialize()
    expected = {}
    assert result == expected


# LLM-generated content at query #20
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

def test___reduce___with_missing_attributes():
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
        x = field()
        y = field()
    instance = TestClass()
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {}

def test___reduce___pickle_roundtrip():
    import pickle
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    original = TestClass(x=5, y=15)
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)
    assert restored == original
    assert restored.x == 5
    assert restored.y == 15

def test___reduce___with_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    instance = TestClass(x=100)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 100}

def test___reduce___with_initial_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=99)
        y = field()
    instance = TestClass()
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 99}

def test___reduce___with_factory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
        y = field()
    instance = TestClass(x=10, y=30)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 20, 'y': 30}

def test___reduce___after_set():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    modified = instance.set(x=50)
    result = modified.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 50, 'y': 2}

def test___reduce___with_serializer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(serializer=lambda v, f: str(v))
        y = field()
    instance = TestClass(x=42, y=84)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 42, 'y': 84}

def test___reduce___with_invariant():
    from pyrsistent import PClass, field, InvariantException
    def positive_invariant(value):
        return value > 0, 'value must be positive'
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
        y = field()
    instance = TestClass(x=5, y=10)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 2
    assert result[1][0] == TestClass
    assert isinstance(result[1][1], dict)
    assert result[1][1] == {'x': 5, 'y': 10}


# LLM-generated content at query #21
#--------------------------

def test_remove_when_item_in_data():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.remove('a')
    assert result is evolver
    assert 'a' not in evolver._pclass_evolver_data
    assert 'a' not in evolver._factory_fields
    assert evolver._pclass_evolver_data_is_dirty is True


# LLM-generated content at query #22
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
    instance = TestClass()
    assert instance.x == 100

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
            return (('value must be non-negative',),)
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
            return (('sum must be 100',),)
        return ()
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [global_invariant]
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


# LLM-generated content at query #23
#--------------------------

def test___reduce___returns_correct_tuple_for_pickling():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 10, 'y': 20}

def test___reduce___handles_missing_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10)
    result = instance.__reduce__()
    assert result[1][1] == {'x': 10}

def test___reduce___works_with_no_fields():
    from pyrsistent import PClass
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = instance.__reduce__()
    assert result[1][1] == {}

def test___reduce___preserves_field_order():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    instance = TestClass(a=1, b=2, c=3)
    result = instance.__reduce__()
    assert list(result[1][1].keys()) == ['a', 'b', 'c']
    assert list(result[1][1].values()) == [1, 2, 3]


# LLM-generated content at query #24
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_invariant_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True, invariant=lambda v: (v > 0, 'positive'))
    try:
        TestClass(x=-1)
    except InvariantException as e:
        assert e.error_codes == ('positive',)
        assert e.missing_fields == ()
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #25
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
    instance = TestClass(x=5, y="test")
    result = instance.serialize()
    expected = {'x': 10, 'y': 'TEST'}
    assert result == expected

def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    def custom_serializer(value, format):
        if format == 'double':
            return value * 2
        return value
    class TestClass(PClass):
        x = field(serializer=custom_serializer)
    instance = TestClass(x=7)
    result = instance.serialize(format='double')
    expected = {'x': 14}
    assert result == expected

def test_serialize_missing_field_with_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=100)
        y = field()
    instance = TestClass(y=50)
    result = instance.serialize()
    expected = {'x': 100, 'y': 50}
    assert result == expected

def test_serialize_empty_pclass():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = instance.serialize()
    expected = {}
    assert result == expected


# LLM-generated content at query #26
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
    instance = TestClass()
    assert instance.x == 100

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

def test_pclass_constructor_with_ignore_extra_false_and_extra_field():
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
    def invariant_check(value):
        if value < 0:
            return (f"Value must be non-negative, got {value}",)
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
        __invariants__ = [invariant(lambda c: c.x + c.y > 0, "Sum must be positive")]
    try:
        TestClass(x=-10, y=5)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_creates_frozen_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    try:
        instance.x = 20
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)


# LLM-generated content at query #27
#--------------------------

def test_persistent_returns_original_when_no_changes():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.persistent()
    assert result is original

def test_persistent_returns_new_instance_when_data_is_dirty():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 3)
    result = evolver.persistent()
    assert result is not original
    assert result._factory_fields == {'a'}
    assert result.a == 3
    assert result.b == 2

def test_persistent_returns_new_instance_after_setitem():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver['a'] = 3
    result = evolver.persistent()
    assert result is not original
    assert result._factory_fields == {'a'}
    assert result.a == 3
    assert result.b == 2

def test_persistent_returns_new_instance_after_remove():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.remove('a')
    result = evolver.persistent()
    assert result is not original
    assert result._factory_fields == set()
    assert not hasattr(result, 'a')
    assert result.b == 2

def test_persistent_returns_new_instance_after_delitem():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    del evolver['a']
    result = evolver.persistent()
    assert result is not original
    assert result._factory_fields == set()
    assert not hasattr(result, 'a')
    assert result.b == 2

def test_persistent_returns_new_instance_after_setattr():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.c = 3
    result = evolver.persistent()
    assert result is not original
    assert result._factory_fields == {'c'}
    assert result.a == 1
    assert result.b == 2
    assert result.c == 3

def test_persistent_returns_new_instance_with_multiple_changes():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 10)
    evolver.set('b', 20)
    evolver.set('c', 30)
    result = evolver.persistent()
    assert result is not original
    assert result._factory_fields == {'a', 'b', 'c'}
    assert result.a == 10
    assert result.b == 20
    assert result.c == 30

def test_persistent_returns_original_when_set_same_value():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 1)
    result = evolver.persistent()
    assert result is original

def test_persistent_returns_new_instance_after_remove_and_set():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.remove('a')
    evolver.set('b', 3)
    result = evolver.persistent()
    assert result is not original
    assert result._factory_fields == {'b'}
    assert not hasattr(result, 'a')
    assert result.b == 3


# LLM-generated content at query #28
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
        assert str(e) == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #29
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True, invariant=lambda x: (x > 0, 'error1'))
    try:
        TestClass(x=-1)
    except InvariantException as e:
        assert e.error_codes == ('error1',)
        assert e.missing_fields == ()
        assert e.msg == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"

def test_missing_fields_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
    except InvariantException as e:
        assert e.error_codes == ()
        assert e.missing_fields == ('TestClass.x',)
        assert e.msg == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"

def test_invariant_errors_and_missing_fields_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True, invariant=lambda x: (x > 0, 'error1'))
        y = field(mandatory=True)
    try:
        TestClass(x=-1)
    except InvariantException as e:
        assert 'error1' in e.error_codes
        assert 'TestClass.y' in e.missing_fields
        assert e.msg == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #30
#--------------------------

def test___eq___returns_true_for_same_class_and_equal_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=1, y=2)
    result = obj1.__eq__(obj2)
    assert result is True

def test___eq___returns_false_for_same_class_and_different_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(x=3, y=4)
    result = obj1.__eq__(obj2)
    assert result is False

def test___eq___returns_not_implemented_for_different_class():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    obj1 = TestClass1(x=1)
    obj2 = TestClass2(x=1)
    result = obj1.__eq__(obj2)
    assert result is NotImplemented

def test___eq___returns_not_implemented_for_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    obj1 = TestClass(x=1)
    obj2 = object()
    result = obj1.__eq__(obj2)
    assert result is NotImplemented

def test___eq___handles_missing_attributes_correctly():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1)
    obj2 = TestClass(x=1)
    result = obj1.__eq__(obj2)
    assert result is True

def test___eq___returns_false_when_one_attribute_differs():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    obj1 = TestClass(x=1, y=2, z=3)
    obj2 = TestClass(x=1, y=2, z=99)
    result = obj1.__eq__(obj2)
    assert result is False


# LLM-generated content at query #31
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
    unpickled = pickle.loads(pickled)
    assert unpickled == original
    assert unpickled.a == 100
    assert unpickled.b == 200


# LLM-generated content at query #32
#--------------------------

def test_PClassMeta_new_single_inheritance():
    class CheckedType:
        pass
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    result = TestClass()
    assert hasattr(result, '_pclass_fields')
    assert hasattr(result, '_pclass_invariants')
    assert '_pclass_frozen' in result.__slots__

def test_PClassMeta_new_with_fields():
    class CheckedType:
        pass
    class TestClass(metaclass=PClassMeta):
        field = _PField()
    result = TestClass()
    assert 'field' in result._pclass_fields
    assert 'field' not in result.__dict__

def test_PClassMeta_new_inherits_invariants():
    class CheckedType:
        pass
    class Base(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class Derived(Base, metaclass=PClassMeta):
        pass
    result = Derived()
    assert len(result._pclass_invariants) == 1

def test_PClassMeta_new_multiple_invariants():
    class CheckedType:
        pass
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
        __invariant__ = lambda self: (False, ('error',))
    result = TestClass()
    assert len(result._pclass_invariants) == 2

def test_PClassMeta_new_invalid_invariant():
    class CheckedType:
        pass
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = 'not callable'
        result = False
    except TypeError:
        result = True
    assert result

def test_PClassMeta_new_slots_include_weakref():
    class CheckedType:
        pass
    class TestClass(CheckedType, metaclass=PClassMeta):
        pass
    assert '__weakref__' in TestClass.__slots__

def test_PClassMeta_new_slots_exclude_weakref():
    class CheckedType:
        pass
    class Base(metaclass=PClassMeta):
        pass
    class Derived(Base, metaclass=PClassMeta):
        pass
    assert '__weakref__' not in Derived.__slots__


# LLM-generated content at query #33
#--------------------------

def test_is_pclass_false_for_non_pclass_bases():
    class NonPClassBase:
        pass
    class TestClass(metaclass=PClassMeta):
        pass
    class DerivedClass(TestClass, NonPClassBase, metaclass=PClassMeta):
        pass
    assert '__weakref__' not in DerivedClass.__slots__


# LLM-generated content at query #34
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
    assert 'a' not in evolver._factory_fields

def test_remove_non_existing_item():
    original = object()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    try:
        evolver.remove('b')
        assert False
    except AttributeError as e:
        assert str(e) == 'b'

def test_remove_item_clears_factory_fields():
    original = object()
    initial_dict = {'x': 10}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('x', 20)
    result = evolver.remove('x')
    assert result is evolver
    assert 'x' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'x' not in evolver._factory_fields

def test_remove_item_after_set():
    original = object()
    initial_dict = {'key': 'old'}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('key', 'new')
    result = evolver.remove('key')
    assert result is evolver
    assert 'key' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'key' not in evolver._factory_fields

def test_remove_with_delitem():
    original = object()
    initial_dict = {'item': 42}
    evolver = _PClassEvolver(original, initial_dict)
    del evolver['item']
    assert 'item' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'item' not in evolver._factory_fields


# LLM-generated content at query #2
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
    except PTypeError:
        pass

def test___new___applies_initial_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
    instance = TestClass()
    assert instance.x == 10

def test___new___raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test___new___raises_on_extra_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, y=2)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test___new___handles_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
    instance = TestClass(x=5)
    assert instance.x == 10

def test___new___checks_field_invariant():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(invariant=lambda x: (x > 0, "positive"))
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test___new___checks_global_invariant():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field()
        y = field()
        @staticmethod
        def _invariant(obj):
            return (obj.x + obj.y > 0, "sum_positive")
    try:
        TestClass(x=-5, y=2)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___sets_frozen_flag():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert instance._pclass_frozen == True

def test___new___with_ignore_extra_compliant_factory():
    from pyrsistent import PClass, field
    def factory_with_ignore_extra(value, ignore_extra=False):
        return value * 2
    class TestClass(PClass):
        x = field(factory=factory_with_ignore_extra)
    instance = TestClass(x=3, ignore_extra=True)
    assert instance.x == 6

def test___new___with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    instance = TestClass()
    assert instance.x == 42

def test___new___with_non_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=100)
    instance = TestClass()
    assert instance.x == 100

def test___new___with_factory_fields_parameter():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v + 1)
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=5, y=10)
    assert instance.x == 6
    assert instance.y == 10

def test___new___with_ignore_extra_parameter():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1, ignore_extra=True)
    assert instance.x == 1

def test___new___with_ignore_extra_and_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1, y=2, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')


# LLM-generated content at query #3
#--------------------------

def test___hash___returns_same_hash_for_equal_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y="hello")
    instance2 = TestClass(x=10, y="hello")
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 == hash2

def test___hash___returns_different_hash_for_different_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y="hello")
    instance2 = TestClass(x=20, y="world")
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 != hash2

def test___hash___works_with_missing_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    instance1 = TestClass(x=10)
    instance2 = TestClass(x=10)
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 == hash2

def test___hash___consistent_across_multiple_calls():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    instance = TestClass(a=5, b=3.14)
    hash1 = instance.__hash__()
    hash2 = instance.__hash__()
    assert hash1 == hash2

def test___hash___handles_nested_pclass():
    from pyrsistent import PClass, field
    class Inner(PClass):
        val = field()
    class Outer(PClass):
        inner = field()
    inner1 = Inner(val=42)
    outer1 = Outer(inner=inner1)
    inner2 = Inner(val=42)
    outer2 = Outer(inner=inner2)
    hash1 = outer1.__hash__()
    hash2 = outer2.__hash__()
    assert hash1 == hash2

def test___hash___different_for_different_field_order():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=2, y=1)
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 != hash2

def test___hash___works_with_boolean_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        flag = field()
    instance1 = TestClass(flag=True)
    instance2 = TestClass(flag=False)
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 != hash2

def test___hash___works_with_none_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        item = field()
    instance1 = TestClass(item=None)
    instance2 = TestClass(item=None)
    hash1 = instance1.__hash__()
    hash2 = instance2.__hash__()
    assert hash1 == hash2


# LLM-generated content at query #4
#--------------------------

def test_set_updates_data_and_flags():
    original = object()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('c', 3)
    assert evolver._pclass_evolver_data == {'a': 1, 'b': 2, 'c': 3}
    assert evolver._factory_fields == {'c'}
    assert evolver._pclass_evolver_data_is_dirty == True

def test_set_with_same_value_does_not_update():
    original = object()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 1)
    assert evolver._pclass_evolver_data == {'a': 1}
    assert evolver._factory_fields == set()
    assert evolver._pclass_evolver_data_is_dirty == False

def test_set_overwrites_existing_key():
    original = object()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 100)
    assert evolver._pclass_evolver_data == {'a': 100}
    assert evolver._factory_fields == {'a'}
    assert evolver._pclass_evolver_data_is_dirty == True

def test_set_returns_self():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set('x', 10)
    assert result is evolver


# LLM-generated content at query #5
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
    def serialize_x(value, format):
        return value * 2
    def serialize_y(value, format):
        return value.upper()
    class TestClass(PClass):
        x = field(type=int, serializer=serialize_x)
        y = field(type=str, serializer=serialize_y)
    instance = TestClass(x=5, y="test")
    result = instance.serialize()
    expected = {'x': 10, 'y': 'TEST'}
    assert result == expected

def test_serialize_with_format_parameter():
    from pyrsistent import PClass, field
    def custom_serializer(value, format):
        if format == 'json':
            return str(value) + '_json'
        return value
    class TestClass(PClass):
        data = field(serializer=custom_serializer)
    instance = TestClass(data=100)
    result = instance.serialize(format='json')
    expected = {'data': '100_json'}
    assert result == expected

def test_serialize_missing_field_with_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=42)
        y = field()
    instance = TestClass(y="world")
    result = instance.serialize()
    expected = {'x': 42, 'y': 'world'}
    assert result == expected

def test_serialize_empty_pclass():
    from pyrsistent import PClass, field
    class EmptyClass(PClass):
        pass
    instance = EmptyClass()
    result = instance.serialize()
    expected = {}
    assert result == expected

def test_serialize_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_serialize_ignores_extra_kwargs_when_ignore_extra_true():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
    instance = TestClass.create({'a': 1, 'b': 2}, ignore_extra=True)
    result = instance.serialize()
    expected = {'a': 1}
    assert result == expected

def test_serialize_after_set_operation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=3)
    result = new_instance.serialize()
    expected = {'x': 3, 'y': 2}
    assert result == expected

def test_serialize_with_none_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=None, y=None)
    result = instance.serialize()
    expected = {'x': None, 'y': None}
    assert result == expected

def test_serialize_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = TestClass.create(instance, _factory_fields={'x'})
    result = new_instance.serialize()
    expected = {'x': 1, 'y': 2}
    assert result == expected


# LLM-generated content at query #6
#--------------------------

def test___new___single_inheritance_without_fields():
    class Base(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class Derived(Base):
        pass
    assert Derived._pclass_fields == {}
    assert len(Derived._pclass_invariants) == 1
    assert Derived.__slots__ == ('_pclass_frozen', '__weakref__')

def test___new___multiple_inheritance_without_fields():
    class Base1(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class Base2(metaclass=PClassMeta):
        pass
    class Derived(Base1, Base2):
        pass
    assert Derived._pclass_fields == {}
    assert len(Derived._pclass_invariants) == 1
    assert Derived.__slots__ == ('_pclass_frozen',)

def test___new___with_fields():
    from pyrsistent._field_common import _PField
    field = _PField(type=int, invariant=lambda x: (True, ()))
    class MyClass(metaclass=PClassMeta):
        my_field = field
    assert MyClass._pclass_fields == {'my_field': field}
    assert 'my_field' not in MyClass.__dict__
    assert MyClass.__slots__ == ('_pclass_frozen', 'my_field')

def test___new___inherits_fields():
    from pyrsistent._field_common import _PField
    field1 = _PField(type=int, invariant=lambda x: (True, ()))
    field2 = _PField(type=str, invariant=lambda x: (True, ()))
    class Base(metaclass=PClassMeta):
        base_field = field1
    class Derived(Base):
        derived_field = field2
    assert Derived._pclass_fields == {'base_field': field1, 'derived_field': field2}
    assert Derived.__slots__ == ('_pclass_frozen', 'base_field', 'derived_field')

def test___new___inherits_invariants():
    invariant1 = lambda self: (True, ())
    invariant2 = lambda self: (True, ())
    class Base(metaclass=PClassMeta):
        __invariant__ = invariant1
    class Derived(Base):
        __invariant__ = invariant2
    assert len(Derived._pclass_invariants) == 2
    assert Derived._pclass_invariants[0].__wrapped__ == invariant2
    assert Derived._pclass_invariants[1].__wrapped__ == invariant1

def test___new___non_callable_invariant_raises():
    try:
        class Invalid(metaclass=PClassMeta):
            __invariant__ = "not callable"
        assert False
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test___new___invariant_wrapping():
    def multi_result_invariant(self):
        return [(True, ()), (False, "error")]
    class MyClass(metaclass=PClassMeta):
        __invariant__ = multi_result_invariant
    result = MyClass._pclass_invariants[0](None)
    assert result == (False, ("error",))

def test___new___single_bool_invariant():
    def single_invariant(self):
        return False, "single error"
    class MyClass(metaclass=PClassMeta):
        __invariant__ = single_invariant
    result = MyClass._pclass_invariants[0](None)
    assert result == (False, "single error")


# LLM-generated content at query #7
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

def test_set_with_factory_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(factory=int)
        y = field()
    a = AClass(x='5', y=10)
    a2 = a.set(x='7')
    assert a.x == 5
    assert a2.x == 7
    assert a.y == 10
    assert a2.y == 10


# LLM-generated content at query #8
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
    instance = TestClass(x=5, y="hello")
    result = repr(instance)
    expected = "TestClass(x=5, y='hello')"
    assert result == expected

def test_repr_with_no_fields():
    from pyrsistent import PClass
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = repr(instance)
    expected = "TestClass()"
    assert result == expected

def test_repr_with_field_containing_special_characters():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        field_name = field()
    instance = TestClass(field_name="value")
    result = repr(instance)
    expected = "TestClass(field_name='value')"
    assert result == expected

def test_repr_with_none_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=None)
    result = repr(instance)
    expected = "TestClass(x=None)"
    assert result == expected

def test_repr_with_list_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        items = field()
    instance = TestClass(items=[1, 2, 3])
    result = repr(instance)
    expected = "TestClass(items=[1, 2, 3])"
    assert result == expected

def test_repr_with_dict_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        config = field()
    instance = TestClass(config={'key': 'value'})
    result = repr(instance)
    expected = "TestClass(config={'key': 'value'})"
    assert result == expected

def test_repr_with_initial_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=100)
    instance = TestClass()
    result = repr(instance)
    expected = "TestClass(x=100)"
    assert result == expected

def test_repr_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
    except Exception as e:
        pass
    else:
        assert False, "Should have raised an exception"

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

def test_repr_with_boolean_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        flag = field()
    instance = TestClass(flag=True)
    result = repr(instance)
    expected = "TestClass(flag=True)"
    assert result == expected

def test_repr_with_integer_zero():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        count = field()
    instance = TestClass(count=0)
    result = repr(instance)
    expected = "TestClass(count=0)"
    assert result == expected

def test_repr_with_empty_string():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        text = field()
    instance = TestClass(text="")
    result = repr(instance)
    expected = "TestClass(text='')"
    assert result == expected

def test_repr_with_tuple_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        data = field()
    instance = TestClass(data=(1, 2, 3))
    result = repr(instance)
    expected = "TestClass(data=(1, 2, 3))"
    assert result == expected

def test_repr_with_custom_object():
    from pyrsistent import PClass, field
    class InnerClass:
        def __repr__(self):
            return "InnerClass()"
    class TestClass(PClass):
        obj = field()
    instance = TestClass(obj=InnerClass())
    result = repr(instance)
    expected = "TestClass(obj=InnerClass())"
    assert result == expected


# LLM-generated content at query #9
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
    except Exception as e:
        assert isinstance(e, InvariantException)
        assert "TestClass.x" in e.missing_fields

def test_pclass_constructor_with_extra_field_raises_attribute_error():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=10, z=30)
    except Exception as e:
        assert isinstance(e, AttributeError)
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
        x = field(type=int)
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

def test_pclass_constructor_with_ignore_extra_false_and_extra_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(ignore_extra=False, x=10, z=30)
    except Exception as e:
        assert isinstance(e, AttributeError)

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
    except Exception as e:
        assert isinstance(e, InvariantException)
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException, invariant
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [invariant(lambda c: c.x + c.y > 0, "Sum must be positive")]
    try:
        TestClass(x=-10, y=5)
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_pclass_constructor_creates_frozen_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    try:
        instance.x = 20
    except Exception as e:
        assert isinstance(e, AttributeError)
        assert "Can't set attribute" in str(e)

def test_pclass_constructor_with_no_arguments_and_all_optional_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=1)
        y = field(initial=2)
    instance = TestClass()
    assert instance.x == 1
    assert instance.y == 2

def test_pclass_constructor_with_mixed_mandatory_and_optional_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field(mandatory=True)
        b = field(initial=0)
        c = field(mandatory=True)
    instance = TestClass(a=100, c=200)
    assert instance.a == 100
    assert instance.b == 0
    assert instance.c == 200

def test_pclass_constructor_with_factory_field_and_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
        y = field()
    instance = TestClass(_factory_fields={'x'}, ignore_extra=True, x=10, y=20, z=30)
    assert instance.x == 10
    assert instance.y == 20
    assert not hasattr(instance, 'z')

def test_pclass_constructor_with_factory_field_and_no_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
        y = field()
    try:
        TestClass(_factory_fields={'x'}, ignore_extra=False, x=10, y=20, z=30)
    except Exception as e:
        assert isinstance(e, AttributeError)

def test_pclass_constructor_with_field_ignore_extra_compliant():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, ignore_extra=True)
        y = field()
    instance = TestClass(ignore_extra=True, x=10, y=20, extra=100)
    assert instance.x == 10
    assert instance.y == 20
    assert not hasattr(instance, 'extra')

def test_pclass_constructor_with_field_not_ignore_extra_compliant():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, ignore_extra=False)
        y = field()
    try:
        TestClass(ignore_extra=True, x=10, y=20, extra=100)
    except Exception as e:
        assert isinstance(e, AttributeError)


# LLM-generated content at query #10
#--------------------------

```python
def test_is_pclass_returns_false_for_non_pclass_bases():
    class NonPClass:
        pass

    class TestClass(metaclass=PClassMeta):
        pass

    class DerivedClass(TestClass, NonPClass, metaclass=PClassMeta):
        pass

    result = '_pclass_frozen' in DerivedClass.__slots__
    assert result


# LLM-generated content at query #11
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_invariant_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True, invariant=lambda x: (x > 0, 'error1'))
    try:
        TestClass(x=-1)
    except InvariantException as e:
        assert e.error_codes == ('error1',)
        assert e.missing_fields == ()
    else:
        assert False


# LLM-generated content at query #12
#--------------------------

def test_set_with_existing_field_uses_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=3)
    assert new_instance.x == 3
    assert new_instance.y == 2

def test_set_with_new_field_not_in_kwargs_adds_existing_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=3)
    assert 'y' in new_instance._to_dict()

def test_set_with_positional_args_adds_to_kwargs():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set('x', 3)
    assert new_instance.x == 3
    assert new_instance.y == 2

def test_set_with_multiple_fields_updates_only_specified():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
        z = field()
    instance = TestClass(x=1, y=2, z=3)
    new_instance = instance.set(x=10, z=30)
    assert new_instance.x == 10
    assert new_instance.y == 2
    assert new_instance.z == 30

def test_set_with_no_args_returns_same_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    new_instance = instance.set()
    assert new_instance is not instance
    assert new_instance.x == 1


# LLM-generated content at query #13
#--------------------------

def test_repr_returns_correct_format():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y="hello")
    expected = "TestClass(x=10, y='hello')"
    actual = repr(instance)
    assert actual == expected

def test_repr_with_no_fields():
    from pyrsistent import PClass, field
    class EmptyClass(PClass):
        pass
    instance = EmptyClass()
    expected = "EmptyClass()"
    actual = repr(instance)
    assert actual == expected

def test_repr_with_one_field():
    from pyrsistent import PClass, field
    class SingleFieldClass(PClass):
        name = field()
    instance = SingleFieldClass(name="Alice")
    expected = "SingleFieldClass(name='Alice')"
    actual = repr(instance)
    assert actual == expected

def test_repr_with_special_characters_in_field_value():
    from pyrsistent import PClass, field
    class SpecialClass(PClass):
        text = field()
    instance = SpecialClass(text='line1\nline2')
    expected = "SpecialClass(text='line1\\nline2')"
    actual = repr(instance)
    assert actual == expected

def test_repr_with_numeric_field_names_and_values():
    from pyrsistent import PClass, field
    class NumericClass(PClass):
        a = field()
        b = field()
    instance = NumericClass(a=1, b=2.5)
    expected = "NumericClass(a=1, b=2.5)"
    actual = repr(instance)
    assert actual == expected

def test_repr_after_set_operation():
    from pyrsistent import PClass, field
    class UpdateClass(PClass):
        x = field()
        y = field()
    instance = UpdateClass(x=1, y=2)
    new_instance = instance.set(x=100)
    expected = "UpdateClass(x=100, y=2)"
    actual = repr(new_instance)
    assert actual == expected

def test_repr_with_boolean_and_none_values():
    from pyrsistent import PClass, field
    class MixedClass(PClass):
        flag = field()
        value = field()
    instance = MixedClass(flag=True, value=None)
    expected = "MixedClass(flag=True, value=None)"
    actual = repr(instance)
    assert actual == expected

def test_repr_uses_to_dict_for_field_retrieval():
    from pyrsistent import PClass, field
    class DictCheckClass(PClass):
        a = field()
        b = field()
    instance = DictCheckClass(a=5, b=10)
    dict_repr = instance._to_dict()
    expected_keys = {'a', 'b'}
    actual_keys = set(dict_repr.keys())
    assert actual_keys == expected_keys
    assert dict_repr['a'] == 5
    assert dict_repr['b'] == 10
    repr_output = repr(instance)
    assert 'a=5' in repr_output
    assert 'b=10' in repr_output

def test_repr_includes_class_name_correctly():
    from pyrsistent import PClass, field
    class CustomClassName(PClass):
        field1 = field()
    instance = CustomClassName(field1="test")
    repr_output = repr(instance)
    assert repr_output.startswith("CustomClassName(")
    assert repr_output.endswith(")")

def test_repr_orders_fields_as_in_to_dict_items():
    from pyrsistent import PClass, field
    class OrderedClass(PClass):
        z = field()
        a = field()
        m = field()
    instance = OrderedClass(z=3, a=1, m=2)
    dict_items = list(instance._to_dict().items())
    expected_order = [('z', 3), ('a', 1), ('m', 2)]
    assert dict_items == expected_order
    repr_output = repr(instance)
    expected_repr = "OrderedClass(z=3, a=1, m=2)"
    assert repr_output == expected_repr


# LLM-generated content at query #14
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true_when_conditions_met():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    class TestClass(PClass):
        x = field(type=str, factory=lambda v, ignore_extra=False: v.upper())
    instance = TestClass(x="test")
    field_obj = TestClass._pclass_fields['x']
    result = is_field_ignore_extra_complaint(PClass, field_obj, True)
    assert result == True


# LLM-generated content at query #15
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test___new___raises_AttributeError_for_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, y=2)
        assert False
    except AttributeError as e:
        assert "'y' are not among the specified fields for TestClass" in str(e)

def test___new___uses_initial_for_missing_non_mandatory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test___new___raises_InvariantException_for_missing_mandatory_fields():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)
        assert ("TestClass.x",) == e.missing_fields

def test___new___raises_PTypeError_for_invalid_field_type():
    from pyrsistent import PClass, field, PTypeError
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="string")
        assert False
    except PTypeError as e:
        assert "Invalid type for field TestClass.x" in str(e)

def test___new___applies_field_invariant_and_raises_InvariantException_on_failure():
    from pyrsistent import PClass, field, InvariantException
    def positive_invariant(value):
        return value > 0, "value_not_positive"
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)
        assert ("value_not_positive",) == e.invariant_errors

def test___new___checks_global_invariants_and_raises_InvariantException_on_failure():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x + obj.y > 0, "sum_not_positive"
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [global_invariant]
    try:
        TestClass(x=-5, y=2)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)
        assert ("sum_not_positive",) == e.error_codes

def test___new___handles_factory_fields_with_ignore_extra():
    from pyrsistent import PClass, field
    def factory_with_ignore_extra(value, ignore_extra=False):
        return value * 2
    class TestClass(PClass):
        x = field(type=int, factory=factory_with_ignore_extra)
    instance = TestClass(x=5, ignore_extra=True)
    assert instance.x == 10

def test___new___sets_frozen_attribute_to_True():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert instance._pclass_frozen == True

def test___new___handles_callable_initial():
    from pyrsistent import PClass, field
    def initial_value():
        return 42
    class TestClass(PClass):
        x = field(initial=initial_value)
    instance = TestClass()
    assert instance.x == 42


# LLM-generated content at query #16
#--------------------------

def test_check_and_set_attr_valid():
    class MockField:
        type = (int,)
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
    assert invariant_errors == []

def test_check_and_set_attr_invalid_type():
    class MockField:
        type = (int,)
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
        assert e.expected_type == (int,)
        assert e.actual_type == str

def test_check_and_set_attr_invariant_fails():
    class MockField:
        type = (int,)
        def invariant(self, value):
            return False, "error_code"
    class MockClass:
        pass
    field = MockField()
    name = "test_field"
    value = 42
    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    assert not hasattr(result, name)
    assert invariant_errors == ["error_code"]

def test_check_and_set_attr_no_type_check():
    class MockField:
        type = None
        def invariant(self, value):
            return True, None
    class MockClass:
        pass
    field = MockField()
    name = "test_field"
    value = "any_value"
    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, name, value, result, invariant_errors)
    assert getattr(result, name) == value
    assert invariant_errors == []

def test_check_and_set_attr_multiple_types():
    class MockField:
        type = (int, str)
        def invariant(self, value):
            return True, None
    class MockClass:
        pass
    field = MockField()
    name = "test_field"
    value_int = 42
    value_str = "hello"
    result_int = MockClass()
    result_str = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, field, name, value_int, result_int, invariant_errors)
    _check_and_set_attr(MockClass, field, name, value_str, result_str, invariant_errors)
    assert getattr(result_int, name) == value_int
    assert getattr(result_str, name) == value_str
    assert invariant_errors == []


# LLM-generated content at query #17
#--------------------------

def test_is_pclass_true():
    bases = (CheckedType,)
    result = _is_pclass(bases)
    assert result == True

def test_is_pclass_false_multiple_bases():
    bases = (CheckedType, object)
    result = _is_pclass(bases)
    assert result == False

def test_is_pclass_false_different_base():
    bases = (object,)
    result = _is_pclass(bases)
    assert result == False

def test_is_pclass_false_empty():
    bases = ()
    result = _is_pclass(bases)
    assert result == False


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

def test_pclass_constructor_with_ignore_extra_true():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(ignore_extra=True, x=10, z=30)
    assert instance.x == 10
    assert not hasattr(instance, 'z')

def test_pclass_constructor_with_ignore_extra_false_and_extra_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(ignore_extra=False, x=10, z=30)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

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
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException, invariant
    class TestClass(PClass):
        x = field()
        y = field()
        @invariant(lambda self: self.x + self.y == 100)
        def sum_invariant(self):
            return self.x + self.y == 100
    try:
        TestClass(x=30, y=80)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_creates_frozen_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    try:
        instance.x = 20
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


# LLM-generated content at query #19
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

def test_hash_handles_missing_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj = TestClass(x=1)
    hash_value = hash(obj)
    assert isinstance(hash_value, int)

def test_hash_consistent_across_multiple_calls():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj = TestClass(x=1, y=2)
    assert hash(obj) == hash(obj)

def test_hash_uses_all_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    obj1 = TestClass(a=1, b=2, c=3)
    obj2 = TestClass(a=1, b=2, c=4)
    assert hash(obj1) != hash(obj2)

def test_hash_with_none_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=None, y=None)
    obj2 = TestClass(x=None, y=None)
    assert hash(obj1) == hash(obj2)

def test_hash_with_complex_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=[1, 2], y={'a': 3})
    obj2 = TestClass(x=[1, 2], y={'a': 3})
    assert hash(obj1) == hash(obj2)

def test_hash_different_for_different_field_order():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    obj1 = TestClass(x=1, y=2)
    obj2 = TestClass(y=2, x=1)
    assert hash(obj1) == hash(obj2)


# LLM-generated content at query #20
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


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

def test_constructor_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_constructor_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=20)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_constructor_with_extra_field_raises_attribute_error():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=10, z=30)
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)

def test_constructor_with_initial_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
        y = field()
    instance = TestClass(y=20)
    assert instance.x == 5
    assert instance.y == 20

def test_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 100)
    instance = TestClass()
    assert instance.x == 100

def test_constructor_with_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(_factory_fields={'x'}, x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_constructor_with_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10, z=30, ignore_extra=True)
    assert instance.x == 10
    assert not hasattr(instance, 'z')

def test_constructor_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def check_positive(value):
        if value <= 0:
            return (f"Value must be positive: {value}",)
        return ()
    class TestClass(PClass):
        x = field(invariant=check_positive)
    try:
        TestClass(x=-5)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        if obj.x + obj.y != 100:
            return ("Sum must be 100",)
        return ()
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [global_invariant]
    try:
        TestClass(x=30, y=80)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_constructor_creates_frozen_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    try:
        instance.x = 20
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)


# LLM-generated content at query #23
#--------------------------

def test_constructor_creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_constructor_raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=2)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_constructor_uses_initial_value_for_non_provided_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_constructor_raises_on_extra_field_when_not_ignored():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, z=3)
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)

def test_constructor_ignores_extra_field_when_ignore_extra_true():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1, z=3, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')

def test_constructor_calls_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 100)
    instance = TestClass()
    assert instance.x == 100

def test_constructor_invokes_field_factory():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, factory=lambda v: v * 2)
    instance = TestClass(x=5)
    assert instance.x == 10

def test_constructor_checks_invariant_and_raises_on_failure():
    from pyrsistent import PClass, field, InvariantException
    def positive_invariant(value):
        return value > 0, "Value must be positive"
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    try:
        TestClass(x=-1)
        assert False
    except Exception as e:
        assert isinstance(e, InvariantException)

def test_constructor_supports_factory_fields_parameter():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v + 1)
        y = field()
    instance = TestClass(x=5, y=10, _factory_fields={'x'})
    assert instance.x == 6
    assert instance.y == 10

def test_constructor_freezes_instance_after_creation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except Exception as e:
        assert isinstance(e, AttributeError)


# LLM-generated content at query #24
#--------------------------

def test_serialize_skips_missing_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1)
    serialized = instance.serialize()
    assert 'x' in serialized
    assert 'y' not in serialized


# LLM-generated content at query #25
#--------------------------

def test___new___single_inheritance():
    class CheckedType:
        pass
    class Base(CheckedType, metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class Derived(Base):
        pass
    assert Derived._pclass_invariants is not None
    assert isinstance(Derived._pclass_invariants, tuple)
    assert len(Derived._pclass_invariants) == 1
    assert Derived._pclass_fields == {}
    assert '_pclass_frozen' in Derived.__slots__
    assert '__weakref__' not in Derived.__slots__

def test___new___multiple_inheritance():
    class CheckedType:
        pass
    class Base1(CheckedType, metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class Base2(CheckedType, metaclass=PClassMeta):
        pass
    class Derived(Base1, Base2):
        pass
    assert Derived._pclass_invariants is not None
    assert isinstance(Derived._pclass_invariants, tuple)
    assert len(Derived._pclass_invariants) == 1
    assert Derived._pclass_fields == {}
    assert '_pclass_frozen' in Derived.__slots__
    assert '__weakref__' not in Derived.__slots__

def test___new___with_fields():
    class CheckedType:
        pass
    class _PField:
        def __init__(self):
            pass
    field = _PField()
    class MyClass(CheckedType, metaclass=PClassMeta):
        my_field = field
    assert MyClass._pclass_fields == {'my_field': field}
    assert 'my_field' not in MyClass.__dict__
    assert 'my_field' in MyClass.__slots__
    assert '_pclass_frozen' in MyClass.__slots__
    assert '__weakref__' in MyClass.__slots__

def test___new___invariant_not_callable():
    class CheckedType:
        pass
    try:
        class MyClass(CheckedType, metaclass=PClassMeta):
            __invariant__ = "not callable"
        assert False
    except TypeError:
        pass

def test___new___no_invariants():
    class CheckedType:
        pass
    class MyClass(CheckedType, metaclass=PClassMeta):
        pass
    assert MyClass._pclass_invariants == ()
    assert MyClass._pclass_fields == {}
    assert '_pclass_frozen' in MyClass.__slots__
    assert '__weakref__' in MyClass.__slots__

def test___new___inherited_invariants():
    class CheckedType:
        pass
    invariant1 = lambda self: (True, ())
    invariant2 = lambda self: (False, ("error",))
    class Base1(CheckedType, metaclass=PClassMeta):
        __invariant__ = invariant1
    class Base2(CheckedType, metaclass=PClassMeta):
        __invariant__ = invariant2
    class Derived(Base1, Base2):
        pass
    assert len(Derived._pclass_invariants) == 2
    assert Derived._pclass_invariants[0].__wrapped__ is invariant1
    assert Derived._pclass_invariants[1].__wrapped__ is invariant2
    assert Derived._pclass_fields == {}
    assert '_pclass_frozen' in Derived.__slots__
    assert '__weakref__' not in Derived.__slots__


# LLM-generated content at query #26
#--------------------------

def test___reduce___returns_correct_tuple_for_pickling():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] == TestClass
    assert result[1][1] == {'x': 10, 'y': 20}

def test___reduce___handles_missing_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10)
    result = instance.__reduce__()
    assert result[1][1] == {'x': 10}

def test___reduce___works_with_no_fields():
    from pyrsistent import PClass
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = instance.__reduce__()
    assert result[1][1] == {}

def test___reduce___preserves_field_order():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
        c = field()
    instance = TestClass(a=1, b=2, c=3)
    result = instance.__reduce__()
    assert list(result[1][1].keys()) == ['a', 'b', 'c']


# LLM-generated content at query #27
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

def test_pclass_constructor_ignore_extra_true():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1, z=3, ignore_extra=True)
    assert instance.x == 1

def test_pclass_constructor_ignore_extra_false():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, z=3, ignore_extra=False)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

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
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_global_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x + obj.y > 0, "sum must be positive"
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [global_invariant]
    try:
        TestClass(x=-5, y=2)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test_pclass_constructor_with_existing_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance1 = TestClass(x=1)
    instance2 = TestClass.create(instance1)
    assert instance2 is instance1

def test_pclass_constructor_frozen_after_creation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

def test_pclass_constructor_hash_equality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    instance3 = TestClass(x=3, y=4)
    assert instance1 == instance2
    assert instance1 != instance3
    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)

def test_pclass_constructor_repr():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    repr_str = repr(instance)
    assert repr_str.startswith("TestClass")
    assert "x=1" in repr_str
    assert "y=2" in repr_str

def test_pclass_constructor_with_no_fields():
    from pyrsistent import PClass
    class TestClass(PClass):
        pass
    instance = TestClass()
    assert isinstance(instance, TestClass)

def test_pclass_constructor_pickling_support():
    import pickle
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    pickled = pickle.dumps(instance)
    unpickled = pickle.loads(pickled)
    assert unpickled == instance
    assert unpickled.x == 1
    assert unpickled.y == 2


