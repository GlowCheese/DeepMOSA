####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_PClassMeta_new_single_inheritance():
    class CheckedType:
        pass
    class TestClass(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    result = type(TestClass)
    assert result is PClassMeta
    assert hasattr(TestClass, '_pclass_fields')
    assert isinstance(TestClass._pclass_fields, dict)
    assert hasattr(TestClass, '_pclass_invariants')
    assert isinstance(TestClass._pclass_invariants, tuple)
    assert len(TestClass._pclass_invariants) == 1
    assert '__weakref__' in TestClass.__slots__

def test_PClassMeta_new_multiple_inheritance():
    class CheckedType:
        pass
    class Base1(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class Base2(metaclass=PClassMeta):
        __invariant__ = lambda self: (False, ('error',))
    class TestClass(Base1, Base2):
        pass
    invariants = TestClass._pclass_invariants
    assert len(invariants) == 2
    assert invariants[0](None)[0] is True
    assert invariants[1](None)[0] is False

def test_PClassMeta_new_no_invariants():
    class CheckedType:
        pass
    class TestClass(metaclass=PClassMeta):
        pass
    assert TestClass._pclass_invariants == ()

def test_PClassMeta_new_with_fields():
    class CheckedType:
        pass
    class TestClass(metaclass=PClassMeta):
        field = _PField()
    fields = TestClass._pclass_fields
    assert 'field' in fields
    assert fields['field'] is TestClass.field
    assert 'field' not in TestClass.__dict__

def test_PClassMeta_new_slots_contain_fields():
    class CheckedType:
        pass
    class TestClass(metaclass=PClassMeta):
        field1 = _PField()
        field2 = _PField()
    slots = TestClass.__slots__
    assert 'field1' in slots
    assert 'field2' in slots
    assert '_pclass_frozen' in slots

def test_PClassMeta_new_non_callable_invariant_raises():
    class CheckedType:
        pass
    raised = False
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = 'not callable'
    except TypeError:
        raised = True
    assert raised is True

def test_PClassMeta_new_inherited_invariants():
    class CheckedType:
        pass
    class Base(metaclass=PClassMeta):
        __invariant__ = lambda self: (True, ())
    class Derived(Base):
        __invariant__ = lambda self: (False, ('derived error',))
    invariants = Derived._pclass_invariants
    assert len(invariants) == 2
    assert invariants[0](None)[0] is True
    assert invariants[1](None)[0] is False

def test_PClassMeta_new_wrap_invariant_merges_results():
    class CheckedType:
        pass
    def invariant_returning_list(self):
        return [(True, ()), (False, ('err1',)), (True, ()), (False, ('err2',))]
    class TestClass(metaclass=PClassMeta):
        __invariant__ = invariant_returning_list
    wrapped = TestClass._pclass_invariants[0]
    result = wrapped(None)
    assert result[0] is False
    assert result[1] == ('err1', 'err2')

def test_PClassMeta_new_wrap_invariant_single_bool():
    class CheckedType:
        pass
    def invariant_single_bool(self):
        return False, ('single error',)
    class TestClass(metaclass=PClassMeta):
        __invariant__ = invariant_single_bool
    wrapped = TestClass._pclass_invariants[0]
    result = wrapped(None)
    assert result[0] is False
    assert result[1] == ('single error',)

def test_PClassMeta_new_not_checked_type_no_weakref():
    class OtherBase:
        pass
    class TestClass(OtherBase, metaclass=PClassMeta):
        pass
    assert '__weakref__' not in TestClass.__slots__


# LLM-generated content at query #2
#--------------------------

def test_set_updates_data_when_value_different():
    original = object()
    initial_dict = {"key1": "value1"}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set("key1", "new_value")
    result = evolver._pclass_evolver_data["key1"]
    assert result == "new_value"

def test_set_marks_data_dirty_when_value_different():
    original = object()
    initial_dict = {"key1": "value1"}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set("key1", "new_value")
    result = evolver._pclass_evolver_data_is_dirty
    assert result is True

def test_set_adds_key_to_factory_fields_when_value_different():
    original = object()
    initial_dict = {"key1": "value1"}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set("key1", "new_value")
    result = "key1" in evolver._factory_fields
    assert result is True

def test_set_does_not_update_data_when_value_same():
    original = object()
    initial_dict = {"key1": "value1"}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set("key1", "value1")
    result = evolver._pclass_evolver_data_is_dirty
    assert result is False

def test_set_returns_self():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set("key1", "value1")
    assert result is evolver

def test_set_with_new_key_updates_data():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set("new_key", "new_value")
    result = evolver._pclass_evolver_data["new_key"]
    assert result == "new_value"

def test_set_with_new_key_marks_data_dirty():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set("new_key", "new_value")
    result = evolver._pclass_evolver_data_is_dirty
    assert result is True

def test_set_with_new_key_adds_to_factory_fields():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set("new_key", "new_value")
    result = "new_key" in evolver._factory_fields
    assert result is True

def test_set_with_missing_value_comparison():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set("key", _MISSING_VALUE)
    result = evolver._pclass_evolver_data["key"]
    assert result is _MISSING_VALUE


# LLM-generated content at query #3
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

def test_serialize_missing_field_with_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=42)
        y = field()
    instance = TestClass(y="present")
    result = instance.serialize()
    expected = {'x': 42, 'y': 'present'}
    assert result == expected

def test_serialize_empty_pclass():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = instance.serialize()
    expected = {}
    assert result == expected


# LLM-generated content at query #4
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

def test_set_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(mandatory=True)
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a2.x == 2

def test_set_with_initial_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(initial=5)
    a = AClass()
    a2 = a.set(x=10)
    assert a.x == 5
    assert a2.x == 10

def test_set_ignores_extra_kwargs_in_original_creation():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1, ignore_extra=True)
    a2 = a.set(x=2)
    assert a2.x == 2

def test_set_with_factory_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1, _factory_fields={'x'})
    a2 = a.set(x=2)
    assert a2.x == 2

def test_set_raises_attribute_error_for_unknown_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    try:
        a.set(z=3)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass


# LLM-generated content at query #5
#--------------------------

def test_is_pclass_returns_true_for_pclass_bases():
    class MockPClass:
        pass
    bases = (MockPClass,)
    _is_pclass = lambda bs: any(hasattr(b, '_pclass_fields') for b in bs)
    MockPClass._pclass_fields = {}
    result = _is_pclass(bases)
    assert result == True


# LLM-generated content at query #6
#--------------------------

def test_remove_existing_item():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.remove('a')
    assert result is evolver
    assert 'a' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert evolver._factory_fields == set()

def test_remove_existing_item_from_factory_fields():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('c', 3)
    result = evolver.remove('c')
    assert result is evolver
    assert 'c' not in evolver._pclass_evolver_data
    assert evolver._pclass_evolver_data_is_dirty is True
    assert 'c' not in evolver._factory_fields

def test_remove_non_existing_item():
    original = type('Original', (), {})()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    try:
        evolver.remove('b')
        assert False
    except AttributeError as e:
        assert str(e) == 'b'

def test_remove_does_not_affect_other_items():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2, 'c': 3}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.remove('b')
    assert evolver._pclass_evolver_data['a'] == 1
    assert evolver._pclass_evolver_data['c'] == 3

def test_remove_twice():
    original = type('Original', (), {})()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.remove('a')
    try:
        evolver.remove('a')
        assert False
    except AttributeError as e:
        assert str(e) == 'a'


# LLM-generated content at query #7
#--------------------------

def test_check_and_set_attr_valid_type_and_invariant():
    class MockField:
        type = (int,)
        def invariant(self, value):
            return (True, None)
    class MockClass:
        pass
    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField, 'test_field', 42, result, invariant_errors)
    assert result.test_field == 42
    assert invariant_errors == []

def test_check_and_set_attr_invalid_type():
    class MockField:
        type = (int,)
        def invariant(self, value):
            return (True, None)
    class MockClass:
        pass
    result = MockClass()
    invariant_errors = []
    try:
        _check_and_set_attr(MockClass, MockField, 'test_field', 'not_an_int', result, invariant_errors)
        assert False
    except PTypeError as e:
        assert e.destination_cls == MockClass
        assert e.field_name == 'test_field'
        assert e.expected_types == (int,)
        assert e.actual_type == str

def test_check_and_set_attr_failed_invariant():
    class MockField:
        type = (int,)
        def invariant(self, value):
            return (False, 'invalid_value')
    class MockClass:
        pass
    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField, 'test_field', 42, result, invariant_errors)
    assert not hasattr(result, 'test_field')
    assert invariant_errors == ['invalid_value']

def test_check_and_set_attr_no_type_check():
    class MockField:
        type = None
        def invariant(self, value):
            return (True, None)
    class MockClass:
        pass
    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField, 'test_field', 'any_value', result, invariant_errors)
    assert result.test_field == 'any_value'
    assert invariant_errors == []

def test_check_and_set_attr_multiple_types_valid():
    class MockField:
        type = (int, str)
        def invariant(self, value):
            return (True, None)
    class MockClass:
        pass
    result = MockClass()
    invariant_errors = []
    _check_and_set_attr(MockClass, MockField, 'test_field', 'a_string', result, invariant_errors)
    assert result.test_field == 'a_string'
    assert invariant_errors == []

def test_check_and_set_attr_multiple_types_invalid():
    class MockField:
        type = (int, str)
        def invariant(self, value):
            return (True, None)
    class MockClass:
        pass
    result = MockClass()
    invariant_errors = []
    try:
        _check_and_set_attr(MockClass, MockField, 'test_field', 3.14, result, invariant_errors)
        assert False
    except PTypeError as e:
        assert e.destination_cls == MockClass
        assert e.field_name == 'test_field'
        assert e.expected_types == (int, str)
        assert e.actual_type == float


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

def test___reduce___with_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    instance = TestClass(x=5, y=15)
    result = instance.__reduce__()
    assert result[1][1] == {'x': 5, 'y': 15}

def test___reduce___with_initial_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=100)
        y = field()
    instance = TestClass(y=200)
    result = instance.__reduce__()
    assert result[1][1] == {'x': 100, 'y': 200}

def test___reduce___pickle_roundtrip():
    import pickle
    from pyrsistent import PClass, field
    class TestClass(PClass):
        name = field()
        value = field()
    original = TestClass(name='test', value=42)
    pickled = pickle.dumps(original)
    unpickled = pickle.loads(pickled)
    assert unpickled == original
    assert unpickled.name == 'test'
    assert unpickled.value == 42

def test___reduce___after_set():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    instance = TestClass(a=1, b=2)
    updated = instance.set(a=99)
    result = updated.__reduce__()
    assert result[1][1] == {'a': 99, 'b': 2}

def test___reduce___with_complex_values():
    from pyrsistent import PClass, field, pvector
    class TestClass(PClass):
        items = field()
        mapping = field()
    vec = pvector([1, 2, 3])
    mapping = {'key': 'value'}
    instance = TestClass(items=vec, mapping=mapping)
    result = instance.__reduce__()
    assert result[1][1]['items'] == vec
    assert result[1][1]['mapping'] == mapping

def test___reduce___ignores_extra_attributes():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    instance._extra_attr = 'should_not_appear'
    result = instance.__reduce__()
    assert '_extra_attr' not in result[1][1]


# LLM-generated content at query #9
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test___new___raises_AttributeError_on_extra_fields():
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

def test___new___raises_InvariantException_on_missing_mandatory_fields():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=2)
        assert False
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields

def test___new___raises_InvariantException_on_field_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant(value):
        return value > 0, "value must be positive"
    class TestClass(PClass):
        x = field(invariant=invariant)
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "value must be positive" in e.invariant_errors

def test___new___raises_PTypeError_on_type_mismatch():
    from pyrsistent import PClass, field, PTypeError
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="string")
        assert False
    except PTypeError as e:
        assert "Invalid type for field TestClass.x" in str(e)

def test___new___applies_global_invariants():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x != obj.y, "x and y must be different"
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [global_invariant]
    try:
        TestClass(x=1, y=1)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___handles_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    instance = TestClass()
    assert instance.x == 42

def test___new___sets_frozen_flag():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert instance._pclass_frozen == True

def test___new___with_factory_fields_and_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, factory=lambda v, ignore_extra=False: v * 2)
    instance = TestClass(x=5, ignore_extra=True)
    assert instance.x == 10

def test___new___propagates_ignore_extra_to_factory():
    from pyrsistent import PClass, field
    factory_called = False
    def factory(v, ignore_extra=False):
        nonlocal factory_called
        factory_called = True
        assert ignore_extra == True
        return v
    class TestClass(PClass):
        x = field(type=int, factory=factory)
    instance = TestClass(x=1, ignore_extra=True)
    assert factory_called == True
    assert instance.x == 1

def test___new___without_factory_fields_uses_raw_value():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(_factory_fields=set(), x=5)
    assert instance.x == 5

def test___new___with_factory_fields_uses_factory():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(factory=lambda v: v * 2)
    instance = TestClass(_factory_fields={'x'}, x=5)
    assert instance.x == 10


# LLM-generated content at query #10
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true_for_ignore_extra_compliant_field():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    class TestClass(PClass):
        x = field(type=dict, factory=lambda v, ignore_extra=False: v)
    test_instance = TestClass.create({'x': {}}, ignore_extra=True)
    test_field = TestClass._pclass_fields['x']
    result = is_field_ignore_extra_complaint(PClass, test_field, True)
    assert result is True


# LLM-generated content at query #11
#--------------------------

```python
def test_is_pclass_returns_true_for_pclass_bases():
    from pyrsistent._pclass import _is_pclass, PClassMeta
    class BasePClass(metaclass=PClassMeta):
        pass
    class DerivedPClass(BasePClass):
        pass
    result = _is_pclass((BasePClass,))
    assert result is True


# LLM-generated content at query #12
#--------------------------

def test_repr_with_single_field():
    from pyrsistent import PClass, field
    class SimpleClass(PClass):
        x = field()
    instance = SimpleClass(x=42)
    result = repr(instance)
    expected = "SimpleClass(x=42)"
    assert result == expected

def test_repr_with_multiple_fields():
    from pyrsistent import PClass, field
    class MultiClass(PClass):
        a = field()
        b = field()
        c = field()
    instance = MultiClass(a=1, b='test', c=3.14)
    result = repr(instance)
    expected = "MultiClass(a=1, b='test', c=3.14)"
    assert result == expected

def test_repr_with_no_fields():
    from pyrsistent import PClass
    class EmptyClass(PClass):
        pass
    instance = EmptyClass()
    result = repr(instance)
    expected = "EmptyClass()"
    assert result == expected

def test_repr_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class MandatoryClass(PClass):
        x = field(mandatory=True)
    try:
        instance = MandatoryClass()
    except Exception:
        pass

def test_repr_with_initial_field():
    from pyrsistent import PClass, field
    class InitialClass(PClass):
        x = field(initial=10)
    instance = InitialClass()
    result = repr(instance)
    expected = "InitialClass(x=10)"
    assert result == expected

def test_repr_with_callable_initial():
    from pyrsistent import PClass, field
    class CallableInitialClass(PClass):
        x = field(initial=lambda: 100)
    instance = CallableInitialClass()
    result = repr(instance)
    expected = "CallableInitialClass(x=100)"
    assert result == expected

def test_repr_with_nested_pclass():
    from pyrsistent import PClass, field
    class Inner(PClass):
        val = field()
    class Outer(PClass):
        inner = field()
    inner_instance = Inner(val=5)
    outer_instance = Outer(inner=inner_instance)
    result = repr(outer_instance)
    expected = "Outer(inner=Inner(val=5))"
    assert result == expected

def test_repr_with_special_characters_in_string():
    from pyrsistent import PClass, field
    class StringClass(PClass):
        text = field()
    instance = StringClass(text='line1\nline2')
    result = repr(instance)
    expected = "StringClass(text='line1\\nline2')"
    assert result == expected

def test_repr_with_boolean_and_none():
    from pyrsistent import PClass, field
    class MixedClass(PClass):
        a = field()
        b = field()
        c = field()
    instance = MixedClass(a=True, b=False, c=None)
    result = repr(instance)
    expected = "MixedClass(a=True, b=False, c=None)"
    assert result == expected

def test_repr_after_set_operation():
    from pyrsistent import PClass, field
    class UpdateClass(PClass):
        x = field()
        y = field()
    instance = UpdateClass(x=1, y=2)
    new_instance = instance.set(x=99)
    result = repr(new_instance)
    expected = "UpdateClass(x=99, y=2)"
    assert result == expected


# LLM-generated content at query #13
#--------------------------

def test___eq___with_same_class_and_equal_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    result = instance1 == instance2
    assert result is True

def test___eq___with_same_class_and_different_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=3, y=4)
    result = instance1 == instance2
    assert result is False

def test___eq___with_different_class():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    result = instance1 == instance2
    assert result is False

def test___eq___with_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    other = object()
    result = instance == other
    assert result is NotImplemented

def test___eq___with_missing_field_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    result = instance1 == instance2
    assert result is True

def test___eq___with_one_missing_and_one_present_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1, y=2)
    result = instance1 == instance2
    assert result is False


# LLM-generated content at query #14
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

def test_persistent_returns_new_instance_after_remove():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.remove('a')
    result = evolver.persistent()
    assert result is not original
    assert 'a' not in result._pclass_evolver_data
    assert result.b == 2

def test_persistent_returns_new_instance_after_setitem():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver['c'] = 3
    result = evolver.persistent()
    assert result is not original
    assert result.c == 3

def test_persistent_returns_new_instance_after_delitem():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    del evolver['a']
    result = evolver.persistent()
    assert result is not original
    assert 'a' not in result._pclass_evolver_data

def test_persistent_returns_new_instance_after_setattr():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.d = 4
    result = evolver.persistent()
    assert result is not original
    assert result.d == 4

def test_persistent_returns_original_when_set_same_value():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 1)
    result = evolver.persistent()
    assert result is original

def test_persistent_returns_new_instance_with_combined_changes():
    original = type('Original', (), {})()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    evolver.set('a', 10)
    evolver.remove('b')
    evolver['c'] = 30
    result = evolver.persistent()
    assert result is not original
    assert result.a == 10
    assert not hasattr(result, 'b')
    assert result.c == 30


# LLM-generated content at query #15
#--------------------------

def test_set_with_existing_field():
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

def test_set_with_new_field_via_kwargs():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(y=4)
    assert a2.x == 1
    assert a2.y == 4
    assert a.x == 1
    assert a.y == 2

def test_set_with_positional_args():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set('x', 5)
    assert a2.x == 5
    assert a2.y == 2
    assert a.x == 1
    assert a.y == 2

def test_set_with_multiple_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
        z = field()
    a = AClass(x=1, y=2, z=3)
    a2 = a.set(x=10, z=30)
    assert a2.x == 10
    assert a2.y == 2
    assert a2.z == 30
    assert a.x == 1
    assert a.y == 2
    assert a.z == 3

def test_set_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(mandatory=True)
        y = field()
    a = AClass(x=1)
    a2 = a.set(y=5)
    assert a2.x == 1
    assert a2.y == 5
    assert a.x == 1
    assert getattr(a, 'y', None) is None

def test_set_with_initial_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(initial=10)
        y = field()
    a = AClass(y=2)
    a2 = a.set(x=20)
    assert a2.x == 20
    assert a2.y == 2
    assert a.x == 10
    assert a.y == 2

def test_set_preserves_factory_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(x=3)
    assert a2._pclass_fields is AClass._pclass_fields
    assert a._pclass_fields is AClass._pclass_fields

def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(x=3)
    assert a is not a2
    assert a != a2

def test_set_with_no_args_no_kwargs():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set()
    assert a2.x == 1
    assert a2.y == 2
    assert a == a2
    assert a is not a2

def test_set_with_extra_kwargs_raises_attribute_error():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    try:
        a.set(z=3)
        assert False
    except AttributeError:
        pass


# LLM-generated content at query #16
#--------------------------

def test_constructor_creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    assert instance.x == 1
    assert instance.y == 2

def test_constructor_raises_attribute_error_for_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, y=2)
        assert False
    except AttributeError as e:
        assert "'y' are not among the specified fields for TestClass" in str(e)

def test_constructor_uses_initial_value_for_missing_non_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=5)
    assert instance.x == 10
    assert instance.y == 5

def test_constructor_raises_invariant_exception_for_missing_mandatory_field():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
        assert False
    except InvariantException as e:
        assert "TestClass.x" in e.missing_fields

def test_constructor_raises_invariant_exception_for_field_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant(value):
        return (value > 0, "value must be positive")
    class TestClass(PClass):
        x = field(invariant=invariant)
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "value must be positive" in str(e)

def test_constructor_with_ignore_extra_ignores_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass.create({"x": 1, "y": 2}, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'y')

def test_constructor_with_factory_fields_uses_factory_for_specified_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, factory=lambda v, ignore_extra: v * 2)
        y = field(type=int)
    instance = TestClass.create({"x": 1, "y": 2}, _factory_fields={"x"})
    assert instance.x == 2
    assert instance.y == 2

def test_constructor_sets_frozen_attribute_to_true():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    assert instance._pclass_frozen is True

def test_constructor_raises_attribute_error_when_setting_attribute_after_creation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

def test_constructor_handles_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
    instance = TestClass()
    assert instance.x == 42


# LLM-generated content at query #17
#--------------------------

```python
def test_invariant_errors_or_missing_fields_trigger_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field(invariant=lambda v: (v > 0, 'y_positive'))
    try:
        TestClass(y=-1)
    except InvariantException as e:
        assert 'y_positive' in e.invariant_errors
        assert 'TestClass.x' in e.missing_fields
    else:
        assert False


# LLM-generated content at query #18
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_invariant_exception():
    from pyrsistent import PClass, field, InvariantException
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
    except InvariantException as e:
        assert e.missing_fields == ('TestClass.x',)
        assert e.error_codes == ()
        assert e.message == 'Field invariant failed'
    else:
        assert False, "Expected InvariantException"


# LLM-generated content at query #19
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

def test___hash___handles_missing_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
        z = field(mandatory=False)
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert hash(instance1) == hash(instance2)

def test___hash___consistent_with_equality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    instance1 = TestClass(a=10, b=20)
    instance2 = TestClass(a=10, b=20)
    instance3 = TestClass(a=30, b=40)
    assert (instance1 == instance2) == (hash(instance1) == hash(instance2))
    assert (instance1 == instance3) == (hash(instance1) == hash(instance3))

def test___hash___uses_all_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        f1 = field()
        f2 = field()
        f3 = field()
    instance = TestClass(f1=100, f2=200, f3=300)
    expected_hash = hash((('f1', 100), ('f2', 200), ('f3', 300)))
    assert hash(instance) == expected_hash

def test___hash___works_with_none_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=None, y=None)
    instance2 = TestClass(x=None, y=None)
    assert hash(instance1) == hash(instance2)

def test___hash___works_with_boolean_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        flag = field()
        active = field()
    instance1 = TestClass(flag=True, active=False)
    instance2 = TestClass(flag=True, active=False)
    assert hash(instance1) == hash(instance2)

def test___hash___works_with_string_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        name = field()
        desc = field()
    instance1 = TestClass(name="test", desc="description")
    instance2 = TestClass(name="test", desc="description")
    assert hash(instance1) == hash(instance2)

def test___hash___works_with_tuple_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        coords = field()
        items = field()
    instance1 = TestClass(coords=(1, 2), items=(3, 4))
    instance2 = TestClass(coords=(1, 2), items=(3, 4))
    assert hash(instance1) == hash(instance2)

def test___hash___works_with_custom_class_values():
    from pyrsistent import PClass, field
    class InnerClass(PClass):
        val = field()
    class TestClass(PClass):
        inner = field()
    inner_instance = InnerClass(val=5)
    instance1 = TestClass(inner=inner_instance)
    instance2 = TestClass(inner=inner_instance)
    assert hash(instance1) == hash(instance2)


# LLM-generated content at query #20
#--------------------------

```python
def test_check_type_raises_ptypeerror_when_type_mismatch():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    from pyrsistent._pclass import PTypeError

    class MockField:
        type = (int,)
        def invariant(self, value):
            return True, None

    class DestinationClass:
        pass

    field = MockField()
    name = "test_field"
    value = "not_an_int"
    try:
        check_type(DestinationClass, field, name, value)
        assert False, "Expected PTypeError"
    except PTypeError as e:
        assert e.destination_cls == DestinationClass
        assert e.field_name == name
        assert e.expected_types == (int,)
        assert e.actual_type == str
        assert "Invalid type for field DestinationClass.test_field, was str" in str(e)

def test_check_type_passes_when_type_matches():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type

    class MockField:
        type = (int, float)
        def invariant(self, value):
            return True, None

    class DestinationClass:
        pass

    field = MockField()
    name = "test_field"
    value = 42
    check_type(DestinationClass, field, name, value)

def test_check_type_with_multiple_allowed_types():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type

    class MockField:
        type = (int, str)
        def invariant(self, value):
            return True, None

    class DestinationClass:
        pass

    field = MockField()
    name = "test_field"
    value = "valid_string"
    check_type(DestinationClass, field, name, value)

def test_check_type_with_no_type_specified():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type

    class MockField:
        type = None
        def invariant(self, value):
            return True, None

    class DestinationClass:
        pass

    field = MockField()
    name = "test_field"
    value = object()
    check_type(DestinationClass, field, name, value)

def test_check_type_with_custom_class_type():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type

    class CustomClass:
        pass

    class MockField:
        type = (CustomClass,)
        def invariant(self, value):
            return True, None

    class DestinationClass:
        pass

    field = MockField()
    name = "test_field"
    value = CustomClass()
    check_type(DestinationClass, field, name, value)

def test_check_and_set_attr_invariant_fails():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type

    class MockField:
        type = (int,)
        def invariant(self, value):
            return False, "invariant_failed"

    class DestinationClass:
        pass

    cls = DestinationClass
    field = MockField()
    name = "test_field"
    value = 42
    result = type('Result', (), {})()
    invariant_errors = []
    _check_and_set_attr(cls, field, name, value, result, invariant_errors)
    assert len(invariant_errors) == 1
    assert invariant_errors[0] == "invariant_failed"
    assert not hasattr(result, name)

def test_check_and_set_attr_invariant_passes():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type

    class MockField:
        type = (int,)
        def invariant(self, value):
            return True, None

    class DestinationClass:
        pass

    cls = DestinationClass
    field = MockField()
    name = "test_field"
    value = 42
    result = type('Result', (), {})()
    invariant_errors = []
    _check_and_set_attr(cls, field, name, value, result, invariant_errors)
    assert len(invariant_errors) == 0
    assert getattr(result, name) == value

def test_check_and_set_attr_type_check_fails():
    from pyrsistent._field_common import check_type
    from pyrsistent._checked_types import get_type
    from pyrsistent._pclass import PTypeError

    class MockField:
        type = (int,)
        def invariant(self, value):
            return True, None

    class DestinationClass:
        pass

    cls = DestinationClass
    field = MockField()
    name = "test_field"
    value = "not_an_int"
    result = type('Result', (), {})()
    invariant_errors = []
    try:
        _check_and_set_attr(cls, field, name, value, result, invariant_errors)
        assert False, "Expected PTypeError"
    except PTypeError:
        pass
    assert len(invariant_errors) == 0
    assert not hasattr(result, name)


# LLM-generated content at query #21
#--------------------------

```python
def test_is_field_ignore_extra_complaint_returns_true_when_conditions_met():
    from pyrsistent import PClass, field
    from pyrsistent._field_common import is_field_ignore_extra_complaint
    from pyrsistent._checked_types import get_type
    import inspect

    class MockField:
        def __init__(self, type_spec, factory_params):
            self.type = type_spec
            self.factory = lambda x, **kwargs: x
            self.factory.__signature__ = inspect.signature(lambda x, **kwargs: x) if 'ignore_extra' in factory_params else inspect.signature(lambda x: x)

    class TestPClass(PClass):
        x = field(type=str, mandatory=True)

    field_instance = MockField(type_spec=(str,), factory_params=['ignore_extra'])
    result = is_field_ignore_extra_complaint(PClass, field_instance, ignore_extra=True)
    assert result == True


# LLM-generated content at query #22
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


# LLM-generated content at query #23
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
        y = field(initial=10)
    instance = TestClass(x=5)
    hash_value = hash(instance)
    assert isinstance(hash_value, int)

def test___hash___consistent_across_multiple_calls():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    instance = TestClass(a="test", b=42)
    hash1 = hash(instance)
    hash2 = hash(instance)
    assert hash1 == hash2

def test___hash___works_with_nested_structures():
    from pyrsistent import PClass, field, pvector
    class Inner(PClass):
        val = field()
    class Outer(PClass):
        inner = field()
        items = field()
    inner = Inner(val=7)
    outer1 = Outer(inner=inner, items=pvector([1, 2]))
    outer2 = Outer(inner=inner, items=pvector([1, 2]))
    assert hash(outer1) == hash(outer2)

def test___hash___different_for_different_field_order():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    instance1 = TestClass(a=1, b=2)
    instance2 = TestClass(a=2, b=1)
    assert hash(instance1) != hash(instance2)

def test___hash___uses_all_fields_in_calculation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        f1 = field()
        f2 = field()
        f3 = field()
    instance1 = TestClass(f1=10, f2=20, f3=30)
    instance2 = TestClass(f1=10, f2=20, f3=31)
    assert hash(instance1) != hash(instance2)


# LLM-generated content at query #24
#--------------------------

def test_serialize_includes_only_fields_with_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1)
    serialized = instance.serialize()
    assert 'x' in serialized
    assert serialized['x'] == 1
    assert 'y' not in serialized


# LLM-generated content at query #25
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=True)
    instance = TestClass(x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test___new___raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
    try:
        TestClass()
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___applies_initial_value_for_non_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
        y = field(mandatory=True)
    instance = TestClass(y=10)
    assert instance.x == 5
    assert instance.y == 10

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
        y = field(mandatory=True)
    instance = TestClass(y=200)
    assert instance.x == 100
    assert instance.y == 200

def test___new___checks_type_and_raises_on_invalid():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="not_an_int")
        assert False
    except Exception as e:
        assert "Invalid type" in str(e)

def test___new___invariant_failure_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(val):
        return val > 0, "ERR"
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    try:
        TestClass(x=-5)
        assert False
    except InvariantException as e:
        assert "ERR" in str(e)

def test___new___global_invariant_failure_raises_exception():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x + obj.y > 0, "GLOBAL_ERR"
    class TestClass(PClass):
        x = field()
        y = field()
        __invariants__ = [global_invariant]
    try:
        TestClass(x=-10, y=5)
        assert False
    except InvariantException as e:
        assert "GLOBAL_ERR" in str(e)

def test___new___with_factory_fields_and_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=dict, factory=lambda d, ignore_extra=False: d if ignore_extra else d)
    instance = TestClass.create({"x": {"a": 1}, "extra": 2}, ignore_extra=True)
    assert instance.x == {"a": 1}

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


# LLM-generated content at query #26
#--------------------------

def test___reduce___returns_tuple_with_restore_pickle_and_class_and_data():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    result = instance.__reduce__()
    assert result[0].__name__ == '_restore_pickle'
    assert result[1][0] is TestClass
    assert result[1][1] == {'x': 10, 'y': 20}


# LLM-generated content at query #27
#--------------------------

def test_eq_returns_true_for_same_class_and_equal_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    assert instance1 == instance2


# LLM-generated content at query #28
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
    from pyrsistent import PClass
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
    instance = SingleFieldClass(name="test")
    expected = "SingleFieldClass(name='test')"
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

def test_repr_with_boolean_and_none_values():
    from pyrsistent import PClass, field
    class MixedClass(PClass):
        flag = field()
        empty = field()
    instance = MixedClass(flag=True, empty=None)
    expected = "MixedClass(flag=True, empty=None)"
    actual = repr(instance)
    assert actual == expected

def test_repr_uses_to_dict_method():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    instance = TestClass(a=1, b=2)
    dict_repr = instance._to_dict()
    expected = "TestClass(a=1, b=2)"
    actual = repr(instance)
    assert actual == expected
    assert isinstance(dict_repr, dict)
    assert dict_repr == {'a': 1, 'b': 2}

def test_repr_with_initial_field_values():
    from pyrsistent import PClass, field
    class WithInitial(PClass):
        x = field(initial=5)
        y = field()
    instance = WithInitial(y=10)
    expected = "WithInitial(x=5, y=10)"
    actual = repr(instance)
    assert actual == expected

def test_repr_after_set_operation():
    from pyrsistent import PClass, field
    class Changeable(PClass):
        value = field()
    instance = Changeable(value=1)
    new_instance = instance.set(value=2)
    expected = "Changeable(value=2)"
    actual = repr(new_instance)
    assert actual == expected

def test_repr_with_complex_nested_structure():
    from pyrsistent import PClass, field, pvector
    class NestedClass(PClass):
        items = field()
    instance = NestedClass(items=pvector([1, 2, 3]))
    expected = "NestedClass(items=pvector([1, 2, 3]))"
    actual = repr(instance)
    assert actual == expected


# LLM-generated content at query #29
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
        assert "missing_fields" in str(e)

def test_constructor_uses_initial_for_non_provided_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=2)
    assert instance.x == 10
    assert instance.y == 2

def test_constructor_raises_on_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=1, z=3)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test_constructor_ignores_extra_fields_when_ignore_extra_true():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1, z=3, ignore_extra=True)
    assert instance.x == 1
    assert not hasattr(instance, 'z')

def test_constructor_invokes_factory_for_fields():
    from pyrsistent import PClass, field
    def custom_factory(val):
        return val * 2
    class TestClass(PClass):
        x = field(factory=custom_factory)
    instance = TestClass(x=5)
    assert instance.x == 10

def test_constructor_checks_invariants():
    from pyrsistent import PClass, field, InvariantException
    def invariant_check(val):
        if val < 0:
            return (f"Value must be non-negative, got {val}",)
        return ()
    class TestClass(PClass):
        x = field(invariant=invariant_check)
    try:
        TestClass(x=-1)
        assert False
    except InvariantException as e:
        assert "invariant_errors" in str(e)

def test_constructor_supports_callable_initial():
    from pyrsistent import PClass, field
    def initial_func():
        return 42
    class TestClass(PClass):
        x = field(initial=initial_func)
    instance = TestClass()
    assert instance.x == 42

def test_constructor_handles_factory_fields_parameter():
    from pyrsistent import PClass, field
    def custom_factory(val):
        return val + 100
    class TestClass(PClass):
        x = field(factory=custom_factory)
        y = field()
    instance = TestClass(x=5, y=10, _factory_fields={'x'})
    assert instance.x == 105
    assert instance.y == 10

def test_constructor_freezes_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)


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
    assert a is not a2

def test_set_with_positional_arguments():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set('x', 3)
    assert a.x == 1
    assert a2.x == 3
    assert a is not a2

def test_set_multiple_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(x=10, y=20)
    assert a.x == 1 and a.y == 2
    assert a2.x == 10 and a2.y == 20

def test_set_partial_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(x=5)
    assert a.x == 1 and a.y == 2
    assert a2.x == 5 and a2.y == 2

def test_set_unchanged_returns_new_instance():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=1)
    assert a.x == a2.x
    assert a is not a2

def test_set_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(mandatory=True)
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(y=3)
    assert a.x == 1 and a.y == 2
    assert a2.x == 1 and a2.y == 3

def test_set_with_initial_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(initial=10)
        y = field()
    a = AClass(y=5)
    a2 = a.set(x=20)
    assert a.x == 10 and a.y == 5
    assert a2.x == 20 and a2.y == 5

def test_set_raises_on_extra_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    try:
        a.set(z=2)
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test_set_preserves_factory_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(type=int)
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(x=3)
    assert isinstance(a2.x, int)
    assert a2.x == 3

def test_set_with_factory_fields_and_ignore_extra():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(x=5)
    assert a2.x == 5 and a2.y == 2


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_creates_instance_with_given_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_constructor_uses_initial_value_for_missing_non_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
        y = field()
    instance = TestClass(y=15)
    assert instance.x == 5
    assert instance.y == 15

def test_constructor_raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    try:
        TestClass(y=15)
        assert False
    except Exception as e:
        assert "missing_fields" in str(e)

def test_constructor_raises_on_extra_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    try:
        TestClass(x=10, z=30)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test_constructor_ignores_extra_field_when_ignore_extra_true():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass.create({"x": 10, "z": 30}, ignore_extra=True)
    assert instance.x == 10
    assert not hasattr(instance, 'z')

def test_constructor_raises_on_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def positive_invariant(value):
        return value > 0, "Value must be positive"
    class TestClass(PClass):
        x = field(invariant=positive_invariant)
    try:
        TestClass(x=-5)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test_constructor_supports_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
        y = field(type=int)
    instance = TestClass(_factory_fields={'x'}, x=10, y=20)
    assert instance.x == 10
    assert instance.y == 20

def test_constructor_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 42)
        y = field()
    instance = TestClass(y=15)
    assert instance.x == 42
    assert instance.y == 15

def test_constructor_creates_frozen_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=10)
    try:
        instance.x = 20
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

def test_constructor_equality_based_on_field_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=20)
    instance3 = TestClass(x=10, y=30)
    assert instance1 == instance2
    assert instance1 != instance3

def test_constructor_hash_consistency_with_equality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y=20)
    instance2 = TestClass(x=10, y=20)
    instance3 = TestClass(x=10, y=30)
    assert hash(instance1) == hash(instance2)
    assert hash(instance1) != hash(instance3)

def test_constructor_pickle_support():
    import pickle
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    original = TestClass(x=10, y=20)
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)
    assert original == restored
    assert original.x == restored.x
    assert original.y == restored.y


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
    assert a is not a2

def test_set_with_positional_arguments():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set('x', 2)
    assert a.x == 1
    assert a2.x == 2
    assert a is not a2

def test_set_multiple_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(x=10, y=20)
    assert a.x == 1 and a.y == 2
    assert a2.x == 10 and a2.y == 20

def test_set_returns_new_instance():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a is not a2
    assert isinstance(a2, AClass)

def test_set_preserves_other_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
        y = field()
    a = AClass(x=1, y=2)
    a2 = a.set(x=10)
    assert a2.y == 2
    assert a2.x == 10

def test_set_with_mandatory_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(mandatory=True)
        y = field()
    a = AClass(x=1)
    a2 = a.set(x=5)
    assert a2.x == 5
    assert not hasattr(a2, 'y')

def test_set_with_initial_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(initial=10)
        y = field()
    a = AClass()
    a2 = a.set(y=20)
    assert a2.x == 10
    assert a2.y == 20

def test_set_raises_attribute_error_for_unknown_field():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    try:
        a.set(z=5)
        assert False
    except AttributeError:
        pass

def test_set_with_factory_fields():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field(type=int)
    a = AClass(x=1)
    a2 = a.set(x=2)
    assert a2.x == 2
    assert isinstance(a2.x, int)

def test_set_maintains_immutability():
    from pyrsistent import PClass, field
    class AClass(PClass):
        x = field()
    a = AClass(x=1)
    a2 = a.set(x=2)
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

def test___new___raises_error_on_invalid_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="not_an_int")
        assert False
    except Exception as e:
        assert "Invalid type" in str(e)

def test___new___uses_initial_value_for_non_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, initial=5)
        y = field(type=str, mandatory=True)
    instance = TestClass(y="test")
    assert instance.x == 5
    assert instance.y == "test"

def test___new___raises_error_on_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, mandatory=True)
    try:
        TestClass()
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___raises_error_on_extra_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x=1, y=2)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test___new___invokes_field_invariant_and_raises_on_failure():
    from pyrsistent import PClass, field
    def invariant_check(value):
        return value > 0, "value_not_positive"
    class TestClass(PClass):
        x = field(type=int, invariant=invariant_check)
    try:
        TestClass(x=-1)
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___checks_global_invariants():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x + obj.y == 10, "sum_not_ten"
    class TestClass(PClass):
        x = field(type=int)
        y = field(type=int)
        __invariants__ = [global_invariant]
    try:
        TestClass(x=3, y=4)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___with_factory_fields_and_ignore_extra():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, factory=lambda v, ignore_extra=False: v * 2)
    instance = TestClass(x=5, ignore_extra=True)
    assert instance.x == 10

def test___new___freezes_instance_after_creation():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

def test___new___with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, initial=lambda: 42)
        y = field(type=str, mandatory=True)
    instance = TestClass(y="test")
    assert instance.x == 42
    assert instance.y == "test"


# LLM-generated content at query #4
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
    else:
        assert False


# LLM-generated content at query #5
#--------------------------

def test_set_method_with_existing_field():
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

def test_set_method_with_positional_args():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set('x', 3)
    assert new_instance.x == 3
    assert new_instance.y == 2
    assert instance.x == 1
    assert instance.y == 2

def test_set_method_with_multiple_fields():
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
    assert instance.x == 1
    assert instance.y == 2
    assert instance.z == 3

def test_set_method_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    instance = TestClass(x=1)
    new_instance = instance.set(y=5)
    assert new_instance.x == 1
    assert new_instance.y == 5
    assert instance.x == 1
    assert not hasattr(instance, 'y')

def test_set_method_with_initial_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=10)
        y = field()
    instance = TestClass(y=2)
    new_instance = instance.set(y=20)
    assert new_instance.x == 10
    assert new_instance.y == 20
    assert instance.x == 10
    assert instance.y == 2

def test_set_method_preserves_factory_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=100)
    assert new_instance._pclass_frozen == True
    assert instance._pclass_frozen == True

def test_set_method_with_no_args_updates_nothing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set()
    assert new_instance.x == 1
    assert new_instance.y == 2
    assert instance.x == 1
    assert instance.y == 2

def test_set_method_creates_new_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    new_instance = instance.set(x=99)
    assert new_instance is not instance
    assert new_instance.__class__ is instance.__class__


# LLM-generated content at query #6
#--------------------------

def test_PClassMeta_new_with_no_fields_or_invariants():
    class TestClass(metaclass=PClassMeta):
        pass
    assert TestClass._pclass_fields == {}
    assert TestClass._pclass_invariants == ()
    assert TestClass.__slots__ == ('_pclass_frozen', '__weakref__')

def test_PClassMeta_new_with_fields():
    class TestClass(metaclass=PClassMeta):
        _field1 = _PField()
        _field2 = _PField()
    assert 'field1' in TestClass._pclass_fields
    assert 'field2' in TestClass._pclass_fields
    assert TestClass._pclass_fields['field1'] is TestClass._field1
    assert TestClass._pclass_fields['field2'] is TestClass._field2
    assert '_field1' not in TestClass.__dict__
    assert '_field2' not in TestClass.__dict__
    assert TestClass.__slots__ == ('_pclass_frozen', 'field1', 'field2', '__weakref__')

def test_PClassMeta_new_with_inherited_fields():
    class BaseClass(metaclass=PClassMeta):
        _field1 = _PField()
    class DerivedClass(BaseClass, metaclass=PClassMeta):
        _field2 = _PField()
    assert 'field1' in DerivedClass._pclass_fields
    assert 'field2' in DerivedClass._pclass_fields
    assert DerivedClass._pclass_fields['field1'] is BaseClass._pclass_fields['field1']
    assert DerivedClass._pclass_fields['field2'] is DerivedClass._field2
    assert DerivedClass.__slots__ == ('_pclass_frozen', 'field1', 'field2', '__weakref__')

def test_PClassMeta_new_with_invariant():
    def sample_invariant(instance):
        return True, ()
    class TestClass(metaclass=PClassMeta):
        __invariant__ = sample_invariant
    assert len(TestClass._pclass_invariants) == 1
    assert callable(TestClass._pclass_invariants[0])
    assert TestClass._pclass_invariants[0](None) == (True, ())

def test_PClassMeta_new_with_inherited_invariants():
    def invariant1(instance):
        return True, ()
    def invariant2(instance):
        return False, ('error',)
    class BaseClass(metaclass=PClassMeta):
        __invariant__ = invariant1
    class DerivedClass(BaseClass, metaclass=PClassMeta):
        __invariant__ = invariant2
    assert len(DerivedClass._pclass_invariants) == 2
    result1 = DerivedClass._pclass_invariants[0](None)
    result2 = DerivedClass._pclass_invariants[1](None)
    assert result1 == (True, ())
    assert result2 == (False, ('error',))

def test_PClassMeta_new_with_non_callable_invariant_raises():
    try:
        class TestClass(metaclass=PClassMeta):
            __invariant__ = 'not callable'
        assert False
    except TypeError as e:
        assert 'Invariants must be callable' in str(e)

def test_PClassMeta_new_with_multiple_inheritance_and_invariants():
    def invariantA(instance):
        return True, ()
    def invariantB(instance):
        return True, ()
    class BaseA(metaclass=PClassMeta):
        __invariant__ = invariantA
    class BaseB(metaclass=PClassMeta):
        __invariant__ = invariantB
    class Derived(BaseA, BaseB, metaclass=PClassMeta):
        pass
    assert len(Derived._pclass_invariants) == 2
    assert Derived._pclass_invariants[0](None) == (True, ())
    assert Derived._pclass_invariants[1](None) == (True, ())

def test_PClassMeta_new_with_complex_invariant_returning_tuple():
    def complex_invariant(instance):
        return [(True, ()), (False, ('err1',)), (True, ()), (False, ('err2',))]
    class TestClass(metaclass=PClassMeta):
        __invariant__ = complex_invariant
    result = TestClass._pclass_invariants[0](None)
    assert result == (False, ('err1', 'err2'))

def test_PClassMeta_new_with_non_checked_type_base_has_no_weakref():
    class RegularBase:
        pass
    class TestClass(RegularBase, metaclass=PClassMeta):
        pass
    assert '__weakref__' not in TestClass.__slots__


# LLM-generated content at query #7
#--------------------------

def test_set_updates_data_and_flags_when_value_changes():
    original = object()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set('a', 3)
    assert evolver._pclass_evolver_data == {'a': 3, 'b': 2}
    assert evolver._factory_fields == {'a'}
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver

def test_set_does_not_update_when_value_unchanged():
    original = object()
    initial_dict = {'a': 1, 'b': 2}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set('a', 1)
    assert evolver._pclass_evolver_data == {'a': 1, 'b': 2}
    assert evolver._factory_fields == set()
    assert evolver._pclass_evolver_data_is_dirty is False
    assert result is evolver

def test_set_adds_new_key():
    original = object()
    initial_dict = {'a': 1}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set('b', 2)
    assert evolver._pclass_evolver_data == {'a': 1, 'b': 2}
    assert evolver._factory_fields == {'b'}
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver

def test_set_with_missing_value_constant():
    original = object()
    initial_dict = {}
    evolver = _PClassEvolver(original, initial_dict)
    result = evolver.set('key', _MISSING_VALUE)
    assert evolver._pclass_evolver_data == {'key': _MISSING_VALUE}
    assert evolver._factory_fields == {'key'}
    assert evolver._pclass_evolver_data_is_dirty is True
    assert result is evolver


# LLM-generated content at query #8
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


# LLM-generated content at query #9
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
        return f"{format}:{value}"
    class TestClass(PClass):
        data = field(serializer=custom_serializer)
    instance = TestClass(data="info")
    result = instance.serialize(format="json")
    expected = {'data': 'json:info'}
    assert result == expected

def test_serialize_missing_field_with_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=100)
        y = field()
    instance = TestClass(y=200)
    result = instance.serialize()
    expected = {'x': 100, 'y': 200}
    assert result == expected

def test_serialize_only_fields_with_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field(mandatory=True)
        b = field(initial=0)
        c = field()
    instance = TestClass(a=1)
    result = instance.serialize()
    expected = {'a': 1, 'b': 0}
    assert result == expected


# LLM-generated content at query #10
#--------------------------

def test___eq___with_same_class_and_equal_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=2)
    result = instance1.__eq__(instance2)
    assert result is True

def test___eq___with_same_class_and_different_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1, y=3)
    result = instance1.__eq__(instance2)
    assert result is False

def test___eq___with_different_class():
    from pyrsistent import PClass, field
    class TestClass1(PClass):
        x = field()
    class TestClass2(PClass):
        x = field()
    instance1 = TestClass1(x=1)
    instance2 = TestClass2(x=1)
    result = instance1.__eq__(instance2)
    assert result is NotImplemented

def test___eq___with_non_pclass_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=1)
    other = object()
    result = instance.__eq__(other)
    assert result is NotImplemented

def test___eq___with_missing_field_in_one_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1, y=2)
    instance2 = TestClass(x=1)
    result = instance1.__eq__(instance2)
    assert result is False

def test___eq___with_missing_field_in_both_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=1)
    instance2 = TestClass(x=1)
    result = instance1.__eq__(instance2)
    assert result is True

def test___eq___with_nested_pclass_fields():
    from pyrsistent import PClass, field
    class InnerClass(PClass):
        a = field()
    class OuterClass(PClass):
        inner = field()
    inner1 = InnerClass(a=5)
    inner2 = InnerClass(a=5)
    instance1 = OuterClass(inner=inner1)
    instance2 = OuterClass(inner=inner2)
    result = instance1.__eq__(instance2)
    assert result is True

def test___eq___with_nested_pclass_fields_different():
    from pyrsistent import PClass, field
    class InnerClass(PClass):
        a = field()
    class OuterClass(PClass):
        inner = field()
    inner1 = InnerClass(a=5)
    inner2 = InnerClass(a=6)
    instance1 = OuterClass(inner=inner1)
    instance2 = OuterClass(inner=inner2)
    result = instance1.__eq__(instance2)
    assert result is False


# LLM-generated content at query #11
#--------------------------

def test_set_does_not_modify_when_value_is_same_object():
    from pyrsistent import _PClassEvolver, _MISSING_VALUE
    original = object()
    initial_dict = {'key': 'value'}
    evolver = _PClassEvolver(original, initial_dict.copy())
    same_value = initial_dict['key']
    evolver.set('key', same_value)
    assert evolver._pclass_evolver_data_is_dirty == False
    assert evolver._factory_fields == set()
    assert evolver._pclass_evolver_data == initial_dict


# LLM-generated content at query #12
#--------------------------

def test_repr_with_single_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=42)
    result = repr(instance)
    expected = "TestClass(x=42)"
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

def test_repr_with_mandatory_field_missing():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(mandatory=True)
        y = field()
    instance = TestClass(x=10)
    result = repr(instance)
    expected = "TestClass(x=10)"
    assert result == expected

def test_repr_with_initial_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=5)
        y = field()
    instance = TestClass(y=20)
    result = repr(instance)
    expected = "TestClass(x=5, y=20)"
    assert result == expected

def test_repr_with_callable_initial():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(initial=lambda: 100)
        y = field()
    instance = TestClass(y=30)
    result = repr(instance)
    expected = "TestClass(x=100, y=30)"
    assert result == expected

def test_repr_with_no_fields():
    from pyrsistent import PClass
    class TestClass(PClass):
        pass
    instance = TestClass()
    result = repr(instance)
    expected = "TestClass()"
    assert result == expected


# LLM-generated content at query #13
#--------------------------

```python
def test_invariant_errors_or_missing_fields_raises_exception():
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


# LLM-generated content at query #14
#--------------------------

def test___hash___returns_same_hash_for_equal_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y="test")
    instance2 = TestClass(x=10, y="test")
    assert hash(instance1) == hash(instance2)

def test___hash___returns_different_hash_for_different_instances():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=10, y="test")
    instance2 = TestClass(x=20, y="test")
    assert hash(instance1) != hash(instance2)

def test___hash___handles_missing_values_consistently():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field(mandatory=False)
    instance1 = TestClass(x=10)
    instance2 = TestClass(x=10)
    assert hash(instance1) == hash(instance2)

def test___hash___works_with_none_values():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance1 = TestClass(x=None, y=None)
    instance2 = TestClass(x=None, y=None)
    assert hash(instance1) == hash(instance2)

def test___hash___produces_integer():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
    instance = TestClass(x=5)
    assert isinstance(hash(instance), int)

def test___hash___consistent_with_equality():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        a = field()
        b = field()
    instance1 = TestClass(a=1, b=2)
    instance2 = TestClass(a=1, b=2)
    assert instance1 == instance2
    assert hash(instance1) == hash(instance2)

def test___hash___different_for_different_field_order_in_tuple():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field()
        y = field()
    instance = TestClass(x=1, y=2)
    expected_tuple = (("x", 1), ("y", 2))
    assert hash(instance) == hash(expected_tuple)


# LLM-generated content at query #15
#--------------------------

def test___new___creates_instance_with_valid_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
        y = field(type=str, mandatory=True)
    instance = TestClass(x=10, y="test")
    assert instance.x == 10
    assert instance.y == "test"

def test___new___raises_on_invalid_type():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x="not_an_int")
        assert False
    except Exception as e:
        assert "Invalid type" in str(e)

def test___new___uses_initial_value_when_not_provided():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, initial=5)
        y = field(type=str, mandatory=True)
    instance = TestClass(y="test")
    assert instance.x == 5
    assert instance.y == "test"

def test___new___raises_on_missing_mandatory_field():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int, mandatory=True)
    try:
        TestClass()
        assert False
    except Exception as e:
        assert "Field invariant failed" in str(e)

def test___new___raises_on_extra_fields():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    try:
        TestClass(x=1, extra=2)
        assert False
    except AttributeError as e:
        assert "are not among the specified fields" in str(e)

def test___new___handles_factory_fields_with_ignore_extra():
    from pyrsistent import PClass, field
    def factory_func(value, ignore_extra=False):
        return value
    class TestClass(PClass):
        x = field(type=int, factory=factory_func)
    instance = TestClass(x=10, ignore_extra=True)
    assert instance.x == 10

def test___new___raises_on_field_invariant_failure():
    from pyrsistent import PClass, field, InvariantException
    def invariant_func(value):
        return value > 0, "error"
    class TestClass(PClass):
        x = field(type=int, invariant=invariant_func)
    try:
        TestClass(x=-5)
        assert False
    except InvariantException as e:
        assert "Field invariant failed" in str(e)

def test___new___checks_global_invariants():
    from pyrsistent import PClass, field, InvariantException
    def global_invariant(obj):
        return obj.x + obj.y == 10, "global_error"
    class TestClass(PClass):
        x = field(type=int)
        y = field(type=int)
        _pclass_invariants = (global_invariant,)
    try:
        TestClass(x=3, y=8)
        assert False
    except InvariantException as e:
        assert "Global invariant failed" in str(e)

def test___new___freezes_instance():
    from pyrsistent import PClass, field
    class TestClass(PClass):
        x = field(type=int)
    instance = TestClass(x=1)
    try:
        instance.x = 2
        assert False
    except AttributeError as e:
        assert "Can't set attribute" in str(e)

def test___new___with_callable_initial():
    from pyrsistent import PClass, field
    def initial_func():
        return 42
    class TestClass(PClass):
        x = field(type=int, initial=initial_func)
        y = field(type=str, mandatory=True)
    instance = TestClass(y="test")
    assert instance.x == 42
    assert instance.y == "test"


